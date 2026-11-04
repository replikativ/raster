(ns raster.gpu.cuda-copy-test
  "Resident-copy contracts restored from 063e/#1338; common-owner race controls
  exercise current lifetime authority. Native driver calls and exact cuda:0
  admission are test-local; the production CUDA registry stays disabled."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-backend :as backend]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.cuda-lifecycle-test :as fixture]))

(defn- with-driver [f]
  (let [original-type backend/backend-type
        original-descriptor backend/descriptor
        descriptor {:namespace 'raster.gpu.cuda-runtime
                    :kernel-body-c-dialect :cuda
                    :memory-space :device :coherence :explicit-transfer
                    :owned-slice? true}]
    (with-redefs [backend/backend-type
                  (fn [device] (if (= :cuda:0 device) :cuda (original-type device)))
                  backend/descriptor
                  (fn [device] (if (= :cuda:0 device) descriptor
                                  (original-descriptor device)))]
      (#'fixture/with-driver f))))

(defn- copy-buffer-range! [& args]
  ;; Resolve at the actual contract call, so the unchanged runtime can load and
  ;; the public/API absence is an observed behavioural failure, not an import error.
  (apply (requiring-resolve 'raster.gpu.cuda-runtime/copy-buffer-range!) args))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(deftest public-resident-copy-uses-view-offsets-and-returns-identical-destination
  (with-driver
    (fn [ops _]
      (let [session (gpu/make-session :cuda:0)]
        (try
          (gpu/alloc! session {:source [:float 20 nil] :destination [:float 20 nil]})
          (let [source (gpu/buffer-view session :source {:byte-offset 12 :shape [5]})
                destination (gpu/buffer-view session :destination {:byte-offset 28 :shape [5]})
                before (count @ops)
                result (try (gpu/copy-range! session source destination {:elements 5})
                            (catch Throwable error error))
                [_ [dst src bytes stream]] (get @ops before)]
            (is (identical? destination result)
                "public resident D2D copy must return the original destination view")
            (is (= [:copy :stream-sync] (mapv first (drop before @ops))))
            (is (= (+ 28 (:device-pointer (gpu/buffer session :destination))) dst))
            (is (= (+ 12 (:device-pointer (gpu/buffer session :source))) src))
            (is (= 20 bytes))
            (is (identical? (:stream @runtime/state) stream))
            (is (identical? stream (first (second (get @ops (inc before)))))))
          (finally (gpu/close-session! session)))))))

(deftest copy-validation-zero-and-overlap-reject-before-driver-contact
  (with-driver
    (fn [ops _]
      (let [session (gpu/make-session :cuda:0) other (gpu/make-session :cuda:0)]
        (try
          (gpu/alloc! session {:x [:float 20 nil] :half [:half 20 nil]})
          (gpu/alloc! other {:x [:float 20 nil]})
          (let [left (gpu/buffer-view session :x {:byte-offset 4 :shape [5]})
                overlap (gpu/buffer-view session :x {:byte-offset 12 :shape [5]})
                disjoint (gpu/buffer-view session :x {:byte-offset 40 :shape [5]})
                foreign (gpu/buffer-view other :x) before (count @ops)]
            (doseq [[src dst spec] [[left overlap {:elements 5}] [left left {:elements 5}]
                                   [left :half {:elements 1}] [left foreign {:elements 1}]
                                   [left disjoint {:elements 6}] [left disjoint {:elements -1}]
                                   [left disjoint {:src-element Long/MAX_VALUE :elements 1}]
                                   [left disjoint {:dst-element Long/MAX_VALUE :elements 1}]]]
              (is (some? (error-of #(gpu/copy-range! session src dst spec)))))
            (is (= before (count @ops)))
            (is (identical? left (gpu/copy-range! session left left {:elements 0})))
            (is (= before (count @ops)) "zero length is a no-op even on the same view")
            (is (identical? disjoint (gpu/copy-range! session left disjoint {:elements 5})))
            (doseq [range [[-1 0 1] [0 -1 1] [0 0 -1] [20 0 1] [0 20 1]
                           [Long/MAX_VALUE 0 1] [0 Long/MAX_VALUE 1]
                           [0 0 Long/MAX_VALUE] [0 0 (inc (bigint Long/MAX_VALUE))]]]
              (let [before (count @ops) b (gpu/buffer session :x)]
                (is (= :cuda-copy-range
                       (:reason (ex-data (error-of #(apply copy-buffer-range! b b range))))))
                (is (= before (count @ops))))))
          (finally (gpu/close-session! session) (gpu/close-session! other))))))
  (with-driver
    (fn [ops _]
      (let [src (runtime/make-buffer 4 :float) dst (runtime/make-buffer 4 :float)]
        (try
          (let [before (count @ops)
                foreign (assoc src ::runtime/context (Object.))]
            (is (= :runtime-generation-mismatch
                   (:reason (ex-data (error-of #(copy-buffer-range! foreign dst 0 0 1))))))
            (is (= before (count @ops))))
          (runtime/free-buffer! src)
          (let [before (count @ops)]
            (is (= :owner-releasing
                   (:reason (ex-data (error-of #(copy-buffer-range! src dst 0 0 1))))))
            (is (= before (count @ops))))
          (finally (runtime/free-buffer! src) (runtime/free-buffer! dst)))))))

(deftest copy-unknown-completion-keeps-both-buffers-and-root-with-one-native-wait
  (with-driver
    (fn [ops fault]
      (let [src (runtime/make-buffer 4 :float) dst (runtime/make-buffer 4 :float)
            failure (ex-info "copy completion unknown" {})]
        (reset! fault {:operation :stream-sync :error failure})
        (is (identical? failure (error-of #(copy-buffer-range! src dst 0 0 4))))
        (is (= 1 (count (filter #(= :stream-sync (first %)) @ops))))
        (is (= 2 (root/lease-count runtime/state)))
        (is (= 1 (count @(:debts (:prepared-execution @runtime/state)))))
        (doseq [buffer [src dst]]
          (is (= :owner-in-use
                 (:reason (ex-data (error-of #(runtime/free-buffer! buffer)))))))
        (reset! fault nil)
        (let [before (count @ops)]
          (is (identical? failure (error-of runtime/synchronize-async!)))
          (is (= before (count @ops)))
          (is (= 2 (root/lease-count runtime/state)))
          (is (not-any? #(contains? #{:free :context-destroy} (first %)) @ops)))))))

(deftest copy-explicitly-retry-safe-completion-can-retire-retained-debt
  (with-driver
    (fn [ops fault]
      (let [src (runtime/make-buffer 4 :half) dst (runtime/make-buffer 4 :half)
            failure (ex-info "copy drain retry allowed" {:cleanup-retry-safe? true})]
        (reset! fault {:operation :stream-sync :error failure})
        (is (identical? failure (error-of #(copy-buffer-range! src dst 0 0 4))))
        (is (= 2 (root/lease-count runtime/state)))
        (is (= 1 (count @(:debts (:prepared-execution @runtime/state)))))
        (doseq [buffer [src dst]]
          (is (= :owner-in-use
                 (:reason (ex-data (error-of #(runtime/free-buffer! buffer)))))))
        (reset! fault nil)
        (let [waits (count (filter #(= :stream-sync (first %)) @ops))]
          (runtime/synchronize-async!)
          (is (= (inc waits) (count (filter #(= :stream-sync (first %)) @ops)))))
        (is (empty? @(:debts (:prepared-execution @runtime/state))))
        (is (= 2 (root/lease-count runtime/state)))
        (runtime/free-buffer! src) (runtime/free-buffer! dst)
        (is (= 0 (root/lease-count runtime/state)))
        (is (= 2 (count (filter #(= :free (first %)) @ops))))))))

(deftest retirement-winning-either-copy-owner-does-not-invert-owner-and-root-locks
  (doseq [retire-endpoint [:source :destination]]
    (with-driver
      (fn [ops _]
        (let [src (runtime/make-buffer 4 :float) dst (runtime/make-buffer 4 :float)
              retiring (if (= :source retire-endpoint) src dst)
              target-owner (cleanup/lifetime-owner retiring)
              original cleanup/retain-use! native driver/*invoke*
              reserving (promise) reserve (promise) freeing (promise) finish-free (promise)]
          (binding [driver/*invoke*
                    (fn [operation args]
                      (when (and (= :free operation)
                                 (= (:device-pointer retiring) (first args)))
                        (deliver freeing true)
                        (deref finish-free 5000 :timeout))
                      (native operation args))]
            (with-redefs [cleanup/retain-use!
                          (fn [owner]
                            (when (Thread/holdsLock runtime/state)
                              (deliver reserving :wrong-lock-order)
                              (throw (ex-info "root held before copy owner admission" {})))
                            (when (identical? target-owner owner)
                              (deliver reserving true)
                              (deref reserve 5000 :timeout))
                            (original owner))]
              (let [copy (future
                           (try (error-of #(copy-buffer-range! src dst 0 0 4))
                                (finally (deliver reserving :no-owner-admission))))]
                (try
                  (is (= true (deref reserving 5000 :timeout)) (str retire-endpoint))
                  (when (= true (deref reserving 0 :timeout))
                    (let [retirement (future (runtime/free-buffer! retiring))]
                      (try
                        (is (= true (deref freeing 5000 :timeout)))
                        (is (= :available
                               (deref (future (locking runtime/state :available))
                                      5000 :timeout)))
                        (finally (deliver reserve true) (deliver finish-free true)))
                      (is (nil? (deref retirement 5000 :timeout)))
                      (is (= :owner-releasing (:reason (ex-data (deref copy 5000 nil)))))
                      (is (not-any? #(= :copy (first %)) @ops))))
                  (finally (deliver reserve true) (deliver finish-free true)))
                (is (not= :timeout (deref copy 5000 :timeout))))))
          (runtime/free-buffer! src) (runtime/free-buffer! dst))))))

(defn- assert-both-in-use! [ops buffers]
  (let [before (count @ops)]
    (doseq [buffer buffers]
      (is (= :owner-in-use
             (:reason (ex-data (error-of #(runtime/free-buffer! buffer)))))))
    (is (= before (count @ops)) "active endpoint retirement must not contact native free")))

(deftest admission-winning-copy-keeps-both-owners-through-native-completion
  (with-driver
    (fn [ops _]
      (let [src (runtime/make-buffer 4 :float) dst (runtime/make-buffer 4 :float)
            native driver/*invoke* observed (atom []) stream (:stream @runtime/state)
            before (count @ops)]
        (binding [driver/*invoke*
                  (fn [operation args]
                    (when (contains? #{:copy :stream-sync} operation)
                      (swap! observed conj operation)
                      (assert-both-in-use! ops [src dst])
                      (is (identical? stream
                                      (if (= :copy operation) (nth args 3) (first args)))))
                    (native operation args))]
          (is (identical? dst (copy-buffer-range! src dst 0 0 4))))
        (is (= [:copy :stream-sync] @observed))
        (is (= [:copy :stream-sync] (mapv first (drop before @ops))))
        (is (empty? @(:debts (:prepared-execution @runtime/state))))
        (is (= 2 (root/lease-count runtime/state)))
        (runtime/free-buffer! src) (runtime/free-buffer! dst)
        (is (= 0 (root/lease-count runtime/state)))
        (is (= 2 (count (filter #(= :free (first %)) @ops))))))))

(deftest throwing-enqueue-drains-before-retirement-or-retains-unknown-completion
  (doseq [completion [:known :unknown]]
    (with-driver
      (fn [ops _]
        (let [src (runtime/make-buffer 4 :float) dst (runtime/make-buffer 4 :float)
              native driver/*invoke* stream (:stream @runtime/state)
              started (atom false) enqueue-failure (ex-info "enqueue returned after possible work" {})
              completion-failure (ex-info "rollback completion indeterminate" {})]
          (binding [driver/*invoke*
                    (fn [operation args]
                      (case operation
                        :copy (do
                                (assert-both-in-use! ops [src dst])
                                (is (identical? stream (nth args 3)))
                                (native operation args)
                                ;; Record/accept the fake enqueue before throwing; a
                                ;; driver exception cannot prove that no work started.
                                (reset! started true)
                                (throw enqueue-failure))
                        :stream-sync (do
                                       (is @started)
                                       (assert-both-in-use! ops [src dst])
                                       (is (identical? stream (first args)))
                                       (native operation args)
                                       (when (= :unknown completion) (throw completion-failure)))
                        (native operation args)))]
            (is (identical? enqueue-failure
                            (error-of #(copy-buffer-range! src dst 0 0 4))))
            (is (= 1 (count (filter #(= :copy (first %)) @ops))))
            (is (= 1 (count (filter #(= :stream-sync (first %)) @ops))))
            (is (= 2 (root/lease-count runtime/state)))
            (if (= :known completion)
              (do
                (is (empty? @(:debts (:prepared-execution @runtime/state))))
                (is (empty? (.getSuppressed enqueue-failure)))
                (runtime/free-buffer! src) (runtime/free-buffer! dst)
                (is (= 0 (root/lease-count runtime/state)))
                (is (= 2 (count (filter #(= :free (first %)) @ops)))))
              (do
                (is (= 1 (count @(:debts (:prepared-execution @runtime/state)))))
                (is (some #(identical? completion-failure %) (.getSuppressed enqueue-failure)))
                (assert-both-in-use! ops [src dst])
                (let [before (count @ops)]
                  (is (identical? completion-failure (error-of runtime/synchronize-async!)))
                  (is (= before (count @ops)))
                  (is (= 2 (root/lease-count runtime/state))))
                (is (not-any? #(contains? #{:free :context-destroy} (first %)) @ops))))))))))
