(ns raster.gpu.cuda-copy-device-test
  "Native-only D2D qualification adapted from the 063e resident-copy proof.
  Generic CI may visibly skip an unavailable device; REQUIRE_CUDA_DEVICE=1
  requires an actual ready probe and never accepts a skip. No generated kernels."
  (:require [clojure.string :as string]
            [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-backend :as backend]
            [raster.gpu.runtime-root :as root]))

(defn- preserving-primary [body retire!]
  (let [primary (volatile! nil)]
    (try (body)
         (catch Throwable failure (vreset! primary failure) (throw failure))
         (finally
           (try (retire!)
                (catch Throwable failure
                  (if-let [original @primary]
                    (when-not (identical? original failure) (.addSuppressed original failure))
                    (throw failure))))))))

(defn- with-cuda-admission [f]
  (let [original-type backend/backend-type original-descriptor backend/descriptor
        descriptor {:namespace 'raster.gpu.cuda-runtime :kernel-body-c-dialect :cuda
                    :memory-space :device :coherence :explicit-transfer :owned-slice? true}]
    (with-redefs [backend/backend-type
                  (fn [device] (if (= :cuda:0 device) :cuda (original-type device)))
                  backend/descriptor
                  (fn [device] (if (= :cuda:0 device) descriptor (original-descriptor device)))]
      (f))))

(defn- raw-bits [dt host]
  (mapv #(if (= :float dt) (Float/floatToRawIntBits (float %))
             (bit-and 65535 (int %))) host))

(defn- source-data [dt total phase]
  (if (= :float dt)
    (float-array (map #(Float/intBitsToFloat (int (+ 1056964608 (* phase 1048576) (* % 2048))))
                      (range total)))
    (short-array (map #(unchecked-short (+ 123 (* phase 7919) (* % 1543))) (range total)))))

(defn- copy-proof! [dt ops]
  (let [n 257 total (+ n 13) width (if (= :float dt) 4 2)
        sentinel (if (= :float dt) (float -777) (short 7))
        initial (if (= :float dt) (float-array (repeat total sentinel))
                    (short-array (repeat total sentinel)))
        guard-bits (first (raw-bits dt initial)) session (gpu/make-session :cuda:0)]
    (preserving-primary
      (fn []
        (gpu/alloc! session {:source [dt total nil] :destination [dt total initial]})
        (let [source (gpu/buffer-view session :source {:byte-offset (* 3 width) :shape [n]})
              destination (gpu/buffer-view session :destination {:byte-offset (* 7 width) :shape [n]})
              previous (volatile! nil)]
          (doseq [phase [0 1]]
            (let [data (source-data dt total phase) source-bits (raw-bits dt data)
                  expected (vec (concat (repeat 7 guard-bits)
                                        (subvec source-bits 3 (+ 3 n))
                                        (repeat 6 guard-bits)))]
              (when @previous (is (not= @previous expected) "replay must change copied bits"))
              (vreset! previous expected)
              (gpu/upload! session :source data)
              (let [before (count @ops)]
                (is (identical? destination (gpu/copy-range! session source destination {:elements n})))
                (is (= [:copy :stream-sync] (mapv first (drop before @ops))))
                (let [[_ [dst src bytes stream]] (get @ops before)]
                  (is (= (+ (:device-pointer (gpu/buffer session :source)) (* 3 width)) src))
                  (is (= (+ (:device-pointer (gpu/buffer session :destination)) (* 7 width)) dst))
                  (is (= (* n width) bytes))
                  (is (identical? (:stream @runtime/state) stream))
                  (is (identical? stream (first (second (get @ops (inc before))))))))
              ;; Read once: identical values feed assertions and retained raw witnesses.
              (let [actual-destination (raw-bits dt (gpu/download session :destination))
                    actual-source (raw-bits dt (gpu/download session :source))]
                (is (= expected actual-destination))
                (is (= source-bits actual-source))
                (prn {:event :cuda-copy-witness :dtype dt :phase phase
                      :elements n :total-elements total :source-element 3 :destination-element 7
                      :input-source-bits source-bits :expected-destination-bits expected
                      :actual-destination-bits actual-destination :actual-source-bits actual-source}))))
          (let [before (count @ops)]
            (is (identical? destination (gpu/copy-range! session source destination {:elements 0})))
            (is (= before (count @ops))))
          (println "[CUDA COPY NATIVE]" {:dtype dt :elements n :total-elements total :passes 2
                                         :source-byte-offset (* 3 width)
                                         :destination-byte-offset (* 7 width)
                                         :device (gpu/execution-device-info session)})))
      #(gpu/close-session! session))
    (is (= 0 (root/lease-count runtime/state)))))

(defn- retire-fixture-root! [ops returned]
  (when-let [entry (::root/entry @runtime/state)]
    (let [leases (count @(:leases entry)) ledger (:prepared-execution @runtime/state)
          pending (count @(:pending ledger)) debts (count @(:debts ledger))]
      (when-not (and (zero? leases) (zero? pending) (zero? debts) (empty? @runtime/kernel-registry))
        (throw (ex-info "Native fixture still owns live children or completion debt"
                        {:leases leases :pending pending :debts debts})))
      ;; Production shutdown!/reset! deliberately reject ever-live roots. This is
      ;; explicit test-owned canonical teardown, not supported public shutdown proof.
      (cleanup/release! (::cleanup/owner entry))
      (when (seq (cleanup/pending (::cleanup/owner entry)))
        (throw (ex-info "Native fixture root teardown remains incomplete" {})))
      (is (= 1 (count (filter #(= :destroy-runtime (first %)) @ops))))
      (is (= 1 (count (filter #(= :destroy-runtime %) @returned))))
      (is (= (count (filter #(= :host-allocate (first %)) @ops))
             (count (filter #(= :host-free (first %)) @ops))))
      (is (= 4 (count (filter #(= :free (first %)) @ops))))
      ;; Reset only this fixture's private projection after confirmed native release.
      (reset! runtime/state {})
      (is (empty? @runtime/state))
      (println "[CUDA COPY RETIRED]" {:leases leases :pending pending :debts debts
                                      :destroy-runtime-returned 1 :fixture-state-reset true
                                      :production-shutdown-qualified false}))))

(deftest native-resident-copy-float-and-raw-half-offset-guards-and-changed-replay
  (let [required? (contains? #{"1" "true" "yes" "on"}
                             (some-> (System/getenv "RASTER_REQUIRE_CUDA_DEVICE") string/lower-case))
        result (try (probe/probe) (catch Throwable error {:status :probe-error :error error}))]
    (if-not (= :ready (:status result))
      (if required?
        (throw (ex-info "Required CUDA resident-copy qualification needs an actual ready device"
                        {:probe result :required true} (:error result)))
        (do (println "[CUDA COPY SKIP]" {:status (:status result) :required false
                                        :reason (some-> (:error result) .getMessage)})
            (is true "cuda-copy-device-skip-marker")))
      (do
        (println "[CUDA COPY READY]" (assoc (dissoc result :error) :required required?))
        (with-cuda-admission
          (fn []
            (with-redefs [runtime/state (atom {}) runtime/kernel-registry (atom {})]
              (let [ops (atom []) returned (atom [])]
                (binding [driver/*invoke*
                          (fn [operation args]
                            (swap! ops conj [operation args])
                            (let [value (binding [driver/*invoke* nil] (driver/invoke! operation args))]
                              (swap! returned conj operation) value))]
                  (preserving-primary #(doseq [dt [:float :half]] (copy-proof! dt ops))
                                      #(retire-fixture-root! ops returned)))))))))))
