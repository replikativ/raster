(ns raster.gpu.ordered-invocation-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.ir.ordered-invocation-realization-test :as allocation-fixture]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as program]))

(defn- run-early-stage! [target]
  (let [source (double-array [7.0])
        destination (double-array 1)
        compilation (equation-first/compile #'fixture/write-before-shape
                                             {:target target :dtype :double})
        initial (invocation-link/begin-ordered-realization
                 compilation [source destination Long/MAX_VALUE] nil)
        ready (invocation-link/realize-ready-equation
               initial (:id (first (:equations (:program initial)))) nil)
        plan (:stage-plan ready)
        session (gpu/make-session target)
        ledger (volatile! [])
        timings (volatile! {})]
    (try
      (let [storage (link/realize-plan-storage!
                     plan session :early-stage (link/allocation-groups plan) {} ledger timings)
            executor (link/resident-program-executor
                      plan session :early-stage (:node-views storage) false false ledger)
            call (:call ready)
            prepared (program/prepare-sequence-with!
                      [{:id :early :kind :graph
                        :call {:graph (:graph call) :bindings (:buffers call)
                               :scalar-values (:scalar-values call)}}]
                      executor)]
        (try
          (program/run-prepared! prepared)
          (let [output-id (first (:outputs plan))
                first-result (double-array 1)
                after-reuse (double-array 1)]
            (gpu/download-range! session (get-in storage [:node-views output-id])
                                 first-result {:elements 1})
            (is (= [8.0] (vec first-result)))
            (is (= [0.0] (vec destination)) "resident writes do not invent host copyback")
            (is (= 2 (count @ledger)))
            (let [reused (link/realize-plan-storage!
                          plan session :early-stage (link/allocation-groups plan) {} ledger timings
                          (:roots storage))]
              (is (empty? (:owned-specs reused)))
              (is (= 2 (count @ledger)))
              (gpu/download-range! session (get-in reused [:node-views output-id])
                                   after-reuse {:elements 1})
              (is (= [8.0] (vec after-reuse)) "the old zero initializer must not erase completed work"))
            (is (= 0 (:source-cursor initial)))
            (is (not (contains? (get-in ready [:state :realized :buffers]) 'out)))
            (let [checks (atom 0)]
              (is (thrown? ArithmeticException
                           (invocation-link/advance-host-source
                            (-> (:state ready) (dissoc :pending) (assoc :source-cursor 1))
                            (fn [step operands]
                              (swap! checks inc)
                              (scalar/evaluate-invocation-step (:source-ns compilation) step operands))
                            (fn [& _] (throw (ex-info "shared host scalar evaluated twice" {}))) nil)))
              (is (= 1 @checks))
              (is (= 2 (count @ledger)) "the failed check acquired no suffix root")
              (gpu/download-range! session (get-in storage [:node-views output-id])
                                   after-reuse {:elements 1})
              (is (= [8.0] (vec after-reuse)) "the late failure preserves completed earlier writes")))
          (finally (program/release-prepared! prepared))))
      (finally (gpu/close-session! session)))))

(deftest opencl-early-stage-completes-without-materializing-the-late-overflow
  (if @ocl-probe/opencl-available?
    (run-early-stage! :ocl:0)
    (ocl-probe/opencl-skip! "ordered invocation early stage and root reuse")))

(deftest level-zero-early-stage-completes-without-materializing-the-late-overflow
  (if @ze-probe/gpu-available?
    (run-early-stage! :ze:0)
    (ze-probe/gpu-skip! "ordered invocation early stage and root reuse")))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- check-source-allocation-zeroing! [target]
  (let [session (gpu/make-session target)]
    (try
      (gpu/alloc! session {:empty [:byte 0 (byte-array 0)]
                           :byte-one [:byte 1 (byte-array [7])]
                           :byte-two [:byte 2 (byte-array [7 8])]
                           :half-one [:half 1 (short-array [15360])]
                           :float-one [:float 1 (float-array [7.0])]
                           :double-one [:double 1 (double-array [7.0])]})
      (doseq [key [:empty :byte-one :byte-two :half-one :float-one :double-one]]
        (is (= key (gpu/zero-buffer! session key)))
        (is (every? zero? (seq (gpu/download session key)))))
      (gpu/upload-range! session :byte-two (byte-array [9 8]) {:elements 2})
      (is (= :gpu-zero-partial-view
             (reason #(gpu/zero-buffer! session (gpu/buffer-view session :byte-two {:shape [1]})))))
      (is (= [9 8] (vec (gpu/download session :byte-two))))
      (let [event (gpu/submit-upload-ranges!
                   session [[:byte-one (byte-array [9]) {:elements 1}]])]
        (try
          (is (= :gpu-zero-pending-events (reason #(gpu/zero-buffer! session :byte-one))))
          (gpu/await-event! session event)
          (finally (gpu/release-event! session event))))
      (gpu/zero-buffer! session :byte-one)
      (is (= [0] (vec (gpu/download session :byte-one))))
      (finally (gpu/close-session! session)))))

(deftest opencl-source-initialization-covers-empty-and-subword-roots
  (if @ocl-probe/opencl-available?
    (check-source-allocation-zeroing! :ocl:0)
    (ocl-probe/opencl-skip! "source allocation zero initialization")))

(deftest level-zero-source-initialization-covers-empty-and-subword-roots
  (if @ze-probe/gpu-available?
    (check-source-allocation-zeroing! :ze:0)
    (ze-probe/gpu-skip! "source allocation zero initialization")))

(defn- check-source-position-allocation! [target]
  (let [compilation (equation-first/compile #'allocation-fixture/source-sized-output
                                           {:target target :dtype :double})
        download (fn [session view count]
                   (let [result (double-array count)]
                     (gpu/download-range! session view result {:elements count})
                     (vec result)))]
    (doseq [n [-1 0 3]]
      (let [session (gpu/make-session target)
            ledger (volatile! [])
            timings (volatile! {})
            initial (invocation-link/begin-ordered-realization
                     compilation [(double-array [7.0]) (double-array 1) n] nil)
            early (invocation-link/realize-ready-equation
                   initial (:equation (first (:source-order initial))) nil)
            plan (:stage-plan early)]
        (try
          (let [storage (link/realize-plan-storage!
                         plan session :source-position (link/allocation-groups plan) {} ledger timings)
                run-stage! (fn [ready storage]
                             (let [call (:call ready)
                                   executor (link/resident-program-executor
                                             (:stage-plan ready) session :source-position
                                             (:node-views storage) false false ledger)
                                   prepared (program/prepare-sequence-with!
                                             [{:id :source-stage :kind :graph
                                               :call {:graph (:graph call) :bindings (:buffers call)
                                                      :scalar-values (:scalar-values call)}}]
                                             executor)]
                               (try (program/run-prepared! prepared)
                                    (finally (program/release-prepared! prepared)))))
                checks (atom 0)
                evaluate (fn [expression scalars]
                           (swap! checks inc)
                           (scalar/evaluate-array-size (:source-ns compilation)
                                                       (update-vals scalars :value) expression))]
            (run-stage! early storage)
            ;; This test harness deliberately composes the reviewed internal boundaries. It is
            ;; not evidence that public LinkedExecutable already drives the source cursor.
            (let [state (-> (:state early)
                            (dissoc :pending)
                            (assoc :source-cursor 1 :acquired-storage (set (keys (:roots storage)))))
                  early-view (get-in storage [:node-views (first (:outputs plan))])]
              (if (neg? n)
                (do
                  (is (thrown? NegativeArraySizeException
                               (invocation-link/prepare-source-allocation state evaluate nil)))
                  (is (= 2 (count @ledger)) "a failed source count acquires no suffix allocation"))
                (let [allocation (invocation-link/prepare-source-allocation state evaluate nil)
                      acquired (link/realize-storage-nodes!
                                (:nodes allocation) session :source-position {} ledger timings
                                (:roots storage))
                      token (:storage allocation)
                      output-view (get-in acquired [:node-views token])]
                  (is (= 3 (count @ledger)))
                  (gpu/zero-buffer! session output-view)
                  (is (= (vec (repeat n 0.0)) (download session output-view n)))
                  (let [next-state (-> (:state allocation)
                                       (dissoc :pending)
                                       (assoc :source-cursor 2
                                              :acquired-storage (set (keys (:roots acquired)))))
                        suffix (invocation-link/realize-ready-equation
                                next-state (:equation (nth (:source-order next-state) 2)) nil)
                        suffix-storage (link/realize-plan-storage!
                                        (:stage-plan suffix) session :source-position
                                        (link/allocation-groups (:stage-plan suffix)) {} ledger timings
                                        (:roots acquired))]
                    (is (empty? (:owned-specs suffix-storage)))
                    (run-stage! suffix suffix-storage)
                    (is (= (vec (repeat n 1.0)) (download session output-view n))))))
              (is (= 1 @checks) "the constructor size is evaluated only at its source position")
              (is (= [8.0] (download session early-view 1))
                  "successful allocation, empty output and failed count preserve the early write")))
          (finally (gpu/close-session! session)))))))

(deftest opencl-source-allocation-between-resident-stages
  (if @ocl-probe/opencl-available?
    (check-source-position-allocation! :ocl:0)
    (ocl-probe/opencl-skip! "source-position allocation between resident stages")))

(deftest level-zero-source-allocation-between-resident-stages
  (if @ze-probe/gpu-available?
    (check-source-position-allocation! :ze:0)
    (ze-probe/gpu-skip! "source-position allocation between resident stages")))
