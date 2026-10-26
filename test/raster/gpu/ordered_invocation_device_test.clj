(ns raster.gpu.ordered-invocation-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.ir.ordered-invocation-realization-test :as allocation-fixture]
            [raster.compiler.fixtures.symbolic-storage :as symbolic-storage]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
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

(defn- check-ordered-prefix-returns! [target]
  (doseq [[source parameter zero-tail?]
          [[#'allocation-fixture/ordered-prefix-buffer-return 'out false]
           [#'allocation-fixture/ordered-prefix-zero-buffer-return 'early true]]]
    (let [compilation (equation-first/compile
                       source {:target target :dtype :double
                               :values {parameter (av/tensor {:dtype :double :shape [8]})}})]
      (is (true? (:required? (invocation-link/ordered-preparation-requirement compilation)))
          "the retained late allocation forces actual ordered preparation")
      (doseq [n [-1 0 3]]
        (let [input (double-array [10 11 12 13 14 15 16 17])
              expected (into [1.0 2.0] (if zero-tail? (repeat 6 0.0) [12.0 13.0 14.0 15.0 16.0 17.0]))
              executable (link/instantiate-ordered-invocation! compilation [input n] {})]
          (try
            (if (neg? n)
              (do
                (is (thrown? NegativeArraySizeException (link/run! executable)))
                (let [runtime (:ordered @(:execution-state executable))
                      token (get-in runtime [:current :realized :buffers 'out])
                      actual (double-array 8)]
                  (gpu/download-range! (:session executable) (get-in runtime [:node-views token])
                                       actual {:elements 8})
                  (is (= expected (vec actual))
                      "the prefix completed before the late failure, preserving the initialized tail")
                  (is (false? (boolean (:resident-boundary @(:execution-state executable)))))
                  (is (zero? @(:completed-replays executable)))
                  (is (= :link-execution-poisoned (reason #(link/outputs executable))))))
              (let [outputs (vec (keys (link/run! executable)))
                    [output late] outputs
                    plan (get-in @(:execution-state executable) [:resident-boundary :plan])
                    view (get-in plan [:nodes output :view])]
                (is (= 2 (count outputs)))
                (is (= [8] (:shape view)))
                (is (= expected (vec (link/download executable output))))
                (is (= [n] (get-in plan [:nodes late :view :shape])))
                (is (= (vec (repeat n 0.0)) (vec (link/download executable late))))
                (is (= [10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0] (vec input))
                    "resident completion does not invent host copyback")
                (if zero-tail?
                  ;; The next replay must execute the source constructor again, not merely
                  ;; reuse initialized bytes from the preceding partial write.
                  (gpu/upload-range! (:session executable) (get (link/outputs executable) output)
                                     (double-array (repeat 8 99.0)) {:elements 8})
                  ;; A captured state buffer must not re-upload its old host initializer.
                  (aset input 7 99.0))
                (link/run! executable)
                (is (= 2 @(:completed-replays executable)))
                (is (= expected (vec (link/download executable output))))
                (is (= (vec (repeat n 0.0)) (vec (link/download executable late))))))
            (finally (link/close! executable))))))))

(defn- check-unused-constructor! [target]
  (let [compilation (equation-first/compile
                     #'allocation-fixture/ordered-unused-constructor-return
                     {:target target :dtype :double
                      :values {'out (av/tensor {:dtype :double :shape [8]})}})]
    (is (true? (:required? (invocation-link/ordered-preparation-requirement compilation))))
    (doseq [n [-1 0 3]]
      (let [input (double-array [10 11 12 13 14 15 16 17])
            executable (link/instantiate-ordered-invocation! compilation [input n] {})]
        (try
          (if (neg? n)
            (do
              (is (thrown? NegativeArraySizeException (link/run! executable)))
              (let [runtime (:ordered @(:execution-state executable))
                    token (get-in runtime [:current :realized :buffers 'out])
                    actual (double-array 8)]
                (gpu/download-range! (:session executable) (get-in runtime [:node-views token])
                                     actual {:elements 8})
                (is (= [1.0 2.0 12.0 13.0 14.0 15.0 16.0 17.0] (vec actual))
                    "the earlier resident write survives a dead constructor's failure"))
              (is (zero? @(:completed-replays executable)))
              (is (= :link-execution-poisoned (reason #(link/outputs executable)))))
            (let [output (first (keys (link/run! executable)))]
              (is (= [1.0 2.0 12.0 13.0 14.0 15.0 16.0 17.0]
                     (vec (link/download executable output))))))
          (finally (link/close! executable)))))))

(deftest opencl-unused-constructor-still-executes-at-source-position
  (if @ocl-probe/opencl-available?
    (check-unused-constructor! :ocl:0)
    (ocl-probe/opencl-skip! "unused source constructor failure")))

(deftest level-zero-unused-constructor-still-executes-at-source-position
  (if @ze-probe/gpu-available?
    (check-unused-constructor! :ze:0)
    (ze-probe/gpu-skip! "unused source constructor failure")))

(deftest opencl-ordered-prefix-returns-preserve-whole-source-buffers
  (if @ocl-probe/opencl-available?
    (check-ordered-prefix-returns! :ocl:0)
    (ocl-probe/opencl-skip! "ordered prefix views and whole source returns")))

(deftest level-zero-ordered-prefix-returns-preserve-whole-source-buffers
  (if @ze-probe/gpu-available?
    (check-ordered-prefix-returns! :ze:0)
    (ze-probe/gpu-skip! "ordered prefix views and whole source returns")))

(defn- check-symbolic-prefix-ranges! [target]
  (doseq [[capacity n m extra prepared-stages expected-reason]
          [[3 4 4 {} 0 :program-link-graph-range]
           [6 6 7 {} 1 :program-link-graph-range]
           ;; Static facts may add an earlier guarded kernel precondition, never permission
           ;; to erase the original pointwise read minimum.
           [3 4 4 {:values {'x (av/tensor {:dtype :float :shape [3]})}} 0 :kernel-precondition-failed]]]
    (let [input (float-array (range capacity))
          original (vec input)
          prepared (compiled/lower #'symbolic-storage/prefix-map [input n m]
                                   (merge {:compiler :equation-first :target target :dtype :float} extra))
          artifact (compiled/instantiate! prepared)
          preparations (atom 0)
          prepare program/prepare-sequence-with-retained-programs!]
      (try
        (with-redefs [program/prepare-sequence-with-retained-programs!
                      (fn [& args] (swap! preparations inc) (apply prepare args))]
          (is (= expected-reason (reason #(artifact)))
              "the current binding checks actual capacity even with supplied static facts"))
        (is (= prepared-stages @preparations)
            "range refusal precedes preparation/submission of the invalid stage")
        (is (zero? @(:completed-replays (:executable artifact))))
        (is (= original (vec input)))
        (is (= :link-execution-poisoned (reason #(artifact))))
        (finally (compiled/close! artifact))))))

(deftest symbolic-prefix-ranges-fail-before-the-invalid-stage
  (if @ocl-probe/opencl-available?
    (check-symbolic-prefix-ranges! :ocl:0)
    (ocl-probe/opencl-skip! "ordered symbolic prefix range failures"))
  (if @ze-probe/gpu-available?
    (check-symbolic-prefix-ranges! :ze:0)
    (ze-probe/gpu-skip! "ordered symbolic prefix range failures")))
