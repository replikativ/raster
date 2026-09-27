(ns raster.gpu.program-memory-order-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.nn :as nn]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.analysis.physical-liveness :as liveness]
            [raster.compiler.ir.link-plan :as plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as program]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as device]))

(deftm four-layers
  [w :- (Array double) b :- (Array double) x :- (Array double)] :- (Array double)
  (let [a (nn/dense w x b)
        middle (nn/dense w a b)
        c (nn/dense w middle b)]
    (nn/dense w c b)))

(defn- arguments []
  [(double-array [1.0 0.25 -0.5 1.0])
   (double-array [0.25 -0.125])
   (double-array [1.0 -2.0])])

(defn- lowered [target args]
  (equation/lower (equation/compile #'four-layers {:target target :dtype :double}) args))

(defn- assert-candidate [linked order]
  (let [report (liveness/report linked order)
        candidates (:proposals report)]
    (is (seq candidates) "real typed writes and selected execution order must meet")
    (doseq [candidate candidates]
      (is (= :witnessed (get-in candidate [:runtime-order :status])))
      (is (= :witnessed (:cross-replay-initialization candidate)))
      (is (= #{:alias-realization :completion-and-escape} (:pending candidate))))
    (is (= :unproven (:reuse report)))
    (is (= :unproven (:completion order)))))

(deftest typed-program-order-meets-complete-write-evidence
  (let [linked (lowered :ocl:0 (arguments))
        call (get-in linked [:instances 0 :call])
        visited (atom [])
        prepared (program/prepare-with!
                  call {:bind! (fn [key graph _ _] {:key key :graph graph})
                        :run! #(swap! visited conj (:key %))
                        :release! (fn [_])})
        executable (link/map->LinkedExecutable
                    {:plan linked :prepared-program prepared
                     :closed? (atom false) :pending-inputs (atom #{})})]
    (try
      (is (= :parallel-program-graph-order
             (try (program/execution-order prepared (constantly nil))
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (with-redefs [gpu/kernel-graph-execution-order
                    (fn [_ handle]
                      {:record-time-prologue []
                       :per-replay (mapv #(hash-map :kernel-phase (:id %))
                                         (get-in handle [:graph :nodes]))})]
        (let [order (link/execution-order executable)]
          (assert-candidate linked order)
          (program/run-prepared! prepared)
          (is (= (:binding-order prepared) @visited))
          (is (= (set (keys (get-in prepared [:plan :step-keys])))
                 (set (map #(get-in % [:source :step]) (:per-replay order)))))
          (is (every? :complete-write?
                      (filter #(= :write (:access %)) (:accesses (plan/memory-report linked)))))))
      (with-redefs [gpu/kernel-graph-execution-order
                    (fn [_ handle]
                      {:record-time-prologue [{:kernel-phase :hoisted}]
                       :per-replay (mapv #(hash-map :kernel-phase (:id %))
                                         (get-in handle [:graph :nodes]))})]
        (is (every? #(= :record-time-prologue (get-in % [:runtime-order :reason]))
                    (:proposals (liveness/report linked (link/execution-order executable))))))
      (finally (program/release-prepared! prepared)))
    (is (= :parallel-program-closed
           (try (program/execution-order prepared identity)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest selected-program-order-is-observation-only-on-local-devices
  (doseq [[target available? skip!] [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
                                    [:ze:0 device/gpu-available? device/gpu-skip!]]]
    (if-not @available?
      (skip! (str "equation-first memory witness on " target))
      (let [args (arguments)
            expected (vec (apply four-layers args))
            linked (lowered target args)]
        (with-open [executable (link/instantiate! linked)]
          (assert-candidate linked (link/execution-order executable))
          (dotimes [_ 2]
            (link/run! executable)
            (is (= expected (vec (link/download executable (first (:outputs linked))))))))))))
