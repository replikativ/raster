(ns raster.compiler.kernel-call-pipeline-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as ra]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]))

(deftm resident-kernel-call-map
  [x :- (Array float) out :- (Array float) scale :- Float n :- Long] :- (Array float)
  (raster.par/map! out i n float (* scale (ra/aget x i))))

(deftm resident-kernel-call-inout
  [state :- (Array float) scale :- Float n :- Long] :- (Array float)
  (raster.par/map! state i n float (* scale (ra/aget state i))))

(deftm resident-kernel-call-reduce
  [x :- (Array float) out :- (Array float) scale :- Double n :- Long] :- Void
  (let [sum (raster.par/reduce acc 0.0 i n
                               (+ acc (* scale (ra/aget x i))))]
    (raster.par/map-void! j n
                          (ra/aset out j (* (ra/aget x j) sum)))))

(deftm resident-kernel-call-map-void
  [x :- (Array float) out :- (Array float) scale :- Float n :- Long] :- Void
  (raster.par/map-void! i n
                        (ra/aset out i (* scale (ra/aget x i)))))

(deftm resident-kernel-call-scan
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/scan out acc 0.0 i n float (+ acc (ra/aget x i))))

(deftm resident-kernel-call-exclusive-scan
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/scan-exclusive out acc 0.0 i n float (+ acc (ra/aget x i))))

(deftest resident-typed-scan-is-one-graph-backed-executable-step
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-scan
                                                 :ze:0 :dtype :float)
        step (first (:steps descriptor))
        executable (:artifact step)]
    (is (= 1 (count (:steps descriptor))))
    (is (= :executable (:convention step)))
    (is (kernel-graph/kernel-graph? executable))
    (is (= 3 (count (:nodes executable))))
    (is (= 1 (count (:temporaries executable))))
    (is (empty? (:allocs descriptor))
        "graph-private scan storage must not leak into program allocations")
    (is (= (:abi executable) (:abi step)))
    (is (= (:arguments executable)
           (mapv (fn [{:keys [kind sym expression]}]
                   (if (= :scalar kind) expression sym))
                 (:argument-specs step))))))

(deftest resident-exclusive-scan-preserves-its-n-plus-one-result-contract
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-exclusive-scan
                                                 :ze:0 :dtype :float)
        step (first (:steps descriptor))
        executable (:artifact step)]
    (is (= [:executable] (mapv :convention (:steps descriptor))))
    (is (= :exclusive (get-in executable [:attributes :scan-mode])))
    (is (= (kernel-launch/sum 'n 1) (:elements (first (:outputs executable)))))
    (is (empty? (:allocs descriptor)))))

(deftest resident-segmap-step-carries-one-executable-call-template
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-map
                                                 :ze:0 :dtype :float)
        step (first (:steps descriptor))
        runtime-params [(float-array 513) (float-array 513) (float 2.0) 513]
        ordered-values
        (mapv (fn [{:keys [kind sym type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn runtime-params)}
                  (keyword (name sym))))
              (:argument-specs step))
        call (kcall/make (:artifact step) ordered-values)]
    (is (= :map (:convention step)))
    (is (kart/kernel-artifact? (:artifact step)))
    (is (= '[x out scale _n_bound]
           (mapv (comp :name :slot) (:argument-specs step))))
    (is (= [:input :output :scalar :scalar]
           (mapv :kind (:argument-specs step))))
    (is (= [256] (get-in call [:geometry :workgroup-size])))
    (is (= [3] (get-in call [:geometry :group-count])))))

(deftest resident-compilation-enriches-the-shared-report-without-a-second-run
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-map
                                                 :ze:0 :dtype :float
                                                 :compiler-report? true)
        report (:compiler-report descriptor)]
    (is (= 1 (:schema-version report)))
    (is (= :opencl (get-in report [:route :backend])))
    (is (= :typed-soac (get-in report [:route :source-dialect])))
    (is (= {:assessed? true
            :resident? true
            :device-scratch-count 0
            :host-array-allocs-in-compute 0
            :internal-host-roundtrips 0}
           (:residency report)))
    (is (= (count (:steps descriptor))
           (get-in report [:emission :kernel-count])))))

(deftest resident-segmap-inout-is-one-physical-result-slot
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-inout
                                                 :ze:0 :dtype :float)
        step (first (:steps descriptor))]
    (is (= :map (:convention step)))
    (is (= '[state scale _n_bound]
           (mapv (comp :name :slot) (:argument-specs step))))
    (is (= [:inout :scalar :scalar]
           (mapv :kind (:argument-specs step))))
    (is (= [:result :parameter :bound]
           (mapv (comp :role :slot) (:argument-specs step))))
    (is (= {'state :output} (:array-roles descriptor)))
    (is (= :state (:output step)))))

(deftest resident-segred-schedule-is-an-explicit-kernel-call-override
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-reduce
                                                 :ze:0 :dtype :float)
        step (first (filter #(= :reduce (:convention %)) (:steps descriptor)))
        runtime-params [(float-array 513) (float-array 513) 0.75 513]
        ordered-values
        (mapv (fn [{:keys [kind sym type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn runtime-params)}
                  (keyword (name sym))))
              (:argument-specs step))
        default-call (kcall/make (:artifact step) ordered-values)
        resident-call (kcall/make (:artifact step) ordered-values {:group-count [1]})]
    (is (kart/kernel-artifact? (:artifact step)))
    (is (= [:input :output :scalar :scalar]
           (mapv :kind (:argument-specs step))))
    (is (= [3] (get-in default-call [:geometry :group-count])))
    (is (= [1] (get-in resident-call [:geometry :group-count])))))

(deftest resident-map-void-step-carries-a-logical-plan-for-one-physical-call
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-map-void
                                                 :ze:0 :dtype :float)
        step (first (:steps descriptor))
        runtime-params [(float-array 513) (float-array 513) (float 2.0) 513]
        logical-values
        (mapv (fn [{:keys [kind sym type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn runtime-params)}
                  (keyword (name sym))))
              (:argument-specs step))
        physical-values (kcall/expand-logical-arguments
                         (:artifact step) logical-values (fn [_ value] [value]))
        call (kcall/make (:artifact step) physical-values)]
    (is (= :map-void (:convention step)))
    (is (:logical-bindings? step))
    (is (kart/kernel-artifact? (:artifact step)))
    (is (= :segmap (get-in step [:artifact :provenance :dialect])))
    (is (= '[x out scale]
           (subvec (kcall/logical-arguments (:artifact step)) 0 3)))
    (is (= [3] (get-in call [:geometry :group-count])))))
