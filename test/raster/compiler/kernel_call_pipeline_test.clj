(ns raster.compiler.kernel-call-pipeline-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as ra]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attention]
            [raster.dl.nn :as nn]))

(deftm resident-kernel-call-map
  [x :- (Array float) out :- (Array float) scale :- Float n :- Long] :- (Array float)
  (raster.par/map! out i n float (* scale (ra/aget x i))))

(deftm resident-kernel-call-inout
  [state :- (Array float) scale :- Float n :- Long] :- (Array float)
  (raster.par/map! state i n float (* scale (ra/aget state i))))

(deftm resident-kernel-call-reduce
  [x :- (Array float) out :- (Array float) scale :- Double n :- Long] :- Void
  (let [sum (raster.par/reduce acc 0.0 i n
                               (+ acc (float (* scale (ra/aget x i)))))]
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

(deftm packed-gelu-between-contractions
  [x :- (Array float) up-weight :- (Array float) down-weight :- (Array float)
   rows :- Long input-width :- Long hidden-width :- Long output-width :- Long] :- (Array float)
  (let [packed (nn/linear-nb x up-weight rows input-width (* 2 hidden-width))
        hidden (float-array (* rows hidden-width))
        effect (nn/gelu-erf-mul-strided! packed hidden rows (* 2 hidden-width)
                                          0 hidden-width hidden-width)]
    (nn/linear-nb hidden down-weight rows hidden-width output-width)))

(deftest a-packed-multistage-consumer-does-not-erase-either-contraction
  (let [descriptor (pipeline/compile-gpu-program
                    #'packed-gelu-between-contractions :ze:0 :dtype :float
                    :on-non-resident :throw)
        steps (:steps descriptor)]
    (is (= 3 (count steps)))
    (is (= [:executable :map-void :executable] (mapv :convention steps)))
    (is (every? some? (map :artifact steps)))
    (is (= 3 (count (:allocs descriptor)))
        "only the packed projection, fused hidden value and final output are materialized")))

(deftm packed-qkv-consumers
  [x :- (Array float) qkv-weight :- (Array float) scores :- (Array float)
   rows :- Long model-width :- Long heads :- Long head-dim :- Long theta :- Double]
  :- (Array float)
  (let [packed (nn/linear-nb x qkv-weight rows model-width (* 3 model-width))
        head-width (* heads head-dim)
        q (float-array (* rows head-width))
        k (float-array (* rows head-width))
        context (float-array (* rows head-width))
        q-effect (attention/rope-prefill-strided!
                  packed q rows heads head-dim theta (* 3 model-width) 0)
        k-effect (attention/rope-prefill-strided!
                  packed k rows heads head-dim theta (* 3 model-width) model-width)
        context-effect (attention/attn-prefill-out-strided!
                        scores packed context rows heads 1 heads head-dim
                        (* 3 model-width) (* 2 model-width))]
    ;; Keep all three view consumers observable in this compiler regression.  A real attention
    ;; graph consumes Q/K through scores; this test uses a cheap continuation instead.
    (raster.par/map! context i (* rows head-width) float
                     (+ (ra/aget context i) (ra/aget q i) (ra/aget k i)))))

(deftest packed-qkv-views-compose-directly-with-the-projection
  (let [descriptor (pipeline/compile-gpu-program
                    #'packed-qkv-consumers :ze:0 :dtype :float :on-non-resident :throw)]
    (is (= [:executable :map-void :map-void :map-void :map-void :map-void :map]
           (mapv :convention (:steps descriptor))))
    (is (= 4 (count (:allocs descriptor)))
        "the packed projection, rotated Q/K and context are the only resident values")
    (is (= 2 (count (filter #(-> % :artifact :provenance :segop-id str
                                 (.startsWith "rstr_initialization_equation_"))
                            (:steps descriptor))))
        "the dense attention output discharges its initializer; paired RoPE stores retain theirs until even-width is proved")))

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
    (doseq [n [0 1 513]]
      (is (= (inc n)
             (kernel-launch/resolve-expression {'n n}
                                               (:elements (first (:outputs executable)))))))
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

(deftest resident-segred-keeps-its-complete-two-phase-schedule
  (let [descriptor (pipeline/compile-gpu-program #'resident-kernel-call-reduce
                                                 :ze:0 :dtype :float)
        step (first (filter #(= :executable (:convention %)) (:steps descriptor)))
        graph (:artifact step)
        partial (get-in graph [:nodes 0 :operation])
        terminal (get-in graph [:nodes 1 :operation])
        workgroup-size (first (get-in partial [:launch :workgroup-size]))
        n (inc (* 2 workgroup-size))
        runtime-params [(float-array n) (float-array n) 0.75 n]
        ordered-values
        (mapv (fn [{:keys [kind sym type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn runtime-params)}
                  (keyword (name sym))))
              (:argument-specs step))
        scalars (into {} (keep identity)
                      (map (fn [slot value]
                             (when (= :scalar (:kind slot)) [(:name slot) (:value value)]))
                           (:abi graph) ordered-values))]
    (is (kernel-graph/kernel-graph? graph))
    (is (= 2 (count (:nodes graph))))
    (is (= 1 (count (:temporaries graph))))
    (is (= [:input :output :scalar :scalar]
           (mapv :kind (:argument-specs step))))
    (is (= 3 (kernel-launch/resolve-expression
              scalars (first (get-in partial [:launch :group-count])))))
    (is (= [1] (get-in terminal [:launch :group-count])))))

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
    (is (= :kernel-body (get-in step [:artifact :provenance :dialect])))
    (is (= :kernel-body (get-in step [:artifact :attributes :emission-route])))
    (is (= '[x out scale]
           (subvec (kcall/logical-arguments (:artifact step)) 0 3)))
    (is (= [3] (get-in call [:geometry :group-count])))))
