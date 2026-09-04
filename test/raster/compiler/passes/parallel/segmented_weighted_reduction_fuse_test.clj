(ns raster.compiler.passes.parallel.segmented-weighted-reduction-fuse-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-fuse :as fuse]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.reference.segmented-weighted-reduction :as reference]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as array-ops]
            [raster.dl.gsdm :as gsdm]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.numeric]
            [raster.runtime.hardware :as hardware]))

;; These are cross-compilation tests.  Own the target facts instead of inheriting a
;; :ze:0 registration from whichever namespace happened to run first in a monolithic
;; test JVM.  No device or driver is required to derive and emit these schedules.
(use-fixtures
  :once
  (fn [f]
    (hardware/reset-hardware!)
    (hardware/init!)
    (hardware/register-target-device!
     :ze:0
     {:name "Intel(R) Arc(TM) Graphics"
      :capabilities {:total-eus 64
                     :threads-per-eu 8
                     :simd-width 16
                     :subgroup-sizes [16 32]
                     :max-workgroup-size 1024
                     :shared-local-memory 131072}})
    (f)))

(defn- chain
  []
  '(let* [raw (raster.dl.array-ops/indexed-dot
               Q K dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          weights (raster.dl.array-ops/scale-clamp-exp
                   raw (raster.numeric// 1.0 (raster.numeric/sqrt dk))
                   5.0 (clojure.core/* n-edges n-heads))
          denominator (raster.dl.array-ops/scatter-add
                       weights dst n-nodes n-edges n-heads)
          weighted (raster.dl.array-ops/scatter-mul-add
                    weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          normalized (raster.dl.array-ops/segment-div
                      weighted denominator n-nodes emb-dim n-heads 1.0e-6)]
         normalized))

(deftm resident-structured-reduction-probe
  [Q :- (Array float) K :- (Array float) V :- (Array float)
   dst :- (Array long) src :- (Array long)
   n-nodes :- Long n-edges :- Long emb-dim :- Long n-heads :- Long]
  :- (Array float)
  (let [dk (quot emb-dim n-heads)
        raw (array-ops/indexed-dot
             Q K dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
        weights (array-ops/scale-clamp-exp
                 raw (/ 1.0 (raster.numeric/sqrt dk)) 5.0 (* n-edges n-heads))
        denominator (array-ops/scatter-add weights dst n-nodes n-edges n-heads)
        weighted (array-ops/scatter-mul-add
                  weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
        normalized (array-ops/segment-div
                    weighted denominator n-nodes emb-dim n-heads 1.0e-6)]
    normalized))

(deftest proven-region-becomes-one-schedule-neutral-marker
  (let [{:keys [form stats]} (fuse/fuse (chain) :float)
        pairs (mapv vec (partition 2 (second form)))
        marker (second (last pairs))
        plan (fuse/marker-plan marker :float)]
    (is (= {:segmented-weighted-reductions-fused 1} stats))
    (is (= 2 (count pairs)) "one allocation plus one fused call replace five operations")
    (is (= 'normalized (ffirst (take-last 1 pairs))))
    (is (fuse/marker? marker))
    (is (= '[Q K V dst src] (mapv :id (:operands plan))))
    (is (= '[n-nodes n-edges emb-dim n-heads dk]
           (:runtime-parameters plan)))
    (is (= (first (first pairs)) (get-in plan [:output :id])))
    (is (= :recognized-indexed-attention-chain
           (get-in plan [:provenance :lowering])))))

(deftest marker-rebinds-generic-operands-and-runtime-parameters
  (let [{:keys [form]} (fuse/fuse (chain) :float)
        marker (second (last (mapv vec (partition 2 (second form)))))
        rebound (with-meta
                  (list fuse/marker-op 'out2 '[q2 k2 v2 dst2 src2]
                        '[(long entities2) edges2 width2 heads2 components2])
                  (meta marker))
        plan (fuse/marker-plan rebound :float)]
    (is (= '[q2 k2 v2 dst2 src2] (mapv :id (:operands plan))))
    (is (= 'out2 (get-in plan [:output :id])))
    (is (= '[entities2 edges2 width2 heads2 components2]
           (:runtime-parameters plan)))
    (is (= 'entities2 (get-in plan [:segment-axes 0 :extent])))
    (is (= 'width2 (get-in plan [:storage :total-dim])))))

(deftest an-unproved-chain-remains-the-identical-fallback-form
  (let [original (chain)
        mismatched (assoc (vec (second original)) 7
                          '(raster.dl.array-ops/scatter-add
                            weights other-dst n-nodes n-edges n-heads))
        original (list 'let* mismatched 'normalized)
        result (fuse/fuse original :float)]
    (is (identical? original (:form result)))
    (is (= 0 (get-in result [:stats :segmented-weighted-reductions-fused])))))

(deftest unresolved-ad-keeps-the-compositional-region-visible
  (let [[op bindings] (chain)
        original (list op bindings '(raster.ad.reverse/value-and-grad normalized))
        result (fuse/fuse original :float)]
    (is (identical? original (:form result)))
    (is (= {:segmented-weighted-reductions-fused 0
            :declined-unresolved-ad-boundary 1}
           (:stats result)))))

(deftest compiler-pass-is-gpu-only
  (let [opts {:inline? false :simd? false :dtype :float}
        cpu (pipeline/run-passes (chain) [:lower :region-copy :structured-reduction-fuse] opts)
        gpu (pipeline/run-passes (chain) [:lower :region-copy :structured-reduction-fuse]
                                 (assoc opts :target-device :ze:0))
        marker-count #(count (filter fuse/marker? (tree-seq coll? seq %)))]
    (is (zero? (marker-count cpu)))
    (is (= 1 (marker-count gpu)))))

(deftest compiled-production-path-selects-one-artifact-backed-step
  (let [descriptor (pipeline/compile-gpu-program
                    #'resident-structured-reduction-probe :ze:0 :dtype :float)
        step (first (:steps descriptor))
        args [(float-array 15) (float-array 15) (float-array 15)
              (long-array 4) (long-array 4) 3 4 5 2]
        call-arguments
        (mapv (fn [{:keys [kind type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn args)}
                  (Object.)))
              (:argument-specs step))
        wide-args (assoc args 7 512)
        wide-call-arguments
        (mapv (fn [{:keys [kind type value-fn]}]
                (if (= :scalar kind)
                  {:type type :value (value-fn wide-args)}
                  (Object.)))
              (:argument-specs step))
        call (kcall/make (:artifact step) call-arguments)]
    (is (= 1 (count (:steps descriptor))))
    (is (= 1 (count (:allocs descriptor))))
    (is (= :contract (:convention step)))
    (is (kdispatch/kernel-dispatch? (:dispatch step)))
    (is (kart/kernel-artifact? (:artifact step)))
    (is (= :indexed-segmented-reduction-reference
           (get-in step [:artifact :attributes :strategy])))
    (is (= [] (get-in step [:artifact :attributes :materialized-intermediates])))
    (is (= [:input :input :input :input :input :output
            :scalar :scalar :scalar :scalar :scalar :scalar]
           (mapv (comp :kind :slot) (:argument-specs step))))
    (is (= [1 3] (get-in call [:geometry :group-count])))
    (is (= :indexed-segmented-reduction-reference
           (kdispatch/alternative-strategy
            (kdispatch/select-alternative (:dispatch step) call-arguments))))
    (is (= :indexed-segmented-reduction-subgroup-score-reuse
           (kdispatch/alternative-strategy
            (kdispatch/select-alternative (:dispatch step) wide-call-arguments))))
    (is (= (:sym (first (:allocs descriptor))) (:result-sym descriptor)))))

(deftest compiler-schedule-can-pin-either-dispatch-alternative
  (doseq [[requested emitted]
          [[:reference :indexed-segmented-reduction-reference]
           [:subgroup-score-reuse
            :indexed-segmented-reduction-subgroup-score-reuse]]]
    (let [descriptor
          (pipeline/compile-gpu-program
           #'resident-structured-reduction-probe :ze:0 :dtype :float
           :schedule {:segmented-weighted-reduction {:strategy requested}})
          step (first (:steps descriptor))]
      (is (nil? (:dispatch step)))
      (is (= emitted (get-in step [:artifact :attributes :strategy]))))))

(deftest compiler-schedule-can-bake-an-offline-measured-selector
  (let [analytic (pipeline/compile-gpu-program
                  #'resident-structured-reduction-probe :ze:0 :dtype :float)
        argument (get-in analytic [:steps 0 :dispatch :selector :argument])
        dispatch-id (get-in analytic [:steps 0 :dispatch :id])
        selector {:kind :runtime-scalar-ranges
                  :argument argument
                  :below :indexed-segmented-reduction-reference
                  :ranges [{:at-least 3
                            :strategy
                            :indexed-segmented-reduction-subgroup-score-reuse}]}
        descriptor
        (pipeline/compile-gpu-program
         #'resident-structured-reduction-probe :ze:0 :dtype :float
         :schedule {:segmented-weighted-reduction
                    {:strategy :auto :measured-selectors {dispatch-id selector}}})
        step (first (:steps descriptor))
        arguments-for
        (fn [args]
          (mapv (fn [{:keys [kind type value-fn]}]
                  (if (= :scalar kind)
                    {:type type :value (value-fn args)}
                    (Object.)))
                (:argument-specs step)))
        narrow [(float-array 15) (float-array 15) (float-array 15)
                (long-array 4) (long-array 4) 3 4 5 2]
        wide (assoc narrow 7 8)]
    (is (= dispatch-id (get-in step [:dispatch :id])))
    (is (= selector (get-in step [:dispatch :selector])))
    (is (= :measured-runtime-shape (get-in step [:dispatch :attributes :selection])))
    (is (= :indexed-segmented-reduction-reference
           (kdispatch/alternative-strategy
            (kdispatch/select-alternative (:dispatch step) (arguments-for narrow)))))
    (is (= :indexed-segmented-reduction-subgroup-score-reuse
           (kdispatch/alternative-strategy
            (kdispatch/select-alternative (:dispatch step) (arguments-for wide)))))))

(deftest compiled-dispatch-projects-resident-abi-and-reference-environment
  (let [descriptor (pipeline/compile-gpu-program
                    #'resident-structured-reduction-probe :ze:0 :dtype :float)
        args [(float-array 15) (float-array 15) (float-array 15)
              (long-array 4) (long-array 4) 3 4 5 2]
        param->key (into {} (map (fn [sym] [sym (keyword (str "resident-" (name sym)))]))
                         (:array-params descriptor))
        alloc->key (into {} (map (fn [{:keys [sym]}] [sym (keyword (str "scratch-" (name sym)))])
                                 (:allocs descriptor)))
        capacities
        (merge (into {} (map (fn [sym]
                               [(param->key sym)
                                (alength (get (zipmap (:all-params descriptor) args) sym))]))
                     (:array-params descriptor))
               (into {} (map (fn [{:keys [sym size-fn]}]
                               [(alloc->key sym) (size-fn args)]))
                     (:allocs descriptor)))
        node-id (merge param->key alloc->key)
        views (into {}
                    (map (fn [[sym key]]
                           [key (gpu/->ResidentBufferView
                                 :test-session key
                                 {:byte-length (* 4 (get capacities key)) :dtype :float})]))
                    node-id)
        executable
        (link/map->LinkedExecutable
         {:plan {:instances [{:id :probe :descriptor descriptor :bindings node-id}]}
          :session (atom {}) :node-views views :closed? (atom false)})
        projected (link/dispatch-arguments executable args)
        contract (get-in projected [:dispatch :attributes :tuning])
        expected (reference/evaluate (get-in contract [:reference :plan])
                                     (:reference-inputs projected))]
    (is (= 0 (:step-index projected)))
    (is (= (mapv (comp views param->key) (take 5 (:array-params descriptor)))
           (subvec (:arguments projected) 0 5)))
    (is (= (views (alloc->key (:result-sym descriptor))) (nth (:arguments projected) 5)))
    (is (= (views (alloc->key (:result-sym descriptor)))
           (get-in projected [:resident-bindings (:result-sym descriptor)])))
    (is (= [3 4 5 2 2 15]
           (mapv :value (subvec (:arguments projected) 6))))
    (is (identical? (first args)
                    (get-in projected [:reference-inputs :buffers 'Q])))
    (is (= {'n-nodes 3 'n-edges 4 'emb-dim 5 'n-heads 2 'dk 2}
           (select-keys (get-in projected [:reference-inputs :scalars])
                        '[n-nodes n-edges emb-dim n-heads dk])))
    (is (= [:segmented-weighted-reduction :measured-selectors]
           (:schedule-path contract)))
    (is (= (get-in descriptor [:steps 0 :dispatch :id]) (:schedule-key contract)))
    (is (= :segmented-weighted-reduction (get-in contract [:reference :kind])))
    (is (= :float (get-in contract [:numerical-mode :accumulate])))
    (is (= :indexed-dense-values (get-in contract [:layout :storage :kind])))
    (is (= :edge-list-by-destination (get-in contract [:layout :membership :kind])))
    (is (= 15 (alength ^doubles expected)))
    (is (every? zero? expected))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"exceeds its resident node view"
         (link/dispatch-arguments executable (assoc args 7 512))))))

(deftest actual-gsdm-region-reaches-generic-structured-reduction-stage
  (let [diagnostic (pipeline/show-pipeline
                    #'gsdm/graph-attention-multihead
                    :dtype :float :simd? false :target-device :ze:0)
        kernels
        (filterv #(= :indexed-segmented-reduction-reference
                     (get-in % [:attributes :strategy]))
                 (:kernels diagnostic))]
    (is (= 1 (get-in diagnostic
                     [:structured-reductions-stats
                      :segmented-weighted-reductions-fused])))
    (is (= 1 (count kernels)))
    (is (true? (get-in (first kernels) [:attributes :dynamic-shape?])))
    (is (= [] (get-in (first kernels) [:attributes :materialized-intermediates])))))
