(ns raster.compiler.passes.parallel.segmented-weighted-reduction-fuse-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [clojure.walk :as walk]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.equation-artifact :as equation-artifact]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-fuse :as fuse]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.reference.segmented-weighted-reduction :as reference]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as array-ops]
            [raster.dl.gsdm :as gsdm]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.numeric]
            [raster.runtime.hardware :as hardware]))

;; These are cross-compilation tests.  Own the target facts instead of inheriting a
;; :ze:0 registration from whichever namespace happened to run first in a monolithic
;; test JVM.  No device or driver is required to derive and emit these schedules.
(use-fixtures
  :once
  hardware-fixture/isolated
  (fn [f]
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

(deftest protected-marker-enters-the-shared-typed-program-only-when-enabled
  (let [{:keys [form]} (fuse/fuse (chain) :float)
        marker (second (last (mapv vec (partition 2 (second form)))))
        plan (fuse/marker-plan marker :float)
        array-types (into {} (map (juxt :id :dtype)) (:operands plan))
        scalar-types (zipmap (filter symbol? (swr/runtime-parameter-values plan))
                             (repeat :long))
        values (into {} (map (fn [{:keys [id dtype elements]}]
                              [id (av/tensor {:dtype dtype :shape [elements]})]))
                     (:operands plan))
        disabled (typed-route/attempt form :float array-types
                                      {:scalar-types scalar-types :values values})
        enabled (typed-route/attempt form :float array-types
                                     {:scalar-types scalar-types :values values
                                      :segmented-plans? true})
        program (:program enabled)
        plan-equation (some #(when (swr/plan? (:algorithm %)) %) (:equations program))]
    (is (nil? disabled) "the existing descriptor route remains selected by default")
    (is (= :typed-parallel (:dialect program)))
    (is (nil? (get-in enabled [:stats :typed-validated]))
        "the pipeline, not the cyclic frontend route, owns mixed-union validation")
    (is (= plan (:algorithm plan-equation)))
    (is (= (vec (distinct (concat (swr/ordered-input-ids plan)
                                  (filter symbol? (swr/runtime-parameter-values plan)))))
           (:operands plan-equation)))
    (is (= '[normalized] (:results plan-equation)))
    (is (identical? program (structured-route/validate-typed-program! program)))
    (is (= [plan] (:operations plan-equation)))))

(deftest protected-plan-preserves-ordinary-equations-on-both-sides
  (let [{:keys [form]} (fuse/fuse (chain) :float)
        [[storage allocation] [result marker]] (mapv vec (partition 2 (second form)))
        source (list 'let*
                     ['q-ready '(raster.par/map! Q i n-nodes float
                                                  (clojure.core/aget Q 0))
                      storage allocation
                      result marker
                      'post '(raster.par/pmap j n-nodes float
                                              (clojure.core/aget normalized 0))]
                     'post)
        plan (fuse/marker-plan marker :float)
        array-types (into {} (map (juxt :id :dtype)) (:operands plan))
        scalar-types (zipmap (filter symbol? (swr/runtime-parameter-values plan))
                             (repeat :long))
        values (into {} (map (fn [{:keys [id dtype elements]}]
                              [id (av/tensor {:dtype dtype :shape [elements]})]))
                     (:operands plan))
        program (:program (typed-route/attempt
                           source :float array-types
                           {:scalar-types scalar-types :values values
                            :segmented-plans? true}))
        descriptor-attempt (typed-route/attempt
                            source :float array-types
                            {:scalar-types scalar-types :values values})
        algorithms (filterv
                    #(or (swr/plan? %)
                         (some (fn [equation] (not= 'scalar (soac/operation-kind equation)))
                               (soac/equations %)))
                    (map :algorithm (:equations program)))]
    (is (= 3 (count algorithms)))
    (is (= :typed-soac (get-in descriptor-attempt [:stats :route])))
    (is (some fuse/marker? (tree-seq coll? seq (get-in descriptor-attempt [:program :source])))
        "descriptor compilation keeps the protected binding between its typed SOAC islands")
    (is (soac/program-form? (first algorithms)))
    (is (= plan (second algorithms)))
    (is (soac/program-form? (last algorithms)))
    (is (= 1 (count (get-in program [:attributes :allocations]))))
    (is (every? empty?
                (map #(get-in (soac/facts %) [:attributes :allocations])
                     [(first algorithms) (last algorithms)]))
        "the plan output allocation is not replayed by adjacent SOAC initialization passes")
    (is (= '[post] (:outputs program)))
    (is (identical? program (structured-route/validate-typed-program! program)))))

(deftest functional-producer-is-a-live-plan-operand
  (let [static-chain (walk/postwalk-replace
                      {'n-nodes 2 'n-edges 3 'emb-dim 5 'n-heads 1 'dk 5}
                      (chain))
        {fused :form} (fuse/fuse static-chain :float)
        [[storage allocation] [result marker]] (mapv vec (partition 2 (second fused)))
        [_ marker-output marker-inputs runtime-values] marker
        rebound-marker (with-meta
                         (list fuse/marker-op marker-output
                               (assoc marker-inputs 0 'q-ready) runtime-values)
                         (meta marker))
        plan (fuse/marker-plan rebound-marker :float)
        source (list 'let*
                     ['q-ready '(raster.par/pmap i 10 float (clojure.core/aget Q i))
                      storage allocation
                      result rebound-marker
                      'post '(raster.par/pmap j 2 float
                                              (clojure.core/aget normalized 0))]
                     'post)
        external-descriptors (remove #(= 'q-ready (:id %)) (:operands plan))
        array-types (into {} (map (juxt :id :dtype)) external-descriptors)
        values (into {} (map (fn [{:keys [id dtype elements]}]
                              [id (av/tensor {:dtype dtype :shape [elements]})]))
                     external-descriptors)
        program (:program (typed-route/attempt source :float array-types
                                               {:values values :segmented-plans? true}))
        [producer protected consumer] (:equations program)]
    (is (soac/program-form? (:algorithm producer)))
    (is (= '[q-ready] (:results producer)))
    (is (= 'q-ready (first (:operands protected))))
    (is (= plan (:algorithm protected)))
    (is (soac/program-form? (:algorithm consumer)))
    (is (some #{'normalized} (:operands consumer)))
    (is (= '[post] (:outputs program)))
    (is (identical? program (structured-route/validate-typed-program! program)))))

(deftest public-equation-first-emits-the-protected-plan
  (let [compilation (equation-first/compile
                     #'resident-structured-reduction-probe {:target :ze:0 :dtype :float})
        semantic-plan (:algorithm (last (:equations (:semantic compilation))))
        scheduled-plan (:algorithm (last (:equations (:scheduled compilation))))]
    (is (swr/plan? semantic-plan))
    (is (= semantic-plan scheduled-plan))
    (is (= :indexed-segmented-reduction-reference
           (get-in compilation [:scheduled :equations
                                (dec (count (get-in compilation [:scheduled :equations])))
                                :operations 0 :attributes :strategy])))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (= 1 (count (:kernels compilation))))))

(deftest public-equation-first-selects-only-explicit-subgroup-policy
  (let [compilation (equation-first/compile
                     #'resident-structured-reduction-probe
                     {:target :ze:0 :dtype :float
                      :schedule {:segmented-weighted-reduction {:strategy :subgroup-score-reuse}}})
        operation (first (:operations (last (:equations (:emitted compilation)))))
        index (dec (count (get-in operation [:body :equations])))
        certificate-path [:body :equations index :operations 0]
        certificate (get-in operation certificate-path)
        arguments [(float-array 15) (float-array 15) (float-array 15)
                   (long-array [0 0 2 2]) (long-array [1 1 0 2]) 3 4 5 2]
        identity {:semantic-request-fingerprint "subgroup-public-test"
                  :compiler-build-fingerprint "test-build"
                  :source-dependency-fingerprint "test-source"
                  :target-descriptor-fingerprint "test-target"}
        restored (equation-artifact/open
                  identity
                  (equation-artifact/decode
                   (equation-artifact/encode (equation-artifact/seal identity compilation))))
        reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= :reassociated (get-in certificate [:numerics :mode])))
    (is (= :indexed-segmented-reduction-subgroup-score-reuse
           (get-in certificate [:attributes :strategy])))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (= 1 (count (:kernels compilation))))
    (is (seq (emitted-equation/complete-write-domains operation)))
    (is (= (:outputs (equation-first/lower compilation arguments))
           (:outputs (equation-first/lower restored arguments))))
    (doseq [damaged [(update-in operation (conj certificate-path :body :operations) pop)
                     (assoc-in operation (conj certificate-path :numerics)
                               {:mode :exact :policy :same-typed-ssa-evaluation-order})]]
      (is (= :emitted-reduction-subgroup-refinement
             (reason #(emitted-equation/complete-write-domains damaged)))))
    (is (= :kernel-precondition-failed
           (reason #(equation-first/lower compilation (assoc arguments 8 6)))))))

(deftest public-protected-plan-validates-runtime-storage-before-allocation
  (let [compilation (equation-first/compile
                     #'resident-structured-reduction-probe {:target :ze:0 :dtype :float})
        arguments [(float-array 15) (float-array 15) (float-array 15)
                   (long-array [0 0 2 2]) (long-array [1 1 0 2]) 3 4 5 2]
        linked (equation-first/lower compilation arguments)
        identity {:semantic-request-fingerprint "segmented-public-test"
                  :compiler-build-fingerprint "test-build"
                  :source-dependency-fingerprint "test-source"
                  :target-descriptor-fingerprint "test-target"}
        restored (equation-artifact/open
                  identity
                  (equation-artifact/decode
                   (equation-artifact/encode (equation-artifact/seal identity compilation))))]
    (is (= 0 (get-in linked [:attributes :driver-allocations])))
    (is (= 1 (count (:outputs linked))))
    (is (= (:outputs linked) (:outputs (equation-first/lower restored arguments))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (equation-first/lower compilation (assoc arguments 2 (float-array 14)))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (equation-first/lower compilation (assoc arguments 8 6))))))

(deftest protected-plan-complete-write-proof-is-bound-to-the-exact-schedule
  (let [compilation (equation-first/compile
                     #'resident-structured-reduction-probe {:target :ze:0 :dtype :float})
        emitted (:emitted compilation)
        operation (first (:operations (last (:equations emitted))))
        numerical-index (dec (count (get-in operation [:body :equations])))
        damaged (update-in operation [:body :equations numerical-index :operations 0
                                      :body :operations] pop)
        reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (seq (emitted-equation/complete-write-domains operation)))
    (is (= :emitted-reduction-reference-refinement
           (reason #(emitted-equation/complete-write-domains damaged))))
    (is (= :emitted-reduction-outer-slice
           (reason #(emitted-program/validate!
                     (assoc-in emitted [:attributes :allocations 0 :initialization] :unspecified)))))))

(deftest mixed-components-initialize-shared-scratch-only-once
  (let [{:keys [form]} (fuse/fuse (chain) :float)
        [[storage allocation] [result marker]] (mapv vec (partition 2 (second form)))
        plan (fuse/marker-plan marker :float)
        effect '(raster.par/map-void! i 1
                  (raster.par/atomic-add! scratch 0 (float 1.0)))
        source (list 'let* ['scratch '(clojure.core/float-array 1)
                           'before effect storage allocation result marker 'after effect]
                     [result 'scratch])
        attempt (typed-route/attempt
                 source :float (into {} (map (juxt :id :dtype)) (:operands plan))
                 {:segmented-plans? true :resident-initialization? true
                  :scalar-types (zipmap (:runtime-parameters plan) (repeat :long))})
        program (:program attempt)]
    (is (some? program) (pr-str (:declined attempt)))
    (is (= 1 (get-in attempt [:stats :initialization-fills])))
    (is (identical? program (structured-route/validate-typed-program! program)))
    (let [[before [_ & after]] (split-with #(not (swr/plan? (:algorithm %)))
                                         (:equations program))
          fills (fn [equations]
                  (filter #(= :zero (get-in % [:attributes :initialization])) equations))]
      (is (= 1 (count (fills before))))
      (is (empty? (fills after))
          "the suffix must observe the prefix update, not reset the allocation"))))

(deftest protected-source-admission-does-not-infer-missing-or-conflicting-types
  (let [{:keys [form]} (fuse/fuse (chain) :float)
        marker (last (second form))
        plan (fuse/marker-plan marker :float)
        array-types (into {} (map (juxt :id :dtype)) (:operands plan))
        options {:segmented-plans? true
                 :scalar-types (zipmap (:runtime-parameters plan) (repeat :long))}]
    (doseq [[arrays opts reason]
            [[(assoc array-types 'Q :double) options :source-value-conflict]
             [array-types (update options :scalar-types dissoc 'dk) :typed-soac-unknown-value]
             [array-types (assoc-in options [:scalar-types 'dk] :float) :source-value-conflict]]]
      (let [attempt (typed-route/attempt form :float arrays opts)]
        (is (nil? (:program attempt)))
        (is (= reason (get-in attempt [:declined :reason])))))
    (let [unproved (assoc-in (vec form) [1 (dec (count (second form)))]
                             (with-meta marker nil))
          unproved (apply list unproved)]
      (is (= :segmented-reduction-marker-missing-plan
             (try (typed-route/attempt unproved :float array-types options)
                  nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
          "losing the recognized algebra proof must not become a fallback or inferred plan"))))

(deftest marker-rebinding-deduplicates-identified-single-head-dimensions
  (let [single-head (walk/postwalk-replace
                     {'dk 'emb-dim 'n-heads 1}
                     (chain))
        {:keys [form stats]} (fuse/fuse single-head :float)
        marker (second (last (mapv vec (partition 2 (second form)))))
        plan (fuse/marker-plan marker :float)]
    (is (= {:segmented-weighted-reductions-fused 1} stats))
    (is (= '[n-nodes n-edges emb-dim 1]
           (:runtime-parameters plan)))
    (is (= 'emb-dim (get-in plan [:score :axis :extent])))
    (is (= 'emb-dim (get-in plan [:storage :total-dim])))
    (is (= 1 (get-in plan [:segment-axes 1 :extent])))))

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
                    (map (fn [[_sym key]]
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
