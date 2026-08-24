(ns raster.compiler.passes.parallel.segmented-weighted-reduction-fuse
  "Protect proven segmented weighted reductions as one schedule-neutral compiler operation.

   Recognition rules prove source algebra; this pass only coordinates disjoint plans and raises
   them. The marker deliberately says nothing about attention, graph topology, cache layout, or a
   target schedule. Backends consume the validated plan and independently select a legal leaf."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.indexed-attention-recognize :as indexed-rule]))

(def marker-op 'raster.compiler/segmented-weighted-reduction!)
(def ^:private plan-key ::plan)

(descriptor/register-op-descriptor!
 marker-op
 {:compiler-ir? true
  :buffer {:allocates? false :in-place-arg 0}
  :effects {:pure? false :mutating? true}})

(def recognition-rules
  "Ordered algebraic recognition rules. Adding a source spelling extends this table; it does not
   add a compiler pass or backend semantic operation."
  [{:id :indexed-dot-scatter-normalize
    :recognize (fn [form dtype]
                 (indexed-rule/recognize form :dtype dtype :accumulator-dtype dtype))}])

(defn marker?
  [form]
  (and (seq? form) (= marker-op (first form))))

(defn marker-plan
  "Recover and validate the schedule-neutral plan after intervening compiler substitutions."
  [marker emitted-dtype]
  (when-not (and (marker? marker) (= 4 (count marker))
                 (vector? (nth marker 2)) (vector? (nth marker 3)))
    (throw (ex-info "segmented weighted-reduction marker has the wrong shape"
                    {:reason :segmented-reduction-invalid-marker :marker marker})))
  (let [[_ output inputs runtime-values] marker
        template (get (meta marker) plan-key)]
    (when-not (swr/plan? template)
      (throw (ex-info "segmented weighted-reduction marker lost its proof"
                      {:reason :segmented-reduction-marker-missing-plan :marker marker})))
    (when-not (= emitted-dtype (:accumulator-dtype template))
      (throw (ex-info "segmented weighted-reduction marker dtype differs from its proof"
                      {:reason :segmented-reduction-marker-dtype-mismatch
                       :expected (:accumulator-dtype template) :actual emitted-dtype})))
    (swr/rebind template output inputs runtime-values marker)))

(defn- array-constructor
  [dtype]
  (case dtype
    :float 'clojure.core/float-array
    :double 'clojure.core/double-array
    (throw (ex-info "structured reduction output storage is unsupported"
                    {:reason :segmented-reduction-output-dtype-unsupported :dtype dtype}))))

(defn- array-tag
  [dtype]
  (case dtype :float 'floats :double 'doubles))

(defn- fresh-storage-symbol
  [used output]
  (loop [candidate (symbol (str (name output) "__segmented_reduction_out"))
         suffix 2]
    (if (contains? used candidate)
      (recur (symbol (str (name output) "__segmented_reduction_out_" suffix)) (inc suffix))
      candidate)))

(defn- marker-form
  [plan storage]
  (with-meta
    (list marker-op storage (swr/ordered-input-ids plan)
          (swr/runtime-parameter-values plan))
    {plan-key plan}))

(defn- recognize
  [form dtype rules]
  (mapcat (fn [{:keys [recognize]}] (recognize form dtype)) rules))

(defn- unresolved-ad-boundary?
  "A protected marker has no VJP of its own. If semantic AD operators are still present, leave the
   compositional program visible so the fixpoint can differentiate it first. Already-expanded
   backward code is safe: recognizers still have to prove exclusive uses of every intermediate."
  [form]
  (boolean
   (some (fn [node]
           (when (seq? node)
             (let [operation (first node)
                   ns-name (when (symbol? operation) (namespace operation))]
               (and ns-name (.startsWith ^String ns-name "raster.ad")))))
         (tree-seq coll? seq form))))

(defn fuse
  "Raise disjoint proven reductions. Failed recognition leaves the identical source form intact."
  ([form dtype] (fuse form dtype recognition-rules))
  ([form dtype rules]
   (let [ad-boundary? (unresolved-ad-boundary? form)
         plans (if ad-boundary? [] (vec (recognize form dtype rules)))]
     (if (empty? plans)
       {:form form :stats (cond-> {:segmented-weighted-reductions-fused 0}
                            ad-boundary? (assoc :declined-unresolved-ad-boundary 1))}
       (let [[let-sym bindings & body] form
             pairs (mapv vec (partition 2 bindings))
             plan-by-output (into {} (map (juxt #(get-in % [:output :id]) identity)) plans)
             removed (into #{} (mapcat #(get-in % [:source-operation :intermediates])) plans)
             used (set (map first pairs))
             rewritten
             (mapcat
              (fn [[sym init]]
                (cond
                  (contains? removed sym) []
                  (contains? plan-by-output sym)
                  (let [plan (get plan-by-output sym)
                        dtype (get-in plan [:output :dtype])
                        storage (fresh-storage-symbol used sym)
                        tag (array-tag dtype)
                        storage (with-meta storage {:tag tag :raster.type/tag tag})
                        result (with-meta sym (merge (meta sym) {:tag tag :raster.type/tag tag}))]
                    [storage (list (array-constructor dtype) (get-in plan [:output :elements]))
                     result (marker-form plan storage)])
                  :else [sym init]))
              pairs)]
         {:form (with-meta (apply list let-sym (vec rewritten) body) (meta form))
          :stats {:segmented-weighted-reductions-fused (count plans)}})))))
