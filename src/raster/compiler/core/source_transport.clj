(ns raster.compiler.core.source-transport
  "Owner-local source transformations, separate from source execution admission.
   Preserve authentic helper inputs at the transformation that consumes them;
   never reconstruct their origins from later symbol names or equal expressions."
  (:require [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.util :as util]))

(defn effective-body
  "Existing multi-form helper convention: a single ordered do expression."
  [body]
  (if (> (count body) 1) [(apply list 'do body)] body))

(deftype ^:private BodyImport
  [frontend parameters source-ns effective hygienic qualified tags renamings observations]
  entry/RetainedProducerInput
  (assert-producer-current! [_]
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (when-not (identical? source-ns (find-ns (ns-name source-ns)))
      (throw (ex-info "imported source namespace was replaced" {:reason :stale-source-import-namespace})))
    (entry/assert-definition-inputs-current! observations)
    true))

(defn- qualify-tags [tags source-ns]
  (if-not source-ns tags
  (when tags
    (mapv (fn [tag]
            (if (or (nil? tag)
                     (contains? '#{double long float int Double Long Float Integer
                                   doubles floats longs ints objects} tag)
                     (.contains (str tag) "."))
              tag
              (let [lookup (symbol (str tag))
                    result (try (ns-resolve source-ns lookup) (catch Exception _ nil))]
                (entry/observe-source-resolution!
                 {:kind :symbol-resolution :source-ns source-ns :symbol lookup :result result})
                (if (class? result) (symbol (.getName ^Class result)) tag)))) tags))))

(defn- prepare-body [frontend body parameters tags source-ns owned?]
  (let [_ (when frontend
            (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
            (when-not (and (identical? body (:body frontend))
                           (= parameters (get-in frontend [:source-entry :params]))
                           (= tags (get-in frontend [:source-entry :tags]))
                           (identical? source-ns (get-in frontend [:source-entry :source-ns])))
              (throw (ex-info "helper transformation differs from its original input"
                              {:reason :source-import-input-mismatch}))))
        _ (when owned?
            (entry/observe-source-body-input! (get-in frontend [:source-entry :source-var]) :var)
            (entry/observe-frontend-source-identity! frontend))
        qualified-tags (qualify-tags tags source-ns)
        effective (effective-body body)
        ;; The free set is scope-aware. Shadowed original binders deliberately
        ;; remain distinct ordered events, never entries in a symbol-keyed map.
        used (when owned? (volatile! (into (set parameters) (mapcat util/free-syms) effective)))
        events (when owned? (volatile! []))
        allocate util/*fresh-binder*
        hygienic (if owned?
                   (binding [util/*fresh-binder*
                             (fn [original]
                               (let [fresh (allocate original)]
                                 (when-not (and (symbol? fresh) (nil? (namespace fresh))
                                                (not (contains? @used fresh))
                                                (identical? (meta original) (meta fresh)))
                                   (throw (ex-info "source import requires fresh metadata-preserving binders"
                                                   {:reason :nonfresh-source-import-binder})))
                                 (vswap! used conj fresh)
                                 (vswap! events conj [original fresh])
                                 fresh))]
                     (mapv util/alpha-convert effective))
                   (mapv util/alpha-convert effective))
        parameter-set (set (map #(if (symbol? %) % (symbol (name %))) parameters))
        qualified (if source-ns
                    (mapv #(inference/qualify-body-symbols % source-ns parameter-set
                                                         entry/observe-source-resolution!) hygienic)
                    hygienic)
        input (when owned?
                (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
                (entry/assert-definition-inputs-current! @entry/*emission-inputs*)
                (BodyImport. frontend parameters source-ns effective hygienic qualified
                             qualified-tags @events @entry/*emission-inputs*))]
    {:walked-body qualified :tags qualified-tags :source-import input}))

(defn prepare-inline-body
  "Perform the inliner's existing alpha-renaming and qualification once.
   An authentic frontend and active owner additionally retain an immutable input
   witness with IMPORT-LOCAL history. Legacy bodies acquire no owner. No formal
   argument completion, call substitution, final-step correspondence, normal
   return or permission to execute the imported body is supplied."
  [frontend body parameters tags source-ns]
  (if (and frontend entry/*emission-inputs*)
    (let [parent entry/*emission-inputs* historical @parent
          _ (entry/assert-definition-inputs-current! historical)
          imported (binding [entry/*emission-inputs* (atom {:static [] :values []})]
                     (prepare-body frontend body parameters tags source-ns true))
          ^BodyImport input (:source-import imported)]
      (when-not (identical? historical @parent)
        (throw (ex-info "source import's enclosing owner changed during construction"
                        {:reason :unstable-definition-proof-input})))
      (entry/assert-definition-inputs-current! historical)
      (entry/adopt-definition-inputs! (.-observations input))
      imported)
    (prepare-body frontend body parameters tags source-ns false)))

(defn body-import-view
  "Read an exact constructed body under its original history in a collecting owner.
   Wrong/copy/ownerless inputs abstain; stale original inputs throw. Successful
   queries adopt the original observations, never refresh them. This is only a
   body-import relation: call arguments and later transformations remain unproved."
  [input body]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? BodyImport input)
                (identical? body (.-qualified ^BodyImport input)))
       (entry/assert-producer-current! input)
       (entry/adopt-definition-inputs! (.-observations ^BodyImport input))
       {:frontend (.-frontend ^BodyImport input) :parameters (.-parameters ^BodyImport input)
        :tags (.-tags ^BodyImport input)
        :effective-body (.-effective ^BodyImport input) :hygienic-body (.-hygienic ^BodyImport input)
        :walked-body body :renamings (.-renamings ^BodyImport input)}))))
