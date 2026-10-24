(ns raster.compiler.core.binding-completion
  "Shared JVM lexical binding completion, separate from initializer inference.
   Process-local lookup guards are never portable type or ownership facts."
  (:require [raster.compiler.core.method-entry :as entry]))

(defn typed-interface?
  "Whether a hint names the existing Raster typed-function interface family."
  [hint]
  (and hint (.contains (str hint) "IFn__")))

(defn declared-binding-hint
  "The emitter's explicit declaration precedence, without initializer inference."
  [binder]
  (or (:tag (meta binder)) (:raster.type/tag (meta binder))))

(defn- checked-loader [context]
  (when-not (and (contains? context :hint-classloader)
                 (or (nil? (:hint-classloader context))
                     (instance? ClassLoader (:hint-classloader context))))
    (throw (ex-info "binding completion requires its actual hint-resolution loader"
                    {:reason :binding-completion-resolution-context})))
  (:hint-classloader context))

(defn- normalized-hint [hint]
  (if (and (symbol? hint) (zero? (count (name hint)))) nil hint))

(defn- class-for-name [name initialize? loader]
  (try (Class/forName name initialize? loader) (catch Exception _ nil)))

(defn- array-hint-class-name [hint]
  ;; Existing emitter hint translation, shared with initializer-free replay.
  (get {'objects "[Ljava.lang.Object;" 'doubles "[D" 'floats "[F"
        'ints "[I" 'longs "[J" 'bytes "[B" 'shorts "[S"} hint))

(defn- completion-plan [hint actual resolved]
  (let [checkcast (when (and resolved (not= resolved Object)) resolved)]
    {:resolved-hint-class resolved
     :checkcast-class checkcast
     ;; Preserve the existing emitter facts exactly. A resolved hint is not
     ;; evidence about the initializer's raw stack carrier.
     :local-facts (cond-> {:type actual :hint hint
                          :verified (boolean (and hint (not= hint 'Object)))}
                    (= :ref actual) (assoc :reference-class (or checkcast Object)))}))

(deftype ^:private BindingResolutionContext [loader inputs])

(defn capture-resolution-context
  "Capture an ALREADY LOADED actual binding-policy Var without invoking it.
   The caller authenticates which emitter policy it will use. Keep its original
   observation independent: only a successful optional proof adopts it."
  [actual-policy-var]
  (when (and (var? actual-policy-var) (bound? actual-policy-var))
    (let [owner (atom {:static [] :values []})]
      (binding [entry/*emission-inputs* owner]
        (entry/observe-definition-input! actual-policy-var (.ns ^clojure.lang.Var actual-policy-var)
                                         (.sym ^clojure.lang.Var actual-policy-var))
        (let [root (get-in @owner [:definition-proofs actual-policy-var :value])]
          (when (instance? clojure.lang.IFn root)
            (entry/assert-definition-inputs-current! @owner)
            (BindingResolutionContext. (.getClassLoader (class root)) @owner)))))))

(defn binding-completion-plan
  "Plan the emitter's existing CHECKCAST and local facts from an established
   raw stack carrier. Context supplies the ACTUAL emitter-function loader.
   General compilation preserves Class.forName-first, then ns-resolve behavior,
   including its original class-initialization behavior."
  [hint actual context]
  (let [hint (normalized-hint hint)
        resolved
        (when (and hint (= actual :ref) (not (typed-interface? hint)))
          (let [loader (checked-loader context)
                array-name (array-hint-class-name hint)]
            (if array-name
              ;; These primitive/bootstrap arrays follow the same first loader.
              (Class/forName array-name true loader)
              (let [name (str hint)]
                (when (pos? (count name))
                  (or (class-for-name name true loader)
                      (when-let [resolved (ns-resolve (or (:source-ns context) *ns*) (symbol name))]
                        (when (class? resolved) resolved))))))))]
    (completion-plan hint actual resolved)))

(defn- object-lookup-decision
  "Bounded initializer-free projection of the same first-loader lookup order."
  [hint source-ns loader]
  (let [array-name (array-hint-class-name hint)
        name (or array-name (str hint))
        primary (class-for-name name false loader)]
    (if primary
      {:branch :class-for-name :result primary}
      ;; ns-resolve can initialize dotted-name Classes via RT's own loader.
      ;; Only a direct namespace mapping is admitted by this optional proof.
      (when (and (nil? array-name) (nil? (namespace hint)) (not (.contains name ".")))
        {:branch :namespace-mapping :result (.getMapping ^clojure.lang.Namespace source-ns hint)}))))

(deftype ^:private BindingClassLookupInput [hint source-ns loader decision]
  entry/RetainedProducerInput
  (assert-producer-current! [this]
    (when-not (and (identical? source-ns (find-ns (ns-name source-ns)))
                   (= decision (object-lookup-decision hint source-ns loader)))
      (throw (ex-info "binding hint resolution changed"
                      {:reason :stale-binding-completion-resolution :hint hint})))
    this))

(defn reference-binding-without-check?
  "Prove a binding CHECKCAST redundant for an independently established actual
   reference upper-bound Class. Caller owns formal/source carrier correspondence,
   exact initializer syntax and its evaluation; no AV/dtype supplies a Class.
   Requires explicit source namespace, sealed actual-policy context and active
   definition ownership. Original policy/lookup history is retained on success.
   Neither optional construction nor replay initializes classes."
  [binder source-ns resolution-context reference-upper-bound]
   (entry/call-with-optional-definition-inputs
    (fn []
      (let [hint (declared-binding-hint binder)]
        (boolean
         (when (and entry/*emission-inputs* (instance? clojure.lang.Namespace source-ns)
                    (class? reference-upper-bound) (not (.isPrimitive ^Class reference-upper-bound))
                    (symbol? hint) (pos? (count (name hint))) (not (typed-interface? hint))
                    (instance? BindingResolutionContext resolution-context))
           (let [^BindingResolutionContext context resolution-context
                 loader (.-loader context)
                 inputs (.-inputs context)
                 _ (entry/assert-definition-inputs-current! inputs)
                 decision (object-lookup-decision hint source-ns loader)
                 plan (completion-plan hint :ref (:result decision))]
             (when (and (class? (:resolved-hint-class plan))
                        (.isAssignableFrom ^Class (:resolved-hint-class plan) reference-upper-bound))
               (entry/adopt-definition-inputs! inputs)
               (entry/adopt-definition-inputs!
                {:producer-inputs [(BindingClassLookupInput. hint source-ns loader decision)]})
               (entry/assert-definition-inputs-current! @entry/*emission-inputs*)
               true))))))))

(defn object-binding-without-check?
  "The Object-only specialization; a generic Object carrier cannot justify
   narrowing to an array/reference hint. Two-arg calls abstain without loader
   authority. Caller additionally owns exact syntax and lexical deadness."
  ([binder source-ns] false)
  ([binder source-ns resolution-context]
   (reference-binding-without-check? binder source-ns resolution-context Object)))
