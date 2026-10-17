(ns raster.compiler.core.ordered-call
  "Ordered normalization of an already-resolved JVM invocation.

   Plans are ephemeral compiler-owner data, not persistent form metadata or a
   second function registry. The producer owns source descriptor selection,
   lexical correspondence, definition guards and compilation policy. This
   consumer performs no overload selection or operand dtype inference."
  (:require [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion])
  (:import [java.lang.reflect Method Modifier]))

(defn- input-contract [^Class carrier source-policy]
  (cond
    (.isPrimitive carrier) (completion/primitive-input-contract carrier source-policy)
    (= Object carrier) completion/object-input-contract
    (.isArray carrier) (completion/reference-check-contract carrier)
    ;; A host reference parameter can have FISupport adaptation. Its Class is
    ;; not by itself evidence that the source merely performs CHECKCAST.
    :else (throw (ex-info "unclosed JVM reference input conversion"
                          {:reason :unclosed-call-input-conversion :carrier carrier}))))

(defn plan-resolved-call
  "Describe a call whose exact Method and source policy the producer owns.

   Supported source contexts are static host methods, typed invk interfaces,
   and Object-ABI IFn.invoke. The last does not stand for invokePrim/direct
   linking. No source syntax, global root or scalar tag selects a Method here.
   The producer must verify source-to-descriptor correspondence independently.
   Runtime executable selection remains the ORIGINAL receiver expression."
  [resolved-method receiver arguments source-policy]
  (when-not (and (instance? Method resolved-method)
                 (vector? arguments) (completion/source-policy? source-policy))
    (throw (ex-info "ordered call requires a resolved Method, arguments and source policy"
                    {:reason :unclosed-ordered-call})))
  (let [^Method method resolved-method
        owner (.getDeclaringClass method)
        static? (Modifier/isStatic (.getModifiers method))
        parameter-classes (vec (.getParameterTypes method))
        typed? (and (.isInterface owner) (= "invk" (.getName method)))
        ifn? (and (= clojure.lang.IFn owner) (= "invoke" (.getName method))
                  (every? #(= Object %) parameter-classes))]
    (when-not (and (Modifier/isPublic (.getModifiers method))
                   (Modifier/isPublic (.getModifiers owner))
                   (not (.isVarArgs method))
                   (= (count arguments) (.getParameterCount method))
                   (or (and static? (nil? receiver))
                       (and (not static?) (or typed? ifn?))))
      (throw (ex-info "unsupported resolved invocation context"
                      {:reason :unsupported-ordered-invocation :method method})))
    {:kind :ordered-jvm-call :method method :source-policy source-policy
     :receiver receiver :arguments arguments
     :input-contracts (mapv #(input-contract % source-policy) parameter-classes)
     :receiver-contract (when-not static? (completion/reference-check-contract owner))}))

(defn verify-source-method!
  "Check exact invocation Class correspondence in the ACTIVE source compiler
   loader. Call immediately before actual source compilation in that scope;
   this is not a portable certificate or authority for a later/different loader.
   Lookup does not initialize the classes or execute the target method."
  [^Method method]
  (doseq [^Class carrier (distinct (concat [(.getDeclaringClass method) (.getReturnType method)]
                                          (.getParameterTypes method)))
          :when (not (.isPrimitive carrier))]
    (let [resolved (try (clojure.lang.RT/classForNameNonLoading (.getName carrier))
                        (catch ClassNotFoundException _ nil))]
      (when-not (identical? carrier resolved)
        (throw (ex-info "source invocation resolves a different Class identity"
                        {:reason :source-invocation-class-mismatch
                         :expected carrier :resolved resolved :method method})))))
  ;; Parameter tags do not select a return descriptor. JVM bridge/covariant
  ;; methods can share a name and parameter list; don't silently substitute a
  ;; different Method while claiming to preserve the supplied descriptor.
  (let [candidates (filter (fn [^Method candidate]
                             (and (= (.getName method) (.getName candidate))
                                  (= (vec (.getParameterTypes method))
                                     (vec (.getParameterTypes candidate)))))
                           (.getMethods (.getDeclaringClass method)))]
    (when-not (= [method] (vec candidates))
      (throw (ex-info "source parameter signature does not close the exact Method"
                      {:reason :source-invocation-method-ambiguous :method method}))))
  method)

(defn normalize-resolved-call
  "Capture the executable, then each completed ORIGINAL operand, then invoke.

   Returns ordered bindings, invocation syntax and owner-private Class ports.
   The owner must bind each port to its actual Class, compile in the original
   lexical scope under :source-policy, and preserve exact method loader identity.
   `lexical-identifiers` must include ALL enclosing owner binders, even unused
   parameters: a generated Class port must not be shadowed by an unused binder.
   The returned plan does not certify those later compilation steps. It is not
   a default AD normalizer and adds no Class.cast/RT.box derivative templates."
  [plan fresh lexical-identifiers]
  (when-not (and (set? lexical-identifiers) (every? symbol? lexical-identifiers))
    (throw (ex-info "ordered call requires its enclosing lexical identifiers"
                    {:reason :unclosed-ordered-call-scope})))
  ;; Reconstruct structural contracts rather than trusting caller-edited plans.
  (let [source-identifiers (volatile! lexical-identifiers)
        _ (util/postwalk-preserving-meta
           (fn [form] (when (symbol? form) (vswap! source-identifiers conj form)) form)
           [(:receiver plan) (:arguments plan)])
        used source-identifiers
        fresh! (fn [prefix]
                 (let [id (fresh prefix)]
                   (when-not (and (symbol? id) (nil? (namespace id))
                                  (not (contains? @used id)))
                     (throw (ex-info "ordered call requires distinct fresh lexical identifiers"
                                     {:reason :nonfresh-ordered-call-identifier :identifier id})))
                   (vswap! used conj id)
                   ;; A fresh name carries no prior JVM representation facts.
                   (with-meta id nil)))
        {:keys [method receiver arguments source-policy] :as checked}
        (plan-resolved-call (:method plan) (:receiver plan) (:arguments plan) (:source-policy plan))
        _ (when-not (= checked plan)
            (throw (ex-info "ordered invocation plan differs from its descriptor"
                            {:reason :invalid-ordered-call-plan})))
        ^Method method method
        static? (Modifier/isStatic (.getModifiers method))
        owner (.getDeclaringClass method)
        receiver-value (when-not static? (fresh! "executable"))
        receiver-bindings (if static? []
                             [receiver-value (completion/complete-value (:receiver-contract checked) receiver)])
        input-values (mapv (fn [_] (fresh! "input")) arguments)
        bindings (into receiver-bindings
                       (mapcat (fn [value contract operand]
                                 [value (completion/complete-value contract operand)])
                               input-values (:input-contracts checked) arguments))
        ;; Qualified method syntax plus explicit parameter Classes retains the
        ;; selected overload after primitive inputs become boxed temporaries.
        invoke-head (with-meta (symbol (.getName owner) (str (when-not static? ".") (.getName method)))
                      {:param-tags (vec (.getParameterTypes method))})
        classes (distinct (keep (fn [contract]
                                  (when (= :jvm-checkcast (:source-context contract))
                                    (:target-carrier contract)))
                                (cond-> (:input-contracts checked)
                                  (:receiver-contract checked) (conj (:receiver-contract checked)))))
        class-ports (mapv (fn [carrier] {:class carrier :port (fresh! "reference_class")}) classes)]
    {:bindings (vec bindings) :input-values input-values
     :invocation (list* invoke-head (if static? input-values (into [receiver-value] input-values)))
     :class-ports class-ports :source-policy source-policy
     ;; Owner-side data only. A downstream source/AD consumer must derive the
     ;; local receiver's correspondence from this binding equation and scope.
     :receiver-binding (when-not static? (vec receiver-bindings))
     :method method}))
