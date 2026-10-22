(ns raster.compiler.core.call-boundary
  "Physical invocation evidence, separate from semantic dispatch and dtype.

   Projects an ALREADY resolved JVM Method against an existing selected typed
   definition. Signature queries do not resolve overloads, execute source,
   initialize classes, infer operand representations or authorize reordering.
   Explicit selected-emission capture can prepare an actual helper through the
   caller's compiler callback, without executing its numerical body. Existing
   definition/owner guards remain responsible for later reuse."
  (:require [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.types :as types])
  (:import [java.lang.reflect Method Modifier]))

(defn selected-typed-signature
  "Check the physical signature of an already selected global typed invocation.

   `resolved-method` must come from the source invocation's resolved descriptor,
   not overload resolution performed by a normalizer. Match its declaring Class
   by identity against the existing interface registry and check the live source
   and executable correspondence through selected-definition.

   A known signature describes JVM carriers only: its Object result can have a
   logical scalar dtype. It proves neither non-nullness nor conversion identity,
   purity, ownership, completion order, or support for an AD rule/backend.
   This is a GLOBAL definition/signature query, not call-site evidence. The
   caller must first establish from retained call syntax and lexical scope that
   the executable really is this selected global definition. Dynamic selection
   must not be frozen into a global root by this query. Missing/stale definition
   or physical descriptor evidence returns an explicit unknown."
  [operation implementation-op argument-count resolved-method]
  (if-not (instance? Method resolved-method)
    {:status :unknown :reason :unresolved-invocation-method}
    (if-let [{:keys [entry] :as definition}
             (method-entry/selected-definition operation implementation-op argument-count)]
      (let [interface-name (:typed-iface entry)
            ;; This lookup uses the existing registered Class, never Class/forName
            ;; or a semantic tag -> Class fallback. No class loading is requested.
            interface (get @types/fn-interface-by-name interface-name)
            interface-class (:iface-class interface)
            ^Method method resolved-method]
        (cond
          (not (class? interface-class))
          {:status :unknown :reason :unregistered-invocation-interface}

          (not (and (.isInterface ^Class interface-class)
                    (identical? interface-class (.getDeclaringClass method))
                    (= "invk" (.getName method))
                    (not (Modifier/isStatic (.getModifiers method)))
                    (= argument-count (.getParameterCount method))))
          {:status :unknown :reason :invocation-descriptor-mismatch}

          (not (.isInstance ^Class interface-class (:typed-impl entry)))
          {:status :unknown :reason :invocation-receiver-class-mismatch}

          :else
          {:status :known :kind :typed-interface
           :method method :declaring-class interface-class
           :parameter-classes (vec (.getParameterTypes method))
           :return-class (.getReturnType method)
           :definition definition}))
      {:status :unknown :reason :unverified-selected-definition})))

(defn ^:no-doc capture-selected-emission!
  "Capture an emission through an already resolved global typed call.
   No operand tags select a method here. The original call-site producer must
   prove global lexical correspondence and supply its actual resolved Method.
   Realizing the selected lazy implementation never executes its numeric body.
   Capture is independent of derivative activity: an inactive call still needs
   the same source/receiver ownership and historical-input checks. `prepare!`
   must be the existing non-evaluating owned compiler entry; it is not a model
   invocation. Preparation does not supply missing original call-site evidence."
  [operation implementation-op argument-count resolved-method prepare!]
  (method-entry/call-with-definition-inputs
   (fn []
    (let [observed (method-entry/observed-selected-definition
                    :typed operation implementation-op argument-count)
          select! (fn []
                  (let [signature (selected-typed-signature
                                   operation implementation-op argument-count resolved-method)]
                    (when-not (= :known (:status signature))
                      (throw (ex-info "paired helper has no verified selected typed invocation"
                                      {:reason :ad-paired-helper-selected-definition
                                       :operation operation :implementation-op implementation-op
                                       :decline (:reason signature)})))
                    signature))
        initial (select!)
        _ (when-not (and observed
                         (identical? (:receiver-var observed)
                                     (get-in initial [:definition :receiver-var])))
            (throw (ex-info "selected helper has no original definition observation"
                            {:reason :ad-paired-helper-selected-definition
                             :operation operation :implementation-op implementation-op})))
        _ (method-entry/prepare-owned-definition-inputs! prepare! [(:receiver-var observed)])
        {:keys [definition declaring-class] :as signature} (select!)
        {:keys [source-var receiver-var parameters tags return-tag entry]} definition
        emission (:raster.core/bytecode-emission (meta receiver-var))]
    (when-not (and (= :jvm-bytecode (:backend emission))
                   (identical? (:implementation emission) @receiver-var)
                   (identical? (:implementation emission) (:typed-impl entry))
                   (= (:params emission) parameters) (= (:tags emission) tags)
                   (= (:return-tag emission) return-tag)
                   (identical? (:source-ns emission) (.ns ^clojure.lang.Var source-var))
                   (= (:typed-interface emission) (:typed-iface entry)))
      (throw (ex-info "selected helper does not own its retained bytecode emission"
                      {:reason :ad-paired-helper-emission-correspondence
                       :operation operation :implementation-op implementation-op})))
    (let [observations (mapv method-entry/var-snapshot [source-var receiver-var])
          selection-guard
          (fn []
            (let [current (select!)]
              (when-not (and (every? method-entry/var-snapshot-current? observations)
                             (identical? declaring-class (:declaring-class current))
                             (identical? source-var (get-in current [:definition :source-var]))
                             (identical? receiver-var (get-in current [:definition :receiver-var]))
                             (identical? (:implementation emission)
                                         (get-in current [:definition :entry :typed-impl])))
                (throw (ex-info "selected paired helper definition changed"
                                {:reason :ad-paired-helper-selected-definition
                                 :operation operation :implementation-op implementation-op})))))
          emission-guard (method-entry/prepare-emission-input-guard! emission)
          guard (fn [] (selection-guard)
                  (emission-guard)
                  nil)
          _ (guard)]
      {:emission emission :guard guard :signature signature :method resolved-method})))))
