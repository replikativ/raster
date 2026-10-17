(ns raster.compiler.core.call-boundary
  "Physical invocation evidence, separate from semantic dispatch and dtype.

   Projects an ALREADY resolved JVM Method against an existing selected typed
   definition. Does not resolve overloads, execute source, initialize classes,
   infer operand representations, or authorize argument reordering. Existing
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
