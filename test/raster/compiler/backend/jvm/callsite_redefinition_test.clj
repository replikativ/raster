(ns raster.compiler.backend.jvm.callsite-redefinition-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as entry]))

(defn- receiver [scope name]
  (let [source (core/resolve-deftm-var (ns-resolve scope name))]
    (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))))

(deftest emitted-typed-call-selects-new-root-after-number-conversion-without-manual-invalidation
  (let [scope (create-ns (gensym "raster.callsite-redefinition."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(do (raster.core/deftm old-helper [value :- Long] :- Long 1)
                   (raster.core/deftm new-helper [value :- Long] :- Long 2))))
      (let [old (receiver scope 'old-helper) new (receiver scope 'new-helper)]
        (core/ensure-bytecode-implementation! old)
        (core/ensure-bytecode-implementation! new)
        (let [original @old replacement @new
              operation (symbol (str (ns-name scope)) "old-helper")
              implementation (symbol (str (ns-name scope)) (str (.sym ^clojure.lang.Var old)))
              selected (entry/selected-definition operation implementation 1)
              interface (get-in selected [:entry :typed-iface])
              call-receiver (with-meta implementation
                              {:raster.type/tag interface :raster.type/ret-tag 'long})]
          (try
            (binding [*ns* scope]
              (doseq [name '[cold-caller warm-caller]]
                (eval (list 'raster.core/deftm name '[value :- Object] ':- 'Long
                            (list 'clojure.core/long (list '.invk call-receiver 'value))))))
            (doseq [[name warm?] [['cold-caller false] ['warm-caller true]]]
              (alter-var-root old (constantly original))
              (let [caller (receiver scope name)]
                (core/ensure-bytecode-implementation! caller)
                (when warm? (is (= 1 (@caller (long 0)))))
                (let [conversions (atom 0)
                      argument (proxy [Number] []
                                 (longValue []
                                   (swap! conversions inc)
                                   (alter-var-root old (constantly replacement))
                                   (long 0))
                                 (intValue [] (int 0)) (doubleValue [] (double 0))
                                 (floatValue [] (float 0)))]
                  (is (= 2 (@caller argument)))
                  (is (= 1 @conversions)))))
            (finally (alter-var-root old (constantly original))))))
      (finally (remove-ns (ns-name scope))))))
