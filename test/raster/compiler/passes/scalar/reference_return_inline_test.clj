(ns raster.compiler.passes.scalar.reference-return-inline-test
  "A ready reference-return module may supply an ordinary whole-call recipe.
   It remains source expansion evidence only, never native placement authority."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.source-import :as source-import]))

(deftm ready-array-first [first-value :- (Array double)
                          second-value :- (Array double)
                          n :- Long] :- (Array double)
  first-value)

(deftm ready-checked-return [value :- Object n :- Long] :- (Array double)
  value)

(deftm ready-quoted-array [value :- (Array double) n :- Long] :- (Array double)
  (let [ignored '(let [reference_input__0 :quoted] reference_input__0)]
    value))

(deftm ready-local-effects [values :- (Array double)
                            events :- (Array long)
                            n :- Long] :- (Array double)
  (let [alias values
        before (clojure.core/aset events 0
                                  (inc (clojure.core/aget events 0)))
        old (clojure.core/aget alias n)
        changed (clojure.core/aset alias n (+ old 1.0))
        result alias]
    result))

(deftm ready-top-effect [values :- (Array double)
                         events :- (Array long)
                         n :- Long] :- (Array double)
  (let [ignored (clojure.core/aset events 0 n)]
    values))

;; This pair is deliberately never invoked or compiled elsewhere in the namespace.
(deftm cold-array-first [value :- (Array double) n :- Long] :- (Array double)
  value)

(defmacro define-reference-callers []
  (letfn [(receiver [operation implementation arity return-tag]
            (let [selected (entry/selected-definition operation implementation arity)]
              (with-meta implementation
                {:raster.type/tag (get-in selected [:entry :typed-iface])
                 :raster.type/ret-tag return-tag})))]
    (let [ns-name (str (ns-name *ns*))
          ready-op (symbol ns-name "ready-array-first")
          ready-impl (symbol ns-name "ready-array-first_m_doubles_doubles_long-impl")
          ready-receiver (receiver ready-op ready-impl 3 'doubles)
          checked-op (symbol ns-name "ready-checked-return")
          checked-impl (symbol ns-name "ready-checked-return_m_Object_long-impl")
          checked-receiver (receiver checked-op checked-impl 2 'doubles)
          quoted-op (symbol ns-name "ready-quoted-array")
          quoted-impl (symbol ns-name "ready-quoted-array_m_doubles_long-impl")
          quoted-receiver (receiver quoted-op quoted-impl 2 'doubles)
          effects-op (symbol ns-name "ready-local-effects")
          effects-impl (symbol ns-name "ready-local-effects_m_doubles_longs_long-impl")
          effects-receiver (receiver effects-op effects-impl 3 'doubles)
          top-effect-op (symbol ns-name "ready-top-effect")
          top-effect-impl (symbol ns-name "ready-top-effect_m_doubles_longs_long-impl")
          top-effect-receiver (receiver top-effect-op top-effect-impl 3 'doubles)
          cold-op (symbol ns-name "cold-array-first")
          cold-impl (symbol ns-name "cold-array-first_m_doubles_long-impl")
          cold-receiver (receiver cold-op cold-impl 2 'doubles)]
      (list
       'do
       (list 'raster.core/deftm 'ordered-ready-reference-caller
             '[first-value :- Object second-value :- (Array double) n :- Long] ':- 'Object
             (list 'let ['result (list '.invk ready-receiver 'first-value 'second-value 'n)]
                   'result))
       (list 'raster.core/deftm 'checked-ready-reference-caller
             '[value :- Object n :- Long] ':- 'Object
             (list 'let ['result (list '.invk checked-receiver 'value 'n)] 'result))
       (list 'raster.core/deftm 'quoted-ready-reference-caller
             '[value :- (Array double) n :- Long] ':- 'Object
             (list 'let ['result (list '.invk quoted-receiver 'value 'n)] 'result))
       (list 'raster.core/deftm 'local-effects-ready-reference-caller
             '[values :- Object events :- (Array long) n :- Long] ':- 'Object
             (list 'let ['result (list '.invk effects-receiver 'values 'events 'n)] 'result))
       (list 'raster.core/deftm 'top-effect-ready-reference-caller
             '[values :- Object events :- (Array long) n :- Long] ':- 'Object
             (list 'let ['result (list '.invk top-effect-receiver 'values 'events 'n)] 'result))
       (list 'raster.core/deftm 'cold-reference-caller
             '[value :- (Array double) n :- Long] ':- 'Object
             (list 'let ['result (list '.invk cold-receiver 'value 'n)] 'result))))))

(declare ordered-ready-reference-caller checked-ready-reference-caller
         quoted-ready-reference-caller local-effects-ready-reference-caller
         top-effect-ready-reference-caller cold-reference-caller)
(define-reference-callers)

(defn- implementation-var [source-var]
  (ns-resolve (.ns ^clojure.lang.Var source-var)
              (symbol (str (:name (meta source-var)) "-impl"))))

(defn- ensure-ready! [helper]
  (let [source-var (core/resolve-deftm-var helper)
        implementation (implementation-var source-var)]
    (core/ensure-current-owned-bytecode! implementation)
    {:source-var source-var :implementation implementation}))

(defn- caller-request [caller]
  (let [source-var (core/resolve-deftm-var caller)
        facts (pipeline/deftm-frontend-facts source-var :double)
        source (first (pipeline/get-walked-body source-var :double))
        occurrence (some #(when (and (seq? %) (= '.invk (first %))) %)
                         (tree-seq coll? seq source))]
    {:source source :occurrence occurrence
     :options {:source-var source-var :source-ns (:source-ns facts)
               :active-params (:parameters facts)
               :public-parameters (:parameters facts)}}))

(defn- compile-through-ready-fallback [caller]
  (let [fallback source-import/instantiate-ready-reference-helper
        accepted (atom 0)]
    [(with-redefs [source-import/instantiate-static-helper (fn [& _] nil)
                   source-import/instantiate-ready-reference-helper
                   (fn [& arguments]
                     (let [application (apply fallback arguments)]
                       (when application (swap! accepted inc))
                       application))]
       (pipeline/compile-aot caller :simd? false))
     @accepted]))

(deftest ready-reference-fallback-preserves-input-slots-null-and-return-checks
  (ensure-ready! #'ready-array-first)
  (ensure-ready! #'ready-checked-return)
  (let [[ordered accepted] (compile-through-ready-fallback #'ordered-ready-reference-caller)
        [checked checked-accepted] (compile-through-ready-fallback #'checked-ready-reference-caller)
        second-value (double-array [2.0])]
    (is (pos? accepted) "the public caller consumed an actual ready-module recipe")
    (is (pos? checked-accepted))
    (doseq [invoke [ordered-ready-reference-caller ordered]]
      (is (thrown? ClassCastException (invoke "wrong array" second-value 0)))
      (is (nil? (invoke nil second-value 0)) "CHECKCAST preserves null"))
    (doseq [invoke [checked-ready-reference-caller checked]]
      (is (nil? (invoke nil 0)))
      (is (thrown? ClassCastException (invoke "wrong returned class" 0))))))

(deftest ready-reference-import-keeps-quoted-data-opaque
  (ensure-ready! #'ready-quoted-array)
  (let [[compiled accepted] (compile-through-ready-fallback #'quoted-ready-reference-caller)
        value (double-array [1.0 2.0])]
    (is (pos? accepted))
    (is (identical? value (quoted-ready-reference-caller value 0)))
    (is (identical? value (compiled value 0)))))

(deftest complete-ready-reference-expression-keeps-locals-effects-and-result-identity
  (ensure-ready! #'ready-local-effects)
  (ensure-ready! #'ready-top-effect)
  (let [[compiled accepted]
        (compile-through-ready-fallback #'local-effects-ready-reference-caller)
        [top-compiled top-accepted]
        (compile-through-ready-fallback #'top-effect-ready-reference-caller)]
    (is (pos? accepted))
    (is (pos? top-accepted))
    (doseq [invoke [local-effects-ready-reference-caller compiled]]
      (let [values (double-array [2.0]) events (long-array 1)
            result (invoke values events 0)]
        (is (identical? values result))
        (is (= [3.0] (vec values)) "the helper-local store executes exactly once")
        (is (= [1] (vec events)) "the visible counter store executes exactly once"))
      (let [events (long-array 1)]
        (is (thrown? ClassCastException (invoke "wrong array" events 0)))
        (is (= [0] (vec events)) "the input CHECKCAST precedes the helper body"))
      (let [events (long-array 1)]
        (is (thrown? NullPointerException (invoke nil events 0)))
        (is (= [1] (vec events)) "null passes CHECKCAST and fails at the original read site")))
    ;; This helper's tail is the input itself. Reassembling the artifact from
    ;; argument-bindings + result-form appears to work but silently loses the
    ;; extracted unused-store binding; consuming the complete expression retains it.
    (doseq [invoke [top-effect-ready-reference-caller top-compiled]]
      (let [values (double-array [4.0]) events (long-array 1)
            result (invoke values events 7)]
        (is (identical? values result))
        (is (= [4.0] (vec values)))
        (is (= [7] (vec events))))
      (let [events (long-array 1)]
        (is (thrown? ClassCastException (invoke "wrong array" events 7)))
        (is (= [0] (vec events))))
      (let [events (long-array 1)]
        (is (nil? (invoke nil events 7)))
        (is (= [7] (vec events)))))))

(deftest cold-reference-query-does-not-prepare-a-module
  (let [helper (core/resolve-deftm-var #'cold-array-first)
        implementation (implementation-var helper)
        {:keys [source occurrence options]} (caller-request #'cold-reference-caller)]
    (is (some? occurrence))
    (is (nil? (:raster.core/bytecode-emission (meta implementation))))
    (with-redefs [core/ensure-bytecode-implementation!
                  (fn [& _] (throw (AssertionError. "reference query must not compile")))]
      (is (nil? (entry/call-with-definition-inputs
                 #(source-import/instantiate-ready-reference-helper source occurrence options))))
      (is (nil? (:raster.core/bytecode-emission (meta implementation)))))))

(deftest ready-reference-receipt-rejects-stale-source-and-replaced-module
  (let [{helper :source-var implementation :implementation} (ensure-ready! #'ready-array-first)
        {:keys [source occurrence options]} (caller-request #'ordered-ready-reference-caller)
        construct #(entry/call-with-definition-inputs
                    (fn []
                      (source-import/instantiate-ready-reference-helper source occurrence options)))]
    (doseq [[owner change]
            [[helper #(assoc % ::changed-source true)]
             [implementation
              #(assoc-in % [:raster.core/bytecode-emission :implementation] (Object.))]]]
      (let [metadata (meta owner)
            application (construct)]
        (is (some? application))
        (is (= '[first-value second-value n]
               (mapv :operand (get-in application [:application :input-slots]))))
        (is (= 3 (count (:argument-bindings application)))
            "every actual input keeps its ordered completion slot, including unused formals")
        (try
          (alter-meta! owner change)
          (is (thrown? clojure.lang.ExceptionInfo
                       (source-import/validate-helper-application! application)))
          (finally (reset-meta! owner metadata)))))))
