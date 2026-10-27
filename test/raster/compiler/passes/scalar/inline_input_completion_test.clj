(ns raster.compiler.passes.scalar.inline-input-completion-test
  "Inlining must preserve the actual typed invocation, not merely its body."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.pipeline :as pipeline]
            [raster.runtime.callsites :as callsites]))

(deftm narrow-float [x :- Float] :- Float x)
(deftm unused-arrays [a :- (Array double) b :- (Array double)] :- Long 7)

(defmacro define-typed-callers []
  (let [operation (symbol (str (ns-name *ns*)) "narrow-float")
        implementation (symbol (str (ns-name *ns*)) "narrow-float_m_float-impl")
        selected (entry/selected-definition operation implementation 1)
        receiver (with-meta implementation
                   {:raster.type/tag (get-in selected [:entry :typed-iface])
                    :raster.type/ret-tag 'float})
        arrays-operation (symbol (str (ns-name *ns*)) "unused-arrays")
        arrays-implementation (symbol (str (ns-name *ns*)) "unused-arrays_m_doubles_doubles-impl")
        arrays-selected (entry/selected-definition arrays-operation arrays-implementation 2)
        arrays-receiver (with-meta arrays-implementation
                          {:raster.type/tag (get-in arrays-selected [:entry :typed-iface])
                           :raster.type/ret-tag 'long})]
    (list 'do
          (list 'raster.core/deftm 'narrow-caller
                '[x :- Double] ':- 'Double
                (list 'let ['result (list '.invk receiver 'x)] '(double result)))
          (list 'raster.core/deftm 'ordered-input-caller
                '[first-value :- Object events :- (Array long) second-value :- (Array double)] ':- 'Long
                (list 'let
                      ['result (list '.invk arrays-receiver 'first-value
                                     '(do (clojure.core/aset events 0
                                                            (inc (clojure.core/aget events 0)))
                                          second-value))]
                      '(long result))))))

(declare narrow-caller ordered-input-caller)
(define-typed-callers)

(defmacro define-return-identity-caller []
  (let [operation (symbol (str (ns-name *ns*)) "narrow-float")
        implementation (symbol (str (ns-name *ns*)) "narrow-float_m_float-impl")
        selected (entry/selected-definition operation implementation 1)
        receiver (with-meta implementation
                   {:raster.type/tag (get-in selected [:entry :typed-iface])
                    :raster.type/ret-tag 'float})]
    (list 'raster.core/deftm 'boxed-return-identity-caller '[x :- Float] ':- 'Boolean
          (list 'let ['result (list '.invk receiver 'x)]
                '(clojure.core/identical? result result)))))

(declare boxed-return-identity-caller)
(define-return-identity-caller)

(deftest inlined-typed-call-preserves-float-input-rounding
  ;; The caller's Double is legal for this actual Float formal: the existing
  ;; invocation emitter performs JVM d2f. Substituting the body directly loses it.
  (let [compiled (pipeline/compile-aot #'narrow-caller :simd? false)]
    (doseq [x [0.1 1.00000006 1.00000008 -1.00000008]]
      (let [expected (Double/doubleToRawLongBits (double (float x)))]
        (is (= expected (Double/doubleToRawLongBits (narrow-caller x))))
        (is (= expected (Double/doubleToRawLongBits (compiled x))))))))

(deftest first-input-check-precedes-second-input-effects-even-with-unused-formals
  (let [compiled (pipeline/compile-aot #'ordered-input-caller :simd? false)
        second-value (double-array [2.0])]
    (doseq [invoke [ordered-input-caller compiled]]
      (let [events (long-array 1)]
        (is (thrown? ClassCastException (invoke "wrong array" events second-value)))
        (is (= [0] (vec events)) "argument one is checked before evaluating argument two"))
      (let [events (long-array 1)]
        (is (= 7 (invoke nil events second-value)) "null passes the actual array CHECKCAST")
        (is (= [1] (vec events)) "the second argument is evaluated exactly once")))))

(deftest inlined-typed-call-retains-one-boxed-result-for-repeated-uses
  (let [compiled (pipeline/compile-aot #'boxed-return-identity-caller :simd? false)
        value (float 0.25)]
    (is (true? (boxed-return-identity-caller value)))
    (is (true? (compiled value))
        "the interface Object result is boxed once, not once per use")))

(def callback-callee-events (atom []))

(deftm callback-old-helper [x :- Double] :- Long
  (do (swap! callback-callee-events conj :old) 11))
(deftm callback-new-helper [x :- Double] :- Long
  (do (swap! callback-callee-events conj :new) 22))

(defmacro define-callback-callers []
  (let [operation (symbol (str (ns-name *ns*)) "callback-old-helper")
        implementation (symbol (str (ns-name *ns*)) "callback-old-helper_m_double-impl")
        selected (entry/selected-definition operation implementation 1)
        receiver (with-meta implementation
                   {:raster.type/tag (get-in selected [:entry :typed-iface])
                    :raster.type/ret-tag 'long})]
    (list* 'do
           (for [caller '[callback-cold-caller callback-warm-caller callback-throw-caller]]
             (list 'raster.core/deftm caller '[value :- Object sentinel :- Long] ':- 'Long
                   (list 'let ['result (list '.invk receiver 'value)]
                         '(clojure.core/long result)))))))

(declare callback-cold-caller callback-warm-caller callback-throw-caller)
(define-callback-callers)

(defn- callback-implementation-var [generic]
  (let [source (core/resolve-deftm-var generic)]
    (ns-resolve (:ns (meta source)) (symbol (str (:name (meta source)) "-impl")))))

(defn- replace-callback-implementation! [receiver implementation]
  (alter-var-root receiver (constantly implementation))
  (callsites/invalidate! receiver implementation))

(defn- redefining-number [replace! events failure]
  (proxy [Number] []
    (doubleValue []
      (swap! events conj :convert)
      (replace!)
      (if failure (throw failure) 3.0))
    (intValue [] (throw (AssertionError. "unexpected int conversion")))
    (longValue [] (throw (AssertionError. "unexpected long conversion")))
    (floatValue [] (throw (AssertionError. "unexpected float conversion")))))

(deftest reference-input-conversion-cannot-freeze-the-callee-before-its-callback
  ;; Both implementations and their interface are real registered typed entries.
  ;; No positive planner facts or emission/source records are mocked. The call's
  ;; runtime target must be selected AFTER Number.doubleValue, cold or linked.
  (let [receiver (callback-implementation-var #'callback-old-helper)
        replacement (callback-implementation-var #'callback-new-helper)]
    (core/ensure-current-owned-bytecode! receiver)
    (core/ensure-current-owned-bytecode! replacement)
    (let [original (var-get receiver)
          next-implementation (var-get replacement)]
      (try
        (doseq [[caller warm? throws?]
                [[#'callback-cold-caller false false]
                 [#'callback-warm-caller true false]
                 [#'callback-throw-caller true true]]
                inline? [false true]]
          (replace-callback-implementation! receiver original)
          (let [invoke (pipeline/compile-aot caller :inline? inline? :simd? false)
                conversions (atom [])
                failure (when throws? (ex-info "conversion failed after target replacement" {}))]
            (when warm?
              (is (= 11 (invoke 1.0 0))))
            (reset! callback-callee-events [])
            (let [value (redefining-number
                         #(replace-callback-implementation! receiver next-implementation)
                         conversions failure)]
              (if throws?
                (let [caught (try (invoke value 0) nil (catch Throwable error error))]
                  (is (identical? failure caught))
                  (is (= [] @callback-callee-events)
                      "failed conversion invokes neither old nor new target"))
                (do
                  (is (= 22 (invoke value 0)) (str "inline?=" inline? ", warm?=" warm?))
                  (is (= [:new] @callback-callee-events))))
              (is (= [:convert] @conversions)))))
        (finally
          (replace-callback-implementation! receiver original)
          (reset! callback-callee-events []))))))

(def prefix-replacement (atom nil))

(defn mutate-prefix-target! []
  (swap! callback-callee-events conj :prefix)
  (when-let [implementation @prefix-replacement]
    (replace-callback-implementation!
     (callback-implementation-var #'callback-old-helper) implementation)))

(deftm mutating-prefix-helper [x :- Double] :- Long
  (do (mutate-prefix-target!) 0))

(defmacro define-prefix-callers []
  (let [receiver (fn [name]
                   (let [operation (symbol (str (ns-name *ns*)) name)
                         implementation (symbol (str (ns-name *ns*)) (str name "_m_double-impl"))
                         selected (entry/selected-definition operation implementation 1)]
                     (with-meta implementation
                       {:raster.type/tag (get-in selected [:entry :typed-iface])
                        :raster.type/ret-tag 'long})))
        prefix (receiver "mutating-prefix-helper")
        target (receiver "callback-old-helper")]
    (list* 'do
           (for [caller '[prefix-cold-caller prefix-warm-caller]]
             (list 'raster.core/deftm caller '[value :- Double] ':- 'Long
                   (list 'let ['ignored (list '.invk prefix 'value)
                               'result (list '.invk target 'value)]
                         '(clojure.core/long result)))))))

(declare prefix-cold-caller prefix-warm-caller)
(define-prefix-callers)

(deftest earlier-helper-effects-cannot-freeze-a-later-call-target
  (let [receiver (callback-implementation-var #'callback-old-helper)
        replacement (callback-implementation-var #'callback-new-helper)]
    (core/ensure-current-owned-bytecode! receiver)
    (core/ensure-current-owned-bytecode! replacement)
    (let [original (var-get receiver)
          next-implementation (var-get replacement)]
      (try
        (doseq [[caller warm?] [[#'prefix-cold-caller false] [#'prefix-warm-caller true]]
                inline? [false true]]
          (replace-callback-implementation! receiver original)
          (reset! prefix-replacement nil)
          (let [invoke (pipeline/compile-aot caller :inline? inline? :simd? false)]
            (when warm? (is (= 11 (invoke 1.0))))
            (reset! callback-callee-events [])
            (reset! prefix-replacement next-implementation)
            (is (= 22 (invoke 1.0)) (str "inline?=" inline? ", warm?=" warm?))
            (is (= [:prefix :new] @callback-callee-events))))
        (finally
          (replace-callback-implementation! receiver original)
          (reset! prefix-replacement nil)
          (reset! callback-callee-events []))))))
