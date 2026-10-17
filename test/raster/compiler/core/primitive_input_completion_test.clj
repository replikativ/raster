(ns raster.compiler.core.primitive-input-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion])
  (:import [java.lang.reflect Proxy InvocationHandler]))

(defn- primitive-call [carrier]
  (let [interface (types/ensure-fn-interface! [(symbol (.getName ^Class carrier)) 'long])
        ^Class interface-class (:iface-class interface)
        target (Proxy/newProxyInstance
                (.getClassLoader interface-class) (into-array Class [interface-class])
                (reify InvocationHandler
                  (invoke [_ _ _ arguments] (aget ^objects arguments 0))))]
    {:interface interface-class :target target}))

(defn- observation [value]
  (cond
    (instance? Float value) {:class Float :bits (Float/floatToRawIntBits value)}
    (instance? Double value) {:class Double :bits (Double/doubleToRawLongBits value)}
    :else {:class (class value) :value value}))

(defn- pair-of-calls [carrier policy operand]
  (let [{:keys [interface target]} (primitive-call carrier)
        later '(do (swap! events conj :later) 0)
        original (list '.invk 'target operand later)
        term (completion/complete-value
              (completion/primitive-input-contract carrier policy) operand)
        transformed (list 'let* ['saved term] (list '.invk 'target 'saved later))
        compile-call (fn [body]
                       ;; Actual source compilation uses the SAME retained
                       ;; policy, not merely a matching projection argument.
                       (binding [*unchecked-math* (:unchecked-math policy)]
                         ;; Clojure local hints require the registered type's
                         ;; SYMBOL, not a Class value (which leaves an unresolved
                         ;; reflective call and different conversion semantics).
                         (eval (list 'fn* [(with-meta 'target {:tag (symbol (.getName ^Class interface))}) 'x 'events]
                                     (completion/project-to-jvm-source body policy)))))
        outcome (fn [f input]
                  (let [events (atom [])]
                    (try (assoc (observation (f target input events)) :events @events)
                         (catch Throwable error {:exception (class error) :events @events}))))]
    {:original (partial outcome (compile-call original))
     :transformed (partial outcome (compile-call transformed))}))

(deftest all-primitive-contexts-complete-before-later-inputs
  (doseq [[carrier input expected-class] [[Byte/TYPE 12 Byte] [Short/TYPE 12 Short]
                                        [Integer/TYPE 12 Integer] [Long/TYPE 12 Long]
                                        [Float/TYPE 1.5 Float] [Double/TYPE 1.5 Double]
                                        [Boolean/TYPE false Boolean] [Character/TYPE \a Character]]]
    (let [{:keys [original transformed]} (pair-of-calls carrier {:unchecked-math false} 'x)
          positive (original input)]
      (is (= expected-class (:class positive)))
      (is (= [:later] (:events positive)))
      (is (= positive (transformed input)))
      (is (= {:exception NullPointerException :events []} (original nil)))
      (is (= (original nil) (transformed nil)))
      (is (= {:exception ClassCastException :events []} (original "wrong-class")))
      (is (= (original "wrong-class") (transformed "wrong-class"))))))

(deftest strict-wrapper-checks-are-not-truthiness-or-general-char-casts
  (doseq [carrier [Boolean/TYPE Character/TYPE]]
    (let [{:keys [original transformed]} (pair-of-calls carrier {:unchecked-math false} 'x)]
      (is (= {:exception ClassCastException :events []} (original 42)))
      (is (= (original 42) (transformed 42))))))

(deftest primitive-and-boxed-floating-inputs-retain-different-overflow-behavior
  (let [policy {:unchecked-math false}
        boxed (pair-of-calls Float/TYPE policy 'x)
        primitive (pair-of-calls Float/TYPE policy '(double x))]
    (is (= {:exception IllegalArgumentException :events []}
           ((:original boxed) Double/MAX_VALUE)))
    (is (= ((:original boxed) Double/MAX_VALUE) ((:transformed boxed) Double/MAX_VALUE)))
    (is (= {:class Float :bits (Float/floatToRawIntBits Float/POSITIVE_INFINITY) :events [:later]}
           ((:original primitive) Double/MAX_VALUE)))
    (is (= ((:original primitive) Double/MAX_VALUE) ((:transformed primitive) Double/MAX_VALUE)))))

(deftest retained-checked-and-unchecked-policies-reach-actual-compilation
  (doseq [setting [false true :warn-on-boxed]
          operand ['x '(long x)]]
    (let [{:keys [original transformed]}
          (pair-of-calls Integer/TYPE {:unchecked-math setting} operand)
          expected (if setting {:class Integer :value -1 :events [:later]}
                       {:exception ArithmeticException :events []})]
      (is (= expected (original Long/MAX_VALUE)))
      (is (= (original Long/MAX_VALUE) (transformed Long/MAX_VALUE))))))

(deftest re-unboxing-does-not-add-rounding-traps-or-change-fp-bits
  (doseq [[carrier inputs]
          [[Float/TYPE [(Float/valueOf (float -0.0)) Float/POSITIVE_INFINITY
                        (Float/intBitsToFloat 2143289410)]]
           [Double/TYPE [(Double/valueOf -0.0) Double/POSITIVE_INFINITY
                         (Double/longBitsToDouble 9221120237041090626)]]]]
    (let [{:keys [original transformed]} (pair-of-calls carrier {:unchecked-math false} 'x)]
      (doseq [input inputs]
        (is (= (assoc (observation input) :events [:later]) (original input)))
        (is (= (original input) (transformed input)))))))

(deftest boxed-number-conversion-is-once-only-and-may-have-effects-or-trap
  (let [{:keys [original transformed]} (pair-of-calls Double/TYPE {:unchecked-math false} 'x)]
    (doseq [fail? [false true]
            invoke [original transformed]]
      (let [calls (atom 0)
            input (proxy [Number] []
                    (doubleValue []
                      (swap! calls inc)
                      (if fail? (throw (ArithmeticException. "conversion trap")) 3.5))
                    (intValue [] (throw (IllegalStateException. "wrong conversion")))
                    (longValue [] (throw (IllegalStateException. "wrong conversion")))
                    (floatValue [] (throw (IllegalStateException. "wrong conversion"))))]
        (is (= (if fail? {:exception ArithmeticException :events []}
                   {:class Double :bits (Double/doubleToRawLongBits 3.5) :events [:later]})
               (invoke input)))
        (is (= 1 @calls) "Re-unboxing the completed Double must not reexecute the caller's Number method")))))

(deftest source-context-policy-and-result-declarations-remain-explicit
  (let [policy {:unchecked-math false}
        operand (with-meta 'x {:raster.type/tag 'double :tag 'double :line 9 :ownership :fresh})
        term (completion/complete-value (completion/primitive-input-contract Float/TYPE policy) operand)
        projected (completion/project-to-jvm-source term policy)]
    (is (= {:line 9} (meta term)) "Narrowing must not preserve a stale Double declaration")
    (is (= {:line 9} (meta projected)))
    (is (identical? operand (:operand (completion/canonical-parts term))))
    (is (= [Float/TYPE] (:param-tags (meta (first projected)))))
    (is (= :unclosed-value-completion-policy
           (try (completion/project-to-jvm-source term) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :value-completion-policy-mismatch
           (try (completion/project-to-jvm-source term {:unchecked-math true}) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= (list 'quote [term])
           (completion/project-to-jvm-source (list 'quote [term]))))
    (binding [*clojure-version* {:major 1 :minor 11}]
      (is (= :unsupported-value-completion-source-dialect
             (try (completion/project-to-jvm-source term policy) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    ;; Clojure structural equality ignores metadata. The verified projection
    ;; must return its own exact method tags, never the supplied executable.
    (let [tampered (with-meta (list (with-meta (first projected) {:param-tags [Double/TYPE]}) operand)
                    (meta projected))
          verified (completion/verified-jvm-source-projection term tampered policy)]
      (is (= [Float/TYPE] (:param-tags (meta (first verified))))))))

(deftest unknown-or-nonprimitive-contexts-are-not-guessed
  (doseq [[carrier policy] [[nil {:unchecked-math false}] [Double {:unchecked-math false}]
                            [Void/TYPE {:unchecked-math false}] [Float/TYPE nil]
                            [Float/TYPE {}] [Float/TYPE {:unchecked-math :unknown}]
                            [Float/TYPE {:unchecked-math false :other :unverified}]]]
    (is (= :unsupported-value-completion
           (try (completion/primitive-input-contract carrier policy) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
