(ns raster.compiler.pe-test
  (:require [clojure.test :refer [deftest testing is are]]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.core :refer [deftm]]))

;; ================================================================
;; Constant propagation
;; ================================================================

(deftest constant-propagation-test
  (testing "known constant substituted"
    (is (= '(* 3.0 y)
           (pe/pe '(* x y) {'x 3.0}))))
  (testing "multiple constants"
    (is (= 12.0
           (pe/pe '(* x y) {'x 3.0 'y 4.0}))))
  (testing "unknown variable preserved"
    (is (= '(* x y)
           (pe/pe '(* x y) {})))))

;; ================================================================
;; Let inlining
;; ================================================================

(deftest let-inlining-test
  (testing "constant let binding propagated"
    (is (= '(* 3.0 x)
           (pe/pe '(let* [a 3.0] (* a x))))))
  (testing "single-use trivial binding inlined"
    (is (= '(* y x)
           (pe/pe '(let* [a y] (* a x))))))
  (testing "multi-use binding preserved"
    (let [result (pe/pe '(let* [a y] (* a a)))]
      (is (seq? result))
      (is (= 'let* (first result)))))
  (testing "nested let constant propagation"
    (is (= '(* 6.0 x)
           (pe/pe '(let* [a 3.0 b (* a 2.0)] (* b x)))))))

;; ================================================================
;; Dead binding elimination
;; ================================================================

(deftest dead-binding-test
  (testing "unused binding eliminated"
    (is (= 'x (pe/pe '(let* [a 42] x)))))
  (testing "used binding preserved"
    (let [result (pe/pe '(let* [a y] (+ a 1)))]
      ;; a is used once and is trivial (symbol), so should be inlined
      (is (= '(+ y 1) result)))))

;; ================================================================
;; Dead branch elimination
;; ================================================================

(deftest dead-branch-test
  (testing "(if true x y) => x"
    (is (= 'x (pe/pe '(if true x y)))))
  (testing "(if false x y) => y"
    (is (= 'y (pe/pe '(if false x y)))))
  (testing "(if nil x y) => y"
    (is (= 'y (pe/pe '(if nil x y)))))
  (testing "unknown test preserved"
    (let [result (pe/pe '(if p x y))]
      (is (seq? result))
      (is (= 'if (first result))))))

;; ================================================================
;; Do form simplification
;; ================================================================

(deftest do-simplification-test
  (testing "single-form do unwrapped"
    (is (= 'x (pe/pe '(do x)))))
  (testing "constant-only init forms removed"
    (is (= 'x (pe/pe '(do 42 "hello" x))))))

;; ================================================================
;; Combined: constant prop + simplify
;; ================================================================

(deftest combined-test
  (testing "Lorenz-style parameter specialization"
    ;; Lorenz: dx = sigma * (y - x)
    ;; With known sigma = 10.0, this should simplify
    (let [result (pe/pe '(* sigma (- y x)) {'sigma 10.0})]
      (is (= '(* 10.0 (- y x)) result))))

  (testing "constant arithmetic chain"
    ;; (let [a 2.0 b 3.0] (* a b x)) => (* 6.0 x)
    (is (= '(* 6.0 x)
           (pe/pe '(let* [a 2.0 b 3.0] (* (* a b) x))))))

  (testing "zero multiplication elimination"
    ;; (let [a 0.0] (* a x)) => 0.0
    (is (= 0.0
           (pe/pe '(let* [a 0.0] (* a x)))))))

;; ================================================================
;; Loop preservation
;; ================================================================

(deftest loop-preservation-test
  (testing "loop bindings not eliminated (needed by recur)"
    (let [result (pe/pe '(loop [i 0 acc 0.0]
                           (if (< i 10)
                             (recur (+ i 1) (+ acc 1.0))
                             acc)))]
      ;; Loop form should be preserved (not eliminated)
      (is (seq? result))
      ;; The head should still be loop or loop*
      (is (#{'loop 'loop*} (first result))))))

;; ================================================================
;; Fixpoint behavior
;; ================================================================

(deftest fixpoint-test
  (testing "fixpoint converges"
    ;; Multiple rounds needed: first simplify inner, then outer
    (is (= 'x (pe/pe '(+ (+ x 0) 0)))))
  (testing "fixpoint with derivative cleanup"
    (is (= 'x (pe/pe '(+ (* 1.0 x) 0.0))))))

;; ================================================================
;; PE on walked body
;; ================================================================

(deftest pe-walked-body-test
  (testing "specialize with known params"
    (let [body ['(* sigma (- y x))]
          result (pe/pe-walked-body
                  '[sigma y x]
                  {'sigma 10.0}
                  body)]
      (is (= ['(* 10.0 (- y x))] result))))
  (testing "all params known => fully constant"
    (let [body ['(+ a b)]
          result (pe/pe-walked-body '[a b] {'a 3.0 'b 4.0} body)]
      (is (= [7.0] result))))
  (testing "no params known => no change"
    (let [body ['(+ a b)]
          result (pe/pe-walked-body '[a b] {} body)]
      (is (= ['(+ a b)] result)))))

;; ================================================================
;; .invk preservation
;; ================================================================

(deftest invk-preservation-test
  (testing ".invk args are PE'd"
    (let [result (pe/pe '(.invk some-impl x 3.0) {'x 2.0})]
      ;; Both args constant → should be a simplified .invk or constant
      (is (or (number? result)
              (and (seq? result) (= '.invk (first result)))))))
  (testing ".invk with non-constant args preserved"
    (let [result (pe/pe '(.invk some-impl x y))]
      (is (seq? result))
      (is (= '.invk (first result))))))

;; ================================================================
;; Edge cases
;; ================================================================

(deftest edge-cases-test
  (testing "nil form"
    (is (nil? (pe/pe nil))))
  (testing "literal passthrough"
    (is (= 42 (pe/pe 42)))
    (is (= "hello" (pe/pe "hello"))))
  (testing "symbol passthrough"
    (is (= 'x (pe/pe 'x))))
  (testing "quote preserved"
    (is (= '(quote foo) (pe/pe '(quote foo)))))
  (testing "fn preserved"
    (let [result (pe/pe '(fn [x] (* x x)))]
      (is (seq? result))
      (is (#{'fn 'fn*} (first result))))))

;; ================================================================
;; Alias propagation soundness under sequential shadowing (#72 find)
;; ================================================================

(deftest alias-shadowed-rhs-test
  (testing "a symbol alias whose RHS is REBOUND later must be kept"
    ;; Kh := out, then out is rebound; the closure and the later call must
    ;; keep reading the FIRST out. Dropping Kh + env substitution would make
    ;; them read the rebound out (silent wrong value) — PE must keep Kh.
    (let [form '(let* [out (clojure.core/double-array 3)
                       Kh out
                       out (clojure.core/double-array 4)
                       r (some.ns/f Kh out)]
                      r)
          result (pe/pe form)
          bindings (partition 2 (second result))]
      (is (some #(= 'Kh (first %)) bindings)
          "alias with later-shadowed RHS survives")
      (is (= '(some.ns/f Kh out) (second (last bindings)))
          "use site still references the alias, not the rebound RHS")))

  (testing "a symbol alias with a STABLE RHS is still propagated"
    (let [form '(let* [a b
                       r (some.ns/f a)]
                      r)
          result (pe/pe form)]
      (is (not (some #{'a} (flatten result)))
          "stable alias is inlined away")))

  (testing "kept rebinding invalidates a previously propagated constant"
    ;; x := 1.0 propagates; then x is REBOUND to a call — later refs must
    ;; use the new x, not the stale constant.
    (let [form '(let* [x 1.0
                       x (some.ns/g)
                       r (some.ns/f x)]
                      r)
          result (pe/pe form)
          bindings (partition 2 (second result))]
      (is (= '(some.ns/f x) (second (last bindings)))
          "post-rebind reference is NOT replaced by the stale constant"))))

(deftest fn*-wrapped-arity-substitution-test
  (testing "env substitution reaches (fn* ([params] body)) — the reified
            AD pullback shape (dangling-ref regression, #72 find)"
    ;; h := x is droppable (x never rebound); the closure body must get the
    ;; substitution — otherwise its h is a dangling symbol at bytecode emit.
    (let [form '(let* [h x
                       pb (fn* ([dy] (some.ns/f h dy)))]
                      pb)
          result (pe/pe form)]
      (is (not (some #{'h} (flatten result)))
          "alias substituted inside the wrapped-arity closure body"))))

;; ================================================================
;; Specialize-var API on real deftm
;; ================================================================

(deftm pe-test-fn [^double a ^double b ^double x]
  (+ (* a x) b))

(deftest specialize-var-test
  (testing "specialize with known a and b"
    (let [mangled-sym 'raster.compiler.pe-test/pe-test-fn_m_double_double_double
          result (pe/specialize-var (resolve mangled-sym) {'a 2.0 'b 3.0})]
      (is (vector? (:params result)))
      (is (= '[a b x] (:params result)))
      (is (vector? (:body result)))
      ;; The body should have a and b inlined
      (is (not (some #{'a 'b} (flatten (:body result))))
          "Known params should be substituted")
      (is (some #{'x} (flatten (:body result)))
          "Unknown param x should remain")))
  (testing "specialize-body convenience"
    (let [mangled-sym 'raster.compiler.pe-test/pe-test-fn_m_double_double_double
          body (pe/specialize-body (resolve mangled-sym) {'a 2.0 'b 3.0})]
      (is (vector? body)))))
