(ns raster.compiler.ir.form-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.ir.form :as form]))

;; ================================================================
;; form-info classification
;; ================================================================

(deftest form-info-leaf-test
  (testing "symbols are leaves"
    (let [info (form/form-info 'x)]
      (is (= :leaf (:kind info)))
      (is (false? (:introduces-scope? info)))
      (is (true? (:liftable? info)))))

  (testing "numbers are leaves"
    (let [info (form/form-info 42)]
      (is (= :leaf (:kind info)))))

  (testing "strings are leaves"
    (let [info (form/form-info "hello")]
      (is (= :leaf (:kind info)))))

  (testing "nil is a leaf"
    (let [info (form/form-info nil)]
      (is (= :leaf (:kind info))))))

(deftest form-info-binding-test
  (testing "let is a binding form"
    (let [info (form/form-info '(let [x 1] x))]
      (is (= :binding (:kind info)))
      (is (false? (:introduces-scope? info)))
      (is (true? (:liftable? info)))))

  (testing "let* is a binding form"
    (let [info (form/form-info '(let* [x 1] x))]
      (is (= :binding (:kind info)))
      (is (true? (:liftable? info))))))

(deftest form-info-scope-test
  (testing "dotimes introduces scope"
    (let [info (form/form-info '(dotimes [i 10] (println i)))]
      (is (= :scope (:kind info)))
      (is (true? (:introduces-scope? info)))
      (is (false? (:liftable? info)))))

  (testing "loop introduces scope"
    (let [info (form/form-info '(loop [x 0] (recur (inc x))))]
      (is (= :scope (:kind info)))
      (is (true? (:introduces-scope? info)))
      (is (false? (:liftable? info))))))

(deftest form-info-branch-test
  (testing "if is a branch"
    (let [info (form/form-info '(if true 1 0))]
      (is (= :branch (:kind info)))
      (is (false? (:introduces-scope? info)))
      (is (false? (:liftable? info)))))

  (testing "when is a branch"
    (let [info (form/form-info '(when true 1))]
      (is (= :branch (:kind info)))
      (is (false? (:liftable? info)))))

  (testing "case* is a branch"
    (let [info (form/form-info '(case* x 0 :a 1 :b))]
      (is (= :branch (:kind info))))))

(deftest form-info-lambda-test
  (testing "fn is a lambda"
    (let [info (form/form-info '(fn [x] x))]
      (is (= :lambda (:kind info)))
      (is (true? (:introduces-scope? info)))
      (is (false? (:liftable? info)))))

  (testing "fn* is a lambda"
    (let [info (form/form-info '(fn* [x] x))]
      (is (= :lambda (:kind info))))))

(deftest form-info-par-test
  (testing "raster.par/map! is a par form"
    (let [info (form/form-info '(raster.par/map! out n (fn [i] (+ i 1))))]
      (is (= :par (:kind info)))
      (is (true? (:introduces-scope? info)))
      (is (false? (:liftable? info)))
      (is (= 0 (:return-type-arg info))
          "mutating par! returns type of arg 0 (output buffer)")))

  (testing "raster.par/reduce is a par form"
    (let [info (form/form-info '(raster.par/reduce out init n (fn [acc i] acc)))]
      (is (= :par (:kind info)))
      (is (= 1 (:return-type-arg info))
          "par/reduce returns type of arg 1 (init accumulator)"))))

(deftest form-info-special-test
  (testing "try is special"
    (let [info (form/form-info '(try (+ 1 2) (catch Exception e 0)))]
      (is (= :special (:kind info)))
      (is (false? (:liftable? info)))))

  (testing "throw is special"
    (let [info (form/form-info '(throw (Exception. "err")))]
      (is (= :special (:kind info)))))

  (testing "new is special"
    (let [info (form/form-info '(new Object))]
      (is (= :special (:kind info))))))

(deftest form-info-call-test
  (testing "regular function call"
    (let [info (form/form-info '(Math/sin x))]
      (is (= :call (:kind info)))
      (is (false? (:introduces-scope? info)))
      (is (true? (:liftable? info)))))

  (testing ".invk typed call"
    (let [info (form/form-info '(.invk impl x y))]
      (is (= :invk (:kind info)))
      (is (true? (:liftable? info)))))

  (testing "do block"
    (let [info (form/form-info '(do 1 2 3))]
      (is (= :do (:kind info)))
      (is (true? (:liftable? info))))))

;; ================================================================
;; Convenience predicates
;; ================================================================

(deftest binding-form-pred-test
  (testing "let/let* are binding forms"
    (is (true? (form/binding-form? '(let [x 1] x))))
    (is (true? (form/binding-form? '(let* [x 1] x)))))

  (testing "non-binding forms return false"
    (is (false? (form/binding-form? '(if true 1 0))))
    (is (false? (form/binding-form? '(loop [x 0] x))))
    (is (false? (form/binding-form? 42)))))

(deftest introduces-scope-pred-test
  (testing "scope-introducing forms"
    (is (true? (form/introduces-scope? '(dotimes [i 10] i))))
    (is (true? (form/introduces-scope? '(loop [x 0] x))))
    (is (true? (form/introduces-scope? '(fn [x] x))))
    (is (true? (form/introduces-scope? '(raster.par/map! out n f)))))

  (testing "non-scope forms"
    (is (false? (form/introduces-scope? '(let [x 1] x))))
    (is (false? (form/introduces-scope? '(if true 1 0))))
    (is (false? (form/introduces-scope? '(+ 1 2))))
    (is (false? (form/introduces-scope? 42)))))

(deftest liftable-pred-test
  (testing "liftable forms"
    (is (true? (form/liftable? '(let [x 1] x))))
    (is (true? (form/liftable? '(.invk impl x y))))
    (is (true? (form/liftable? '(Math/sin x))))
    (is (true? (form/liftable? '(do 1 2))))
    (is (true? (form/liftable? 42))))

  (testing "non-liftable forms"
    (is (false? (form/liftable? '(if true 1 0))))
    (is (false? (form/liftable? '(dotimes [i 10] i))))
    (is (false? (form/liftable? '(loop [x 0] x))))
    (is (false? (form/liftable? '(fn [x] x))))
    (is (false? (form/liftable? '(raster.par/map! out n f))))))

(deftest scope-form-pred-test
  (testing "scope forms include scope, lambda, and par"
    (is (true? (form/scope-form? '(dotimes [i 10] i))))
    (is (true? (form/scope-form? '(loop [x 0] x))))
    (is (true? (form/scope-form? '(fn [x] x))))
    (is (true? (form/scope-form? '(raster.par/map! out n f)))))

  (testing "non-scope forms"
    (is (false? (form/scope-form? '(let [x 1] x))))
    (is (false? (form/scope-form? '(if true 1 0))))
    (is (false? (form/scope-form? '(+ 1 2))))))

(deftest call-form-pred-test
  (testing "call forms include .invk and regular calls"
    (is (true? (form/call-form? '(.invk impl x y))))
    (is (true? (form/call-form? '(Math/sin x))))
    (is (true? (form/call-form? '(f x y)))))

  (testing "non-call forms"
    (is (false? (form/call-form? '(let [x 1] x))))
    (is (false? (form/call-form? '(if true 1 0))))
    (is (false? (form/call-form? '(fn [x] x))))
    (is (false? (form/call-form? 42)))))

;; ================================================================
;; known-form-heads set
;; ================================================================

(deftest known-form-heads-test
  (testing "all classified heads are in known-form-heads"
    (doseq [head '[let let* loop loop* dotimes if when case*
                   fn fn* ftm try catch finally recur throw new
                   do .invk var]]
      (is (contains? form/known-form-heads head)
          (str head " should be in known-form-heads"))))

  (testing "special Clojure forms are included"
    (doseq [head '[quote set! letfn* reify* . def]]
      (is (contains? form/known-form-heads head)
          (str head " should be in known-form-heads")))))

;; ================================================================
;; par form mutating? detection via return-type-arg
;; ================================================================

(deftest par-mutating-detection-test
  (testing "mutating par forms (ending in !) have return-type-arg 0"
    (let [info (form/form-info '(raster.par/map! out n f))]
      (is (= 0 (:return-type-arg info)))))

  (testing "non-mutating par forms have return-type-arg 1"
    (let [info (form/form-info '(raster.par/reduce out init n f))]
      (is (= 1 (:return-type-arg info)))))

  (testing "raster.par/scan! is mutating"
    (let [info (form/form-info '(raster.par/scan! out n f))]
      (is (= 0 (:return-type-arg info))))))

;; ================================================================
;; scope-info — binder decomposition + rebuild round-trip
;; ================================================================

(deftest scope-info-non-binder-test
  (testing "non-binding forms return nil (caller recurses generically)"
    (is (nil? (form/scope-info '(clojure.core/+ a b))))
    (is (nil? (form/scope-info '(.invk impl a b))))
    (is (nil? (form/scope-info '(if a b c))))
    (is (nil? (form/scope-info '(try (foo) (finally (bar))))) "try itself binds nothing")
    (is (nil? (form/scope-info 'x))
        "a bare symbol is not a binder")))

(deftest scope-info-decomposition-test
  (testing "let* — one sequential scope, binders+inits paired"
    (let [{:keys [scopes sequential? outer]} (form/scope-info '(let* [a 1 b (+ a 2)] (+ a b)))]
      (is (true? sequential?))
      (is (= [] outer))
      (is (= 1 (count scopes)))
      (is (= '[a b] (:binders (first scopes))))
      (is (= '[1 (+ a 2)] (:inits (first scopes))))
      (is (= '[(+ a b)] (:body (first scopes))))))
  (testing "dotimes — idx binder, bound expr is outer"
    (let [{:keys [scopes outer]} (form/scope-info '(dotimes [i n] (aset a i i)))]
      (is (= '[i] (:binders (first scopes))))
      (is (= '[n] outer))))
  (testing "fn* multi-arity — one scope per arity"
    (let [{:keys [scopes]} (form/scope-info '(fn* ([x] x) ([x y] (+ x y))))]
      (is (= 2 (count scopes)))
      (is (= '[x] (:binders (first scopes))))
      (is (= '[x y] (:binders (second scopes))))))
  (testing "letfn* — recursive scope (binders visible to inits)"
    (let [{:keys [rec? scopes]} (form/scope-info '(letfn* [f (fn* [x] (g x)) g (fn* [y] (f y))] (f 1)))]
      (is (true? rec?))
      (is (= '[f g] (:binders (first scopes))))))
  (testing "catch — binds ex-sym to catch body"
    (let [{:keys [scopes]} (form/scope-info '(catch Exception e (handle e)))]
      (is (= '[e] (:binders (first scopes))))
      (is (= '[(handle e)] (:body (first scopes))))))
  (testing "par/reduce — acc + idx binders, init/bound outer"
    (let [{:keys [scopes outer]} (form/scope-info '(raster.par/reduce acc init i n (+ acc (aget a i))))]
      (is (= '[acc i] (:binders (first scopes))))
      (is (= '[init n] outer)))))

(deftest scope-info-rebuild-identity-test
  (testing "rebuild ∘ scope-info is identity for every closed-core binder form"
    (doseq [f ['(let* [a 1 b (+ a 2)] (+ a b))
               '(loop* [i 0 acc 0] (recur (inc i) (+ acc i)))
               '(dotimes [i n] (aset arr i i))
               '(fn* [x y] (+ x y))
               '(fn* self [n] (self (dec n)))
               '(fn* ([x] x) ([x y] (+ x y)))
               '(ftm [x] :- Double :raster.walker/source-body [(+ x 1)] (.invk impl x))
               '(letfn* [f (fn* [x] (g x)) g (fn* [y] (f y))] (f 1))
               '(catch Exception e (handle e))
               '(reify* [IFoo] (bar [this x] (+ x 1)) (baz [this] 0))
               '(raster.par/map! out i n nil (+ (aget in i) 1.0))
               '(raster.par/map! out i n :offset base nil (aget in i))
               '(raster.par/reduce acc 0.0 i n (+ acc (aget in i)))
               '(raster.par/scan res acc 0.0 i n nil (+ acc (aget in i)))]]
      (let [{:keys [scopes outer rebuild]} (form/scope-info f)]
        (is (= f (rebuild scopes outer)) (str "round-trip: " (first f)))))))
