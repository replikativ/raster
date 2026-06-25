(ns raster.compiler.core.util-test
  "Tests for free variable analysis — correctness of scoping across all binding forms."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.core.util :as util]))

;; ================================================================
;; free-syms: scoped free variable collection
;; ================================================================

(deftest free-syms-simple-test
  (testing "bare symbols"
    (is (= #{'x} (util/free-syms 'x)))
    (is (= #{} (util/free-syms 42)))
    (is (= #{} (util/free-syms 'clojure.core/+))))
  (testing "simple call — clojure.core vars are not free"
    (is (= #{'x 'y} (util/free-syms '(+ x y)))
        "bare + resolves to clojure.core/+, not a local")
    (is (= #{'x 'y} (util/free-syms '(clojure.core/+ x y)))
        "qualified head is not a local variable")
    (is (= #{'my-fn 'x} (util/free-syms '(my-fn x)))
        "unknown bare symbol is free (could be a local fn)"))
  (testing "dot-prefixed symbols are method calls, not variables"
    (is (= #{'x 'y} (util/free-syms '(.add x y)))
        ".add is a method call, not a free variable")
    (is (= #{} (util/free-syms '.invk))
        "bare .invk is a method symbol, not free"))
  (testing "cast intrinsics are not free"
    (is (= #{} (util/free-syms 'double))
        "double is clojure.core/double, not a local")
    (is (= #{} (util/free-syms 'long))
        "long is clojure.core/long, not a local")
    (is (= #{'x} (util/free-syms '(double x)))
        "only x is free in (double x)")))

(deftest free-syms-let-binding-test
  (testing "let* binding scopes correctly"
    (is (= #{'x} (util/free-syms '(let* [a x] a)))
        "a is bound, x is free")
    (is (= #{'x 'y} (util/free-syms '(let* [a x b y] (+ a b))))
        "a and b are bound; x and y are free; + is clojure.core"))
  (testing "sequential binding visibility"
    (is (= #{'x} (util/free-syms '(let* [a x b a] b)))
        "second binding can see first; only x is free")
    (is (= #{'x} (util/free-syms '(let* [a x b (+ a 1)] b)))
        "init of b sees a from prior binding; + is clojure.core")))

(deftest free-syms-loop-test
  (testing "loop* scopes vars into body"
    (is (= #{'n} (util/free-syms '(loop* [i 0] (if (< i n) (recur (+ i 1)) i))))
        "i is bound by loop; n is free; <, + are clojure.core")))

(deftest free-syms-dotimes-test
  (testing "dotimes scopes index"
    (is (= #{'n 'arr 'v} (util/free-syms '(dotimes [i n] (aset arr i v))))
        "i is bound, n/arr/v are free; aset is clojure.core")))

(deftest free-syms-fn-test
  (testing "fn* params are local"
    (is (= #{'y} (util/free-syms '(fn* [x] (+ x y))))
        "x is param (bound), y is free; + is clojure.core")
    (is (= #{} (util/free-syms '(fn* [x y] (+ x y))))
        "x y bound by params, + is clojure.core")))

(deftest free-syms-try-catch-test
  (testing "catch binds exception variable"
    (is (= #{'foo 'x 'bar} (util/free-syms (list 'try '(foo x) (list 'catch 'Exception 'e '(bar e)))))
        "x/foo/bar are free, e is bound by catch")))

(deftest free-syms-nested-test
  (testing "nested let + fn"
    (is (= #{'outer} (util/free-syms '(let* [a outer] (fn* [x] (+ a x)))))
        "outer is free, a bound by let, x by fn, + is clojure.core"))
  (testing "deeply nested"
    (is (= #{'z} (util/free-syms
                  (list 'let* '[a 1]
                        (list 'let* '[b a]
                              (list 'fn* '[x]
                                    (list 'loop* '[i 0]
                                          '(if (< i x) (recur (+ i z)) b)))))))
        "z is free — a, b bound by let, x by fn, i by loop; <, + are clojure.core")))

;; ================================================================
;; free-syms: par form scoping
;; ================================================================

(deftest free-syms-par-map-test
  (testing "par/map! scopes idx into body"
    (let [result (util/free-syms '(raster.par/map! out i n double (+ (aget a i) 1.0)))]
      (is (contains? result 'out) "out is free")
      (is (contains? result 'n) "bound expr is free")
      (is (contains? result 'a) "array in body is free")
      (is (not (contains? result 'i)) "idx is scoped — not free")
      (is (not (contains? result 'double)) "double is clojure.core, not free")
      (is (not (contains? result '+)) "+ is clojure.core, not free")
      (is (not (contains? result 'aget)) "aget is clojure.core, not free"))))

(deftest free-syms-par-reduce-test
  (testing "par/reduce scopes acc and idx into body"
    (is (= #{'a 'n}
           (util/free-syms '(raster.par/reduce acc 0.0 i n (+ acc (aget a i)))))
        "acc and i are scoped; a, n free; +, aget are clojure.core")))

(deftest free-syms-par-map-void-test
  (testing "par/map-void! scopes idx into body"
    (is (= #{'n 'arr 'v}
           (util/free-syms '(raster.par/map-void! i n (aset arr i v))))
        "i is scoped, n/arr/v are free; aset is clojure.core")))

;; ================================================================
;; free-syms: NEW closed-core binders (letfn*, reify*, quote)
;; ================================================================

(deftest free-syms-letfn-test
  (testing "letfn* binds fn names (mutually recursive); only outer refs are free"
    (is (= #{'q}
           (util/free-syms '(letfn* [f (fn* [x] (g x)) g (fn* [y] (f y))] (f q))))
        "f, g bound (incl. cross-references in their bodies); q free")))

(deftest free-syms-reify-test
  (testing "reify* binds this + method params"
    (is (= #{'z}
           (util/free-syms '(reify* [IFoo] (bar [this x] (+ x z)))))
        "this, x bound by the method; z free; + is clojure.core")))

(deftest free-syms-quote-opaque-test
  (testing "quote is opaque — quoted code-as-data contributes no free vars"
    (is (= #{} (util/free-syms '(quote (+ q r))))
        "q, r are quoted data, not free variable references")
    (is (= #{'live}
           (util/free-syms '(let* [a 1] (clojure.core/+ live (quote (+ q r))))))
        "only the live reference counts; the quoted form is data")))

;; ================================================================
;; subst-syms: capture-avoiding substitution
;; ================================================================

(deftest subst-syms-basic-test
  (testing "plain substitution of all free occurrences"
    (is (= '(clojure.core/+ A (clojure.core/* A 2))
           (util/subst-syms '{a A} '(clojure.core/+ a (clojure.core/* a 2)))))
    (is (= '(+ a a) (util/subst-syms {} '(+ a a))) "empty smap is identity")))

(deftest subst-syms-key-capture-test
  (testing "a binder that is an smap KEY is rebound — not substituted within"
    (is (= '(dotimes [i n] (clojure.core/+ i 1))
           (util/subst-syms '{i 5} '(dotimes [i n] (clojure.core/+ i 1))))
        "dotimes rebinds i; the 5 must not leak inside")
    (is (= '(let* [a q] a)
           (util/subst-syms '{a Z} '(let* [a q] a)))
        "let* rebinds a; body a refers to the binding, not Z")))

(deftest subst-syms-value-capture-test
  (testing "a binder that shadows an smap VALUE is alpha-renamed"
    ;; subst {x -> i} into (dotimes [i n] (+ x i)): the substituted i must stay
    ;; FREE (refer to outer i), so the binder i is renamed.
    (let [r (util/subst-syms '{x i} '(dotimes [i n] (clojure.core/+ x i)))
          binder (first (second r))]
      (is (not= 'i binder) "the colliding binder i was alpha-renamed")
      (is (contains? (util/free-syms r) 'i) "the substituted i stays free (no capture)")
      (is (not (contains? (util/free-syms r) 'x)) "x was substituted away")))
  (testing "value-capture avoidance in a par form (the inline.clj gap)"
    (let [r (util/subst-syms '{x i} '(raster.par/map! out i n nil (clojure.core/+ x (clojure.core/aget in i))))]
      (is (not= 'i (nth r 2)) "par idx binder alpha-renamed to avoid capture")
      (is (contains? (util/free-syms r) 'i) "substituted i stays free")))
  (testing "value-capture in the par variants the inliner used to MISS (map2!/scan/stencil)"
    ;; the former inline.clj subst-syms had no clause for these → :else capture.
    (doseq [f ['(raster.par/map2! o1 o2 i n cast (clojure.core/+ x (clojure.core/aget a i)) (clojure.core/* x 2))
               '(raster.par/scan res acc 0.0 i n cast (clojure.core/+ acc x))]]
      (let [r (util/subst-syms '{x i} f)]
        (is (contains? (util/free-syms r) 'i) "substituted i stays free (binder renamed)")
        (is (not (contains? (set (util/free-syms r)) 'x)) "x substituted away"))))
  (testing "value-capture in fn* renames the PARAM consistently with body refs"
    ;; regression: the fn*/reify* rebuild must use the renamed binder for the param
    ;; vector, not the original — else binder and body refs diverge and leak.
    (let [r (util/subst-syms '{x i} '(fn* [i] (clojure.core/+ x i)))
          param (first (second r))]
      (is (not= 'i param) "param renamed")
      (is (= #{'i} (util/free-syms r)) "only the substituted i is free; no leaked α-var"))))

(deftest subst-syms-quote-opaque-test
  (testing "quote is opaque to substitution"
    (is (= '(quote (+ a a)) (util/subst-syms '{a A} '(quote (+ a a)))))))

(deftest ftm-source-body-stays-in-sync
  (testing "subst/alpha rename the ftm :raster.walker/source-body CONSISTENTLY with the walked body"
    ;; Regression: the source-body is opaque to free-syms (raw unqualified payload)
    ;; but shares the arity scope — subst/alpha MUST rename it in lockstep with the
    ;; walked body, else the AD source-body desyncs (silent miscompile on re-derive).
    (let [r (util/alpha-convert '(let* [a 1] (ftm [x] :raster.walker/source-body [(+ x a)] (.invk impl x a))))
          atoms (set (flatten r))]
      ;; if the source-body were left stale, the ORIGINAL names a/x would survive
      ;; (the let binder a and ftm param x are both renamed everywhere else)
      (is (not (contains? atoms 'a)) "outer binder a fully renamed (incl. source-body)")
      (is (not (contains? atoms 'x)) "ftm param x fully renamed (incl. source-body)"))
    (let [r (util/subst-syms '{g h} '(ftm [x] :raster.walker/source-body [(+ x g)] (.invk impl x g)))]
      (is (= '(ftm [x] :raster.walker/source-body [(+ x h)] (.invk impl x h)) r)
          "g→h applied to BOTH source-body and walked body")))
  (testing "free-syms still ignores the raw source-body payload"
    (is (= #{'impl 'w}
           (util/free-syms '(ftm [x] :raster.walker/source-body [(+ x BOGUS)] (.invk impl x w))))
        "BOGUS in source-body is not a free var; only the walked body counts")))

(deftest scope-utils-map-set-literal-test
  (testing "free-syms traverses map and set literals"
    (is (= #{'foo 'x 'k 'y} (util/free-syms '(foo {k x :b y}))) "map keys and vals scanned")
    (is (= #{'foo 'x 'y} (util/free-syms '(foo #{x y}))) "set elements scanned"))
  (testing "subst-syms substitutes inside map and set literals (AD-template parity)"
    (is (= '(foo {:a Q :b [Q]}) (util/subst-syms '{x Q} '(foo {:a x :b [x]}))))
    (is (= '(foo #{Q y}) (util/subst-syms '{x Q} '(foo #{x y})))))
  (testing "alpha-convert renames binders referenced inside map/set literals"
    (let [r (util/alpha-convert '(let* [a 1] (foo {:k a} #{a})))]
      (is (= #{'foo} (util/free-syms r)) "only foo free; a renamed consistently in map+set"))))

;; ================================================================
;; alpha-convert: hygienic renaming of all binders
;; ================================================================

(deftest alpha-convert-test
  (testing "every bound var is freshened; free vars unchanged"
    (let [r (util/alpha-convert '(let* [a 1] (clojure.core/+ a free)))]
      (is (= #{'free} (util/free-syms r)) "free vars preserved")
      (is (not= 'a (first (second r))) "binder renamed")))
  (testing "non-sequential binder forms do not crash and preserve free vars"
    ;; regression: alpha-scope's simultaneous branch crashed on every non-seq form
    (doseq [[f expected-free] [['(fn* [x] (clojure.core/+ x y))            #{'y}]
                               ['(fn* [a & more] (foo a more))             #{'foo}]
                               ['(dotimes [i n] (foo i v))                  #{'n 'foo 'v}]
                               ['(letfn* [f (fn* [x] (g x)) g (fn* [y] (f y))] (f q)) #{'q}]
                               ['(reify* [IFoo] (bar [this a] (+ a z)))     #{'z}]
                               ['(catch Exception e (handle e v))           #{'handle 'v}]
                               ['(raster.par/map! out i n nil (foo i w))    #{'out 'n 'foo 'w}]]]
      (let [r (util/alpha-convert f)]
        (is (= expected-free (util/free-syms r)) (str "free preserved for " (first f))))))
  (testing "shadowing is resolved — inner and outer binders become distinct"
    (let [r (util/alpha-convert '(let* [a 1] (let* [a 2] (clojure.core/+ a a))))
          outer-a (first (second r))
          inner   (nth r 2)
          inner-a (first (second inner))
          body    (nth inner 2)]
      (is (not= outer-a inner-a) "outer and inner a are distinct fresh names")
      (is (= inner-a (second body)) "inner body refers to the inner binder")))
  (testing "free vars are not captured by a same-named binder in an init"
    ;; (let* [a 1 b a] b): the b-init `a` refers to the let's a, must track rename
    (let [r (util/alpha-convert '(let* [a 1 b a] b))
          binds (second r)
          a' (nth binds 0) b-init (nth binds 3)]
      (is (= a' b-init) "b's init references the renamed a, not a stale name"))))

