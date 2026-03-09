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
