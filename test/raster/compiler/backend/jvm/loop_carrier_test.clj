(ns raster.compiler.backend.jvm.loop-carrier-test
  (:refer-clojure :exclude [aget aset])
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays :refer [aget aset alloc-like]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.pipeline :as pipeline]))

(deftm grouped-carry
  (All [T] [src :- (Array T) rows :- Long group :- Long slab :- Long] :- (Array T)
    (let [out (alloc-like src (* rows slab))]
      (par/map-void! o (clojure.core/* rows slab)
        (let [g (quot o slab) i (rem o slab) gg (clojure.core/* g group)]
          (aset out o
            (loop [r 1 acc (aget src (clojure.core/+ (clojure.core/* gg slab) i))]
              (if (< r group)
                (recur (inc r)
                       (n/+ acc (aget src (clojure.core/+ (clojure.core/* (clojure.core/+ gg r) slab) i))))
                acc)))))
      out)))

(deftm widening-carry [x :- Double count :- Long] :- Double
  (loop [i 0 acc 0]
    (if (< i count) (recur (inc i) (+ acc x)) acc)))

(deftm coupled-carry [step :- Double] :- Double
  (loop [a (float 1.0e8) b (float 0.0) i 0]
    (if (< i 2) (recur (+ a step) (+ b a) (inc i)) b)))

(deftest mapped-float-carry-retains-per-add-rounding
  (let [input (float-array [1.0e8 4.0 1.0 2.0 -1.0e8 -1.0])
        ordinary (grouped-carry input 1 3 2)
        compiled (pipeline/compile-aot #'grouped-carry :dtype :float)]
    (is (= [0.0 5.0] (vec ordinary)))
    (is (= [0.0 5.0] (vec (compiled input 1 3 2))))))

(deftest mapped-double-carry-retains-double-arithmetic
  (let [input (double-array [1.0e8 1.0 -1.0e8])
        ordinary (grouped-carry input 1 3 1)
        compiled (pipeline/compile-aot #'grouped-carry :dtype :double)]
    (is (= [1.0] (vec ordinary)))
    (is (= [1.0] (vec (compiled input 1 3 1))))))

(deftest genuine-recurrence-widening-is-not-truncated
  (is (== 1.5 (widening-carry 0.5 3)))
  (is (== 1.5 ((pipeline/compile-aot #'widening-carry) 0.5 3))))

(deftest dependent-carry-widening-reaches-a-fixed-point
  (let [pairs [['a '(float 1.0e8)] ['b '(float 0.0)] ['i 0]]
        body '((if (< i 2) (recur (+ a step) (+ b a) (inc i)) b))]
    (is (= [:double :double :int]
           (#'bytecode/loop-recur-types-in-scope pairs body {'step {:type :double}})))))

(deftest coupled-carries-preserve-the-wider-value
  (is (== 200000001.0 (coupled-carry 1.0)))
  (is (== 200000001.0 ((pipeline/compile-aot #'coupled-carry) 1.0))))

(deftest branch-width-joins-do-not-forget-earlier-carry-widening
  (is (= [:double :double]
         (#'bytecode/loop-recur-types-in-scope
           [['a 0] ['b '(long 1)]]
           '((recur (if p b (long 0)) (+ b step)))
           {'step {:type :double} 'p {:type :bool}}))))

(deftest long-and-float-recurrences-use-the-common-numeric-width
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((if p (recur (float 1.0)) (recur (long 100000001)))) {'p {:type :bool}}))))

(deftest unknown-carry-shadows-an-outer-type
  (is (= [:ref]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(opaque)]] '((recur a)) {'a {:type :float}}))))

(deftest unknown-nested-local-shadows-the-loop-carry
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((let* [a (opaque)] (recur (+ a (float 1.0))))) {}))))

(deftest recurrence-value-preserves-local-scope-and-tail-joins
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((recur (let* [a (opaque)] (+ a (float 1.0))))) {})))
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((recur (do (opaque) (if p (long 100000001) (float 1.0)))))
           {'p {:type :bool}}))))

(deftest recurrence-branches-and-initializers-preserve-nested-scope
  (doseq [expression
          '[(if p (let* [a (opaque)] (+ a (float 1.0))) (float 0.0))
            (let* [b (let* [a (opaque)] (+ a (float 1.0)))] b)
            (+ (float 1.0) (let* [a (opaque)] (+ a (float 1.0))))]]
    (is (= [:double]
           (#'bytecode/loop-recur-types-in-scope
             [['a '(float 0.0)]] (list (list 'recur expression))
             {'p {:type :bool}})))))

(deftest noncanonical-array-read-name-is-not-an-intrinsic
  (let [ns-name (gensym "raster.compiler.backend.jvm.noncanonical-read-probe-")
        probe (create-ns ns-name)]
    (try
      (intern probe 'aget (fn [_ _] -3.25))
      (binding [*ns* probe]
        (is (nil? (#'bytecode/infer-arg-stack-type '(aget src 0) {'src {:type :ref :hint 'floats}})))
        (is (= :double (#'bytecode/infer-arg-stack-type '(Math/abs (aget src 0))
                                                       {'src {:type :ref :hint 'floats}}))))
      ;; Caller/source namespace differences cannot turn a bare name into a
      ;; canonical read. Explicit lowered reads remain typed in either context.
      (binding [*ns* (the-ns 'clojure.core)]
        (is (nil? (#'bytecode/infer-arg-stack-type '(aget src 0) {'src {:type :ref :hint 'floats}})))
        (is (= :float (#'bytecode/infer-arg-stack-type '(clojure.core/aget src 0)
                                                       {'src {:type :ref :hint 'floats}}))))
      (is (nil? (#'bytecode/infer-arg-stack-type '(aget src 0)
                     {'aget {:type :ref} 'src {:type :ref :hint 'floats}})))
      (finally (remove-ns ns-name)))))
