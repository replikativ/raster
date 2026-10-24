(ns raster.compiler.jvm-loop-anf-test
  "Loop operands in call position are A-normalized for JVM on-stack replacement, and the
   rewrite preserves the evaluation order of EVERY operand, including Var reads.
   Needs no native library: the compiled case is a host-expanded par/contract."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as rn]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bc]
            [raster.compiler.pipeline :as pipeline]))

(def ^:dynamic source 1)

(defn- run [form]
  (binding [*ns* (the-ns 'raster.compiler.jvm-loop-anf-test)]
    ((eval (list 'fn [] (list 'clojure.core/binding '[source 1] form))))))

(defn- lift [form]
  (binding [*ns* (the-ns 'raster.compiler.jvm-loop-anf-test)]
    (bc/lift-loop-operands form)))

(def ^:private set-source-loop
  "A loop that changes `source` to 9 and evaluates to 1."
  '(loop* [i 0] (if (< i 1) (do (set! source 9) (recur (inc i))) i)))

(deftest review-counterexample-keeps-its-result
  ;; Review counterexample (2026-10-24): lifting only the later loop delayed the earlier
  ;; mutable Var read, turning [1 1] into [9 1].
  (let [form (list 'vector 'source set-source-loop)]
    (is (= [1 1] (run form)))
    (is (= [1 1] (run (lift form))))
    (testing "a non-inline Var head is read before its operands, so the call is not rewritten"
      (is (= form (lift form))))))

(deftest var-read-operands-are-captured-in-place
  (testing "an inline head is rewritten, and the earlier Var read is bound before the loop"
    (let [form (list 'clojure.core/+ 'source set-source-loop)
          lifted (lift form)]
      (is (= 2 (run form)))
      (is (= 2 (run lifted)))
      (is (= 'let* (first lifted)))
      (is (= 'source (second (second lifted))) "the Var read is the first captured operand")
      (is (= 4 (count (second lifted))))))
  (testing "a Var array index read before a loop that changes it"
    (let [form (list 'clojure.core/let ['out '(long-array 2)]
                     (list 'clojure.core/aset 'out 'source
                           '(loop* [i 0] (if (< i 1) (do (set! source 0) (recur (inc i))) 7)))
                     '(vec out))]
      (is (= [0 7] (run form)))
      (is (= [0 7] (run (walk/macroexpand-all form))))
      (is (= [0 7] (run (lift (walk/macroexpand-all form)))))))
  (testing "qualified static-field reads are captured, not treated as locals"
    (let [lifted (lift '(clojure.core/+ Integer/MAX_VALUE (loop* [x 1] x)))]
      (is (= 'Integer/MAX_VALUE (second (second lifted)))))))

(deftest locals-and-constants-stay-in-place
  (let [form '(clojure.core/aset out i (loop* [l 0 acc 0.0] (if (< l 3) (recur (inc l) (+ acc l)) acc)))
        lifted (lift form)]
    (is (= 'let* (first lifted)))
    (is (= 2 (count (second lifted))) "only the loop is bound")
    (is (= '(clojure.core/aset out i) (take 3 (nth lifted 2))))
    (let [out (double-array 2)]
      ((eval `(fn [~'out ~'i] ~lifted)) out 1)
      (is (= [0.0 3.0] (vec out)))))
  (let [lifted (lift '(clojure.core/+ 1.0 (loop* [x 1.0] x)))]
    (is (= 2 (count (second lifted))) "constants are not bound")
    (is (= 1.0 (second (nth lifted 2))))))

(deftest declined-and-unchanged-forms
  (testing "forms without loops in operand position are unchanged"
    (is (= '(f a (g b)) (lift '(f a (g b)))))
    (is (= '(if (loop* [x 1] x) a b) (lift '(if (loop* [x 1] x) a b)))))
  (testing "a Var-resolved method target is not turned into a local"
    (is (= '(.invk raster.numeric/+ (loop* [x 1] x))
           (lift '(.invk raster.numeric/+ (loop* [x 1] x))))))
  (testing "operands inside reify*/deftype* are left to their own scoping"
    (let [form '(reify* [clojure.lang.IDeref] (deref [this] (clojure.core/+ 1 (loop* [x 1] x))))]
      (is (= 'reify* (first (lift form)))))))

(deftm anf-host-mm [A :- (Array double) B :- (Array double) m :- Long k :- Long n :- Long] :- (Array double)
  (let [C (double-array (* m n))]
    (par/contract C [[i m] [j n]] [[l k]]
                  (rn/* (ra/aget A (+ (* i k) l)) (ra/aget B (+ (* l n) j))))
    C))

(defn- naive-mm [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (loop [l 0 acc 0.0]
                (if (< l k) (recur (inc l) (+ acc (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))) acc)))))
    C))

(deftest host-expanded-contraction-is-lifted-and-correct
  (let [lifted (atom 0)
        original bc/lift-loop-operands
        f (with-redefs [bc/lift-loop-operands
                        (fn [form]
                          (let [r (original form)]
                            (when-not (= r form) (swap! lifted inc))
                            r))]
            (pipeline/compile-aot #'anf-host-mm))
        [m k n] [7 5 3]
        r (java.util.Random. 7)
        A (double-array (repeatedly (* m k) #(.nextGaussian r)))
        B (double-array (repeatedly (* k n) #(.nextGaussian r)))]
    (is (pos? @lifted) "the host contraction's inner loop was A-normalized")
    (is (= (vec (naive-mm A B m k n)) (vec (f A B m k n)))
        "same left-to-right summation as the scalar reference")))
