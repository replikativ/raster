(ns raster.compiler.jvm-dense-contraction-blas-test
  "The JVM dense-contraction BLAS leaf is an opt-in schedule choice with checked admission:
   numerical permission, run-time capacity/range, runtime aliasing and provider availability
   each fall back to the generated host expansion. Tests that need a native BLAS check for it
   individually; every fallback test runs without one."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as rn]
            [raster.par :as par]
            [raster.dl.nn :as nn]
            [raster.linalg.blas :as blas]
            [raster.compiler.backend.jvm.par-simd :as ps]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.pipeline :as pipeline]))

(defn- blas? [] (blas/gemm-provider-ready?))

(defmacro ^:private when-blas [& body]
  `(if (blas?) (do ~@body) (println "[SKIP] no BLAS provider:" ~(str *ns*))))

;; A BLAS-spelled product: canonicalized to par/contract with :abstract-blas-product arithmetic.
(deftm blas-leaf-mm [A :- (Array double) B :- (Array double) m :- Long k :- Long n :- Long] :- (Array double)
  (nn/matmul A B m k n))

;; The same product written as an ordered contraction: retained typed-SSA arithmetic.
(deftm ssa-leaf-mm [A :- (Array double) B :- (Array double) m :- Long k :- Long n :- Long] :- (Array double)
  (let [C (double-array (* m n))]
    (par/contract C [[i m] [j n]] [[l k]]
                  (rn/* (ra/aget A (+ (* i k) l)) (ra/aget B (+ (* l n) j))))
    C))

(defn- rand-array [n seed]
  (let [r (java.util.Random. seed)] (double-array (repeatedly n #(.nextGaussian r)))))

(defn- naive-mm [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (loop [l 0 acc 0.0]
                (if (< l k) (recur (inc l) (+ acc (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))) acc)))))
    C))

(defn- close? [xs ys]
  (and (= (count xs) (count ys))
       (every? #(< (Math/abs (double %)) 1e-12) (map - xs ys))))

(defn- compiled [leaf var]
  (binding [ps/*dense-contraction-leaf* leaf]
    {:stats (:backend-applied-stats (pipeline/show-pipeline var))
     :f (pipeline/compile-aot var)}))

(def ^:private shape [7 5 3])

(deftest generated-leaf-is-the-default
  (is (= :generated ps/*dense-contraction-leaf*))
  (let [{:keys [stats f]} (compiled ps/*dense-contraction-leaf* #'blas-leaf-mm)
        [m k n] shape A (rand-array (* m k) 1) B (rand-array (* k n) 2)]
    (is (nil? (:blas-contractions stats)))
    (is (= (vec (naive-mm A B m k n)) (vec (f A B m k n))) "ordered generated loop")))

(deftest blas-leaf-runs-an-admitted-product
  (when-blas
   (let [{:keys [stats f]} (compiled :blas #'blas-leaf-mm)
         [m k n] shape A (rand-array (* m k) 3) B (rand-array (* k n) 4)
         guard-calls (atom 0)
         original blas/dense-gemm-admissible?
         C (with-redefs [blas/dense-gemm-admissible? (fn [& args] (swap! guard-calls inc) (apply original args))]
             (f A B m k n))]
     (is (= 1 (:blas-contractions stats)))
     (is (= 1 @guard-calls) "the run-time guard selects the leaf on every call")
     (is (close? (vec C) (vec (naive-mm A B m k n)))))))

(deftest numerical-permission-is-required
  (testing "an ordered (retained typed-SSA) contraction keeps its generated loop under :blas"
    (let [{:keys [stats f]} (compiled :blas #'ssa-leaf-mm)
          [m k n] shape A (rand-array (* m k) 5) B (rand-array (* k n) 6)]
      (is (nil? (:blas-contractions stats)))
      (is (= {:numerical-permission 1} (:blas-declines stats)))
      (is (= (vec (naive-mm A B m k n)) (vec (f A B m k n))))))
  (testing "the plan states the reason"
    (let [form (list 'raster.par/contract 'C '[[i m] [j n]] '[[l k]]
                     '(clojure.core/* (clojure.core/aget A (clojure.core/+ (clojure.core/* i k) l))
                                      (clojure.core/aget B (clojure.core/+ (clojure.core/* l n) j))))]
      (is (= :numerical-permission
             (:reason (ps/dense-contract-blas-plan (with-meta form {:raster.type/elem-type :double})))))
      (is (= :element-type
             (:reason (ps/dense-contract-blas-plan
                       (with-meta form {:raster.type/elem-type :long
                                        :raster.source/arithmetic numerics/retained-source-arithmetic}))))))))

(deftest provider-unavailable-falls-back
  (testing "at compile time"
    (with-redefs [blas/gemm-provider-ready? (constantly false)]
      (let [{:keys [stats f]} (compiled :blas #'blas-leaf-mm)
            [m k n] shape A (rand-array (* m k) 7) B (rand-array (* k n) 8)]
        (is (nil? (:blas-contractions stats)))
        (is (= {:provider-unavailable 1} (:blas-declines stats)))
        (is (= (vec (naive-mm A B m k n)) (vec (f A B m k n)))))))
  (testing "at run time (a leaf compiled while the provider was present)"
    (when-blas
     (let [{:keys [f]} (compiled :blas #'blas-leaf-mm)
           [m k n] shape A (rand-array (* m k) 9) B (rand-array (* k n) 10)]
       (with-redefs [blas/gemm-provider-ready? (constantly false)]
         (is (= (vec (naive-mm A B m k n)) (vec (f A B m k n))) "exact generated loop"))))))

(deftest capacity-violations-keep-generated-bounds-checks
  (let [[m k n] shape
        A (rand-array (dec (* m k)) 11) B (rand-array (* k n) 12)
        failure (fn [f] (try (f A B m k n) nil (catch Throwable t (class t))))
        generated (failure (:f (compiled :generated #'blas-leaf-mm)))]
    (is (some? generated) "an A one element short is out of bounds")
    (when-blas
     (is (= generated (failure (:f (compiled :blas #'blas-leaf-mm))))
         "the BLAS leaf never reads past the array; it fails exactly like the generated loop"))))

(deftest runtime-guard-checks-alias-capacity-and-range
  (let [a (double-array 12) b (double-array 12) c (double-array 12)]
    (with-redefs [blas/gemm-provider-ready? (constantly true)]
      (is (true? (blas/dense-gemm-admissible? :double a b c 3 4 3)))
      (is (true? (blas/dense-gemm-admissible? :double a a c 3 4 3)) "A and B may be the same input")
      (is (false? (blas/dense-gemm-admissible? :double a b a 3 4 3)) "C aliases A")
      (is (false? (blas/dense-gemm-admissible? :double a b b 3 4 3)) "C aliases B")
      (is (false? (blas/dense-gemm-admissible? :double a b c 4 4 3)) "m·k > |A|")
      (is (false? (blas/dense-gemm-admissible? :double a b c 3 4 5)) "k·n > |B|")
      (is (false? (blas/dense-gemm-admissible? :double a b c -1 4 3)))
      (is (false? (blas/dense-gemm-admissible? :double a b c 3 (inc Integer/MAX_VALUE) 0)))
      (is (false? (blas/dense-gemm-admissible? :float a b c 3 4 3)) "element class")
      (is (false? (blas/dense-gemm-admissible? :double a b nil 3 4 3))))
    (with-redefs [blas/gemm-provider-ready? (constantly false)]
      (is (false? (blas/dense-gemm-admissible? :double a b c 3 4 3))))))

(deftest aliased-output-runs-the-generated-loop
  ;; Distinct argument names do not prove distinct buffers: an admitted plan whose C is the
  ;; same array as A at run time must take the generated in-place loop, never BLAS.
  (let [form (with-meta
               (list 'raster.par/contract 'C '[[i m] [j n]] '[[l k]]
                     '(clojure.core/* (clojure.core/aget A (clojure.core/+ (clojure.core/* i k) l))
                                      (clojure.core/aget B (clojure.core/+ (clojure.core/* l n) j))))
               {:raster.type/elem-type :double
                :raster.source/arithmetic (numerics/blas-source-arithmetic :double)})
        fallback (walk/macroexpand-all form)
        params '[A B C m k n]
        hint (fn [body] (list 'fn ['A 'B 'C 'm 'k 'n]
                              (list 'let* '[A (clojure.core/doubles A) B (clojure.core/doubles B)
                                            C (clojure.core/doubles C) m (clojure.core/long m)
                                            k (clojure.core/long k) n (clojure.core/long n)]
                                    body)))
        generated (eval (hint fallback))
        have-blas (blas?)
        m 3 k 3 n 3]
    (with-redefs [blas/gemm-provider-ready? (constantly true)]
      (let [plan (ps/dense-contract-blas-plan form)
            leaf (eval (hint (ps/dense-contract-blas-form plan fallback)))]
        (is (:ok plan) (pr-str plan))
        (is (= params (vec (concat [(:A plan) (:B plan) (:out plan)] (:dimensions plan)))))
        (let [in-place-generated (rand-array 9 13) in-place-leaf (aclone in-place-generated)
              B (rand-array 9 14)]
          (generated in-place-generated B in-place-generated m k n)
          (leaf in-place-leaf B in-place-leaf m k n)
          (is (= (vec in-place-generated) (vec in-place-leaf))
              "C identical to A: same in-place result as the generated loop"))
        (if have-blas
          (let [A (rand-array 9 15) B (rand-array 9 16) C (double-array 9)]
            (leaf A B C m k n)
            (is (close? (vec C) (vec (naive-mm A B m k n))) "disjoint buffers take the GEMM"))
          (println "[SKIP] no BLAS provider: disjoint-buffer GEMM"))))))
