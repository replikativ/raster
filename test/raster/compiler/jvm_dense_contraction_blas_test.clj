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

(defn- dense-form [elem variant]
  (let [left (case variant
               :tn '(clojure.core/+ (clojure.core/* l m) i)
               '(clojure.core/+ (clojure.core/* i k) l))
        right (case variant
                :nt '(clojure.core/+ (clojure.core/* j k) l)
                '(clojure.core/+ (clojure.core/* l n) j))]
    (with-meta
      (list 'raster.par/contract 'C '[[i m] [j n]] '[[l k]]
            (list 'clojure.core/* (list 'clojure.core/aget 'A left)
                  (list 'clojure.core/aget 'B right)))
      {:raster.type/elem-type elem :raster.source/arithmetic (numerics/blas-source-arithmetic elem)})))

(defn- leaf-function [elem plan fallback]
  (let [array-cast (case elem :double 'clojure.core/doubles :float 'clojure.core/floats)
        array-tag (case elem :double 'doubles :float 'floats)
        typed-array #(with-meta % {:tag array-tag})
        parameters (into (mapv typed-array '[A B C]) '[m k n])]
    (eval (list 'fn parameters
                (list 'let* (vec (concat
                                  (mapcat (fn [id] [(typed-array id) (list array-cast id)]) '[A B C])
                                  (mapcat (fn [id] [id (list 'clojure.core/long id)]) '[m k n])))
                      (ps/dense-contract-blas-form plan fallback))))))

(deftest readiness-prepares-the-selected-gemm-before-branch-selection
  (doseq [[elem handle] [[:double #'blas/dgemm-mh] [:float #'blas/sgemm-mh]]]
    (let [preparations (atom 0)
          missing (delay (swap! preparations inc)
                         (throw (ex-info "missing typed GEMM" {:dtype elem})))
          other (if (= elem :double) #'blas/sgemm-mh #'blas/dgemm-mh)]
      (with-redefs-fn {#'blas/gemm-provider (delay true) handle missing other (delay ::prepared)}
        (fn []
          (is (false? (blas/gemm-provider-ready? elem)))
          (is (true? (blas/gemm-provider-ready? (if (= elem :double) :float :double))))
          (when (= elem :double) (is (false? (blas/gemm-provider-ready?))))
          (is (= 1 @preparations)))))))

(deftest typed-readiness-is-used-by-compiler-and-runtime
  (doseq [elem [:double :float]]
    (let [seen (atom [])
          array (if (= elem :double) double-array float-array)]
      (with-redefs [blas/gemm-provider-ready? (fn [actual] (swap! seen conj actual) false)]
        (is (= :provider-unavailable (:reason (ps/dense-contract-blas-plan (dense-form elem :nn)))))
        (is (false? (blas/dense-gemm-admissible? elem (array 1) (array 1) (array 1) 1 1 1)))
        (is (= [elem elem] @seen))))))

(deftest qualified-native-leaves-cover-both-dtypes-orientations-and-empty-domains
  (doseq [elem [:double :float] variant [:nn :nt :tn]]
    (if-not (blas/gemm-provider-ready? elem)
      (println "[SKIP] typed GEMM unavailable:" elem variant)
      (let [form (dense-form elem variant)
            plan (ps/dense-contract-blas-plan form)
            leaf (leaf-function elem plan (walk/macroexpand-all form))
            array (if (= elem :double) double-array float-array)
            guard blas/dense-gemm-admissible?]
        (is (:ok plan) (pr-str plan))
        (doseq [[m k n] [[2 3 2] [0 3 2] [2 3 0] [2 0 3]]]
          (let [A (array (range 1 (inc (* m k))))
                B (array (range 7 (+ 7 (* k n))))
                C (array (repeat (+ 2 (* m n)) -7.0))
                selected (atom [])
                expected (vec
                          (for [i (range m) j (range n)]
                            (reduce + 0.0
                                    (for [l (range k)]
                                      (* (aget A (if (= variant :tn) (+ (* l m) i) (+ (* i k) l)))
                                         (aget B (if (= variant :nt) (+ (* j k) l) (+ (* l n) j))))))))]
            (with-redefs [blas/dense-gemm-admissible?
                          (fn [& args]
                            (let [admitted? (apply guard args)]
                              (swap! selected conj admitted?) admitted?))]
              (is (identical? C (leaf A B C m k n))))
            (is (= [true] @selected) "the qualified BLAS branch, not fallback, executed")
            (is (= expected (subvec (vec C) 0 (* m n))))
            (is (= [-7.0 -7.0] (subvec (vec C) (* m n))))))))))

(deftest negative-zero-neutral-keeps-its-generated-result
  (let [base (dense-form :double :nn)
        form (with-meta (apply list (concat base [:init -0.0])) (meta base))
        fallback (walk/macroexpand-all form)]
    (with-redefs [blas/gemm-provider-ready? (constantly true)]
      (is (= :signed-zero-init (:reason (ps/dense-contract-blas-plan form)))))
    (let [generated (eval (list 'fn '[A B C m k n] fallback))
          C (double-array [7.0])]
      (generated (double-array 0) (double-array 0) C 1 0 1)
      (is (= (Double/doubleToRawLongBits -0.0)
             (Double/doubleToRawLongBits (aget C 0)))))))

(deftest selected-gemm-failure-is-not-replayed-through-the-fallback
  (let [A (double-array [2.0]) B (double-array [3.0]) C (double-array [-1.0])
        fallback-calls (atom 0)
        failure (ex-info "GEMM failed after mutation" {:test :native-failure})]
    (with-redefs [blas/gemm-provider-ready? (constantly true)
                  blas/dgemm! (fn [_left _right output & _arguments]
                                (aset ^doubles output 0 9.0)
                                (throw failure))
                  ;; A named sentinel keeps eval's fallback closed without
                  ;; depending on lexical captures or touching any provider.
                  blas/dgemm-nt! (fn [& _] (swap! fallback-calls inc) C)]
      (let [plan (ps/dense-contract-blas-plan (dense-form :double :nn))
            leaf (leaf-function :double plan
                                '(raster.linalg.blas/dgemm-nt! A B C m k n 1.0 0.0))]
        (is (:ok plan))
        (is (identical? failure
                        (try (leaf A B C 1 1 1) nil (catch Throwable actual actual))))
        (is (= [9.0] (vec C)))
        (is (zero? @fallback-calls))))))

;; A BLAS-spelled product: canonicalized to par/contract with :abstract-blas-product arithmetic.
(deftm blas-leaf-mm [A :- (Array double) B :- (Array double) m :- Long k :- Long n :- Long] :- (Array double)
  (nn/matmul A B m k n))

;; Exercise the actual typed fallback, rather than a raw macroexpansion whose
;; arithmetic has not passed through Raster's Float lowering.
(deftm blas-leaf-mm-into-double
  [A :- (Array double) B :- (Array double) C :- (Array double)
   m :- Long k :- Long n :- Long] :- (Array double)
  (nn/matmul! A B C m k n))

(deftm blas-leaf-mm-into-float
  [A :- (Array float) B :- (Array float) C :- (Array float)
   m :- Long k :- Long n :- Long] :- (Array float)
  (nn/matmul! A B C m k n))

(deftest missing-gemm-handle-retains-the-generated-loop-without-invoking-native
  (doseq [[elem handle fixture]
          [[:double #'blas/dgemm-mh #'blas-leaf-mm-into-double]
           [:float #'blas/sgemm-mh #'blas-leaf-mm-into-float]]]
    (let [{:keys [stats f]}
          (binding [ps/*dense-contraction-leaf* :blas]
            (with-redefs [blas/gemm-provider-ready? (constantly true)]
              {:stats (:backend-applied-stats (pipeline/show-pipeline fixture :dtype elem))
               :f (pipeline/compile-aot fixture :dtype elem)}))
          array (if (= elem :double) double-array float-array)
          A (array [2]) B (array [3]) C (array [-1])
          preparations (atom 0)
          native-calls (atom 0)
          selected (atom [])
          guard blas/dense-gemm-admissible?]
      (is (= 1 (:blas-contractions stats)) "the typed compiler selected the guarded BLAS leaf")
      (with-redefs-fn
        {#'blas/gemm-provider (delay true)
         handle (delay (swap! preparations inc)
                       (throw (ex-info "GEMM preparation failed" {:dtype elem})))
         #'blas/invoke-native! (fn [& _]
                                (swap! native-calls inc)
                                (throw (ex-info "native invocation must not be reached" {:dtype elem})))
         #'blas/dense-gemm-admissible?
         (fn [& args]
           (let [admitted? (apply guard args)]
             (swap! selected conj admitted?)
             admitted?))}
        (fn []
          (is (identical? C (f A B C 1 1 1)))
          (is (= [6.0] (vec C)))
          (is (= [false] @selected))
          (is (= 1 @preparations))
          (is (zero? @native-calls)))))))

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
