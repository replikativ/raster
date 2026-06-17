(ns raster.core-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm ftm defvalue methods-of print-methods]]))

;; ================================================================
;; Basic single dispatch
;; ================================================================

(deftm single-long [^long x] (* x x))
(deftm single-long [^String x] (count x))
(deftm single-long [^double x] (+ x 0.5))

(deftest single-dispatch-test
  (testing "long dispatch"
    (is (= 1764 (single-long 42)))
    (is (= 0 (single-long 0))))
  (testing "String dispatch"
    (is (= 5 (single-long "hello")))
    (is (= 0 (single-long ""))))
  (testing "double dispatch"
    (is (= 3.5 (single-long 3.0)))))

;; ================================================================
;; Specificity (subtype ordering)
;; ================================================================

(deftm specific [^Object x] :object)
(deftm specific [^String x] :string)
(deftm specific [^Number x] :number)
(deftm specific [^Long x] :long)

(deftest specificity-test
  (testing "most specific type wins"
    (is (= :long (specific 42)))
    (is (= :string (specific "hi")))
    (is (= :number (specific 3.14)))
    (is (= :object (specific :keyword)))
    (is (= :object (specific [1 2 3])))))

;; ================================================================
;; Metadata-form type annotations: ^{:- T} (Rich Hickey / TypedClojure style)
;; ================================================================

(deftm meta-ann [^{:- Long} x] :long)
(deftm meta-ann [^{:- String} x] :string)
;; TypedClojure's fully-qualified metadata key is also accepted
(deftm meta-ann [^{:typed.clojure/type Double} x] :double)
;; inline and metadata forms may be mixed in one arglist
(deftm meta-mixed [^{:- Long} x y :- String] [x y])
;; return type via metadata: ^{:- Ret} goes BEFORE the arg vector,
;; mirroring Clojure's own ^long [x] return-hint convention.
(deftm meta-ret ^{:- Long} [x :- Long] (* x x))

(deftest metadata-annotation-test
  (testing "^{:- T} dispatches identically to inline x :- T"
    (is (= :long   (meta-ann 42)))
    (is (= :string (meta-ann "hi"))))
  (testing "^{:typed.clojure/type T} is recognized"
    (is (= :double (meta-ann 3.14))))
  (testing "metadata and inline forms mix freely"
    (is (= [7 "ok"] (meta-mixed 7 "ok"))))
  (testing "ftm accepts the metadata form too"
    (is (= 6.25 ((ftm [^{:- Double} y] :- Double (* y y)) 2.5))))
  (testing "return type via ^{:- Ret} before the arg vector"
    (is (= 49 (meta-ret 7)))
    ;; the metadata return type drives the same ::return-tag as inline :- Long
    (let [v (->> (ns-publics 'raster.core-test) keys
                 (filter #(re-find #"meta-ret_m_" (str %)))
                 (map #(ns-resolve 'raster.core-test %)) first)]
      (is (= 'long (:raster.core/return-tag (meta v))))))
  (testing "ftm return type via ^{:- Ret} before the arg vector"
    (is (= 9.0 ((ftm ^{:- Double} [y :- Double] (* y y)) 3.0)))))

;; ================================================================
;; Multi-arity dispatch
;; ================================================================

(deftm multi-arity [^long x] (* x 10))
(deftm multi-arity [^long x ^long y] (+ x y))
(deftm multi-arity [^long x ^long y ^long z] (+ x y z))
(deftm multi-arity [^String a ^String b] (str a b))

(deftest multi-arity-test
  (testing "arity 1"
    (is (= 50 (multi-arity 5))))
  (testing "arity 2 long"
    (is (= 7 (multi-arity 3 4))))
  (testing "arity 2 string"
    (is (= "ab" (multi-arity "a" "b"))))
  (testing "arity 3"
    (is (= 6 (multi-arity 1 2 3)))))

;; ================================================================
;; Error: no matching method
;; ================================================================

(deftm strict-typed [^long x] x)

(deftest no-match-test
  (testing "throws on type mismatch"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"No matching method"
                          (strict-typed "wrong"))))
  (testing "throws on arity mismatch"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"No matching method"
                          (strict-typed 1 2)))))

;; ================================================================
;; Open extension (adding methods later)
;; ================================================================

(deftm extendable [^long x] :original)

(deftm extendable [^String x] :extended)

(deftest open-extension-test
  (testing "original method"
    (is (= :original (extendable 1))))
  (testing "extended with new type"
    (is (= :extended (extendable "hi")))
    (is (= :original (extendable 1)))))

;; ================================================================
;; defvalue
;; ================================================================

(defvalue Point2D [x :- Double, y :- Double])

(deftm describe [^Point2D p] (str "(" (:x p) "," (:y p) ")"))
(deftm describe [^String s] s)

(def ^:private valhalla?
  "True if JDK supports Valhalla value classes (JDK 27+ with --enable-preview)."
  (try
    (Class/forName "java.lang.classfile.ClassFile")
    (>= (.feature (Runtime/version)) 27)
    (catch ClassNotFoundException _ false)))

(deftest defvalue-test
  (testing "creates record"
    (let [p (->Point2D 1.0 2.0)]
      (is (= 1.0 (:x p)))
      (is (= 2.0 (:y p)))))
  (testing "dispatch on value type"
    (is (= "(1.0,2.0)" (describe (->Point2D 1.0 2.0))))
    (is (= "hi" (describe "hi"))))
  (when valhalla?
    (testing "value class: no ACC_IDENTITY (identity-free)"
      (let [flags (.accessFlags (class (->Point2D 0.0 0.0)))
            ;; Resolve AccessFlag/IDENTITY at runtime to avoid compile error on JDK < 27
            identity-flag (eval '(java.lang.reflect.AccessFlag/IDENTITY))]
        (is (not (.contains flags identity-flag))
            "Value class should not have ACC_IDENTITY")
        (is (.contains flags (java.lang.reflect.AccessFlag/FINAL))
            "Value class should be ACC_FINAL")))
    (testing "value class: identity-free equality"
      (let [a (->Point2D 1.0 2.0)
            b (->Point2D 1.0 2.0)]
        (is (= a b) "Same-valued value objects should be equal")))
    (testing "value class: keyword access via ILookup"
      (let [p (->Point2D 3.0 4.0)]
        (is (= 3.0 (:x p)))
        (is (= 4.0 (:y p)))
        (is (nil? (:missing p)))
        (is (= :default (:missing p :default)))))
    (testing "value class: array helpers"
      (let [cls (class (->Point2D 0.0 0.0))
            make-arr (.getMethod cls "makeArray" (into-array Class [Integer/TYPE]))
            arr (.invoke make-arr nil (object-array [(int 3)]))]
        (is (= 3 (alength ^objects arr)))
        ;; arraySet and arrayGet
        (let [set-m (.getMethod cls "arraySet"
                                (into-array Class [(.getClass arr) Integer/TYPE cls]))
              get-m (.getMethod cls "arrayGet"
                                (into-array Class [(.getClass arr) Integer/TYPE]))]
          (.invoke set-m nil (object-array [arr (int 0) (->Point2D 10.0 20.0)]))
          (.invoke set-m nil (object-array [arr (int 1) (->Point2D 30.0 40.0)]))
          (let [p0 (.invoke get-m nil (object-array [arr (int 0)]))]
            (is (= 10.0 (:x p0)))
            (is (= 20.0 (:y p0))))
          (let [p1 (.invoke get-m nil (object-array [arr (int 1)]))]
            (is (= 30.0 (:x p1)))))))
    (testing "value class: toString"
      (let [p (->Point2D 1.5 2.5)]
        (is (= "Point2D(1.5, 2.5)" (.toString p)))))))

;; ================================================================
;; Introspection
;; ================================================================

(deftest introspection-test
  (testing "methods-of returns dispatch table"
    (let [table (methods-of #'single-long)]
      (is (map? table))
      (is (contains? table 1))
      (is (= 3 (count (get table 1))))))
  (testing "print-methods runs without error"
    (is (string? (with-out-str (print-methods #'single-long))))))

;; ================================================================
;; Zero-arity
;; ================================================================

(deftm zero-arity [] :zero)

(deftest zero-arity-test
  (is (= :zero (zero-arity))))

;; ================================================================
;; Primitive arrays
;; ================================================================

(deftm arr-fn [^longs xs] (aget xs 0))
(deftm arr-fn [^doubles xs] (aget xs 0))

(deftest primitive-array-test
  (testing "long array dispatch"
    (is (= 42 (arr-fn (long-array [42 43])))))
  (testing "double array dispatch"
    (is (= 3.14 (arr-fn (double-array [3.14 2.71]))))))

;; ================================================================
;; 5+ arg deftm with all primitive doubles
;; ================================================================

(deftm five-arg-prim
  [a :- Double, b :- Double, c :- Double, d :- Double, e :- Double]
  :- Double
  (+ a b c d e))

(deftest five-arg-primitive-test
  (testing "5-arg deftm with all doubles returns correct result"
    (is (= 15.0 (five-arg-prim 1.0 2.0 3.0 4.0 5.0))))
  (testing "5-arg deftm generates typed-impl"
    (let [table (methods-of #'five-arg-prim)
          entry (first (get table 5))]
      (is (some? (:typed-impl entry))
          "MethodEntry should have a typed-impl")
      (is (some? (:typed-iface entry))
          "MethodEntry should have a typed-iface"))))

;; ================================================================
;; ftm — typed anonymous function
;; ================================================================

(def add-ftm
  "ftm that adds two doubles"
  (ftm [x :- Double, y :- Double] :- Double (+ x y)))

(def array-scale-ftm
  "ftm that scales an array by a scalar (in-place)"
  (ftm [du :- (Array double), u :- (Array double), t :- Double]
       (let [n (alength u)]
         (dotimes [i n]
           (aset du i (* 2.0 (aget u i)))))))

(deftest ftm-basic-test
  (testing "ftm via IFn invoke"
    (is (= 7.0 (add-ftm 3.0 4.0))))
  (testing "ftm via .invk (typed path)"
    (is (= 7.0 (.invk add-ftm 3.0 4.0))))
  (testing "ftm with array params via IFn"
    (let [du (double-array [0.0 0.0])
          u  (double-array [3.0 5.0])]
      (array-scale-ftm du u 1.0)
      (is (= 6.0 (aget du 0)))
      (is (= 10.0 (aget du 1)))))
  (testing "ftm with array params via .invk"
    (let [du (double-array [0.0 0.0])
          u  (double-array [3.0 5.0])]
      (.invk array-scale-ftm du u 1.0)
      (is (= 6.0 (aget du 0)))
      (is (= 10.0 (aget du 1))))))

;; ================================================================
;; (Fn ...) annotation + devirtualization
;; ================================================================

(deftm apply-fn
  [f :- (Fn [Double Double] Double), x :- Double, y :- Double]
  :- Double
  (f x y))

(deftest fn-annotation-test
  (testing "(Fn ...) annotated param works with ftm via .invk"
    (is (= 7.0 (apply-fn add-ftm 3.0 4.0))))
  (testing "(Fn ...) annotated param works with another ftm"
    (is (= 12.0 (apply-fn (ftm [a :- Double, b :- Double] :- Double (* a b)) 3.0 4.0))))
  (testing "(Fn ...) annotated param rejects plain fn with ClassCastException"
    (is (thrown? ClassCastException
                 (apply-fn (fn [x y] (+ (double x) (double y))) 3.0 4.0)))))

;; ================================================================
;; deftm source metadata for specialization
;; ================================================================

(deftm source-meta-add
  [u :- (Array double), v :- (Array double)]
  :- (Array double)
  (let [n (alength u)
        out (double-array n)]
    (dotimes [i n]
      (aset out i (+ (aget u i) (aget v i))))
    out))

(deftest deftm-source-metadata-test
  (let [impl-var #'source-meta-add_m_doubles_doubles
        m (meta impl-var)]
    (testing "deftm works normally"
      (let [u (double-array [1.0 2.0 3.0])
            v (double-array [4.0 5.0 6.0])
            r (source-meta-add u v)]
        (is (= 5.0 (aget r 0)))
        (is (= 7.0 (aget r 1)))
        (is (= 9.0 (aget r 2)))))
    (testing "deftm stores source metadata used by specialization"
      (is (= '[u v] (:raster.core/deftm-params m)))
      (is (= '[doubles doubles] (:raster.core/deftm-tags m)))
      (is (vector? (:raster.core/deftm-source-body m)))
      (is (seq (:raster.core/deftm-source-body m))))))
