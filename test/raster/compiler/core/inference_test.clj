(ns raster.compiler.core.inference-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.inference :as inf]))

;; Helper to build a minimal type-env for infer-binding-tag
(defn- type-env
  ([] {})
  ([flat-env]
   (reduce-kv (fn [m s t] (assoc m s {:tag t})) {} flat-env)))

(defn- opts [] {:use-tc? false :source-ns *ns*})

;; ================================================================
;; literal-tag
;; ================================================================

(deftest literal-tag-test
  (testing "doubles"
    (is (= 'double (inf/literal-tag 3.14)))
    (is (= 'double (inf/literal-tag 0.0)))
    (is (= 'double (inf/literal-tag Double/NaN))))
  (testing "longs"
    (is (= 'long (inf/literal-tag 42)))
    (is (= 'long (inf/literal-tag 0)))
    (is (= 'long (inf/literal-tag -1))))
  (testing "non-numeric returns class or nil"
    ;; literal-tag returns the Class for string literals
    (is (some? (inf/literal-tag "hello")))
    (is (nil? (inf/literal-tag nil)))
    (is (nil? (inf/literal-tag 'x)))))

;; ================================================================
;; hint-tag
;; ================================================================

(deftest hint-tag-test
  (testing "type-hinted symbols"
    (is (= 'double (inf/hint-tag (with-meta 'x {:tag 'double}))))
    (is (= 'doubles (inf/hint-tag (with-meta 'x {:tag 'doubles})))))
  (testing "unhinted"
    (is (nil? (inf/hint-tag 'x)))
    (is (nil? (inf/hint-tag 42)))))

;; ================================================================
;; infer-binding-tag
;; ================================================================

(deftest infer-binding-tag-literal-test
  (testing "literal init"
    (is (= 'double (inf/infer-binding-tag 'x 3.14 nil (type-env) (opts))))
    (is (= 'long (inf/infer-binding-tag 'x 42 nil (type-env) (opts))))))

(deftest infer-binding-tag-env-lookup-test
  (testing "symbol init with known type"
    (is (= 'double (inf/infer-binding-tag 'y 'x nil (type-env {'x 'double}) (opts))))
    (is (= 'doubles (inf/infer-binding-tag 'y 'x nil (type-env {'x 'doubles}) (opts))))))

(deftest infer-binding-tag-array-alloc-test
  (testing "array allocation"
    (is (= 'doubles (inf/infer-binding-tag 'buf '(double-array 16) nil (type-env) (opts))))
    (is (= 'floats (inf/infer-binding-tag 'buf '(float-array 16) nil (type-env) (opts))))
    (is (= 'longs (inf/infer-binding-tag 'buf '(long-array 16) nil (type-env) (opts))))
    (is (= 'ints (inf/infer-binding-tag 'buf '(int-array 16) nil (type-env) (opts))))))

;; aclone type preservation is handled by TC at invocation time

(deftest infer-binding-tag-hinted-sym-test
  (testing "type-hinted symbol"
    (let [sym (with-meta 'z {:tag 'double})]
      (is (= 'double (inf/infer-binding-tag sym 'x nil (type-env) (opts)))))))

(deftest infer-binding-tag-cast-test
  (testing "primitive cast in init"
    (is (= 'double (inf/infer-binding-tag 'x '(double y) nil (type-env) (opts))))
    (is (= 'long (inf/infer-binding-tag 'x '(long y) nil (type-env) (opts))))))

;; ================================================================
;; infer-rewritten-tag
;; ================================================================

(deftest infer-rewritten-tag-cast-test
  (testing "primitive casts"
    (is (= 'double (inf/infer-rewritten-tag '(double x) '(double x) {})))
    (is (= 'long (inf/infer-rewritten-tag '(long x) '(long x) {})))
    (is (= 'float (inf/infer-rewritten-tag '(float x) '(float x) {})))
    (is (= 'int (inf/infer-rewritten-tag '(int x) '(int x) {})))))

;; ================================================================
;; try-resolve-call
;; ================================================================

(deftest try-resolve-call-arithmetic-test
  (testing "resolves raster.numeric/+ for doubles"
    (let [result (inf/try-resolve-call 'raster.numeric/+ '(x y) (type-env {'x 'double 'y 'double}))]
      (is (some? result))
      (is (contains? result :mangled-sym))
      (is (contains? result :typed-iface))
      (is (= '[double double] (:tags result)))))
  (testing "returns nil for unknown fn"
    (is (nil? (inf/try-resolve-call 'no.such/fn '(x y) (type-env {'x 'double 'y 'double}))))))

;; ================================================================
;; generic-fn?
;; ================================================================

(deftest generic-fn-test
  (testing "raster.numeric ops are generic"
    (is (true? (inf/generic-fn? 'raster.numeric/+)))
    (is (true? (inf/generic-fn? 'raster.numeric/*))))
  (testing "non-deftm syms are not generic"
    (is (not (inf/generic-fn? 'clojure.core/+)))
    (is (not (inf/generic-fn? 'no.such/fn)))))

;; ================================================================
;; trace-inference
;; ================================================================

(deftest trace-inference-test
  (testing "trace-inference prints provenance when enabled"
    (let [output (with-out-str
                   (binding [inf/*trace-inference* true]
                     (inf/infer-binding-tag 'x 3.14 nil (type-env) (opts))))]
      (is (.contains output "INFER"))
      (is (.contains output "literal")))
    (let [output (with-out-str
                   (binding [inf/*trace-inference* true]
                     (inf/infer-binding-tag 'x '(double-array 16) nil (type-env) (opts))))]
      (is (.contains output "array-alloc")))
    (let [output (with-out-str
                   (binding [inf/*trace-inference* true]
                     (inf/infer-binding-tag 'y 'x nil (type-env {'x 'double}) (opts))))]
      (is (.contains output "env"))))
  (testing "no output when trace disabled"
    (let [output (with-out-str
                   (inf/infer-binding-tag 'x 3.14 nil (type-env) (opts)))]
      (is (= "" output)))))
