(ns raster.compiler.integral-reduction-test
  (:require [clojure.test :refer [deftest is]]
            [raster.par]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.passes.parallel.segred-body :as tree]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]))

(defn source [storage operator]
  (let [cast (if (= :int storage) 'clojure.core/int 'clojure.core/long)
        limits (:limits (dtype/info storage))
        neutral (case operator :sum 0 :product 1 :min (:max limits) :max (:min limits))
        combine (case operator
                  :sum (if (= :int storage) 'clojure.core/unchecked-add-int 'clojure.core/unchecked-add)
                  :product (if (= :int storage) 'clojure.core/unchecked-multiply-int 'clojure.core/unchecked-multiply)
                  :min 'clojure.core/min :max 'clojure.core/max)]
    (list 'let*
          ['result (list 'raster.par/reduce 'acc (list cast neutral) 'i 'n
                         (list combine 'acc '(clojure.core/aget a i)))]
          'result)))

(defn emit
  ([storage operator policy] (emit storage operator policy :ze:0))
  ([storage operator policy device]
  (let [options {:target-device device :dtype policy :array-types {'a storage}
                 :scalar-types {'n :long}}
        packet (:form (lower/segop-lower-pass (source storage operator) options))]
    (opencl/opencl-pass packet :device-id device :dtype policy :min-elements 0
                        :array-types {'a storage} :scalar-types {'n :long}))))

(defn modular-reference [storage operator values]
  (let [{:keys [min max]} (:limits (dtype/info storage))
        modulus (.shiftLeft java.math.BigInteger/ONE (* 8 (dtype/bytes-of storage)))
        wrap (fn [value] (long (+' min (mod (-' value min) modulus))))]
    (case operator
      :sum (reduce #(wrap (+' %1 %2)) 0 values)
      :product (reduce #(wrap (*' %1 %2)) 1 values)
      :min (reduce clojure.core/min max values)
      :max (reduce clojure.core/max min values))))

(defn- operations [body]
  (mapcat (fn [op]
            (cons op (concat (operations (:operations op))
                             (operations (:then-operations op))
                             (operations (:else-operations op)))))
          body))

(deftest integral-tree-preserves-storage-and-certified-combine
  (doseq [storage [:int :long] operator [:sum :product :min :max] policy [:float :double]]
    (let [emitted (emit storage operator policy)
          kernels (:kernels emitted)
          expressions (for [kernel kernels
                            op (operations (get-in kernel [:attributes :kernel-body :operations]))
                            :when (and (= "ScalarCompute" (some-> op class .getSimpleName))
                                       (= storage (get-in op [:expression :result-type])))]
                        (:expression op))]
      (is (= 2 (count kernels)))
      (is (some #{(dtype/jvm-array-constructor storage)} (flatten (:form emitted))))
      (doseq [kernel kernels]
        (is (= #{storage} (into #{} (map :dtype) (remove #(= :scalar (:kind %)) (:abi kernel)))))
        (is (= :exact (get-in kernel [:provenance :scheduled-operation :numerics :mode]))))
      (if (contains? #{:sum :product} operator)
        (do (is (seq expressions))
            (is (every? #(= :wrap (get-in % [:options :overflow])) expressions)))
        (is (not (re-find #"\b(isnan|fmin|fmax)\(" (apply str (map :source kernels)))))))))

(deftest modular-oracle-agrees-with-source-at-signed-limits
  (doseq [storage [:int :long] operator [:sum :product :min :max]
          values [[] [1] [(:max (:limits (dtype/info storage))) 1]
                  [(:min (:limits (dtype/info storage))) -1]]]
    (let [array ((resolve (dtype/jvm-array-constructor storage)) values)
          function (eval (list 'fn ['a 'n] (source storage operator)))]
      (is (= (modular-reference storage operator values) (function array (count values)))))))

(deftest contradictory-storage-and-overflow-contracts-decline
  (doseq [storage [:int :long]]
    (let [options {:target-device :ze:0 :dtype :double :array-types {'a storage}
                   :scalar-types {'n :long}}
          packet (:form (lower/segop-lower-pass (source storage :sum) options))
          operation (first (get-in packet [:equations 0 :operations]))]
      (is (thrown? clojure.lang.ExceptionInfo
                   (tree/schedule operation (assoc options :array-types {'a :double}))))
      (doseq [overflow [nil :trap]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (tree/schedule (assoc-in operation [:reduction :algebra :overflow] overflow)
                                    options)))))))

(deftest result-storage-uses-canonical-dtype-facets
  (doseq [storage [:float :double :int :long :byte :half :i64 :f32]]
    (let [constructor (dtype/jvm-array-constructor storage)
          array ((resolve constructor) 1)]
      (is (= "clojure.core" (namespace constructor)))
      (is (= (dtype/canon storage) (dtype/dtype-for-jvm-array array))))))

(deftest packed-half-storage-does-not-become-a-host-numeric-scalar
  (is (= 'clojure.core/short-array (dtype/jvm-array-constructor :half)))
  (doseq [storage [:half :f16 :byte]]
    (is (nil? (#'opencl/host-scalar-constructor storage))))
  (doseq [storage [:int :i64 :float :f64]]
    (is (= (dtype/jvm-array-constructor storage) (#'opencl/host-scalar-constructor storage)))))

(deftest integer-element-arithmetic-retains-source-overflow
  (doseq [storage [:int :long]
          [operator expected] [[(if (= :int storage) 'clojure.core/unchecked-multiply-int
                                   'clojure.core/unchecked-multiply) :wrap]
                               ['clojure.core/* :trap]]
          :when (or (= :wrap expected) (= :long storage))]
    (let [expression (with-meta (list operator '(clojure.core/aget a i)
                                      (if (= :int storage) (int 2) (long 2)))
                       {:raster.type/tag (dtype/scalar-tag-for-dtype storage)})
          lowered (tree/lower-element-operations
                   expression {:index 'i :coordinate 'element-index :dtype storage
                               :arrays #{'a} :array-types {'a storage} :scalars #{}})
          computations (filter #(= "ScalarCompute" (some-> % class .getSimpleName))
                               (:operations lowered))]
      (is (seq computations))
      (is (= expected (get-in (last computations) [:expression :options :overflow]))))))
