(ns raster.compiler.passes.parallel.typed-scalar-packet-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]))

(def source
  '(let* [prefix (identity xs)
          sum (raster.par/reduce acc (float 0.0) i (clojure.core/alength xs)
                                (+ acc (* (float (aget xs i)) (float (aget xs i)))))
          norm (Math/sqrt (double sum))]
     norm))

(def options {:dtype :float :target-device :ocl:0 :array-types {'xs :float}})

(def causal-source
  '(let* [^{:raster.type/tag long} n (clojure.core/long (clojure.core/alength xs))
          sum (raster.par/reduce acc (float 0.0) i n
                                (+ acc (* (float (aget xs i)) (float (aget xs i)))))
          norm (Math/sqrt (double sum))]
     norm))

(deftest compatibility-composition-retains-certified-host-length-equations
  (let [result (lower/segop-lower-pass causal-source
                                       (assoc options :values {'xs (av/tensor {:dtype :float :shape [32]})}))
        scheduled (:form result)
        {:keys [body graph]} (graph/make-for-equation scheduled (last (:equations scheduled)))]
    (is (= 1 (get-in result [:stats :segops-lowered])))
    (is (= 1 (get-in result [:stats :typed-soac-reused])))
    (is (= 1 (get-in result [:stats :typed-scalar-equations])))
    (is (= [32] (get-in scheduled [:values 'xs :shape])))
    (is (= ['n] (:results (first (:equations body)))))
    (is (true? (get-in body [:equations 0 :attributes :host-only])))
    (is (= {'n '(extent xs)} (graph/derived-scalar-expressions (:values body) (:equations body))))
    (is (= 'n (:elements (first (:inputs graph)))))))

(deftest unsupported-host-bindings-remain-outside-the-certified-packet
  (doseq [expression ['(clojure.core/long (do (swap! counter inc) (clojure.core/alength xs)))
                      '(clojure.core/long (clojure.core/aget xs 0))
                      '(clojure.core/long (clojure.core/double (clojure.core/alength xs)))
                      '(clojure.core/long (clojure.core/int (clojure.core/long (clojure.core/alength xs))))]]
    (let [source (list 'let* [(with-meta 'n {:raster.type/tag 'long}) expression] 'n)
          result (lower/segop-lower-pass source options)]
      (is (zero? (get-in result [:stats :typed-scalar-equations])))
      (is (= expression (second (second (:source (:form result)))))))))

(deftest retaining-a-pure-tail-does-not-remove-or-replay-an-earlier-effect
  (let [source '(let* [^{:raster.type/tag long} n
                       (do (swap! counter inc) (clojure.core/long (clojure.core/alength xs)))] n)
        normalized (:source (:form (lower/segop-lower-pass source options)))
        counter (atom 0)
        execute (eval (list 'fn '[xs counter] normalized))]
    (is (= 3 (execute (float-array 3) counter)))
    (is (= 1 @counter))))

(deftest host-only-proof-is-opt-in-and-does-not-count-as-parallel-work
  (let [source '(let* [^{:raster.type/tag long} n (clojure.core/long (clojure.core/alength xs))] n)
        default (raster.compiler.passes.parallel.typed-soac-route/attempt
                 source :float {'xs :float})
        result (lower/segop-lower-pass source options)]
    (is (nil? (:program default)))
    (is (= 1 (get-in result [:stats :typed-scalar-equations])))
    (is (zero? (get-in result [:stats :segops-lowered])))
    (is (zero? (get-in result [:stats :typed-soac-reused])))
    (is (= 3 ((eval (list 'fn '[xs] (:source (:form result)))) (float-array 3))))))

(def nested-source
  (let [leaf (get (into {} (map vec (partition 2 (second source)))) 'sum)]
    (list 'let* ['prefix '(identity xs)
                 'norm (list 'Math/sqrt (list 'double leaf))] 'norm)))

(deftest imported-scalar-packet-keeps-one-logical-and-scheduled-result-identity
  (let [result (lower/segop-lower-pass source options)
        scheduled (:form result)
        equation (last (:equations scheduled))
        kernel-graph (:graph (graph/make-for-equation scheduled equation))
        emitted (opencl/opencl-pass scheduled :device-id :ocl:0 :dtype :float
                                   :compile-spirv? false :min-elements 0)]
    (is (= 1 (get-in result [:stats :typed-soac-reused])))
    (is (= 1 (get-in result [:stats :typed-scalar-equations])))
    (is (= ['sum] (:results equation)))
    (is (= ['sum] (mapv :id (:outputs kernel-graph))))
    (is (= scheduled (program/validate! scheduled)))
    (is (seq (:kernels emitted)))
    (is (zero? (get-in emitted [:stats :fallback] 0)))))

(deftest unary-scalar-consumer-retains-the-complete-reduction-packet
  (let [result (lower/segop-lower-pass nested-source options)
        scheduled (:form result)
        equation (last (:equations scheduled))
        emitted (opencl/opencl-pass scheduled :device-id :ocl:0 :dtype :float
                                   :compile-spirv? false :min-elements 0)]
    (is (= 1 (get-in result [:stats :typed-soac-reused])))
    (is (= 1 (get-in result [:stats :typed-scalar-equations])))
    (is (= [:binding 'norm] (get-in equation [:provenance :compatibility-source-site])))
    (is (= scheduled (program/validate! scheduled)))
    (is (seq (:dispatches emitted)))
    (is (zero? (get-in emitted [:stats :fallback] 0)))))

(deftest input-shadowing-cannot-import-a-conflicting-typed-result
  (let [source '(let* [xs (raster.par/reduce acc 0.0 i n (+ acc (aget xs i)))] xs)
        options {:dtype :double :target-device :ocl:0 :array-types {'xs :double}
                 :scalar-types {'n :long}}
        result (try (lower/segop-lower-pass source options)
                    (catch clojure.lang.ExceptionInfo error {:refused error}))]
    (is (instance? clojure.lang.ExceptionInfo (:refused result)))
    (is (= :raster/bug (:reason (ex-data (:refused result)))))
    (is (= :typed-reduction-packet (:stage (ex-data (:refused result)))))))

(deftest legacy-equations-retain-their-separate-binding-identity
  (let [source '(let* [result (raster.par/map! out i n (aget xs i))] result)
        options {:dtype :double :array-types {'xs :double 'out :double} :scalar-types {'n :long}}]
    (with-redefs [raster.compiler.passes.parallel.typed-soac-route/attempt
                  (fn [& _] {:declined {:reason :test-legacy-boundary}})]
      (let [scheduled (:form (lower/segop-lower-pass source options))]
        (is (= [[:binding 'result]] (mapcat :results (:equations scheduled))))))))

(deftest repeated-source-binders-remain-distinct-before-packet-import
  (let [source '(let* [value (raster.par/reduce acc (long 0) i n
                                              (unchecked-add acc (long i)))
                       value (raster.par/reduce acc 0.0 i n (+ acc (double value)))]
                 value)
        scheduled (:form (lower/segop-lower-pass source
                                                {:dtype :double :scalar-types {'n :long}}))
        [producer consumer] (:equations scheduled)
        original (eval (list 'fn '[n] source))
        normalized (eval (list 'fn '[n] (:source scheduled)))]
    (is (not= (:results producer) (:results consumer)))
    (is (some (set (:results producer)) (:operands consumer)))
    (is (= scheduled (program/validate! scheduled)))
    (doseq [n [0 1 5]] (is (= (original n) (normalized n))))))
