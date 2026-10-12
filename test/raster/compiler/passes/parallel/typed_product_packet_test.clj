(ns raster.compiler.passes.parallel.typed-product-packet-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.backend.jvm.par-simd :as jvm]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]))

(defn host-control [value] value)

(def events (atom []))
(defn record-event [event] (swap! events conj event) event)

(def source
  '(let* [prefix (raster.compiler.passes.parallel.typed-product-packet-test/host-control xs)
          ^doubles sums (clojure.core/double-array 1)
          ^longs counts (clojure.core/long-array 1)
          effect (raster.par/product-reduce!
                  [sums counts]
                  [[sum 0.0 :double] [count (long 0) :long]]
                  [[segment 1]] i n []
                  [(double (aget xs i)) (long 1)]
                  [[left-sum right-sum] [left-count right-count]] []
                  [(+ left-sum right-sum) (unchecked-add left-count right-count)]
                  {:associative? true :commutative? true :order :implementation-defined})]
     sums))

(def options
  {:dtype :float :target-device :ocl:0
   :array-types {'xs :float} :scalar-types {'n :long}})

(deftest product-packet-preserves-local-mixed-storage-and-complete-emission
  (let [result (lower/segop-lower-pass source options)
        scheduled (:form result)
        equation (first (filter #(seq (:operations %)) (:equations scheduled)))
        algorithm (:algorithm equation)
        facts (dialect/facts algorithm)
        emitted (opencl/opencl-pass scheduled :device-id :ocl:0 :dtype :float
                                   :compile-spirv? false :min-elements 0)]
    (is (= 1 (get-in result [:stats :typed-soac-reused])))
    (is (= 'product-reduce (dialect/operation-kind (first (dialect/equations algorithm)))))
    (is (= [:double :long] (mapv #(get-in facts [:values % :dtype]) ['sums 'counts])))
    (is (= ['sums 'counts] (dialect/physical-results facts (first (dialect/equations algorithm)))))
    (is (= [:double :long] (mapv #(get-in scheduled [:values % :dtype]) ['sums 'counts])))
    (is (= scheduled (program/validate! scheduled)))
    (is (seq (:kernels emitted)))
    (is (zero? (get-in emitted [:stats :fallback] 0)))
    (let [host (jvm/simd-pass scheduled :min-elements 0)
          execute (eval (list 'fn '[xs n] (:form host)))]
      (is (= [4.5] (vec (execute (float-array [-1 2 3.5]) 3))))
      (is (= [0.0] (vec (execute (float-array [-1 2 3.5]) 0)))))))

(deftest sequential-wrapper-exposes-product-without-dropping-neighbor-effects
  (let [wrapped (list 'let* ['result
                             (list 'do
                                   '(raster.compiler.passes.parallel.typed-product-packet-test/record-event :prefix)
                                   source
                                   '(raster.compiler.passes.parallel.typed-product-packet-test/record-event :suffix))]
                      'result)
        result (lower/segop-lower-pass wrapped options)
        scheduled (:form result)
        emitted (opencl/opencl-pass scheduled :device-id :ocl:0 :dtype :float
                                   :compile-spirv? false :min-elements 0)
        host (jvm/simd-pass scheduled :min-elements 0)
        execute (eval (list 'fn '[xs n] (:form host)))]
    (is (= 1 (get-in result [:stats :typed-soac-reused])))
    (is (= [:binding 'result]
           (get-in (first (:equations scheduled)) [:provenance :compatibility-source-site])))
    (is (seq (:kernels emitted)))
    (is (zero? (get-in emitted [:stats :fallback] 0)))
    (reset! events [])
    (is (= :suffix (execute (float-array [-1 2 3.5]) 3)))
    (is (= [:prefix :suffix] @events))))

(deftest flattened-origin-does-not-mislabel-later-original-bindings
  (let [source (list 'let* ['result source
                           'later source] 'later)
        result (lower/segop-lower-pass source options)
        origins (mapv #(get-in % [:provenance :compatibility-source-site])
                      (filter #(seq (:operations %)) (get-in result [:form :equations])))]
    (is (= [[:binding 'result] [:binding 'later]] origins))))

(deftest conditional-arm-product-remains-outside-sequential-wrapper-admission
  (let [guarded (list 'let* ['result (list 'if 'condition source nil)] 'result)
        result (lower/segop-lower-pass guarded (assoc-in options [:scalar-types 'condition] :boolean))]
    (is (zero? (get-in result [:stats :typed-soac-reused] 0)))
    (is (empty? (get-in result [:form :equations])))
    (is (thrown? clojure.lang.ExceptionInfo
                 (opencl/opencl-pass (:form result) :device-id :ocl:0 :dtype :float
                                     :compile-spirv? false :min-elements 0)))))

(deftest unsupported-compound-product-extent-remains-a-precise-refusal
  (let [source (walk/postwalk-replace {'n '(clojure.core/alength xs)} source)
        result (lower/segop-lower-pass source options)
        diagnostic (first (get-in result [:stats :segops-declined]))]
    (is (zero? (get-in result [:stats :typed-soac-reused])))
    (is (= :typed-soac-syntax (:reason diagnostic)))
    (is (= :typed-admission (:stage diagnostic)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (opencl/opencl-pass (:form result) :device-id :ocl:0 :dtype :float
                                     :compile-spirv? false :min-elements 0)))))

(deftest physical-storage-authorities-must-agree
  (let [expression (last (second source))
        options (update options :array-types merge {'sums :double 'counts :long})
        packet (#'lower/singleton-packet 'result expression options (atom #{}) 'write-region)
        forged (assoc-in packet [:program :values 'sums] (av/tensor {:dtype :float :shape ['n]}))
        failure (try (#'lower/packet-equations forged [:binding 'result] (atom 0))
                     nil (catch clojure.lang.ExceptionInfo error error))]
    (is (= :parallel-program-value-conflict (:reason (ex-data failure))))
    (is (= 'sums (:id (ex-data failure))))))

(deftest value-table-join-uses-canonical-shape-refinement-only
  (let [unknown (av/tensor {:dtype :double :shape ['(unknown-dimension x)]})
        known (av/tensor {:dtype :double :shape ['n]})
        join (fn [left right] (#'lower/merge-values {'x left} {'x right}))]
    (is (= {'x known} (join unknown known)))
    (is (= {'x known} (join known unknown)))
    (doseq [incompatible [(av/tensor {:dtype :double :shape ['m]})
                          (av/tensor {:dtype :float :shape ['n]})
                          (assoc known :ownership :borrowed)
                          (assoc known :logical-layout {:strides [2]})]]
      (is (thrown? clojure.lang.ExceptionInfo (join known incompatible))))))

(deftest physical-views-do-not-redeclare-a-causal-logical-extent
  (let [definition (av/tensor {:dtype :double :shape ['rows]})
        later-view (av/tensor {:dtype :double :shape ['allocation-length]})
        values {'bias definition}
        equation {:id :later-update}]
    (is (= values (#'lower/merge-physical-views values {'bias later-view} equation)))
    (is (= {'fresh later-view}
           (#'lower/merge-physical-views {} {'fresh later-view} equation)))
    (doseq [conflict [(assoc later-view :dtype :float)
                      (assoc later-view :representation {:kind :quantized :scheme :q4-k})]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (#'lower/merge-physical-views values {'bias conflict} equation))))))
