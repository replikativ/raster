(ns raster.compiler.passes.parallel.typed-write-packet-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.backend.jvm.par-simd :as jvm]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.par]))

(defn host-control [value] value)

(def options
  {:dtype :float :target-device :ocl:0
   :array-types {'out :float 'other :float 'vals :float 'keys :int}
   :scalar-types {'n :long 'stride :long}})

(defn source [expression]
  (list 'let* ['prefix '(raster.compiler.passes.parallel.typed-write-packet-test/host-control vals)
              'step expression] 'step))

(deftest isolated-write-packets-retain-canonical-conflicts-and-prefixes
  (doseq [expression
          ['(raster.par/scatter! out vals keys n stride)
           '(raster.par/map-void! i (* n stride)
              (aset out (+ (* (aget keys (quot i stride)) stride) (rem i stride))
                    (+ (aget out (+ (* (aget keys (quot i stride)) stride) (rem i stride)))
                       (aget vals i))))]]
    (let [scheduled (:form (lower/segop-lower-pass (source expression) options))
          equations (filter #(seq (:operations %)) (:equations scheduled))
          equation (first equations)
          algorithm (:algorithm equation)
          conflict (-> algorithm dialect/equations first dialect/operation-parts
                       :attributes :conflict)
          host (jvm/simd-pass scheduled :min-elements 0)
          execute (eval (list 'fn '[out other vals keys n stride] (:form host)))
          out (float-array [10 20 30 40])
          emitted (opencl/opencl-pass scheduled :device-id :ocl:0 :dtype :float
                                     :compile-spirv? false :min-elements 0)]
      (is (= 1 (count equations)))
      (is (= ['prefix '(raster.compiler.passes.parallel.typed-write-packet-test/host-control vals)]
             (vec (take 2 (second (:source scheduled))))))
      (is (= 'scatter (dialect/operation-kind (first (dialect/equations algorithm)))))
      (is (= (dialect/reducing-scatter-conflict '+ :float) conflict))
      (is (false? (dialect/reducing-scatter-conflict? (assoc conflict :dtype :double))))
      (is (false? (dialect/reducing-scatter-conflict? (assoc conflict :identity 1.0))))
      (is (= :reduce (:write-conflict (first (:operations equation)))))
      (let [returned (execute out (float-array 4) (float-array [1 2 3 4 5 6])
                              (int-array [1 0 1]) 3 2)]
        (if (= 'raster.par/scatter! (first expression))
          (is (identical? out returned))
          (is (nil? returned))))
      (is (= [13.0 24.0 36.0 48.0] (vec out)))
      (is (= 1 (get-in emitted [:stats :segop-reused])))
      (is (= :reducing-scatter (get-in emitted [:kernels 0 :effects :kind]))))))

(deftest mixed-destinations-retain-distinct-write-contracts
  (let [expression '(raster.par/map-void! i n
                     (do (aset out i (float 3))
                         (raster.par/atomic-add! other 0 (float 1))))
        scheduled (:form (lower/segop-lower-pass (source expression) options))
        equation (first (filter #(seq (:operations %)) (:equations scheduled)))
        algorithm (:algorithm equation)]
    (is (= 'effect-map (dialect/operation-kind (first (dialect/equations algorithm)))))
    (is (= #{'out 'other} (:outputs (first (:operations equation)))))
    (is (= 2 (count (:results equation))))
    (is (every? #(= :float (get-in scheduled [:values % :dtype])) (:results equation)))
    (is (= #{'out 'other}
           (set (dialect/physical-results (dialect/facts algorithm)
                                         (first (dialect/equations algorithm))))))
    (is (= 1 (get-in (lower/segop-lower-pass (source expression) options) [:stats :typed-soac-reused])))))

(deftest unique-direct-stores-retain-their-canonical-map-and-destination
  (let [result (lower/segop-lower-pass
                (source '(raster.par/map-void! i n (aset out i (float 3)))) options)
        equation (first (filter #(seq (:operations %)) (:equations (:form result))))]
    (is (= 1 (get-in result [:stats :typed-soac-reused])))
    (is (= 'map (dialect/operation-kind (first (dialect/equations (:algorithm equation))))))
    (is (= #{'out} (:outputs (first (:operations equation)))))))

(deftest unknown-effects-do-not-gain-a-write-packet
  (let [result (lower/segop-lower-pass
                (source '(raster.par/map-void! i n (unknown-effect! out i))) options)]
    (is (zero? (get-in result [:stats :typed-soac-reused])))
    (is (seq (get-in result [:stats :segops-declined])))))

(deftest double-scatter-does-not-gain-float-atomic-semantics
  (let [result (lower/segop-lower-pass
                (source '(raster.par/scatter! out vals keys n stride))
                (assoc options :dtype :double :array-types {'out :double 'vals :double 'keys :int}))]
    (is (zero? (get-in result [:stats :typed-soac-reused])))
    (is (some #(= :typed-admission (:stage %)) (get-in result [:stats :segops-declined])))))

(deftest repeated-effect-packets-freshen-logical-results-not-physical-storage
  (let [region '(raster.par/map-void! i n
                 (do (aset out i (float 3)) (raster.par/atomic-add! other 0 (float 1))))
        source (list 'let* ['prefix '(raster.compiler.passes.parallel.typed-write-packet-test/host-control vals)
                            'first-effect region 'second-effect region] 'second-effect)
        lowered (lower/segop-lower-pass source options)
        scheduled (:form lowered)
        equations (filterv #(seq (:operations %)) (:equations scheduled))
        results (mapcat :results equations)]
    (is (= 2 (get-in lowered [:stats :typed-soac-reused])))
    (is (= 2 (count equations)))
    (is (= 4 (count results) (count (set results))))
    (is (every? #(= :float (get-in scheduled [:values % :dtype])) results))
    (doseq [equation equations]
      (let [algorithm (:algorithm equation)]
        (is (= #{'out 'other}
               (set (dialect/physical-results (dialect/facts algorithm)
                                             (first (dialect/equations algorithm))))))))))
