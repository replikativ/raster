(ns raster.compiler.passes.parallel.typed-gather-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def ^:private source
  '(let* [step (raster.par/gather out src idx n)]
         step))

(def ^:private strided-source
  '(let* [step (raster.par/gather out src idx n stride)]
         step))

(def ^:private array-types
  {'out :float 'src :float 'idx :int})

(def ^:private scalar-types
  {'n :long 'stride :long})

(defn- scheduled-program []
  (let [{:keys [program]} (route/attempt source :float array-types
                                         {:scalar-types scalar-types})]
    (:form (segop-lower/segop-lower-pass
            program {:dtype :float :target-device :ocl:0
                     :array-types array-types :scalar-types scalar-types}))))

(deftest flat-gather-is-an-ordinary-typed-map
  (let [{:keys [program stats]} (route/attempt source :float array-types
                                               {:scalar-types scalar-types})
        equation (first (dialect/equations (get-in program [:equations 0 :algorithm])))
        operation (dialect/operation-parts equation)
        {:keys [parameters body-results]} (dialect/lambda-parts (:lambda operation))]
    (is (= :typed-soac (:dialect program)))
    (is (= :analyzed-source (:front-end stats)))
    (is (= 'map (:kind operation)))
    (is (= '[idx] (:arrays operation))
        "the index map is pointwise")
    (is (= '[src] (:captures operation))
        "the indirectly read source remains a stable tensor capture")
    (is (= '(clojure.core/aget %capture0 %element0) (first body-results)))
    (is (= 2 (count parameters)))
    (is (= [{:destination 'out :access :write :host-return :buffer}]
           (get-in program [:equations 0 :attributes :result-storage])))
    (is (not-any? #{'raster.par/gather}
                  (tree-seq coll? seq (:source program))))))

(deftest typed-gather-has-target-specific-schedules-without-semantic-reparsing
  (let [scheduled (scheduled-program)
        operation (-> scheduled :equations first :operations first)
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        gpu (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                     :dtype :float :min-elements 0
                                     :array-types array-types
                                     :scalar-types scalar-types)
        kernel (first (:kernels gpu))
        kernel-source (:source kernel)]
    (is (instance? raster.compiler.ir.segop.SegMap operation))
    (is (= :typed-soac (:algorithm-dialect operation)))
    (testing "JVM recognizes hardware vgather from the scheduled scalar region"
      (is (= 1 (get-in jvm [:stats :segop-reused])))
      (is (nil? (get-in jvm [:stats :segop-relowered])))
      (is (str/includes? (pr-str (:form jvm)) "FloatVector/fromArray")))
    (testing "portable GPU emission consumes the same SegMap and keeps C names hygienic"
      (is (= 1 (get-in gpu [:stats :segop-reused])))
      (is (zero? (get-in gpu [:stats :fallback])))
      (is (= [:int :float :float :long] (mapv :dtype (:abi kernel))))
      (is (= :kernel-body (get-in kernel [:provenance :dialect])))
      (is (re-find #"int rstr_map_load_[0-9]+ = .*idx\[" kernel-source))
      (is (re-find #"src\[.*rstr_map_load_[0-9]+" kernel-source)
          "the index tensor load is explicit SSA feeding the stable source load")
      (is (not (str/includes? kernel-source "src[idx[idx]]"))))))

(deftest strided-gather-is-a-flattened-typed-map
  (let [{:keys [program stats]} (route/attempt strided-source :float array-types
                                               {:scalar-types scalar-types})
        scalar-equation (first (:equations program))
        map-equation (second (:equations program))
        algorithm (:algorithm map-equation)
        operation (-> algorithm dialect/equations first dialect/operation-parts)
        body (-> operation :lambda dialect/lambda-parts :body-results first)]
    (is (= :typed-soac (:dialect program)))
    (is (= :analyzed-source (:front-end stats)))
    (is (= 'scalar (-> scalar-equation :algorithm dialect/equations first
                       dialect/operation-parts :kind)))
    (is (= 'map (:kind operation)))
    (is (= (first (:results scalar-equation))
           (get-in operation [:attributes :extent])))
    (is (some #{'clojure.core/quot} (flatten body)))
    (is (some #{'clojure.core/rem} (flatten body)))
    (is (not-any? #{'raster.par/gather}
                  (tree-seq coll? seq (:source program))))))

(deftest strided-gather-reuses-one-schedule-across-jvm-and-gpu
  (let [{:keys [program]} (route/attempt strided-source :float array-types
                                         {:scalar-types scalar-types})
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:dtype :float :target-device :ocl:0
                                   :array-types array-types :scalar-types scalar-types}))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        gpu (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                     :dtype :float :min-elements 0
                                     :array-types array-types
                                     :scalar-types scalar-types)
        kernel-source (:source (first (:kernels gpu)))
        evaluable-jvm-form (walk/postwalk
                            #(if (symbol? %) (with-meta % nil) %)
                            (:form jvm))
        jvm-values (eval
                    (list 'let* ['out (list 'float-array 4)
                                 'src (list 'float-array [10.0 11.0 20.0 21.0 30.0 31.0])
                                 'idx (list 'int-array [2 0])
                                 'n 2
                                 'stride 2]
                          (list 'vec evaluable-jvm-form)))]
    (testing "JVM consumes the typed map and honestly scalarizes the unsupported vector schedule"
      (is (= 1 (get-in jvm [:stats :segop-reused])))
      (is (nil? (get-in jvm [:stats :segop-relowered])))
      (is (= 1 (get-in jvm [:stats :fallback])))
      (is (= [30.0 31.0 10.0 11.0] jvm-values)))
    (testing "GPU emits the same flattened scalar region"
      (is (= 1 (get-in gpu [:stats :segop-reused])))
      (is (zero? (get-in gpu [:stats :fallback])))
      (is (str/includes? kernel-source " / stride"))
      (is (str/includes? kernel-source " % stride")))))
