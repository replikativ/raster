(ns raster.compiler.passes.parallel.typed-reducing-scatter-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.pipeline :as pipeline]))

(def ^:private float-types
  {'out :float 'vals :float 'keys :int})

(def ^:private scalar-types
  {'n :long 'stride :long})

(defn- schedule [source dtype array-types target]
  (pipeline/schedule-parallel-form
   source {:target-device target :dtype dtype
           :array-types array-types :scalar-types scalar-types}))

(deftest scatter-add-carries-a-checked-conflict-algebra
  (let [source '(let* [step (raster.par/scatter! out vals keys n)] step)
        {:keys [form stats]} (schedule source :float float-types :cpu:0)
        algorithm (get-in form [:equations 0 :algorithm])
        equation (first (dialect/equations algorithm))
        operation (dialect/operation-parts equation)
        conflict (get-in operation [:attributes :conflict])]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= 'scatter (:kind operation)))
    (is (= :reduce (:kind conflict)))
    (is (= '+ (:operator conflict)))
    (is (= :float (:dtype conflict)))
    (is (scan/associative-scan? (:algebra conflict)))
    (is (= conflict (dialect/reducing-scatter-conflict '+ :float)))
    (is (= [{:destination 'out :access :read-write :host-return :buffer}]
           (get-in (dialect/facts algorithm)
                   [:equations 0 :attributes :result-storage])))
    (is (false? (dialect/reducing-scatter-conflict?
                 (assoc conflict :identity 1.0))))))

(deftest reducing-scatter-shares-one-schedule-boundary-across-jvm-and-opencl
  (let [source '(let* [step (raster.par/scatter! out vals keys n)] step)
        {scheduled :form} (schedule source :float float-types :ocl:0)
        operation (-> scheduled :equations first :operations first)
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[out vals keys n] (:form jvm)))
        out (float-array [10.0 20.0 30.0])
        gpu (opencl-pass/opencl-pass
             scheduled :device-id :ocl:0 :dtype :float :min-elements 0
             :array-types float-types :scalar-types scalar-types)
        kernel (first (:kernels gpu))]
    (testing "the JVM consumes the SegMap as an exact sequential update schedule"
      (is (= :reduce (:write-conflict operation)))
      (is (= 1 (get-in jvm [:stats :segop-reused])))
      (is (= 1 (get-in jvm [:stats :scalar-effect-maps])))
      (is (zero? (get-in jvm [:stats :fallback])))
      (is (identical? out (execute out (float-array [1 2 3 4])
                                   (int-array [2 0 2 0]) 4)))
      (is (= [16.0 20.0 34.0] (mapv double out))))
    (testing "OpenCL selects typed atomic addition without a legacy scatter generator"
      (is (= 1 (get-in gpu [:stats :segop-reused])))
      (is (zero? (get-in gpu [:stats :fallback])))
      (is (= :reducing-scatter (get-in kernel [:effects :kind])))
      (is (= :reduce (get-in kernel [:effects :write-conflict])))
      (is (= :inout (:kind (second (:abi kernel)))))
      (is (str/includes? (:source kernel) "float atomic_add_float"))
      (is (str/includes? (:source kernel) "atomic_add_float(out_ + keys[idx], vals[idx])")))))

(deftest strided-scatter-is-the-same-proof-carrying-reduction
  (let [source '(let* [step (raster.par/scatter! out vals keys n stride)] step)
        {scheduled :form stats :stats} (schedule source :float float-types :ocl:0)
        operation (-> scheduled :equations second :operations first)
        algorithm (-> scheduled :equations second :algorithm)
        conflict (-> algorithm dialect/equations first dialect/operation-parts
                     :attributes :conflict)
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[out vals keys n stride] (:form jvm)))
        out (float-array [10.0 20.0 30.0 40.0])
        gpu (opencl-pass/opencl-pass
             scheduled :device-id :ocl:0 :dtype :float :min-elements 0
             :array-types float-types :scalar-types scalar-types)
        kernel (first (:kernels gpu))
        kernel-source (:source kernel)]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= :reduce (:kind conflict)))
    (is (= '+ (:operator conflict)))
    (is (= :reduce (:write-conflict operation)))
    (is (not-any? #{'raster.par/scatter!}
                  (tree-seq coll? seq (:source scheduled))))
    (testing "JVM performs the exact sequential reduction through the scheduled effect region"
      (is (= 1 (get-in jvm [:stats :segop-reused])))
      (is (nil? (get-in jvm [:stats :segop-relowered])))
      (is (identical? out (execute out (float-array [1 2 3 4 5 6])
                                   (int-array [1 0 1]) 3 2)))
      (is (= [13.0 24.0 36.0 48.0] (mapv double out))))
    (testing "GPU derives an atomic add from the same conflict certificate"
      (is (= 1 (get-in gpu [:stats :segop-reused])))
      (is (zero? (get-in gpu [:stats :fallback])))
      (is (= :reducing-scatter (get-in kernel [:effects :kind])))
      (is (str/includes? kernel-source "atomic_add_float"))
      (is (str/includes? kernel-source " / stride"))
      (is (str/includes? kernel-source " % stride")))))

(deftest additive-effect-recognition-refuses-an-extra-destination-read
  (let [unsafe
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (clojure.core/aset
                  out (clojure.core/aget keys i)
                  (clojure.core/+
                   (clojure.core/aget out (clojure.core/aget keys i))
                   (clojure.core/aget out i))))]
               effect)
        routed (pipeline/schedule-parallel-form
                unsafe {:target-device :cpu:0 :dtype :float
                        :array-types float-types :scalar-types scalar-types})]
    (is (not= :typed-soac (get-in routed [:stats :source-dialect])))
    (is (= :typed-soac-source-coverage
           (get-in routed [:stats :typed-soac-declined :reason])))
    (is (= :unsupported-parallel-operation
           (get-in routed [:stats :typed-soac-declined :bindings 0 :reason])))))

(deftest verified-array-dtype-selects-the-target-atomic-spelling
  (let [operation '(raster.par/atomic-add! out i contribution)
        emit (fn [config]
               (binding [c-emit/*emit-config* config
                         c-emit/*array-dtypes* {'out :float}
                         c-emit/*scalar-type* "float"]
                 (c-emit/emit-stmt operation 'i #{'out} "idx")))]
    (is (= "atomic_add_float(out_ + idx, contribution);"
           (emit c-emit/opencl-config)))
    (is (= "atomicAdd(out_ + idx, contribution);"
           (emit c-emit/hip-config)))))

(deftest reduce-by-key-is-the-same-typed-additive-conflict-operation
  (let [source '(let* [step (raster.par/reduce-by-key out keys vals n +)] step)
        {scheduled :form stats :stats} (schedule source :int
                                                 {'out :int 'keys :int 'vals :int}
                                                 :ocl:0)
        operation (-> scheduled :equations first :operations first)
        gpu (opencl-pass/opencl-pass
             scheduled :device-id :ocl:0 :dtype :int :min-elements 0
             :array-types {'out :int 'keys :int 'vals :int}
             :scalar-types scalar-types)
        source (:source (first (:kernels gpu)))]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= :reduce (:write-conflict operation)))
    (is (= :int (get-in operation [:conflict-contract :dtype])))
    (is (str/includes? source "atomic_add(out_ + keys[idx], vals[idx])"))
    (is (not (str/includes? source "atomic_add_float")))))
