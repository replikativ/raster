(ns raster.compiler.backend.gpu.matrix-target-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.kernel-body-target :as body-target]
            [raster.compiler.backend.gpu.matrix-target :as matrix-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.kernel-call :as kernel-call]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]))

(def ^:private mma
  {:family :mma :m 16 :n 16 :k 16 :subgroup 32})

(defn- mma-tile []
  (assoc (hardware/derive-gemm-tile
          {:device-type :gpu :backend :cuda :compute-capability [8 0]
           :subgroup-size 32 :grf-bytes-per-lane 256 :matrix mma})
         :matrix mma))

(defn- mma-body []
  (schedule/matrix-body
   {:id :matrix-target-test :row 'a :col 'b :out 'c
    :dimensions [128 128 64] :dimension-parameters ['m 'n 'k]
    :tile (mma-tile) :result-dtype :float}))

(deftest target-lowering-forks-after-one-verified-body
  (let [body (mma-body)
        cuda (matrix-target/emit-matrix-kernel "matrix_target_cuda" body :cuda)]
    (is (= :cuda-c (:target cuda)))
    (is (= :cuda (:target-dialect cuda)))
    (is (identical? body (:kernel-body cuda)))
    (is (re-find #"wmma::mma_sync" (:source cuda)))
    (testing "an instruction cannot leak into an unrelated target spelling"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"no builtin"
           (matrix-target/emit-matrix-kernel "mma_as_dpas" body :opencl-intel))))
    (testing "unimplemented families decline at the single target boundary"
      (try
        (matrix-target/emit-matrix-kernel "mma_as_mfma" body :hip)
        (is false "HIP matrix target unexpectedly accepted an unimplemented MFMA row")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :matrix-target-dialect-not-lowered (:reason (ex-data exception)))))))))

(deftest scheduled-matrix-entry-no-longer-implies-opencl
  (let [emitted
        (gemm/emit-scheduled-matrix-kernel
         {:kernel-name "scheduled_cuda_mma"
          :a 'a :b 'b :c 'c
          :m 128 :n 128 :k 64
          :dimension-parameters ['m 'n 'k]
          :tile (mma-tile)
          :result-dtype :float
          :target-dialect :cuda})]
    (is (= :cuda-c (:target emitted)))
    (is (= :cuda (:target-dialect emitted)))
    (is (= ['a 'b 'c 'm 'n 'k]
           (mapv :id (get-in emitted [:kernel-body :parameters]))))
    (is (re-find #"extern \"C\" __global__" (:source emitted)))
    (is (not (re-find #"__kernel" (:source emitted))))))

(deftest verified-body-owns-the-complete-target-artifact
  (let [body (-> (mma-body)
                 (assoc :provenance {:scheduled-operation :body-certificate
                                     :dialect :matrix-schedule})
                 (assoc-in [:attributes :instruction-family] :mfma))
        scheduled (scheduled-body/make
                   {:source :body-certificate
                    :body body
                    :arguments ['a 'b 'c 128 128 64]
                    :effects {:kind :tensor-contraction-stage
                              :uses [{:value 'a :access :read}
                                     {:value 'b :access :read}
                                     {:value 'c :access :write}]}
                    :legality {:kind :matrix-instruction-tiling}
                    :numerics {:mode :reassociated :policy :tiled-contraction
                               :rounding :nearest-even :accumulator-dtype :float}})
        artifact (body-target/emit-artifact
                  "matrix_artifact_cuda" scheduled :cuda
                  {:provenance {:lowering :caller-lie :target-dialect :hip}
                   :attributes {:kernel-body :caller-lie :instruction-family :mfma}})
        aligned-buffer (fn [id] {:id id :alignment 32})
        call (kernel-call/make
              artifact
              [(aligned-buffer :a-buffer)
               (aligned-buffer :b-buffer)
               (aligned-buffer :c-buffer)
               {:type :int :value 128}
               {:type :int :value 128}
               {:type :int :value 64}])]
    (is (kernel-artifact/kernel-artifact? artifact))
    (is (= :cuda-c (:target artifact)))
    (is (= ['a 'b 'c 128 128 64] (:arguments artifact)))
    (is (= [1 1] (get-in call [:geometry :group-count])))
    (is (empty? (mapcat kernel-launch/expression-references
                        (concat (get-in artifact [:launch :workgroup-size])
                                (get-in artifact [:launch :group-count]))))
        "static dimension specialization closes launch expressions")
    (is (= [:no-write-alias :no-write-alias nil]
           (mapv :aliasing (take 3 (:abi artifact))))
        "stable body reads survive target ABI projection")
    (is (= [:lhs :rhs :result :dimension :dimension :dimension]
           (mapv :role (:abi artifact))))
    (is (= [32 32 32]
           (mapv :alignment (take 3 (:abi artifact))))
        "WMMA load/store alignment is a checked call precondition")
    (is (identical? body (get-in artifact [:attributes :kernel-body])))
    (is (= :matrix (get-in artifact [:attributes :body-family])))
    (is (= :scheduled-kernel-body (get-in artifact [:provenance :lowering])))
    (is (= :cuda (get-in artifact [:provenance :target-dialect])))
    (is (identical? scheduled (get-in artifact [:provenance :scheduled-operation])))
    (is (= :mma (get-in artifact [:attributes :target-facts :instruction-family]))
        "physical identity comes from MatrixMad, not descriptive body attributes")
    (let [downgraded (-> artifact
                         (update :abi #(mapv (fn [slot] (dissoc slot :alignment)) %))
                         (assoc-in [:attributes :target-facts :pointer-alignment] nil)
                         (assoc-in [:provenance :target-facts :pointer-alignment] nil))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"requires its verified pointer alignment"
           (scheduled-body/validate-artifact-projection! scheduled downgraded))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"alignment contract"
         (kernel-call/make
          artifact
          [{:id :a-buffer :alignment 16}
           (aligned-buffer :b-buffer)
           (aligned-buffer :c-buffer)
           {:type :int :value 128}
           {:type :int :value 128}
           {:type :int :value 64}])))))

(deftest target-symbols-are-validated-before-source-emission
  (let [body (mma-body)]
    (testing "the public entry point is already a portable identifier"
      (try
        (matrix-target/emit-matrix-kernel "bad-name" body :cuda)
        (is false "invalid entry point unexpectedly reached CUDA emission")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :matrix-target-entry-point (:reason (ex-data exception)))))))
    (testing "the CUDA C++ keyword set is part of the public-name contract"
      (try
        (matrix-target/emit-matrix-kernel "default" body :cuda)
        (is false "C++ keyword unexpectedly reached CUDA emission")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :matrix-target-entry-point (:reason (ex-data exception)))))))
    (testing "ABI spelling cannot shadow an emitter-owned local"
      (try
        (matrix-target/emit-matrix-kernel
         "valid_name" body :opencl-intel {:parameter-names {'a "m_base"}})
        (is false "generated-local collision unexpectedly reached OpenCL emission")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :matrix-target-name-collision (:reason (ex-data exception)))))))
    (testing "CUDA currently has one exact verified spelling"
      (try
        (matrix-target/emit-matrix-kernel
         "valid_name" body :cuda {:parameter-names {'a "lhs"}})
        (is false "unsupported CUDA spelling override was accepted")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :cuda-mma-parameter-spelling-unsupported
                 (:reason (ex-data exception)))))))
    (testing "a dynamically emitted grid-Z local shares the ABI collision check"
      (let [grid-body
            (schedule/matrix-body
             {:id :grid-z-name-collision :row 'a :col 'b :out 'c
              :dimensions [128 128 64] :dimension-parameters ['m 'n 'k]
              :tile (assoc (hardware/derive-gemm-tile {})
                           :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16})
              :result-dtype :float
              :additional-parameters
              [(kernel-body/->KernelParameter 'k_slice :scalar :int [] nil nil :schedule)]
              :additional-indices [(kernel-body/->IndexBinding 'k-slice :group 2)]
              :launch-group-count [1 1 (kernel-launch/runtime-value 'k_slice)]})]
        (try
          (matrix-target/emit-matrix-kernel "grid_z_collision" grid-body :opencl-intel)
          (is false "grid-Z local unexpectedly shadowed an ABI parameter")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :matrix-target-name-collision (:reason (ex-data exception))))))))))
