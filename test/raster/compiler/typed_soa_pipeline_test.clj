(ns raster.compiler.typed-soa-pipeline-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm defvalue]]
            [raster.gpu.device-probe :as probe]
            [raster.par]))

(defvalue TypedParticle [x :- Float, id :- Long])

(deftm project-particle-x!
  [particles :- TypedParticleSoA, out :- (Array float), n :- Long] :- (Array float)
  (let [effect (raster.par/map-void!
                index n
                (aset out index (.x (aget particles index))))]
    out))

(deftest resident-soa-is-scalar-replaced-before-typed-soac
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "typed resident SoA pipeline")
    (let [program (pipeline/compile-gpu-program
                   #'project-particle-x! :ocl:0 :dtype :float :compiler-report? true)
          kernel (get-in program [:steps 0 :artifact])]
      (is (= '[particles out n] (:all-params program)))
      (is (= '[particles out] (:array-params program)))
      (is (= ["x" "id"]
             (mapv :field (get-in program [:value-specs 'particles :leaves]))))
      (is (= 'particles (get-in kernel [:abi 0 :binding])))
      (is (= "x" (get-in kernel [:abi 0 :field])))
      (is (= :typed-soac (get-in program [:compiler-report :route :source-dialect])))
      (is (true? (get-in program [:compiler-report :route :typed-validated])))
      (is (= {:kernel-body 1} (get-in program [:compiler-report :emission :routes])))
      (is (= :kernel-body (artifact/emission-route kernel)))
      (is (zero? (get-in program [:compiler-report :lowering :backend-relowered])))
      (is (zero? (get-in program [:compiler-report :lowering :fallback])))
      (let [lowering (resident-plan/lower
                      {:id :typed-soa-test :target :ocl:0 :descriptor program
                       :arguments [{"x" (float-array 8) "id" (long-array 8)}
                                   (float-array 8) 8]})]
        (is (= ["x" "id"]
               (mapv :name
                     (get-in lowering [:plan :values [:typed-soa-test 'particles] :leaves]))))))))
