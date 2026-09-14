(ns raster.compiler.typed-soa-pipeline-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-artifact :as artifact]
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
      (is (= '[particles_x particles_id out n] (:all-params program)))
      (is (= '[particles_x particles_id out] (:array-params program)))
      (is (= :typed-soac (get-in program [:compiler-report :route :source-dialect])))
      (is (true? (get-in program [:compiler-report :route :typed-validated])))
      (is (= {:kernel-body 1} (get-in program [:compiler-report :emission :routes])))
      (is (= :kernel-body (artifact/emission-route kernel)))
      (is (zero? (get-in program [:compiler-report :lowering :backend-relowered])))
      (is (zero? (get-in program [:compiler-report :lowering :fallback]))))))
