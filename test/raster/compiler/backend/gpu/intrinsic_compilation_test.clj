(ns raster.compiler.backend.gpu.intrinsic-compilation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.c-emit :as c]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.kernel-body-opencl :as scalar]
            [raster.compiler.backend.gpu.staged-contraction-fixtures :as fixtures]
            [raster.compiler.passes.parallel.staged-contraction-body :as staged]))

(def selection {:dp4a :opencl-packed-dot})

(deftest helpers-carry-only-consumed-requirements
  (is (= {:source "" :compilation {}}
         (c/intrinsic-helper-module "x + y" :opencl-portable selection)))
  (let [module (c/intrinsic-helper-module "rstr_dp4a(a,b,c)" :opencl-portable selection)]
    (is (= {:language-standard "CL3.0" :extensions #{"cl_khr_integer_dot_product"}}
           (:compilation module)))
    (is (re-find #"dot_4x8packed_ss_int" (:source module)))
    (is (re-find #"as_uint\(acc\)" (:source module)))
    (is (not (re-find #"dot_acc_sat" (:source module)))))
  (doseq [dialect [:cuda :hip]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unsupported intrinsic"
                         (c/intrinsic-helper-module "rstr_dp4a(a,b,c)" dialect selection)))))

(deftest typed-body-projection-preserves-native-helper-contract
  (let [scheduled (staged/lower (fixtures/packed-facts 1 128 32 32))
        ordinary (target/emit-artifact "ordinary_dot" scheduled :opencl-portable)
        native (target/emit-artifact "native_dot" scheduled :opencl-portable
                                     {:target-features {:intrinsic-implementations selection}})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"need emit-scalar-module"
                         (scalar/emit-scalar-kernel "cannot_drop_contract" (:body scheduled)
                           {:target-dialect :opencl-portable
                            :target-features {:intrinsic-implementations selection}})))
    (is (nil? (get-in ordinary [:attributes :compilation])))
    (is (= "CL3.0" (get-in native [:attributes :compilation :language-standard])))
    (is (re-find #"dot_4x8packed_ss_int" (:source native)))
    (is (= (select-keys ordinary [:abi :arguments :effects :launch])
           (select-keys native [:abi :arguments :effects :launch])))))
