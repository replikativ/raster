(ns raster.compiler.ir.transfer-capabilities-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.transfer-capabilities :as capabilities]))

(deftest every-backend-states-the-whole-contract
  ;; pure fact tables: no device is needed to read them
  (require 'raster.gpu.ocl-runtime 'raster.gpu.ze-runtime)
  (let [opencl ((resolve 'raster.gpu.ocl-runtime/transfer-capabilities))
        level-zero ((resolve 'raster.gpu.ze-runtime/transfer-capabilities))]
    (is (= opencl (capabilities/validate! opencl)))
    (is (= level-zero (capabilities/validate! level-zero)))
    (testing "the facts a scheduler plans against are honest, not inferred from the API"
      (is (true? (:independent-physical-queue? opencl)))
      (is (false? (:physically-serialized? opencl)))
      (is (= :device-completion (:event-semantics opencl)))
      (is (true? (:physically-serialized? level-zero)))
      (is (= :already-complete (:event-semantics level-zero)))
      (is (false? (:peer-transfer? level-zero)) "no peer route exists yet, and it says so"))))

(deftest a-missing-or-contradictory-fact-is-refused
  (let [complete (capabilities/validate! ((resolve 'raster.gpu.ocl-runtime/transfer-capabilities)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"state every fact"
                          (capabilities/validate! (dissoc complete :async-d2h?))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"physically serialized"
                          (capabilities/validate!
                           (assoc complete :independent-physical-queue? false
                                  :physically-serialized? false))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mechanisms must agree"
                          (capabilities/validate! (assoc complete :peer-transfer? true))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not asynchronous"
                          (capabilities/validate!
                           (assoc complete :submission :inline-host-copy))))))
