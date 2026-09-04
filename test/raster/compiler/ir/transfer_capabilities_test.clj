(ns raster.compiler.ir.transfer-capabilities-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.transfer-capabilities :as capabilities]
            ;; pure fact tables: no device is needed to read them
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

(deftest every-backend-states-the-whole-contract
  (let [opencl (ocl/transfer-capabilities)
        level-zero (ze/transfer-capabilities)]
    (is (= opencl (capabilities/validate! opencl)))
    (is (= level-zero (capabilities/validate! level-zero)))
    (testing "the facts a scheduler plans against are honest, not inferred from the API"
      (is (true? (:independent-physical-queue? opencl)))
      (is (false? (:physically-serialized? opencl)))
      (is (= :device-completion (:event-semantics opencl)))
      (is (true? (:host-lease-until-await? opencl))
          "a download's host destination is written at await time")
      (is (true? (:physically-serialized? level-zero)))
      (is (= :already-complete (:event-semantics level-zero)))
      (is (= :none (:host-staging level-zero)) "an inline copy has no in-flight staging copy")
      (is (= :any (:host-memory level-zero)) "the host argument is any array or segment")
      (is (false? (:peer-transfer? level-zero)) "no peer route exists yet, and it says so"))))

(deftest a-missing-or-contradictory-fact-is-refused
  (let [complete (capabilities/validate! (ocl/transfer-capabilities))]
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
                           (assoc complete :submission :inline-host-copy
                                  :queue-ordering :inline :event-semantics :already-complete
                                  :physically-serialized? true))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"inline ordering"
                          (capabilities/validate!
                           (assoc (ze/transfer-capabilities) :event-semantics :device-completion))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"queued device work"
                          (capabilities/validate!
                           (assoc complete :queue-ordering :inline
                                  :physically-serialized? true))))))

(deftest an-out-of-order-shared-queue-may-overlap
  ;; one out-of-order device queue for compute and transfers is legal and not serialized
  (is (capabilities/validate!
       {:submission :device-event :host-staging :none :independent-physical-queue? false
        :queue-ordering :out-of-order :async-h2d? true :async-d2h? true
        :peer-transfer? false :peer-mechanisms [] :event-semantics :device-completion
        :host-lease-until-await? true :host-memory :pinned :physically-serialized? false})))
