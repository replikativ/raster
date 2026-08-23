(ns raster.compiler.ir.kernel-artifact-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-launch :as klaunch]))

(def ^:private abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot '_n_bound :scalar :int :role :bound)])

(def ^:private base
  {:kernel-name "reduce0"
   :source (str "__kernel void reduce0(__global const float* x, "
                "__global float* out, int _n_bound) {}")
   :abi abi
   :arguments '[x out n]
   :launch (klaunch/spec {:workgroup-size [256]
                          :group-count [(klaunch/ceil-div 'n 256)]})
   :temporaries []
   :effects {:kind :pure-reduction}
   :provenance {:dialect :segred :segop-id 7}
   :attributes {:dtype :float :c-op "+" :identity-val 0.0}})

(deftest artifact-is-a-verified-compiler-value
  (let [a (kart/make base)]
    (is (kart/kernel-artifact? a))
    (is (= :opencl-c (:target a)))
    (is (= [256] (kart/launch-value a :workgroup-size)))
    (is (= "+" (kart/attribute a :c-op)))
    (is (identical? a (kart/validate! a)))))

(deftest artifact-refuses-an-unverified-module-or-call
  (testing "source and ABI parameter order cannot diverge"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"signature does not match"
         (kart/make (assoc base :source
                           "__kernel void reduce0(__global float* out, __global const float* x, int _n_bound) {}")))))
  (testing "compiler arguments cannot be silently dropped"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"argument count mismatch"
         (kart/make (assoc base :arguments '[x out])))))
  (testing "an entry without executable launch geometry is not a kernel artifact"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"invalid launch contract"
         (kart/make (assoc base :launch {:dimensions 1}))))))
