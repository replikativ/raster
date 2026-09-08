(ns raster.compiler.passes.parallel.staged-result-transform-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.staged-contracts :as public]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.contraction-closure :as closure]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.passes.parallel.staged-scalar-body :as staged]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.staged-scalar-body-test :as fixture]
            [raster.gpu.device-probe :as probe]
            [raster.runtime.hardware :as hardware]
            [raster.gpu.link :as link]))

(defn source []
  (facts/from-components
   (-> (select-keys (fixture/three-stage-facts) [:out :free-axes :contract-axes :body :opts])
       (assoc :dtype :float)
       (assoc-in [:opts :epilogue]
                 {:acc 'value :expr '(+ value (aget bias _)) :dtype :float
                  :operands [{:sym 'bias :dtype :float :map {:groups '[[[j 3]]]}}]}))))

(deftest result-transforms-share-storage-and-scalar-lowering
  (let [s (source)
        scheduled (staged/lower s :scalar-types {'scale :float})]
    (is (= '[a b bias weights out scale] (:arguments scheduled)))
    (is (= 3 (get-in scheduled [:legality :storage-elements 'bias])))
    (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (is (= [:float :float :float :float :float :float]
             (mapv :dtype (:abi (target/emit-artifact "staged_result_transform" scheduled dialect))))))))

(deftest result-transforms-cannot-read-closed-reduction-indices
  (doseq [mutate [#(assoc-in % [:epilogue :expr] '(+ value t))
                  #(assoc-in % [:epilogue :acc] 'scale)
                  #(assoc-in % [:epilogue :operands 0 :map] {:groups '[[[t 4]]]})]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (staged/lower (mutate (source)) :scalar-types {'scale :float}))))
  (let [s (source)
        attrs {:contraction s :array-parameters '[a b bias weights] :capture-parameters '[scale]}]
    (is (= {'bias 3}
           (into {} (map (juxt :parameter :elements))
                 (filter #(= 'bias (:parameter %)) (closure/storage-requirements attrs)))))))

(deftest result-transform-declared-types-cannot-retype-captures-or-storage
  (doseq [s [(assoc-in (source) [:epilogue :scalars] [{:sym 'scale :dtype :double}])
             (assoc-in (source) [:epilogue :scalars] [{:sym 'scale :dtype :float}
                                                    {:sym 'scale :dtype :float}])
             (-> (source)
                 (assoc-in [:epilogue :expr] '(+ value (aget a _)))
                 (assoc-in [:epilogue :operands 0 :sym] 'a)
                 (assoc-in [:epilogue :operands 0 :dtype] :double))]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (staged/lower s :scalar-types {'scale :float})))))

(deftest destination-reading-result-transforms-decline-before-closure-admission
  (let [s (-> (source)
              (assoc-in [:epilogue :expr] '(+ value (aget out _)))
              (assoc-in [:epilogue :operands 0 :sym] 'out))
        s (assoc-in s [:opts :epilogue] (:epilogue s))
        failure (try (staged/analyze! s :scalar-types {'scale :float}) nil
                     (catch clojure.lang.ExceptionInfo e (ex-data e)))]
    (is (= :staged-scalar-body-declined (:reason failure)))
    (is (= :result-transform-inout (:missing-rule failure)))
    (is (nil? (frontend/form->program
               (list 'let* ['effect (facts/surface-form s)] nil)
               {:dtype :float :array-types {'a :float 'b :float 'weights :float 'out :float}
                :scalar-types {'scale :float}})))))

(deftest public-result-transforms-have-checked-allocation-free-invocations
  (doseq [vendor [:cuda :hip]]
    (let [device (keyword (str (name vendor) ":staged-result-transform-test"))]
      (hardware/register-target-device!
       device {:type vendor :capabilities (cond-> {:warp-size 32 :subgroup-sizes [32]
                                                   :max-workgroup-size 1024}
                                           (= vendor :cuda) (assoc :compute-capability [8 0])
                                           (= vendor :hip) (assoc :gfx-arch :gfx1100))})
      (let [compilation (equation-first/compile #'public/floating-result-transform!
                                               {:target device :dtype :float})
            arguments [(float-array 16) (float-array 8) (float-array 2) (float-array 2) (float 1.5)]]
        (is (= :none (get-in compilation [:stats :fallback])))
        (is (= 0 (get-in (equation-first/lower compilation arguments) [:attributes :driver-allocations])))
        (is (thrown? clojure.lang.ExceptionInfo
                     (equation-first/lower compilation (assoc arguments 2 (float-array 1)))))))))

(deftest public-result-transform-executes-after-reduction
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "public staged result transform")
    (let [output (float-array (repeat 2 -555.0))
          compilation (equation-first/compile #'public/floating-result-transform! {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compilation
                                     [(float-array (repeat 16 1.0)) (float-array (repeat 8 2.0))
                                      (float-array [1.0 3.0]) output (float 1.5)])
          output-id (some (fn [[id node]] (when (identical? output (:source node)) id)) (:nodes plan))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (some? output-id))
      (let [executable (link/instantiate! plan)]
        (try
          (link/run! executable)
          (is (= [18.5 20.5] (vec (link/download executable output-id))))
          (finally (link/close! executable)))))))
