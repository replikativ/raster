(ns raster.compiler.backend.gpu.static-loop-emission-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.gpu.device-probe :as probe]))

(defn- loop-body [pipeline? type lower upper step]
  (let [kernel (if pipeline? (fixtures/pipelined-staging-body 4 :preferred)
                  (body/make
                   {:id :counted-loop-test
                    :operations [(body/->ForLoop (body/value 'iteration :int) 0 1 1 []
                                                  [(body/->Yield [])] [] {})]
                    :launch (launch/spec {:workgroup-size [1] :group-count [1]})}))
        position (if pipeline? 4 0)
        index-id (get-in kernel [:operations position :index :id])
        bound #(if (and (= type :long) (integer? %)) (body/index-cast % :long :exact) %)]
    (-> kernel
        (update :parameters conj (body/->KernelParameter 'bound :scalar type [] nil nil :bound))
        (update-in [:operations position] assoc :index (body/value index-id type)
                   :lower (bound lower) :upper (bound upper) :step step)
        body/validate!)))

(deftest static-counted-loops-use-proved-canonical-advance
  (doseq [pipeline? [false true] type [:int]
          dialect [:opencl-portable :cuda :hip]
          [lower upper step] [[0 8 1] [0 8 3] [8 8 3] [9 8 3] [-5 4 3]]]
    (let [source (emit/emit-scalar-kernel "counted_loop"
                                         (loop-body pipeline? type lower upper step)
                                         {:target-dialect dialect})]
      (is (str/includes? source (str " += " step ") {")))
      (is (not (str/includes? source "break;"))))))

(deftest signed-limits-and-unknown-bounds-keep-the-safe-policy
  (doseq [pipeline? [false true] type [:int :long] dialect [:opencl-portable :cuda :hip]]
    (let [maximum (if (= type :int) Integer/MAX_VALUE Long/MAX_VALUE)
          source #(emit/emit-scalar-kernel "counted_edge" (loop-body pipeline? type %1 %2 %3)
                                           {:target-dialect dialect})]
      ;; Long bounds carry explicit IndexCasts in KernelBody. They deliberately retain checks;
      ;; this slice does not strip casts or introduce another constant evaluator.
      (is (str/includes? (source (dec maximum) maximum 1)
                         (if (= type :int) " += 1) {" "break;")))
      (is (str/includes? (source (dec maximum) maximum 2) "break;"))
      (is (str/includes? (source 0 'bound 1) "break;"))
      (is (str/includes? (source (body/index-cast 0 type :exact) 8 1) "break;")))))

(deftest counted-loop-exit-preserves-carried-values-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "canonical counted-loop carries")
    (let [cases [[0 8 1] [0 8 3] [8 8 3] [9 8 3] [-5 4 3]]
          kernel (body/make
                  {:id :counted-loop-carried-values
                   :parameters [(body/->KernelParameter 'out :output :float [5] :global
                                                        (layout/row-major [5] :float) :result)]
                   :operations
                   (vec (mapcat
                         (fn [position [lower upper step]]
                           (let [[index carry next-value result]
                                 (mapv #(symbol (str % position)) ["i" "carry" "next" "result"])]
                             [(body/->ForLoop
                               (body/value index :int) lower upper step
                               [(body/->LoopArg (body/value carry :float) (body/literal 7.0 :float))]
                               [(body/->ScalarCompute (body/value next-value :float)
                                                      (body/scalar-expression :+ :float [carry (body/literal 1.0 :float)]))
                                (body/->Yield [next-value])]
                               [(body/value result :float)] {})
                              (body/->ScalarStore 'out [position] result nil)]))
                         (range) cases))
                   :launch (launch/spec {:workgroup-size [1] :group-count [1]})})
          refinement (scheduled/make
                      {:source {:kind :counted-loop-fixture :cases cases} :body kernel :arguments '[out]
                       :effects {:kind :counted-loop-fixture :uses (scheduled/derive-uses kernel '[out])}
                       :legality {:kind :fixture} :numerics {:mode :exact :policy :small-exact-floats}})
          compiled (target/emit-artifact "counted_loop_carries" refinement :opencl-portable)
          ocl (find-ns 'raster.gpu.ocl-runtime)
          op #(ns-resolve ocl %)
          out ((op 'buffer-of-array) (float-array (repeat 5 -555.0)) :float)]
      (try
        ((op 'register-kernel!) (:kernel-name compiled) compiled)
        (let [prepared ((op 'bind-kernel-call) (call/make compiled [out]))]
          (try ((op 'launch-registered-bound!) prepared)
               (finally ((op 'destroy-prepared!) prepared))))
        (is (= [15.0 10.0 7.0 7.0 10.0] (vec ((op 'buffer->array) out))))
        (finally ((op 'free-buffer!) out))))))
