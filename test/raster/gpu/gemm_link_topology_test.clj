(ns raster.gpu.gemm-link-topology-test
  "Model-free reachability for the public LinkPlan GEMM path.

   A semantic GEMM may select a conversion/layout/split graph.  This test pins the exact
   flattening and constant-prologue boundary without loading a driver or allocating the large
   logical buffers that make split-K profitable."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link]))

(defn- split-gemm-descriptor
  []
  (let [tile (hardware/derive-gemm-tile {})]
    {:dtype :float
     :gemm-precision :mixed-f16-f32
     :schedule {:precision :mixed-f16-f32}
     :all-params '[a b c m n k]
     :array-params '[a b c]
     :array-roles {'a :input 'b :input 'c :output}
     :scalar-params '[m n k]
     :allocs []
     :steps
     [{:convention :gemm :variant :nt :A 'a :B 'b :C 'c
       :dispatch
       (gemm/emit-executable
        {:id "mock-linked-gemm" :a 'a :b 'b :c 'c
         :m :gemm-m :n :gemm-n :k :gemm-k :variant :nt
         :precision :mixed-f16-f32 :tile tile :fill-workgroups 32})
       :argument-specs [{:kind :input :sym 'a}
                        {:kind :input :sym 'b}
                        {:kind :output :sym 'c}
                        {:kind :scalar :type :int :value-fn #(nth % 3)}
                        {:kind :scalar :type :int :value-fn #(nth % 4)}
                        {:kind :scalar :type :int :value-fn #(nth % 5)}]
       :strategy-selection {:path [:precision]
                            :mapping {:f32-scalar :f32-scalar}
                            :default :auto}
       :phase :gemm}]
     :result-sym 'c}))

(defn- split-gemm-plan
  [descriptor m n k]
  (link-plan/make
   {:id :split-gemm-topology
    :target :ze:0
    :nodes [(link-plan/node {:id :a :dtype :float :shape [m k]
                             :device :ze:0 :role :input})
            (link-plan/node {:id :b :dtype :float :shape [n k]
                             :device :ze:0 :role :constant})
            (link-plan/node {:id :c :dtype :float :shape [m n]
                             :device :ze:0 :role :output})]
    :instances [(link-plan/instance
                 {:id :projection
                  :descriptor descriptor
                  :bindings {'a :a 'b :b 'c :c}
                  :scalars {'m m 'n n 'k k}})]
    :outputs [:c]}))

(defn- element-bytes
  [dtype]
  (case dtype
    (:byte :int8) 1
    (:half :short) 2
    (:double :long) 8
    4))

(deftest link-plan-flattens-selected-gemm-graph-and-hoists-constant-transforms
  (let [m 13 n 640 k 262144
        descriptor (split-gemm-descriptor)
        plan (split-gemm-plan descriptor m n k)
        session (atom {:device-id :ze:0 :session-id :mock-session
                       :buffers {} :allocations {} :prepared {} :graphs {}
                       :kernel-graphs {} :events {} :closed? false})
        registered (atom [])
        recorded (atom [])
        replayed (atom [])
        runtime-function
        (fn [_ name]
          (case name
            "make-buffer"
            (fn [elements dtype]
              {:dtype dtype :n-elements elements
               :byte-size (* (long elements) (element-bytes dtype))
               :alignment 64})

            "array->buffer!" (fn [buffer _] buffer)
            "buffer-as-float-buffer" (fn [& _] (throw (AssertionError. "unused")))
            "buffer-as-int-buffer" (fn [& _] (throw (AssertionError. "unused")))
            "register-kernel!" (fn [name artifact] (swap! registered conj [name artifact]))
            "bind-kernel-call" (fn [call] {:kernel-call call})
            "record-graph!" (fn [bounds & [options]]
                              (let [graph {:bounds (vec bounds) :options options}]
                                (swap! recorded conj graph)
                                graph))
            "replay-graph!" (fn [graph] (swap! replayed conj graph))
            (throw (ex-info "unexpected mocked runtime function" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) runtime-function}
      (fn []
        (let [executable (gpu-link/instantiate! plan {:session session})
              phase (first (:phases executable))
              bound (get-in @session [:prepared phase])
              prepareds (:prepareds bound)
              graph-entry (get-in @session [:graphs (:graph-key executable)])]
          (testing "the selected schedule is flattened behind one semantic LinkPlan step"
            (is (= 1 (count (:phases executable))))
            (is (= 5 (count prepareds))
                "A/B conversion, B transpose, split contraction, and combine are all reachable")
            (is (= 4 (count (:temporary-buffers bound)))
                "A16, B16, transposed B16, and split partials remain graph-private")
            (is (= 5 (count @registered)))
            (is (every? #(contains? % :kernel-call) prepareds)))
          (testing "only transforms derived entirely from the constant weight enter the prologue"
            (is (= [2 3] (mapv (comp count :bounds) @recorded)))
            (is (= 2 (count (filter :const-prologue? prepareds))))
            (is (= 3 (count (remove :const-prologue? prepareds))))
            (is (= 1 (count @replayed)) "the constant prologue executes once at instantiation")
            (is (= 2 (count (:bounds (:prologue-graph graph-entry)))))
            (is (= 3 (count (:bounds (:replay-graph graph-entry)))))))))))
