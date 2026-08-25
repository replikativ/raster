(ns raster.gpu.gemm-graph-program-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.core.hardware :as hardware]
            [raster.gpu.core :as gpu]))

(defn- descriptor
  []
  (let [tile (hardware/derive-gemm-tile {})]
    {:dtype :float
     :gemm-precision :f16-xmx
     :schedule {:precision :f16-xmx}
     :all-params ['a 'b 'c 'm 'n 'k]
     :array-params ['a 'b 'c]
     :array-roles {'a :input 'b :input 'c :output}
     :scalar-params ['m 'n 'k]
     :allocs []
     :steps
     [{:convention :gemm :variant :nt :A 'a :B 'b :C 'c
       :dispatch
       (gemm/emit-executable
        {:id "mock-resident-gemm" :a 'a :b 'b :c 'c
         :m :gemm-m :n :gemm-n :k :gemm-k :variant :nt
         :precision :f16-xmx :tile tile :fill-workgroups 32})
       :argument-specs [{:kind :input :sym 'a}
                        {:kind :input :sym 'b}
                        {:kind :output :sym 'c}
                        {:kind :scalar :type :int :value-fn #(nth % 3)}
                        {:kind :scalar :type :int :value-fn #(nth % 4)}
                        {:kind :scalar :type :int :value-fn #(nth % 5)}]
       :strategy-selection {:path [:precision]
                            :mapping {:f32-scalar :f32-scalar}
                            :default :auto}
       :phase :gpu-step-0}]
     :result-sym 'c}))

(deftest resident-program-flattens-the-selected-executable-graph
  (let [m 13 n 640 k 262144
        arguments [(float-array (* m k)) (float-array (* k n)) (float-array (* m n)) m n k]
        session (atom {:device-id :ze:0 :programs {} :buffers {}})
        registered (atom [])
        replayed (atom [])
        allocate! (fn [sess specs]
                    (swap! sess update :buffers merge
                           (into {}
                                 (map (fn [[key [dtype elements _]]]
                                        [key {:key key :dtype dtype :n-elements elements}]))
                                 specs)))
        runtime-function
        (fn [_ name]
          (case name
            "kernel-registry-entry" (fn [_] nil)
            "register-kernel!" (fn [name artifact] (swap! registered conj [name artifact]))
            "bind-registered-map-void-kernel" (fn [& _] (throw (ex-info "unused" {})))
            "bind-kernel-call" (fn [call] {:bound (:artifact call) :kernel-call call})
            "make-buffer" (fn [elements dtype] {:dtype dtype :n-elements elements
                                                :private? true})
            "record-graph!" (fn [bounds & [options]] {:bounds (vec bounds) :options options})
            "replay-graph!" (fn [graph] (swap! replayed conj graph))
            (throw (ex-info "unexpected mocked runtime function" {:name name}))))]
    (with-redefs-fn
      {#'gpu/alloc! allocate!
       (ns-resolve 'raster.gpu.core 'rt-resolve) runtime-function}
      (fn []
        (gpu/bind-program! session (descriptor) arguments {'b :constant})
        (let [{:keys [bounds scratch-buffers prologue-graph graph]}
              (get-in @session [:programs :program])]
          (is (= 5 (count bounds)) "convert A/B, transpose B, split contraction, combine")
          (is (= 4 (count scratch-buffers)) "A16, B16, BT16, and partials are graph-owned")
          (is (= 5 (count @registered)))
          (is (= 2 (count (:bounds prologue-graph)))
              "constant B conversion and its dependent transpose are hoisted")
          (is (= 3 (count (:bounds graph))))
          (is (= 1 (count @replayed)))
          (is (every? #(contains? % :kernel-call) bounds)))))))
