(ns raster.gpu.link-initialization-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as runtime]))

(defn- copy-plan []
  (let [kernel (artifact/make
                {:kernel-name "copy" :source "__kernel void copy(__global const float* x, __global float* y) {}"
                 :abi [(abi/slot 'x :input :float) (abi/slot 'y :output :float)]
                 :arguments '[x y]
                 :launch (launch/spec {:workgroup-size [1] :group-count [2]})
                 :effects {:kind :map :reads '[x] :writes '[y]}})
        descriptor {:dtype :float :all-params '[x] :array-params '[x] :scalar-params []
                    :array-roles {'x :input}
                    :allocs [{:sym 'y :dtype :float :size-fn (fn [_] 2)}]
                    :steps [{:phase :copy :kernel-name "copy" :convention :map :artifact kernel
                             :argument-specs [{:kind :input :sym 'x} {:kind :output :sym 'y}]}]
                    :result-sym 'y}
        node (fn [id role] (link/node {:id id :role role :device :ze:0 :dtype :float :shape [2]}))
        instance (fn [id x y] (link/instance {:id id :descriptor descriptor
                                             :bindings {'x x 'y y} :scalars {}}))]
    (link/make {:id :initialization :target :ze:0
                :nodes [(node :x :input) (node :unused :input)
                        (node :scratch :state) (node :out :output)]
                :instances [(instance :first :x :scratch) (instance :second :scratch :out)]
                :outputs [:out]})))

(deftest runtime-gates-use-local-preconditions-not-node-roles
  (let [replays (atom 0)
        uploads (atom 0)
        session (atom {:device-id :ze:0})]
    ;; Only backend operations are stubbed: real validation, instantiation, upload and run gates
    ;; remain active. This is a hardware-free contract test, not numerical kernel validation.
    (with-redefs [gpu/alloc! (fn [& _])
                  gpu/buffer-view (fn [_ key opts] (assoc opts :buffer-key key))
                  gpu/bind-step! (fn [& _])
                  gpu/record-graph! (fn [& _])
                  gpu/upload-range! (fn [& _] (swap! uploads inc))
                  gpu/replay! (fn [& _] (swap! replays inc))]
      (let [executable (runtime/instantiate! (copy-plan) {:session session})]
        (is (= #{:x} @(:pending-inputs executable))
            "unused input and locally produced state require no caller upload")
        (is (= :link-pending-inputs
               (try (runtime/run! executable) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (zero? @replays))
        (runtime/upload! executable :x (float-array [1 2]))
        (runtime/run! executable)
        (runtime/run! executable)
        (is (= 1 @uploads))
        (is (= 2 @replays)))
      (let [plan (update (copy-plan) :outputs conj :unused)
            executable (runtime/instantiate! plan {:session session})]
        (is (= #{:x :unused} @(:pending-inputs executable))
            "an exported pass-through input remains a real initialization requirement")))))
