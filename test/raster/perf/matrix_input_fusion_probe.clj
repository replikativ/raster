(ns raster.perf.matrix-input-fusion-probe
  "Opt-in device replay oracle for staged versus fused matrix input conversion. No timing claims."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.gpu.core :as gpu]
            [raster.perf.production-canary :as canary]))

(defn run!
  "Check both generated schedules with changing activations and poisoned output on one device.
   Tiny fixed geometry includes a partial M tile. Failures propagate; unavailable hardware is
   not reported as a passing or skipped benchmark. Device allocations are session-owned."
  ([] (run! :ocl:0))
  ([device]
   (let [shape [13 32 32] [m n k] shape
         [a b] (canary/gemm-arguments shape)
         activations [a (float-array (map - a))]
         expected (mapv #(vec (canary/gemm-reference % b shape)) activations)
         poison (float-array (* m n) Float/NaN)
         spec {:id :input-fusion-replay :a 'A :b 'B :c 'C :m m :n n :k k :variant :nn
               :fill-workgroups 16
               :tile {:block-m 16 :block-n 32 :sg-m 8 :sg-n 16 :block-k 32 :num-stages 1
                      :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}}
         candidates [(first (:alternatives (gemm/emit-matrix-alternatives spec)))
                     (gemm/emit-matrix-input-fusion-alternative spec)]]
     (gpu/with-gpu-session [sess device]
       (gpu/alloc! sess {:a [:float (* m k) a] :b [:float (* k n) b]
                         :c [:float (* m n) poison]})
       {:device device :shape shape :comparison :exact :timing? false
        :candidates
        (mapv (fn [g]
                (let [strategy (get-in g [:attributes :strategy])
                      handle (gpu/bind-kernel-graph! sess strategy g {'A :a 'B :b 'C :c} {})]
                  (try
                    (doseq [iteration (range 4)
                            :let [input-index (mod iteration 2)]]
                      (gpu/upload! sess :a (nth activations input-index))
                      (gpu/upload! sess :c poison)
                      (gpu/run-kernel-graph! sess handle)
                      (let [actual (vec (gpu/download sess :c))]
                        (when-not (= (nth expected input-index) actual)
                          (throw (ex-info "matrix input fusion replay differs from independent host reference"
                                          {:strategy strategy :iteration iteration
                                           :expected-head (take 8 (nth expected input-index))
                                           :actual-head (take 8 actual)})))))
                    {:strategy strategy :passed? true :replays 4 :checked-elements (* 4 m n)
                     :stages (mapv (comp last :id) (:nodes g))
                     :temporaries (mapv (comp last :id) (:temporaries g))}
                    (finally (gpu/release-kernel-graph! sess handle))))) candidates)}))))
