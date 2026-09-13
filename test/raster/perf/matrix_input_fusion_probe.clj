(ns raster.perf.matrix-input-fusion-probe
  "Opt-in device replay oracle and paired timing for staged versus fused matrix input conversion."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.gpu.core :as gpu]
            [raster.gpu.measurement :as measurement]
            [raster.perf.production-canary :as canary]))

(defn run!
  "Check both generated schedules with changing activations and poisoned output on one device.
   Tiny fixed geometry includes a partial M tile. Failures propagate; unavailable hardware is
   not reported as a passing or skipped benchmark. Device allocations are session-owned."
  ([] (run! :ocl:0))
  ([device] (run! device {}))
  ([device {:keys [timing? rounds warmup-rounds]
            :or {timing? false rounds 12 warmup-rounds 4}}]
   (when-not (and (boolean? timing?) (integer? rounds) (<= 2 rounds 120) (even? rounds)
                  (integer? warmup-rounds) (<= 0 warmup-rounds 120))
     (throw (ex-info "probe requires bounded even rounds and nonnegative warmup rounds"
                     {:rounds rounds :warmup-rounds warmup-rounds :timing? timing?})))
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
                     (gemm/emit-matrix-input-fusion-alternative spec)]
         live (atom []) profiles (atom [])]
     (gpu/with-gpu-session [sess device]
       (gpu/alloc! sess {:a [:float (* m k) a] :b [:float (* k n) b]
                         :c [:float (* m n) poison]})
       (try
         (let [samplers
               (mapv (fn [g]
                (let [strategy (get-in g [:attributes :strategy])
                      handle (gpu/bind-kernel-graph! sess strategy g {'A :a 'B :b 'C :c} {}
                                                     {:profile? timing?})
                      counter (atom -1)]
                  (swap! live conj handle)
                  {:id strategy
                   :sample-fn
                   (fn []
                    (let [iteration (swap! counter inc) input-index (mod iteration 2)]
                      (gpu/upload! sess :a (nth activations input-index))
                      (gpu/upload! sess :c poison)
                      (let [duration
                            (if timing?
                              (let [profile (gpu/profile-bound-kernel-graph! sess handle)
                                    span (:device-wall-ms profile)
                                    phase (cond (< iteration 4) :validation
                                                (< iteration (+ 4 warmup-rounds)) :warmup
                                                :else :measurement)]
                                (when-not (and (number? span) (Double/isFinite (double span))
                                               (not (neg? (double span))))
                                  (throw (ex-info "probe requires a finite device span" {:profile profile})))
                                (swap! profiles conj {:candidate strategy :replay-index iteration
                                                      :sampling-phase phase :input-index input-index
                                                      :sample-index (when (= :measurement phase)
                                                                      (- iteration 4 warmup-rounds))
                                                      :profile profile})
                                (* 1.0e6 span))
                              (do (gpu/run-kernel-graph! sess handle) 0.0))
                            actual (vec (gpu/download sess :c))]
                        (when-not (= (nth expected input-index) actual)
                          (throw (ex-info "matrix input fusion replay differs from independent host reference"
                                          {:strategy strategy :iteration iteration
                                           :expected-head (take 8 (nth expected input-index))
                                           :actual-head (take 8 actual)})))
                        duration)))})) candidates)
               _ (doseq [{:keys [sample-fn]} samplers _ (range 4)] (sample-fn))
               comparison (when timing?
                            (measurement/measure-interleaved!
                             samplers :rounds rounds :warmup-rounds warmup-rounds
                             :timing-source :device-event))
               replays (+ 4 (if timing? (+ rounds warmup-rounds) 0))]
           (cond->
            {:device device :shape shape :comparison :exact :timing? timing?
             :candidates (mapv (fn [g]
                                {:strategy (get-in g [:attributes :strategy])
                                 :passed? true :replays replays :checked-elements (* replays m n)
                                 :stages (mapv (comp last :id) (:nodes g))
                                 :temporaries (mapv (comp last :id) (:temporaries g))}) candidates)}
             timing? (assoc :replay-profiles @profiles
                            :scope {:timing-source :device-event :public-deftm-path? false
                                    :weight-conversion-included? true :transfers-included? false
                                    :validation-included? false :promotion? false}
                            :measurement (update comparison :measurements
                                                 #(update-vals % (fn [m] (into {} m)))))))
         (finally (doseq [handle (reverse @live)] (gpu/release-kernel-graph! sess handle))))))))
