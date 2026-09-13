(ns raster.perf.matrix-input-fusion-probe
  "Opt-in device replay oracle and paired timing for staged versus fused matrix input conversion."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]
            [raster.perf.production-canary :as canary]))

(def ^:private tiles
  {:single-fragment {:block-m 16 :block-n 32 :sg-m 8 :sg-n 16 :block-k 32 :num-stages 1
                     :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}
   :multi-fragment {:block-m 32 :block-n 64 :sg-m 16 :sg-n 32 :block-k 32 :num-stages 2
                    :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}})

(defn rounded-inputs
  "Independent Java binary16 round-to-nearest-even oracle, not compiler scalar lowering."
  [values]
  (float-array (map #(Float/float16ToFloat (Float/floatToFloat16 (float %))) values)))

(defn constant-weight-plan
  "Use ordinary LinkPlan roles and initialization to hoist weight-only representation stages."
  [g device [m n k] variant a b]
  (let [descriptor {:dtype :float :all-params '[A B C] :array-params '[A B C]
                    :array-roles {'A :input 'B :constant 'C :output}
                    :scalar-params [] :allocs [] :result-sym 'C
                    :steps [{:convention :executable :artifact g :abi (:abi g)
                             :argument-specs (mapv (fn [slot arg] {:kind (:kind slot) :sym arg})
                                                   (:abi g) (:arguments g))
                             :output 'C :phase :matrix}]}]
    (link-plan/make
     {:id :matrix-input-constant-weights :target device
      :nodes [(link-plan/node {:id :a :dtype :float :shape [m k] :device device :role :input :source a})
              (link-plan/node {:id :b :dtype :float :shape (if (= :nt variant) [n k] [k n])
                               :device device :role :constant :source b})
              (link-plan/node {:id :c :dtype :float :shape [m n] :device device :role :output})]
      :instances [(link-plan/instance {:id :matrix :descriptor descriptor
                                      :bindings {'A :a 'B :b 'C :c} :scalars {}})]
      :outputs [:c]})))

(defn run!
  "Check both generated schedules with changing activations and poisoned output on one device.
   Bounded geometry defaults to a partial M tile. Failures propagate; unavailable hardware is
   not reported as a passing or skipped benchmark. Device allocations are session-owned."
  ([] (run! :ocl:0))
  ([device] (run! device {}))
  ([device {:keys [timing? rounds warmup-rounds shape tile-policy input-policy variant residency]
            :or {timing? false rounds 12 warmup-rounds 4 shape [13 32 32]
                 tile-policy :single-fragment input-policy :dyadic variant :nn residency :all-stages}}]
   (when-not (and (boolean? timing?) (integer? rounds) (<= 2 rounds 120) (even? rounds)
                  (integer? warmup-rounds) (<= 0 warmup-rounds 120))
     (throw (ex-info "probe requires bounded even rounds and nonnegative warmup rounds"
                     {:rounds rounds :warmup-rounds warmup-rounds :timing? timing?})))
   (when-not (and (vector? shape) (= 3 (count shape))
                  (every? #(and (integer? %) (<= 1 % 4096)) shape)
                  (<= (apply *' shape) 8388608)
                  (contains? tiles tile-policy) (contains? #{:dyadic :half-rounding-ties} input-policy)
                  (contains? #{:nn :nt} variant)
                  (contains? #{:all-stages :constant-weights} residency))
     (throw (ex-info "probe geometry, work budget, tile or input policy is unsupported"
                     {:shape shape :tile-policy tile-policy :input-policy input-policy :variant variant
                      :residency residency})))
   (let [[m n k] shape
         [a b] (canary/gemm-arguments shape)
         a (if (= :half-rounding-ties input-policy)
             (float-array (map #(+ (double %) (/ 1.0 4096.0)) a)) a)
         resident-b (if (= :nt variant)
                      (float-array (for [j (range n) p (range k)] (aget ^floats b (+ (* p n) j)))) b)
         activations [a (float-array (map - a))]
         expected (mapv #(vec (canary/gemm-reference (rounded-inputs %) (rounded-inputs b) shape)) activations)
         poison (float-array (* m n) Float/NaN)
         spec {:id :input-fusion-replay :a 'A :b 'B :c 'C :m m :n n :k k :variant variant
               :fill-workgroups 16
               :tile (get tiles tile-policy)}
         candidates [(first (:alternatives (gemm/emit-matrix-alternatives spec)))
                     (gemm/emit-matrix-input-fusion-alternative spec)]
         live (atom []) profiles (atom [])]
     (gpu/with-gpu-session [sess device]
       (when (= :all-stages residency)
         (gpu/alloc! sess {:a [:float (* m k) a] :b [:float (* k n) resident-b]
                           :c [:float (* m n) poison]}))
       (try
         (let [samplers
               (mapv (fn [g]
                (let [strategy (get-in g [:attributes :strategy])
                      linked? (= :constant-weights residency)
                      handle (if linked?
                               (link/instantiate! (constant-weight-plan g device shape variant a resident-b)
                                                  {:session sess :profile? timing?})
                               (gpu/bind-kernel-graph! sess strategy g {'A :a 'B :b 'C :c} {}
                                                      {:profile? timing?}))
                      upload! (if linked? #(link/upload! handle %1 %2) #(gpu/upload! sess %1 %2))
                      download! (if linked? #(link/download handle %) #(gpu/download sess %))
                      profile! (if linked? #(link/profile! handle) #(gpu/profile-bound-kernel-graph! sess handle))
                      replay! (if linked? #(link/run! handle) #(gpu/run-kernel-graph! sess handle))
                      counter (atom -1)]
                  (swap! live conj (if linked? #(link/close! handle) #(gpu/release-kernel-graph! sess handle)))
                  {:id strategy
                   :sample-fn
                   (fn []
                    (let [iteration (swap! counter inc) input-index (mod iteration 2)]
                      (upload! :a (nth activations input-index))
                      (upload! :c poison)
                      (let [duration
                            (if timing?
                              (let [profile (profile!)
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
                              (do (replay!) 0.0))
                            actual (vec (download! :c))]
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
             :variant variant :tile-policy tile-policy :tile (:tile spec) :input-policy input-policy
             :residency residency
             :oracle :java-rne-binary16-inputs-host-double-dot-float-output
             :candidates (mapv (fn [g]
                                {:strategy (get-in g [:attributes :strategy])
                                 :passed? true :replays replays :checked-elements (* replays m n)
                                 :stage-scope :compiled-graph
                                 :stages (mapv (comp last :id) (:nodes g))
                                 :profiled-replay-kernels
                                 (when timing?
                                   (mapv :kernel-name
                                         (get-in (first (filter #(= (get-in g [:attributes :strategy])
                                                                    (:candidate %)) @profiles))
                                                 [:profile :profile])))
                                 :temporaries (mapv (comp last :id) (:temporaries g))}) candidates)}
             timing? (assoc :replay-profiles @profiles
                            :scope {:timing-source :device-event :public-deftm-path? false
                                    :weight-conversion-included? (= :all-stages residency)
                                    :one-time-initialization-included? false
                                    :transfers-included? false
                                    :validation-included? false :promotion? false}
                            :measurement (update comparison :measurements
                                                 #(update-vals % (fn [m] (into {} m)))))))
         (finally (doseq [close! (reverse @live)] (close!))))))))
