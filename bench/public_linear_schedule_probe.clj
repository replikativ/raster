(ns public-linear-schedule-probe
  "Opt-in device-event comparison of schedules for one public nn/linear! program.

   This is an internal Raster schedule probe, not an external baseline and not a tuning
   promotion. It keeps compilation, binding, transfers and validation outside timed samples."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.nn :as nn]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.ocl-runtime :as ocl]))

(def ^:private compared-strategies
  [:portable-segred :xmx-direct :xmx-direct-tile-inputs])

(defn- checked-options!
  [{:keys [shape revision environment rounds warmup-rounds target residency]}]
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (pos? %)) shape)
                 (string? revision) (seq revision)
                 (string? environment) (seq environment)
                 (= :ocl:0 target)
                 (contains? #{:all-stages :constant-weights} residency)
                 (integer? rounds) (<= 2 rounds 120) (even? rounds)
                 (integer? warmup-rounds) (<= 0 warmup-rounds 30))
    (throw (ex-info "linear probe requires Arc OpenCL, positive [rows,in,out], identities and bounded rounds"
                    {:target target :shape shape :residency residency
                     :rounds rounds :warmup-rounds warmup-rounds})))
  (let [[rows in out] shape
        products (*' rows in out)
        bytes (*' 4 (+' (*' rows in) (*' out in) out (*' rows out)))]
    (when (or (> products 16000000) (> bytes (* 64 1024 1024)))
      (throw (ex-info "linear probe exceeds its host-work or logical-storage budget"
                      {:products products :bytes bytes}))))
  shape)

(defn- f16 [value]
  (double (Float/float16ToFloat (Float/floatToFloat16 (float value)))))

(defn- reference
  [^floats x ^floats weights ^floats bias rows in out]
  (vec
   (for [row (range rows) column (range out)]
     (float
      (+ (aget bias column)
         (reduce + 0.0
                 (for [k (range in)]
                   (* (f16 (aget x (+ (* row in) k)))
                      (f16 (aget weights (+ (* column in) k)))))))))))

(defn- runtime-arguments
  [candidate scalar-values]
  (mapv (fn [slot argument]
          (if (= :scalar (:kind slot))
            {:type (:dtype slot)
             :value (or (get scalar-values argument)
                        (throw (ex-info "linear probe lacks a scalar argument"
                                        {:argument argument :strategy (executable/strategy candidate)})))}
            argument))
        (executable/abi candidate) (executable/arguments candidate)))

(defn- constant-weight-plan
  [candidate target [rows in out] x weights bias scalar-values]
  (let [arguments (executable/arguments candidate)
        abi (executable/abi candidate)
        pointer-arguments (mapv second (filter #(not= :scalar (:kind (first %)))
                                               (map vector abi arguments)))
        scalar-arguments (mapv second (filter #(= :scalar (:kind (first %)))
                                              (map vector abi arguments)))
        descriptor
        {:dtype :float
         :all-params arguments
         :array-params pointer-arguments
         :array-roles {'W :constant 'x :input 'y :output 'b :constant}
         :scalar-params scalar-arguments :allocs [] :result-sym 'y
         :steps [{:convention :executable :artifact candidate :abi abi
                  :argument-specs (mapv (fn [slot argument]
                                          (if (= :scalar (:kind slot))
                                            {:kind :scalar :sym argument :type (:dtype slot)
                                             :value-fn (constantly (get scalar-values argument))}
                                            {:kind (:kind slot) :sym argument}))
                                        abi arguments)
                  :output 'y :phase (executable/strategy candidate)}]}]
    (link-plan/make
     {:id [:public-linear-residency (executable/strategy candidate)]
      :target target
      :nodes [(link-plan/node {:id :x :dtype :float :shape [rows in]
                               :device target :role :input :source x})
              (link-plan/node {:id :W :dtype :float :shape [out in]
                               :device target :role :constant :source weights})
              (link-plan/node {:id :b :dtype :float :shape [out]
                               :device target :role :constant :source bias})
              (link-plan/node {:id :y :dtype :float :shape [rows out]
                               :device target :role :output})]
      :instances [(link-plan/instance
                   {:id :linear :descriptor descriptor
                    :bindings {'W :W 'x :x 'y :y 'b :b}
                    :scalars (select-keys scalar-values scalar-arguments)})]
      :outputs [:y]})))

(defn run!
  "Compare portable, materialized XMX and tile-local-input schedules for public nn/linear!.

   Shape is [rows,input-width,output-width]. Inputs are exactly binary16-representable, allowing
   all three numerical policies to share an exact oracle. `:all-stages` includes materialized
   transforms on every replay; `:constant-weights` uses the ordinary LinkPlan role contract to
   hoist weight-only transforms into an untimed one-time prologue."
  [{:keys [shape revision environment rounds warmup-rounds target residency]
    :or {shape [32 256 256] rounds 12 warmup-rounds 4 target :ocl:0
         residency :all-stages}
    :as options}]
  (let [[rows in out] (checked-options!
                       (merge {:shape shape :rounds rounds :warmup-rounds warmup-rounds
                               :target target :residency residency}
                              options))
        x (float-array (map #(/ (- (mod % 13) 6) 8.0) (range (* rows in))))
        weights (float-array (map #(/ (- (mod % 11) 5) 8.0) (range (* out in))))
        bias (float-array (map #(/ (- (mod % 7) 3) 4.0) (range out)))
        expected (reference x weights bias rows in out)
        poison (float-array (repeat (* rows out) Float/NaN))
        descriptor (pipeline/compile-gpu-program #'nn/linear! target :dtype :float
                                                 :gemm-precision :mixed-f16-f32)
        choice (:dispatch (first (:steps descriptor)))
        candidates (mapv (fn [strategy]
                           (or (dispatch/alternative choice strategy)
                               (throw (ex-info "linear probe schedule is unavailable"
                                               {:strategy strategy
                                                :available (mapv executable/strategy
                                                                 (:alternatives choice))}))))
                         compared-strategies)
        scalar-values {'n (* rows out) 'batch rows 'in-f in 'out-f out}]
    (gpu/with-gpu-session [session target]
      (when (= :all-stages residency)
        (gpu/alloc! session {'x [:float (alength x) x]
                             'W [:float (alength weights) weights]
                             'b [:float (alength bias) bias]
                             'y [:float (* rows out) nil]}))
      (let [bound (atom [])]
        (try
          (doseq [candidate candidates]
            (let [strategy (executable/strategy candidate)
                  linked? (= :constant-weights residency)
                  started (System/nanoTime)
                  handle (if linked?
                            (link/instantiate!
                            (constant-weight-plan candidate target shape x weights bias scalar-values)
                            {:session session :profile? true})
                           (gpu/bind-kernel-executable!
                            session strategy candidate
                            (runtime-arguments candidate scalar-values) {:profile? true}))
                  poison! (if linked?
                            #(link/upload! handle :y poison)
                            #(gpu/upload! session 'y poison))
                  replay! (if linked?
                            #(link/run! handle)
                            #(gpu/run-kernel-graph! session handle))
                  profile! (if linked?
                             #(link/profile! handle)
                             #(gpu/profile-bound-kernel-graph! session handle))
                  download (if linked?
                             #(link/download handle :y)
                             #(gpu/download session 'y))
                  close! (if linked?
                           #(link/close! handle)
                           #(gpu/release-kernel-graph! session handle))
                  entry {:id strategy :handle handle :poison! poison! :profile! profile!
                         :close! close!
                         :bind-ms (/ (- (System/nanoTime) started) 1.0e6)
                         :signature (tuning/executable-signature candidate)
                         :kernel-count (count (executable/artifacts candidate))}]
              (swap! bound conj entry)
              (poison!)
              (replay!)
              (let [actual (vec (download))]
                (when-not (= expected actual)
                  (throw (ex-info "linear schedule failed the exact rounded-input oracle"
                                  {:strategy strategy :expected-head (take 8 expected)
                                   :actual-head (take 8 actual)}))))))
          {:kind :public-linear-schedule-comparison
           :version 2 :shape shape :revision revision :environment environment
           :target target :device (ocl/selected-device-info)
           :input-recipe {:activation :index-mod13-minus6-over8
                          :weight :index-mod11-minus5-over8
                          :bias :index-mod7-minus3-over4}
           :validation {:passed? true :comparison :exact
                        :oracle :host-binary16-rounded-dot-plus-f32-bias
                        :every-candidate? true}
           :scope {:public-compiler-path? true :timing-source :device-event
                   :cache-state :warm-resident :transfers-included? false
                   :residency residency
                   :materialized-weight-transform
                   (if (= :constant-weights residency)
                     :one-time-prologue-excluded
                     :included-every-replay)
                   :one-time-initialization-included? false
                   :validation-included? false :promotion? false}
           :candidates (mapv #(dissoc % :handle :poison! :profile! :close!) @bound)
           :comparison
           (update
            (measurement/measure-interleaved!
             (mapv (fn [{:keys [id poison! profile!]}]
                     {:id id
                      :sample-fn
                      (fn []
                        (poison!)
                        (let [profile (profile!)
                              duration (:device-wall-ms profile)]
                          (when-not (and (number? duration)
                                         (Double/isFinite (double duration))
                                         (not (neg? (double duration))))
                            (throw (ex-info "linear probe requires a finite device-event span"
                                            {:strategy id :profile profile})))
                          (* 1.0e6 duration)))})
                   @bound)
             :rounds rounds :warmup-rounds warmup-rounds
             :timing-source :device-event)
            :measurements #(update-vals % (fn [measurement] (into {} measurement))))}
          (finally
            (doseq [{:keys [close!]} (reverse @bound)]
              (close!))))))))
