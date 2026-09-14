(ns public-linear-schedule-probe
  "Opt-in device-event comparison of schedules for one public nn/linear! program.

   This is an internal Raster schedule probe, not an external baseline and not a tuning
   promotion. It keeps compilation, binding, transfers and validation outside timed samples."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.nn :as nn]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.ocl-runtime :as ocl]))

(def ^:private compared-strategies
  [:portable-segred :xmx-direct :xmx-direct-tile-inputs])

(defn- checked-options!
  [{:keys [shape revision environment rounds warmup-rounds target]}]
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (pos? %)) shape)
                 (string? revision) (seq revision)
                 (string? environment) (seq environment)
                 (= :ocl:0 target)
                 (integer? rounds) (<= 2 rounds 120) (even? rounds)
                 (integer? warmup-rounds) (<= 0 warmup-rounds 30))
    (throw (ex-info "linear probe requires Arc OpenCL, positive [rows,in,out], identities and bounded rounds"
                    {:target target :shape shape :rounds rounds :warmup-rounds warmup-rounds})))
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

(defn run!
  "Compare portable, materialized XMX and tile-local-input schedules for public nn/linear!.

   Shape is [rows,input-width,output-width]. Inputs are exactly binary16-representable, allowing
   all three numerical policies to share an exact oracle. The materialized schedule includes its
   cast/transpose stages on every replay; constant-weight initialization is a separate experiment."
  [{:keys [shape revision environment rounds warmup-rounds target]
    :or {shape [32 256 256] rounds 12 warmup-rounds 4 target :ocl:0}
    :as options}]
  (let [[rows in out] (checked-options!
                       (merge {:shape shape :rounds rounds :warmup-rounds warmup-rounds
                               :target target}
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
      (gpu/alloc! session {'x [:float (alength x) x]
                           'W [:float (alength weights) weights]
                           'b [:float (alength bias) bias]
                           'y [:float (* rows out) nil]})
      (let [bound (atom [])
            poison! #(gpu/upload! session 'y poison)]
        (try
          (doseq [candidate candidates]
            (let [strategy (executable/strategy candidate)
                  started (System/nanoTime)
                  handle (gpu/bind-kernel-executable!
                          session strategy candidate (runtime-arguments candidate scalar-values)
                          {:profile? true})]
              (swap! bound conj {:id strategy :handle handle
                                 :bind-ms (/ (- (System/nanoTime) started) 1.0e6)
                                 :signature (tuning/executable-signature candidate)
                                 :kernel-count (count (executable/artifacts candidate))})
              (poison!)
              (gpu/run-kernel-graph! session handle)
              (let [actual (vec (gpu/download session 'y))]
                (when-not (= expected actual)
                  (throw (ex-info "linear schedule failed the exact rounded-input oracle"
                                  {:strategy strategy :expected-head (take 8 expected)
                                   :actual-head (take 8 actual)}))))))
          {:kind :public-linear-schedule-comparison
           :version 1 :shape shape :revision revision :environment environment
           :target target :device (ocl/selected-device-info)
           :input-recipe {:activation :index-mod13-minus6-over8
                          :weight :index-mod11-minus5-over8
                          :bias :index-mod7-minus3-over4}
           :validation {:passed? true :comparison :exact
                        :oracle :host-binary16-rounded-dot-plus-f32-bias
                        :every-candidate? true}
           :scope {:public-compiler-path? true :timing-source :device-event
                   :cache-state :warm-resident :transfers-included? false
                   :materialized-weight-transform :included-every-replay
                   :validation-included? false :promotion? false}
           :candidates (mapv #(dissoc % :handle) @bound)
           :comparison
           (update
            (gpu/measure-bound-kernel-graphs-interleaved!
             session
             (mapv #(assoc (select-keys % [:id :handle]) :before-sample! poison!) @bound)
             :rounds rounds :warmup-rounds warmup-rounds)
            :measurements #(update-vals % (fn [measurement] (into {} measurement))))}
          (finally
            (doseq [{:keys [handle]} (reverse @bound)]
              (gpu/release-kernel-graph! session handle))))))))
