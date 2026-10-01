(ns raster-strict-gemm
  "Opt-in device-event counterpart to comparison/clblast_sgemm.cpp."
  (:require [raster.perf.production-canary :as canary]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]
            [raster.runtime.hardware :as hardware]))

(defn- dimension [text]
  (let [n (Long/parseLong text)]
    (when-not (<= 1 n 4096)
      (throw (ex-info "GEMM dimensions must be in 1..4096" {:dimension n})))
    n))

(defn- check-output! [^floats expected actual]
  (let [^floats actual (float-array actual)]
    (when-not (= (alength expected) (alength actual))
      (throw (ex-info "GEMM output length differs from oracle" {})))
    (when-not (every? #(Float/isFinite (float %)) actual)
      (throw (ex-info "GEMM produced a nonfinite output" {})))
    (let [error (reduce max 0.0
                        (map (fn [want got]
                               (Math/abs (- (double want) (double got))))
                             expected actual))]
      (when-not (and (Double/isFinite error) (<= error 1.0e-4))
        (throw (ex-info "GEMM differs from independent CPU oracle"
                        {:max-absolute-error error})))
      error)))

(defn- profile-duration-ns [profile field]
  (let [ms (get profile field)]
    (when-not (and (number? ms) (Double/isFinite (double ms)) (pos? ms)
                   (< (double ms) (/ Long/MAX_VALUE 1.0e6)))
      (throw (ex-info "GEMM profile requires a finite positive device duration"
                      {:field field :value ms})))
    (let [ns (long (* 1.0e6 ms))]
      (when-not (pos? ns)
        (throw (ex-info "GEMM profile duration is below one nanosecond"
                        {:field field :value ms})))
      ns)))

(defn benchmark! [shape]
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (<= 1 % 4096)) shape))
    (throw (ex-info "GEMM shape must contain three dimensions in 1..4096" {:shape shape})))
  (let [[m n k] shape]
    (when (> (*' m n k) 64000000)
      (throw (ex-info "reference work exceeds 64M products" {:shape shape})))
    (hardware/init!)
    (let [[a b :as args] (canary/gemm-arguments shape)
          expected (canary/gemm-reference a b shape)
          started (System/nanoTime)
          prepared (canary/prepare-gemm :ocl:0 args shape
                                       {:variant :plain :gemm-precision :f32-scalar
                                        :constants ['B]})
          compile-ns (- (System/nanoTime) started)
          started (System/nanoTime)
          instance (compiled/instantiate! prepared {:profile? true})
          bind-ns (- (System/nanoTime) started)]
      (try
        (let [resident (:executable instance)
              output (some #(when (= 'C (:sym %)) %) (:out-tree instance))
              _ (when-not output
                  (throw (ex-info "GEMM has no semantic C output" {})))
              _ (dotimes [_ 4] (link/profile! resident))
              pre-error (check-output! expected (link/download resident (:node output)))
              profiles (vec (repeatedly 12 #(link/profile! resident)))
              post-error (check-output! expected (link/download resident (:node output)))
              samples (mapv #(profile-duration-ns % :device-wall-ms) profiles)]
          {:baseline :raster-generated-contract
           :precision :strict-f32 :shape shape
           :device (hardware/device-signature :ocl:0)
           :clock :device-event-span :warmups 4 :samples-ns samples
           :measurement (into {} (measurement/summarize samples
                                   :timing-source :device-event
                                   :warmup-iterations 4))
           :kernel-total-ns (mapv #(profile-duration-ns % :kernel-total-ms) profiles)
           :kernel-names (mapv :kernel-name (:profile (first profiles)))
           :compile-ns compile-ns :bind-ns bind-ns
           :schedule-precision (get-in prepared [:schedule :precision])
           :compilation (canary/compilation-evidence prepared)
           :max-absolute-error (max pre-error post-error)})
        (finally (compiled/close! instance))))))

(defn -main [& dimensions]
  (when-not (= 3 (count dimensions))
    (throw (ex-info "usage: clojure -M:bench -m raster-strict-gemm M N K" {})))
  (prn (benchmark! (mapv dimension dimensions))))
