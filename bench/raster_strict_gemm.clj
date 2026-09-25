(ns raster-strict-gemm
  "Opt-in device-event counterpart to comparison/clblast_sgemm.cpp."
  (:require [raster.perf.production-canary :as canary]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
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
    (let [error (reduce max 0.0
                        (map (fn [want got]
                               (Math/abs (- (double want) (double got))))
                             expected actual))]
      (when (> error 1.0e-4)
        (throw (ex-info "GEMM differs from independent CPU oracle"
                        {:max-absolute-error error})))
      error)))

(defn benchmark! [shape]
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
              samples (mapv #(long (* 1.0e6 (:device-wall-ms %))) profiles)]
          (when-not (every? pos? samples)
            (throw (ex-info "device-event span is missing or nonpositive"
                            {:samples-ns samples})))
          {:baseline :raster-generated-contract
           :precision :strict-f32 :shape shape
           :device (hardware/device-signature :ocl:0)
           :clock :device-event-span :warmups 4 :samples-ns samples
           :kernel-total-ns (mapv #(long (* 1.0e6 (:kernel-total-ms %))) profiles)
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
