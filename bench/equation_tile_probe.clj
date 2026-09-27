(ns equation-tile-probe
  "Opt-in paired device-event comparison of public equation-first NN schedules.
   No timing assertion, tuning promotion or new kernel implementation. Load in a :dev REPL."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]
            [raster.perf.production-canary :as canary]
            [raster.runtime.hardware :as hardware]))

(defn run!
  [{:keys [target shape revision environment rounds warmup-rounds]
    :or {target :ocl:0 shape [8 256 256] rounds 8 warmup-rounds 3}}]
  (when-not (and (string? revision) (seq revision)
                 (string? environment) (seq environment))
    (throw (ex-info "benchmark requires revision and environment provenance" {})))
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (pos? %)) shape)
                 (<= (reduce *' shape) (* 64 1024 1024))
                 (let [[m n k] shape]
                   (<= (+' (*' m k) (*' k n) (*' 3 m n)) (* 4 1024 1024))))
    (throw (ex-info "probe exceeds bounded shape/work budget" {:shape shape})))
  (let [[a b _ :as arrays] (canary/gemm-arguments shape)
        expected (vec (canary/gemm-reference a b shape))
        bound (atom [])
        failure (volatile! nil)]
    (try
      (doseq [strategy [:portable :register-tiled]]
        (let [args (into (assoc arrays 2 (float-array (count expected))) (map long shape))
              started (System/nanoTime)
              prepared (compiled/lower
                        #'canary/gemm-mnk! args
                        {:compiler :equation-first :target target :dtype :float
                         :constants '[A B] :outputs '[C]
                         :schedule {:precision :mixed-f16-f32
                                    :typed-contraction {:strategy strategy}}})
              prepare-ns (- (System/nanoTime) started)
              started (System/nanoTime)
              live (compiled/instantiate! prepared {:profile? true})
              bind-ns (- (System/nanoTime) started)]
          ;; After instantiate! returns, own cleanup even if validation/metadata capture fails.
          (swap! bound conj {:id strategy :live live})
          (let [profile (compiled/profile live)
                actual (vec (get-in profile [:result :C]))]
            (when-not (= expected actual)
              (throw (ex-info "equation-first tile failed independent CPU oracle"
                              {:strategy strategy :shape shape
                               :expected (take 8 expected) :actual (take 8 actual)}))))
          (swap! bound update (dec (count @bound)) merge
                 {:prepare-ns prepare-ns :bind-ns bind-ns
                  :template (get (compiled/preparation-report prepared) :template)
                  :execution (compiled/execution-info live)
                  :signatures
                  (mapv (comp tuning/executable-signature :graph)
                        (filter program-call/emitted-equation-call?
                                (get-in (compiled/plan prepared) [:instances 0 :call :steps])))})))
      {:kind :equation-first-nn-schedule-comparison
       :version 1 :revision revision :environment environment
       :target target :device (hardware/device-signature target) :shape shape
       :validation {:passed? true :oracle :independent-cpu-double-dot
                    :inputs :deterministic-dyadic-f32 :comparison :exact}
       :scope {:storage :float :accumulator :float :target-contraction-permitted? true
               :timing-source :device-event :cache-state :warm-resident
               :transfers-included? false :compilation-included? false
               :validation-included? false :promotion? false}
       :candidates (mapv #(dissoc % :live) @bound)
       :comparison
       (measurement/measure-interleaved!
        (mapv (fn [{:keys [id live]}]
                {:id id
                 :sample-fn (fn []
                              (* 1.0e6 (double (:device-wall-ms
                                                (link/profile! (:executable live))))))})
              @bound)
        :rounds rounds :warmup-rounds warmup-rounds)}
      (catch Throwable error
        (vreset! failure error)
        (throw error))
      (finally
        (let [cleanup-errors
              (reduce (fn [errors {:keys [live]}]
                        (try (compiled/close! live) errors
                             (catch Throwable error (conj errors error))))
                      [] (reverse @bound))]
          (when-let [primary (or @failure (first cleanup-errors))]
            (doseq [error (if @failure cleanup-errors (rest cleanup-errors))]
              (when-not (identical? primary error) (.addSuppressed ^Throwable primary error)))
            (when-not @failure (throw primary))))))))
