(ns raster.gpu.parallel-program
  "Stage-once execution of a checked emitted parallel program call.

   Every numerical graph is bound before the first launch. Structured control is initially
   realized as one pre-bound graph per host iteration; all buffers remain resident and suffix
   equations consume the exact carry bindings selected by the call IR."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.structured-loop-call :as loop-call]))

(defn staging-plan
  "Return ordered graph bindings for a prepared program call without contacting a driver."
  [call execution-id]
  (let [call (program-call/validate! call)]
    (vec
     (mapcat
      (fn [step-index step]
        (cond
          (program-call/evaluated-host-equation? step)
          []

          (program-call/emitted-equation-call? step)
          [{:key [:parallel-program execution-id step-index]
            :graph (:graph step)
            :buffers (:buffers step)
            :scalar-values (:scalar-values step)}]

          (loop-call/structured-loop-call? step)
          (mapv (fn [iteration]
                  (let [{:keys [buffers scalar-values]}
                        (loop-call/iteration-binding step iteration)]
                    {:key [:parallel-program execution-id step-index iteration]
                     :graph (:graph step)
                     :buffers buffers
                     :scalar-values scalar-values}))
                (range (:trip-count step)))))
      (range) (:steps call)))))

(defn run-with!
  "Bind the complete call, then execute it through an injected graph executor.

   `executor` contains `:bind!`, `:run!`, and `:release!`. A staging failure releases all prior
   handles without launching; an execution failure releases the entire staged program."
  [call {:keys [bind! run! release!] :as executor}]
  (let [call (program-call/validate! call)]
    (doseq [[operation function] [[:bind! bind!] [:run! run!] [:release! release!]]]
      (when-not (ifn? function)
        (throw (ex-info "parallel program executor requires callable operations"
                        {:reason :parallel-program-executor
                         :operation operation :executor executor}))))
    (let [staged (volatile! [])]
      (try
        (doseq [{:keys [key graph buffers scalar-values]}
                (staging-plan call (random-uuid))]
          (vswap! staged conj (bind! key graph buffers scalar-values)))
        (doseq [handle @staged]
          (run! handle))
        (:outputs call)
        (finally
          (doseq [handle (rseq @staged)]
            (release! handle)))))))

(defn run!
  "Run a prepared emitted parallel program in one GPU session."
  [session call]
  (let [bind-graph! (requiring-resolve 'raster.gpu.core/bind-kernel-graph!)
        run-graph! (requiring-resolve 'raster.gpu.core/run-kernel-graph!)
        release-graph! (requiring-resolve 'raster.gpu.core/release-kernel-graph!)]
    (run-with!
     call
     {:bind! (fn [key graph buffers scalars]
               (bind-graph! session key graph buffers scalars))
      :run! (fn [handle] (run-graph! session handle))
      :release! (fn [handle] (release-graph! session handle))})))
