(ns raster.gpu.structured-loop
  "Host-repeated execution of a checked structured-loop call.

   This is the correctness schedule: each iteration binds and runs the ordinary emitted
   KernelGraph with the call plan's carry/scalar projection. Later replay caching and persistent
   workgroup schedules must preserve this behavior."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.structured-loop-call :as loop-call]))

(defn run-with!
  "Execute `call` through an injected graph executor.

   `executor` contains `:bind!`, `:run!`, and `:release!`. Bind receives a unique logical key, the
   emitted graph, buffer bindings, and typed scalar bindings. Every successfully bound iteration
   is released even when execution fails. This injection keeps sequencing and lifetime semantics
   hardware-free testable."
  [call {:keys [bind! run! release!] :as executor}]
  (let [call (loop-call/validate! call)]
    (doseq [[key function] [[:bind! bind!] [:run! run!] [:release! release!]]]
      (when-not (ifn? function)
        (throw (ex-info "structured loop executor requires callable operations"
                        {:reason :structured-loop-executor :operation key
                         :executor executor}))))
    (let [execution-id (random-uuid)]
      (dotimes [index (:trip-count call)]
        (let [{:keys [buffers scalar-values]} (loop-call/iteration-binding call index)
              handle (bind! [:structured-loop execution-id index]
                            (:graph call) buffers scalar-values)]
          (try
            (run! handle)
            (finally
              (release! handle))))))
    (:outputs call)))

(defn run!
  "Run a StructuredLoopCall in one GPU session through ordinary KernelGraph operations."
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
