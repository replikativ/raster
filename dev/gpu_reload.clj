(ns gpu-reload
  "One-command ordered reload of the GPU compile pipeline.

   WHY: `require ... :reload` reloads ONLY the named namespace, not its dependencies. The GPU
   kernel generators (c-emit, par-opencl, segop-opencl) sit DEEP under pipeline/gpu.core, so
   reloading just pipeline + gpu.core after editing a generator leaves the generator STALE in the
   REPL — producing kernels from old code while the surface looks updated (the recurring
   'works only on a fresh REPL' confusion: a stale par-opencl dropped the fp64 pragma, etc.).

   FIX: reload the whole chain LEAVES-FIRST in dependency order, so each namespace re-requires
   already-reloaded deps and every edit is picked up. Then reset the GPU runtime for clean device
   state. Load once per REPL with (load-file \"dev/gpu_reload.clj\"); call (gpu-reload/reload!)
   after any edit to the GPU path — no REPL restart needed.")

(def gpu-pipeline-nses
  "GPU compile-pipeline namespaces, LEAVES FIRST (deps before dependents). Reloading in this
   order means each ns sees the latest of everything it depends on."
  '[raster.compiler.backend.gpu.c-emit
    raster.compiler.backend.gpu.opencl-codegen
    raster.compiler.backend.gpu.par-opencl
    raster.compiler.ir.soac
    raster.compiler.passes.parallel.soac-lower
    raster.compiler.backend.gpu.segop-opencl
    raster.compiler.backend.gpu.opencl-pass
    raster.compiler.passes.scalar.dce
    raster.compiler.pipeline
    raster.gpu.core])

(defn reload!
  "Reload the GPU compile pipeline leaves-first + reset the GPU runtime. Returns the reloaded
   namespace count. opts :ze-runtime? true also reloads ze-runtime (do this only after editing it
   — it clears the kernel registry and bounces the device); :reset? false skips the GPU reset."
  ([] (reload! {}))
  ([{:keys [ze-runtime? reset?] :or {reset? true}}]
   (let [nses (cond-> gpu-pipeline-nses
                ze-runtime? (-> vec (conj 'raster.gpu.ze-runtime)))]
     (doseq [n nses] (require n :reload))
     (when reset? ((requiring-resolve 'raster.gpu.ze-runtime/reset!)))
     {:reloaded (count nses) :nses nses})))
