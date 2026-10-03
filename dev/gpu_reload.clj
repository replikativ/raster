(ns gpu-reload
  "One-command ordered reload of the GPU compile pipeline.

   WHY: `require ... :reload` reloads ONLY the named namespace, not its dependencies. The GPU
   kernel generators (c-emit, par-opencl, segop-opencl) sit DEEP under pipeline/gpu.core, so
   reloading just pipeline + gpu.core after editing a generator leaves the generator STALE in the
   REPL — producing kernels from old code while the surface looks updated (the recurring
   'works only on a fresh REPL' confusion: a stale par-opencl dropped the fp64 pragma, etc.).

   FIX: reload the whole chain LEAVES-FIRST in dependency order, so each namespace re-requires
   already-reloaded deps and every edit is picked up. Load once per REPL with
   (load-file \"dev/gpu_reload.clj\"). Native runtime namespaces must not be reloaded over live
   owners; restart the REPL process after native representation changes. Live runtime reset
   currently fails closed until the complete child-resource lease contract lands.")

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
  "Reload the compiler pipeline leaves-first. Native runtime reload is rejected before changes;
   use a process restart. :reset? true requests checked reset (currently fail-closed for live roots)."
  ([] (reload! {}))
  ([{:keys [ze-runtime? reset?] :or {reset? false}}]
   (when ze-runtime?
     (throw (ex-info "Native runtime reload would discard live ownership; restart the REPL process"
                     {:reason :runtime-reload-requires-process-restart})))
   ;; Admit an explicitly requested reset before reloading any compiler namespaces.
   (when reset? ((requiring-resolve 'raster.gpu.ze-runtime/reset!)))
   (let [nses gpu-pipeline-nses]
     (doseq [n nses] (require n :reload))
     {:reloaded (count nses) :nses nses})))
