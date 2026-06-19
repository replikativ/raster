(ns sandbox.core
  "Entry point for the browser sandbox. Runs the WebGPU compute experiment and
   the WebGL2 render experiment. See node/wasm_dotproduct.mjs for the wasm
   codegen experiment (runs under node, no browser needed)."
  (:require [sandbox.webgpu-compute :as gpu]
            [sandbox.webgl-triangle :as gl]
            [sandbox.wasm-interop :as wasm]))

(defn init []
  (js/console.log "raster cljs sandbox: init")
  (wasm/run!)
  (gpu/run!)
  (gl/run!))
