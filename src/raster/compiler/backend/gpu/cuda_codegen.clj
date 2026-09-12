(ns raster.compiler.backend.gpu.cuda-codegen
  "CUDA-C entry point for verified direct matrix KernelBody values.
   Admission, fragment mainloop spelling and uniform typed epilogues share the fragment backend."
  (:require [raster.compiler.backend.gpu.matrix-fragment-source :as fragment-source]))

(defn emit-matrix-kernel
  "Lower the supported CUDA WMMA subset of a verified matrix KernelBody.
   No tile, dimension, launch or ABI side channel is accepted."
  [kernel-name kernel-body]
  (fragment-source/emit-matrix-kernel kernel-name kernel-body :cuda))
