(ns raster.gpu
  "GPU computing — public entry point.

  Provides a unified session layer that auto-detects backend
  (Level Zero for Intel, OpenCL for portable) and handles
  compilation, memory management, and kernel dispatch.

  Usage:
    (require '[raster.gpu :as gpu])
    (gpu/with-gpu-session [sess]
      (gpu/compile! sess ...)
      (gpu/alloc! sess :buf 1024 :double)
      (gpu/upload! sess :buf my-array)
      (gpu/invoke! sess :kernel :buf)
      (gpu/download sess :buf))

  Sub-namespaces for backend-specific access:
    raster.gpu.core        — unified session API
    raster.gpu.ze-runtime  — Intel Level Zero (Panama FFM)
    raster.gpu.ocl-runtime — OpenCL ICD (Panama FFM)"
  (:require [raster.gpu.core :as core]
            [raster.support :refer [import-vars]]))

;; ================================================================
;; Session lifecycle
;; ================================================================

(import-vars raster.gpu.core
             make-session close-session!
             with-gpu-session)

;; ================================================================
;; Compilation
;; ================================================================

(import-vars raster.gpu.core
             compile! compile-phases!)

;; ================================================================
;; Memory management
;; ================================================================

(import-vars raster.gpu.core
             alloc! free-buffer!
             upload! download
             buffer sync-to-arrays!)

;; ================================================================
;; Kernel execution
;; ================================================================

(import-vars raster.gpu.core
             invoke! invoke-scan! invoke-rng-fill! invoke-active-ids!
             kernel)
