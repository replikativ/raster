(ns raster.compiler.passes.parallel.device
  "Device helpers for compiler backend selection and allocation policy.

  Device ids are concrete keywords such as :cpu:0, :cuda:0, :ze:0, and :ocl:0.
  The compiler uses these concrete ids directly rather than a second layer of
  :cpu/:cuda pseudo-devices."
  (:require [clojure.string :as str]))

(defn device-type
  "Extract device type (:cpu, :cuda, :ze, or :ocl) from a device-id keyword.
  E.g. :cpu:0 -> :cpu, :cuda:1 -> :cuda, :ze:0 -> :ze, :ocl:0 -> :ocl"
  [device-id]
  (when device-id
    (let [s (name device-id)]
      (cond
        (= s "cpu")                :cpu
        (= s "cuda")               :cuda
        (= s "ze")                 :ze
        (= s "ocl")                :ocl
        (str/starts-with? s "cpu")  :cpu
        (str/starts-with? s "cuda") :cuda
        (str/starts-with? s "ze")   :ze
        (str/starts-with? s "ocl")  :ocl
        :else (keyword (first (str/split s #":")))))))

(defn gpu-target?
  "True when device-id selects a GPU backend family."
  [device-id]
  (contains? #{:cuda :ze :ocl} (device-type device-id)))

(defn level-zero-target?
  "True when device-id selects a Level Zero target family."
  [device-id]
  (= :ze (device-type device-id)))

(defn preferred-dtype
  "Return the preferred default dtype for a target device, or nil when there is
  no device-specific preference."
  [device-id]
  (when (level-zero-target? device-id)
    :float))

;; ================================================================
;; Dtype-polymorphic allocation
;; ================================================================

(defn dtype->array-ctor
  "Return array constructor symbol for dtype."
  [dtype]
  (case dtype
    :double 'double-array
    :float  'float-array
    :long   'long-array
    :int    'int-array
    (throw (ex-info (str "dtype->array-ctor: unknown dtype `" dtype "`. Expected :double, :float, :long, or :int.")
                    {:dtype dtype}))))

(defn alloc-expr
  "Generate (ctor size) allocation S-expression for dtype."
  [dtype size-expr]
  (list (dtype->array-ctor dtype) size-expr))

;; ================================================================
;; Device-aware allocation
;; ================================================================

(defn select-backend
  "Select backend for a par form based on device and size hint.
  Returns :cuda, :opencl, :simd, or :scalar.

  device-id: keyword like :cpu:0, :cuda:0, :ze:0, or :ocl:0
  size-hint: known or estimated element count (may be nil)"
  [device-id size-hint]
  (let [dt (device-type (or device-id :cpu:0))]
    (cond
      (= :cuda dt)                         :cuda
      (= :ze dt)                           :opencl
      (= :ocl dt)                          :opencl
      (and size-hint (>= size-hint 8))     :simd
      (nil? size-hint)                     :simd
      :else                                :scalar)))

(defn select-runtime-backend
  "Choose the runtime backend from user options.

  target-device wins when present; otherwise simd? controls the CPU fallback."
  [target-device simd? size-hint]
  (if target-device
    (select-backend target-device size-hint)
    (if simd? :simd :scalar)))
