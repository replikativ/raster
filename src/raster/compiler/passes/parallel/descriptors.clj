(ns raster.compiler.passes.parallel.descriptors
  "Shared descriptors for explicit `raster.par/*` forms."
  (:require [clojure.set]
            [raster.compiler.ir.par :as par]))

(defn map-form-info
  "Return a normalized descriptor for `raster.par/map!` forms, or nil."
  [form]
  (when (par/par-map-form? form)
    (let [info (par/extract-par-map-info form)]
      (assoc info
             :type :map
             :form form
             :inputs (par/collect-aget-arrays (:body info))))))

(defn map-void-form-info
  "Return a normalized descriptor for `raster.par/map-void!` forms, or nil."
  [form]
  (when (par/par-map-void-form? form)
    (let [info (par/extract-par-map-void-info form)]
      (assoc info
             :type :map-void
             :form form
             :inputs (par/collect-aget-arrays (:body info))))))

(defn reduce-form-info
  "Return a normalized descriptor for `raster.par/reduce` forms, or nil."
  [form]
  (when (par/par-reduce-form? form)
    (let [info (par/extract-par-reduce-info form)]
      (assoc info
             :type :reduce
             :form form
             :inputs (par/collect-aget-arrays (:body info))))))

(defn map2-form-info
  "Return a normalized descriptor for `raster.par/map2!` forms, or nil."
  [form]
  (when (par/par-map2-form? form)
    (let [info (par/extract-par-map2-info form)]
      (assoc info
             :type :map2
             :form form
             :inputs (clojure.set/union
                      (par/collect-aget-arrays (:body1 info))
                      (par/collect-aget-arrays (:body2 info)))))))

(defn stencil-form-info
  "Return a normalized descriptor for `raster.par/stencil!` forms, or nil."
  [form]
  (when (par/par-stencil-form? form)
    (let [info (par/extract-par-stencil-info form)]
      (assoc info
             :type :stencil
             :form form
             :inputs (set (:in-arrays info))))))

(defn butterfly-form-info
  "Return a normalized descriptor for `raster.par/butterfly!` forms, or nil."
  [form]
  (when (par/par-butterfly-form? form)
    (let [info (par/extract-par-butterfly-info form)]
      (assoc info
             :type :butterfly
             :form form
             :inputs #{(:re info) (:im info) (:wr info) (:wi info)}))))

(defn par-form-info
  "Return a normalized descriptor for any explicit `raster.par/*` form, or nil."
  [form]
  (or (map-form-info form)
      (map2-form-info form)
      (map-void-form-info form)
      (reduce-form-info form)
      (stencil-form-info form)
      (butterfly-form-info form)))

(defn par-form?
  "Predicate for explicit `raster.par/*` forms recognized by this layer."
  [form]
  (boolean (par-form-info form)))

(defn bound-expr
  "Return the normalized bound expression for an explicit `raster.par/*` form."
  [form]
  (:bound (par-form-info form)))