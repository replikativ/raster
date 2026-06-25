(ns raster.compiler.passes.scalar.validate-types
  "Type-completeness validator for the compiled IR.

  Checks invariants that catch silent type-loss bugs:
   1. Helper methods extracted by closure/hoist-and-compile shouldn't have
      Object-typed parameters (those become megamorphic IFn dispatch sites
      at runtime, killing inlining + SIMD).
   2. par/map!/par/reduce/par/scan! forms that survived to the backend
      should carry :raster.type/elem-type meta (else SIMD codegen falls
      back to :double regardless of the actual array type — the f32
      [F → [D cast bug shape).

  Currently surfaces violations as warnings (default) or hard errors when
  raster.strict.types=true is set. Used by the pipeline to convert
  silent regressions into compile-time signals."
  (:require [clojure.walk :as walk]
            [raster.compiler.ir.par :as par]))

(defn ^:private object-typed?
  "True if the symbol's metadata says its type is Object (i.e. erased)."
  [sym]
  (= 'Object (or (:tag (meta sym))
                 (:raster.type/tag (meta sym)))))

(defn helper-object-params
  "Return [{:helper :params}] for every non-compute helper in fn-specs
  that has Object-typed parameters."
  [fn-specs]
  (vec (keep (fn [{:keys [name params]}]
               (let [obj (filterv object-typed? params)]
                 (when (and (not= name 'compute) (seq obj))
                   {:helper name :params (mapv str obj)})))
             fn-specs)))

(defn par-form-missing-elem-type
  "Walk form, return [{:form-head :out :loc}] for each par/map!,
  par/reduce or par/scan! call that has no :raster.type/elem-type
  metadata. These forms will pick :double in SIMD codegen by default,
  which silently miscompiles f32 arrays into DoubleVector calls."
  [form]
  (let [out (atom [])]
    (walk/postwalk
     (fn [f]
       (when (and (seq? f)
                  (or (par/par-map-form? f)
                      (par/par-reduce-form? f)))
         (when-not (:raster.type/elem-type (meta f))
           (swap! out conj {:form-head (first f)
                            :out (when (>= (count f) 2) (second f))})))
       f)
     form)
    @out))

(defn report-violations!
  "Emit a warning (or throw, if raster.strict.types=true) describing each
  Object-typed helper param. Pure side-effect; returns nil."
  [pass-key violations]
  (when (seq violations)
    (let [strict? (= "true" (System/getProperty "raster.strict.types"))
          msg (with-out-str
                (println (format "[%s] type-completeness violations:" pass-key))
                (doseq [{:keys [helper params]} violations]
                  (println (format "  helper '%s': %d Object-typed param(s) %s"
                                   helper (count params) params))))]
      (if strict?
        (throw (ex-info (str "STRICT TYPE CHECK failed: " msg)
                        {:pass pass-key :violations violations}))
        (binding [*out* *err*] (print msg))))))

(defn report-elem-type-gaps!
  "Emit a warning (or throw, if raster.strict.types=true) for par forms
  missing :raster.type/elem-type meta."
  [pass-key gaps]
  (when (seq gaps)
    (let [strict? (= "true" (System/getProperty "raster.strict.types"))
          msg (with-out-str
                (println (format "[%s] par form(s) missing :raster.type/elem-type:" pass-key))
                (doseq [{:keys [form-head out]} gaps]
                  (println (format "  %s into %s" form-head out))))]
      (if strict?
        (throw (ex-info (str "STRICT TYPE CHECK failed: " msg)
                        {:pass pass-key :gaps gaps}))
        (binding [*out* *err*] (print msg))))))
