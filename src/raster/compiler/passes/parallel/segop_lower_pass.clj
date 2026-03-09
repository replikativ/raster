(ns raster.compiler.passes.parallel.segop-lower-pass
  "Pipeline pass: lower par forms to SegOp records.

   Walks let* bindings and converts raster.par/* forms to SegOp IR via
   SOAC intermediate. SegOp records are attached as metadata on binding
   symbols for downstream backend consumption.

   This decouples hardware-aware execution planning from backend codegen:
   - Lowering decides phase decomposition, launch params, accumulator count
   - Backend translates SegOp to target code (SIMD, OpenCL, scalar)"
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.device :as device]
            [raster.runtime.hardware :as hw]
            [raster.compiler.ir.form :as form]))

(def ^:private id-counter (atom 0))

(defn- par-form->segops
  "Convert a par form to SegOp records via SOAC intermediate.
   sym: binding symbol (or gensym for body-position forms).
   form: the par/* S-expression.
   Returns [SegOp ...] or nil if not a par form."
  [sym form device-id dtype]
  (when (seq? form)
    (let [id (swap! id-counter inc)
          soac-node (try (soac/par-form->soac sym form id)
                         (catch Exception e
                           (binding [*out* *err*]
                             (println (str "WARNING: SegOp lowering failed for " sym ": " (.getMessage e))))
                           nil))]
      (when soac-node
        (try
          (soac-lower/lower-soac soac-node
                                 (or device-id :cpu:0)
                                 :dtype (or dtype :double))
          (catch Exception e
            (binding [*out* *err*]
              (println (str "WARNING: SegOp lowering failed for " sym ": " (.getMessage e))))
            nil))))))

(defn- annotate-binding
  "Attach SegOp metadata to a binding symbol."
  [sym segops]
  (vary-meta sym assoc ::segops segops))

(defn get-segops
  "Retrieve SegOp records from a binding symbol's metadata."
  [sym]
  (::segops (meta sym)))

(defn segop-lower-pass
  "Pipeline pass: convert par forms in let* bindings to SegOp records.

   Walks the form's let* bindings. For each par/map!, par/reduce, par/scan!
   binding, converts to SegOp and attaches as metadata on the binding symbol.

   Returns {:form annotated-form :stats {:segops-lowered N}}.

   Options from pipeline opts:
     :target-device — device for launch param computation
     :dtype — element type (:double or :float)"
  [form opts]
  (if-not (form/binding-form? form)
    {:form form :stats {:segops-lowered 0}}
    (let [[let-sym bindings-vec & body-exprs] form
          pairs (partition 2 bindings-vec)
          device-id (:target-device opts)
          dtype (:dtype opts)
          lowered (atom 0)
          ;; Annotate bindings with SegOp metadata
          new-pairs
          (mapv (fn [[sym init]]
                  (if-let [segops (par-form->segops sym init device-id dtype)]
                    (do (swap! lowered inc)
                        [(annotate-binding sym segops) init])
                    [sym init]))
                pairs)
          ;; Also check body expressions for par forms
          new-body
          (mapv (fn [expr]
                  (let [tmp-sym (gensym "body_par_")]
                    (if-let [segops (par-form->segops tmp-sym expr device-id dtype)]
                      (do (swap! lowered inc)
                          (with-meta (list 'do expr)
                            {::body-segops segops}))
                      expr)))
                body-exprs)
          new-bindings (vec (mapcat identity new-pairs))]
      {:form (list* let-sym new-bindings new-body)
       :stats {:segops-lowered @lowered}})))
