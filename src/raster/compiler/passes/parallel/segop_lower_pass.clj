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

(def ^:private fatal-reasons
  "A violated invariant is not a missing lowering rule. Recording one as a conversion decline would
   let the pipeline continue on the legacy path with the bug intact — the loud-to-silent trade this
   whole change exists to prevent."
  #{:raster/fatal :raster/bug})

(defn- diagnostic
  "The structured record north-star §3.5 asks for in place of a warning: WHICH operation, in which
   binding, for which target dialect and device, what rule was missing, and what happens instead."
  [sym form stage ^Exception e device-id dtype]
  {:op (when (seq? form) (first form))
   :sym sym
   :stage stage                     ; :soac (par form → SOAC) or :segop (SOAC → SegOp)
   :target-dialect :segop
   :device (or device-id :cpu:0)
   :dtype (or dtype :double)
   :reason (or (:reason (ex-data e)) :no-lowering-rule)
   :message (.getMessage e)
   ;; a fallback is a stated outcome, not the absence of one
   :fallback :no-segop-metadata-backend-lowers-or-uses-legacy-codegen})

(defn- lower-attempt
  "Lower a par form to SegOp records via the SOAC intermediate.

   Returns `{:segops [...]}` on success, `{:declined <diagnostic>}` when the form IS a parallel
   primitive but no lowering rule applies, and nil when the form is simply not a par form — the
   common, correctly-silent case (most bindings are ordinary values).

   This used to `println` a WARNING to stderr and return nil for BOTH of the last two cases. That
   conflation is the defect: `nil` meant \"nothing to do here\" and \"a parallel form the middle end
   cannot represent\" at once, so a real coverage gap looked exactly like an ordinary binding and the
   only trace was a line on stderr that no pass, stat, or diagnostic could see. north-star §3.5 names
   this precise code: SegOp lowering \"may no longer warn and return nil\".

   The lowering ATTEMPT is unchanged — still tried for any seq, so nothing that lowered before stops
   lowering now. What is new is that a failure on a recognized par form is reported as data.

   NB success/failure is carried in an explicit `{:ok …}`/`{:err …}` rather than a truthy value: a
   SOAC node is a RECORD, and records satisfy `map?` and are always truthy, so a compact
   `or`/`if-let` version silently misread every successful lowering as a decline marker."
  [sym form device-id dtype array-types]
  (when (seq? form)
    (let [par? (par/par-form? form)
          decline (fn [stage e] (when (contains? fatal-reasons (:reason (ex-data e))) (throw e))
                    (when par? {:declined (diagnostic sym form stage e device-id dtype)}))
          ;; capture value-or-exception in one call: the alternative (catch a sentinel, then call
          ;; again to get the exception) re-runs a side-effecting conversion
          ;; Only an intentional, structured conversion refusal may become a decline. A raw
          ;; NullPointerException/ClassCastException/etc. is an implementation bug and must escape;
          ;; treating it as "no lowering rule" would silently select a different backend path.
          attempt (fn [f] (try {:ok (f)} (catch clojure.lang.ExceptionInfo e {:err e})))
          soac (attempt #(soac/par-form->soac sym form (swap! id-counter inc) :dtype (or dtype :double)))]
      (cond
        (:err soac) (decline :soac (:err soac))

        (nil? (:ok soac))
        (when par? {:declined (diagnostic sym form :soac
                                          (ex-info "par-form->soac produced no SOAC node" {})
                                          device-id dtype)})

        :else
        (let [segops (attempt #(soac-lower/lower-soac (:ok soac) (or device-id :cpu:0)
                                                      :dtype (or dtype :double)))]
          (cond
            (:err segops) (decline :segop (:err segops))
            (seq (:ok segops)) (cond-> {:segops (:ok segops)}
                                 (soac-lower/scan-soac? (:ok soac))
                                 (assoc :kernel-graph
                                        (soac-lower/scan-kernel-graph
                                         (:ok soac) (:ok segops) {:array-types array-types})))
            :else (when par? {:declined (diagnostic sym form :segop
                                                    (ex-info "lower-soac produced no SegOps" {})
                                                    device-id dtype)})))))))

(defn- annotate-binding
  "Attach scheduled compiler values to a binding symbol."
  [sym {:keys [segops kernel-graph]}]
  (cond-> (vary-meta sym assoc ::segops segops)
    kernel-graph (vary-meta assoc ::kernel-graph kernel-graph)))

(defn get-segops
  "Retrieve SegOp records from a binding symbol's metadata."
  [sym]
  (::segops (meta sym)))

(defn get-kernel-graph
  "Retrieve a scheduled KernelGraph from a binding symbol's metadata."
  [sym]
  (::kernel-graph (meta sym)))

(defn segop-lower-pass
  "Pipeline pass: convert par forms in let* bindings to SegOp records.

   Walks the form's let* bindings. For each par/map!, par/reduce, par/scan!
   binding, converts to SegOp and attaches as metadata on the binding symbol.

   Scan decomposition additionally records one verified KernelGraph with its intermediate buffers
   and dependencies. Returns both `:segops-lowered` and `:kernel-graphs-lowered` stats.

   Options from pipeline opts:
     :target-device — device for launch param computation
     :dtype — element type (:double or :float)"
  [form opts]
  (if-not (form/binding-form? form)
    {:form form :stats {:segops-lowered 0 :kernel-graphs-lowered 0}}
    (let [[let-sym bindings-vec & body-exprs] form
          pairs (partition 2 bindings-vec)
          device-id (:target-device opts)
          dtype (:dtype opts)
          array-types (:array-types opts)
          lowered (atom 0)
          graphs-lowered (atom 0)
          ;; Every par form the middle end could NOT represent, as data. Previously these went to
          ;; stderr as `WARNING: …` and vanished — invisible to stats, to explain-pipeline, and to
          ;; anyone diagnosing why a kernel took the legacy path.
          declined (atom [])
          attempt (fn [sym init]
                    (let [r (lower-attempt sym init device-id dtype array-types)]
                      (when-let [d (:declined r)] (swap! declined conj d))
                      (when (:segops r) r)))
          ;; Annotate bindings with SegOp metadata
          new-pairs
          (mapv (fn [[sym init]]
                  (if-let [lowered-values (attempt sym init)]
                    (do (swap! lowered inc)
                        (when (:kernel-graph lowered-values) (swap! graphs-lowered inc))
                        [(annotate-binding sym lowered-values) init])
                    [sym init]))
                pairs)
          ;; Also check body expressions for par forms
          new-body
          (mapv (fn [expr]
                  (let [tmp-sym (gensym "body_par_")]
                    (if-let [{:keys [segops kernel-graph]} (attempt tmp-sym expr)]
                      (do (swap! lowered inc)
                          (when kernel-graph (swap! graphs-lowered inc))
                          (with-meta (list 'do expr)
                            (cond-> {::body-segops segops}
                              kernel-graph (assoc ::body-kernel-graph kernel-graph))))
                      expr)))
                body-exprs)
          new-bindings (vec (mapcat identity new-pairs))]
      {:form (list* let-sym new-bindings new-body)
       :stats (cond-> {:segops-lowered @lowered
                       :kernel-graphs-lowered @graphs-lowered}
                (seq @declined) (assoc :segops-declined @declined))})))
