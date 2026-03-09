(ns raster.compiler.passes.scalar.hoist
  "Reusable buffer hoisting utilities built on buffer fusion."
  (:require [raster.compiler.ir.form :as form]
            [raster.compiler.passes.scalar.buffer-fuse :as buffer-fuse]))

(defn hoist-buffers
  "Split a fused let* form into hoistable allocations and per-call body."
  [fused-let-form]
  (let [[_ bindings-vec & body-exprs] fused-let-form
        pairs (partition 2 bindings-vec)
        hoistable? (fn [[sym _]] (:raster.buffer/hoistable (meta sym)))
        hoistable (vec (filter hoistable? pairs))
        inner (vec (remove hoistable? pairs))
        alloc-bindings (mapv vec hoistable)
        buf-syms (mapv first hoistable)
        inner-bindings (vec (mapcat identity inner))
        inner-form (list* 'let* inner-bindings body-exprs)]
    {:alloc-bindings alloc-bindings
     :buf-syms buf-syms
     :inner-form inner-form}))

(defn fuse-walked-body
  "Apply buffer fusion to let* forms in a walked body."
  [body-forms]
  (let [stats (atom {:fused 0 :fresh-allocs 0 :unchanged 0})
        rewritten
        (mapv (fn [form]
                (if (and (seq? form)
                         (form/binding-form? form))
                  (let [result (buffer-fuse/fuse-let form)]
                    (swap! stats (fn [current-stats]
                                   (merge-with + current-stats (:stats result))))
                    (:form result))
                  form))
              body-forms)]
    {:forms rewritten
     :stats @stats}))

;; compile-fused-fn removed — replaced by closure/hoist-and-compile