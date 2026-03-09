(ns raster.compiler.passes.scalar.buffer-fuse
  "Buffer reuse pass: rewrite allocating ops to reuse dead buffers."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]
            [raster.analysis.memory :as ma]))

(defn- can-reuse-arg?
  "Check if arg-sym's buffer can be reused at binding-idx.
  Uses the pre-computed memory analysis which includes escape analysis
  (closure capture and return escape) in addition to liveness checks."
  [arg-sym binding-idx analysis]
  (let [used-after (get-in analysis [:used-after binding-idx] #{})
        aliases (get-in analysis [:alias-state :aliases])
        arg-aliases (ma/transitive-closure #{arg-sym} aliases)
        ;; Find the binding that defines arg-sym and check its escape analysis
        arg-binding (first (filter #(= arg-sym (:binding-name %))
                                   (:bindings analysis)))]
    (and (= arg-aliases #{arg-sym})
         (not (contains? used-after arg-sym))
         ;; Must not escape via return or closure capture
         (or (nil? arg-binding) (not (:escapes? arg-binding))))))

(def ^:private call-head util/call-head)
(def ^:private call-args util/call-args)

(defn fuse-let
  "Fuse buffer allocations in a (let* [...] body) form."
  [let-form & {:keys [dtype]}]
  (let [analysis (ma/analyze-sexp-let let-form)
        [_ bindings-vec & body-exprs] let-form
        pairs (vec (partition 2 bindings-vec))
        fused (atom 0)
        fresh-allocs (atom 0)
        unchanged (atom 0)
        new-pairs
        (vec
         (mapcat
          (fn [[idx [sym init]]]
            (let [head (call-head init)
                  resolved (when head (descriptor/resolve-buffer-semantics head))]
              (if-let [[entry _base-op] resolved]
                (if (:allocates? entry)
                  (let [args (call-args init)
                        in-place-idx (:in-place-arg entry)]
                    (if (and in-place-idx
                             (< in-place-idx (count args))
                             (symbol? (nth args in-place-idx))
                             (can-reuse-arg? (nth args in-place-idx) idx analysis))
                      (do (swap! fused inc)
                          [[sym ((:rewrite-fn entry) args (nth args in-place-idx))]])
                      (if-let [alloc-fn (:alloc-form entry)]
                        (let [;; Determine write mode: check if the rewrite calls an op
                              ;; with a registered buffer-write mode, or if it's a par/map!
                              ;; (which always overwrites every element)
                              rewrite-sample ((:rewrite-fn entry) args 'buf__probe)
                              write-mode (cond
                                           ;; Check if rewrite body is a par/map! (overwrite)
                                           (and (seq? rewrite-sample)
                                                (= :par (:kind (form/form-info rewrite-sample))))
                                           :overwrite
                                           ;; Check rewrite head for registered write mode
                                           (and (seq? rewrite-sample) (symbol? (first rewrite-sample)))
                                           (let [wm (descriptor/get-buffer-write-mode (first rewrite-sample))]
                                             (or (:mode wm) :accumulate))
                                           ;; Let form: check the first effectful form inside
                                           (and (seq? rewrite-sample) (form/binding-form? rewrite-sample))
                                           (let [body-forms (drop 2 rewrite-sample)
                                                 first-effect (first (filter seq? body-forms))
                                                 ;; For .invk calls, the op name is the 2nd element
                                                 effect-op (when first-effect
                                                             (if (= '.invk (first first-effect))
                                                               (second first-effect)
                                                               (first first-effect)))]
                                             (if (and effect-op (symbol? effect-op))
                                               (let [wm (descriptor/get-buffer-write-mode effect-op)]
                                                 (or (:mode wm) :accumulate))
                                               :accumulate))
                                           ;; Default: accumulate (conservative — buffer must be zeroed).
                                           ;; Use :overwrite only when proven (par/map!, registered ops).
                                           :else :accumulate)
                              buf-sym (with-meta (gensym (str "buf_" (name sym) "_"))
                                        {:raster.buffer/hoistable true :raster.buffer/write-mode write-mode})
                              alloc-expr (alloc-fn args {:dtype dtype})]
                          (swap! fresh-allocs inc)
                          [[buf-sym alloc-expr]
                           [sym ((:rewrite-fn entry) args buf-sym)]])
                        (do (swap! unchanged inc)
                            [[sym init]]))))
                  (do (swap! unchanged inc)
                      [[sym init]]))
                (do (swap! unchanged inc)
                    [[sym init]]))))
          (map-indexed vector pairs)))
        all-bindings (vec (mapcat identity new-pairs))
        new-form (let [r (list* 'let* all-bindings body-exprs)]
                   (if-let [m (meta let-form)] (with-meta r m) r))]
    {:form new-form
     :stats {:fused @fused
             :fresh-allocs @fresh-allocs
             :unchanged @unchanged}}))