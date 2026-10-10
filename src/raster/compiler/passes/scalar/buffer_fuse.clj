(ns raster.compiler.passes.scalar.buffer-fuse
  "Buffer reuse pass: rewrite allocating ops to reuse dead buffers."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.ir.form :as form]
            [raster.analysis.memory :as ma]))

(defn- fresh-buffer-init?
  "The binding allocates its own buffer: an array constructor or allocator, or
  an op whose buffer facet allocates its result."
  [init]
  (when-let [head (util/call-head init)]
    (or (descriptor/alloc-op? head)
        (some-> (descriptor/resolve-buffer-semantics head) first :allocates?))))

(defn- can-reuse-arg?
  "Check if arg-sym's buffer can be reused at binding-idx.
  Uses the pre-computed memory analysis which includes escape analysis
  (closure capture and return escape) in addition to liveness checks. Only a
  buffer this let owns is reused: a parameter or any other borrowed array may
  be held by the caller, whatever its local last use."
  [arg-sym binding-idx analysis owned]
  (let [used-after (get-in analysis [:used-after binding-idx] #{})
        aliases (get-in analysis [:alias-state :aliases])
        arg-aliases (ma/transitive-closure #{arg-sym} aliases)
        ;; Find the binding that defines arg-sym and check its escape analysis
        arg-binding (first (filter #(= arg-sym (:binding-name %))
                                   (:bindings analysis)))]
    (and (contains? owned arg-sym)
         (= arg-aliases #{arg-sym})
         (not (contains? used-after arg-sym))
         ;; Must not escape via return or closure capture
         (or (nil? arg-binding) (not (:escapes? arg-binding))))))

(def ^:private call-head util/call-head)
(def ^:private call-args util/call-args)

(defn- known-argument-tag
  "Use the existing result-alternative authority; one branch is not admission."
  [expression environment]
  (let [tags (inference/infer-result-tags expression environment *ns*)]
    (when (= 1 (count tags)) (first tags))))

(defn- same-parameter-carrier?
  "Removing a specialization call must not remove a coercion or check.
   Require known equal carriers; Object fallback is not type evidence."
  [actual declared source-ns]
  (when (and actual declared)
    (let [actual-class (types/tag->check-class actual)
          declared-class (binding [*ns* (or source-ns *ns*)]
                           (types/tag->check-class declared))]
      (and (= actual-class declared-class)
           (or (not= Object actual-class)
               (= 'Object actual declared))))))

(defn fuse-let
  "Fuse buffer allocations in a (let* [...] body) form."
  [let-form & {:keys [dtype param-env]}]
  (let [analysis (ma/analyze-sexp-let let-form)
        [_ bindings-vec & body-exprs] let-form
        pairs (vec (partition 2 bindings-vec))
        owned (into #{} (keep (fn [[sym init]] (when (fresh-buffer-init? init) sym))) pairs)
        fused (atom 0)
        fresh-allocs (atom 0)
        unchanged (atom 0)
        new-pairs
        (vec
         (mapcat
          (fn [[idx [sym init]]]
            (binding [util/*shadowing-locals*
                      (into util/*shadowing-locals*
                            (concat (keys param-env) (map first (take idx pairs))))]
            (let [head (call-head init)
                  resolved (when head (descriptor/resolve-buffer-semantics head))
                  actual-tags (when (:auto-detected? (first resolved))
                                ;; Keeping the original prefix here retains
                                ;; alternatives of prior locals instead of
                                ;; resurrecting a single stale metadata tag.
                                (mapv #(known-argument-tag
                                        (list 'let* (vec (mapcat identity (take idx pairs))) %)
                                        (or param-env {}))
                                      (call-args init)))]
              (if-let [[entry _base-op] resolved]
                (if (and (:allocates? entry)
                         (or (not (:auto-detected? entry))
                             (and (= (count (:parameters entry)) (count (call-args init))
                                     (count (:parameter-tags entry)))
                                  (every? true? (map #(same-parameter-carrier? %1 %2 (:source-ns entry))
                                                     actual-tags (:parameter-tags entry))))))
                  (let [original-args (call-args init)
                        argument-pairs
                        (when (:auto-detected? entry)
                          (mapv (fn [argument tag]
                                    [(with-meta (gensym "buffer_arg__")
                                       (cond-> (meta argument) tag (assoc :raster.type/tag tag)))
                                     argument]) original-args actual-tags))
                        args (if (:auto-detected? entry) (mapv first argument-pairs) original-args)
                        in-place-idx (:in-place-arg entry)]
                    (if (and in-place-idx
                             (< in-place-idx (count args))
                             (symbol? (nth args in-place-idx))
                             (can-reuse-arg? (nth args in-place-idx) idx analysis owned))
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
                                        {:raster.buffer/hoistable (not (:auto-detected? entry))
                                         :raster.buffer/no-hoist (:auto-detected? entry)
                                         :raster.buffer/write-mode write-mode})
                              alloc-expr (alloc-fn args {:dtype dtype})]
                          (swap! fresh-allocs inc)
                          (concat argument-pairs
                                  [[buf-sym alloc-expr]
                                   [sym ((:rewrite-fn entry) args buf-sym)]]))
                        (do (swap! unchanged inc)
                            [[sym init]]))))
                  (do (swap! unchanged inc)
                      [[sym init]]))
                (do (swap! unchanged inc)
                    [[sym init]])))))
          (map-indexed vector pairs)))
        all-bindings (vec (mapcat identity new-pairs))
        new-form (let [r (list* 'let* all-bindings body-exprs)]
                   (if-let [m (meta let-form)] (with-meta r m) r))]
    {:form new-form
     :stats {:fused @fused
             :fresh-allocs @fresh-allocs
             :unchanged @unchanged}}))
