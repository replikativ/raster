(ns raster.compiler.backend.wasm.emit
  "Track A: walk raster's POST-PASS IR (the S-expr compile-aot hands the bytecode
   backend) and emit a WebAssembly module via encoder.clj. Reads the semantic op
   from :raster.op/original metadata (never the mangled impl name), per the
   compiler design rules.

   v1 node set:
     scalar body         — value expr → function returns it
     (loop [v init ...] (if cond <recur-or-effects+recur> <result>))
     let* / do / if / recur / .invk(raster.numeric +,-,*,/)
     clojure.core/{aget,aset,<,<=,>,>=,inc,dec} · long/int/double casts
     param & loop-local syms · numeric literals
   Arrays live in linear memory as (ptr,len); index/count are i32 (v1 — 32-bit
   memory). Float element type from the array param's tag (doubles→f64, floats→f32).
   Every loop returns its non-recur (else) value, so the function result type is
   the deftm's return tag."
  (:require [raster.compiler.backend.wasm.encoder :as e]))

;; --- tag → wasm types ------------------------------------------------------
(def ^:private array-tag->elem
  {'doubles {:vt :f64 :bytes 8 :align 3 :load :f64.load :store :f64.store}
   'floats  {:vt :f32 :bytes 4 :align 2 :load :f32.load :store :f32.store}
   'longs   {:vt :i64 :bytes 8 :align 3 :load :i64.load :store :i64.store}
   'ints    {:vt :i32 :bytes 4 :align 2 :load :i32.load :store :i32.store}})

(defn- array-tag? [tag] (contains? array-tag->elem tag))

(defn- tag->vt
  "Scalar raster/annotation tag → wasm valtype. Arrays are i32 pointers; long/int
   indices are i32 in v1 (32-bit linear memory)."
  [tag]
  (cond
    (#{'double 'Double} tag) :f64
    (#{'float 'Float} tag)   :f32
    (#{'long 'int 'Long 'Integer} tag) :i32
    (array-tag? tag) :i32
    :else :i32))

(defn- arith-op [sem vt]
  (let [pfx (case vt :f64 "f64" :f32 "f32" "i32")]
    (case sem
      raster.numeric/+ (keyword (str pfx ".add"))
      raster.numeric/- (keyword (str pfx ".sub"))
      raster.numeric/* (keyword (str pfx ".mul"))
      raster.numeric// (keyword (str pfx ".div"))   ; f64/f32 only; integer / is rare
      nil)))

(defn- cmp-op
  "Comparison opcode for base (lt/gt/le/ge/eq/ne) at operand valtype vt.
   Float compares are unsigned-name-free; integer compares are signed."
  [base vt]
  (case vt
    :f64 (keyword (str "f64." base))
    :f32 (keyword (str "f32." base))
    (keyword (str "i32." (case base "lt" "lt_s" "gt" "gt_s" "le" "le_s" "ge" "ge_s" base)))))

;; ctx: {:env {sym {:idx :vt}}  :elems {sym elem-map}}
(declare emit-val infer-vt emit-effect alloc!)

(defn- addr
  "byte address of array element: ptr + idx*elem-bytes  (i32)."
  [ctx arr-sym idx-node]
  (let [pidx (get-in ctx [:env arr-sym :idx])
        eb   (get-in ctx [:elems arr-sym :bytes])]
    (-> (e/local-get pidx)
        (into (emit-val ctx idx-node))
        (into (e/i32-const eb)) (into (e/i :i32.mul)) (into (e/i :i32.add)))))

(defn- emit-val
  "Emit a value-producing expression → byte vector."
  [ctx node]
  (cond
    (integer? node) (e/i32-const node)
    (float? node)   (e/f64-const node)
    (symbol? node)  (let [{:keys [idx]} (get-in ctx [:env node])]
                      (when-not idx (throw (ex-info (str "unbound sym " node) {:node node})))
                      (e/local-get idx))
    (seq? node)
    (let [h (first node), A (vec (rest node))]
      (cond
        (#{'long 'int 'double} h) (emit-val ctx (first A)) ; index/f64 casts: identity
        (= h 'float)                                        ; f32 cast / literal
        (let [x (first A)]
          (cond (number? x) (e/f32-const x)                 ; (float 0.0) → f32 literal
                :else (into (emit-val ctx x) (e/i :f32.demote_f64)))) ; (float <f64>) → demote
        (= h 'clojure.core/aget)
        (let [[arr idx] A elem (get-in ctx [:elems arr])]
          (into (addr ctx arr idx) (e/mem-load (:load elem) (:align elem) 0)))
        ;; comparisons — match by local name (bare or clojure.core-qualified)
        (#{"<" "<=" ">" ">=" "==" "=" "not="} (name h))
        (let [v0 (infer-vt ctx (A 0))
              vt (if (#{:f64 :f32} v0) v0 (infer-vt ctx (A 1)))  ; float if either operand is float
              base (case (name h) "<" "lt" "<=" "le" ">" "gt" ">=" "ge" "not=" "ne" "eq")]
          (-> (emit-val ctx (A 0)) (into (emit-val ctx (A 1))) (into (e/i (cmp-op base vt)))))
        ;; integer step (inc/dec, incl. unchecked-*-int)
        (#{"inc" "unchecked-inc-int" "unchecked-inc"} (name h))
        (-> (emit-val ctx (A 0)) (into (e/i32-const 1)) (into (e/i :i32.add)))
        (#{"dec" "unchecked-dec-int" "unchecked-dec"} (name h))
        (-> (emit-val ctx (A 0)) (into (e/i32-const 1)) (into (e/i :i32.sub)))
        (= h '.invk)
        (let [sem (:raster.op/original (meta node))
              vt  (case (:raster.type/tag (meta node)) double :f64 float :f32 :i32)
              opk (or (arith-op sem vt) (throw (ex-info (str "unhandled .invk op " sem) {})))
              [_impl o1 o2] A]
          (-> (emit-val ctx o1) (into (emit-val ctx o2)) (into (e/i opk))))
        ;; bare raster.numeric/{+,-,*,/} — SROA output isn't re-devirtualized, and
        ;; the op IS the head (no .invk wrapper); type from the operands.
        (#{'raster.numeric/+ 'raster.numeric/- 'raster.numeric/* 'raster.numeric//} h)
        (let [[o1 o2] A
              v0 (infer-vt ctx o1)
              vt (if (#{:f64 :f32} v0) v0 (infer-vt ctx o2))
              opk (arith-op h vt)]
          (-> (emit-val ctx o1) (into (emit-val ctx o2)) (into (e/i opk))))
        ;; let* / do in VALUE position (introduced by inlined deftm calls)
        (= h 'let*)
        (let [[binds & body] A
              [ctx' init-bytes] (reduce (fn [[c acc] [s init]]
                                          (let [vt (infer-vt c init) idx (alloc! c vt)]
                                            [(assoc-in c [:env s] {:idx idx :vt vt})
                                             (-> acc (into (emit-val c init)) (into (e/local-set idx)))]))
                                        [ctx []] (partition 2 binds))
              tail (if (= 1 (count body)) (first body) (cons 'do body))]
          (into init-bytes (emit-val ctx' tail)))
        (= h 'do)
        (into (vec (mapcat #(emit-effect ctx %) (butlast A))) (emit-val ctx (last A)))
        :else (throw (ex-info (str "unhandled value head " h) {:node node}))))
    :else (throw (ex-info (str "unhandled value node " (pr-str node)) {}))))

(defn- infer-vt
  "Result valtype of an expression, from env + carried :raster.type/tag."
  [ctx node]
  (cond
    (symbol? node)  (get-in ctx [:env node :vt] :i32)
    (float? node)   :f64
    (integer? node) :i32
    (seq? node)
    (case (:raster.type/tag (meta node))
      double :f64, float :f32, long :i32, int :i32
      (cond
        (= 'float (first node))  :f32
        (= 'double (first node)) :f64
        (#{'long 'int} (first node)) (infer-vt ctx (second node))
        (#{'let* 'do} (first node)) (infer-vt ctx (last node))  ; value of tail
        (= 'clojure.core/aget (first node))  (get-in ctx [:elems (second node) :vt] :f64)
        :else :i32))
    :else :i32))

(declare emit-dotimes)

(defn- emit-effect
  "Side-effecting statement that leaves nothing on the stack (aset / when / do /
   dotimes)."
  [ctx node]
  (let [h (first node), A (vec (rest node))]
    (cond
      (= h 'clojure.core/aset)
      (let [[arr idx v] A elem (get-in ctx [:elems arr])]
        (-> (addr ctx arr idx) (into (emit-val ctx v)) (into (e/mem-store (:store elem) (:align elem) 0))))
      (= h 'do) (vec (mapcat #(emit-effect ctx %) A))
      (= h 'when)                                   ; (when cond body…) — void, no else
      (-> (emit-val ctx (first A))
          (into [(e/op :if) e/empty-block])
          (into (vec (mapcat #(emit-effect ctx %) (rest A))))
          (into [(e/op :end)]))
      (= h 'dotimes) (emit-dotimes ctx node)
      :else (throw (ex-info (str "unhandled effect head " h) {:node node})))))

(defn- emit-recur
  "(recur v0 v1 ...) → push all new values, set loop locals in REVERSE (so each
   value is read from the OLD locals before any are overwritten), then branch to
   the loop header at depth `loop-depth`."
  [ctx args loop-idxs loop-depth]
  (-> (vec (mapcat #(emit-val ctx %) args))
      (into (vec (mapcat #(e/local-set %) (reverse loop-idxs))))
      (into (e/br loop-depth))))

(defn- emit-then
  "then-branch of the loop's if: optional effects (a `do`) ending in a recur."
  [ctx node loop-idxs loop-depth]
  (cond
    (and (seq? node) (= 'recur (first node)))
    (emit-recur ctx (vec (rest node)) loop-idxs loop-depth)
    (and (seq? node) (= 'do (first node)))
    (let [stmts (vec (rest node))]
      (-> (vec (mapcat #(emit-effect ctx %) (butlast stmts)))
          (into (emit-then ctx (last stmts) loop-idxs loop-depth))))
    :else (throw (ex-info (str "loop then-branch must end in recur, got " (pr-str node)) {}))))

(defn- alloc!
  "Reserve a new function-local of valtype vt, return its index. Locals are
   collected in the ctx :locals atom; index = base (after params) + position."
  [ctx vt]
  (let [a (:locals ctx)
        idx (+ (:base ctx) (count @a))]
    (swap! a conj vt)
    idx))

(declare emit-result)

(defn- emit-loop
  "(loop [v init ...] (if cond <recur|do…recur> <result>)) → [bytes ret-vt].
   Returns the non-recur (else) value via block(result vt){ loop ; unreachable }."
  [ctx node]
  (let [[_ bindvec ifform] node
        pairs (vec (partition 2 bindvec))
        ;; allocate loop-var locals; inits emitted with the OUTER env (loop vars
        ;; not yet in scope for their own inits)
        loop-vars (mapv (fn [[sym init]]
                          (let [vt (infer-vt ctx init)]  ; 0→i32, 0.0→f64, (float 0.0)→f32
                            {:sym sym :idx (alloc! ctx vt) :vt vt :init init}))
                        pairs)
        inits (vec (mapcat (fn [{:keys [idx init]}]
                             (into (emit-val ctx init) (e/local-set idx))) loop-vars))
        ctx' (reduce (fn [c {:keys [sym idx vt]}] (assoc-in c [:env sym] {:idx idx :vt vt}))
                     ctx loop-vars)
        loop-idxs (mapv :idx loop-vars)
        [_ cnd then els] ifform
        ret-vt (infer-vt ctx' els)
        ;; depths from inside the if: if=0, loop=1, block=2
        if-bytes (-> (emit-val ctx' cnd)
                     (into [(e/op :if) e/empty-block])
                     (into (emit-then ctx' then loop-idxs 1))
                     (into [(e/op :else)])
                     (into (emit-val ctx' els)) (into (e/br 2))
                     (into [(e/op :end)]))]
    [(into inits (e/block-t ret-vt (into (e/loop* if-bytes) (e/i :unreachable))))
     ret-vt]))

(defn- emit-dotimes
  "(dotimes [i n] body…) → a void counted loop. Emits:
     i = 0
     block { loop { br_if 1 (i >= n) ; body-effects ; i++ ; br 0 } }
   Labels from inside the loop body: loop=0, block=1."
  [ctx node]
  (let [[_ [isym ncnt] & body] node
        idx  (alloc! ctx :i32)
        ctx' (assoc-in ctx [:env isym] {:idx idx :vt :i32})
        loop-body (-> (e/local-get idx)
                      (into (emit-val ctx' ncnt))
                      (into (e/i :i32.ge_s))
                      (into (e/br-if 1))                       ; exit block when i >= n
                      (into (vec (mapcat #(emit-effect ctx' %) body)))
                      (into (e/local-get idx)) (into (e/i32-const 1)) (into (e/i :i32.add))
                      (into (e/local-set idx))
                      (into (e/br 0)))]                        ; re-loop
    (-> (e/i32-const 0) (into (e/local-set idx))
        (into (e/block (e/loop* loop-body))))))

(defn- emit-result
  "Emit an expression in result (tail) position → [bytes ret-vt] (ret-vt = nil for
   void). Handles let* (a local per binding), do (effects then tail), loop, dotimes
   (void), when (void), and scalar value expressions."
  [ctx node]
  (cond
    (and (seq? node) (= 'dotimes (first node))) [(emit-dotimes ctx node) nil]
    (and (seq? node) (= 'when (first node)))    [(emit-effect ctx node) nil]

    (and (seq? node) (= 'let* (first node)))
    (let [[_ binds & body] node
          [ctx' init-bytes] (reduce (fn [[c acc] [s init]]
                                      (let [vt (infer-vt c init)
                                            idx (alloc! c vt)]
                                        [(assoc-in c [:env s] {:idx idx :vt vt})
                                         (-> acc (into (emit-val c init)) (into (e/local-set idx)))]))
                                    [ctx []] (partition 2 binds))
          tail (if (= 1 (count body)) (first body) (cons 'do body))
          [b ret] (emit-result ctx' tail)]
      [(into init-bytes b) ret])

    (and (seq? node) (= 'do (first node)))
    (let [stmts (vec (rest node))
          [b ret] (emit-result ctx (last stmts))]
      [(into (vec (mapcat #(emit-effect ctx %) (butlast stmts))) b) ret])

    (and (seq? node) (#{'loop 'loop*} (first node))) (emit-loop ctx node)

    :else [(emit-val ctx node) (infer-vt ctx node)]))

(defn compile-kernel
  "Compile a kernel IR to a wasm module byte[].
   params: ordered [{:sym :tag}].  ir: post-pass S-expr. The result type is
   inferred from the IR (loop else-value / scalar body).
   Returns {:bytes byte[] :name :param-types [vt...] :result-types [vt...]}."
  [{:keys [name params ir mem-pages] :or {mem-pages 256}}]
  (let [param-vts (mapv #(tag->vt (:tag %)) params)
        env0 (into {} (map-indexed (fn [idx {:keys [sym tag]}] [sym {:idx idx :vt (tag->vt tag)}]) params))
        elems (into {} (keep (fn [{:keys [sym tag]}] (when (array-tag? tag) [sym (array-tag->elem tag)])) params))
        ctx {:env env0 :elems elems :base (count params) :locals (atom [])}
        [body-bytes ret-vt] (emit-result ctx ir)
        result-types (if ret-vt [ret-vt] [])     ; nil ret-vt = void (array-mutating)
        locals @(:locals ctx)]
    {:name name
     :param-types param-vts
     :result-types result-types
     :bytes (e/build-module
             {:types [(e/functype param-vts result-types)]
              :funcs [{:name name :type-idx 0
                       :param-types param-vts :result-types result-types
                       :locals locals :body body-bytes}]
              :memory {:min mem-pages :max mem-pages :export "memory"}})}))
