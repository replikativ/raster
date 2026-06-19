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

(defn- void-effect-form?
  "A form producing no value (side-effect only): dotimes/when, or a do whose tail
   is itself a void effect. Such a form bound in a let* (e.g. the inliner's
   (let* [_ret (dotimes …)] _ret) body-lift) is emitted as an effect, not a value."
  [node]
  (and (seq? node)
       (let [h (first node)]
         (or (#{'dotimes 'when} h)
             (and (= 'do h) (void-effect-form? (last node)))))))

(defn- emit-val
  "Emit a value-producing expression → byte vector."
  [ctx node]
  (cond
    (integer? node) (e/i32-const node)
    (float? node)   (e/f64-const node)
    (symbol? node)  (let [{:keys [idx void]} (get-in ctx [:env node])]
                      (cond void []                       ; void-bound sym carries no value
                            idx  (e/local-get idx)
                            :else (throw (ex-info (str "unbound sym " node) {:node node}))))
    (seq? node)
    (let [h (first node), A (vec (rest node))]
      (cond
        (#{'long 'int} h) (emit-val ctx (first A))          ; integer index casts: identity (i32)
        (= h 'double)                                       ; f64 cast / literal — type-aware
        (let [x (first A)]
          (if (number? x)
            (e/f64-const (double x))
            (let [xv (infer-vt ctx x)]
              (cond-> (emit-val ctx x)
                (= xv :f32) (into (e/i :f64.promote_f32))     ; (double <f32>) → promote
                (= xv :i32) (into (e/i :f64.convert_i32_s)))))) ; (double <i32>) → convert
        (= h 'float)                                        ; f32 cast / literal — type-aware
        (let [x (first A)]
          (if (number? x)
            (e/f32-const x)                                  ; (float 0.0) → f32 literal
            (let [xv (infer-vt ctx x)]
              (cond-> (emit-val ctx x)
                (= xv :f64) (into (e/i :f32.demote_f64))      ; (float <f64>) → demote
                (= xv :i32) (into (e/i :f32.convert_i32_s)))))) ; (float <i32>) → convert
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
                                          (if (void-effect-form? init)
                                            [(assoc-in c [:env s] {:void true})
                                             (into acc (emit-effect c init))]
                                            (let [vt (infer-vt c init) idx (alloc! c vt)]
                                              [(assoc-in c [:env s] {:idx idx :vt vt})
                                               (-> acc (into (emit-val c init)) (into (e/local-set idx)))])))
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

(declare emit-dotimes emit-dotimes-scalar)

(defn- emit-effect
  "Side-effecting statement that leaves nothing on the stack (aset / when / do /
   let* / dotimes)."
  [ctx node]
  (let [h (first node), A (vec (rest node))]
    (cond
      (= h 'clojure.core/aset)
      (let [[arr idx v] A elem (get-in ctx [:elems arr])]
        (-> (addr ctx arr idx) (into (emit-val ctx v)) (into (e/mem-store (:store elem) (:align elem) 0))))
      (= h 'do) (vec (mapcat #(emit-effect ctx %) A))
      ;; let* in effect position (e.g. soa-lower floats arg bindings out of a
      ;; store: (let* [args…] (do (aset …) …))). Bind locals, emit body as effects.
      (= h 'let*)
      (let [[binds & body] A
            [ctx' init-bytes] (reduce (fn [[c acc] [s init]]
                                        (if (void-effect-form? init)
                                          [(assoc-in c [:env s] {:void true}) (into acc (emit-effect c init))]
                                          (let [vt (infer-vt c init) idx (alloc! c vt)]
                                            [(assoc-in c [:env s] {:idx idx :vt vt})
                                             (-> acc (into (emit-val c init)) (into (e/local-set idx)))])))
                                      [ctx []] (partition 2 binds))]
        (into init-bytes (vec (mapcat #(emit-effect ctx' %) body))))
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

(declare emit-result emit-loop-scalar simd-loop-plan emit-loop-simd)

(defn- emit-loop
  "(loop [v init ...] (if cond <recur|do…recur> <result>)) → [bytes ret-vt].
   When ctx :simd? is set and the loop is an elementwise counted map, emits a
   v128 main loop + scalar remainder; otherwise the scalar loop."
  [ctx node]
  (if-let [plan (and (:simd? ctx) (simd-loop-plan ctx node))]
    (emit-loop-simd ctx node plan)
    (emit-loop-scalar ctx node)))

(defn- emit-loop-scalar
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

;; ─────────────────────────────────────────────────────────────────────────
;; SIMD128 vectorization of elementwise dotimes-maps (opt-in via ctx :simd?).
;; Targets (dotimes [i n] (aset out i <pure-elementwise>)…) where every store
;; indexes exactly i and the value is composed only of (aget arr i), arithmetic
;; (+,-,*,/), scalar literals and loop-invariant scalars. Emits a v128 main loop
;; (2-wide f64 / 4-wide f32) plus a scalar remainder loop. Cannot run under
;; Chicory (no v128) — validate execution via node/V8.
;; ─────────────────────────────────────────────────────────────────────────

(def ^:private simd-op-suffix {"+" "add" "-" "sub" "*" "mul" "/" "div"})

(defn- simd-op-of
  "If node is a vectorizable arithmetic op, return its suffix (add/sub/mul/div)."
  [node]
  (let [h (first node)
        sym (cond (= h '.invk) (:raster.op/original (meta node))
                  (#{'raster.numeric/+ 'raster.numeric/- 'raster.numeric/* 'raster.numeric//} h) h
                  :else nil)]
    (when sym (simd-op-suffix (name sym)))))

(defn- simd-op-args [node] (if (= '.invk (first node)) (drop 2 node) (rest node)))

(defn- simd-index-ok?
  "Array index is exactly the loop var (optionally wrapped in a long/int cast)."
  [i-sym idx]
  (or (= idx i-sym)
      (and (seq? idx) (#{'long 'int} (first idx)) (= (second idx) i-sym))))

(defn- simd-pure?
  "node is an elementwise-pure value: aget@i, arith of pure, scalar literal, or a
   loop-invariant scalar (NOT the bare loop var — it differs per lane)."
  [i-sym node]
  (cond
    (number? node) true
    (symbol? node) (not= node i-sym)
    (seq? node)
    (let [h (first node)]
      (cond
        (= h 'clojure.core/aget) (simd-index-ok? i-sym (nth node 2))
        (#{'long 'int 'double 'float} h) (simd-pure? i-sym (last node))
        (and (= 2 (count node)) (simd-op-suffix (name h))) false  ; unary +/- unsupported
        (simd-op-of node) (every? #(simd-pure? i-sym %) (simd-op-args node))
        :else false))
    :else false))

(defn- simd-arr-vts
  "All array element vts referenced (stored-to + aget-loaded) in stores."
  [ctx stores]
  (letfn [(go [node acc]
              (cond
                (and (seq? node) (= 'clojure.core/aget (first node)))
                (conj acc (get-in ctx [:elems (second node) :vt]))
                (seq? node) (reduce #(go %2 %1) acc (rest node))
                :else acc))]
    (reduce (fn [acc [arr _ v]] (go v (conj acc (get-in ctx [:elems arr :vt])))) #{} stores)))

(defn- simd-vectorizable?
  "Analyze a dotimes for SIMD. Returns {:lane-vt :lanes :stores :n :body} or nil.
   Every body form must be an (aset arr i pure) store; all arrays one f64/f32 type."
  [ctx node]
  (let [[_ [isym ncnt] & body] node
        flat (mapcat (fn f [x]
                       (cond (and (seq? x) (= 'do (first x))) (mapcat f (rest x))
                             (and (seq? x) (= 'let* (first x)) (empty? (second x))) (mapcat f (drop 2 x))
                             :else [x]))
                     body)
        stores (mapv (fn [x]
                       (when (and (seq? x) (= 'clojure.core/aset (first x)) (= 4 (count x))
                                  (simd-index-ok? isym (nth x 2)) (simd-pure? isym (nth x 3)))
                         [(nth x 1) (nth x 2) (nth x 3)]))
                     flat)]
    (when (and (seq flat) (every? some? stores))
      (let [vts (simd-arr-vts ctx stores)]
        (when (and (= 1 (count vts)) (#{:f64 :f32} (first vts)))
          (let [lane-vt (first vts)]
            {:lane-vt lane-vt :lanes (case lane-vt :f64 2 :f32 4)
             :stores stores :n ncnt :body body :isym isym}))))))

(declare emit-vexpr)

(defn- emit-vexpr
  "Emit an elementwise-pure expression as a v128 value (lane-vt-wide)."
  [ctx lane-vt node]
  (let [splat (if (= lane-vt :f64) :f64x2.splat :f32x4.splat)]
    (cond
      (number? node)
      (into (if (= lane-vt :f64) (e/f64-const node) (e/f32-const (float node))) (e/v splat))
      (symbol? node)
      (into (e/local-get (get-in ctx [:env node :idx])) (e/v splat))
      (seq? node)
      (let [h (first node)]
        (cond
          (= h 'clojure.core/aget)
          (let [[_ arr idx] node] (into (addr ctx arr idx) (e/v128-load 0 0)))
          (#{'long 'int 'double 'float} h) (emit-vexpr ctx lane-vt (last node))
          :else
          (let [suf (simd-op-of node)
                [o1 o2] (simd-op-args node)
                pfx (if (= lane-vt :f64) "f64x2" "f32x4")]
            (-> (emit-vexpr ctx lane-vt o1)
                (into (emit-vexpr ctx lane-vt o2))
                (into (e/v (keyword (str pfx "." suf)))))))))))

(defn- emit-dotimes-simd
  "Vectorized counted loop: v128 main loop (step = lanes) + scalar remainder.
     i=0; block{ loop{ (i+lanes)>n → br1; <v128 stores>; i+=lanes; br0 }}
          block{ loop{ i>=n → br1; <scalar stores>; i++; br0 }}"
  [ctx node {:keys [lane-vt lanes stores n body isym]}]
  (let [idx  (alloc! ctx :i32)
        ctx' (assoc-in ctx [:env isym] {:idx idx :vt :i32})
        vmain (-> (e/local-get idx) (into (e/i32-const lanes)) (into (e/i :i32.add))
                  (into (emit-val ctx' n)) (into (e/i :i32.gt_s)) (into (e/br-if 1))
                  (into (vec (mapcat (fn [[arr idx-node v]]
                                       (-> (addr ctx' arr idx-node)
                                           (into (emit-vexpr ctx' lane-vt v))
                                           (into (e/v128-store 0 0))))
                                     stores)))
                  (into (e/local-get idx)) (into (e/i32-const lanes)) (into (e/i :i32.add))
                  (into (e/local-set idx)) (into (e/br 0)))
        srem (-> (e/local-get idx) (into (emit-val ctx' n)) (into (e/i :i32.ge_s)) (into (e/br-if 1))
                 (into (vec (mapcat #(emit-effect ctx' %) body)))
                 (into (e/local-get idx)) (into (e/i32-const 1)) (into (e/i :i32.add))
                 (into (e/local-set idx)) (into (e/br 0)))]
    (-> (e/i32-const 0) (into (e/local-set idx))
        (into (e/block (e/loop* vmain)))
        (into (e/block (e/loop* srem))))))

(defn- counted-inc?
  "node increments i-sym by exactly 1 (inc / unchecked-inc-int / unchecked-inc)."
  [i-sym node]
  (and (seq? node) (symbol? (first node))
       (#{"inc" "unchecked-inc-int" "unchecked-inc"} (name (first node)))
       (= (second node) i-sym)))

(defn- simd-loop-plan
  "Match a vectorizable counted loop the passes lower dotimes into:
     (loop* [i 0] (if (< i n) (do <aset-stores…> (recur (inc i))) els))
   Returns {:lane-vt :lanes :stores :n :isym :els :then-body} or nil."
  [ctx node]
  (let [[_ bindvec ifform] node]
    (when (and (vector? bindvec) (= 2 (count bindvec))
               (seq? ifform) (= 4 (count ifform)) (= 'if (first ifform)))
      (let [isym (first bindvec)
            init (second bindvec)
            zero? (or (= init 0) (and (seq? init) (#{'int 'long} (first init)) (= 0 (second init))))
            [_ cnd then els] ifform]
        (when (and zero? (symbol? isym)
                   (seq? cnd) (symbol? (first cnd)) (= "<" (name (first cnd))) (= isym (second cnd))
                   (seq? then) (= 'do (first then)))
          (let [n      (nth cnd 2)
                stores (mapv (fn [x]
                               (when (and (seq? x) (= 'clojure.core/aset (first x)) (= 4 (count x))
                                          (simd-index-ok? isym (nth x 2)) (simd-pure? isym (nth x 3)))
                                 [(nth x 1) (nth x 2) (nth x 3)]))
                             (butlast (rest then)))
                tail   (last then)]
            (when (and (seq stores) (every? some? stores)
                       (seq? tail) (= 'recur (first tail)) (counted-inc? isym (second tail)))
              (let [vts (simd-arr-vts ctx stores)]
                (when (and (= 1 (count vts)) (#{:f64 :f32} (first vts)))
                  (let [lane-vt (first vts)]
                    {:lane-vt lane-vt :lanes (if (= lane-vt :f64) 2 4)
                     :stores stores :n n :isym isym :els els :then-body (vec (rest then))}))))))))))

(defn- emit-loop-simd
  "Vectorized counted loop → [bytes ret-vt]. v128 main loop (step = lanes) + scalar
   remainder, then the loop's else value (matches emit-loop-scalar's return)."
  [ctx node {:keys [lane-vt lanes stores n isym els then-body]}]
  (let [idx  (alloc! ctx :i32)
        ctx' (assoc-in ctx [:env isym] {:idx idx :vt :i32})
        vmain (-> (e/local-get idx) (into (e/i32-const lanes)) (into (e/i :i32.add))
                  (into (emit-val ctx' n)) (into (e/i :i32.gt_s)) (into (e/br-if 1))
                  (into (vec (mapcat (fn [[arr idx-node v]]
                                       (-> (addr ctx' arr idx-node)
                                           (into (emit-vexpr ctx' lane-vt v))
                                           (into (e/v128-store 0 0))))
                                     stores)))
                  (into (e/local-get idx)) (into (e/i32-const lanes)) (into (e/i :i32.add))
                  (into (e/local-set idx)) (into (e/br 0)))
        srem (-> (e/local-get idx) (into (emit-val ctx' n)) (into (e/i :i32.ge_s)) (into (e/br-if 1))
                 (into (vec (mapcat #(emit-effect ctx' %) (butlast then-body))))  ; stores (drop recur)
                 (into (e/local-get idx)) (into (e/i32-const 1)) (into (e/i :i32.add))
                 (into (e/local-set idx)) (into (e/br 0)))]
    [(-> (e/i32-const 0) (into (e/local-set idx))
         (into (e/block (e/loop* vmain)))
         (into (e/block (e/loop* srem)))
         (into (emit-val ctx' els)))
     (infer-vt ctx' els)]))

(defn- emit-dotimes
  "(dotimes [i n] body…) → a void counted loop. Emits:
     i = 0
     block { loop { br_if 1 (i >= n) ; body-effects ; i++ ; br 0 } }
   Labels from inside the loop body: loop=0, block=1.
   When ctx :simd? is set and the loop is an elementwise map, emits a v128 main
   loop + scalar remainder instead."
  [ctx node]
  (if-let [plan (and (:simd? ctx) (simd-vectorizable? ctx node))]
    (emit-dotimes-simd ctx node plan)
    (emit-dotimes-scalar ctx node)))

(defn- emit-dotimes-scalar
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

    ;; void-bound symbol in tail position (e.g. (let* [_ret (dotimes …)] _ret))
    (and (symbol? node) (get-in ctx [:env node :void])) [[] nil]

    (and (seq? node) (= 'let* (first node)))
    (let [[_ binds & body] node
          [ctx' init-bytes] (reduce (fn [[c acc] [s init]]
                                      (if (void-effect-form? init)
                                        [(assoc-in c [:env s] {:void true})
                                         (into acc (emit-effect c init))]
                                        (let [vt (infer-vt c init)
                                              idx (alloc! c vt)]
                                          [(assoc-in c [:env s] {:idx idx :vt vt})
                                           (-> acc (into (emit-val c init)) (into (e/local-set idx)))])))
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
   :simd? enables SIMD128 vectorization of elementwise dotimes-maps.
   Returns {:bytes byte[] :name :param-types [vt...] :result-types [vt...]}."
  [{:keys [name params ir mem-pages simd?] :or {mem-pages 256}}]
  (let [param-vts (mapv #(tag->vt (:tag %)) params)
        env0 (into {} (map-indexed (fn [idx {:keys [sym tag]}] [sym {:idx idx :vt (tag->vt tag)}]) params))
        elems (into {} (keep (fn [{:keys [sym tag]}] (when (array-tag? tag) [sym (array-tag->elem tag)])) params))
        ctx {:env env0 :elems elems :base (count params) :locals (atom []) :simd? simd?}
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
