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
  (:require [raster.compiler.backend.wasm.encoder :as e]
            [raster.compiler.backend.wasm.transcendental :as tr]
            [raster.compiler.backend.intrinsics :as ix]))

;; --- tag → wasm types ------------------------------------------------------
(def ^:private array-tag->elem
  {'doubles {:vt :f64 :bytes 8 :align 3 :load :f64.load :store :f64.store}
   'floats  {:vt :f32 :bytes 4 :align 2 :load :f32.load :store :f32.store}
   'longs   {:vt :i64 :bytes 8 :align 3 :load :i64.load :store :i64.store}
   'ints    {:vt :i32 :bytes 4 :align 2 :load :i32.load :store :i32.store}
   'bytes   {:vt :i32 :bytes 1 :align 0 :load :i32.load8_s :store :i32.store8}})

(defn- array-tag? [tag] (contains? array-tag->elem tag))

(defn- tag->vt
  "Scalar raster/annotation tag → wasm valtype. Arrays are i32 pointers; long/int
   scalars are i32 in v1 — 32-bit linear memory, so indices/counters are exact, but a
   genuine 64-bit `long` value would TRUNCATE. Kernels here use long only as an
   index/counter (the dominant case); true i64 math needs a dedicated path (see the
   'longs reject-guard in compile-fn). Out-of-range array indices are unchecked (C-like,
   by design): an OOB index traps or hits adjacent memory."
  [tag]
  (cond
    (#{'double 'Double} tag) :f64
    (#{'float 'Float} tag)   :f32
    (#{'long 'int 'Long 'Integer} tag) :i32
    (array-tag? tag) :i32
    :else :i32))

;; Operator/intrinsic lowering (which wasm opcode for op@vt) is owned by
;; raster.compiler.backend.intrinsics — emit dispatches through ix/wasm-op.

;; ctx: {:env {sym {:idx :vt}}  :elems {sym elem-map}}
(declare emit-val infer-vt emit-effect alloc! emit-intrinsic emit-loop ends-in-recur? emit-bindings emit-operand-at)

(defn- addr
  "byte address of array element: ptr + idx*elem-bytes  (i32)."
  [ctx arr-sym idx-node]
  (let [pidx (get-in ctx [:env arr-sym :idx])
        eb   (get-in ctx [:elems arr-sym :bytes])]
    (-> (e/local-get pidx)
        (into (emit-val ctx idx-node))
        (into (e/i32-const eb)) (into (e/i :i32.mul)) (into (e/i :i32.add)))))

(defn- void-effect-form?
  "A form producing no value (side-effect only): nil, an array store (aset),
   dotimes/when, a statement loop ending in nil, or a do/let* whose TAIL is
   itself void. Such a form bound in a let* — e.g. the inliner's
   (let* [_ret (dotimes …)] _ret) body-lift, the par lowering's
   (let* [body_result (let* [n …] (dotimes …) nil)] body_result), or trailing
   `(aset …)` statements lifted to `_eff_` bindings — is emitted as an effect,
   not a value."
  [node]
  (or (nil? node)
      (and (seq? node)
           (let [h (first node)]
             (or (#{'dotimes 'when} h)
                 (and (symbol? h) (= "aset" (name h)))
                 ;; statement loop: (loop [..] (if c <recur…> nil)) — void exit
                 (and (#{'loop 'loop*} h)
                      (let [[_ _ ifform] node]
                        (and (seq? ifform) (= 'if (first ifform))
                             (let [[_ _ t e] ifform]
                               (or (and (ends-in-recur? t) (void-effect-form? e))
                                   (and (ends-in-recur? e) (void-effect-form? t)))))))
                 (and (#{'do 'let* 'let} h) (void-effect-form? (last node))))))))

(defn- synth-case*
  "Lower the macroexpanded `case*` special form
     (case* test shift mask default {hash [testval result] …} switch-type test-type)
   into a let*-bound nested-if chain: the test is bound once and compared against
   each integer testval. The exhaustive-case `default` (a throw) becomes the final
   else and lowers to `unreachable`. Only `:int` test-types occur in numeric
   kernels (the walker keys cases on integers)."
  [args]
  (let [[test _shift _mask default clause-map] args
        g (gensym "case_g_")
        clauses (vals clause-map)]                       ; each = [testval result]
    (list 'let* [g test]
          (reduce (fn [els [k e]] (list 'if (list 'clojure.core/== g k) e els))
                  default (reverse clauses)))))

(defn- const-if-taken
  "For an `if` whose condition is a compile-time constant, which branch is statically
   taken: :then (true / a keyword — e.g. a `cond`'s :else), :else (nil / false), or nil
   (a real runtime condition). wasm has no keyword/nil truthiness, so these must fold."
  [c]
  (cond (or (true? c) (keyword? c)) :then
        (or (nil? c) (false? c))    :else
        :else nil))

(defn- sym-vt
  "Valtype from the walker's stamped :raster.type/tag on a binding symbol — the
   single type source (TC-fed). nil when unstamped (pass-introduced bindings)."
  [s]
  (when-let [tag (:raster.type/tag (meta s))]
    (case tag
      double :f64  float :f32  (long int byte short char boolean) :i32
      nil)))                                     ; array/ref tags: not a scalar local

(defn- emit-bindings
  "Lower a let* binding vector → [ctx' init-bytes]. The local's type comes from the
   STAMPED symbol tag (walker/TC — the single type source); init inference is only
   the fallback for pass-introduced unstamped bindings. Init values are coerced to
   the local's valtype (emit-operand-at), so a widened local takes a narrower init."
  [ctx binds]
  (reduce (fn [[c acc] [s init]]
            (if (void-effect-form? init)
              [(assoc-in c [:env s] {:void true}) (into acc (emit-effect c init))]
              (let [vt (or (sym-vt s) (infer-vt c init))
                    idx (alloc! c vt)]
                [(assoc-in c [:env s] {:idx idx :vt vt})
                 (-> acc (into (emit-operand-at c vt init)) (into (e/local-set idx)))])))
          [ctx []] (partition 2 binds)))

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
        (and (symbol? h) (#{"long" "int"} (name h)))        ; integer cast → i32
        (let [x (first A) xv (infer-vt ctx x)]
          (cond-> (emit-val ctx x)
            (= xv :f64) (into (e/i :i32.trunc_f64_s))        ; (long (Math/floor d)) etc.
            (= xv :f32) (into (e/i :i32.trunc_f32_s))))
        (and (symbol? h) (= "double" (name h)))             ; f64 cast / literal — type-aware
        (let [x (first A)]
          (if (number? x)
            (e/f64-const (double x))
            (let [xv (infer-vt ctx x)]
              (cond-> (emit-val ctx x)
                (= xv :f32) (into (e/i :f64.promote_f32))     ; (double <f32>) → promote
                (= xv :i32) (into (e/i :f64.convert_i32_s)))))) ; (double <i32>) → convert
        (and (symbol? h) (= "float" (name h)))              ; f32 cast / literal — type-aware
        (let [x (first A)]
          (if (number? x)
            (e/f32-const x)                                  ; (float 0.0) → f32 literal
            (let [xv (infer-vt ctx x)]
              (cond-> (emit-val ctx x)
                (= xv :f64) (into (e/i :f32.demote_f64))      ; (float <f64>) → demote
                (= xv :i32) (into (e/i :f32.convert_i32_s)))))) ; (float <i32>) → convert
        (#{'clojure.core/aget 'raster.arrays/aget} h)
        (let [[arr idx] A elem (get-in ctx [:elems arr])]
          (into (addr ctx arr idx) (e/mem-load (:load elem) (:align elem) 0)))
        ;; integer step (inc/dec, incl. unchecked-*-int) — index/counter steppers
        (#{"inc" "unchecked-inc-int" "unchecked-inc"} (name h))
        (-> (emit-val ctx (A 0)) (into (e/i32-const 1)) (into (e/i :i32.add)))
        (#{"dec" "unchecked-dec-int" "unchecked-dec"} (name h))
        (-> (emit-val ctx (A 0)) (into (e/i32-const 1)) (into (e/i :i32.sub)))
        ;; numeric predicates → compare against a typed zero (bool/i32 result)
        (#{"zero?" "pos?" "neg?"} (name h))
        (let [x (A 0) vt (infer-vt ctx x)
              zc (case vt :f64 (e/f64-const 0.0) :f32 (e/f32-const 0.0) (e/i32-const 0))
              k  (case (name h) "zero?" :eq "pos?" :gt "neg?" :lt)]
          (-> (emit-val ctx x) (into zc) (into (e/i (ix/wasm-op k vt)))))
        ;; value-position if → typed if/else block. Constant-truthy conditions
        ;; (a `cond`'s :else, or true/nil/false) fold to the taken branch — wasm
        ;; has no notion of a keyword/nil "condition".
        (= h 'if)
        (let [[c t e] A]
          (case (const-if-taken c)
            :then (emit-val ctx t)
            :else (emit-val ctx e)
            (let [tv (infer-vt ctx t)
                  vt (if (= tv :i32) (infer-vt ctx e) tv)]
              (-> (emit-val ctx c)
                  (into (e/if-t vt (emit-operand-at ctx vt t) (emit-operand-at ctx vt e)))))))
        ;; case* (macroexpanded) → lower to a let*+if-chain, emitted via normal path
        (= h 'case*)
        (emit-val ctx (synth-case* A))
        ;; exhaustive-case default / unreachable code
        (= h 'throw)
        (e/i :unreachable)
        ;; loop in VALUE position (e.g. an inlined reduction as a let* binding init)
        ;; — emit-loop already yields a value-producing block(result vt); take bytes.
        (#{'loop 'loop*} h)
        (first (emit-loop ctx node))
        ;; array access .invk (aget is a deftm and may arrive devirtualized OR
        ;; survive as an un-inlined callee) — ALWAYS emit through the elems
        ;; machinery: outlining it as a sibling function loses the per-array
        ;; element type (the callee monomorphizes under the CALLER's dtype and
        ;; e.g. f32-loads an int array — seen in qgemm's int-aget helper).
        (and (= h '.invk)
             (let [op (:raster.op/original (meta node))]
               (contains? #{'raster.arrays/aget 'clojure.core/aget} op)))
        (emit-val ctx (with-meta (cons (:raster.op/original (meta node)) (rest A))
                                 (meta node)))
        ;; non-intrinsic deftm callee kept as a real wasm function (not inlined):
        ;; push args, then `call` its function index. Intrinsics (raster.numeric,
        ;; Math, …) fall through to inline emission below.
        (and (= h '.invk)
             (let [op (:raster.op/original (meta node))]
               (and (not (and (symbol? op) (ix/canonical op)))
                    (contains? (:fn-index ctx) (first A)))))
        (-> (vec (mapcat #(emit-val ctx %) (drop 1 A)))   ; args (drop the impl sym)
            (into (e/call (get (:fn-index ctx) (first A)))))
        ;; devirtualized typed call: op + element vt from carried metadata
        (= h '.invk)
        (let [tag (:raster.type/tag (meta node))
              vt  (case tag
                    double :f64, float :f32, (long int byte short char boolean) :i32
                    ;; No semantic tag — a devirtualization site failed to stamp it
                    ;; (the walker, inline-invk, and resolve-generic-deftm-calls all
                    ;; should). Recover the element type from the first operand
                    ;; rather than silently defaulting to i32, which would miscompile
                    ;; f64 arithmetic to integer ops. Warn so the missing stamp is
                    ;; visible instead of producing wrong code.
                    (let [v (infer-vt ctx (first (drop 1 A)))]
                      (binding [*out* *err*]
                        (println (str "WARNING: wasm emit — .invk " (:raster.op/original (meta node))
                                      " has no :raster.type/tag; inferred " v
                                      " from its operand. A devirtualization site didn't stamp it.")))
                      v))]
          (emit-intrinsic ctx (:raster.op/original (meta node)) (drop 1 A) vt))
        ;; par/dp4a — 4-way signed-int8 dot + accumulate: the ONE quantization
        ;; primitive, spelled per backend (CUDA __dp4a / AMD sdot4 / OpenCL helper /
        ;; CPU-C dpbusd). Wasm Tier-1 scalar: extract sign-extended bytes by
        ;; shl/shr_s pairs, 4 muls, chained adds onto acc. (Tier-2 SIMD via the
        ;; existing i32x4.dot int8-dot loop plan when the enclosing loop matches.)
        (= :dp4a (ix/canonical h))    ; any spelling — canonical collapses them
        (let [[a b acc] A
              la (alloc! ctx :i32) lb (alloc! ctx :i32)
              lane (fn [l k]                               ; sign-extended byte k
                     (let [sh (- 24 (* 8 k))]
                       (cond-> (e/local-get l)
                         (pos? sh) (-> (into (e/i32-const sh)) (into (e/i :i32.shl)))
                         true      (-> (into (e/i32-const 24)) (into (e/i :i32.shr_s))))))]
          (-> (emit-val ctx a) (into (e/local-set la))
              (into (emit-val ctx b)) (into (e/local-set lb))
              (into (emit-val ctx acc))
              (into (vec (mapcat (fn [k]
                                   (-> (lane la k) (into (lane lb k))
                                       (into (e/i :i32.mul)) (into (e/i :i32.add))))
                                 (range 4))))))
        ;; any registered numeric op / intrinsic (arith, comparison, rem/mod,
        ;; Math, min/max) — dispatched through backend.intrinsics (single table).
        (and (symbol? h) (ix/canonical h))
        (emit-intrinsic ctx h A nil)
        ;; let* / do in VALUE position (introduced by inlined deftm calls)
        (#{'let* 'let} h)
        (let [[binds & body] A
              [ctx' init-bytes] (emit-bindings ctx binds)
              tail (if (= 1 (count body)) (first body) (cons 'do body))]
          (into init-bytes (emit-val ctx' tail)))
        (= h 'do)
        (into (vec (mapcat #(emit-effect ctx %) (butlast A))) (emit-val ctx (last A)))
        :else (throw (ex-info (str "unhandled value head " h) {:node node}))))
    (nil? node) []                                  ; void leaf (statement-loop exit, if-void arm)
    :else (throw (ex-info (str "unhandled value node " (pr-str node)) {}))))

(defn- emit-rem-mod
  "rem/mod via the registry's :special path. i32 → native rem_s; float → a - q*b
   (q = floor for mod, trunc for rem). a,b are pure value exprs (loads + arith)
   so duplicating a is sound."
  [ctx k a b]
  (let [av (infer-vt ctx a)
        vt (if (#{:f64 :f32} av) av :i32)]
    (if (= vt :i32)
      (-> (emit-val ctx a) (into (emit-val ctx b)) (into (e/i :i32.rem_s)))
      (let [f64? (= vt :f64)
            qop (if (= k :mod) (if f64? :f64.floor :f32.floor) (if f64? :f64.trunc :f32.trunc))
            dop (if f64? :f64.div :f32.div) mop (if f64? :f64.mul :f32.mul) sop (if f64? :f64.sub :f32.sub)]
        (-> (emit-val ctx a)                                       ; a
            (into (emit-val ctx a)) (into (emit-val ctx b)) (into (e/i dop)) (into (e/i qop)) ; q
            (into (emit-val ctx b)) (into (e/i mop))               ; q*b
            (into (e/i sop)))))))                                  ; a - q*b

(defn- emit-operand-at
  "Emit `node`, coercing its value to valtype `vt` when its own type differs — so a
   mixed-type op (e.g. `(>= (double dx) (long 0))`, valid in Clojure via numeric
   promotion) gets matching operand types on the wasm stack instead of an i32/f64
   mismatch. Only the genuine cross-type cases insert a conversion."
  [ctx vt node]
  (let [nv (infer-vt ctx node)
        b  (emit-val ctx node)]
    (cond
      (= nv vt) b
      (= nv :diverge) b                       ; throw/unreachable: value never flows
      (and (= vt :f64) (= nv :i32)) (into b (e/i :f64.convert_i32_s))
      (and (= vt :f32) (= nv :i32)) (into b (e/i :f32.convert_i32_s))
      (and (= vt :f64) (= nv :f32)) (into b (e/i :f64.promote_f32))
      (and (= vt :f32) (= nv :f64)) (into b (e/i :f32.demote_f64))
      (and (= vt :i32) (= nv :f64)) (into b (e/i :i32.trunc_f64_s))
      (and (= vt :i32) (= nv :f32)) (into b (e/i :i32.trunc_f32_s))
      :else b)))

(defn- emit-intrinsic
  "Emit a registered numeric op / intrinsic via the backend.intrinsics table.
   `op` is the canonical-able op form; `args` the (already op-stripped) operands;
   `vt-hint` the element valtype from .invk metadata (nil → infer from operands).
   Comparisons take their opcode at the operand vt but yield i32 (bool)."
  [ctx op args vt-hint]
  (let [k (ix/canonical op)
        d (ix/descriptor op)
        operand-vt (fn [] (let [v0 (infer-vt ctx (first args))]
                            (if (#{:f64 :f32} v0) v0 (infer-vt ctx (second args)))))]
    (case (:kind d)
      :special (emit-rem-mod ctx k (first args) (second args))
      :cmp (let [vt (operand-vt)
                 opk (or (ix/wasm-op k vt) (throw (ex-info (str "no wasm lowering for " k " @ " vt) {:op op})))]
             (-> (emit-operand-at ctx vt (first args)) (into (emit-operand-at ctx vt (second args))) (into (e/i opk))))
      :infix (if (and (= k :-) (= 1 (count args)))
               ;; unary minus → negate (the n-ary fold would wrongly emit identity)
               (let [vt (or vt-hint (infer-vt ctx (first args)))]
                 (case vt
                   :f64 (-> (emit-val ctx (first args)) (into (e/i :f64.neg)))
                   :f32 (-> (emit-val ctx (first args)) (into (e/i :f32.neg)))
                   (-> (e/i32-const 0) (into (emit-val ctx (first args))) (into (e/i :i32.sub)))))
               (let [vt (or vt-hint (operand-vt))
                     opk (or (ix/wasm-op k vt) (throw (ex-info (str "no wasm lowering for " k " @ " vt) {:op op})))]
                 ;; left-fold n-ary operands so (* a b c) can't silently drop c;
                 ;; coerce each to the op vt (mixed f64/int operands)
                 (reduce (fn [acc a] (-> acc (into (emit-operand-at ctx vt a)) (into (e/i opk))))
                         (emit-operand-at ctx vt (first args)) (rest args))))
      :fn (if (ix/wasm-poly? k)
            ;; transcendental → inline polynomial form, emitted via the normal path.
            ;; Args are ALWAYS cast-wrapped to f64 so the synthetic form is
            ;; homogeneous by construction — a raw f32 arg inside an f64 poly
            ;; otherwise makes individual ops pick f32 (first-operand rule) while
            ;; the caller believes f64 (mixed-width miscompile). f32 result demotes.
            (let [vt (or vt-hint (infer-vt ctx (first args)))
                  form (tr/form k (map #(list 'double %) args))]
              (if (= vt :f32)
                (emit-val ctx (list 'float form))
                (emit-val ctx form)))
            (let [vt (or vt-hint (infer-vt ctx (first args)))
                  opk (or (ix/wasm-op k vt)
                          (throw (ex-info (str "no wasm lowering for intrinsic " k " @ " vt
                                               " (transcendental? needs import/polynomial)") {:op op})))]
              (if (= 2 (:arity d))
                (-> (emit-operand-at ctx vt (first args)) (into (emit-operand-at ctx vt (second args))) (into (e/i opk)))
                (-> (emit-operand-at ctx vt (first args)) (into (e/i opk))))))
      (throw (ex-info (str "intrinsic not emittable: " op) {:op op :kind (:kind d)})))))

(def ^:dynamic *lenient-types?*
  "When true, missing type info warns and defaults to i32 (the historical behavior)
   instead of throwing. Types come from walker/TC stamps — a miss is a front-half
   bug; bind this only to diagnose."
  false)

(defn- warn-unknown-vt
  "G1: surface a type-inference miss (mirrors the .invk loud-recovery policy) and
   default to :i32, instead of *silently* assuming i32 — a silent i32 on a float expr
   miscompiles. A warning here means a node reached the emitter without a resolvable
   type: a missing :raster.type/tag stamp, an unhandled form head, or an unbound symbol."
  [node why]
  (if *lenient-types?*
    (do (binding [*out* *err*]
          (println (str "WARNING: wasm infer-vt — no type for " why ": " (pr-str node)
                        " → defaulting to i32 (may miscompile a float expr).")))
        :i32)
    (throw (ex-info (str "wasm emit: no type for " why " — a node reached the emitter "
                         "without a resolvable :raster.type/tag (front-half stamping gap)")
                    {:node node :why why}))))

(defn- infer-vt
  "node-vt: the TOTAL type reader (never an inferrer). Sources, in order:
   stamped :raster.type/tag (walker/TC — symbols and forms), a literal's Clojure
   class (0.5 IS a double; retyping a literal is the narrowing pass's job via
   cast-wrapping), the binding env, and a CLOSED set of dialect rules for the
   untagged heads the lowered dialect defines (clojure.core index arith → i32,
   comparisons/predicates → i32, casts, aget element types, dp4a → i32,
   control forms = their tail, synthetic math forms = their first operand —
   homogeneous by construction). Anything else THROWS (front-half stamping gap);
   bind *lenient-types?* only to diagnose. Emission emits exactly this type and
   consumers coerce at typed slots — prediction ≡ lookup, divergence impossible."
  [ctx node]
  (cond
    (symbol? node)  (if (contains? (:env ctx) node)
                      (get-in ctx [:env node :vt] :i32)   ; in env w/o :vt = a void binding
                      (or (sym-vt node)                    ; stamped tag (walker/TC)
                          (warn-unknown-vt node "unbound symbol")))
    (float? node)   :f64
    (integer? node) :i32
    (seq? node)
    (case (:raster.type/tag (meta node))
      double :f64, float :f32, (long int byte short char boolean) :i32
      (let [h (first node)]
        (cond
          (and (symbol? h) (= "float" (name h)))  :f32
          (and (symbol? h) (= "double" (name h))) :f64
          (and (symbol? h) (#{"long" "int"} (name h))) :i32 ; an integer cast yields i32
          ;; let* — bind its locals so the tail's refs resolve (a let* used as an
          ;; operand, e.g. (/ (let* …) (let* …)), must infer its tail's vt, which
          ;; may reference a binding introduced inside it).
          (#{'let* 'let} h) (let [c' (reduce (fn [c [s init]]
                                               (assoc-in c [:env s]
                                                         {:vt (or (sym-vt s) (infer-vt c init))}))
                                             ctx (partition 2 (second node)))]
                              (infer-vt c' (last node)))
          (= 'do h) (infer-vt ctx (last node))           ; value of tail
          (= 'if h) (let [c (nth node 1)]               ; fold constant-truthy conditions
                      (case (const-if-taken c)
                        :then (infer-vt ctx (nth node 2))
                        :else (infer-vt ctx (nth node 3))
                        (let [tv (infer-vt ctx (nth node 2))]
                          (if (= tv :i32) (infer-vt ctx (nth node 3)) tv))))
          (= 'case* h) (infer-vt ctx (second (first (vals (nth node 5))))) ; first clause's result
          (and (symbol? h) (#{"zero?" "pos?" "neg?"} (name h))) :i32   ; predicate → bool
          ;; integer steppers (match emit-val's inc/dec) — i32 result, made explicit
          ;; so it's intentional rather than relying on the unknown-head default
          (and (symbol? h) (#{"inc" "unchecked-inc-int" "unchecked-inc"
                              "dec" "unchecked-dec-int" "unchecked-dec"} (name h))) :i32
          (#{'loop 'loop*} h)                                         ; loop result = its non-recur branch
          (let [c' (reduce (fn [c [s init]]
                             (assoc-in c [:env s] {:vt (or (sym-vt s) (infer-vt c init))}))
                           ctx (partition 2 (nth node 1)))  ; bind loop vars so the result ref resolves
                ifform (nth node 2) then (nth ifform 2) els (nth ifform 3)]
            (infer-vt c' (if (ends-in-recur? then) els then)))
          (= 'throw h) :diverge                        ; never produces a value
          (= :dp4a (ix/canonical h)) :i32              ; int8 dot-accumulate → i32
          (#{'clojure.core/aget 'raster.arrays/aget} h) (get-in ctx [:elems (second node) :vt] :f64)
          ;; registered numeric op/intrinsic: comparisons → bool (i32); arith /
          ;; math / rem-mod → element type of first operand
          (ix/canonical h) (if (= :cmp (ix/kind h)) :i32 (infer-vt ctx (second node)))
          :else (warn-unknown-vt node (str "unhandled head " h)))))
    :else (warn-unknown-vt node "non-expression node")))

(declare emit-dotimes emit-dotimes-scalar emit-loop-void)

(defn- emit-effect
  "Side-effecting statement that leaves nothing on the stack (aset / when / do /
   let* / dotimes)."
  [ctx node]
  (if (nil? node)
    []                                              ; statement nil (void tails)
  (let [h (first node), A (vec (rest node))]
    (cond
      ;; devirtualized statement call: recover the semantic op and re-dispatch
      ;; (raster.arrays/aset is a deftm — it can arrive as (.invk impl arr i v))
      (and (= h '.invk) (:raster.op/original (meta node)))
      (emit-effect ctx (with-meta (cons (:raster.op/original (meta node)) (rest A))
                                  (meta node)))
      (#{'clojure.core/aset 'raster.arrays/aset} h)
      (let [[arr idx v] A elem (get-in ctx [:elems arr])]
        (when-not elem
          (throw (ex-info (str "aset target " arr " has no element info (not an array param?)")
                          {:arr arr :node node})))
        ;; the store's element type is ground truth — coerce the value like an
        ;; intrinsic operand (kernels' explicit (float ...) casts make this a
        ;; no-op; a mixed-width expression gets the correct convert instead of
        ;; a validation failure)
        (-> (addr ctx arr idx)
            (into (emit-operand-at ctx (:vt elem) v))
            (into (e/mem-store (:store elem) (:align elem) 0))))
      (= h 'do) (vec (mapcat #(emit-effect ctx %) A))
      ;; let* in effect position (e.g. soa-lower floats arg bindings out of a
      ;; store: (let* [args…] (do (aset …) …))). Bind locals, emit body as effects.
      (#{'let* 'let} h)
      (let [[binds & body] A
            [ctx' init-bytes] (emit-bindings ctx binds)]
        (into init-bytes (vec (mapcat #(emit-effect ctx' %) body))))
      (= h 'if)                                     ; (if c then [else]) in void position
      (let [[c t e] A]
        (case (const-if-taken c)
          :then (emit-effect ctx t)
          :else (if (some? e) (emit-effect ctx e) [])
          (-> (emit-val ctx c)
              (into [(e/op :if) e/empty-block])
              (into (emit-effect ctx t))
              (into (if (some? e) (into [(e/op :else)] (emit-effect ctx e)) []))
              (into [(e/op :end)]))))
      (= h 'case*) (emit-effect ctx (synth-case* A))
      (= h 'throw) (e/i :unreachable)               ; exhaustive-case default
      (= h 'dotimes) (emit-dotimes ctx node)
      (= h 'when) (emit-effect ctx (list 'if (second node) (list* 'do (drop 2 node))))
      ;; statement-position loop (model kernels: packing/softmax inner loops that
      ;; end in nil) — the value-loop emission with a void exit branch.
      (#{'loop 'loop*} h) (emit-loop-void ctx node)
      :else (throw (ex-info (str "unhandled effect head " h) {:node node}))))))

(defn- emit-recur
  "(recur v0 v1 ...) → push all new values COERCED to their loop-var valtypes
   (the typed slot, from the stamped loop-var tags), set loop locals in REVERSE
   (so each value is read from the OLD locals before any are overwritten), then
   branch to the loop header at depth `loop-depth`."
  [ctx args loop-idxs loop-vts loop-depth]
  (-> (vec (mapcat (fn [a vt] (emit-operand-at ctx vt a)) args loop-vts))
      (into (vec (mapcat #(e/local-set %) (reverse loop-idxs))))
      (into (e/br loop-depth))))

(defn- ends-in-recur?
  "Does this branch hand control back to the loop — a recur, possibly after a do /
   let*, or inside an if whose branches lead to a recur? An `if` with a recur on
   one side and a loop-result value on the other is a CONTROL branch (handled by
   emit-then), not a plain value, so it 'ends in recur' if EITHER side does."
  [node]
  (and (seq? node)
       (let [h (first node)]
         (or (= 'recur h)
             (and (#{'do 'let* 'let} h) (ends-in-recur? (last node)))
             (and (= 'if h) (or (ends-in-recur? (nth node 2))
                                (ends-in-recur? (nth node 3 nil))))))))

(defn- emit-then
  "Emit a loop's recur-branch: effects/bindings ending in a recur, or an `if` that
   splits into a recur side and a loop-result side. Recur args coerce to the
   loop-var valtypes; a result branch coerces to the block's ret-vt (nil ret-vt =
   void loop: result side emits as effects). `loop-depth`/`block-depth` as before;
   both increment through each nested if."
  [ctx node loop-idxs loop-vts ret-vt loop-depth block-depth]
  (cond
    (and (seq? node) (= 'recur (first node)))
    (emit-recur ctx (vec (rest node)) loop-idxs loop-vts loop-depth)
    (and (seq? node) (= 'do (first node)))
    (let [stmts (vec (rest node))]
      (-> (vec (mapcat #(emit-effect ctx %) (butlast stmts)))
          (into (emit-then ctx (last stmts) loop-idxs loop-vts ret-vt loop-depth block-depth))))
    (and (seq? node) (#{'let* 'let} (first node)))
    (let [[binds & body] (rest node)
          [ctx' init-bytes] (emit-bindings ctx binds)
          tail (if (= 1 (count body)) (first body) (cons 'do body))]
      (into init-bytes (emit-then ctx' tail loop-idxs loop-vts ret-vt loop-depth block-depth)))
    (and (seq? node) (= 'if (first node)))
    (let [[_ c a b] node
          ld (inc loop-depth), bd (inc block-depth)
          branch (fn [br-node]
                   (if (ends-in-recur? br-node)
                     (emit-then ctx br-node loop-idxs loop-vts ret-vt ld bd)
                     (-> (if (nil? ret-vt)
                           (emit-effect ctx br-node)
                           (emit-operand-at ctx ret-vt br-node))
                         (into (e/br bd)))))]
      (-> (emit-val ctx c)
          (into [(e/op :if) e/empty-block])
          (into (branch a))
          (into [(e/op :else)])
          (into (branch b))
          (into [(e/op :end)])))
    :else (throw (ex-info (str "loop recur-branch must end in recur, got " (pr-str node)) {}))))

(defn- alloc!
  "Reserve a new function-local of valtype vt, return its index. Locals are
   collected in the ctx :locals atom; index = base (after params) + position."
  [ctx vt]
  (let [a (:locals ctx)
        idx (+ (:base ctx) (count @a))]
    (swap! a conj vt)
    idx))

(declare emit-result emit-loop-scalar simd-loop-plan emit-loop-simd)

(defn- strip-int-cast
  "Strip (long …)/(int …) wrappers to reach the underlying expression."
  [x]
  (if (and (seq? x) (#{'long 'int} (first x))) (recur (second x)) x))

(defn- aget-call
  "Normalize an aget call to [arr index], handling BOTH the direct
   (clojure.core/aget arr i) form and the devirtualized
   (.invk raster.arrays/aget…-impl arr i) form — keyed on the :raster.op/original
   semantic-op metadata, never the mangled impl name (per the compiler's
   semantic-identity contract). Returns nil for non-aget forms."
  [e]
  (cond
    (not (seq? e)) nil
    (= 'clojure.core/aget (first e)) [(nth e 1) (nth e 2)]
    (= '.invk (first e))
    (let [op (:raster.op/original (meta e))]
      (when (and (symbol? op) (= "aget" (name op)))
        [(nth e 2) (nth e 3)]))            ; (.invk impl arr idx …)
    :else nil))

(defn- byte-aget-arr
  "If `expr` is (aget arr i-sym) on an int8 (byte, elem-size 1) array indexed
   exactly by i-sym, return arr; else nil. Robust to devirtualization."
  [ctx expr i-sym]
  (when-let [[arr idx] (aget-call (strip-int-cast expr))]
    (when (and (= 1 (get-in ctx [:elems arr :bytes]))
               (= (strip-int-cast idx) i-sym))
      arr)))

(defn- int8-dot-loop-plan
  "Recognize the int8 quantized dot reduction:
     (loop* [i 0 acc 0]
       (if (< i n) (recur (inc i) (+ acc (* (aget x i) (aget y i)))) acc))
   over two int8 (byte) arrays x,y. Returns {:i-sym :acc-sym :x :y :n-node} or nil.
   This is the pattern that maps to i32x4.dot_i16x8_s (widen i8→i16, dot, accumulate)."
  [ctx node]
  (when (and (seq? node) (= 'loop* (first node)))
    (let [[_ bindvec ifform] node
          pairs (partition 2 bindvec)]
      (when (and (= 2 (count pairs)) (seq? ifform) (= 'if (first ifform)))
        (let [[i-sym i-init] (first pairs)
              [acc-sym acc-init] (second pairs)
              [_ cnd then els] ifform]
          (when (and (= 0 i-init) (= 0 acc-init) (symbol? i-sym) (symbol? acc-sym)
                     (= els acc-sym)                                   ; recur-in-then, else returns acc
                     (seq? cnd) (= 'clojure.core/< (first cnd))
                     (= (strip-int-cast (nth cnd 1)) i-sym)
                     (seq? then) (= 'recur (first then)))
            (let [n-node (nth cnd 2)
                  [_ i-step acc-step] then
                  is (strip-int-cast i-step)
                  as (strip-int-cast acc-step)]
              (when (and (seq? is) (= 'clojure.core/inc (first is)) (= (strip-int-cast (second is)) i-sym)
                         (seq? as) (= 'clojure.core/+ (first as)) (= (strip-int-cast (nth as 1)) acc-sym))
                (let [prod (strip-int-cast (nth as 2))]
                  (when (and (seq? prod) (= 'clojure.core/* (first prod)))
                    (let [x (byte-aget-arr ctx (nth prod 1) i-sym)
                          y (byte-aget-arr ctx (nth prod 2) i-sym)]
                      (when (and x y)
                        {:i-sym i-sym :acc-sym acc-sym :x x :y y :n-node n-node}))))))))))))

(defn- emit-loop-int8-dot
  "Emit a v128 int8 dot: 16-wide main loop (widen i8→i16, i32x4.dot_i16x8_s,
   accumulate into an i32x4) + horizontal sum + scalar remainder. → [bytes :i32]."
  [ctx {:keys [i-sym acc-sym x y n-node]}]
  (let [i-idx    (alloc! ctx :i32)
        acc-idx  (alloc! ctx :i32)
        vacc-idx (alloc! ctx :v128)
        vx-idx   (alloc! ctx :v128)
        vy-idx   (alloc! ctx :v128)
        ctx'     (-> ctx (assoc-in [:env i-sym] {:idx i-idx :vt :i32})
                     (assoc-in [:env acc-sym] {:idx acc-idx :vt :i32}))
        n-bytes  (fn [] (emit-val ctx' n-node))         ; re-emit n each use (pure load)
        gi       (fn [] (e/local-get i-idx))
        vload    (fn [arr] (-> (addr ctx' arr i-sym) (into (e/v128-load 0 0))))
        ;; main loop body: break when i+16 > n, else dot & accumulate, i += 16
        main-body (-> []
                      (into (gi)) (into (e/i32-const 16)) (into (e/i :i32.add))
                      (into (n-bytes)) (into (e/i :i32.gt_s)) (into (e/br-if 1))
                      (into (vload x)) (into (e/local-set vx-idx))
                      (into (vload y)) (into (e/local-set vy-idx))
                      (into (e/local-get vx-idx)) (into (e/v :i16x8.extend-low-i8x16-s))
                      (into (e/local-get vy-idx)) (into (e/v :i16x8.extend-low-i8x16-s))
                      (into (e/v :i32x4.dot-i16x8-s))
                      (into (e/local-get vx-idx)) (into (e/v :i16x8.extend-high-i8x16-s))
                      (into (e/local-get vy-idx)) (into (e/v :i16x8.extend-high-i8x16-s))
                      (into (e/v :i32x4.dot-i16x8-s))
                      (into (e/v :i32x4.add))
                      (into (e/local-get vacc-idx)) (into (e/v :i32x4.add)) (into (e/local-set vacc-idx))
                      (into (gi)) (into (e/i32-const 16)) (into (e/i :i32.add)) (into (e/local-set i-idx))
                      (into (e/br 0)))
        hsum (-> (e/local-get acc-idx)
                 (into (e/local-get vacc-idx)) (into (e/v-lane :i32x4.extract-lane 0)) (into (e/i :i32.add))
                 (into (e/local-get vacc-idx)) (into (e/v-lane :i32x4.extract-lane 1)) (into (e/i :i32.add))
                 (into (e/local-get vacc-idx)) (into (e/v-lane :i32x4.extract-lane 2)) (into (e/i :i32.add))
                 (into (e/local-get vacc-idx)) (into (e/v-lane :i32x4.extract-lane 3)) (into (e/i :i32.add))
                 (into (e/local-set acc-idx)))
        ;; scalar remainder: while i < n: acc += x[i]*y[i]; i++
        rem-body (-> []
                     (into (gi)) (into (n-bytes)) (into (e/i :i32.ge_s)) (into (e/br-if 1))
                     (into (e/local-get acc-idx))
                     (into (addr ctx' x i-sym)) (into (e/mem-load :i32.load8_s 0 0))
                     (into (addr ctx' y i-sym)) (into (e/mem-load :i32.load8_s 0 0))
                     (into (e/i :i32.mul)) (into (e/i :i32.add)) (into (e/local-set acc-idx))
                     (into (gi)) (into (e/i32-const 1)) (into (e/i :i32.add)) (into (e/local-set i-idx))
                     (into (e/br 0)))]
    [(-> []
         (into (e/i32-const 0)) (into (e/local-set i-idx))
         (into (e/i32-const 0)) (into (e/local-set acc-idx))
         (into (e/i32-const 0)) (into (e/v :i32x4.splat)) (into (e/local-set vacc-idx))
         (into (e/block (e/loop* main-body)))
         (into hsum)
         (into (e/block (e/loop* rem-body)))
         (into (e/local-get acc-idx)))
     :i32]))

(defn- emit-loop
  "(loop [v init ...] (if cond <recur|do…recur> <result>)) → [bytes ret-vt].
   When ctx :simd? is set and the loop is an elementwise counted map, emits a
   v128 main loop + scalar remainder; otherwise the scalar loop. int8 dot
   reductions get a dedicated i32x4.dot_i16x8_s vectorization."
  [ctx node]
  (cond
    (and (:simd? ctx) (int8-dot-loop-plan ctx node))
    (emit-loop-int8-dot ctx (int8-dot-loop-plan ctx node))
    (and (:simd? ctx) (simd-loop-plan ctx node))
    (emit-loop-simd ctx node (simd-loop-plan ctx node))
    :else (emit-loop-scalar ctx node)))

(defn- emit-loop-scalar
  "(loop [v init ...] (if cond <recur|do…recur> <result>)) → [bytes ret-vt].
   Returns the non-recur (else) value via block(result vt){ loop ; unreachable }."
  [ctx node]
  (let [[_ bindvec ifform] node
        pairs (vec (partition 2 bindvec))
        ;; allocate loop-var locals; inits emitted with the OUTER env (loop vars
        ;; not yet in scope for their own inits)
        loop-vars (mapv (fn [[sym init]]
                          ;; stamped symbol tag first (TC loop inference via walker)
                          (let [vt (or (sym-vt sym) (infer-vt ctx init))]
                            {:sym sym :idx (alloc! ctx vt) :vt vt :init init}))
                        pairs)
        inits (vec (mapcat (fn [{:keys [idx vt init]}]
                             (into (emit-operand-at ctx vt init) (e/local-set idx))) loop-vars))
        ctx' (reduce (fn [c {:keys [sym idx vt]}] (assoc-in c [:env sym] {:idx idx :vt vt}))
                     ctx loop-vars)
        loop-idxs (mapv :idx loop-vars)
        [_ cnd then els] ifform
        ;; the loop's `if` may have the recur in EITHER branch:
        ;;   (if cont? (recur …) result)   — recur in then
        ;;   (if done? result (recur …))   — recur in else (e.g. fbm reductions)
        recur-then? (ends-in-recur? then)
        result (if recur-then? els then)
        ret-vt (infer-vt ctx' result)
        loop-vts (mapv :vt loop-vars)
        ;; depths from inside the if: if=0, loop=1, block=2. The result branch
        ;; leaves its value (coerced to ret-vt) then br 2; the recur branch br 1.
        if-bytes (if recur-then?
                   (-> (emit-val ctx' cnd)
                       (into [(e/op :if) e/empty-block])
                       (into (emit-then ctx' then loop-idxs loop-vts ret-vt 1 2))
                       (into [(e/op :else)])
                       (into (emit-operand-at ctx' ret-vt els)) (into (e/br 2))
                       (into [(e/op :end)]))
                   (-> (emit-val ctx' cnd)
                       (into [(e/op :if) e/empty-block])
                       (into (emit-operand-at ctx' ret-vt then)) (into (e/br 2))
                       (into [(e/op :else)])
                       (into (emit-then ctx' els loop-idxs loop-vts ret-vt 1 2))
                       (into [(e/op :end)])))]
    [(into inits (e/block-t ret-vt (into (e/loop* if-bytes) (e/i :unreachable))))
     ret-vt]))

(defn- emit-loop-void
  "(loop [v init ...] (if cond <recur-branch> <exit>)) in STATEMENT position —
   the exit branch is effects/nil, the block carries no result. Model kernels'
   packing/softmax inner loops end in nil; rms/softmax normalize loops end in a
   final effect."
  [ctx node]
  (let [[_ bindvec ifform] node
        pairs (vec (partition 2 bindvec))
        loop-vars (mapv (fn [[sym init]]
                          ;; stamped symbol tag first (TC loop inference via walker)
                          (let [vt (or (sym-vt sym) (infer-vt ctx init))]
                            {:sym sym :idx (alloc! ctx vt) :vt vt :init init}))
                        pairs)
        inits (vec (mapcat (fn [{:keys [idx vt init]}]
                             (into (emit-operand-at ctx vt init) (e/local-set idx))) loop-vars))
        ctx' (reduce (fn [c {:keys [sym idx vt]}] (assoc-in c [:env sym] {:idx idx :vt vt}))
                     ctx loop-vars)
        loop-idxs (mapv :idx loop-vars)
        [_ cnd then els] ifform
        recur-then? (ends-in-recur? then)
        exit (if recur-then? els then)
        rec  (if recur-then? then els)
        exit-bytes (if (nil? exit) [] (emit-effect ctx' exit))
        loop-vts (mapv :vt loop-vars)
        ;; depths from inside the if: if=0, loop=1, void block=2
        if-bytes (if recur-then?
                   (-> (emit-val ctx' cnd)
                       (into [(e/op :if) e/empty-block])
                       (into (emit-then ctx' rec loop-idxs loop-vts nil 1 2))
                       (into [(e/op :else)])
                       (into exit-bytes) (into (e/br 2))
                       (into [(e/op :end)]))
                   (-> (emit-val ctx' cnd)
                       (into [(e/op :if) e/empty-block])
                       (into exit-bytes) (into (e/br 2))
                       (into [(e/op :else)])
                       (into (emit-then ctx' rec loop-idxs loop-vts nil 1 2))
                       (into [(e/op :end)])))]
    (into inits (e/block-v (into (e/loop* if-bytes) (e/i :unreachable))))))

;; ─────────────────────────────────────────────────────────────────────────
;; SIMD128 vectorization of elementwise dotimes-maps (opt-in via ctx :simd?).
;; Targets (dotimes [i n] (aset out i <pure-elementwise>)…) where every store
;; indexes exactly i and the value is composed only of (aget arr i), arithmetic
;; (+,-,*,/), scalar literals and loop-invariant scalars. Emits a v128 main loop
;; (2-wide f64 / 4-wide f32) plus a scalar remainder loop. Cannot run under
;; Chicory (no v128) — validate execution via node/V8.
;; ─────────────────────────────────────────────────────────────────────────

;; canonical arith key → SIMD lane-op suffix (f64x2.<suffix> / f32x4.<suffix>)
(def ^:private simd-suffix {:+ "add" :- "sub" :* "mul" :div "div"})

(defn- simd-op-of
  "If node is a SIMD-vectorizable arithmetic op (registry :infix arith), return
   its lane-op suffix (add/sub/mul/div); else nil."
  [node]
  (let [op (if (= '.invk (first node)) (:raster.op/original (meta node)) (first node))]
    (simd-suffix (ix/canonical op))))

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
        (and (= 2 (count node)) (simd-suffix (ix/canonical h))) false  ; unary +/- unsupported
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

(defn- simd-plan-from-stores
  "Shared SIMD store matcher (dotimes + loop* analyzers): each candidate form must be an
   (aset arr i pure) store indexed exactly by isym, and all referenced arrays must share
   one f64/f32 element type. Returns {:lane-vt :lanes :stores} or nil."
  [ctx isym forms]
  (let [stores (mapv (fn [x]
                       (when (and (seq? x) (= 'clojure.core/aset (first x)) (= 4 (count x))
                                  (simd-index-ok? isym (nth x 2)) (simd-pure? isym (nth x 3)))
                         [(nth x 1) (nth x 2) (nth x 3)]))
                     forms)]
    (when (and (seq forms) (every? some? stores))
      (let [vts (simd-arr-vts ctx stores)]
        (when (and (= 1 (count vts)) (#{:f64 :f32} (first vts)))
          (let [lane-vt (first vts)]
            {:lane-vt lane-vt :lanes (if (= lane-vt :f64) 2 4) :stores stores}))))))

(defn- simd-vectorizable?
  "Analyze a dotimes for SIMD. Returns {:lane-vt :lanes :stores :n :body :isym} or nil.
   Every body form must be an (aset arr i pure) store; all arrays one f64/f32 type."
  [ctx node]
  (let [[_ [isym ncnt] & body] node
        flat (mapcat (fn f [x]
                       (cond (and (seq? x) (= 'do (first x))) (mapcat f (rest x))
                             (and (seq? x) (= 'let* (first x)) (empty? (second x))) (mapcat f (drop 2 x))
                             :else [x]))
                     body)]
    (when-let [plan (simd-plan-from-stores ctx isym flat)]
      (assoc plan :n ncnt :body body :isym isym))))

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

(defn- emit-simd-blocks
  "The two blocks shared by both SIMD emitters: a v128 main loop (step = lanes) over the
   vectorizable stores, then a scalar remainder loop (step 1) whose per-iteration body is
   `srem-bytes`. `idx` is the pre-allocated counter local; `ctx'` has isym bound to it.
     i=0; block{ loop{ (i+lanes)>n → br1; <v128 stores>; i+=lanes; br0 }}
          block{ loop{ i>=n → br1; <srem-bytes>;          i++;       br0 }}"
  [ctx' idx lane-vt lanes stores n srem-bytes]
  (let [vmain (-> (e/local-get idx) (into (e/i32-const lanes)) (into (e/i :i32.add))
                  (into (emit-val ctx' n)) (into (e/i :i32.gt_s)) (into (e/br-if 1))
                  (into (vec (mapcat (fn [[arr idx-node v]]
                                       (-> (addr ctx' arr idx-node)
                                           (into (emit-vexpr ctx' lane-vt v))
                                           (into (e/v128-store 0 0))))
                                     stores)))
                  (into (e/local-get idx)) (into (e/i32-const lanes)) (into (e/i :i32.add))
                  (into (e/local-set idx)) (into (e/br 0)))
        srem (-> (e/local-get idx) (into (emit-val ctx' n)) (into (e/i :i32.ge_s)) (into (e/br-if 1))
                 (into srem-bytes)
                 (into (e/local-get idx)) (into (e/i32-const 1)) (into (e/i :i32.add))
                 (into (e/local-set idx)) (into (e/br 0)))]
    (-> (e/i32-const 0) (into (e/local-set idx))
        (into (e/block (e/loop* vmain)))
        (into (e/block (e/loop* srem))))))

(defn- emit-dotimes-simd
  "Vectorized counted dotimes → void (v128 main + scalar remainder over the whole body)."
  [ctx node {:keys [lane-vt lanes stores n body isym]}]
  (let [idx  (alloc! ctx :i32)
        ctx' (assoc-in ctx [:env isym] {:idx idx :vt :i32})]
    (emit-simd-blocks ctx' idx lane-vt lanes stores n
                      (vec (mapcat #(emit-effect ctx' %) body)))))

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
          (let [n    (nth cnd 2)
                tail (last then)]
            (when (and (seq? tail) (= 'recur (first tail)) (counted-inc? isym (second tail)))
              (when-let [plan (simd-plan-from-stores ctx isym (butlast (rest then)))]
                (assoc plan :n n :isym isym :els els :then-body (vec (rest then)))))))))))

(defn- emit-loop-simd
  "Vectorized counted loop → [bytes ret-vt]. v128 main loop (step = lanes) + scalar
   remainder, then the loop's else value (matches emit-loop-scalar's return)."
  [ctx node {:keys [lane-vt lanes stores n isym els then-body]}]
  (let [idx  (alloc! ctx :i32)
        ctx' (assoc-in ctx [:env isym] {:idx idx :vt :i32})
        ;; scalar remainder runs the loop body's stores (drop the trailing recur)
        blocks (emit-simd-blocks ctx' idx lane-vt lanes stores n
                                 (vec (mapcat #(emit-effect ctx' %) (butlast then-body))))]
    [(into blocks (emit-val ctx' els)) (infer-vt ctx' els)]))

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
    ;; void `if` (no else) in tail position — a macroexpanded `when`/`when-let`.
    (and (seq? node) (= 'if (first node)) (= 3 (count node))) [(emit-effect ctx node) nil]

    ;; void-bound symbol in tail position (e.g. (let* [_ret (dotimes …)] _ret))
    (and (symbol? node) (get-in ctx [:env node :void])) [[] nil]

    (and (seq? node) (#{'let* 'let} (first node)))
    (let [[_ binds & body] node
          [ctx' init-bytes] (emit-bindings ctx binds)
          tail (if (= 1 (count body)) (first body) (cons 'do body))
          [b ret] (emit-result ctx' tail)]
      [(into init-bytes b) ret])

    (and (seq? node) (= 'do (first node)))
    (let [stmts (vec (rest node))
          [b ret] (emit-result ctx (last stmts))]
      [(into (vec (mapcat #(emit-effect ctx %) (butlast stmts))) b) ret])

    (and (seq? node) (#{'loop 'loop*} (first node))) (emit-loop ctx node)

    ;; value-type return: (->Type e0 e1 …) → wasm multi-value return. Each field
    ;; expr is pushed; ret-vt is a VECTOR of valtypes (the value type's fields).
    ;; (soa-lower/lower-value-fn leaves the constructor tail intact for us.)
    (and (seq? node) (symbol? (first node)) (.startsWith (name (first node)) "->"))
    (let [args (rest node)]
      [(vec (mapcat #(emit-val ctx %) args)) (mapv #(infer-vt ctx %) args)])

    :else [(emit-val ctx node) (infer-vt ctx node)]))

(defn- compile-fn
  "Compile one kernel IR to its per-function pieces (no module wrapper).
   {:name :params [{:sym :tag}] :ir :simd?} →
   {:name :param-types :result-types :locals :body}. Result type is inferred
   from the IR (loop else-value / scalar body); nil = void (array-mutating)."
  [{:keys [name params ir simd? fn-index]}]
  ;; G2 guard: 'longs (i64) array params are declared in array-tag->elem but not
  ;; coherently supported — an aget loads i64 while scalar arithmetic runs at i32
  ;; (tag->vt long→i32), and the encoder has no i64 div/rem. Reject loudly rather
  ;; than emit a stack-type mismatch / wrong code. Use 'ints for index/count data.
  (when-let [bad (some (fn [{:keys [sym tag]}] (when (= 'longs tag) sym)) params)]
    (throw (ex-info (str "wasm backend (" name "): 'longs (i64) array param '" bad
                         "' not supported — i64 arrays mix with i32 scalar arithmetic. "
                         "Use 'ints, or extend the backend with a full i64 path first.")
                    {:fn name :param bad})))
  (let [param-vts (mapv #(tag->vt (:tag %)) params)
        env0 (into {} (map-indexed (fn [idx {:keys [sym tag]}] [sym {:idx idx :vt (tag->vt tag)}]) params))
        elems (into {} (keep (fn [{:keys [sym tag]}] (when (array-tag? tag) [sym (array-tag->elem tag)])) params))
        ctx {:env env0 :elems elems :base (count params) :locals (atom []) :simd? simd?
             :fn-index (or fn-index {})}
        [body-bytes ret-vt] (emit-result ctx ir)]
    {:name name :param-types param-vts
     ;; ret-vt: nil = void · keyword = single value · vector = multi-value (value-type return)
     :result-types (cond (nil? ret-vt) [] (vector? ret-vt) ret-vt :else [ret-vt])
     :locals @(:locals ctx) :body body-bytes}))

(defn compile-module
  "Compile several kernels into ONE wasm module that shares a single linear
   memory — so a program's SoA data lives in one buffer addressed by all the
   exported kernels (vs one module/one memory per kernel).
   kernels: [{:name :params :ir :simd?} …].
   Returns {:bytes byte[] :exports [{:name :param-types :result-types} …]}."
  [kernels & {:keys [mem-pages] :or {mem-pages 256}}]
  (let [;; callee dispatch table: each kernel's :call-key (the .invk impl sym other
        ;; kernels reference) → its function index. Kernels with no :call-key (the
        ;; independent-kernel case, e.g. asteroids) simply don't get called.
        fn-index (into {} (keep-indexed (fn [i k] (when-let [ck (:call-key k)] [ck i])) kernels))
        fns   (mapv #(compile-fn (assoc % :fn-index fn-index)) kernels)
        types (mapv #(e/functype (:param-types %) (:result-types %)) fns)
        funcs (vec (map-indexed (fn [i f] (assoc f :type-idx i
                                                 :export? (:export? (nth kernels i) true)))
                                fns))]
    {:bytes (e/build-module
             {:types types :funcs funcs
              :memory {:min mem-pages :max mem-pages :export "memory"}})
     :exports (vec (keep-indexed (fn [i f] (when (:export? (nth kernels i) true)
                                             (select-keys f [:name :param-types :result-types])))
                                 fns))}))

(defn compile-kernel
  "Compile a single kernel IR to a one-function wasm module (its own memory).
   For multiple kernels sharing one memory use compile-module.
   Returns {:bytes byte[] :name :param-types [vt…] :result-types [vt…]}."
  [{:keys [mem-pages] :or {mem-pages 256} :as kernel}]
  (let [m (compile-module [kernel] :mem-pages mem-pages)
        x (first (:exports m))]
    {:bytes (:bytes m) :name (:name x)
     :param-types (:param-types x) :result-types (:result-types x)}))
