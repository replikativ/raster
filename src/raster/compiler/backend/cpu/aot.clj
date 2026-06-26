(ns raster.compiler.backend.cpu.aot
  "Monolithic CPU-C AOT backend for compile-aot.

  Runs compile-aot's forward passes to obtain the fused SCALAR IR (one inlined
  body, no SIMD lowering, buffers materialized as `double-array` bindings), then
  emits the WHOLE body as a single C host function — control flow and all — via
  the shared expression/statement emitter (raster.compiler.backend.gpu.c-emit)
  under the native-C config. Compiles at -O3 -march=native and loads via Panama
  FFM (zero-copy on-heap arrays). One native call per compiled function: no
  per-op JVM dispatch, no per-kernel orchestration.

  This is the path to fast quantized inference: the layer is composed from
  standard deftms (which fuse here), and the int8-MAC is the one op-descriptor
  whose C lowering emits the maddubs intrinsic (added separately).

  Scope today: single-output element-wise / reduction deftms over double/float
  arrays (silu, rms-norm, residual, ...). Multi-buffer layers + the int8-MAC
  lowering build on this."
  (:require [clojure.walk :as walk]
            [raster.compiler.pipeline :as pl]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.backend.cpu.codegen :as cpu]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.ir.form :as form]
            [raster.compiler.passes.scalar.peephole :as peephole]))

;; ---------------------------------------------------------------------------
;; Fused-IR access — reuse compile-aot's forward pipeline at the scalar backend.
;; ---------------------------------------------------------------------------

(defn fused-scalar-form
  "Run compile-aot's forward passes with the SCALAR backend (no SIMD lowering)
  to get the inlined, buffer-fused body as plain loops/lets — the dialect the C
  emitter consumes."
  [f-var dtype]
  (let [gwb   @#'pl/get-walked-body
        gp    @#'pl/get-params
        bpe   @#'pl/build-param-env
        cp    @#'pl/clean-params
        wb    (gwb f-var dtype)
        params (gp f-var dtype)
        raw   (if (= 1 (count wb)) (first wb) (list* 'do wb))
        source-ns (or (when (var? f-var) (.ns ^clojure.lang.Var f-var)) *ns*)
        opts  {:inline? true :simd? false :dtype dtype
               :active-params (cp params)
               :param-env (bpe f-var dtype)
               :source-ns source-ns}]
    {:form (pl/run-passes raw pl/forward-passes opts)
     :params (cp params)
     :param-env (bpe f-var dtype)}))

;; ---------------------------------------------------------------------------
;; Form normalization for C emission.
;; ---------------------------------------------------------------------------

;; element-keyword -> C type, and the maps from a deftm tag / array ctor to it.
(def ^:private elem->ctype
  {:double "double" :float "float" :long "long" :int "int" :byte "int8_t"})
(def ^:private tag->elem
  '{doubles :double double :double floats :float float :float longs :long long :long
    ints :int int :int bytes :byte byte :byte})
(def ^:private ctor->elem
  '{clojure.core/double-array :double clojure.core/float-array :float
    clojure.core/long-array :long clojure.core/int-array :int clojure.core/byte-array :byte})

(def ^:private array-ctors '#{clojure.core/double-array clojure.core/float-array
                              clojure.core/byte-array
                              clojure.core/long-array clojure.core/int-array})

(defn- semantic-op
  "The walker-stamped semantic op of a call form (:raster.op/original on a
  devirtualized .invk). Falls back to the impl symbol for .invk, else the head.
  Lets passes match ops by MEANING instead of pattern-matching mangled impl names."
  [form]
  (when (seq? form)
    (or (:raster.op/original (meta form))
        (if (= '.invk (first form)) (second form) (first form)))))

(defn- alength-call?
  "Match an alength call (raster.arrays/alength devirtualized, or clojure.core/alength)
  by its semantic op."
  [form]
  (contains? #{'raster.arrays/alength 'clojure.core/alength} (semantic-op form)))

(defn- alength-arg [form]
  (if (= '.invk (first form)) (nth form 2) (second form)))

(defn- desugar-dotimes
  "(dotimes [iv bound] body...) -> (loop* [iv (int 0)] (if (< iv bound)
     (do body... (recur (+ iv 1))) iv)). Base is the induction var (harmless;
   the loop is side-effecting). Lets the shared emit-stmt / our map-loop matcher
   treat it as a counted loop."
  [form]
  (let [[_ binds & body] form
        iv (first binds) bound (second binds)]
    (list 'loop* [iv (list 'int 0)]
          (list 'if (list 'clojure.core/< iv bound)
                (concat (list 'do) body (list (list 'recur (list 'clojure.core/+ iv 1))))
                iv))))

(defn normalize-for-c
  "Rewrite IR idioms the bare emitter doesn't yet lower to C:
   - (dotimes [i bound] ...)  -> counted loop* (see desugar-dotimes)
   - (alength arr)            -> a per-array length symbol  len_<arr>
   - (unchecked-inc-int x)    -> (clojure.core/+ x 1)
   Returns {:form rewritten :length-syms {arr len-sym}}."
  [form]
  (let [length-syms (atom {})
        rewritten
        (walk/postwalk
         (fn [f]
           (cond
             (alength-call? f)
             (let [arr (alength-arg f)
                   ls (or (get @length-syms arr)
                          (let [s (symbol (str "len_" (name arr)))]
                            (swap! length-syms assoc arr s) s))]
               ls)

             (and (seq? f) (#{'clojure.core/unchecked-inc-int 'clojure.core/unchecked-inc
                              'unchecked-inc-int 'unchecked-inc} (first f)))
             (list 'clojure.core/+ (second f) 1)

             (and (seq? f) (= 'dotimes (first f)))
             (desugar-dotimes f)

             ;; devirtualized explicit array ops -> clojure.core forms (matched by
             ;; SEMANTIC op, not impl name) so c-emit's aget/aset handlers fire and
             ;; the element type follows the array's :raster.type/tag (e.g. int8 out).
             (and (seq? f) (= '.invk (first f)) (= 'raster.arrays/aget (semantic-op f)))
             (apply list 'clojure.core/aget (nnext f))

             (and (seq? f) (= '.invk (first f)) (= 'raster.arrays/aset (semantic-op f)))
             (apply list 'clojure.core/aset (nnext f))

             :else f))
         form)]
    {:form rewritten :length-syms @length-syms}))

;; ---------------------------------------------------------------------------
;; Host function emission.
;; ---------------------------------------------------------------------------

(defn- buffer-alloc?
  "A binding init that allocates an array buffer: a core array-ctor (double-array …)
  or raster.arrays/zeros-like (matched by semantic op, not impl name). Both yield a
  zero-filled buffer (Java arrays are zero-initialized — no extra fill needed)."
  [init]
  (and (seq? init)
       (let [op (semantic-op init)]
         (or (contains? array-ctors op) (= 'raster.arrays/zeros-like op)))))

(defn- buffer-size
  "The size expression of a buffer-alloc init."
  [init]
  (if (contains? array-ctors (semantic-op init))
    (second init)        ; (double-array <size>)
    (last init)))        ; (.invk zeros-like arr <size>)

(defn- buffer-elem-type
  "Element keyword of a buffer-alloc init: from the array ctor (double-array → :double,
  byte-array → :byte) or, for zeros-like, the walker-stamped :raster.type/tag."
  [init]
  (or (ctor->elem (semantic-op init))
      (tag->elem (:raster.type/tag (meta init)))
      :double))

(defn split-let
  "Split the outermost let* into buffer-alloc bindings and a stripped let* whose
  bindings are only the compute bindings (buffers become free vars = params).
  emit-stmt's let* handler then routes loop-valued bindings correctly (the bare
  loop* path mis-emits the loop's exit value).
  Returns {:buffers [[sym size-expr elem-type]...] :stripped <let*-without-allocs>}."
  [form]
  (assert (and (seq? form) (= 'let* (first form))) (str "expected let*, got " (when (seq? form) (first form))))
  (let [[_ bindings & body] form
        pairs (partition 2 bindings)
        buffers (vec (for [[sym init] pairs :when (buffer-alloc? init)]
                       [sym (buffer-size init) (buffer-elem-type init)]))
        ;; scalar bindings (simple int/arith, not buffers or loops) — needed to
        ;; resolve buffer sizes that reference them (e.g. arg_n = (* n 1)).
        scalar-bindings (vec (for [[sym init] pairs
                                   :when (not (buffer-alloc? init))
                                   :when (not (and (seq? init)
                                                   (#{'loop* 'loop 'dotimes 'let* 'let} (first init))))]
                               [sym init]))
        kept (vec (mapcat (fn [[s init]] (if (buffer-alloc? init) [] [s init])) pairs))]
    {:buffers buffers
     :scalar-bindings scalar-bindings
     :stripped (cons 'let* (cons kept body))}))

(defn resolve-int-expr
  "Evaluate an integer IR expression (buffer size, offset) to a number given an
  env of {sym value}. Handles devirtualized .invk long/int arith, (int x), and
  clojure core +/-/* — enough for buffer-size and length expressions."
  [expr env]
  (cond
    (number? expr) expr
    (symbol? expr) (get env expr)
    (seq? expr)
    (let [op (first expr)]
      (cond
        (= 'int op)  (long (resolve-int-expr (second expr) env))
        (= 'long op) (long (resolve-int-expr (second expr) env))
        ;; arithmetic — classify by the SEMANTIC op (:raster.op/original metadata,
        ;; falling back to the head/impl symbol) via the shared intrinsics registry.
        :else
        (let [args (if (= '.invk op) (nnext expr) (rest expr))
              k (intrinsics/canonical (semantic-op expr))
              as (map #(resolve-int-expr % env) args)]
          (case k
            :+   (apply + as)
            :-   (if (= 1 (count as)) (- (first as)) (apply - as))
            :*   (apply * as)
            :div (apply quot as)
            (throw (ex-info (str "resolve-int-expr: unhandled op " (pr-str expr)) {:expr expr}))))))
    :else (throw (ex-info (str "resolve-int-expr: unhandled " (pr-str expr)) {:expr expr}))))

(def ^:private ctype {:double "double" :float "float" :long "long" :int "int"})

;; scalar induction/length vars are ints
(def ^:private ct-int "int")

(defn- map-loop?
  "Canonical counted map loop: single induction var, body is an `if` whose then
  branch ends in `recur` and whose else (the loop's exit value) is a buffer/sym
  we discard. (loop* [iv init] (if test (do stmts... (recur step)) base))"
  [form]
  (and (seq? form) (= 'loop* (first form))
       (= 2 (count (second form)))               ; single induction var
       (let [body (nth form 2 nil)]
         (and (seq? body) (= 'if (first body))
              (let [then (nth body 2 nil)]
                (and (seq? then) (= 'do (first then))
                     (let [lst (last then)]
                       (and (seq? lst) (= 'recur (first lst))))))))))

(defn- emit-expr* [e array-syms]
  (ce/emit-expr e nil array-syms "idx"))

(defn- emit-host-stmt
  "Emit a fused scalar form as host-C statements. Handles the outer let* (scalar
  decls + buffer-writing loops) and the canonical counted map loop as a clean
  `for`; delegates everything else (reductions, nested let*, aset, if) to the
  shared emit-stmt. `ct` is the array element C type for scalar decls."
  [form array-syms ct]
  (cond
    (map-loop? form)
    (let [[_ binds ifbody] form
          iv (first binds) init (second binds)
          test (nth ifbody 1)
          then (nth ifbody 2)
          do-stmts (butlast (rest then))
          step (second (last then))]
      ;; iv is the int counter declared by this for-loop — visible to the body so
      ;; index math derived from it (and the test/step) types as int.
      (binding [ce/*int-vars* (conj ce/*int-vars* iv)]
        (str "for (int " (ce/c-symbol iv) " = " (emit-expr* init array-syms) "; "
             (emit-expr* test array-syms) "; "
             (ce/c-symbol iv) " = " (emit-expr* step array-syms) ") {\n    "
             (clojure.string/join "\n    " (map #(ce/emit-stmt % nil array-syms "idx") do-stmts))
             "\n  }")))

    (and (seq? form) (= 'let* (first form)))
    (let [[_ binds & body] form
          pairs (partition 2 binds)
          loop-init? (fn [init] (or (map-loop? init)
                                    (and (seq? init) (#{'let* 'loop*} (first init)))))
          ;; outer scalar bindings are declared `int` (ct-int) — seed them so loops
          ;; that index off them type correctly.
          scalar-syms (for [[sym init] pairs :when (not (loop-init? init))] sym)]
      (binding [ce/*int-vars* (into ce/*int-vars* scalar-syms)]
        (str (clojure.string/join
              "\n  "
              (for [[sym init] pairs]
                (if (loop-init? init)
                ;; buffer-writing loop / nested compute — emit for side effects,
                ;; the binding sym (a result buffer) is discarded.
                  (emit-host-stmt init array-syms ct)
                ;; scalar binding (e.g. n = (int len_x)) — declare it.
                  (str ct-int " " (ce/c-symbol sym) " = " (emit-expr* init array-syms) ";"))))
             "\n  "
           ;; body: emit statement forms only. The function is void; the trailing
           ;; result value (a buffer symbol, or a vector [q scales] of them for a
           ;; multi-output kernel) is data for the host wrapper, NOT a C statement.
             (clojure.string/join
              "\n  "
              (for [b body :when (not (or (symbol? b) (vector? b)))]
                (emit-host-stmt b array-syms ct))))))

    :else (ce/emit-stmt form nil array-syms "idx")))

(defn emit-c-fn
  "Emit a C host function for a fused, normalized scalar form.
   - kernel-name  : C function name
   - dtype        : :double/:float (array element type)
   - array-params : input array param symbols (from the deftm signature)
   - scalar-params: [[sym ctype]...] scalar params (e.g. n : int)
   - buffers      : [[sym size-expr]...] output/intermediate buffers (caller-allocated)
   - length-syms  : {arr len-sym} integer length params for (alength arr)
   - stripped     : the let* body (buffer-allocs already removed) to emit
   Returns the C source string."
  [kernel-name dtype array-params scalar-params buffers length-syms stripped]
  (let [ct (ctype dtype "double")  ; default scalar type for the body (the dominant dtype)
        array-syms (set (concat (map first array-params) (map first buffers)))
        ;; param order: input arrays, output buffers, scalar params, array lengths —
        ;; each array gets ITS OWN element type (e.g. float in, int8_t out for quant).
        param-strs (concat
                    (map (fn [[s c]] (str "const " c "* restrict " (ce/c-symbol s))) array-params)
                    (map (fn [[s _ elem]] (str (elem->ctype elem "double") "* restrict " (ce/c-symbol s))) buffers)
                    (map (fn [[s t]] (str t " " (ce/c-symbol s))) scalar-params)
                    (map (fn [[_ ls]] (str "int " (ce/c-symbol ls))) (sort-by (comp str val) length-syms)))
        ;; scalar params and array-length params are all int — seed them so index
        ;; arithmetic bound to a name (base = b*n) types as int, not *scalar-type*.
        int-seed (set (concat (map first scalar-params) (vals length-syms)))
        body-c (binding [ce/*emit-config* cpu/cpu-config
                         ce/*scalar-type* ct
                         ce/*int-vars* int-seed]
                 (emit-host-stmt stripped array-syms ct))]
    (str "#include <math.h>\n#include <stdbool.h>\n#include <stdint.h>\n"
         "void " kernel-name "(" (clojure.string/join ", " param-strs) ") {\n  "
         body-c
         "\n}\n")))

;; ---------------------------------------------------------------------------
;; End-to-end: deftm var -> Panama-loaded native fn.
;; ---------------------------------------------------------------------------

(def ^:private array-tags '#{doubles floats longs ints bytes})

(defn result-buffers
  "Which buffer(s) the function returns, and in what shape. Inspects the stripped
  let* body tail: a bare symbol returns ONE buffer (the common case); a vector
  literal [a b ...] returns SEVERAL (the fused multi-output kernel, e.g. [q scales]
  from norm+blockquant). Each returned position is traced through the compute-binding
  aliases (sym -> the symbol its init returns) to a buffer symbol.
  Returns {:shape :scalar|:vector :indices [i ...]} (indices into `buffers`,
  defaulting to the last buffer when a position can't be resolved)."
  [stripped buffers]
  (let [[_ binds & body] stripped
        buf-syms (mapv first buffers)
        buf-set (set buf-syms)
        aliases (into {} (for [[sym init] (partition 2 binds)]
                           [sym (form/tail-symbol init)]))
        resolve-buf (fn resolve-buf [s seen]
                      (cond (buf-set s) s
                            (or (nil? s) (contains? seen s)) nil
                            :else (recur (get aliases s) (conj seen s))))
        idx-of (fn [x]
                 (let [s (if (symbol? x) x (form/tail-symbol x))
                       b (resolve-buf s #{})]
                   (or (when b (.indexOf buf-syms b)) (dec (count buffers)))))
        tail (last body)]
    (if (vector? tail)
      {:shape :vector :indices (mapv idx-of tail)}
      {:shape :scalar :indices [(idx-of tail)]})))

(defn- alloc-array [elem n]
  (case elem :double (double-array n) :float (float-array n)
        :long (long-array n) :int (int-array n) :byte (byte-array n)))

(defn compile-aot-c
  "Compile a deftm var to a single native C function via the monolithic CPU-C
  backend. Returns a fn of the deftm's args that allocates output buffers, calls
  the native code (zero-copy), and returns the result buffer."
  [f-var dtype]
  (let [{:keys [form params param-env]} (fused-scalar-form f-var dtype)
        {nform :form length-syms :length-syms} (normalize-for-c form)
        {:keys [buffers scalar-bindings stripped]} (split-let nform)
        ;; canonical copy-propagation: resolve aliases read downstream (e.g. a binding
        ;; r = (let* [..writes buf..] buf) from inlining residual-add, where r feeds a
        ;; later norm's reduction). Shared with the AD peephole — no bespoke aliasing.
        stripped (peephole/copy-propagate-let stripped)
        ;; input array params as [sym ctype] (per-element type), scalars as [sym ctype]
        array-params  (vec (for [p params :when (contains? array-tags (get param-env p))]
                             [p (elem->ctype (tag->elem (get param-env p)) "double")]))
        scalar-params (vec (for [p params :when (not (contains? array-tags (get param-env p)))]
                             [p (ctype (get param-env p) "int")]))
        ;; stable length-sym order shared by signature + call site
        len-order (vec (sort-by (comp str val) length-syms))
        kernel-name (str "aot_" (ce/c-symbol (:name (meta f-var))) "_" (name dtype))
        src (emit-c-fn kernel-name dtype array-params scalar-params buffers length-syms stripped)
        so  (cpu/compile-source! src)
        native (cpu/load-kernel so kernel-name
                                (+ (count array-params) (count buffers))
                                (+ (count scalar-params) (count len-order)))
        {result-shape :shape result-indices :indices} (result-buffers stripped buffers)
        ;; Output buffers are reused across calls (the hoisted-buffer model the JVM
        ;; backend uses) — keyed by the resolved size signature, so repeated
        ;; same-shape calls (inference) don't re-allocate. Single-consumer: the
        ;; returned buffer is overwritten on the next call.
        cache (volatile! nil)]
    (with-meta
      (fn [& args]
        (let [base-env (zipmap params args)
              len-env (reduce (fn [m [arr ls]] (assoc m ls (count (get base-env arr))))
                              base-env len-order)
              env (reduce (fn [m [sym init]] (assoc m sym (resolve-int-expr init m)))
                          len-env scalar-bindings)
              sizes (mapv (fn [[_ size-expr]] (long (resolve-int-expr size-expr env))) buffers)
              c @cache
              buf-arrs (if (and c (= (:sizes c) sizes))
                         (:bufs c)
                         (let [b (mapv (fn [[_ _ elem] size] (alloc-array elem size)) buffers sizes)]
                           (vreset! cache {:sizes sizes :bufs b}) b))
              in-arrs  (map (comp base-env first) array-params)
              scalars  (map (fn [[p _]] (get base-env p)) scalar-params)
              lengths  (map (fn [[_ ls]] (get env ls)) len-order)]
          (apply native (concat in-arrs buf-arrs scalars lengths))
          (case result-shape
            :vector (mapv #(nth buf-arrs %) result-indices)
            (nth buf-arrs (first result-indices)))))
      {:c-source src :kernel-name kernel-name})))
