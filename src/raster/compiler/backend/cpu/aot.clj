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
            [raster.compiler.backend.cpu.codegen :as cpu]))

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

(def ^:private array-ctors '#{clojure.core/double-array clojure.core/float-array
                              clojure.core/long-array clojure.core/int-array})

(def ^:private alength-impls
  "Devirtualized alength impl symbols (any element type)."
  #{'raster.arrays/alength_m_doubles-impl
    'raster.arrays/alength_m_floats-impl
    'raster.arrays/alength_m_longs-impl
    'raster.arrays/alength_m_ints-impl})

(defn- alength-call?
  "Match (.invk raster.arrays/alength_*-impl arr) or (clojure.core/alength arr)."
  [form]
  (and (seq? form)
       (or (and (= '.invk (first form)) (contains? alength-impls (second form)))
           (= 'clojure.core/alength (first form)))))

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

             :else f))
         form)]
    {:form rewritten :length-syms @length-syms}))

;; ---------------------------------------------------------------------------
;; Host function emission.
;; ---------------------------------------------------------------------------

(def ^:private zeros-like-impls
  #{'raster.arrays/zeros-like_m_doubles_long-impl
    'raster.arrays/zeros-like_m_floats_long-impl})

(defn- buffer-alloc?
  "A binding init that allocates an array buffer: (double-array <size>) or
  (.invk raster.arrays/zeros-like_* arr size). Both yield a zero-filled buffer
  (Java arrays are zero-initialized, so the wrapper needs no extra fill)."
  [init]
  (and (seq? init)
       (or (contains? array-ctors (first init))
           (and (= '.invk (first init)) (contains? zeros-like-impls (second init))))))

(defn- buffer-size
  "The size expression of a buffer-alloc init."
  [init]
  (if (contains? array-ctors (first init))
    (second init)        ; (double-array <size>)
    (last init)))        ; (.invk zeros-like arr <size>)

(defn split-let
  "Split the outermost let* into buffer-alloc bindings and a stripped let* whose
  bindings are only the compute bindings (buffers become free vars = params).
  emit-stmt's let* handler then routes loop-valued bindings correctly (the bare
  loop* path mis-emits the loop's exit value).
  Returns {:buffers [[sym size-expr]...] :stripped <let*-without-allocs>}."
  [form]
  (assert (and (seq? form) (= 'let* (first form))) (str "expected let*, got " (when (seq? form) (first form))))
  (let [[_ bindings & body] form
        pairs (partition 2 bindings)
        buffers (vec (for [[sym init] pairs :when (buffer-alloc? init)]
                       [sym (buffer-size init)]))
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
        (= '.invk op)
        (let [nm (name (second expr))
              as (map #(resolve-int-expr % env) (nnext expr))
              [a b] as]
          (cond (re-find #"_star_" nm)  (* a b)
                (re-find #"_plus_" nm)  (+ a b)
                (re-find #"_minus_" nm) (if (= 2 (count as)) (- a b) (- a))
                (re-find #"_div_" nm)   (quot a b)
                :else a))
        (#{'* 'clojure.core/*} op) (apply * (map #(resolve-int-expr % env) (rest expr)))
        (#{'+ 'clojure.core/+} op) (apply + (map #(resolve-int-expr % env) (rest expr)))
        (#{'- 'clojure.core/-} op) (apply - (map #(resolve-int-expr % env) (rest expr)))
        :else (throw (ex-info (str "resolve-int-expr: unhandled " (pr-str expr)) {:expr expr}))))
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
      (str "for (int " (ce/c-symbol iv) " = " (emit-expr* init array-syms) "; "
           (emit-expr* test array-syms) "; "
           (ce/c-symbol iv) " = " (emit-expr* step array-syms) ") {\n    "
           (clojure.string/join "\n    " (map #(ce/emit-stmt % nil array-syms "idx") do-stmts))
           "\n  }"))

    (and (seq? form) (= 'let* (first form)))
    (let [[_ binds & body] form]
      (str (clojure.string/join
            "\n  "
            (for [[sym init] (partition 2 binds)]
              (if (or (map-loop? init)
                      (and (seq? init) (#{'let* 'loop*} (first init))))
                ;; buffer-writing loop / nested compute — emit for side effects,
                ;; the binding sym (a result buffer) is discarded.
                (emit-host-stmt init array-syms ct)
                ;; scalar binding (e.g. n = (int len_x)) — declare it.
                (str ct-int " " (ce/c-symbol sym) " = " (emit-expr* init array-syms) ";"))))
           "\n  "
           ;; body: emit non-symbol forms (drop the trailing result sym)
           (clojure.string/join
            "\n  "
            (for [b body :when (not (symbol? b))]
              (emit-host-stmt b array-syms ct)))))

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
  (let [ct (ctype dtype "double")
        array-syms (set (concat array-params (map first buffers)))
        ;; param order: input arrays, output buffers, scalar params, array lengths
        param-strs (concat
                    (map #(str "const " ct "* restrict " (ce/c-symbol %)) array-params)
                    (map #(str ct "* restrict " (ce/c-symbol (first %))) buffers)
                    (map (fn [[s t]] (str t " " (ce/c-symbol s))) scalar-params)
                    (map (fn [[_ ls]] (str "int " (ce/c-symbol ls))) (sort-by (comp str val) length-syms)))
        body-c (binding [ce/*emit-config* cpu/cpu-config
                         ce/*scalar-type* ct]
                 (emit-host-stmt stripped array-syms ct))]
    (str "#include <math.h>\n#include <stdbool.h>\n"
         "void " kernel-name "(" (clojure.string/join ", " param-strs) ") {\n  "
         body-c
         "\n}\n")))

;; ---------------------------------------------------------------------------
;; End-to-end: deftm var -> Panama-loaded native fn.
;; ---------------------------------------------------------------------------

(def ^:private array-tags '#{doubles floats longs ints})

(defn- returned-sym
  "The symbol an init expression ultimately evaluates to — follows
  let*/loop*/do/if to the value leaf. Used to map a compute binding to the buffer
  its loop writes & returns (for multi-buffer result tracking)."
  [expr]
  (cond
    (symbol? expr) expr
    (not (seq? expr)) nil
    (contains? #{'let* 'let} (first expr)) (returned-sym (last expr))
    (contains? #{'loop* 'loop} (first expr)) (returned-sym (nth expr 2 nil)) ; loop body
    (= 'do (first expr)) (returned-sym (last expr))
    (= 'if (first expr)) (or (returned-sym (nth expr 3 nil))   ; base/else first
                             (returned-sym (nth expr 2 nil)))
    :else nil))

(defn result-buffer-index
  "Which buffer the function returns: trace the let* body result through the
  compute-binding aliases (sym -> buffer its init returns) to a buffer symbol.
  Returns the index into `buffers`, defaulting to the last buffer."
  [stripped buffers]
  (let [[_ binds & body] stripped
        buf-syms (mapv first buffers)
        buf-set (set buf-syms)
        aliases (into {} (for [[sym init] (partition 2 binds)]
                           [sym (returned-sym init)]))
        resolve (fn resolve [s seen]
                  (cond (buf-set s) s
                        (or (nil? s) (contains? seen s)) nil
                        :else (resolve (get aliases s) (conj seen s))))
        result-sym (resolve (returned-sym (last body)) #{})]
    (or (when result-sym (.indexOf buf-syms result-sym))
        (dec (count buffers)))))

(defn- alloc-array [dtype n]
  (case dtype :double (double-array n) :float (float-array n)
        :long (long-array n) :int (int-array n)))

(defn compile-aot-c
  "Compile a deftm var to a single native C function via the monolithic CPU-C
  backend. Returns a fn of the deftm's args that allocates output buffers, calls
  the native code (zero-copy), and returns the result buffer."
  [f-var dtype]
  (let [{:keys [form params param-env]} (fused-scalar-form f-var dtype)
        {nform :form length-syms :length-syms} (normalize-for-c form)
        {:keys [buffers scalar-bindings stripped]} (split-let nform)
        array-params  (vec (filter #(contains? array-tags (get param-env %)) params))
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
        result-idx (result-buffer-index stripped buffers)
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
                         (let [b (mapv #(alloc-array dtype %) sizes)]
                           (vreset! cache {:sizes sizes :bufs b}) b))
              in-arrs  (map base-env array-params)
              scalars  (map (fn [[p _]] (get base-env p)) scalar-params)
              lengths  (map (fn [[_ ls]] (get env ls)) len-order)]
          (apply native (concat in-arrs buf-arrs scalars lengths))
          (nth buf-arrs result-idx)))
      {:c-source src :kernel-name kernel-name})))
