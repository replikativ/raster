(ns raster.params
  "User-facing API for HMap/HVec-parameterized deftms.

  Lets you write models with structured params:

      (defmodel mlp [w :- (Params (HMap :mandatory {:W1 (Param (Array double))
                                                    :b1 (Param (Array double))
                                                    :W2 (Param (Array double))
                                                    :b2 (Param (Array double))}))
                     x :- (Array double)] :- (Array double)
        (let [h (relu (linear x (:W1 w) (:b1 w)))]
          (linear h (:W2 w) (:b2 w))))

      ;; user-facing call:
      (mlp {:W1 ... :b1 ... :W2 ... :b2 ...} x)

      ;; compiled:
      (def fast (compile-aot #'mlp))
      (fast {:W1 ... ...} x)

  Internally, the macro pre-flattens the body so the HMap arg becomes a list of
  flat positional args (canonical sorted-key order). The walker, TC, bytecode
  emission, and lazy JIT path see no HMap — they see a flat-arg deftm. A
  user-facing wrapper handles flatten-on-call.

  The wrapper kind on each leaf — `(Param T)` vs `(Frozen T)` — is preserved in
  treedef metadata so AD and the optimizer can dispatch at runtime."
  (:refer-clojure :exclude [compile])
  (:require [clojure.string :as str]
            [raster.core]            ; deftm machinery: ensure-walked-body!, jit-walk-with-tc
            [raster.ad.reverse]      ; AD: resolve-deftm-var, grad-expr, value+grad
            [raster.arrays]
            [raster.compiler.core.params :as params]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.params-flatten :as pf]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.optim :as optim]))

;; ----------------------------------------------------------------------
;; Internal: parse a typed param vector
;; ----------------------------------------------------------------------

(defn- parse-typed-params
  "Return [params annotations] from a typed param vector like
  [a :- Long, b :- (HMap ...)]. Untyped params become annotation nil.
  Also accepts the metadata form `^{:- T} a` (see types/param-type-meta)."
  [param-vec]
  (loop [items (seq param-vec)
         params []
         anns   []]
    (if-not items
      [params anns]
      (let [item (first items)
            rest-items (next items)]
        (if (and rest-items (= ':- (first rest-items)))
          (recur (next (next rest-items))
                 (conj params (types/strip-type-meta item))
                 (conj anns (second rest-items)))
          (recur rest-items
                 (conj params (types/strip-type-meta item))
                 (conj anns (types/param-type-meta item))))))))

(defn- parse-defmodel-args
  "Parse defmodel's args: [param-vec :- ret-type body...] OR
   [param-vec body...] (no ret type). Returns {:params :anns :ret :body}."
  [args]
  (let [param-vec (first args)
        rest1 (rest args)
        [ret body] (if (= :- (first rest1))
                     [(second rest1) (drop 2 rest1)]
                     [(types/meta-type-annotation param-vec) rest1])
        [params anns] (parse-typed-params param-vec)]
    {:params params :anns anns :ret ret :body (vec body)}))

;; Defined below; declared here for use by functions above them in file order.
(declare container-class-cache gen-container-class! compile-flatten-wrapper-positional
         value+grad-positional alloc-zeros-like build-container)

;; ----------------------------------------------------------------------
;; Model-metadata registry (avoids the 64KB-method limit)
;; ----------------------------------------------------------------------
;; A defmodel's structured metadata (treedefs, original-args, annotations,
;; containers) can be very large for big models (e.g. a 3-layer/16-dim GSDM has
;; 66 flat leaves). Splicing it as a quoted LITERAL into the expansion makes
;; stock Clojure compile a giant constructor method that overflows the JVM 64KB
;; per-method bytecode limit (it surfaces as "Method code too large!" when
;; compiling the synthesized `(alter-meta! ...)` form). Mirroring deftm's
;; walked-body-registry approach, we store the blob in a runtime atom at
;; macro-expansion time and look it up by a string key — so NO large literal is
;; ever compiled into a method.
(defonce ^:private model-meta-registry (atom {}))

(defn model-meta-for
  "Retrieve stored defmodel metadata by registry key (set at macroexpansion)."
  [key]
  (get @model-meta-registry key))

;; ----------------------------------------------------------------------
;; Adapter: structured args -> flat args
;; ----------------------------------------------------------------------

(defn- compile-flatten-wrapper
  "Build a *fixed-arity* wrapper around inner-fn that splices each tree arg
  into per-leaf positional args via the callee's treedef. Generated via eval
  once per defmodel / compile-aot call — eliminates the per-call rest-args
  seq, transient loop, spec walk, and apply that the older runtime adapter
  performed.

  Validates that:
    - the runtime tree shape matches the spec (HVec lengths, HMap keys);
    - no two leaves of the same tree share JVM identity (naked weight tying)."
  [inner-fn original-args treedefs]
  (compile-flatten-wrapper-positional inner-fn original-args treedefs))

(defn- compile-flatten-wrapper-positional
  [inner-fn original-args treedefs]
  (let [user-syms (mapv (fn [arg] (gensym (str (name arg) "_"))) original-args)
        validations
        (vec (for [[arg sym] (map vector original-args user-syms)
                   :when (get treedefs arg)]
               (let [td (get treedefs arg)]
                 `(do
                    (raster.compiler.core.params-flatten/assert-tree-shape!
                     '~(:spec td) ~sym)
                    (raster.compiler.core.params-flatten/assert-no-identity-collisions!
                     '~(:spec td) ~sym)))))
        flat-args (vec (mapcat (fn [arg sym]
                                 (if-let [td (get treedefs arg)]
                                   (mapv (fn [{:keys [path]}]
                                           (pf/access-expr-for-path sym path))
                                         (:leaves td))
                                   [sym]))
                               original-args
                               user-syms))
        ;; Use direct positional invocation when flat arity ≤ 20
        ;; (Clojure's positional-arg limit), else fall back to apply.
        ;; inner-fn-sym must be the SAME symbol the surrounding (fn [<sym>] …)
        ;; binds — we use an explicit gensym, not auto-gensyms (which differ
        ;; across separate syntax-quote scopes).
        inner-fn-sym (gensym "inner_fn_")
        call-form (if (<= (count flat-args) 20)
                    (cons inner-fn-sym flat-args)
                    (list 'apply inner-fn-sym (vec flat-args)))
        builder (eval (list 'fn [inner-fn-sym]
                            (list 'fn (vec user-syms)
                                  (cons 'do (concat validations [call-form])))))]
    (builder inner-fn)))

;; ----------------------------------------------------------------------
;; Value-class container for a param bundle (used by compile-train-step)
;; ----------------------------------------------------------------------
;; A param tree's leaves are packed into ONE typed `defvalue` (a field per leaf),
;; so the compiled train step takes one container arg per role (w/m/v) instead of
;; N positional leaf params — dodging the JVM 255-arg method limit (≤254 leaves).
;; The container is also a GPU-explodable SoA. The loss itself stays POSITIONAL
;; (it is inlined via value+grad, so its N leaf params become locals, no wall);
;; only the top-level train step uses a container.

(defn- container-class-sym
  "Name the container class by a hash of its leaf signature (syms + types), so
  the same shape dedups, a DIFFERENT shape gets a DIFFERENT class, and
  redefining a model with a changed spec is safe (no stale cached class — value
  classes can't be redefined in place once loaded)."
  [fn-name arg leaves]
  (let [sig (mapv (juxt :sym :type) leaves)]
    (symbol (str (name fn-name) "__" (name arg) "__C"
                 (Integer/toUnsignedString (hash sig) 36)))))

;; ----------------------------------------------------------------------
;; defmodel macro
;; ----------------------------------------------------------------------

(defmacro defmodel
  "Define a deftm-style typed function whose params can include HMap/HVec
  trees. Body uses (:k m) / (get m :k) / (nth v i) for access; the macro
  pre-flattens the body so the underlying compiled function takes flat
  positional args. The user-facing var accepts the structured args and
  flattens at call time.

  See ns docstring for examples."
  {:arglists '([name [params...] :- RetType & body])}
  [fn-name & args]
  (let [{:keys [params anns ret body]} (parse-defmodel-args args)
        prepared (pf/prepare-deftm params anns (if (= 1 (count body))
                                                 (first body)
                                                 (cons 'do body)))
        treedefs    (:treedefs prepared)
        ;; The loss stays POSITIONAL — its leaf params are flattened by
        ;; prepare-deftm. (Containerization happens only at the top-level train
        ;; step, in compile-train-step, where the leaves would otherwise overflow
        ;; the JVM arg limit.)
        flat-params (:params prepared)
        flat-anns   (:annotations prepared)
        flat-body   (:body prepared)
        flat-name   (symbol (str fn-name "--flat"))
        flat-param-vec (vec (mapcat (fn [p ann]
                                      (if ann [p :- ann] [p]))
                                    flat-params flat-anns))
        ;; Stash the (potentially huge) structured metadata in the runtime
        ;; registry at macro-expansion time so the emitted code only references
        ;; it by a small string key — never compiling it as an inline literal,
        ;; which would overflow the JVM 64KB per-method limit for large models.
        reg-key     (str *ns* "/" fn-name)
        _           (swap! model-meta-registry assoc reg-key
                           {:treedefs       treedefs
                            :original-args  params
                            :arg-annotations anns
                            :ret-type       ret})]
    `(do
       ;; Inner flat-arg deftm — what the compiler sees
       (raster.core/deftm ~flat-name ~flat-param-vec
         ~@(when ret [:- ret])
         ~flat-body)

       ;; User-facing var: wraps the flat deftm with a flatten-shim. The big
       ;; structured args are pulled from the registry at runtime (no literals).
       (let [mm# (model-meta-for ~reg-key)]
         (def ~(with-meta fn-name
                 {::flat-var (list 'var flat-name)})
           (#'compile-flatten-wrapper @(var ~flat-name)
                                      (:original-args mm#)
                                      (:treedefs mm#)))
         (alter-meta! (var ~fn-name) assoc
                      ::treedefs       (:treedefs mm#)
                      ::flat-var       (var ~flat-name)
                      ::original-args  (:original-args mm#)
                      ::arg-annotations (:arg-annotations mm#)
                      ::ret-type       (:ret-type mm#)))
       (var ~fn-name))))

;; ----------------------------------------------------------------------
;; compile-aot for a defmodel var
;; ----------------------------------------------------------------------

(defn compile-aot
  "Compile a defmodel var. If the var has tree treedefs, compile the
  underlying flat deftm via pipeline/compile-aot and wrap with a flatten-shim
  matching the original structured args. Otherwise pass through."
  [model-var & opts]
  (let [m (meta model-var)]
    (if-let [flat-var (::flat-var m)]
      (let [compiled-flat (apply pipeline/compile-aot flat-var opts)]
        (compile-flatten-wrapper compiled-flat (::original-args m) (::treedefs m)))
      (apply pipeline/compile-aot model-var opts))))

;; ----------------------------------------------------------------------
;; Trainable-leaf extraction (for AD / optimizer interop)
;; ----------------------------------------------------------------------

(defn- path-key
  "Stable keyword key for a leaf path. (path-key [:layers 0 :Wq]) => :layers.0.Wq.
  Uses '.' as separator: '/' would make a multi-slash keyword that violates the
  keyword spec (≤1 slash) and that rewrite-clj/cljfmt can't parse, while '__'
  could collide with the wrapper sym names."
  [path]
  (keyword (str/join "."
                     (map (fn [p] (if (keyword? p) (name p) (str p)))
                          path))))

(defn leaves-by-kind
  "Return a flat map of {path-key -> leaf-value} for leaves matching `kind`.
  spec is the tree type spec; value is a runtime tree.
  kind ∈ {:param :frozen :plain}."
  [kind spec value]
  (let [descs (pf/flatten-spec spec)
        vals  (pf/flatten-value spec value)]
    (into {}
          (keep (fn [[d v]]
                  (when (= kind (:kind d))
                    [(path-key (:path d)) v])))
          (map vector descs vals))))

(defn trainable-leaves
  "Flat {path-key -> array} of trainable (Param) leaves only."
  [spec value]
  (leaves-by-kind :param spec value))

(defn unflatten-by-kind
  "Inverse of `leaves-by-kind`: given a spec, the original full tree, and a
  flat map of {path-key -> updated-leaf-value} for one kind, return a new
  tree with those leaves replaced. Other-kind leaves pass through."
  [kind spec original-value updated-flat]
  (let [descs (pf/flatten-spec spec)
        vals  (pf/flatten-value spec original-value)
        new-vals (mapv (fn [d v]
                         (if (= kind (:kind d))
                           (get updated-flat (path-key (:path d)) v)
                           v))
                       descs vals)]
    (pf/unflatten-value spec new-vals)))

(defn value+grad
  "Like raster.ad.reverse/value+grad but works on a defmodel var.
  Returns a function that takes the same structured args as the model and
  produces [value grad-arg-1 grad-arg-2 ...] where tree args yield grad
  trees of the same shape as the input tree (one entry per leaf,
  including Frozen leaves whose grads are typically zero arrays).

  For non-tree args, the grad is the raw flat output from value+grad."
  [model-var]
  (let [m            (meta model-var)
        flat-var     (or (::flat-var m) (throw (ex-info "Not a defmodel var" {:var model-var})))
        treedefs     (::treedefs m)
        original-args (::original-args m)]
    (value+grad-positional model-var m flat-var treedefs original-args)))

(defn- value+grad-positional
  [model-var m flat-var treedefs original-args]
  (let [;; Avoid hard-required dependency on ad.reverse at namespace load
        vg           (raster.ad.reverse/value+grad flat-var)]
    (fn [& user-args]
      (let [;; Same flatten as the forward adapter
            flat (loop [args original-args, vals user-args, acc (transient [])]
                   (if-let [a (first args)]
                     (let [v (first vals)]
                       (recur (rest args) (rest vals)
                              (if-let [td (get treedefs a)]
                                (reduce conj! acc (pf/flatten-value (:spec td) v))
                                (conj! acc v))))
                     (persistent! acc)))
            flat-out (apply vg flat)
            value    (first flat-out)
            flat-grads (rest flat-out)]
        ;; Walk arg list in lockstep with flat-grads, restructuring tree grads
        (loop [args original-args
               flat-grads flat-grads
               out  (transient [value])]
          (if-let [a (first args)]
            (if-let [td (get treedefs a)]
              (let [n (count (:leaves td))
                    arg-grads (take n flat-grads)
                    rest-flat (drop n flat-grads)
                    structured (pf/unflatten-value (:spec td) (vec arg-grads))]
                (recur (rest args) rest-flat (conj! out structured)))
              (recur (rest args) (rest flat-grads) (conj! out (first flat-grads))))
            (persistent! out)))))))

(defn strip-leaf-wrappers
  "Walk a tree spec and remove Param/Frozen wrappers from every leaf,
  preserving HMap/HVec structure. Used to derive m/v specs from a weights
  spec — optimizer-state arrays don't carry trainability semantics."
  [spec]
  (cond
    (pf/hmap-spec? spec)
    (list 'HMap :mandatory
          (into {} (map (fn [[k t]] [k (strip-leaf-wrappers t)]))
                (pf/hmap-mandatory spec)))

    (pf/hvec-spec? spec)
    (list 'HVec (mapv strip-leaf-wrappers (pf/hvec-elems spec)))

    :else
    (params/inner-type spec)))

(defn- alloc-zeros-like
  "Return a zeroed array of the same dtype + length as `arr`."
  [arr]
  (let [n (alength arr)]
    (cond
      (instance? (Class/forName "[D") arr) (double-array n)
      (instance? (Class/forName "[F") arr) (float-array n)
      :else (double-array n))))

(defn init-adam-state
  "Allocate parallel m and v trees matching the structure of `weights`.
  At every leaf position (regardless of Param/Frozen), allocate a zero array
  of the same dtype and length. Returns {:m <m-tree> :v <v-tree>}.

  Frozen-leaf entries are wasted (the train step never reads them) but the
  shape parallelism keeps pre-flatten and the canonical leaf order trivial."
  [spec weights]
  (let [descs (pf/flatten-spec spec)
        flat  (pf/flatten-value spec weights)
        m-flat (mapv (fn [_ v] (alloc-zeros-like v)) descs flat)
        v-flat (mapv (fn [_ v] (alloc-zeros-like v)) descs flat)]
    {:m (pf/unflatten-value spec m-flat)
     :v (pf/unflatten-value spec v-flat)}))

;; ----------------------------------------------------------------------
;; compile-train-step
;; ----------------------------------------------------------------------

(defn- leaf->arr-tag
  "Array dispatch tag (doubles/floats/longs/ints) for a leaf's `:type`, or nil."
  [l]
  (let [t (:type l)]
    (when (and (sequential? t) (= 'Array (first t)))
      (case (second t) double 'doubles, float 'floats, long 'longs, int 'ints, nil))))

(defn- gen-train-step-body-container
  "Synthesize the train-step body for the VALUE-CLASS CONTAINER parameter model.
  w/m/v arrive as ONE typed `defvalue` container each (a field per leaf), so the
  compiled method takes 3 args instead of 3N positional leaf params — dodging the
  JVM 255-arg method limit (good up to 254 leaves; beyond that nest containers or
  use a flat buffer). Each leaf is read with a TYPED field access `(.leaf v)`
  (direct unboxed array fetch, no checkcast) — OUTSIDE the differentiated region,
  so no AD is needed for it. value+grad runs on the unchanged POSITIONAL loss
  (no container↔container bridging), then per-Param clip+optimizer steps mutate
  the leaf arrays in place (kept alive by the DCE alias-liveness fix). The same
  container is an SoA the GPU backend explodes into per-field array kernel args."
  [loss-flat-var loss-args weights-arg leaves optimizer]
  (let [data-args (vec (remove #(= % weights-arg) loss-args))
        fld   (fn [cont l] (list (symbol (str "." (:sym l))) cont))
        ;; Unpack ALL leaves from w-cont (typed field access; canonical order =
        ;; loss-flat positional arg order). Bind each to its leaf sym.
        w-syms  (mapv :sym leaves)
        unpacks (vec (mapcat (fn [l] [(:sym l) (fld 'w-cont l)]) leaves))
        vg-call `((raster.ad.reverse/value+grad
                   (var ~(symbol (str (.ns ^clojure.lang.Var loss-flat-var))
                                 (str (.sym ^clojure.lang.Var loss-flat-var)))))
                  ~@w-syms ~@data-args)
        ;; Per-Param leaf: grad d__i (tagged so f32/f64 dispatch is correct),
        ;; clip, optimizer step in place; m/v read from their containers by field.
        param-pairs (keep-indexed (fn [i l] (when (= :param (:kind l)) [i l])) leaves)
        d-sym  (fn [i l] (let [t (leaf->arr-tag l) s (symbol (str "d__" i))]
                           (if t (with-meta s {:tag t}) s)))
        grad-binds (vec (mapcat (fn [[i l]] [(d-sym i l) `(clojure.core/nth ~'__vg-out ~(inc i))])
                                param-pairs))
        clip-effs  (map (fn [[i l]] `(raster.dl.optim/clip-grad-norm!
                                      ~(d-sym i l) (raster.arrays/alength ~(:sym l)) ~'max-grad-norm))
                        param-pairs)
        opt-effs   (case optimizer
                     :adam (map (fn [[i l]] `(raster.dl.optim/adam-step!
                                              ~(:sym l) ~(d-sym i l) ~(fld 'm-cont l) ~(fld 'v-cont l)
                                              (raster.arrays/alength ~(:sym l))
                                              ~'lr ~'beta1 ~'beta2 ~'eps ~'adam-t))
                                param-pairs)
                     :sgd  (map (fn [[i l]] `(raster.dl.optim/sgd-step!
                                              ~(:sym l) ~(d-sym i l) (raster.arrays/alength ~(:sym l)) ~'lr))
                                param-pairs))]
    `(~'let [~@unpacks
             ~'__vg-out ~vg-call
             ~'loss (clojure.core/nth ~'__vg-out 0)
             ~@grad-binds]
            ~@clip-effs
            ~@opt-effs
            ~'loss)))

(defn- ensure-train-step-deps! []
  ;; The synthesized train-step body uses fully-qualified raster.dl.optim and
  ;; raster.arrays calls; the eval'd defmodel needs these namespaces loaded.
  (require 'raster.dl.optim 'raster.arrays 'raster.ad.reverse))

(defn compile-train-step
  "Build a fused training-step compiled function from a defmodel loss var.

  Options:
    :optimizer  — :adam (default) or :sgd

  Adam returns:
    {:train-fn   IFn (weights, m, v, ...data..., max-grad-norm, lr, beta1, beta2, eps, adam-t)
     :init-state (fn [weights] -> {:m <m-tree> :v <v-tree>})
     :var        underlying defmodel var
     :spec       weights tree spec}

  SGD returns:
    {:train-fn   IFn (weights, ...data..., max-grad-norm, lr)
     :init-state (fn [_] nil)        ;; SGD has no per-Param state
     :var        underlying defmodel var
     :spec       weights tree spec}

  The synthesized body is evaluated as a defmodel inside raster.params, so the
  pre-flatten pass converts all (:k w) / (:k m) / (:k v) accesses to flat args
  and compile-aot fuses value+grad with the per-leaf optimizer updates."
  ([loss-var] (compile-train-step loss-var {}))
  ([loss-var {:keys [optimizer target-device dtype] :or {optimizer :adam}}]
   (ensure-train-step-deps!)
   (when-not (#{:adam :sgd} optimizer)
     (throw (ex-info "Unknown optimizer (expected :adam or :sgd)"
                     {:optimizer optimizer})))
   (let [m-meta        (meta loss-var)
         treedefs      (or (::treedefs m-meta)
                           (throw (ex-info "Not a defmodel var" {:var loss-var})))
         original-args (::original-args m-meta)
         arg-anns      (::arg-annotations m-meta)
         flat-var      (::flat-var m-meta)
         ;; Assume the FIRST tree arg is the weights — common case.
         [w-arg w-td]  (first treedefs)
         spec          (:spec w-td)
         leaves        (:leaves w-td)
         m-spec        (strip-leaf-wrappers spec)
         ;; Build the train-step's arg vector: [w (m v)? ...non-tree-args... +hyperparams]
         non-tree      (vec (keep (fn [[a ann]] (when (not (pf/tree-arg? ann)) [a ann]))
                                  (map vector original-args arg-anns)))
         ;; Optimizer in the name disambiguates overloads when the same loss
         ;; var is compiled under both :adam and :sgd (compile-aot otherwise
         ;; sees two same-name dispatch entries with different arities).
         train-name    (symbol (str (.sym loss-var) "--train-step-" (name optimizer)))
         ;; CONTAINER model: generate a typed `defvalue` (one field per leaf) and
         ;; take w/m/v as ONE container arg each — 3 args, not 3N positional leaf
         ;; params — so no JVM 255-arg wall (good up to 254 leaves; beyond that
         ;; nest containers or use a flat buffer). Field access is typed; value+grad
         ;; runs on the positional loss; the user-facing train-fn builds the
         ;; containers from the param TREES (call contract unchanged). The same
         ;; container is a GPU-explodable SoA (per-field array kernel args).
         target-ns     (.ns ^clojure.lang.Var loss-var)
         cont-sym      (container-class-sym train-name w-arg leaves)
         _             (gen-container-class! cont-sym leaves target-ns)
         body          (gen-train-step-body-container flat-var original-args w-arg leaves optimizer)
         param-vec     (vec (concat
                             ['w-cont :- cont-sym]
                             (when (= optimizer :adam)
                               ['m-cont :- cont-sym
                                'v-cont :- cont-sym])
                             (mapcat (fn [[a ann]] [a :- ann]) non-tree)
                             ['max-grad-norm :- 'Double
                              'lr            :- 'Double]
                             (when (= optimizer :adam)
                               ['beta1  :- 'Double
                                'beta2  :- 'Double
                                'eps    :- 'Double
                                'adam-t :- 'Long])))
         form          `(raster.core/deftm ~train-name ~param-vec :- ~'Double
                          ~body)
         train-var     (binding [*ns* target-ns]
                         (eval form)
                         (find-var (symbol (str (.name target-ns)) (str train-name))))
         ;; Thread :target-device / :dtype so the fused forward+backward+optimizer
         ;; step can be lowered to a GPU program (D2). The container is a
         ;; GPU-explodable SoA (per-field array kernel args); at :dtype :float the
         ;; (All [T]) kernels monomorphize to float for the device.
         compiled      (apply pipeline/compile-aot train-var
                              (concat (when target-device [:target-device target-device])
                                      (when dtype [:dtype dtype])))
         cinfo         (@container-class-cache cont-sym)
         mk            (fn [s tree] (build-container cinfo s tree))
         train-fn      (case optimizer
                         :adam (fn [w m v & rest-args]
                                 (apply compiled (mk spec w) (mk m-spec m) (mk m-spec v) rest-args))
                         :sgd  (fn [w & rest-args]
                                 (apply compiled (mk spec w) rest-args)))]
     {:train-fn   train-fn
      :init-state (case optimizer
                    :adam (fn [weights] (init-adam-state spec weights))
                    :sgd  (fn [_] nil))
      :var        train-var
      :spec       spec
      :optimizer  optimizer})))

;; ----------------------------------------------------------------------
;; Value-class container for tree params (replaces positional-arg splicing)
;; ----------------------------------------------------------------------
;; A tree arg's leaves are packed into ONE typed value-class container instead
;; of N positional flat params — removing the JVM/Clojure ~200-arg ceiling.
;; Each leaf becomes a TYPED field (double[]/float[]/long[]...), so access via
;; (.-<leaf-sym> container) returns the original array type (no Object boxing,
;; no checkcast). On Valhalla this is a real value class (defvalue); otherwise
;; defvalue falls back to a defrecord. The container holds the leaf-array
;; REFERENCES, so it is built once and reused across training steps (the
;; optimizer mutates the arrays in place).

(defonce ^:private container-class-cache (atom {}))

(defn gen-container-class!
  "Generate (once, cached by class-sym) a value-class container whose typed
  fields are the canonical leaves of a tree arg. `leaves` are the treedef leaf
  descriptors ({:sym :type ...}); `source-ns` is where the class + factory are
  interned. Returns {:class-sym S :factory IFn :field-syms [...] :leaves ...}."
  [class-sym leaves source-ns]
  (or (@container-class-cache class-sym)
      (let [field-vec (vec (mapcat (fn [{:keys [sym type]}] [sym :- type]) leaves))
            form (list 'raster.core/defvalue class-sym field-vec)]
        (binding [*ns* source-ns] (eval form))
        (let [factory @(ns-resolve source-ns (symbol (str "->" class-sym)))
              info {:class-sym class-sym
                    :factory factory
                    :field-syms (mapv :sym leaves)
                    :leaves (vec leaves)}]
          (swap! container-class-cache assoc class-sym info)
          info))))

(defn build-container
  "Build a container instance from a runtime tree value, by flattening the
  value to its canonical leaves and calling the generated factory. Leaf order
  matches the value class' field order (both are the treedef canonical order)."
  [{:keys [factory]} spec value]
  (apply factory (pf/flatten-value spec value)))

(defn model-treedef
  "Look up the treedef for a defmodel var's first tree arg.
  If multiple tree args, returns the first; for arg-name access use
  (-> (meta v) ::treedefs (get arg-name))."
  [model-var]
  (let [tds (-> model-var meta ::treedefs)]
    (when (seq tds)
      (val (first tds)))))
