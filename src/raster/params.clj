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
            [raster.ad.reverse]      ; required so synthesized train-step bodies resolve
            [raster.arrays]
            [raster.compiler.core.params :as params]
            [raster.compiler.core.params-flatten :as pf]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.optim]))

;; ----------------------------------------------------------------------
;; Internal: parse a typed param vector
;; ----------------------------------------------------------------------

(defn- parse-typed-params
  "Return [params annotations] from a typed param vector like
  [a :- Long, b :- (HMap ...)]. Untyped params become annotation nil."
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
                 (conj params item)
                 (conj anns (second rest-items)))
          (recur rest-items
                 (conj params item)
                 (conj anns nil)))))))

(defn- parse-defmodel-args
  "Parse defmodel's args: [param-vec :- ret-type body...] OR
   [param-vec body...] (no ret type). Returns {:params :anns :ret :body}."
  [args]
  (let [param-vec (first args)
        rest1 (rest args)
        [ret body] (if (= :- (first rest1))
                     [(second rest1) (drop 2 rest1)]
                     [nil rest1])
        [params anns] (parse-typed-params param-vec)]
    {:params params :anns anns :ret ret :body (vec body)}))

;; ----------------------------------------------------------------------
;; Adapter: structured args -> flat args
;; ----------------------------------------------------------------------

(defn- compile-flatten-wrapper
  "Build a *fixed-arity* wrapper around inner-fn that splices each tree arg
  into per-leaf positional args via the callee's treedef. Generated via eval
  once per defmodel / compile-aot call — eliminates the per-call rest-args
  seq, transient loop, spec walk, and apply that the older runtime adapter
  performed.

  Validates that no two leaves of the same tree share JVM identity (naked
  weight tying) — throws a clear error if they do."
  [inner-fn original-args treedefs]
  (let [user-syms (mapv (fn [arg] (gensym (str (name arg) "_"))) original-args)
        validations
        (vec (for [[arg sym] (map vector original-args user-syms)
                   :when (get treedefs arg)]
               (let [td (get treedefs arg)]
                 `(raster.compiler.core.params-flatten/assert-no-identity-collisions!
                    '~(:spec td) ~sym))))
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
        flat-params (:params prepared)
        flat-anns   (:annotations prepared)
        flat-body   (:body prepared)
        treedefs    (:treedefs prepared)
        flat-name   (symbol (str fn-name "--flat"))
        flat-param-vec (vec (mapcat (fn [p ann]
                                      (if ann [p :- ann] [p]))
                                    flat-params flat-anns))]
    `(do
       ;; Inner flat-arg deftm — what the compiler sees
       (raster.core/deftm ~flat-name ~flat-param-vec
         ~@(when ret [:- ret])
         ~flat-body)

       ;; User-facing var: wraps the flat deftm with a flatten-shim
       (def ~(with-meta fn-name
               {::treedefs       (list 'quote treedefs)
                ::flat-var       (list 'var flat-name)
                ::original-args  (list 'quote params)})
         (#'compile-flatten-wrapper @(var ~flat-name)
                                 '~params
                                 '~treedefs))
       (alter-meta! (var ~fn-name) assoc
                    ::treedefs '~treedefs
                    ::flat-var (var ~flat-name)
                    ::original-args '~params
                    ::arg-annotations '~anns
                    ::ret-type '~ret)
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
  "Stable keyword key for a leaf path. (path-key [:layers 0 :Wq]) => :layers/0/Wq.
  Uses '/' as separator since '__' could collide with the wrapper sym names."
  [path]
  (keyword (str/join "/"
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
        original-args (::original-args m)
        ;; Avoid hard-required dependency on ad.reverse at namespace load
        vg           ((requiring-resolve 'raster.ad.reverse/value+grad) flat-var)]
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

(defn- access-expr
  "Build the expression `(:k1 (:k2 ... (:kN root)))` corresponding to path
  [k1 k2 ... kN] under the symbol root. For integer steps, uses (nth ... i)."
  [root path]
  (reduce (fn [acc step]
            (cond
              (keyword? step) (list step acc)
              (integer? step) (list 'clojure.core/nth acc step)
              :else (throw (ex-info "Bad path step" {:step step}))))
          root
          path))

(defn- gen-train-step-body
  "Synthesize the train-step body S-expression. The body computes the loss
  via (value+grad #'loss-flat-var), clips each Param leaf's gradient, then
  applies the per-Param optimizer step in-place on (:path w).

  optimizer ∈ {:adam :sgd}:
    :adam — uses raster.dl.optim/adam-step! and references m, v, lr, beta1,
            beta2, eps, adam-t in the surrounding scope.
    :sgd  — uses raster.dl.optim/sgd-step! and references only lr."
  [loss-flat-var loss-args weights-arg leaves _loss-treedef optimizer]
  (let [;; loss-args is the loss var's original arg list, e.g. '[w values ...]
        ;; weights-arg is the symbol of the tree-typed arg, usually 'w
        ;; The flat call to value+grad expects: tree leaves first (canonical
        ;; order), then non-tree args in declared order.
        flat-call-args (concat
                         (map (fn [{:keys [path]}] (access-expr weights-arg path)) leaves)
                         (filter (fn [a] (not= a weights-arg)) loss-args))
        vg-call `((raster.ad.reverse/value+grad (var ~(symbol (str (.ns loss-flat-var))
                                                              (str (.sym loss-flat-var)))))
                  ~@flat-call-args)
        ;; Per-Param destructure: d_<idx> = (nth vg-out (inc idx)).
        ;; Stamp each d__i with the proper array tag (doubles/floats) from
        ;; the leaf type so the walker can resolve clip-grad-norm! and
        ;; sgd-step!/adam-step! to the correct float vs double specialization
        ;; instead of defaulting to the doubles overload (which would inline
        ;; a Double-init reduce! into the train-step and crash with [F→[D
        ;; in the SegRed SIMD codegen for f32 paths).
        param-leaf-pairs (keep-indexed (fn [i l] (when (= :param (:kind l)) [i l]))
                                       leaves)
        leaf->arr-tag (fn [l]
                        (let [t (:type l)]
                          (cond
                            (and (sequential? t) (= 'Array (first t)) (= 'float (second t))) 'floats
                            (and (sequential? t) (= 'Array (first t)) (= 'double (second t))) 'doubles
                            (and (sequential? t) (= 'Array (first t)) (= 'long (second t))) 'longs
                            (and (sequential? t) (= 'Array (first t)) (= 'int (second t))) 'ints
                            :else nil)))
        d-syms (into {} (map (fn [[i l]]
                               (let [tag (leaf->arr-tag l)
                                     base (symbol (str "d__" i))]
                                 [(:path l)
                                  (if tag (with-meta base {:tag tag}) base)]))
                             param-leaf-pairs))
        ;; Bind every effect to a non-underscore name (effect-clip-i, effect-step-i)
        ;; so compile-aot's DCE doesn't eliminate them. Side effects surface as
        ;; `effect_NNNNN` bindings in the compiled form.
        grad-bindings
        (vec (mapcat (fn [[i l]]
                       [(d-syms (:path l)) `(clojure.core/nth ~'__vg-out ~(inc i))])
                     param-leaf-pairs))
        clip-bindings
        (vec (mapcat (fn [[i l]]
                       (let [d (d-syms (:path l))
                             w-acc (access-expr weights-arg (:path l))]
                         [(symbol (str "effect-clip-" i))
                          `(raster.dl.optim/clip-grad-norm!
                             ~d (raster.arrays/alength ~w-acc) ~'max-grad-norm)]))
                     param-leaf-pairs))
        opt-bindings
        (case optimizer
          :adam
          (vec (mapcat (fn [[i l]]
                         (let [d (d-syms (:path l))
                               w-acc (access-expr weights-arg (:path l))
                               m-acc (access-expr 'm (:path l))
                               v-acc (access-expr 'v (:path l))]
                           [(symbol (str "effect-step-" i))
                            `(raster.dl.optim/adam-step!
                               ~w-acc ~d ~m-acc ~v-acc
                               (raster.arrays/alength ~w-acc)
                               ~'lr ~'beta1 ~'beta2 ~'eps ~'adam-t)]))
                       param-leaf-pairs))
          :sgd
          (vec (mapcat (fn [[i l]]
                         (let [d (d-syms (:path l))
                               w-acc (access-expr weights-arg (:path l))]
                           [(symbol (str "effect-step-" i))
                            `(raster.dl.optim/sgd-step!
                               ~w-acc ~d (raster.arrays/alength ~w-acc) ~'lr)]))
                       param-leaf-pairs))
          (throw (ex-info "Unknown optimizer (expected :adam or :sgd)"
                          {:optimizer optimizer})))]
    `(~'let [~'__vg-out ~vg-call
             ~'loss (clojure.core/nth ~'__vg-out 0)
             ~@grad-bindings]
        ~@(map second (partition 2 clip-bindings))
        ~@(map second (partition 2 opt-bindings))
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
  ([loss-var {:keys [optimizer] :or {optimizer :adam}}]
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
         body          (gen-train-step-body flat-var original-args w-arg leaves w-td optimizer)
         ;; HMap/HVec arg types are recognized directly by prepare-deftm — no
         ;; Params wrapper needed. The (:k w) accesses pre-flatten into flat
         ;; positional symbols at compile time.
         param-vec     (vec (concat
                              [w-arg :- spec]
                              (when (= optimizer :adam)
                                ['m :- m-spec
                                 'v :- m-spec])
                              (mapcat (fn [[a ann]] [a :- ann]) non-tree)
                              ['max-grad-norm :- 'Double
                               'lr            :- 'Double]
                              (when (= optimizer :adam)
                                ['beta1  :- 'Double
                                 'beta2  :- 'Double
                                 'eps    :- 'Double
                                 'adam-t :- 'Long])))
         form          `(raster.params/defmodel ~train-name ~param-vec :- ~'Double
                          ~body)
         target-ns     (.ns ^clojure.lang.Var loss-var)
         train-var     (binding [*ns* target-ns]
                         (eval form)
                         (find-var (symbol (str (.name target-ns)) (str train-name))))
         train-fn      (compile-aot train-var)]
     {:train-fn   train-fn
      :init-state (case optimizer
                    :adam (fn [weights] (init-adam-state spec weights))
                    :sgd  (fn [_] nil))
      :var        train-var
      :spec       spec
      :optimizer  optimizer})))

(defn model-treedef
  "Look up the treedef for a defmodel var's first tree arg.
  If multiple tree args, returns the first; for arg-name access use
  (-> (meta v) ::treedefs (get arg-name))."
  [model-var]
  (let [tds (-> model-var meta ::treedefs)]
    (when (seq tds)
      (val (first tds)))))
