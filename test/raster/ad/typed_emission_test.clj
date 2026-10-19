(ns raster.ad.typed-emission-test
  "Phase 1 (Π) of .internal/ad_typed_emission_plan.md: emit-time
  tangent-typed template emission.

  (i)   tangent-tag (Π) — identity on differentiable tags, nil on ⊥/unknown
  (ii)  a grads-fn emission from tagged primal args closes Π-typed output
        ports by projection; raw expressions use independent value evidence
        for tagging and emission-time devirtualization
  (iii) an emission with UNtagged primal args is byte-identical to the
        pre-change emission — tags are carried, never guessed."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ad.tangent :as tangent]
            [raster.ad.templates :as tmpl]
            [raster.compiler.ad.bind-ctx :as bind-ctx]
            ;; external :grads-fn registration sites (Phase 1d — tagged mints)
            [raster.dl.nn]
            [raster.dl.loss]
            [raster.dl.array-ops]
            [raster.dl.attention]
            [raster.nn]))

(defn- make-test-ctx
  "Deterministic single-arity gensym (the pre-change BindCtx contract) —
  make-ctx must canonicalize it tag-capable."
  []
  (let [counter (atom 0)]
    (bind-ctx/make-ctx (fn [prefix] (symbol (str prefix "__" (swap! counter inc)))))))

(defn- tag-of [x] (some-> x meta :raster.type/tag))

(defn- tagged [sym tag] (with-meta sym {:raster.type/tag tag}))

(defn- compile-scalar-ports [template tags incoming]
  (let [parameters (mapv #(tagged (symbol (str "parameter" %1)) %2) (range) tags)
        [ctx ports] (tmpl/instantiate-template-ctx template parameters nil incoming (make-test-ctx))]
    {:bindings (:bindings ctx) :ports ports
     :invoke (eval (list 'fn '[dy choose]
                         (list 'let* (:bindings ctx) ports)))}))

(deftest scalar-template-ports-materialize-their-declared-tangent-dtypes
  (let [{:keys [invoke ports]} (compile-scalar-ports
                              (tmpl/get-template 'clojure.core/double)
                              '[float] (tagged 'dy 'double))
        input (+ 1.0 (Math/scalb 1.0 (int -24)))
        output (first (invoke input false))]
    (is (= '[float] (mapv tag-of ports)))
    (is (= Float (class output)))
    (is (= (Float/floatToRawIntBits (float input)) (Float/floatToRawIntBits output)))
    (is (= [nil] (invoke nil false)))
    (is (thrown? IllegalArgumentException (invoke Double/MAX_VALUE false))))
  (let [template {:grads-fn (fn [ctx _ _ dy gs]
                             (let [g (gs "raw" 'float)]
                               [(update ctx :bindings into [g dy]) [g]]))}
        {:keys [invoke bindings ports]} (compile-scalar-ports template '[float] (tagged 'dy 'double))]
    (is (= 'double (tag-of (first bindings))) "raw source declaration is not the desired port dtype")
    (is (= 'float (tag-of (first ports))))
    (is (= Float (class (first (invoke 2.0 false)))))))

(deftest shared-template-sources-are-projected-per-port-not-retagged
  (doseq [local? [false true]]
   (let [incoming (tagged 'dy 'double)
        template {:grads-fn (fn [ctx _ _ dy gs]
                             (if local?
                               (let [g (gs "shared" 'float)]
                                 [(update ctx :bindings into [g dy]) [g g g]])
                               [ctx [dy dy dy]]))}
        {:keys [invoke ports]} (compile-scalar-ports template '[float double float] incoming)
        values (invoke 2.0 false)]
    (is (= '[float double float] (mapv tag-of ports)))
    (is (= [Float Double Float] (mapv class values)))
    (is (= [nil nil nil] (invoke nil false)))
    (is (= 'double (tag-of incoming)))
    (is (not= (first ports) (last ports)))))
  (let [template {:grads-fn (fn [ctx _ _ dy _] [ctx [nil dy]])}
        {:keys [invoke]} (compile-scalar-ports template '[double float] (tagged 'dy 'double))]
    (is (= [nil (float 2.0)] (invoke 2.0 false)))
    (is (= Float (class (second (invoke 2.0 false)))))))

(deftest conditional-dispatch-unification-is-not-a-value-conversion-proof
  (let [template {:grads-fn (fn [ctx _ _ dy gs]
                             (let [g (gs "conditional" 'double)]
                               [(update ctx :bindings into [g (list 'if 'choose dy 0.0)]) [g]]))}
        {:keys [invoke ports]} (compile-scalar-ports template '[double] 'dy)]
    (is (= '[double] (mapv tag-of ports)))
    (is (= [Double] (mapv class (invoke (float 2.0) true))))
    (is (= [2.0] (invoke (float 2.0) true)))
    (is (= [0.0] (invoke (float 2.0) false)))
    (is (= [nil] (invoke nil true)))))

(deftest contradictory-shared-array-output-contracts-refuse
  (doseq [tags ['[float floats] '[floats doubles]]]
    (is (= :ad-template-output-interface-conflict
           (try (compile-scalar-ports
                 {:grads-fn (fn [ctx _ _ dy _] [ctx [dy dy]])} tags 'dy)
                nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest intermediate-expression-facts-reach-subsequent-call-selection
  (let [[template _] (tmpl/resolve-template 'raster.dl.loss/mse-loss)
        [ctx _] (tmpl/instantiate-template-ctx
                  template [(tagged 'pred 'floats) (tagged 'target 'floats)
                            (tagged 'n 'long)]
                  nil (tagged 'dy 'double) (make-test-ctx))
        pairs (vec (partition 2 (:bindings ctx)))
        [n-binding scale-binding gradient-binding] pairs
        call (second gradient-binding)]
    (is (= '.invk (first call)))
    (is (= 'raster.dl.loss/mse-grad (:raster.op/original (meta call))))
    (is (= 'floats (tag-of (first gradient-binding))))
    ;; Carrying facts in the resolver environment must not assert that a
    ;; nullable intermediate is an unboxed primitive local.
    (is (nil? (:tag (meta (first n-binding)))))
    (is (nil? (:tag (meta (first scale-binding)))))
    (is (= 'long (first (second n-binding))))))

(deftest unknown-intermediates-mask-earlier-selection-facts
  (let [pred (tagged 'pred 'floats)
        target (tagged 'target 'floats)
        gradient (tagged 'gradient 'floats)
        call (list 'raster.dl.loss/mse-grad pred target 'scale 'count)
        resolve-bindings
        (fn [prefix]
          (first (#'tmpl/close-template-output-ports
                   (into prefix [gradient call]) [pred target] [gradient nil]
                   (:gensym-fn (make-test-ctx)))))]
    (is (= '.invk (first (last (resolve-bindings
                                ['count '(clojure.core/long n)
                                 'scale '(clojure.core/double dy)])))))
    (doseq [prefix [['count '(clojure.core/long n) 'scale 'unknown]
                    ['count '(clojure.core/long n) 'scale '(clojure.core/double dy)
                     'count 'unknown]
                    ['count '(clojure.core/long n) 'scale '(clojure.core/double dy)
                     'scale 'unknown]
                    ['count '(clojure.core/long n) 'scale '(clojure.core/double dy)
                     'scale '(opaque-result dy)]]]
      (is (= 'raster.dl.loss/mse-grad
             (first (last (resolve-bindings prefix))))))))

(defn- nil-safe-projection? [target source expression]
  (let [[head bindings body] expression
        [input init] bindings]
    (and (= 'let* head) (= 2 (count bindings)) (= source init)
         (= (list 'if (list 'clojure.core/nil? input) nil
                  (list (symbol "clojure.core" (name target)) input)) body))))

(deftest materialized-scalar-projection-does-not-infer-a-missing-expression-dtype
  (let [unknown '(opaque-result x)
        dynamic (tangent/project-expr 'float unknown)
        concrete (tangent/project-expr 'float unknown {:materialized? true})]
    (is (nil-safe-projection? 'float unknown dynamic))
    (is (= '(clojure.core/float (opaque-result x)) concrete))
    (is (= 'float (tag-of concrete)))
    (is (= unknown (tangent/project-expr nil unknown {:materialized? true})))))

(deftest scalar-projection-follows-carried-types-through-ssa-aliases
  (let [source (tagged 'source 'double)
        bindings {'outer 'inner 'inner source}
        projected (tangent/project-expr 'float 'outer {:binding-map bindings})]
    (is (nil-safe-projection? 'float 'outer projected))
    (is (= 'float (tag-of projected)))
    (is (= 'outer (tangent/project-expr 'double 'outer {:binding-map bindings}))))
  (is (= 'outer (tangent/project-expr
                'float 'outer {:binding-map {'outer 'inner 'inner (tagged 'source 'float)}})))
  (is (nil-safe-projection? 'float 'outer
                           (tangent/project-expr 'float 'outer
                                                 {:binding-map {'outer 'inner 'inner '(double x)}})))
  (let [source (tagged 'source 'float)]
    (is (= source (tangent/project-expr
                  'float source {:binding-map {'source '(double x)}}))
        "carried source metadata remains authoritative over alias recovery"))
  (doseq [bindings [{'outer 'inner 'inner nil}
                   {'outer 'inner 'inner '(nth grads 0)}
                   {'outer 'inner 'inner '(if condition x nil)}
                   {'outer 'inner 'inner 'outer}
                   {'outer 'unbound}
                   {'outer '(opaque-result x)}]]
    (is (nil-safe-projection? 'float 'outer
                             (tangent/project-expr 'float 'outer {:binding-map bindings})))))

(deftest nullable-scalar-tags-do-not-authorize-unboxing
  (doseq [[source-tag target cast] [['float 'double double] ['double 'float float]]]
    (let [source (tagged 'x source-tag)
          expression (tangent/project-expr target source)
          projection (eval (list 'fn* '[x] expression))]
      (is (nil-safe-projection? target source expression))
      (is (nil? (projection nil)))
      (doseq [value [(float -0.0) 1.0000000596046448 -2.75]]
        (is (= (class (cast value)) (class (projection value))))
        (is (= (Double/doubleToRawLongBits (double (cast value)))
               (Double/doubleToRawLongBits (double (projection value)))))))))

(deftest nullable-projection-preserves-evaluation-and-original-traps
  (let [source '(do (swap! calls inc) x)
        expression (tangent/project-expr 'float source)
        projection (eval (list 'fn* '[x calls] expression))
        calls (atom 0)]
    (is (nil? (projection nil calls)))
    (is (= (float 2.0) (projection 2.0 calls)))
    (is (= 2 @calls)))
  (let [source '(double (do (swap! calls inc) x))
        projection (eval (list 'fn* '[x calls] (tangent/project-expr 'float source)))
        calls (atom 0)]
    (is (thrown? NullPointerException (projection nil calls)))
    (is (= 1 @calls))))

(deftest lexical-cast-shaped-call-does-not-prove-presence-or-dtype
  (doseq [head ['float 'double]
          target ['float 'double]]
    (let [source (with-meta (list head 'x) {:raster.op/lexical-callee true})
          projected (tangent/project-expr target source)]
      (is (nil-safe-projection? target source projected))
      (let [projection (eval (list 'fn* [head 'x] projected))
            calls (atom 0)]
        (is (nil? (projection (fn [_] (swap! calls inc) nil) 1.0)))
        (is (= 1 @calls))
        (let [value (projection (fn [_] (swap! calls inc) 7.25) 1.0)
              expected (if (= target 'float) (float 7.25) (double 7.25))]
          (is (= (class expected) (class value)))
          (is (= expected value))
          (is (= 2 @calls)))))))

(deftest derived-scalar-adjoints-project-before-claiming-their-tangent-tag
  (let [[template _] (tmpl/resolve-template 'raster.par/scale)
        coefficient (tagged 'coefficient 'float)
        input (tagged 'input 'doubles)
        cotangent (tagged 'cotangent 'doubles)
        [ctx gradients] (tmpl/instantiate-template-ctx
                         template [coefficient input] nil cotangent (make-test-ctx))
        scalar-expression (get (into {} (map vec (partition 2 (:bindings ctx))))
                               (first gradients))]
    (is (= '[float doubles] (mapv tag-of gradients)))
    (is (= 'clojure.core/float (first scalar-expression))
        "a desired Float binder tag must be backed by an actual Double-to-Float projection")
    (is (= 'float (tag-of scalar-expression)))
    (is (= 'double (tag-of (second scalar-expression)))
        "the resolved dot-product result remains authoritative before projection")))

;; ================================================================
;; (i) Π — tangent-tag
;; ================================================================

(deftest tangent-tag-test
  (testing "identity on differentiable tags (mirrors differentiable?)"
    (is (= 'double  (tangent/tangent-tag 'double)))
    (is (= 'float   (tangent/tangent-tag 'float)))
    (is (= 'doubles (tangent/tangent-tag 'doubles)))
    (is (= 'floats  (tangent/tangent-tag 'floats))))
  (testing "⊥ (no tangent space) and unknown → nil, never a default"
    (is (nil? (tangent/tangent-tag 'long)))
    (is (nil? (tangent/tangent-tag 'longs)))
    (is (nil? (tangent/tangent-tag 'int)))
    (is (nil? (tangent/tangent-tag 'boolean)))
    (is (nil? (tangent/tangent-tag 'Object)))
    (is (nil? (tangent/tangent-tag nil))))
  (testing "classification distinguishes proven primitive ⊥ from missing evidence"
    (doseq [t '[long int boolean char longs ints booleans]]
      (is (= {:kind :none} (tangent/tangent-kind t)) (str t)))
    (doseq [t ['Object 'objects 'java.lang.Number nil]]
      (is (= {:kind :unknown} (tangent/tangent-kind t)) (str t))))
  (testing "Π agrees with differentiable? on its whole domain"
    (doseq [t '[double float doubles floats long longs int ints boolean Object nil]]
      (is (= (tangent/differentiable? t) (some? (tangent/tangent-tag t)))
          (str "Π and differentiable? disagree on " t)))))

;; ================================================================
;; Tag-capable bind-ctx gensym
;; ================================================================

(deftest tag-capable-gensym-test
  (testing "make-ctx canonicalizes a single-arity gensym-fn to two arities"
    (let [{:keys [gensym-fn]} (make-test-ctx)]
      (is (= 'g__1 (gensym-fn "g")))
      (is (nil? (tag-of (gensym-fn "g"))))
      (let [s (gensym-fn "g" 'floats)]
        (is (= 'floats (tag-of s))))
      (testing "nil tag = plain mint (Π's ⊥ needs no branching at call sites)"
        (is (nil? (tag-of (gensym-fn "g" nil))))))))

;; ================================================================
;; (ii) tagged primal → Π-tagged binding + resolved/tagged call
;; ================================================================

(deftest tagged-emission-test
  (testing "float scalar primals: Π-tagged LHS, devirtualized float RHS (O3)"
    (let [[template _] (tmpl/resolve-template '*)
          a  (tagged 'a0 'float)
          b  (tagged 'b0 'float)
          dy (tagged 'dy0 'float)
          [ctx grad-syms] (tmpl/instantiate-template-ctx template [a b] nil dy (make-test-ctx))
          pairs (partition 2 (:bindings ctx))]
      (is (= '[float float] (mapv tag-of grad-syms))
          "gradient binding LHS carries Π(primal arg tag)")
      (is (every? (fn [[s _]] (= 'float (tag-of s))) pairs))
      (doseq [[_ rhs] pairs]
        (is (= '.invk (first rhs))
            "fully-tagged backward call devirtualizes at emission")
        (is (.contains (name (second rhs)) "_m_float_float")
            "resolved to the FLOAT specialization, not a widened double"))))

  (testing "static :grads alias bindings get Π-tagged LHS"
    ;; '+ grads are [dy dy] — plain symbol RHS, so only the LHS tag applies.
    (let [[template _] (tmpl/resolve-template '+)
          x  (tagged 'x0 'doubles)
          y  (tagged 'y0 'doubles)
          dy (tagged 'dy0 'doubles)
          [_ grad-syms] (tmpl/instantiate-template-ctx template [x y] nil dy (make-test-ctx))]
      (is (= '[doubles doubles] (mapv tag-of grad-syms)))))

  (testing "conditional scalar ports are projected rather than labeled as converted"
    (let [[template _] (tmpl/resolve-template 'Math/min)
          x  (tagged 'x0 'double)
          y  (tagged 'y0 'double)
          dy (tagged 'dy0 'double)
          [ctx grad-syms] (tmpl/instantiate-template-ctx template [x y] nil dy (make-test-ctx))
          pairs (partition 2 (:bindings ctx))
          bindings (into {} (map vec pairs))
          invoke (eval (list 'fn '[x0 y0 dy0]
                             (list 'let* (:bindings ctx) grad-syms)))]
      (is (= '[double double] (mapv tag-of grad-syms)))
      (is (every? #(= 'if (first (second %))) (take 2 pairs)))
      (doseq [port grad-syms]
        (is (= 'let* (first (get bindings port))))
        (is (= 'double (tag-of (get bindings port)))))
      (is (= [1.5 0.0] (invoke 2.0 3.0 1.5)))
      (is (= [0.0 2.5] (invoke 4.0 3.0 2.5)))))

  (testing "mixed known/⊥ primal args: only differentiable slots are tagged"
    ;; aget: (arr idx) — d_arr gets Π(floats)=floats, idx slot grad is nil.
    (let [[template _] (tmpl/resolve-template 'clojure.core/aget)
          arr (tagged 'arr0 'floats)
          [_ grad-syms] (tmpl/instantiate-template-ctx template [arr 'i0] nil (tagged 'dy0 'float) (make-test-ctx))]
      (is (= 'floats (tag-of (first grad-syms))))
      (is (nil? (second grad-syms))))))

;; ================================================================
;; (iii) untagged primal → byte-identical emission (no guessing)
;; ================================================================

(deftest untagged-emission-unchanged-test
  (testing "scalar * with untagged args: exact pre-change bindings, no metadata"
    (let [[template _] (tmpl/resolve-template '*)
          [ctx grad-syms] (tmpl/instantiate-template-ctx template '[a0 b0] nil 'dy0 (make-test-ctx))]
      (is (= '[dg_x__1 (raster.numeric/* dy0 b0)
               dg_y__2 (raster.numeric/* dy0 a0)]
             (:bindings ctx))
          "untagged emission is byte-identical (symbolic, undevirtualized)")
      (is (every? #(nil? (tag-of %)) grad-syms))
      (is (every? #(nil? (tag-of %)) (:bindings ctx))
          "no :raster.type/tag invented anywhere — LHS or RHS")))

  (testing "min with untagged args: exact pre-change if-form"
    (let [[template _] (tmpl/resolve-template 'Math/min)
          [ctx _] (tmpl/instantiate-template-ctx template '[x0 y0] nil 'dy0 (make-test-ctx))]
      (is (= '[dg_x__1 (if (clojure.core/<= (double x0) (double y0)) dy0 0.0)
               dg_y__2 (if (clojure.core/<= (double x0) (double y0)) 0.0 dy0)]
             (:bindings ctx)))
      (is (every? #(nil? (tag-of %)) (:bindings ctx)))))

  (testing "partially tagged args do NOT resolve the call (all-or-nothing)"
    ;; dy tagged, b0 untagged → (raster.numeric/* dy0 b0) must stay symbolic.
    (let [[template _] (tmpl/resolve-template '*)
          dy (tagged 'dy0 'double)
          [ctx grad-syms] (tmpl/instantiate-template-ctx template ['a0 'b0] nil dy (make-test-ctx))]
      (is (= '[dg_x__1 (raster.numeric/* dy0 b0)
               dg_y__2 (raster.numeric/* dy0 a0)]
             (:bindings ctx))
          "a call with ANY unknown arg tag stays symbolic")
      (is (every? #(nil? (tag-of %)) grad-syms)
          "Π of an absent primal tag is nil — LHS stays untagged"))))

;; ================================================================
;; Phase 1d — external :grads-fn registration sites mint tagged syms
;; (dl/nn, dl/array-ops, dl/attention, raster.nn) at the mint site,
;; independent of output-port closure in close-template-output-ports.
;; ================================================================

(defn- instantiate
  "Instantiate `op`'s template with the given actuals; returns
  [binding-pairs grad-syms]."
  [op actuals result-sym adjoint]
  (let [[template _] (tmpl/resolve-template op)
        [ctx grad-syms] (tmpl/instantiate-template-ctx
                         template actuals result-sym adjoint (make-test-ctx))]
    [(partition 2 (:bindings ctx)) grad-syms]))

(deftest external-grads-fn-tagged-mints-test
  (testing "dl/nn rms-norm: float primals ⇒ float-tagged dx/dw at mint"
    (let [[pairs grad-syms]
          (instantiate 'raster.dl.nn/rms-norm
                       [(tagged 'x0 'floats) (tagged 'w0 'floats)
                        'rows0 'feat0 'eps0 'go0]
                       nil (tagged 'dy0 'floats))]
      (is (= '[floats floats nil nil nil nil] (mapv tag-of grad-syms)))
      (is (every? (fn [[s rhs]] (and (= 'floats (tag-of s))
                                     (= 'floats (tag-of rhs))))
                  pairs)
          "both grad bindings carry Π(primal) on LHS and RHS form")))

  (testing "dl/array-ops broadcast-add: dh/dt tagged, integer intermediate ba_n stays ⊥"
    (let [[pairs grad-syms]
          (instantiate 'raster.dl.array-ops/broadcast-add
                       [(tagged 'h0 'floats) (tagged 't0 'floats) 'batch0 'dim0]
                       nil (tagged 'dy0 'floats))
          by-prefix (fn [p] (first (filter #(.startsWith (name (first %)) p) pairs)))]
      (is (= '[floats floats nil nil] (mapv tag-of grad-syms)))
      (is (nil? (tag-of (first (by-prefix "ba_n"))))
          "Π is ⊥ on the integer element-count intermediate — never guessed")
      (is (= 'floats (tag-of (first (by-prefix "dh")))))
      (is (= 'floats (tag-of (first (by-prefix "dt")))))))

  (testing "dl/attention batched-causal-sdpa: dQ/dK/dV tagged from Q/K/V"
    (let [[pairs grad-syms]
          (instantiate 'raster.dl.attention/batched-causal-sdpa
                       [(tagged 'Q0 'floats) (tagged 'K0 'floats) (tagged 'V0 'floats)
                        'b0 'sl0 'hd0]
                       nil (tagged 'dy0 'floats))]
      (is (= '[floats floats floats nil nil nil] (mapv tag-of grad-syms)))
      (is (every? (fn [[s _]] (= 'floats (tag-of s))) pairs))))

  (testing "raster.nn softmax-cross-entropy: the PRIMAL intermediate sce_s is tagged"
    (let [[pairs grad-syms]
          (instantiate 'raster.nn/softmax-cross-entropy
                       [(tagged 'logits0 'floats) (tagged 't0 'floats)]
                       'r0 (tagged 'dy0 'float))
          [[s-sym _] [dl-sym _]] pairs]
      (is (.startsWith (name s-sym) "sce_s"))
      (is (= 'floats (tag-of s-sym))
          "intermediate softmax extraction carries the primal logits tag")
      (is (= 'floats (tag-of dl-sym)))
      (is (= '[floats nil] (mapv tag-of grad-syms))))))

(deftest external-grads-fn-untagged-unchanged-test
  (testing "untagged primals ⇒ emission byte-identical, no tags anywhere"
    (let [[pairs grad-syms]
          (instantiate 'raster.dl.nn/rms-norm
                       '[x0 w0 rows0 feat0 eps0 go0] nil 'dy0)]
      (is (= '[[dx__1 (raster.dl.nn/rms-norm-backward-dx dy0 x0 w0 rows0 feat0 eps0 go0)]
               [dw__2 (raster.dl.nn/rms-norm-backward-dweight dy0 x0 rows0 feat0 eps0)]]
             (mapv vec pairs)))
      (is (every? (fn [[s rhs]] (and (nil? (tag-of s)) (nil? (tag-of rhs)))) pairs)
          "no :raster.type/tag invented on LHS or RHS")
      (is (every? #(nil? (tag-of %)) (remove nil? grad-syms))))))
