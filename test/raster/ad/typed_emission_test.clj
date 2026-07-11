(ns raster.ad.typed-emission-test
  "Phase 1 (Π) of .internal/ad_typed_emission_plan.md: emit-time
  tangent-typed template emission.

  (i)   tangent-tag (Π) — identity on differentiable tags, nil on ⊥/unknown
  (ii)  a grads-fn emission from tagged primal args produces Π-tagged
        binding LHS syms, tagged call forms, and emission-time
        devirtualization when all arg tags are known
  (iii) an emission with UNtagged primal args is byte-identical to the
        pre-change emission — tags are carried, never guessed."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ad.tangent :as tangent]
            [raster.ad.templates :as tmpl]
            [raster.compiler.ad.bind-ctx :as bind-ctx]))

(defn- make-test-ctx
  "Deterministic single-arity gensym (the pre-change BindCtx contract) —
  make-ctx must canonicalize it tag-capable."
  []
  (let [counter (atom 0)]
    (bind-ctx/make-ctx (fn [prefix] (symbol (str prefix "__" (swap! counter inc)))))))

(defn- tag-of [x] (some-> x meta :raster.type/tag))

(defn- tagged [sym tag] (with-meta sym {:raster.type/tag tag}))

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

  (testing "unresolvable-but-tagged emission stamps the call form itself"
    ;; min/max emit (if (<= ...) dy 0.0) — no deftm call to resolve, but the
    ;; grad binding carries Π and the census/downstream read the LHS tag.
    (let [[template _] (tmpl/resolve-template 'Math/min)
          x  (tagged 'x0 'double)
          y  (tagged 'y0 'double)
          dy (tagged 'dy0 'double)
          [ctx grad-syms] (tmpl/instantiate-template-ctx template [x y] nil dy (make-test-ctx))
          pairs (partition 2 (:bindings ctx))]
      (is (= '[double double] (mapv tag-of grad-syms)))
      (doseq [[s rhs] pairs]
        (is (= 'if (first rhs)))
        (is (= (tag-of s) (tag-of rhs))
            "emitted (non-deftm) call form carries the binding's tag"))))

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
