(ns raster.ad.templates-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ad.templates :as tmpl]
            [raster.compiler.ad.bind-ctx :as bind-ctx]))

(defn- make-test-ctx
  "Create a test bind-ctx with a deterministic gensym."
  []
  (let [counter (atom 0)
        gs (fn [prefix] (symbol (str prefix "__" (swap! counter inc))))]
    (bind-ctx/make-ctx gs)))

(defn- ctx->grad-pairs
  "Extract [sym expr] pairs from ctx bindings."
  [ctx]
  (partition 2 (:bindings ctx)))

(defn- with-isolated-registry [f]
  (with-redefs-fn {#'tmpl/template-registry (atom (tmpl/all-templates))} f))

(deftest operation-resolution-preserves-namespaces-and-exact-registered-precedence
  (with-isolated-registry
    (fn []
      (doseq [[expected spellings]
              [['+ '[+ _plus__m_double_double]]
               ['* '[_star__m_double_double-impl]]
               ['Math/sin '[Math/sin java.lang.Math/sin sin_m_double]]
               ['raster.numeric/+ '[raster.numeric/_plus__m_double_double-impl]]
               ['raster.numeric/* '[raster.numeric/_star__m_double_double]]
               ['raster.math/sin '[raster.math/sin_m_double-impl]]]]
        (doseq [op spellings]
          ;; Directly registered alias spellings legitimately retain their exact key.
          (is (= expected (second (tmpl/resolve-template op))) (str op))))
      (doseq [op '[example/sin_custom example/exp_helper example/min_value
                    example/sin_custom_m_double-impl example/sin_m_double-impl
                    sin_helper sin_helper_m_double sin_ _plus_helper]]
        (is (nil? (tmpl/resolve-template op)) (str "must not guess builtin for " op)))
      (let [factory (fn [& _] (fn [dy] [(* 7 dy)]))
            override (fn [& _] (fn [dy] [(* 11 dy)]))]
        (tmpl/register-template! 'example/sin {:pullback-factory factory})
        (doseq [op '[example/sin example/sin_m_double example/sin_m_double-impl]]
          (is (= 'example/sin (second (tmpl/resolve-template op))))
          (is (identical? factory (tmpl/template-pullback op)))
          (is (= [21] (((tmpl/template-pullback op) 0 1) 3))))
        (tmpl/register-template! 'example/sin_m_double-impl {:pullback-factory override})
        (is (identical? override (tmpl/template-pullback 'example/sin_m_double-impl)))))))

(deftest pullback-uses-the-resolved-template-snapshot-including-legacy-factory
  (with-isolated-registry
    (fn []
      (let [old (fn [& _] (fn [dy] [dy]))
            replacement (fn [& _] (fn [dy] [(* 2 dy)]))
            resolve tmpl/resolve-template]
        (tmpl/register-template! 'example/custom {:pullback-factory old})
        (tmpl/register-template! 'example/legacy {:closure old})
        (is (identical? old (tmpl/template-pullback 'example/legacy_m_double-impl)))
        (with-redefs [tmpl/resolve-template
                      (fn [op]
                        (let [resolved (resolve op)]
                          (tmpl/register-template! 'example/custom {:pullback-factory replacement})
                          resolved))]
          (is (identical? old (tmpl/template-pullback 'example/custom))))
        (is (identical? replacement (tmpl/template-pullback 'example/custom)))))))

(deftest derived-jvp-publication-retries-public-registry-replacements
  (doseq [mutation [:explicit :derived :unrelated]]
    (with-isolated-registry
      (fn []
        (let [stale (fn [& _] :stale) fresh (fn [& _] :fresh)
              attempts (atom [])]
          (tmpl/register-template! 'example/race {:structure :before})
          (with-redefs-fn
            {#'tmpl/derive-jvp-fn
             (fn [_ _ structure]
               (swap! attempts conj structure)
               (if (= 1 (count @attempts))
                 (do
                   (case mutation
                     :explicit (tmpl/register-template! 'example/race {:jvp-fn fresh})
                     :derived (tmpl/register-template! 'example/race {:structure :after})
                     :unrelated (tmpl/register-template! 'example/other {:jvp-fn fresh}))
                   stale)
                 fresh))}
            (fn []
              (is (identical? fresh (tmpl/op-jvp-fn 'example/race_m_double-impl)))
              (is (identical? fresh (:jvp-fn (tmpl/get-template 'example/race))))
              (is (identical? fresh (tmpl/op-jvp-fn 'example/race)))
              (is (= (case mutation :explicit [:before] :derived [:before :after]
                           :unrelated [:before :before]) @attempts)))))))))

(deftest derived-jvp-cache-does-not-advance-the-semantic-revision
  (with-isolated-registry
    (fn []
      (let [f (fn [& _] :derived) calls (atom 0)]
        (tmpl/register-template! 'example/cache {:structure :test})
        (let [before (tmpl/registry-revision)]
          (with-redefs-fn {#'tmpl/derive-jvp-fn (fn [& _] (swap! calls inc) f)}
            (fn []
              (is (identical? f (tmpl/op-jvp-fn 'example/cache)))
              (is (identical? f (tmpl/op-jvp-fn 'example/cache_m_double-impl)))
              (is (= 1 @calls))
              (is (= before (tmpl/registry-revision))))))))))

;; ================================================================
;; Registry tests
;; ================================================================

(deftest registry-test
  (testing "arithmetic templates registered"
    (is (tmpl/has-template? '+))
    (is (tmpl/has-template? '-))
    (is (tmpl/has-template? '*))
    (is (tmpl/has-template? '/)))
  (testing "math templates registered"
    (is (tmpl/has-template? 'Math/sin))
    (is (tmpl/has-template? 'Math/cos))
    (is (tmpl/has-template? 'Math/exp))
    (is (tmpl/has-template? 'Math/log))
    (is (tmpl/has-template? 'Math/sqrt))
    (is (tmpl/has-template? 'Math/pow)))
  (testing "cast templates registered"
    (is (tmpl/has-template? 'double))
    (is (tmpl/has-template? 'float))
    (is (tmpl/has-template? 'long))
    (is (tmpl/has-template? 'int)))
  (testing "all templates have :grads-fn (unified path)"
    (doseq [op '[+ - * / Math/sin Math/cos Math/exp Math/log double float]]
      (is (:grads-fn (tmpl/get-template op)) (str op " missing :grads-fn"))))
  (testing "unregistered op returns nil"
    (is (nil? (tmpl/get-template 'no-such-op)))))

;; ================================================================
;; Instantiation tests (via instantiate-template-ctx)
;; ================================================================

(deftest instantiate-addition-test
  (testing "addition: d(x+y)/dx = dy, d(x+y)/dy = dy"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template '+)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['a 'b] 'r_ab 'd_ab ctx)
          pairs (ctx->grad-pairs ctx2)]
      (is (= 2 (count grad-syms)))
      ;; Both grads should be the adjoint symbol
      (is (= 'd_ab (second (first pairs))))
      (is (= 'd_ab (second (second pairs)))))))

(deftest instantiate-multiplication-test
  (testing "multiplication: d(x*y)/dx = dy*y, d(x*y)/dy = dy*x"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template '*)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['a 'b] 'r_ab 'd_ab ctx)
          pairs (ctx->grad-pairs ctx2)]
      (is (= 2 (count grad-syms)))
      ;; d/dx = dy * y → (raster.numeric/* d_ab b)
      (let [[_ grad-x] (first pairs)]
        (is (seq? grad-x))
        (is (= 'raster.numeric/* (first grad-x)))
        (is (= 'd_ab (second grad-x)))
        (is (= 'b (nth (vec grad-x) 2))))
      ;; d/dy = dy * x → (raster.numeric/* d_ab a)
      (let [[_ grad-y] (second pairs)]
        (is (seq? grad-y))
        (is (= 'raster.numeric/* (first grad-y)))
        (is (= 'd_ab (second grad-y)))
        (is (= 'a (nth (vec grad-y) 2)))))))

(deftest instantiate-exp-uses-result-test
  (testing "exp uses forward result: d/dx exp(x) = dy * result"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template 'Math/exp)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['myarg] 'exp_result 'dy_val ctx)
          pairs (ctx->grad-pairs ctx2)]
      (is (= 1 (count grad-syms)))
      (let [[_ grad-expr] (first pairs)]
        ;; Should be (raster.numeric/* dy_val exp_result)
        (is (= 'raster.numeric/* (first grad-expr)))
        (is (= 'dy_val (second grad-expr)))
        (is (= 'exp_result (nth (vec grad-expr) 2)))))))

(deftest instantiate-sin-test
  (testing "sin: d/dx sin(x) = dy * cos(x)"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template 'Math/sin)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['theta] nil 'dy ctx)
          pairs (ctx->grad-pairs ctx2)]
      (is (= 1 (count grad-syms)))
      (let [[_ grad-expr] (first pairs)]
        ;; (raster.numeric/* dy (raster.math/cos theta))
        (is (= 'raster.numeric/* (first grad-expr)))
        (is (= 'dy (second grad-expr)))
        (let [cos-call (nth (vec grad-expr) 2)]
          (is (= 'raster.math/cos (first cos-call)))
          (is (= 'theta (second cos-call))))))))

(deftest instantiate-double-cast-test
  (testing "double cast: identity passthrough"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template 'double)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['val] nil 'dy ctx)
          pairs (ctx->grad-pairs ctx2)]
      (is (= 1 (count grad-syms)))
      (is (= 'dy (second (first pairs)))))))

(deftest instantiate-long-cast-test
  (testing "long cast: zero gradient"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template 'long)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['val] nil 'dy ctx)
          pairs (ctx->grad-pairs ctx2)]
      (is (= 1 (count grad-syms)))
      (is (= 0.0 (second (first pairs)))))))

;; ================================================================
;; Mangled name resolution
;; ================================================================

(deftest resolve-template-test
  (testing "direct lookup"
    (let [[t op] (tmpl/resolve-template '+)]
      (is (some? t))
      (is (= '+ op))))
  (testing "mangled name"
    (let [[t op] (tmpl/resolve-template '_plus__m_double_double)]
      (is (some? t))
      (is (= '+ op))))
  (testing "mangled math name"
    (let [[t op] (tmpl/resolve-template 'sin_m_double)]
      (is (some? t))
      (is (= 'Math/sin op))))
  (testing "-impl suffix stripped"
    (let [[t op] (tmpl/resolve-template '_star__m_double_double-impl)]
      (is (some? t))
      (is (= '* op))))
  (testing "unknown returns nil"
    (is (nil? (tmpl/resolve-template 'unknown_func)))))

;; ================================================================
;; Division template
;; ================================================================

(deftest instantiate-division-test
  (testing "division gradient structure"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template '/)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['a 'b] nil 'dy ctx)
          pairs (ctx->grad-pairs ctx2)]
      (is (= 2 (count grad-syms)))
      ;; d/dx = dy / y → (raster.numeric// dy b)
      (let [[_ grad-x] (first pairs)]
        (is (= 'raster.numeric// (first grad-x))))
      ;; d/dy = -(dy*x / y²) — nested expression
      (let [[_ grad-y] (second pairs)]
        (is (= 'raster.numeric/- (first grad-y)))))))

;; ================================================================
;; Ctx-aware instantiation tests
;; ================================================================

(deftest instantiate-template-ctx-test
  (testing "ctx path for arithmetic templates"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template '+)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx
                            template ['a 'b] 'r_ab 'd_ab ctx)]
      (is (= 2 (count grad-syms)))
      ;; Bindings should have 4 entries (2 pairs of [sym expr])
      (is (= 4 (count (:bindings ctx2))))
      ;; Both grad expressions should be 'd_ab (addition template)
      (let [pairs (partition 2 (:bindings ctx2))]
        (is (= 'd_ab (second (first pairs))))
        (is (= 'd_ab (second (second pairs))))))))

;; ================================================================
;; Gradient count validation
;; ================================================================

(deftest gradient-count-mismatch-test
  (testing "throws when grads-fn returns wrong count"
    (let [bad-template {:grads-fn (fn [ctx actual-params _ _ gensym-fn]
                                    ;; Returns 1 gradient for 2 params — bug
                                    (let [g (gensym-fn "g")]
                                      [(update ctx :bindings into [g 'dy])
                                       [g]]))}
          ctx (make-test-ctx)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"gradient count mismatch"
                            (tmpl/instantiate-template-ctx bad-template ['a 'b] nil 'dy ctx))))))

;; ================================================================
;; Unary/binary - (multi-arity via grads-fn)
;; ================================================================

(deftest unary-negate-test
  (testing "unary - : d(-x)/dx = -dy"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template '-)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['a] nil 'dy ctx)]
      (is (= 1 (count grad-syms)))
      (let [[_ grad-expr] (first (ctx->grad-pairs ctx2))]
        (is (= 'raster.numeric/- (first grad-expr)))
        (is (= 'dy (second grad-expr))))))
  (testing "binary - : d(x-y)/dx = dy, d(x-y)/dy = -dy"
    (let [ctx (make-test-ctx)
          template (tmpl/get-template '-)
          [ctx2 grad-syms] (tmpl/instantiate-template-ctx template ['a 'b] nil 'dy ctx)]
      (is (= 2 (count grad-syms)))
      (let [pairs (ctx->grad-pairs ctx2)]
        (is (= 'dy (second (first pairs))))
        (let [[_ grad-y] (second pairs)]
          (is (= 'raster.numeric/- (first grad-y))))))))
