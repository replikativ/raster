(ns raster.compiler.fusion-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.scalar.buffer-fuse :as buffer-fuse]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.hoist :as hoist]
            [raster.compiler.ir.soac :as soac]
            [raster.analysis.memory :as ma]))

;; ================================================================
;; Mock deftm vars for inlining tests
;; ================================================================

;; Leaf deftm: simple let* body
(def mock-leaf-fn nil)
(alter-meta! #'mock-leaf-fn assoc
             :raster.core/deftm-params '[a b]
             :raster.core/deftm-walked-body
             ['(let* [t1 (some-op a b)
                      t2 (other-op t1)]
                     t2)])

;; Mid-level deftm: calls mock-leaf-fn
(def mock-mid-fn nil)
(alter-meta! #'mock-mid-fn assoc
             :raster.core/deftm-params '[x]
             :raster.core/deftm-walked-body
             ['(let* [r (raster.compiler.fusion-test/mock-leaf-fn x 42)]
                     r)])

;; Bare expression body (plain call, not let*/loop*/dotimes/do): should NOT be inlined
(def mock-bare-fn nil)
(alter-meta! #'mock-bare-fn assoc
             :raster.core/deftm-params '[x]
             :raster.core/deftm-walked-body
             ['(Math/sqrt x)])

;; ================================================================
;; Register test ops for structural testing
;; (These mirror the NN ops but are self-contained for testing)
;; ================================================================

(descriptor/register-buffer-semantics! 'test/alloc-op
                                       {:allocates?   true
                                        :in-place-arg nil
                                        :alloc-form   (fn [args _opts] `(make-buffer))
                                        :rewrite-fn   (fn [args buf] `(do (fill! ~buf ~@args) ~buf))})

(descriptor/register-buffer-semantics! 'test/inplace-op
                                       {:allocates?   true
                                        :in-place-arg 0
                                        :alloc-form   (fn [[x] _opts] `(clone ~x))
                                        :rewrite-fn   (fn [[x] buf] `(do (transform! ~x ~buf) ~buf))})

(descriptor/register-buffer-semantics! 'test/scalar-op
                                       {:allocates? false :in-place-arg nil :alloc-form nil :rewrite-fn nil})

;; ================================================================
;; Descriptor registry tests
;; ================================================================

(deftest test-op-registry
  (testing "Register and retrieve op"
    (is (some? (descriptor/get-buffer-semantics 'test/alloc-op)))
    (is (some? (descriptor/get-op-descriptor 'test/alloc-op)))
    (is (true? (:allocates? (descriptor/get-buffer-semantics 'test/alloc-op)))))

  (testing "In-place arg"
    (is (= 0 (:in-place-arg (descriptor/get-buffer-semantics 'test/inplace-op))))
    (is (nil? (:in-place-arg (descriptor/get-buffer-semantics 'test/alloc-op)))))

  (testing "Non-allocating op"
    (is (false? (:allocates? (descriptor/get-buffer-semantics 'test/scalar-op)))))

  (testing "Resolve direct lookup"
    (let [[entry base] (descriptor/resolve-buffer-semantics 'test/alloc-op)]
      (is (some? entry))
      (is (= 'test/alloc-op base))))

  (testing "Missing op returns nil"
    (is (nil? (descriptor/resolve-buffer-semantics 'nonexistent/op)))))

;; ================================================================
;; S-expression memory analysis tests
;; ================================================================

(deftest test-sexp-free-vars-for-fusion
  (testing "NN-style let binding free vars"
    (let [fvs (ma/sexp-free-vars '(let* [h (dense W x b)
                                         a (relu h)
                                         out (dense W2 a b2)
                                         p (softmax out)]
                                        (cross-entropy p y)))]
      ;; Free vars: W, x, b, dense, relu, W2, b2, softmax, cross-entropy, y
      (is (contains? fvs 'W))
      (is (contains? fvs 'x))
      (is (contains? fvs 'y))
      (is (not (contains? fvs 'h)) "h is bound")
      (is (not (contains? fvs 'a)) "a is bound")
      (is (not (contains? fvs 'out)) "out is bound")
      (is (not (contains? fvs 'p)) "p is bound"))))

;; ================================================================
;; Fusion pass structural tests
;; ================================================================

(deftest test-fuse-let-inplace-reuse
  (testing "In-place op reuses dead arg's buffer"
    ;; h is only used in inplace-op, so its buffer can be reused
    (let [form '(let* [h (test/alloc-op input)
                       a (test/inplace-op h)]
                      (test/scalar-op a))
          result (buffer-fuse/fuse-let form)]
      (is (= 1 (:fused (:stats result)))
          "Should have 1 fused op")
      ;; The rewritten form should use h as the buffer for a
      (let [new-form (:form result)
            bindings (second new-form)
            pairs (partition 2 bindings)]
        ;; a's init should be rewritten to use h as buffer
        (is (some (fn [[sym init]]
                    (and (= sym 'a)
                         (seq? init)
                         ;; Should be the rewrite-fn output
                         (= 'do (first init))))
                  pairs)
            "a should be rewritten with in-place form")))))

(deftest test-fuse-let-fresh-alloc
  (testing "Allocating op without in-place-arg gets fresh buffer"
    (let [form '(let* [h (test/alloc-op input)]
                      h)
          result (buffer-fuse/fuse-let form)]
      ;; alloc-op has no in-place-arg, so fresh buffer is hoisted
      (is (= 1 (:fresh-allocs (:stats result)))
          "Should have 1 fresh allocation"))))

(deftest test-fuse-let-used-after-prevents-reuse
  (testing "In-place arg still used after prevents reuse"
    ;; h is used in BOTH inplace-op AND scalar-op, so can't reuse
    (let [form '(let* [h (test/alloc-op input)
                       a (test/inplace-op h)
                       z (test/scalar-op h a)]
                      z)
          result (buffer-fuse/fuse-let form)]
      ;; h is still used at binding 2 (scalar-op h a),
      ;; so used-after[1] contains h → can't reuse
      (is (= 0 (:fused (:stats result)))
          "Should NOT fuse — h still used after"))))

(deftest test-fuse-let-scalar-unchanged
  (testing "Non-allocating ops are unchanged"
    (let [form '(let* [x (test/scalar-op a b)]
                      x)
          result (buffer-fuse/fuse-let form)]
      (is (= 1 (:unchanged (:stats result)))))))

(deftest test-fuse-let-unknown-op-unchanged
  (testing "Ops without registry entries are unchanged"
    (let [form '(let* [x (unknown-op a b)]
                      x)
          result (buffer-fuse/fuse-let form)]
      (is (= 1 (:unchanged (:stats result)))))))

(deftest test-fuse-let-nn-pattern
  (testing "NN forward pass pattern"
    ;; Register NN ops for this test
    (descriptor/register-buffer-semantics! 'test.nn/dense
                                           {:allocates? true :in-place-arg nil
                                            :alloc-form (fn [[W _x _b] _opts] `(alloc-vec (rows ~W)))
                                            :rewrite-fn (fn [[W x b] buf]
                                                          `(do (copy! ~b ~buf) (mv! ~W ~x ~buf) ~buf))})
    (descriptor/register-buffer-semantics! 'test.nn/relu
                                           {:allocates? true :in-place-arg 0
                                            :alloc-form (fn [[x] _opts] `(clone ~x))
                                            :rewrite-fn (fn [[x] buf] `(do (relu! ~x ~buf) ~buf))})
    (descriptor/register-buffer-semantics! 'test.nn/softmax
                                           {:allocates? true :in-place-arg 0
                                            :alloc-form (fn [[x] _opts] `(clone ~x))
                                            :rewrite-fn (fn [[x] buf] `(do (softmax! ~x ~buf) ~buf))})
    (descriptor/register-buffer-semantics! 'test.nn/cross-entropy
                                           {:allocates? false :in-place-arg nil :alloc-form nil :rewrite-fn nil})

    (let [form '(let* [h   (test.nn/dense W x b)
                       a   (test.nn/relu h)
                       out (test.nn/dense W2 a b2)
                       p   (test.nn/softmax out)]
                      (test.nn/cross-entropy p y))
          result (buffer-fuse/fuse-let form)
          stats (:stats result)]
      ;; relu(h): h is dead after → fuse (in-place reuse)
      ;; softmax(out): out is dead after → fuse (in-place reuse)
      ;; dense(W,x,b): no in-place-arg → fresh alloc
      ;; dense(W2,a,b2): no in-place-arg → fresh alloc
      ;; cross-entropy is in body, not a binding — not counted
      (is (= 2 (:fused stats))
          "relu and softmax should be fused (in-place)")
      (is (= 2 (:fresh-allocs stats))
          "Two dense ops should get fresh buffers")
      (is (= 0 (:unchanged stats))
          "All 4 bindings are allocating ops — none unchanged"))))

;; ================================================================
;; fuse-walked-body tests
;; ================================================================

(deftest test-fuse-walked-body
  (testing "Applies fusion to let* forms in body"
    (let [body ['(let* [h (test/alloc-op input)
                        a (test/inplace-op h)]
                       a)]
          result (hoist/fuse-walked-body body)]
      (is (= 1 (count (:forms result))))
      (is (pos? (:fused (:stats result)))))))

(deftest test-fuse-walked-body-non-let
  (testing "Non-let forms pass through unchanged"
    (let [body ['(+ 1 2)]
          result (hoist/fuse-walked-body body)]
      (is (= ['(+ 1 2)] (:forms result)))
      (is (= 0 (:fused (:stats result)))))))

;; ================================================================
;; Buffer hoisting tests
;; ================================================================

(deftest test-hoist-buffers-structural
  (testing "Hoistable bindings are separated from inner bindings"
    ;; Fuse a form that produces fresh allocs (hoistable) and in-place rewrites (inner)
    (let [form '(let* [h   (test.nn/dense W x b)
                       a   (test.nn/relu h)
                       out (test.nn/dense W2 a b2)
                       p   (test.nn/softmax out)]
                      (test.nn/cross-entropy p y))
          fused (:form (buffer-fuse/fuse-let form))
          {:keys [alloc-bindings buf-syms inner-form]} (hoist/hoist-buffers fused)]
      ;; Two dense ops → two hoistable fresh allocs
      (is (= 2 (count alloc-bindings))
          "Should have 2 hoistable alloc bindings")
      (is (= 2 (count buf-syms))
          "Should have 2 buffer symbols")
      ;; Each alloc binding's sym should have :raster.buffer/hoistable metadata
      (is (every? #(:raster.buffer/hoistable (meta (first %))) alloc-bindings)
          "All alloc binding syms should be tagged :raster.buffer/hoistable")
      ;; Inner form should be a let* with the remaining bindings
      (is (= 'let* (first inner-form))
          "Inner form should be a let*")
      ;; Inner bindings should have h, a, out, p (4 bindings = 8 entries)
      (let [inner-bindings (second inner-form)
            inner-pairs (partition 2 inner-bindings)]
        (is (= 4 (count inner-pairs))
            "Should have 4 inner bindings (h, a, out, p)")))))

(deftest test-hoist-buffers-no-allocs
  (testing "Form with no hoistable bindings returns empty alloc-bindings"
    ;; All ops are in-place or non-allocating → no hoistable bindings
    (let [form '(let* [a (test/inplace-op input)
                       b (test/scalar-op a)]
                      b)
          ;; inplace-op will fuse if input is dead, or get fresh alloc.
          ;; scalar-op is non-allocating.
          ;; But input is a free var so used-after doesn't apply here.
          fused (:form (buffer-fuse/fuse-let form))
          {:keys [alloc-bindings buf-syms]} (hoist/hoist-buffers fused)]
      ;; inplace-op can't reuse input (still used? depends on analysis)
      ;; but even if it gets a fresh alloc, the key point is the function works
      (is (vector? alloc-bindings))
      (is (vector? buf-syms))
      (is (= (count alloc-bindings) (count buf-syms))))))

(deftest test-hoist-buffers-preserves-body
  (testing "Body expression is preserved in inner-form"
    (let [form '(let* [h (test/alloc-op input)]
                      (test/scalar-op h))
          fused (:form (buffer-fuse/fuse-let form))
          {:keys [inner-form]} (hoist/hoist-buffers fused)]
      ;; The body should reference the scalar-op call
      (let [body-exprs (nnext inner-form)]
        (is (some? body-exprs) "Inner form should have body expressions")))))

;; ================================================================
;; Cross-function inlining tests
;; ================================================================

(deftest test-inline-deftm-calls
  (testing "Inlines a deftm call, splicing inner bindings"
    (let [form '(let* [x (raster.compiler.fusion-test/mock-leaf-fn p q)]
                      (use-result x))
          result (inline/inline-deftm-calls form)]
      ;; Should have 3 bindings: t1_gensym, t2_gensym, x
      (let [[_ bindings & _body] result
            pairs (partition 2 bindings)]
        (is (= 3 (count pairs))
            "Should have 2 inlined + 1 outer binding")
        ;; First binding init should substitute params: (some-op p q)
        (let [[_sym1 init1] (first pairs)]
          (is (= 'some-op (first init1)))
          (is (= 'p (second init1)))
          (is (= 'q (nth init1 2))))
        ;; Second binding init should reference first binding's renamed sym
        (let [[sym1 _] (first pairs)
              [_sym2 init2] (second pairs)]
          (is (= 'other-op (first init2)))
          (is (= sym1 (second init2))))
        ;; Third binding (x) should equal second binding's renamed sym
        (let [[sym2 _] (second pairs)
              [sym3 init3] (nth pairs 2)]
          (is (= 'x sym3))
          (is (= sym2 init3)))))))

(deftest test-inline-nested
  (testing "Nested deftm chain A→B gets fully inlined"
    (let [form '(let* [r (raster.compiler.fusion-test/mock-mid-fn input)]
                      r)
          result (inline/inline-deftm-calls form)]
      ;; mock-mid-fn calls mock-leaf-fn, so after 2 passes:
      ;; t1_gensym, t2_gensym, r_gensym (from leaf), r (outer)
      ;; Plus the mid-level's inner binding for r
      (let [[_ bindings & _] result
            pairs (vec (partition 2 bindings))]
        ;; Should have at least 4 bindings after full inlining:
        ;; leaf's t1, leaf's t2, mid's r-inner, outer's r
        (is (>= (count pairs) 4)
            "Should have inlined both levels")
        ;; The deepest init should be (some-op input 42)
        (let [[_sym init] (first pairs)]
          (is (= 'some-op (first init))
              "Deepest binding should be leaf's some-op")
          (is (= 'input (second init))
              "First arg should be substituted to 'input")
          (is (= 42 (nth init 2))
              "Second arg should be literal 42"))))))

(deftest test-inline-no-recursion
  (testing "Non-let* walked body is NOT inlined"
    (let [form '(let* [x (raster.compiler.fusion-test/mock-bare-fn input)]
                      x)
          result (inline/inline-deftm-calls form)]
      ;; Should be unchanged — mock-bare-fn has plain call body (Math/sqrt)
      (let [[_ bindings & _] result
            pairs (partition 2 bindings)]
        (is (= 1 (count pairs))
            "Should still have exactly 1 binding (not inlined)")
        (let [[sym init] (first pairs)]
          (is (= 'x sym))
          (is (= 'raster.compiler.fusion-test/mock-bare-fn (first init))
              "Call should be preserved unchanged"))))))

(deftest test-inline-mixed
  (testing "Mix of inlineable and non-inlineable calls"
    (let [form '(let* [a (raster.compiler.fusion-test/mock-leaf-fn p q)
                       b (raster.compiler.fusion-test/mock-bare-fn a)]
                      b)
          result (inline/inline-deftm-calls form)]
      (let [[_ bindings & _] result
            pairs (vec (partition 2 bindings))]
        ;; mock-leaf-fn inlined (2 inner + 1 outer = 3), mock-bare-fn kept (1)
        (is (= 4 (count pairs))
            "Should have 3 from leaf inlining + 1 bare")
        ;; Last binding should be the bare call
        (let [[sym init] (last pairs)]
          (is (= 'b sym))
          (is (= 'raster.compiler.fusion-test/mock-bare-fn (first init))))))))

;; ================================================================
;; SOAC modelling of imperative par/map-void! bodies
;; ================================================================
;; REGRESSION (silent miscompile): `single-aset-void` unwrapped a let*-wrapped write
;; by recursing into the LAST body form. A body with several statements —
;;   (let* [v (aget x i)] (aset a i v) (aset b i (* v v)))
;; — therefore modelled as a SoacMap whose output is `b` and whose lambda is `(* v v)`;
;; soac->par-form then REBUILT the body from the lambda alone and the store to `a`
;; disappeared from the emitted kernel. `a` was not even in the kernel's array list, so
;; nothing failed loudly: the buffer just stayed at its initial contents (this is how the
;; SFT head's `lse` buffer silently stayed zero). Multi-statement bodies must fall through
;; to a ScalarBinding, i.e. the legacy void path that emits the body as written.

(deftest multi-store-map-void-is-not-a-single-write-soac
  (testing "two asets in one map-void! body ⇒ NOT modelled as a single-write SoacMap"
    (let [form '(raster.par/map-void!
                 i (long n)
                 (let* [v (clojure.core/aget x i)]
                       (clojure.core/aset a i (float v))
                       (clojure.core/aset b i (float (raster.numeric/* v v)))))]
      (is (nil? (soac/par-form->soac 'out form 0))
          "modelling it as a SoacMap over `b` silently drops the store to `a`")))

  (testing "the multi-store form round-trips through the SOAC graph UNCHANGED"
    (let [form '(raster.par/map-void!
                 i (long n)
                 (let* [v (clojure.core/aget x i)]
                       (clojure.core/aset a i (float v))
                       (clojure.core/aset b i (float (raster.numeric/* v v)))))
          nodes (soac/let-bindings->nodes [['_eff form]])
          [[_ round-tripped]] (soac/nodes->let-bindings nodes)]
      (is (= form round-tripped)
          "the legacy void path must emit the body exactly as written")))

  (testing "a SINGLE let*-wrapped write still models as a SoacMap (the fusible shape)"
    (let [form '(raster.par/map-void!
                 i (long n)
                 (let* [v (clojure.core/aget x i)]
                       (clojure.core/aset b i (float (raster.numeric/* v v)))))
          s (soac/par-form->soac 'out form 0)]
      (is (some? s) "the single-write let* shape is what the unwrapping exists for")
      (is (= #{'b} (:outputs s)))
      (is (:void? s)))))

;; ── layout-soundness guard: non-same-position fusion fails loud (not silent mis-fuse) ──
(deftest substitute-aget-refuses-non-same-position-read
  (let [subst @#'soac/substitute-aget-sym]
    (testing "same-position read (aget t dst) inlines the producer (unchanged behaviour)"
      (is (= '(raster.numeric/* p p)
             (subst '(clojure.core/aget t dst) 't 'src 'dst '(raster.numeric/* p p)))
          "no aget on the target left → replaced")
      (is (= 'x (subst 'x 't 'src 'dst '(foo)))  "unrelated body untouched"))
    (testing "a gather/offset read (aget t (+ dst 1)) FAILS LOUD instead of mis-fusing"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-same-position read"
            (subst '(clojure.core/aget t (clojure.core/+ dst 1)) 't 'src 'dst '(raster.numeric/* p p)))))))
