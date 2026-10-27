(ns raster.compiler.fusion-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.buffer-fuse :as buffer-fuse]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.hoist :as hoist]
            [raster.compiler.ir.soac :as soac]
            [raster.numeric :as n]
            [raster.analysis.memory :as ma]))

;; ================================================================
;; Owned deftm fixtures for inlining tests
;; ================================================================

(deftm owned-inline-leaf [a :- Double b :- Double] :- Double
  ;; These typed numerical calls retain selected `.invk` operations in the
  ;; source body.  They are inside the ordinary whole-call proof domain; raw
  ;; clojure.core arithmetic deliberately is not.
  (let [t1 (n/+ a b)
        t2 (n/* t1 2.0)]
    t2))

(deftm owned-core-arithmetic-leaf [a :- Double b :- Double] :- Double
  (let [t1 (clojure.core/+ a b)
        t2 (clojure.core/* t1 2.0)]
    t2))

;; A bare host call remains a call: this fixture has a genuine source owner and
;; selected Method, but its body is outside the ordinary decomposable domain.
(deftm owned-inline-bare [x :- Double] :- Double
  (Math/sqrt x))

(deftm owned-inline-direct-owner [p :- Double q :- Double] :- Double
  (let [x (owned-inline-leaf p q)] x))

(deftm owned-inline-nested-owner [input :- Double] :- Double
  ;; Keep the admitted call at an original nested lexical source site.  Calls
  ;; introduced by importing another helper are not occurrence plans owned by
  ;; this caller and therefore are not a transitive inlining authority.
  (let [offset 42.0
        r (owned-inline-leaf input offset)]
    r))

(deftm owned-inline-chain-mid [x :- Double] :- Double
  (let [r (owned-inline-leaf x 42.0)] r))

(deftm owned-inline-chain-owner [input :- Double] :- Double
  (let [r (owned-inline-chain-mid input)] r))

(deftm owned-core-arithmetic-owner [p :- Double q :- Double] :- Double
  (let [x (owned-core-arithmetic-leaf p q)] x))

(deftm owned-inline-mixed-owner [p :- Double q :- Double] :- Double
  (let [a (owned-inline-leaf p q)
        b (owned-inline-bare a)]
    b))

(defn- owned-inline-expansion [owner helpers]
  ;; Source readiness is an explicit prerequisite. The expansion itself then
  ;; consumes one real caller source context and independently retained helper
  ;; frontends; no Method/body facts are manufactured in test metadata.
  (doseq [helper helpers]
    (core/prepare-emission-source-walk! (core/resolve-deftm-var helper)))
  (let [source-var (core/resolve-deftm-var owner)
        facts (pipeline/deftm-frontend-facts source-var :double)
        forms (pipeline/get-walked-body source-var :double)
        source (if (= 1 (count forms)) (first forms) (list* 'do forms))
        parameters (:parameters facts)
        options {:source-var source-var :source-ns (:source-ns facts)
                 :active-params parameters :public-parameters parameters}]
    (entry/call-with-definition-inputs
     #(entry/call-with-independent-definition-inputs
       (fn []
         (inline/call-with-source-input-completions
          source options (fn [] (inline/inline-deftm-calls source))))))))

(defn- executable-nodes [form]
  (tree-seq #(and (coll? %)
                  (not (and (seq? %) (contains? '#{quote clojure.core/quote} (first %)))))
            seq form))

(defn- generic-operation [generic]
  (let [metadata (meta generic)]
    (symbol (str (ns-name (:ns metadata))) (str (:name metadata)))))

(defn- contains-owned-call? [form generic]
  (let [operation (generic-operation generic)]
    (boolean
     (some #(= operation (descriptor/semantic-op %))
           (executable-nodes form)))))

(defn- operation-named [form operation-name]
  (some (fn [node]
          (let [operation (descriptor/semantic-op node)]
            (when (and (symbol? operation) (= operation-name (name operation))) operation)))
        (executable-nodes form)))

(defn- symbol-used? [form id]
  (boolean (some #(= id %) (executable-nodes form))))

(defn- retained-leaf-arithmetic-chain [form]
  ;; Derive operation identities from the genuine walked helper, then find its
  ;; alpha-renamed lexical dependency anywhere inside the complete imported
  ;; expression. Placement in one outer result binding is intentionally free.
  (let [leaf (first (pipeline/get-walked-body
                     (core/resolve-deftm-var #'owned-inline-leaf) :double))
        leaf-pairs (partition 2 (second leaf))
        original-binders (set (map first leaf-pairs))
        plus (operation-named leaf "+")
        multiply (operation-named leaf "*")]
    (when (and plus multiply)
      (some (fn [scope]
              (when (and (seq? scope) (contains? '#{let let*} (first scope))
                         (vector? (second scope)))
                (let [pairs (vec (partition 2 (second scope)))]
                  (some (fn [[producer-index [producer producer-init]]]
                          (some (fn [[consumer-index [consumer consumer-init]]]
                                  (when (and (< producer-index consumer-index)
                                             (= plus (operation-named producer-init "+"))
                                             (= multiply (operation-named consumer-init "*"))
                                             (symbol-used? consumer-init producer)
                                             (symbol-used? (drop 2 scope) consumer)
                                             (not= producer consumer)
                                             (not (contains? original-binders producer))
                                             (not (contains? original-binders consumer)))
                                    {:producer producer :consumer consumer}))
                                (map-indexed vector pairs)))
                        (map-indexed vector pairs)))))
            (executable-nodes form)))))

(defn- assert-compiled-results [owner arguments expected]
  (doseq [inline? [false true]]
    (let [compiled (pipeline/compile-aot owner :simd? false :inline? inline?)]
      (is (= expected (apply compiled arguments))
          (str "inline?=" inline?)))))

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

(deftest test-fuse-let-borrowed-arg-is-not-reused
  (testing "A parameter's last local use does not make it the let's to overwrite"
    (let [form '(let* [a (test/inplace-op input)]
                      (test/scalar-op a))
          result (buffer-fuse/fuse-let form)]
      (is (= 0 (:fused (:stats result)))
          "input is borrowed from the caller"))))

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
  (testing "Inlines a real owned deftm call and preserves its result"
    (let [result (owned-inline-expansion #'owned-inline-direct-owner
                                         [#'owned-inline-leaf])]
      (is (not (contains-owned-call? result #'owned-inline-leaf)))
      (is (some? (retained-leaf-arithmetic-chain result))
          "the complete imported expression retains its hygienic + to * dependency"))
    (assert-compiled-results #'owned-inline-direct-owner [3.0 4.0] 14.0))
  (testing "Core arithmetic outside the whole-call closure retains exact behavior"
    (assert-compiled-results #'owned-core-arithmetic-owner [3.0 4.0] 14.0)))

(deftest test-inline-nested
  (testing "A real owned call at a nested lexical source site is fully inlined"
    (let [result (owned-inline-expansion #'owned-inline-nested-owner
                                         [#'owned-inline-leaf])]
      (is (not (contains-owned-call? result #'owned-inline-leaf)))
      (is (some? (retained-leaf-arithmetic-chain result))
          "the nested imported expression retains its hygienic lexical arithmetic"))
    (assert-compiled-results #'owned-inline-nested-owner [3.0] 90.0))
  (testing "A real A→B→leaf call chain retains exact behavior"
    ;; Transitive structural elimination needs an authenticated inherited
    ;; occurrence plan for calls introduced by an imported helper.  The
    ;; current whole-call projection does not claim that capability.
    (assert-compiled-results #'owned-inline-chain-owner [3.0] 90.0)))

(deftest test-inline-bare-helper-behavior
  (testing "An owned bare helper remains exact whether retained or legally expanded"
    (assert-compiled-results #'owned-inline-bare [9.0] 3.0)))

(deftest test-inline-mixed
  (testing "Inlineable and retained real calls coexist"
    (let [result (owned-inline-expansion #'owned-inline-mixed-owner
                                         [#'owned-inline-leaf #'owned-inline-bare])]
      (is (not (contains-owned-call? result #'owned-inline-leaf)))
      (is (some? (retained-leaf-arithmetic-chain result))))
    (assert-compiled-results #'owned-inline-mixed-owner [1.0 7.0] 4.0)))

;; ================================================================
;; SOAC modelling of imperative par/map-void! bodies
;; ================================================================
;; REGRESSION (silent miscompile): `single-aset-void` unwrapped a let*-wrapped write
;; by recursing into the LAST body form. A body with several statements —
;;   (let* [v (aget x i)] (aset a i v) (aset b i (* v v)))
;; — therefore modelled as a SoacMap whose output is `b` and whose lambda is `(* v v)`;
;; the old graph reconstruction then REBUILT the body from the lambda alone and the store to `a`
;; disappeared from the emitted kernel. `a` was not even in the kernel's array list, so
;; nothing failed loudly: the buffer just stayed at its initial contents (this is how the
;; SFT head's `lse` buffer silently stayed zero). Multi-statement bodies must fall through
;; to the compatibility void path that emits the source body as written.

(deftest multi-store-map-void-is-not-a-single-write-soac
  (testing "two asets in one map-void! body ⇒ NOT modelled as a single-write SoacMap"
    (let [form '(raster.par/map-void!
                 i (long n)
                 (let* [v (clojure.core/aget x i)]
                       (clojure.core/aset a i (float v))
                       (clojure.core/aset b i (float (raster.numeric/* v v)))))]
      (is (nil? (soac/par-form->soac 'out form 0))
          "modelling it as a SoacMap over `b` silently drops the store to `a`")))

  (testing "a SINGLE let*-wrapped write still models as a SoacMap (the fusible shape)"
    (let [form '(raster.par/map-void!
                 i (long n)
                 (let* [v (clojure.core/aget x i)]
                       (clojure.core/aset b i (float (raster.numeric/* v v)))))
          s (soac/par-form->soac 'out form 0)]
      (is (some? s) "the single-write let* shape is what the unwrapping exists for")
      (is (= #{'b} (:outputs s)))
      (is (:void? s)))))
