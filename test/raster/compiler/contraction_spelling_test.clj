(ns raster.compiler.contraction-spelling-test
  "THE ROUTE A CONTRACTION TAKES MUST NOT DEPEND ON HOW ITS ARRAY READ IS SPELLED. Device-free.

   Four spellings denote the same array read:

     (aget A i)                  bare — what every hand-written test form in this repo used
     (clojure.core/aget A i)     WHAT THE WALKER ACTUALLY EMITS (walker's :array-op handler
                                 qualifies into clojure.core)
     (raster.arrays/aget A i)    what CLAUDE.md tells authors to write, and what
                                 c_emit/normalize-array-prims produces from a devirtualized read
     (.invk impl A i)            devirtualized — array at position 2, index at 3, semantic op
                                 recovered from :raster.op/original metadata

   Twelve matchers on the contraction vertical decided \"is this an array read\" with
   `(= 'aget (first f))`, so on real compiled IR they all saw ZERO operands. Measured consequences:

     • f16 matmul routed to :naive-segred instead of :dpas — 577 vs 4357 GFLOP/s on Arc 140V, where
       the routed :dpas path is at PARITY with the hand-written XMX GEMM. Pure loss.
     • :byte/staged emitted SYNTACTICALLY INVALID OpenCL: `staged_contract(, __global float*
       restrict out, int _nseg)` — leading comma, operands absent from the signature while the body
       still read them.
     • a :decode zero-point SILENTLY DROPPED: `acc += a[l]*b[l]` where the semantics are
       `(a[l]-zp)*b[l]`. A wrong answer, not a slow one.

   WHY STRATEGY EQUALITY IS NOT ENOUGH, and why these tests assert emitted SOURCE. At `:byte` with
   an :nn layout every spelling routes to `:quant-naive` — the strategies agree while the kernel is
   invalid C. A test that compared only `:strategy` would have passed throughout. So the property
   asserted here is the strongest available: for one contraction, every spelling produces the same
   strategy, the same :array-params, and byte-identical kernel source modulo the name gensym."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-route :as cr]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.core.op-descriptor :as od]))

;; ── the same contraction, four spellings ────────────────────────────────────────────
(defn- read-form
  "An array read `arr[idx]` in one of the four spellings."
  [spelling arr idx]
  (case spelling
    :bare   (list 'aget arr idx)
    :core   (list 'clojure.core/aget arr idx)
    :arrays (list 'raster.arrays/aget arr idx)
    :invk   (with-meta (list '.invk 'aget-impl arr idx)
              {:raster.op/original 'raster.arrays/aget})))

(def ^:private spellings [:bare :core :arrays :invk])

(defn- mm
  "C[i,j] = Σ_l A[i,l]·B[l,j]. `:nt` stores B as B[j,l] (K-contiguous, the dp4a layout).
   `:cast?` wraps extents in `(long …)` the way the walked dialect does."
  [spelling m n k & {:keys [nt decode cast?]}]
  (let [ext (fn [e] (if cast? (list 'long e) e))
        row (list 'clojure.core/+ (list 'clojure.core/* 'i (ext k)) 'l)
        col (if nt
              (list 'clojure.core/+ (list 'clojure.core/* 'j (ext k)) 'l)
              (list 'clojure.core/+ (list 'clojure.core/* 'l (ext n)) 'j))]
    (concat
     (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
           (list 'raster.numeric/* (read-form spelling 'A row) (read-form spelling 'B col)))
     (when decode [:decode {'A (list 'clojure.core/- 'x 8)}]))))

(defn- normalize-src
  "Kernel source with the name gensym erased, so two spellings' sources are comparable."
  [src]
  (some-> src (str/replace #"_\d{3,}" "_N")))

(defn- route [form dtype]
  (cr/route-contraction form :dtype dtype))

;; ── the property ────────────────────────────────────────────────────────────────────
(def ^:private cases
  [{:label "f16 canonical matmul → the PEAK tensorized leaf" :dtype :half  :opts {}
    :expect :dpas}
   {:label "f64 → the portable tiled leaf"                   :dtype :double :opts {}
    :expect :regtiled}
   {:label "int8 :nn → widening quant"                       :dtype :byte  :opts {}
    :expect :quant-naive}
   {:label "int8 :nt (K-contiguous) → dp4a"                  :dtype :byte  :opts {:nt true}
    :expect :dp4a}
   {:label "walked dialect: extents wrapped in (long …)"      :dtype :half  :opts {:cast? true}
    :expect :dpas}
   ;; THE CASE THAT WAS MISSING. Every other row either has no casts, or routes to a leaf that
   ;; REGENERATES the index from an axis-map (dpas, staged) and so drops them for free. A leaf that
   ;; emits the body's index VERBATIM is the only place a `(long 640)` extent can reach the C — and
   ;; it did, producing `A[((i * (long)(640)) + l)]` for the walked spelling against `A[((i * 640) +
   ;; l)]` for the bare one. Non-square dims too, so a row/col mix-up cannot hide behind symmetry.
   {:label "walked dialect + a VERBATIM-index leaf (regtiled)" :dtype :double
    :opts {:cast? true} :expect :regtiled :dims [96 160 64]}])

(deftest routing-is-independent-of-the-read-spelling
  (doseq [{:keys [label dtype opts expect dims]} cases]
    (testing label
      (let [[m n k] (or dims [128 128 128])
            rs (into {} (for [sp spellings]
                          [sp (route (apply mm sp m n k (mapcat identity opts)) dtype)]))]
        (doseq [sp spellings]
          (is (= expect (:strategy (get rs sp)))
              (str sp " must reach " expect)))
        (testing "…and :array-params agree — this is the assertion that catches the invalid-C bug,
                  where every spelling routed to the same strategy but the operands vanished from
                  the kernel signature while the body still read them"
          (doseq [sp spellings]
            (is (= '[A B] (vec (:array-params (get rs sp))))
                (str sp " operand list"))))
        (testing "…and the emitted kernel source is byte-identical modulo the name gensym"
          (let [ref (normalize-src (:source (get rs :bare)))]
            (doseq [sp (rest spellings)]
              (is (= ref (normalize-src (:source (get rs sp))))
                  (str sp " source differs from bare")))))))))

(deftest an-invalid-kernel-signature-is-impossible
  (testing "the concrete artifact of the old defect: a signature beginning `(,` — a leading comma
            with zero parameters. Pinned directly so the shape cannot come back by another route"
    (doseq [sp spellings
            dtype [:half :double :byte]]
      (let [src (:source (route (mm sp 128 128 128) dtype))
            sig (re-find #"__kernel void [^)]*\)" (str src))]
        (is (not (re-find #"\(\s*," (str sig)))
            (str sp "/" dtype " emitted a parameterless signature: " sig))
        (doseq [arr ["A" "B"]]
          (is (re-find (re-pattern (str "restrict\\s+" arr "\\b")) (str sig))
              (str sp "/" dtype " reads " arr " but does not declare it: " sig)))))))

;; ── rewriters: what no strategy assertion can see ───────────────────────────────────
(deftest a-decode-zero-point-reaches-the-emitted-kernel
  (testing "the :decode load-lambda is the per-operand zero-point subtraction. Dropping it is a
            WRONG ANSWER that leaves the strategy, the signature and the parameter count intact —
            invisible to every other assertion here"
    (doseq [sp spellings]
      (let [src (str (:source (route (mm sp 128 128 128 :decode true) :byte)))
            acc (re-find #"acc_0 \+=[^;]*;" src)]
        (is (some? acc) (str sp ": no accumulate statement found"))
        (is (re-find #"- 8" (str acc))
            (str sp ": zero-point silently dropped → " acc))))))

(deftest the-router-declines-where-the-emitter-would-throw
  (testing "int8 + :nt + :decode is dp4a-LAYOUT-legal but the dp4a leaf REPLACES the body, so it
            would discard the decode. The emitter throws — correctly, for an explicit
            :tensorize-inner? true — so the ROUTER must stop asking. Before the gate consulted the
            emitter's own predicate, broadening the matchers would have converted a correct
            :quant-naive kernel into a hard compile error"
    (doseq [sp spellings]
      (let [r (try (route (mm sp 128 128 128 :nt true :decode true) :byte)
                   (catch clojure.lang.ExceptionInfo e {:threw (:reason (ex-data e))}))]
        (is (= :quant-naive (:strategy r))
            (str sp ": expected a decline to :quant-naive, got " (pr-str r)))))))

;; ── the second, independent blocker: integer casts in the index ──────────────────────
(deftest affine-matching-sees-through-integer-casts
  (testing "the walked dialect wraps extents in `(long …)`. While `affine` treated `(long 128)` as
            an OPAQUE atom, the index was not the row-major layout, the DPAS orientation gate
            declined with :non-canonical-orientation, and a compiled deftm could not reach the
            tensorized leaf EVEN WITH its operands recognized. Two independent blockers on one path"
    (let [want (am/of-axes '[[i 128] [l 128]])]
      (is (am/index-matches? want '(clojure.core/+ (clojure.core/* i 128) l)))
      (is (am/index-matches? want '(clojure.core/+ (clojure.core/* i (long 128)) l)))
      (is (am/index-matches? want '(clojure.core/+ (clojure.core/* i (int 128)) l)))))
  (testing "…while genuinely opaque subterms stay opaque — distributing quot/mod would make two
            different gathers compare equal, which is a silent wrong-operand bug"
    (is (false? (am/index= '(clojure.core/quot i 4) '(clojure.core/mod i 4) '[i])))))

;; ── the shared helpers ──────────────────────────────────────────────────────────────
(deftest aget-reads-recognizes-every-spelling
  (doseq [sp spellings]
    (testing (str sp)
      (let [f (read-form sp 'A '(clojure.core/+ i 1))
            reads (od/aget-reads (list 'raster.numeric/* f 2.0))]
        (is (= 1 (count reads)))
        (is (= 'A (:sym (first reads))))
        (is (= '(clojure.core/+ i 1) (:idx (first reads))))
        (is (= f (:form (first reads))) "the matched NODE is returned, so a rewriter need not
                                         re-implement matching"))))
  (testing "quoted data is data, not code"
    (is (empty? (od/aget-reads '(quote (aget A i))))))
  (testing "a non-read is not a read"
    (is (empty? (od/aget-reads '(clojure.core/+ i 1))))))

(deftest rewriting-preserves-spelling-positions-and-metadata
  (doseq [sp spellings]
    (testing (str sp " keeps its own shape")
      (let [f (read-form sp 'A 'OLD)
            r (od/rewrite-aget-index f 'NEW)]
        (is (= 'NEW (od/aget-index r)) "index replaced")
        (is (= 'A (od/aget-array-sym r)) "array untouched — a POSITIONAL rewrite would corrupt
                                          .invk into (aget impl idx)")
        (is (= (first f) (first r)) "spelling preserved")
        (is (od/aget-call? r) "still recognizable as a read after rewriting"))))
  (testing "metadata survives — :raster.op/original is what makes a devirtualized read
            recognizable at all, so dropping it blinds the very next matcher"
    (let [r (od/rewrite-aget-index (read-form :invk 'A 'OLD) 'NEW)]
      (is (= 'raster.arrays/aget (:raster.op/original (meta r))))))
  (testing "a type tag on an enclosing form is preserved through a rewrite"
    (let [expr (with-meta (list 'raster.numeric/* (read-form :core 'A 'OLD) 2.0)
                 {:raster.type/tag 'Double})]
      (is (= 'Double (:raster.type/tag (meta (od/rewrite-aget-indices expr '{A NEW})))))))
  (testing "only the named arrays are rewritten"
    (let [expr (list 'raster.numeric/* (read-form :core 'A 'IA) (read-form :core 'B 'IB))
          r (od/rewrite-aget-indices expr '{A NEW})]
      (is (= 'NEW (od/aget-index (nth r 1))))
      (is (= 'IB (od/aget-index (nth r 2))) "B untouched")))
  (testing "quote is opaque to the rewriter"
    (is (= '(quote (clojure.core/aget A OLD))
           (od/rewrite-aget-indices '(quote (clojure.core/aget A OLD)) '{A NEW}))))
  (testing "rewriting a non-read is a bug, not a silent no-op"
    (is (thrown? clojure.lang.ExceptionInfo (od/rewrite-aget-index '(clojure.core/+ i 1) 'NEW)))))

;; ── end to end: the only test that would have caught BOTH defects ───────────────────
(deftest a-real-walked-deftm-reaches-the-peak-leaf
  (testing "Every other test here builds its form by hand. This one takes the compiler's OWN output
            — `ensure-walked-body!` on a literal-dim deftm, which yields
            `(clojure.core/aget A (clojure.core/+ (clojure.core/* i (long 128)) l))` — and asserts
            it routes to the tensorized leaf.

            Before this change it reached :naive-segred. With ONLY the matcher fix it reached
            :regtiled with :non-canonical-orientation, because of the (long 128). It passes only
            when both the registry classification and the affine cast-unwrapping are in place.
            Hand-written bare-`aget` forms could never have shown either."
    (let [ns-form '(do
                     (clojure.core/require '[raster.arrays :as ra] 'raster.par 'raster.numeric)
                     (raster.core/deftm spelling-e2e-mm
                       [A :- (Array double), B :- (Array double)] :- (Array double)
                       (let [C (ra/alloc-like A 16384)]
                         (raster.par/contract C [[i 128] [j 128]] [[l 128]]
                           (raster.numeric/* (ra/aget A (clojure.core/+ (clojure.core/* i 128) l))
                                             (ra/aget B (clojure.core/+ (clojure.core/* l 128) j))))
                         C)))
          v (binding [*ns* (find-ns 'raster.compiler.contraction-spelling-test)] (eval ns-form))
          walked ((requiring-resolve 'raster.core/ensure-walked-body!) v)
          form (first (filter #(and (seq? %) (= 'raster.par/contract (first %)))
                              (tree-seq coll? seq walked)))]
      (is (some? form) "no par/contract survived walking")
      (testing "the walker really does emit the qualified spelling with a cast extent"
        (let [reads (od/aget-reads (nth form 4))]
          (is (= 2 (count reads)))
          (is (every? #(= 'clojure.core/aget (first (:form %))) reads))))
      (is (= :dpas (:strategy (cr/route-contraction form :dtype :half))))
      (is (= :regtiled (:strategy (cr/route-contraction form :dtype :double)))))))
