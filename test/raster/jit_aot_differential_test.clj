(ns raster.jit-aot-differential-test
  "D1 — JIT/AOT differential harness.

  Compiles a corpus of deftms two ways — lazy-JIT (call the deftm directly) and
  compile-aot — on identical inputs and asserts the results agree within
  tolerance. This is the safety net for the compiler-consolidation work: any pass
  that makes AOT (or JIT) diverge on the same body surfaces here rather than as
  wrong numbers in a downstream model. It also exercises the literal-narrowing
  path (element-typed literals like `2.0`, `0.0`) on both paths.

  The harness does NOT check against a reference oracle — only JIT-vs-AOT
  agreement — so corpus entries may use arbitrary consistent shapes."
  (:refer-clojure :exclude [+ - * / aget aset alength])
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.core :refer [deftm]]
            [raster.numeric :refer [+ - * /]]
            [raster.arrays :refer [aget aset alength alloc-like]]
            [raster.dl.nn :as nn]
            [raster.nn :as snn]
            [raster.dl.optim :as optim]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pl]))

;; ---------------------------------------------------------------------------
;; test-local deftms — cover patterns beyond library ops
;; ---------------------------------------------------------------------------

;; elementwise with an element-typed literal (2.0 must narrow to T)
(deftm d-scale-add (All [T] [x :- (Array T) y :- (Array T) n :- Long] :- (Array T)
                        (let [out (alloc-like x n)]
                          (dotimes [i n]
                            (aset out i (+ (* 2.0 (aget x i)) (aget y i))))
                          out)))

;; loop reduction with an element-typed accumulator init (0.0 must narrow to T)
(deftm d-sumsq (All [T] [x :- (Array T) n :- Long] :- T
                    (loop [i 0 s 0.0]
                      (if (clojure.core/< i n)
                        (recur (clojure.core/inc i) (+ s (* (aget x i) (aget x i))))
                        s))))

;; composed body — exercises buffer-fuse aliasing across sub-deftm calls
;; (the compile_aot_composition_bug class). eps forced double per the current
;; escape-hatch convention.
(deftm d-ffn-block (All [T] [x :- (Array T) w1 :- (Array T) b1 :- (Array T)
                             w2 :- (Array T) b2 :- (Array T) g :- (Array T) be :- (Array T)
                             rows :- Long din :- Long dh :- Long] :- (Array T)
                        (let [up   (nn/linear x w1 b1 rows din dh)
                              up2  (nn/gelu up (clojure.core/* rows dh))
                              down (nn/linear up2 w2 b2 rows dh din)]
                          (nn/skip-layer-norm down x g be rows din (clojure.core/double 1e-5)))))

;; D2 regression guard — REBINDING a let var to an ALLOCATING-op result
;; (x = (gelu x)) was the compile_aot_composition_bug: buffer analysis conflated
;; the two SSA versions of the rebound name → downstream read the wrong buffer,
;; corrupting at scale (fine at tiny dims). Fixed by util/uniquify-rebindings.
;; Kept at real-ish dims so the guard actually exercises the size-dependent path.
(deftm d-rebind-block (All [T] [x :- (Array T) w1 :- (Array T) b1 :- (Array T)
                                w2 :- (Array T) b2 :- (Array T)
                                rows :- Long din :- Long dh :- Long] :- (Array T)
                           (let [x (nn/linear x w1 b1 rows din dh)
                                 x (nn/gelu x (clojure.core/* rows dh))
                                 x (nn/linear x w2 b2 rows dh din)]
                             x)))

;; AD-differential axis (task #78): runtime value+grad vs compiled value+grad
;; must agree. The forward-only corpus above CANNOT catch backward miscompiles.
;; cross-entropy's backward is a broadcast `(* (- (/ t (max eps p))) dy)`; a stale
;; :raster.op/original from `pe` let `normalize` rewrite `- -> *` and delete the
;; negation on the compile path only (f64 literal-1.0 seed) — an exact gradient
;; sign flip. A gradient-returning deftm routes straight through run-one's
;; JIT-vs-AOT comparison, so the drop shows up as runtime≠compiled here.
(deftm d-ce-loss (All [T] [logits :- (Array T) y :- (Array T)] :- T
                      (snn/cross-entropy (snn/softmax logits) y)))

;; gradient w.r.t. logits (index 1 of [loss d_logits d_y])
(deftm d-ce-grad (All [T] [logits :- (Array T) y :- (Array T)] :- (Array T)
                      (nth ((rev/value+grad (var d-ce-loss)) logits y) 1)))

;; value+grad + in-place SGD — the train-step shape that surfaced #78 at scale
(deftm d-ce-step (All [T] [w :- (Array T) y :- (Array T) lr :- Double] :- (Array T)
                      (let [vg ((rev/value+grad (var d-ce-loss)) w y)]
                        (optim/sgd-step! w (nth vg 1) (alength w) lr)
                        w)))

;; ---------------------------------------------------------------------------
;; harness
;; ---------------------------------------------------------------------------

(defn- mk [dtype xs]
  (if (= dtype :float) (float-array (map float xs)) (double-array (map double xs))))

(defn- ramp [dtype n]
  (mk dtype (map #(clojure.core/* 0.1 (clojure.core/- (clojure.core/mod % 7) 3)) (range n))))

(defn- result-vec [r]
  (cond (instance? (Class/forName "[F") r) (vec r)
        (instance? (Class/forName "[D") r) (vec r)
        :else [(double r)]))

(defn- maxdiff [a b]
  (if (not= (count a) (count b))
    Double/POSITIVE_INFINITY
    (reduce max 0.0 (map (fn [x y] (Math/abs (clojure.core/- (double x) (double y)))) a b))))

(defn- run-one [{:keys [var make]} dtype]
  (try
    (let [jit (result-vec (apply @var (make dtype)))
          aotf (pl/compile-aot var :dtype dtype)
          aot (result-vec (apply aotf (make dtype)))]
      {:ok true :maxdiff (maxdiff jit aot) :n (count jit)})
    (catch Throwable e
      {:ok false :error (str (.getName (class e)) ": "
                             (first (clojure.string/split-lines (str (.getMessage e)))))})))

;; corpus: :var + :make (dtype -> fresh arg vector). Shapes arbitrary-but-consistent.
(def corpus
  [{:name "gelu"            :var #'nn/gelu
    :make (fn [dt] [(ramp dt 32) 32])}
   {:name "layer-norm"      :var #'nn/layer-norm
    :make (fn [dt] [(ramp dt 32) (ramp dt 8) (ramp dt 8) 4 8 1.0e-5])}
   {:name "skip-layer-norm" :var #'nn/skip-layer-norm
    :make (fn [dt] [(ramp dt 32) (ramp dt 32) (ramp dt 8) (ramp dt 8) 4 8 1.0e-5])}
   {:name "linear"          :var #'nn/linear
    :make (fn [dt] [(ramp dt 32) (ramp dt 128) (ramp dt 16) 4 8 16])}
   {:name "d-scale-add"     :var #'d-scale-add
    :make (fn [dt] [(ramp dt 24) (ramp dt 24) 24])}
   {:name "d-sumsq"         :var #'d-sumsq
    :make (fn [dt] [(ramp dt 24) 24])}
   ;; composed last so sub-op float overloads are warm
   {:name "d-ffn-block"     :var #'d-ffn-block
    :make (fn [dt] [(ramp dt 32) (ramp dt 128) (ramp dt 16) (ramp dt 128) (ramp dt 32)
                    (ramp dt 32) (ramp dt 32) 4 8 16])}
   ;; D2 guard — allocating-op rebind at real-ish dims (rows=32, din=64, dh=256)
   {:name "d-rebind-block"  :var #'d-rebind-block
    :make (fn [dt] [(ramp dt (* 32 64)) (ramp dt (* 64 256)) (ramp dt 256)
                    (ramp dt (* 256 64)) (ramp dt 64) 32 64 256])}
   ;; AD-differential (#78): backward of a template op whose backward is itself a
   ;; deftm (cross-entropy → cross-entropy-backward-dp broadcast). runtime keeps
   ;; it a call; compile inlines it → the pe/normalize sign-drop showed up here.
   ;; :double-only until R4/#77 (tag-safe devirt) fixes the :float [F→[D in the
   ;; backward — then flip :dtypes to enable :float and make this R4's gate.
   {:name "d-ce-grad"       :var #'d-ce-grad :dtypes [:double]
    :make (fn [dt] [(ramp dt 10) (ramp dt 10)])}
   {:name "d-ce-step"       :var #'d-ce-step :dtypes [:double]
    :make (fn [dt] [(ramp dt 10) (ramp dt 10) 0.1])}])

(def tol 1.0e-4)  ;; FP reassociation between serial-JIT and SIMD/fused-AOT

(deftest jit-aot-differential
  ;; An entry may restrict which dtypes it runs via :dtypes (default both). The
  ;; AD-differential entries are :double-only until R4 (tag-safe devirtualization,
  ;; task #77) lands — at :float the backward devirtualizes to a doubles impl and
  ;; throws [F→[D (the same class as LeNet-f32 [J→[S). Flip them to both dtypes
  ;; when R4 lands: this test then becomes R4's regression gate.
  (doseq [entry corpus
          dt (:dtypes entry [:double :float])]
    (testing (str (:name entry) " @ " dt)
      (let [r (run-one entry dt)]
        (if (:ok r)
          (is (<= (:maxdiff r) tol)
              (str (:name entry) " @ " dt " diverged: maxdiff=" (:maxdiff r)))
          (is false (str (:name entry) " @ " dt " errored: " (:error r))))))))
