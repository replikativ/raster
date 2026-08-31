(ns raster.compiler.passes.parallel.epilogue-operand-fusion-test
  "Completes the refusal left by the contract→map epilogue fusion: a consumer map that reads OTHER
   arrays (a per-column bias, a per-row scale, an elementwise residual) now fuses too.

   The consumer is a 1-D map over the FLAT output (`t = i·N + j`), so each operand's index must be
   re-expressed in the contraction's FREE axes before it can be spliced into the store slot — that
   is axis-map/flat-index->map. An index that is not a recognized broadcast shape is REFUSED rather
   than guessed: a wrongly-indexed operand is a silent miscompile."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.passes.parallel.par-fusion :as pf]
            [raster.compiler.passes.parallel.contract-route :as route]))

(def ^:private M 128)
(def ^:private N 256)
(def ^:private K 64)

(defn- mm []
  (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
        (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
              (list 'aget 'B (list '+ (list '* 'l N) 'j)))))

(defn- fuse [mbody]
  (let [b ['C (mm) 'out (list 'raster.par/map! 'out 't (* M N) nil mbody)]]
    (when-let [r (pf/fuse-contract-map b [])]
      (let [fused (second (:bindings r))]
        {:form fused :epilogue (:epilogue (apply hash-map (drop 5 fused)))}))))

(defn- store-line [fused]
  (let [r (route/route-contraction fused :dtype :half)]
    {:store (some #(when (str/includes? % "C[row*N+col] =") (str/trim %))
                  (str/split-lines (:source r)))
     :epi-ops (:epilogue-operands r)
     :declares (fn [sym] (str/includes? (:source r) (str "restrict " sym)))}))

;; ── the flat-index decomposition itself ──────────────────────────────────────────────
(deftest flat-index-decomposition
  (let [fa [['i M] ['j N]]]
    (testing "t → the full row-major output layout (elementwise operand, e.g. a residual)"
      (is (= (am/of-axes fa) (am/flat-index->map 't 't fa))))
    (testing "(mod t N) → the innermost axis only (per-column broadcast, e.g. a bias)"
      (is (= (am/of-axes [['j N]]) (am/flat-index->map (list 'mod 't N) 't fa)))
      (is (= 'j (am/index-expr (am/flat-index->map (list 'mod 't N) 't fa)))))
    (testing "(quot t N) → the outer axes only (per-row broadcast)"
      (is (= (am/of-axes [['i M]]) (am/flat-index->map (list 'clojure.core/quot 't N) 't fa))))
    (testing "a non-matching stride is NOT recognized (must refuse, never guess)"
      (is (nil? (am/flat-index->map (list 'mod 't 7) 't fa)))
      (is (nil? (am/flat-index->map (list 'raster.numeric/* 't 3) 't fa))))))

;; ── fusion of operand-carrying epilogues ─────────────────────────────────────────────
(deftest operand-carrying-epilogues-fuse-with-correct-indices
  (testing "per-column bias: operand map is [j], emitted index is bias[col]"
    (let [f (fuse (list 'raster.numeric/+ (list 'aget 'C 't) (list 'aget 'bias (list 'mod 't N))))
          {:keys [store epi-ops declares]} (store-line (:form f))]
      (is (= '[bias] (mapv :sym (:operands (:epilogue f)))))
      (is (= (am/of-axes [['j N]]) (:map (first (:operands (:epilogue f))))))
      (is (re-find #"bias\[[^]]*col" store))
      (is (= '[bias] epi-ops) "the descriptor must surface the extra kernel arg")
      (is (declares "bias") "…and the signature must declare it")))
  (testing "elementwise residual: full map, emitted index is R[row*N+col]"
    (let [f (fuse (list 'raster.numeric/+ (list 'aget 'C 't) (list 'aget 'R 't)))
          {:keys [store epi-ops]} (store-line (:form f))]
      (is (= (am/of-axes [['i M] ['j N]]) (:map (first (:operands (:epilogue f))))))
      (is (re-find (re-pattern (str "R\\[[^]]*row[^]]*" N "[^]]*col")) store))
      (is (= '[R] epi-ops))))
  (testing "per-row scale: operand map is [i], emitted index is rs[row]"
    (let [f (fuse (list 'raster.numeric/* (list 'aget 'C 't) (list 'aget 'rs (list 'quot 't N))))
          {:keys [store]} (store-line (:form f))]
      (is (re-find #"rs\[[^]]*row" store))))
  (testing "an activation-only body still fuses, with no operands"
    (let [f (fuse (list 'raster.math/exp (list 'aget 'C 't)))
          {:keys [epi-ops]} (store-line (:form f))]
      (is (empty? (:operands (:epilogue f))))
      (is (empty? epi-ops)))))

(deftest refusals-never-guess-an-operand-index
  (testing "an unrecognized stride refuses to fuse"
    (is (nil? (fuse (list 'raster.numeric/+ (list 'aget 'C 't)
                          (list 'aget 'bias (list 'mod 't 7)))))))
  (testing "an operand read at TWO different indices refuses"
    (is (nil? (fuse (list 'raster.numeric/+ (list 'aget 'C 't)
                          (list 'raster.numeric/+ (list 'aget 'bias (list 'mod 't N))
                                (list 'aget 'bias (list 'quot 't N)))))))))

;; ── device: the fused bias+activation equals the two-pass reference ───────────────────
(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(deftest fused-operand-epilogue-matches-two-pass-on-device
  (if-not @gpu?
    (println "[skip] fused-operand-epilogue-device: no GPU")
    (let [ze (find-ns 'raster.gpu.ze-runtime)
          reg! (ns-resolve ze 'register-kernel!) ens! (ns-resolve ze 'ensure-kernel-loaded!)
          mkbuf (ns-resolve ze 'make-buffer) halfs (ns-resolve ze 'buffer-of-floats-as-half)
          a->b (ns-resolve ze 'array->buffer!) l2d (ns-resolve ze 'launch-2d!)
          b->d (ns-resolve ze 'buffer->double-array)
          Ad (float-array (map #(float (* 0.05 (- (mod % 7) 3))) (range (* M K))))
          Bd (float-array (map #(float (* 0.05 (- (mod % 5) 2))) (range (* K N))))
          biasd (float-array (map #(float (* 0.02 (- (mod % 9) 4))) (range N)))
          run (fn [mbody extra]
                (let [fused (:form (fuse mbody))
                      r (route/route-contraction fused :dtype :half)
                      out (mkbuf (* M N) :half)]
                  (reg! (:kernel-name r) {:source (:source r) :dtype :half})
                  (let [{:keys [kernel-handle]} (ens! (:kernel-name r))]
                    (l2d kernel-handle (:wg r) (:grid r)
                         (into [(:segment (halfs Ad)) (:segment (halfs Bd)) (:segment out)
                                {:type :int :value M} {:type :int :value N} {:type :int :value K}]
                               extra))
                    (vec (b->d out)))))
          plain (run (list 'aget 'C 't) [])                       ; identity epilogue = plain GEMM
          bias-buf (a->b (mkbuf N :float) biasd)
          fused (run (list 'raster.numeric/+ (list 'aget 'C 't) (list 'aget 'bias (list 'mod 't N)))
                     [(:segment bias-buf)])]
      (testing "fused bias == plain GEMM followed by a separate per-column bias pass"
        (let [ref (vec (map-indexed (fn [idx v] (+ (double v) (aget biasd (mod idx N)))) plain))]
          (is (= (count ref) (count fused)))
          (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 3.0e-2) fused ref))
              (str "fused " (take 4 fused) " vs two-pass " (take 4 ref))))))))

;; NOTE (deferred to the epilogue-gate work): an operand or scalar sharing a name with a free axis
;; is an ILL-FORMED spec, not something to preserve. Probed: axis `s` + scalar `s` emits `acc * row`
;; — valid C, wrong number, silent — and no substitution mechanism can resolve it, because the spec
;; keeps one symbol namespace for three C-level roles (operand arrays and scalars become kernel
;; params; free axes become store-slot locals). The fix is `:operand-shadows-free-axis` in
;; epilogue-legal?, plus a matching DECLINE in fuse-contract-map so fusion backs off to two kernels
;; rather than the emitter throwing on a program that compiles today.
