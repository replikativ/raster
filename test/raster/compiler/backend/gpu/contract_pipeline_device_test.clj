(ns raster.compiler.backend.gpu.contract-pipeline-device-test
  "W1 integration (the pipeline wire): a (raster.par/contract …) form flows through the ACTUAL
   opencl-pass → routes via the DPAS legality gate → emits invoke-registered-contraction! →
   registers + launches the chosen kernel on device → result matches CPU. End-to-end proof
   that the SOAC contraction path compiles automatically (not just as a standalone emitter).
   Gated on a real GPU."
  (:require [clojure.string]
            [clojure.test :refer [deftest is testing]]
            [raster.arrays :as ra]
            [raster.compiler.backend.gpu.opencl-pass :as ocl]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.ze-runtime :as ze]))

(def ^:private gpu?
  (delay (try (boolean (seq (ze/query-devices))) (catch Throwable _ false))))

(defn- cpuref [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m] (dotimes [j n]
      (aset C (+ (* i n) j)
            (loop [l 0 a 0.0] (if (< l k)
                                (recur (inc l) (+ a (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))))
                                a)))))
    C))

(defn- matmul-form [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(deftm resident-contract-probe
  [A :- (Array float) B :- (Array float)] :- (Array float)
  (let [C (ra/alloc-like A 64)]
    (raster.par/contract C [[i 8] [j 8]] [[l 8]]
      (clojure.core/*
       (ra/aget A (clojure.core/+ (clojure.core/* i 8) l))
       (ra/aget B (clojure.core/+ (clojure.core/* l 8) j))))
    C))

(deftest par-contract-runs-as-a-resident-compiled-step
  (if-not @gpu?
    (println "[skip] resident contract pipeline: no GPU device available")
    (let [descriptor (pipeline/compile-gpu-program #'resident-contract-probe
                                                   :ze:0 :dtype :float)
          A (float-array (map float (range 64)))
          B (float-array (repeat 64 (float 1.0)))
          session (gpu/make-session :ze:0)]
      (try
        (let [program (fixture/instantiate! session descriptor [A B])
              result (get (fixture/run! program [A B]) 'C)]
          ;; The typed contraction route emits one scheduled-executable step whose dispatch is
          ;; the typed contraction dispatch; no convention-specific contraction marker remains.
          (is (= [:executable] (mapv :convention (:steps descriptor))))
          (is (every? #(clojure.string/starts-with? (str (:dispatch-id %))
                                                    "raster_typed_contraction_dispatch")
                      (:steps descriptor)))
          (is (= (vec (mapcat #(repeat 8 (float %))
                              (map (fn [row] (reduce + (range (* row 8) (* (inc row) 8))))
                                   (range 8))))
                 (vec result))))
        (finally
          (gpu/close-session! session))))))

(defn- compile+run
  "Run a matmul par/contract form through opencl-pass at `dt`, register the emitted kernel,
   compile the marker to a fn, launch on device, return {:head :max-abs-diff}.
   Uses :compile-spirv? false → the kernel is compiled at RUNTIME by the Level Zero driver
   (via register-kernel! :source → ensure-kernel-loaded!). NB: the offline ocloc path
   (:compile-spirv? true) cannot compile the DPAS intel_sub_group_2d_block_* intrinsics —
   a pre-existing ocloc/AOT limitation for the tensorized leaf, tracked separately; the
   driver path used here is the runtime path production takes for these kernels."
  [dt m k n]
  (let [{:keys [form kernels]} (ocl/opencl-pass (matmul-form m n k) :dtype dt :compile-spirv? false)
        _ (doseq [kr kernels] (ze/register-kernel! (:kernel-name kr) kr))
        f (eval (list 'fn '[A B C] form))
        A (double-array (map #(* 0.1 (- (double (mod % 7)) 3.0)) (range (* m k))))
        B (double-array (map #(* 0.1 (- (double (mod % 5)) 2.0)) (range (* k n))))
        C (double-array (* m n))]
    (f A B C)
    {:head (first form)
     :max-abs-diff (apply max (map (fn [a b] (Math/abs (- (double a) (double b)))) C (cpuref A B m k n)))}))

(deftest par-contract-compiles-through-opencl-pass
  (testing "opencl-pass recognizes par/contract and emits the invoke marker (device-free)"
    (let [{:keys [form stats]} (ocl/opencl-pass (matmul-form 128 128 128) :dtype :half :compile-spirv? false)]
      (is (= 'raster.gpu.ze-runtime/invoke-registered-contraction! (first form)))
      (is (= 1 (:ze-contracts stats)))))
  (if-not @gpu?
    (println "[skip] contract-pipeline-device: no GPU device available")
    (do
      (testing "f16 matmul routes to DPAS and matches CPU on device"
        (let [{:keys [head max-abs-diff]} (compile+run :half 128 128 128)]
          (is (= 'raster.gpu.ze-runtime/invoke-registered-contraction! head))
          (is (< max-abs-diff 2.0e-2) (str "dpas maxdiff " max-abs-diff))))
      (testing "f64 routes to register-tiled fallback and matches CPU exactly"
        (let [{:keys [max-abs-diff]} (compile+run :double 96 64 96)]
          (is (< max-abs-diff 1.0e-9) (str "regtiled-f64 maxdiff " max-abs-diff))))
      (testing "pitch-unaligned f16 (N=70) falls back to register-tiled and matches CPU"
        (let [{:keys [max-abs-diff]} (compile+run :half 128 96 70)]
          (is (< max-abs-diff 2.0e-2) (str "regtiled-f16 fallback maxdiff " max-abs-diff)))))))

;; ── Phase 0: EVERY routed strategy survives the pipeline (descriptor passed through intact) ──
;; Before the descriptor fix, only :dpas (3 scalar-args) and :regtiled (0) worked: the invoke
;; reconstructed scalar-args from a `case` with no default and destructured [m n k] from :dims,
;; so :segmap / general SegRed (1 scalar-arg, :dims [nseg]) crashed, and quant's f32 output was
;; staged at int8 size (4× undersized).
(defn- run-form [form dt args-map out-len]
  (let [{:keys [form kernels]} (ocl/opencl-pass form :dtype dt :compile-spirv? false)
        _ (doseq [kr kernels] (ze/register-kernel! (:kernel-name kr) kr))
        syms (vec (keys args-map))
        f (eval (list 'fn (conj syms 'C) form))]
    (apply f (conj (mapv args-map syms) (double-array out-len)))))

(deftest phase0-all-strategies-survive-the-pipeline
  (if-not @gpu?
    (println "[skip] phase0-strategies: no GPU")
    (do
      (testing ":segmap (0-contract outer product) through opencl-pass → device"
        (let [a (double-array [1 2 3 4]) b (double-array [10 20 30])
              form (list 'raster.par/contract 'C [['i 4] ['j 3]] []
                         (list '* (list 'raster.arrays/aget 'a 'i) (list 'raster.arrays/aget 'b 'j)))
              out (run-form form :double {'a a 'b b} 12)
              ref (vec (for [i (range 4) j (range 3)] (* (aget a i) (aget b j))))]
          (is (= (vec out) ref))))
      (testing ":portable-segred (3 free axes, batch matmul) through opencl-pass → device"
        (let [B 2 M 3 N 2 K 4
              A (double-array (map #(* 0.1 (double %)) (range (* B M K))))
              Bd (double-array (map #(* 0.2 (double %)) (range (* B K N))))
              form (list 'raster.par/contract 'C [['bb B] ['i M] ['j N]] [['l K]]
                         (list '* (list 'raster.arrays/aget 'A (list '+ (list '* (list '+ (list '* 'bb M) 'i) K) 'l))
                               (list 'raster.arrays/aget 'Bm (list '+ (list '* (list '+ (list '* 'bb K) 'l) N) 'j))))
              out (run-form form :double {'A A 'Bm Bd} (* B M N))
              ref (vec (for [bb (range B) i (range M) j (range N)]
                         (reduce + (for [l (range K)] (* (aget A (+ (* (+ (* bb M) i) K) l))
                                                         (aget Bd (+ (* (+ (* bb K) l) N) j)))))))]
          (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-9) out ref))))))))

(deftest phase0-symbolic-bounds-route-without-crashing
  ;; device-free: symbolic axis bounds previously threw ClassCastException in route-contraction
  ;; before any branch was taken (which made raster.linalg.contract/contract-mm un-routable).
  (testing "symbolic free-axis bounds produce a descriptor with symbolic out-elems/grid"
    (let [r ((requiring-resolve 'raster.compiler.passes.parallel.contract-route/route-contraction)
             '(raster.par/contract C [[i m] [j n]] [[l k]]
                (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))
             :dtype :double)
          artifact (:artifact r)
          scalar-values {"k" 7 "m" 20 "n" 20 "_nseg" 400}
          runtime-arguments
          (mapv (fn [slot]
                  (if (= :scalar (:kind slot))
                    {:type (:kernel-dtype slot)
                     :value (get scalar-values (:c-name slot))}
                    (Object.)))
                (:abi artifact))
          call (kcall/make artifact runtime-arguments)]
      (is (some? (:out-elems r)))
      (is (not (number? (:out-elems r))))            ; carried symbolically
      (is (contains? #{:regtiled :portable-segred} (:strategy r)))
      ;; STRENGTHENED: the original assertion only checked that routing did not CRASH, which let a
      ;; real bug through — the descriptor supplied 1 scalar arg where the kernel declares 4
      ;; (int k, int m, int n, int _nseg), so a caller would have mis-bound at launch. The
      ;; descriptor validator caught it; assert usability here, not merely non-crashing.
      (is (= 4 (count (:scalar-args r)))
          "the symbolic axis bounds must be bound as int params, before the trailing count")
      (is (= '[k m n] (mapv :value (butlast (:scalar-args r))))
          "…in the kernel's declared (name-sorted) order")
      (is (= (:out-elems r) (:value (last (:scalar-args r)))))
      (is (= [256] (get-in call [:geometry :workgroup-size])))
      (is (= [2] (get-in call [:geometry :group-count]))
          "the artifact resolves its symbolic ceil-div grid from ordered ABI values"))))
