(ns raster.dl.gpu-ad-gemm-test
  "GPU-AD validation: the matmuls an AD-transformed train step emits — the forward projection
   (linear-nb → dgemm-nt!, :nt) and its two backward gradients (linear-dx → dgemm!, :nn;
   linear-dW → dgemm-tn!, :tn) — each lower through compile-gpu-program to a RESIDENT XMX GEMM
   graph and match the CPU BLAS reference. This is the empirical end of the framework claim
   that AD-transformed IR flows through the GPU pipeline: the backward kernels ARE ordinary
   GEMMs, so recognizing them on the resident path (incl. the :tn weight-gradient variant) is
   what makes gradient computation run on device.

   The full mse∘linear-nb TRAIN STEP does NOT yet fully lower (its loss reduction + the
   daxpy-diff-into! elementwise gradient have no resident kernel in the composed AD path);
   compile-gpu-program returns nil for it (clean staging-fn fallback). That current boundary is
   asserted below so progress flips it — see .internal/ad_gpu_residency.md.

   Level-Zero / Intel-XMX only (the fp16 DPAS GEMM kernel). Skips cleanly without a GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.pipeline :as pl]
            [raster.dl.array-ops :as ops]
            [raster.dl.nn :as nn]))

(def ^:private gpu-available?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- rnd [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5))))
    a))

(defn- rel-err [gpu cpu]
  (let [err (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) (seq gpu) (seq cpu)))
        mag (reduce max 1e-9 (map #(Math/abs (double %)) cpu))]
    (/ err mag)))

(defn- run-resident [f-var args]
  (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
        make-session (ns-resolve gpu 'make-session)
        bind-program! (ns-resolve gpu 'bind-program!)
        run-program! (ns-resolve gpu 'run-program!)
        close-session! (ns-resolve gpu 'close-session!)
        p (pl/compile-gpu-program f-var :ze:0 :dtype :float)
        s (make-session :ze:0)]
    (try
      (bind-program! s p args {})
      (let [r (run-program! s p args)]
        {:descriptor p :out (or (get r (:result-sym p)) (first (vals r)))})
      (finally (close-session! s)))))

(deftest gpu-ad-gemm-variants
  (if-not @gpu-available?
    (println "  [SKIP] gpu-ad-gemm: no Level Zero GPU")
    ;; XMX-friendly dims (multiples of 16); batch=16, in-f=32, out-f=16.
    (let [b 16 i 32 o 16
          ;; fp16 XMX GEMM: ~2.5e-4 relative error is the precision floor for these dims.
          tol 5e-3]
      (testing "forward linear-nb (:nt) on GPU matches CPU BLAS"
        (let [x (rnd (* b i) 1) W (rnd (* o i) 2)
              cpu (nn/linear-nb x W b i o)
              {:keys [descriptor out]} (run-resident #'nn/linear-nb [x W b i o])]
          (is (= [:nt] (mapv :variant (:steps descriptor)))
              "forward projection lowers to a single :nt resident GEMM step")
          (is (< (rel-err out cpu) tol)
              (str "forward relerr " (rel-err out cpu)))))
      (testing "backward linear-dx (:nn) on GPU matches CPU BLAS"
        (let [dy (rnd (* b o) 3) W (rnd (* o i) 4)
              cpu (nn/linear-dx dy W b i o)
              {:keys [descriptor out]} (run-resident #'nn/linear-dx [dy W b i o])]
          (is (= [:nn] (mapv :variant (:steps descriptor))))
          (is (< (rel-err out cpu) tol)
              (str "linear-dx relerr " (rel-err out cpu)))))
      (testing "backward linear-dW (:tn, weight gradient) on GPU matches CPU BLAS"
        (let [dy (rnd (* b o) 5) x (rnd (* b i) 6)
              cpu (nn/linear-dW dy x b i o)
              {:keys [descriptor out]} (run-resident #'nn/linear-dW [dy x b i o])]
          (is (= [:tn] (mapv :variant (:steps descriptor)))
              "weight-gradient lowers to a :tn resident GEMM step (transpose-A path)")
          (is (< (rel-err out cpu) tol)
              (str "linear-dW relerr " (rel-err out cpu))))))))

(deftest gpu-attention-layout-ops-resident
  ;; The four dense-attention LAYOUT ops (pack/unpack heads + broadcast/sum KV heads)
  ;; used inside the flat gqa-causal-mha are strided permutation/broadcast/segment-reduce
  ;; copies re-expressed over par/map-void!, so each lowers to a SINGLE resident :map-void
  ;; kernel on device (no raw loop / host-scalar fallback) and matches the CPU reference.
  (if-not @gpu-available?
    (println "  [SKIP] gpu-attention-layout: no Level Zero GPU")
    (let [tol 1e-5]
      (testing "pack-heads lowers to a resident :map-void kernel and matches CPU"
        (let [sl 6 nh 4 hd 8 x (rnd (* sl nh hd) 21)
              cpu (ops/pack-heads x sl nh hd)
              {:keys [descriptor out]} (run-resident #'ops/pack-heads [x sl nh hd])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "pack-heads relerr " (rel-err out cpu)))))
      (testing "unpack-heads lowers to a resident :map-void kernel and matches CPU"
        (let [sl 6 nh 4 hd 8 x (rnd (* sl nh hd) 22)
              cpu (ops/unpack-heads x sl nh hd)
              {:keys [descriptor out]} (run-resident #'ops/unpack-heads [x sl nh hd])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "unpack-heads relerr " (rel-err out cpu)))))
      (testing "broadcast-kv-heads (MQA fan-out) lowers resident and matches CPU"
        (let [nkv 2 grp 4 slab 48 src (rnd (* nkv slab) 23)
              cpu (ops/broadcast-kv-heads src nkv grp slab)
              {:keys [descriptor out]} (run-resident #'ops/broadcast-kv-heads [src nkv grp slab])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "broadcast-kv relerr " (rel-err out cpu)))))
      (testing "sum-kv-heads (fan-in segment reduce) lowers resident and matches CPU"
        (let [nkv 2 grp 4 slab 48 big (rnd (* nkv grp slab) 24)
              cpu (ops/sum-kv-heads big nkv grp slab)
              {:keys [descriptor out]} (run-resident #'ops/sum-kv-heads [big nkv grp slab])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "sum-kv relerr " (rel-err out cpu))))))))

(deftest gpu-ad-full-train-step-residency-boundary
  ;; This test does NOT need a GPU — it pins the CURRENT residency boundary at the IR level.
  ;; The full mse∘linear-nb float train step (value+grad + SGD) is AD-transformed and its 3
  ;; matmuls DO lower to resident GEMM steps, but its loss reduction (mse-loss) and the
  ;; daxpy-diff-into! elementwise gradient have no resident kernel in this composed path, so
  ;; compile-gpu-program returns nil (→ compile-aot :target-device uses the staging fn). When
  ;; the saved-activation / elementwise-loss lowering lands (see .internal/ad_gpu_residency.md),
  ;; this returns a descriptor and the assertion below must flip to a full end-to-end run.
  (require 'raster.core 'raster.dl.loss 'raster.dl.optim 'raster.ad.reverse 'raster.arrays)
  (let [loss (eval '(raster.core/deftm gpu-ad-probe-loss
                      [W :- (Array float) x :- (Array float) tgt :- (Array float)
                       batch :- Long in-f :- Long out-f :- Long] :- Double
                      (let [pred (raster.dl.nn/linear-nb x W batch in-f out-f)]
                        (raster.dl.loss/mse-loss pred tgt (clojure.core/* batch out-f)))))
        train (eval '(raster.core/deftm gpu-ad-probe-train
                       [W :- (Array float) x :- (Array float) tgt :- (Array float)
                        batch :- Long in-f :- Long out-f :- Long lr :- Double] :- Double
                       (let [vg ((raster.ad.reverse/value+grad #'gpu-ad-probe-loss)
                                 W x tgt batch in-f out-f)
                             loss (clojure.core/nth vg 0)
                             dW (clojure.core/nth vg 1)]
                         (raster.dl.optim/sgd-step! W dW (raster.arrays/alength W) lr)
                         loss)))]
    (testing "full AD train step is not yet a fully-resident program (documents the gap)"
      ;; :on-non-resident :nil = probe the boundary without throwing (default is :throw now)
      (is (nil? (pl/compile-gpu-program train :ze:0 :dtype :float :on-non-resident :nil))
          "mse reduction + daxpy elementwise gradient have no resident kernel yet"))))
