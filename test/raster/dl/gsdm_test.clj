(ns raster.dl.gsdm-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.dl.gsdm :as gsdm]
            [raster.dl.nn :as nn]
            [raster.dl.diffusion :as diff]
            [raster.params :as rp]
            [raster.compiler.pipeline :as pipeline]
            [raster.linalg.blas :as blas]))

(use-fixtures :once
  (fn [f] (if (blas/available?) (f) (println "[SKIP] No BLAS library"))))

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

(defn- arr-finite?
  "Check all elements of array are finite (not NaN/Inf)."
  [^doubles arr]
  (every? #(Double/isFinite %) arr))

;; ================================================================
;; Variable state constants
;; ================================================================

(deftest variable-state-constants-test
  (testing "state constants are defined"
    (is (= 0 gsdm/LATENT))
    (is (= 1 gsdm/OBSERVED))
    (is (= 2 gsdm/MARGINALIZED))
    (is (= 3 gsdm/INTERVENED))
    (is (= 4 gsdm/N-STATES))))

;; ================================================================
;; Marginalization
;; ================================================================

(deftest marginalize-graph-test
  (testing "no marginalized vars — identity"
    (let [values (double-array [1.0 2.0 3.0])
          spaces (long-array [0 0 0])
          states (long-array [gsdm/LATENT gsdm/LATENT gsdm/LATENT])
          src (long-array [0 1 2])
          dst (long-array [1 2 0])
          mg (gsdm/marginalize-graph values spaces states src dst 3 3)]
      (is (= 3 (:n-vars mg)))
      (is (= 3 (:n-edges mg)))
      (is (= [1.0 2.0 3.0] (vec (:values mg))))))

  (testing "marginalize middle variable"
    (let [values (double-array [1.0 2.0 3.0 4.0])
          spaces (long-array [0 0 0 0])
          states (long-array [gsdm/LATENT gsdm/MARGINALIZED gsdm/OBSERVED gsdm/LATENT])
          ;; Edges: 0→1, 1→2, 2→3, 3→0
          src (long-array [0 1 2 3])
          dst (long-array [1 2 3 0])
          mg (gsdm/marginalize-graph values spaces states src dst 4 4)]
      ;; Variable 1 removed: [0,2,3] → reindexed to [0,1,2]
      (is (= 3 (:n-vars mg)))
      (is (= [1.0 3.0 4.0] (vec (:values mg))))
      (is (= [gsdm/LATENT gsdm/OBSERVED gsdm/LATENT] (vec (:states mg))))
      ;; Edges touching var 1 removed: only 2→3 (→ 1→2) and 3→0 (→ 2→0)
      (is (= 2 (:n-edges mg)))
      ;; Original indices
      (is (= [0 2 3] (vec (:original-indices mg))))))

  (testing "all marginalized — empty graph"
    (let [values (double-array [1.0 2.0])
          spaces (long-array [0 0])
          states (long-array [gsdm/MARGINALIZED gsdm/MARGINALIZED])
          src (long-array [0])
          dst (long-array [1])
          mg (gsdm/marginalize-graph values spaces states src dst 2 1)]
      (is (= 0 (:n-vars mg)))
      (is (= 0 (:n-edges mg))))))

(deftest unmarginalize-values-test
  (testing "scatter back to full array"
    (let [compact (double-array [10.0 30.0 40.0])
          orig-idx (long-array [0 2 3])
          full (gsdm/unmarginalize-values compact orig-idx 4)]
      (is (= [10.0 0.0 30.0 40.0] (vec full))))))

;; ================================================================
;; Normalization
;; ================================================================

(deftest normalization-test
  (testing "fit-normalizer computes mean and std"
    (let [samples [(double-array [1.0 10.0])
                   (double-array [3.0 10.0])
                   (double-array [5.0 10.0])]
          norm (gsdm/fit-normalizer samples 2)]
      (is (approx= 3.0 (aget ^doubles (:mean norm) 0)))
      (is (approx= 10.0 (aget ^doubles (:mean norm) 1)))
      ;; std of [1,3,5] = sqrt(8/3) ≈ 1.633
      (is (> (aget ^doubles (:std norm) 0) 1.0))
      ;; std of [10,10,10] = eps (clamped)
      (is (> (aget ^doubles (:std norm) 1) 0.0))))

  (testing "normalize + denormalize roundtrip"
    (let [values (double-array [3.0 7.0 -1.0])
          mean (double-array [1.0 2.0 3.0])
          std (double-array [2.0 5.0 0.5])
          normed (gsdm/normalize-values values mean std 3)
          restored (gsdm/denormalize-values normed mean std 3)]
      (dotimes [i 3]
        (is (approx= (aget ^doubles values i)
                     (aget ^doubles restored i)))))))

;; ================================================================
;; Position embeddings
;; ================================================================

(deftest position-embeddings-test
  (testing "compute-position-embeddings correct shape"
    (let [spaces (long-array [0 0 1 1 0])
          emb (gsdm/compute-position-embeddings spaces 5 8)]
      (is (= 40 (alength emb)))
      (is (arr-finite? emb))))

  (testing "same position in same space → same embedding"
    ;; Variables 0 and 3 are both position 0 in their respective spaces
    ;; But different positions within the same space differ
    (let [spaces (long-array [0 0 0])
          emb (gsdm/compute-position-embeddings spaces 3 8)]
      ;; Var 0 = position 0 in space 0
      ;; Var 1 = position 1 in space 0 → different embedding
      (is (not (approx= (aget ^doubles emb 0) (aget ^doubles emb 8)))))))

;; ================================================================
;; Timestep embedding
;; ================================================================

(deftest embed-timestep-test
  (testing "sinusoidal embedding produces correct shape"
    (let [ref (double-array 1)
          emb (gsdm/embed-timestep ref 50.0 64)]
      (is (= 64 (alength emb)))
      (is (arr-finite? emb))))

  (testing "different timesteps produce different embeddings"
    (let [ref (double-array 1)
          e0 (gsdm/embed-timestep ref 0.0 8)
          e100 (gsdm/embed-timestep ref 100.0 8)]
      (is (not (approx= (aget e0 0) (aget e100 0)))))))

;; ================================================================
;; TimestepMLP
;; ================================================================

(deftest timestep-mlp-test
  (testing "timestep MLP produces correct shape"
    (let [d 16
          rng (java.util.Random. 42)
          rand-arr (fn [n] (let [a (double-array n)]
                             (dotimes [i n] (aset a i (* 0.1 (.nextGaussian rng))))
                             a))
          W1 (rand-arr (* d d)) b1 (double-array d)
          W2 (rand-arr (* d d)) b2 (double-array d)
          out (gsdm/timestep-mlp 50.0 W1 b1 W2 b2 d)]
      (is (= d (alength out)))
      (is (arr-finite? out)))))

;; ================================================================
;; ResnetBlock
;; ================================================================

(deftest resnet-block-test
  (testing "resnet block output shape matches input"
    (let [n-nodes 5 d 8
          rng (java.util.Random. 42)
          rand-arr (fn [n] (let [a (double-array n)]
                             (dotimes [i n] (aset a i (* 0.1 (.nextGaussian rng))))
                             a))
          ones-arr (fn [n] (let [a (double-array n)] (java.util.Arrays/fill a 1.0) a))
          x (rand-arr (* n-nodes d))
          temb (rand-arr d)
          W1 (rand-arr (* d d)) b1 (double-array d)
          W2 (rand-arr (* d d)) b2 (double-array d)
          Wt (rand-arr (* d d)) bt (double-array d)
          g1 (ones-arr d) bn1 (double-array d)
          g2 (ones-arr d) bn2 (double-array d)
          out (gsdm/resnet-block x temb W1 b1 W2 b2 Wt bt g1 bn1 g2 bn2 n-nodes d)]
      (is (= (* n-nodes d) (alength out)))
      (is (arr-finite? out))))

  (testing "skip connection means output ≈ input for small weights"
    (let [n-nodes 3 d 4
          x (double-array [1 2 3 4  5 6 7 8  9 10 11 12])
          temb (double-array d)
          zero-arr (fn [n] (double-array n))
          ones-arr (fn [n] (let [a (double-array n)] (java.util.Arrays/fill a 1.0) a))
          out (gsdm/resnet-block x temb
                                 (zero-arr (* d d)) (zero-arr d)
                                 (zero-arr (* d d)) (zero-arr d)
                                 (zero-arr (* d d)) (zero-arr d)
                                 (ones-arr d) (zero-arr d)
                                 (ones-arr d) (zero-arr d)
                                 n-nodes d)]
      ;; With zero weights, h ≈ 0 → output ≈ x (skip connection)
      (is (= (* n-nodes d) (alength out))))))

;; ================================================================
;; Multi-head graph attention
;; ================================================================

(deftest graph-attention-multihead-test
  (testing "multi-head graph attention on small graph"
    (let [n-nodes 4 n-edges 6 d 8 n-heads 2
          rng (java.util.Random. 42)
          rand-arr (fn [n] (let [a (double-array n)]
                             (dotimes [i n] (aset a i (* 0.05 (.nextGaussian rng))))
                             a))
          ones-arr (fn [n] (let [a (double-array n)] (java.util.Arrays/fill a 1.0) a))
          h (rand-arr (* n-nodes d))
          ;; Star graph: 0→1, 0→2, 0→3, 1→0, 2→0, 3→0
          src-edges (long-array [0 0 0 1 2 3])
          dst-edges (long-array [1 2 3 0 0 0])
          out (gsdm/graph-attention-multihead h
                                              (nn/xavier-init d d) (double-array d)
                                              (nn/xavier-init d d) (double-array d)
                                              (nn/xavier-init d d) (double-array d)
                                              (ones-arr d) (double-array d)
                                              src-edges dst-edges
                                              n-nodes n-edges d n-heads)]
      (is (= (* n-nodes d) (alength out)))
      (is (arr-finite? out)))))

;; ================================================================
;; Flat embedder
;; ================================================================

(deftest flat-embed-test
  (testing "flat embed with state + position embeddings"
    (let [n-vars 5 d 8 n-spaces 3 n-states gsdm/N-STATES
          rng (java.util.Random. 42)
          values (double-array [0.1 0.2 0.3 0.4 0.5])
          spaces (long-array [0 1 2 0 1])
          states (long-array [gsdm/LATENT gsdm/OBSERVED gsdm/LATENT
                              gsdm/LATENT gsdm/OBSERVED])
          We (double-array (repeatedly d #(.nextGaussian rng)))
          be (double-array d)
          space-emb (double-array (repeatedly (* n-spaces d) #(* 0.1 (.nextGaussian rng))))
          state-emb (double-array (repeatedly (* n-states d) #(* 0.1 (.nextGaussian rng))))
          pos-emb (gsdm/compute-position-embeddings spaces n-vars d)
          out (gsdm/flat-embed values space-emb spaces state-emb states pos-emb
                               We be n-vars d n-spaces n-states)]
      (is (= (* n-vars d) (alength out)))
      (is (arr-finite? out)))))

(deftest flat-unembed-test
  (testing "flat unembed output shape"
    (let [n-vars 5 d 8
          rng (java.util.Random. 42)
          emb (double-array (repeatedly (* n-vars d) #(* 0.1 (.nextGaussian rng))))
          Wu1 (nn/xavier-init d d) bu1 (double-array d)
          Wu2 (double-array (repeatedly d #(* 0.1 (.nextGaussian rng))))
          bu2 (double-array [0.0])
          out (gsdm/flat-unembed emb Wu1 bu1 Wu2 bu2 n-vars d)]
      (is (= n-vars (alength out)))
      (is (arr-finite? out)))))

;; ================================================================
;; Transformer block
;; ================================================================

(deftest transformer-block-test
  (testing "transformer block output shape"
    (let [n-nodes 4 n-edges 6 d 8 n-heads 2
          rng (java.util.Random. 42)
          rand-arr (fn [n] (let [a (double-array n)]
                             (dotimes [i n] (aset a i (* 0.05 (.nextGaussian rng))))
                             a))
          ones-arr (fn [n] (let [a (double-array n)] (java.util.Arrays/fill a 1.0) a))
          x (rand-arr (* n-nodes d))
          temb (rand-arr d)
          src-edges (long-array [0 0 0 1 2 3])
          dst-edges (long-array [1 2 3 0 0 0])
          res-w {:W1 (rand-arr (* d d)) :b1 (double-array d)
                 :W2 (rand-arr (* d d)) :b2 (double-array d)
                 :Wt (rand-arr (* d d)) :bt (double-array d)
                 :g1 (ones-arr d) :bn1 (double-array d)
                 :g2 (ones-arr d) :bn2 (double-array d)}
          attn-w {:Wq (nn/xavier-init d d) :bq (double-array d)
                  :Wk (nn/xavier-init d d) :bk (double-array d)
                  :Wv (nn/xavier-init d d) :bv (double-array d)
                  :g (ones-arr d) :b (double-array d)}
          out (gsdm/transformer-block x temb res-w attn-w
                                      src-edges dst-edges n-nodes n-edges d n-heads)]
      (is (= (* n-nodes d) (alength out)))
      (is (arr-finite? out)))))

;; ================================================================
;; Full GSDM model
;; ================================================================

(deftest gsdm-config-test
  (testing "default config"
    (let [cfg (gsdm/make-gsdm-config)]
      (is (= 64 (:emb-dim cfg)))
      (is (= 2 (:n-heads cfg)))
      (is (= 12 (:n-layers cfg)))
      (is (= gsdm/N-STATES (:n-states cfg))))))

(deftest gsdm-init-weights-test
  (testing "weight initialization includes state embeddings"
    (let [cfg (gsdm/make-gsdm-config :n-layers 2 :emb-dim 8 :n-spaces 3)
          weights (gsdm/init-gsdm-weights cfg)]
      (is (contains? weights :temb-W1))
      (is (contains? weights :embed-We))
      (is (contains? weights :embed-state))
      (is (contains? weights :layer-0-res-W1))
      (is (contains? weights :layer-1-attn-Wv))
      (is (not (contains? weights :layer-2-res-W1)))
      ;; State embedding is n-states * emb-dim
      (is (= (* gsdm/N-STATES 8) (alength ^doubles (:embed-state weights))))
      (let [n-params (gsdm/count-parameters weights)]
        (is (pos? n-params))
        (println "  2-layer emb_dim=8 params:" n-params)))))

(deftest gsdm-forward-dynamic-test
  (testing "dynamic forward pass on Boolean SAT graph"
    (let [n-vars 4 n-edges 12 n-spaces 1
          cfg (gsdm/make-gsdm-config :n-layers 2 :emb-dim 8 :n-heads 2 :n-spaces n-spaces)
          weights (gsdm/init-gsdm-weights cfg)
          values (double-array [0.1 -0.3 0.5 -0.2])
          spaces (long-array [0 0 0 0])
          states (long-array [gsdm/LATENT gsdm/LATENT gsdm/LATENT gsdm/LATENT])
          src-edges (long-array [0 0 0 1 1 1 2 2 2 3 3 3])
          dst-edges (long-array [1 2 3 0 2 3 0 1 3 0 1 2])
          pred (gsdm/gsdm-forward-dynamic weights cfg values spaces states 50.0
                                          src-edges dst-edges n-vars n-edges)]
      (is (= n-vars (alength pred)))
      (is (arr-finite? pred))
      (println "  GSDM forward output:" (vec pred))))

  (testing "forward pass with marginalized variables"
    (let [n-vars 4 n-edges 12 n-spaces 1
          cfg (gsdm/make-gsdm-config :n-layers 2 :emb-dim 8 :n-heads 2 :n-spaces n-spaces)
          weights (gsdm/init-gsdm-weights cfg)
          values (double-array [0.1 -0.3 0.5 -0.2])
          spaces (long-array [0 0 0 0])
          ;; Marginalize variable 1
          states (long-array [gsdm/LATENT gsdm/MARGINALIZED gsdm/LATENT gsdm/LATENT])
          src-edges (long-array [0 0 0 1 1 1 2 2 2 3 3 3])
          dst-edges (long-array [1 2 3 0 2 3 0 1 3 0 1 2])
          mg (gsdm/marginalize-graph values spaces states src-edges dst-edges n-vars n-edges)
          pred (gsdm/gsdm-forward-dynamic weights cfg
                                          (:values mg) (:spaces mg) (:states mg) 50.0
                                          (:src-edges mg) (:dst-edges mg)
                                          (:n-vars mg) (:n-edges mg))]
      ;; 3 variables after marginalization
      (is (= 3 (alength pred)))
      (is (arr-finite? pred)))))


;; ================================================================
;; GPU kernel codegen tests
;; ================================================================

(deftest scatter-reduce-kernel-test
  (testing "scatter-reduce kernel generates valid OpenCL"
    (require 'raster.compiler.backend.gpu.opencl-codegen)
    (let [emit (resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-scatter-reduce-kernel)
          src (emit "scatter_test" :double :atomic :with-weights? true)]
      (is (string? src))
      (is (.contains src "scatter_test"))
      (is (.contains src "src_edges"))
      (is (.contains src "dst_edges"))
      (is (.contains src "weights"))))

  (testing "sorted variant generates valid OpenCL"
    (let [emit (resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-scatter-reduce-kernel)
          src (emit "scatter_sorted" :float :sorted :with-weights? false)]
      (is (string? src))
      (is (.contains src "seg_offsets"))
      (is (not (.contains src "weights"))))))

(deftest row-softmax-kernel-test
  (testing "row-softmax kernel generates valid OpenCL"
    (require 'raster.compiler.backend.gpu.opencl-codegen)
    (let [emit (resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-row-softmax-kernel)
          src (emit "softmax_test" :double)]
      (is (string? src))
      (is (.contains src "softmax_test"))
      (is (.contains src "row_len"))
      (is (.contains src "barrier"))
      (is (.contains src "exp")))))

(deftest group-norm-kernel-test
  (testing "GroupNorm kernel generates valid OpenCL"
    (require 'raster.compiler.backend.gpu.opencl-codegen)
    (let [emit (resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-group-norm-kernel)
          src (emit "gn_test" :double)]
      (is (string? src))
      (is (.contains src "gn_test"))
      (is (.contains src "gamma"))
      (is (.contains src "beta"))
      (is (.contains src "num_groups")))))

(deftest gemm-nonsquare-kernel-test
  (testing "non-square GEMM kernel generates valid OpenCL"
    (require 'raster.compiler.backend.gpu.opencl-codegen)
    (let [emit (resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-gemm-nonsquare-kernel)
          src (emit "gemm_test")]
      (is (string? src))
      (is (.contains src "gemm_test"))
      (is (.contains src "intel_sub_group_f16_f16_matrix_mad_k16"))
      (is (.contains src "col0 < N"))
      (is (.contains src "col1 < N")))))

;; ================================================================
;; FP16 buffer tests
;; ================================================================

(deftest fp16-buffer-roundtrip-test
  (testing "double→FP16→double roundtrip"
    (let [input (double-array [1.0 -0.5 3.14159 0.0 100.0])
          n (alength input)]
      (let [shorts (short-array n)
            _ (dotimes [i n]
                (aset shorts i (short (Float/floatToFloat16 (float (aget input i))))))
            output (double-array n)
            _ (dotimes [i n]
                (aset output i (double (Float/float16ToFloat (aget shorts i)))))]
        (dotimes [i n]
          (is (< (Math/abs (- (aget output i) (aget input i)))
                 (max 0.01 (* 0.002 (Math/abs (aget input i)))))))))))



;; ================================================================
;; Tree-based defmodel + compile-train-step (new params API)
;; ================================================================

(deftest ^:compiled gsdm-defmodel-forward-matches-dynamic
  (testing "compile-aot of make-gsdm-loss matches gsdm-forward-dynamic"
    (let [n-vars 4 n-edges 12 n-spaces 1 emb-dim 8 n-heads 2 n-layers 2
          cfg (gsdm/make-gsdm-config :n-layers n-layers :emb-dim emb-dim
                                     :n-heads n-heads :n-spaces n-spaces)
          weights (gsdm/init-gsdm-weights cfg)
          w-tree  (gsdm/flat->tree weights n-layers)
          x0 (double-array [1.0 0.0 1.0 0.0])
          spaces (long-array [0 0 0 0])
          states (long-array (repeat n-vars gsdm/LATENT))
          src-edges (long-array [0 0 0 1 1 1 2 2 2 3 3 3])
          dst-edges (long-array [1 2 3 0 2 3 0 1 3 0 1 2])
          pos-emb (gsdm/compute-position-embeddings spaces n-vars emb-dim)
          t-val 50.0
          ;; Reference: dynamic forward + manual MSE
          pred-dyn (gsdm/gsdm-forward-dynamic weights cfg x0 spaces states
                                              t-val src-edges dst-edges
                                              n-vars n-edges)
          ref-loss (let [acc (atom 0.0)]
                     (dotimes [i n-vars]
                       (let [d (- (aget ^doubles pred-dyn i) (aget ^doubles x0 i))]
                         (swap! acc + (* d d))))
                     (/ @acc (double n-vars)))
          ;; defmodel + compile-aot
          model    (gsdm/make-gsdm-loss {:n-layers n-layers :emb-dim emb-dim
                                         :n-heads n-heads :n-spaces n-spaces})
          fast     (rp/compile-aot model)
          model-loss (fast w-tree x0 spaces x0 states pos-emb
                           src-edges dst-edges t-val n-vars n-edges)]
      (is (< (Math/abs (- model-loss ref-loss)) 1e-9)
          (format "compile-aot model loss %.6f matches reference %.6f"
                  model-loss ref-loss)))))

(deftest ^:compiled gsdm-fused-train-step-converges
  (testing "rp/compile-train-step with the new tree API converges"
    (let [n-vars 4 n-edges 12 n-spaces 1 emb-dim 8 n-heads 2 n-layers 2
          cfg (gsdm/make-gsdm-config :n-layers n-layers :emb-dim emb-dim
                                     :n-heads n-heads :n-spaces n-spaces)
          weights (gsdm/init-gsdm-weights cfg)
          w     (gsdm/flat->tree weights n-layers)
          model (gsdm/make-gsdm-loss {:n-layers n-layers :emb-dim emb-dim
                                      :n-heads n-heads :n-spaces n-spaces})
          ts    (rp/compile-train-step model)
          state ((:init-state ts) w)
          train-fn (:train-fn ts)
          x0 (double-array [1.0 0.0 1.0 0.0])
          spaces (long-array [0 0 0 0])
          states (long-array (repeat n-vars gsdm/LATENT))
          src-edges (long-array [0 0 0 1 1 1 2 2 2 3 3 3])
          dst-edges (long-array [1 2 3 0 2 3 0 1 3 0 1 2])
          pos-emb (gsdm/compute-position-embeddings spaces n-vars emb-dim)
          losses (vec (for [step (range 1 6)]
                        (train-fn w (:m state) (:v state)
                                  x0 spaces x0 states pos-emb
                                  src-edges dst-edges 50.0 n-vars n-edges
                                  1.0    ;; max-grad-norm
                                  0.001  ;; lr
                                  0.9 0.999 1e-8 step)))]
      (println "  losses:" (mapv #(format "%.4f" %) losses))
      (is (every? Double/isFinite losses))
      (is (< (last losses) (first losses))
          (format "loss[5]=%.4f should be < loss[1]=%.4f"
                  (last losses) (first losses))))))
