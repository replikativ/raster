(ns raster.gpu.link-spike-test
  "C.spike (.internal/artifact_layer_design.md §7.2) — de-risk for the composition/linking PR.
   Link TWO instances of ONE compiled descriptor into ONE command graph with the intermediate a
   device-resident INTERNAL node (never downloaded), using only bind-step! internals + a
   hand-written 2-instance binding-plan (sym→key from DATA).

   SCOPE / HONESTY: this proves the linking primitive for the :map / :reduce / :map-void
   conventions bind-step! handles — the sym→key-as-data plan, the shared internal resident node,
   the single recorded graph. It does NOT de-risk the GEMM half: bind-step! THROWS on a :gemm step
   (spike-gemm-is-the-known-gap below asserts this), so a composite with linear layers is NOT yet
   unblocked. §3.2's remaining work — unifying bind-program!'s GEMM/scatter expansion into the
   per-instance binder — is the composition PR's core task, still un-de-risked by this spike."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.compiler.pipeline :as pl]
            [raster.dl.nn :as nn]))

;; A minimal elementwise forward: y = (x + w) * x  (residual-add then hadamard — the proven-
;; resident dt-two-step shape). Composing it twice with weights w0,w1 gives the intermediate
;; x1 = (x+w0)*x as a device-resident internal node the link shares between the two instances.
(deftm spike-had
  [x :- (Array float) w :- (Array float) n :- Long] :- (Array float)
  (let [s (nn/residual-add x w n)]
    (nn/hadamard s x n)))

(defn- fa ^floats [n seed]
  (let [rng (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (+ 0.5 (* 0.5 (.nextGaussian rng))))))
    a))

(defn- cpu-fwd ^floats [^floats x ^floats w n]
  ;; (x + w) * x, elementwise — residual-add then hadamard.
  (let [out (float-array n)]
    (dotimes [i n] (aset out i (float (* (+ (aget x i) (aget w i)) (aget x i)))))
    out))

(deftest spike-two-instance-link
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "C.spike 2-instance link")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session   (ns-resolve gpu 'make-session)
          bind-step!     (ns-resolve gpu 'bind-step!)
          record-graph!  (ns-resolve gpu 'record-graph!)
          replay!        (ns-resolve gpu 'replay!)
          alloc!         (ns-resolve gpu 'alloc!)
          download       (ns-resolve gpu 'download)
          close-session! (ns-resolve gpu 'close-session!)
          n 4096
          x0 (fa n 1) W0 (fa n 2) W1 (fa n 3)
          args [x0 W0 n]
          prog (pl/compile-gpu-program #'spike-had :ze:0 :dtype :float :on-non-resident :nil)
          _ (is (some? prog) "spike-had must extract fully resident (else the spike can't run)")
          steps (:steps prog)
          result-sym (:result-sym prog)
          scratch-sym (first (remove #(= % result-sym) (map :sym (:allocs prog))))
          x-sym 'x w-sym 'w]
      (testing "the descriptor is a clean 2-step elementwise chain"
        (is (= 2 (count steps)))
        (is (every? #(= :map (:convention %)) steps)))
      (let [sess (make-session :ze:0)]
        (try
          ;; resident buffers: two inputs' weights + shared input + the INTERNAL node x1 + output x2
          ;; + per-instance scratch. x1 is produced by instance 0, consumed by instance 1, never
          ;; downloaded — the whole point.
          (alloc! sess {:x0 [:float n x0] :W0 [:float n W0] :W1 [:float n W1]
                        :s0 [:float n nil] :s1 [:float n nil]
                        :x1 [:float n nil] :x2 [:float n nil]})
          ;; the hand-written 2-instance binding-plan (sym→key as DATA — what the linker promotes):
          (let [plan {0 {x-sym :x0, w-sym :W0, scratch-sym :s0, result-sym :x1}
                      1 {x-sym :x1, w-sym :W1, scratch-sym :s1, result-sym :x2}}
                phases (vec (for [inst [0 1] step steps]
                              (let [ph (keyword (str "i" inst "-" (name (:phase step))))]
                                (bind-step! sess (assoc step :phase ph) args (get plan inst))
                                ph)))]
            (record-graph! sess phases :graph)
            (replay! sess :graph)
            (let [x1 (download sess :x1)
                  x2 (download sess :x2)
                  x1-cpu (cpu-fwd x0 W0 n)
                  x2-cpu (cpu-fwd x1-cpu W1 n)
                  maxdiff (fn [^floats a ^floats b]
                            (reduce max 0.0 (map (fn [p q] (Math/abs (- (double p) (double q)))) a b)))]
              (println "  [spike] x1[0..2] dev:" (mapv #(aget ^floats x1 %) [0 1 2])
                       "cpu:" (mapv #(aget ^floats x1-cpu %) [0 1 2]))
              (println "  [spike] x2[0..2] dev:" (mapv #(aget ^floats x2 %) [0 1 2])
                       "cpu:" (mapv #(aget ^floats x2-cpu %) [0 1 2]))
              (println "  [spike] maxdiff x1:" (maxdiff x1 x1-cpu) " x2:" (maxdiff x2 x2-cpu))
              ;; pure elementwise → no reduction reassociation; the only gap is GPU-FMA vs
              ;; sequential-CPU float rounding (~1e-6), so the gate is a tight relative tolerance.
              (testing "the INTERNAL node x1 = f(x0,W0) is computed on-device"
                (is (< (maxdiff x1 x1-cpu) 1.0e-4)
                    "internal node x1 must equal CPU f(x0,W0) to float precision"))
              (testing "linked 2-instance graph matches CPU f(f(x,W0),W1), internal node never re-uploaded"
                (is (< (maxdiff x2 x2-cpu) 1.0e-3)
                    "device x2 must equal CPU double-composition to float precision"))))
          (finally (close-session! sess)))))))

(deftest spike-gemm-is-the-known-gap
  ;; the OTHER half of §7.2, made explicit and executable: the per-instance binder (bind-step!)
  ;; REJECTS a :gemm step today, so linking a composite with linear layers is NOT yet de-risked.
  ;; This asserts the gap rather than letting the elementwise spike overclaim it — the composition
  ;; PR's core task is unifying bind-program!'s GEMM/scatter expansion into the per-instance binder.
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "C.spike GEMM-convention gap")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session (ns-resolve gpu 'make-session)
          bind-step!   (ns-resolve gpu 'bind-step!)
          close-session! (ns-resolve gpu 'close-session!)
          sess (make-session :ze:0)]
      (try
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"cannot bind a :gemm"
             (bind-step! sess {:convention :gemm :kernel-name "spike_gemm" :phase :g0
                               :arrays [] :n-fn (fn [_] 1) :scalar-specs []}
                         [] identity))
            "bind-step! must reject :gemm — the composition PR must unify the GEMM expansion first")
        (finally (close-session! sess))))))
