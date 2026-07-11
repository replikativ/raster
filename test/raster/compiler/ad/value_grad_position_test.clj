(ns raster.compiler.ad.value-grad-position-test
  "F1 regression: the value+grad/grad AD transform must fire wherever the
   `((value+grad #'f) args…)` call appears, not only when bound DIRECTLY to a let
   symbol. A call nested inside another form — e.g. `(nth ((value+grad #'f) …) k)`
   — used to survive the whole pipeline un-transformed and (on the resident path)
   silently return an INPUT arg instead of the gradient. The inline hoist now lifts
   the nested application into its own binding so the ordinary AD machinery fires;
   a `find-untransformed-vg` backstop throws if a call survives to GPU extraction."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :as rc]
            [raster.compiler.pipeline :as pl]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.ad.reverse :as rev]))

(rc/deftm vgp-loss
  [W :- (Array float) x :- (Array float) tgt :- (Array float)
   batch :- Long in-f :- Long out-f :- Long] :- Double
  (let [pred (nn/linear-nb x W batch in-f out-f)]
    (loss/mse-loss pred tgt (clojure.core/* batch out-f))))

;; grad wrt W bound directly (the position that always worked)
(rc/deftm vgp-train-direct
  [W :- (Array float) x :- (Array float) tgt :- (Array float)
   batch :- Long in-f :- Long out-f :- Long] :- (Array float)
  (let [vg ((rev/value+grad #'vgp-loss) W x tgt batch in-f out-f)
        dW (clojure.core/nth vg 1)]
    dW))

;; grad wrt W read via a NESTED nth — the F1 miscompile position
(rc/deftm vgp-train-nested
  [W :- (Array float) x :- (Array float) tgt :- (Array float)
   batch :- Long in-f :- Long out-f :- Long] :- (Array float)
  (clojure.core/nth ((rev/value+grad #'vgp-loss) W x tgt batch in-f out-f) 1))

(defn- rnd [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5)))) a))

(defn- fd-grad [W x tgt b in-f out-f]
  (let [eps 1e-3 g (float-array (alength W))]
    (dotimes [i (alength W)]
      (let [Wp (aclone W) Wm (aclone W)]
        (aset Wp i (float (+ (aget W i) eps)))
        (aset Wm i (float (- (aget W i) eps)))
        (aset g i (float (/ (- (vgp-loss Wp x tgt b in-f out-f)
                               (vgp-loss Wm x tgt b in-f out-f))
                            (* 2 eps))))))
    g))

(defn- max-rel-err [a ref]
  (let [mag (reduce max 1e-9 (map #(Math/abs (double %)) ref))]
    (/ (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) (seq a) (seq ref))) mag)))

(deftest nested-value+grad-compiles-to-gradient-not-input
  ;; CPU compile-aot: the nested `(nth (vg …) 1)` must produce the GRADIENT (matching
  ;; finite differences), NOT the first input arg W. Regression guard for F1.
  (let [b 4 in-f 3 out-f 2
        W (rnd (* out-f in-f) 1) x (rnd (* b in-f) 2) tgt (rnd (* b out-f) 3)
        fd (fd-grad W x tgt b in-f out-f)
        direct-fn (pl/compile-aot #'vgp-train-direct :dtype :float)
        nested-fn (pl/compile-aot #'vgp-train-nested :dtype :float)
        direct-out (direct-fn W x tgt b in-f out-f)
        nested-out (nested-fn W x tgt b in-f out-f)]
    (testing "direct-position grad matches finite differences"
      (is (< (max-rel-err direct-out fd) 1e-2)))
    (testing "nested-position grad matches finite differences (the F1 fix)"
      (is (< (max-rel-err nested-out fd) 1e-2)
          (str "nested grad " (vec nested-out) " vs FD " (vec fd))))
    (testing "nested grad is NOT the input W (the silent-miscompile symptom)"
      (is (not= (vec nested-out) (vec W))))
    (testing "nested and direct positions agree"
      (is (< (max-rel-err nested-out direct-out) 1e-4)))))

(deftest nested-value+grad-lowers-resident-same-as-direct
  ;; IR-level (no GPU needed): the nested-position train step lowers to the SAME
  ;; resident GEMM program as the direct-position one — the hoist makes the composed
  ;; AD IR identical. compile-gpu-program :on-non-resident :nil returns nil if any
  ;; step falls off the resident path.
  (let [direct (pl/compile-gpu-program #'vgp-train-direct :ze:0 :dtype :float :on-non-resident :nil)
        nested (pl/compile-gpu-program #'vgp-train-nested :ze:0 :dtype :float :on-non-resident :nil)]
    (is (some? direct) "direct train step extracts resident")
    (is (some? nested) "nested train step extracts resident (F1 fix)")
    (when (and direct nested)
      (is (= (mapv (juxt :convention :variant) (:steps direct))
             (mapv (juxt :convention :variant) (:steps nested)))
          "nested lowers to the same resident step sequence as direct"))))

(deftest untransformed-value+grad-backstop-detects
  ;; The fail-loud backstop recognizes a surviving value+grad application in both the
  ;; raw-call shape and the corrupted vector-literal shape, and passes clean IR.
  (let [find (var-get #'raster.compiler.pipeline/find-untransformed-vg)]
    (is (some? (find '(let* [g (clojure.core/nth ((raster.ad.reverse/value+grad (var f)) x tgt) 1)] g)))
        "detects surviving ((value+grad #'f) …) call")
    (is (some? (find '(let* [g (clojure.core/nth [(raster.ad.reverse/value+grad (var f)) x tgt] 1)] g)))
        "detects corrupted [(value+grad #'f) …] vector literal")
    (is (nil? (find '(let* [g (.invk foo x)] g)))
        "clean IR passes")))
