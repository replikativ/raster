(ns raster.dl.broadcast-add-test
  "Pointwise broadcast migration: no BLAS dependency or numerical-policy change."
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse :as rev]
            [raster.arrays :as ra]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :as core :refer [deftm]]
            [raster.dl.array-ops :as ops]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl]
            [raster.numeric :as n]
            [raster.par :as par]))

(defn- bits [values]
  (mapv #(Double/doubleToRawLongBits (double %)) values))

(defn- former-loop-oracle [^doubles h ^doubles t batch dim]
  (let [out (double-array (* batch dim))]
    (dotimes [b batch]
      (dotimes [d dim]
        (let [idx (+ (* b (int dim)) d)]
          (aset out idx (+ (aget h idx) (aget t d))))))
    out))

(deftm broadcast-add-weighted-loss
  [h :- (Array double) t :- (Array double) seed :- (Array double)
   batch :- Long dim :- Long] :- Double
  (let [result (ops/broadcast-add h t batch dim)]
    (par/reduce acc 0.0 i (clojure.core/* batch dim)
                (n/+ acc (n/* (ra/aget result i) (ra/aget seed i))))))

(def ^:private checkpoint-cases
  [[[0.0 -0.0 9007199254740992.0 -9007199254740992.0]
    [-0.0 -0.0] 2 2]
   [[1.0000000000000002 -3.5 0.125 8.0 -16.0 0.5]
    [0.0000000000000002 2.25 -0.125] 2 3]
   [[] [] 0 3]
   [[] [] 3 0]
   [[] [] 0 0]
   [[] [] 0 -3]
   [[] [] -2 -3]])

(deftest broadcast-add-matches-the-former-double-checkpoints
  (doseq [[h-values t-values batch dim] checkpoint-cases]
    (let [h (double-array h-values) t (double-array t-values)
          before [(bits h) (bits t)]
          expected (former-loop-oracle h t batch dim)
          actual (ops/broadcast-add h t batch dim)
          replay (ops/broadcast-add h t batch dim)]
      (is (= (class (double-array 0)) (class actual)))
      (is (= (bits expected) (bits actual) (bits replay)))
      (is (not (identical? actual replay)))
      (is (not (identical? h actual)))
      (is (not (identical? t actual)))
      (is (= before [(bits h) (bits t)]))))
  ;; The allocation still rejects a negative product before touching either input.
  (doseq [[batch dim] [[-1 2] [2 -1]]]
    (is (thrown? NegativeArraySizeException
                 (ops/broadcast-add (double-array 0) (double-array 0) batch dim)))))

(deftest broadcast-add-compiled-jvm-matches-the-former-loop
  ;; Exercise the actual hardware-free compiler, not just the walked map surface.
  (let [compiled (pipeline/compile-aot #'ops/broadcast-add :dtype :double)]
    (doseq [[h-values t-values batch dim] checkpoint-cases]
      (let [h (double-array h-values) t (double-array t-values)
            before [(bits h) (bits t)]
            expected (former-loop-oracle h t batch dim)
            actual (compiled h t batch dim)]
        (is (= (class (double-array 0)) (class actual)))
        (is (= (bits expected) (bits actual)))
        (is (= before [(bits h) (bits t)]))))))

(deftest broadcast-add-retains-both-source-ad-directions
  (let [h (double-array [1.0 -2.0 3.0 4.0 -5.0 6.0])
        t (double-array [0.5 -1.0 2.0])
        seed (double-array [0.25 -0.5 2.0 -1.0 0.5 4.0])
        dh (double-array [0.5 0.25 -0.5 2.0 -1.0 0.125])
        dt (double-array [-0.25 0.5 1.0])
        expected-output (former-loop-oracle h t 2 3)
        expected-value (reduce + 0.0 (map * expected-output seed))
        expected-dt (mapv (fn [d] (+ (aget seed d) (aget seed (+ 3 d)))) (range 3))
        expected-tangent (reduce + 0.0 (map * (former-loop-oracle dh dt 2 3) seed))
        value+grad (rev/value+grad #'broadcast-add-weighted-loss :wrt [0 1])
        push (jvp/jvp #'broadcast-add-weighted-loss :wrt [0 1])
        [value gh gt & inactive] (value+grad h t seed 2 3)
        [jvalue tangent] (push h t seed 2 3 dh dt)]
    (is (= expected-value value jvalue))
    (is (= (bits seed) (bits gh)))
    (is (= (bits expected-dt) (bits gt)))
    (is (every? nil? inactive))
    (is (= expected-tangent tangent))
    (doseq [[argument expected] [[h (vec seed)] [t expected-dt]]
            coordinate (range (alength ^doubles argument))]
      (let [original (aget ^doubles argument coordinate)
            epsilon 0.125
            at (fn [offset]
                 (aset ^doubles argument coordinate (+ original offset))
                 (broadcast-add-weighted-loss h t seed 2 3))
            numerical (try (/ (- (at epsilon) (at (- epsilon))) (* 2.0 epsilon))
                           (finally (aset ^doubles argument coordinate original)))]
        (is (= (nth expected coordinate) numerical))))))

(deftest broadcast-add-declares-one-pointwise-map
  (let [body (first (core/ensure-walked-body! #'ops/broadcast-add))
        heads (keep #(when (seq? %) (first %)) (tree-seq coll? seq body))]
    (is (= 1 (count (filter #{'raster.par/map!} heads))))
    (is (not-any? #{'dotimes 'clojure.core/dotimes} heads))))

(deftest broadcast-add-dynamic-dimension-has-a-resident-compiler-route
  ;; The synthetic target exercises GPU compilation without requiring a device.
  ;; In particular, the positive-dimension guard must admit the dynamic remainder.
  (let [descriptor (pipeline/compile-gpu-program #'ops/broadcast-add :ze:debug
                                                 :dtype :double :on-non-resident :nil)]
    (is (some? descriptor) "the dynamic-dimension map must compile without host fallback")
    (when descriptor
      (is (= 1 (count (:steps descriptor))))
      (is (every? #(contains? #{:map :map-void} (:convention %)) (:steps descriptor))))))

(defn- check-device! [target]
  (let [h (double-array [1.0000000000000002 -3.5 0.125 8.0 -16.0 0.5])
        t (double-array [0.0000000000000002 2.25 -0.125])
        descriptor (pipeline/compile-gpu-program #'ops/broadcast-add target
                                                 :dtype :double :on-non-resident :nil)]
    (is (some? descriptor) "the actual Double primitive must be fully resident")
    (when descriptor
      (is (= 1 (count (:steps descriptor))))
      (is (every? #(contains? #{:map :map-void} (:convention %)) (:steps descriptor)))
      (let [session (gpu/make-session target)]
        (try
          (let [program (fixture/instantiate! session descriptor [h t 2 3])]
            (try
              (doseq [offset [0.0 1.25 -2.0]]
                (let [changed-h (double-array (map #(+ offset %) h))
                      changed-t (double-array (map #(+ % (* 2.0 offset)) t))
                      before [(bits changed-h) (bits changed-t)]
                      expected (former-loop-oracle changed-h changed-t 2 3)
                      result (fixture/run! program [changed-h changed-t 2 3])]
                  (is (= (bits expected) (bits (get result (:result-sym descriptor)))))
                  (is (= before [(bits changed-h) (bits changed-t)]))))
              (finally (fixture/close! program))))
          (finally (gpu/close-session! session)))))))

(deftest broadcast-add-double-map-replays-on-opencl
  (if @opencl/opencl-fp64-available?
    (check-device! :ocl:0)
    (opencl/opencl-skip! "Double broadcast-add map" :fp64)))

(deftest broadcast-add-double-map-replays-on-level-zero
  (if @ze/gpu-available?
    (check-device! :ze:0)
    (ze/gpu-skip! "Double broadcast-add map")))
