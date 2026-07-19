(ns raster.compiler.backend.gpu.tiled-contract-device-test
  "W1 step 3: ON-DEVICE numeric validation of the block-tiled + __local-staged contraction
   emitter (Arc GPU). Tests BOTH tile-divisible and non-divisible dims (boundary padding).
   Uses launch-2d! (workgroup [T T], grid [ceil(N/T) ceil(M/T)]). Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco])
  (:import [java.lang.foreign MemorySegment]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- ref-matmul [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (loop [l 0 acc 0.0]
                (if (< l k)
                  (recur (inc l) (+ acc (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))))
                  acc)))))
    C))

(defn- upload! [^doubles arr]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        alloc (ns-resolve ze 'alloc-shared)
        nbytes (* (alength arr) 8)
        seg (alloc nbytes)]
    (MemorySegment/copy (MemorySegment/ofArray arr) 0 seg 0 nbytes)
    [seg nbytes]))

(defn- run-tiled
  "Emit the tiled kernel for m×n×k, launch on device, return the output vector."
  [m k n]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        alloc (ns-resolve ze 'alloc-shared)
        launch-2d! (ns-resolve ze 'launch-2d!)
        A (double-array (map double (range (* m k))))
        B (double-array (map #(* 0.5 (double %)) (range (* k n))))
        form (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
                   (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
                         (list 'aget 'B (list '+ (list '* 'l n) 'j))))
        sr (cl/contract-form->segred form)
        {:keys [kernel-name source tile]} (sco/generate-tiled-contraction-kernel sr 'C)
        _ (register! kernel-name {:source source :dtype :double})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        [aseg _] (upload! A)
        [bseg _] (upload! B)
        out-bytes (* m n 8)
        oseg (alloc out-bytes)
        T (long tile)
        gcx (long (Math/ceil (/ (double n) T)))   ; columns / N
        gcy (long (Math/ceil (/ (double m) T)))]  ; rows / M
    ;; args order = sorted inputs [A B] then out
    (launch-2d! kernel-handle [T T] [gcx gcy] [aseg bseg oseg])
    (let [out (double-array (* m n))]
      (MemorySegment/copy oseg 0 (MemorySegment/ofArray out) 0 out-bytes)
      {:gpu (vec out) :cpu (vec (ref-matmul A B m k n))})))

(defn- run-regtiled
  "Emit the register-tiled kernel for m×n×k, launch on device, return {:gpu :cpu}."
  [m k n]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        alloc (ns-resolve ze 'alloc-shared)
        launch-2d! (ns-resolve ze 'launch-2d!)
        A (double-array (map double (range (* m k))))
        B (double-array (map #(* 0.5 (double %)) (range (* k n))))
        form (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
                   (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
                         (list 'aget 'B (list '+ (list '* 'l n) 'j))))
        sr (cl/contract-form->segred form)
        {:keys [kernel-name source block micro workgroup]}
        (sco/generate-regtiled-contraction-kernel sr 'C)
        [bm bn _] block
        [wgx wgy] workgroup
        _ (register! kernel-name {:source source :dtype :double})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        [aseg _] (upload! A)
        [bseg _] (upload! B)
        out-bytes (* m n 8)
        oseg (alloc out-bytes)
        gcx (long (Math/ceil (/ (double n) bn)))
        gcy (long (Math/ceil (/ (double m) bm)))]
    (launch-2d! kernel-handle [wgx wgy] [gcx gcy] [aseg bseg oseg])
    (let [out (double-array (* m n))]
      (MemorySegment/copy oseg 0 (MemorySegment/ofArray out) 0 out-bytes)
      {:gpu (vec out) :cpu (vec (ref-matmul A B m k n))})))

(defn- approx-eq? [xs ys]
  (and (= (count xs) (count ys))
       (every? true? (map (fn [a b] (< (Math/abs (- (double a) (double b))) 1.0e-9)) xs ys))))

(deftest tiled-contraction-matches-cpu-on-device
  (if-not @gpu?
    (println "[skip] tiled-contraction-device: no GPU device available")
    (do
      (testing "tile-divisible dims (32×32×32)"
        (let [{:keys [gpu cpu]} (run-tiled 32 32 32)]
          (is (approx-eq? gpu cpu) (str "divisible: GPU " (take 4 gpu) " vs CPU " (take 4 cpu)))))
      (testing "non-divisible dims (30×24×20) — boundary padding path"
        (let [{:keys [gpu cpu]} (run-tiled 30 24 20)]
          (is (approx-eq? gpu cpu) (str "boundary: GPU " (take 4 gpu) " vs CPU " (take 4 cpu)))))
      (testing "skewed dims (17×48×5)"
        (let [{:keys [gpu cpu]} (run-tiled 17 48 5)]
          (is (approx-eq? gpu cpu) "skewed"))))))

(deftest regtiled-contraction-matches-cpu-on-device
  (if-not @gpu?
    (println "[skip] regtiled-contraction-device: no GPU device available")
    (do
      (testing "block-divisible dims (128×64×256)"
        (let [{:keys [gpu cpu]} (run-regtiled 128 256 64)]
          (is (approx-eq? gpu cpu) (str "divisible: GPU " (take 4 gpu) " vs CPU " (take 4 cpu)))))
      (testing "non-divisible dims (100×70×33) — boundary + partial micro-tiles"
        (let [{:keys [gpu cpu]} (run-regtiled 100 33 70)]
          (is (approx-eq? gpu cpu) (str "boundary: GPU " (take 4 gpu) " vs CPU " (take 4 cpu)))))
      (testing "tiny dims (5×7×3) — everything is boundary"
        (let [{:keys [gpu cpu]} (run-regtiled 5 3 7)]
          (is (approx-eq? gpu cpu) "tiny"))))))
