(ns raster.compiler.backend.gpu.contract-pipeline-device-test
  "W1 integration (the pipeline wire): a (raster.par/contract …) form flows through the ACTUAL
   opencl-pass → routes via the DPAS legality gate → emits invoke-registered-contraction! →
   registers + launches the chosen kernel on device → result matches CPU. End-to-end proof
   that the SOAC contraction path compiles automatically (not just as a standalone emitter).
   Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.opencl-pass :as ocl]
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

(defn- compile+run
  "Run a matmul par/contract form through opencl-pass at `dt`, register the emitted kernel,
   compile the marker to a fn, launch on device, return {:head :max-abs-diff}."
  [dt m k n]
  (let [{:keys [form kernels]} (ocl/opencl-pass (matmul-form m n k) :dtype dt :compile-spirv? true)
        _ (doseq [kr kernels] (ze/register-kernel! (:kernel-name kr) (select-keys kr [:source :dtype])))
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
