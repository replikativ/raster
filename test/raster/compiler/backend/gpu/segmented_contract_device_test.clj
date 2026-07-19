(ns raster.compiler.backend.gpu.segmented-contract-device-test
  "W1 step 2c: ON-DEVICE numeric validation of the naive segmented-contraction emitter.
   Emits with LITERAL dims (no int scalar params → matches invoke-registered-kernel's arg
   order: inputs, output, trailing count), compiles SPIR-V, launches on the Arc, and
   compares to an independent CPU reference. Gated on a real GPU (skips cleanly otherwise).
   This is the golden correctness gate for the emit path — contention-insensitive."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]))

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

(defn- approx= [a b] (< (Math/abs (- (double a) (double b))) 1.0e-9))

(deftest segmented-contraction-matches-cpu-on-device
  (if-not @gpu?
    (println "[skip] segmented-contraction-device: no GPU device available")
    (let [ze (find-ns 'raster.gpu.ze-runtime)
          register! (ns-resolve ze 'register-kernel!)
          invoke!   (ns-resolve ze 'invoke-registered-kernel)
          m 3 k 4 n 2
          A (double-array (map double (range (* m k))))
          B (double-array (map #(* 0.5 (double %)) (range (* k n))))
          C (double-array (* m n))
          ;; LITERAL dims → the emitter inlines 3/2/4, no int scalar params.
          form (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
                     (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
                           (list 'aget 'B (list '+ (list '* 'l n) 'j))))
          sr  (cl/contract-form->segred form)
          {:keys [kernel-name source scalar-params]} (sco/generate-segmented-reduce-kernel sr 'C)]
      (testing "literal-dim contraction has no scalar params (dims inlined)"
        (is (empty? scalar-params)))
      (register! kernel-name {:source source :dtype :double :workgroup-size 256})
      (invoke! kernel-name [A B] C [] (* m n))     ; args: A, B, out, int _nseg=(m*n)
      (testing "GPU segmented-contraction result == CPU reference"
        (let [ref (ref-matmul A B m k n)]
          (is (every? true? (map approx= (vec C) (vec ref)))
              (str "GPU " (vec C) " vs CPU " (vec ref))))))))
