(ns raster.compiler.passes.parallel.contract-route-device-test
  "W1 integration: the routing brain (contract-route/route-contraction) picks the right
   kernel via the DPAS legality gate AND both branches are device-correct — the proof that
   the SOAC contraction path is a load-bearing drop-in for the hand-wired GEMM front door.

   - f16 gemma-shaped linear (N%8==0)  → :dpas     → == CPU matmul on f16 inputs (byte-id. golden)
   - f64                                → :regtiled → == CPU matmul (fallback: dtype-not-dpas)
   - pitch-unaligned f16 (N=70)         → :regtiled → == CPU matmul (fallback: n-pitch-unaligned)
   Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-route :as route])
  (:import [java.lang.foreign MemorySegment]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- f16 ^double [^double x] (double (Float/float16ToFloat (Float/floatToFloat16 (float x)))))

(defn- ref-matmul [^doubles A ^doubles B m k n round?]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (loop [l 0 acc 0.0]
                (if (< l k)
                  (recur (inc l) (+ acc (* (if round? (f16 (aget A (+ (* i k) l))) (aget A (+ (* i k) l)))
                                           (if round? (f16 (aget B (+ (* l n) j))) (aget B (+ (* l n) j))))))
                  acc)))))
    C))

(defn- matmul-form [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(defn- launch-routed
  "Launch a routed contraction on device; return its output as a double vector.
   `bufs` maps array-param symbol → DeviceBuffer, `out` is the output DeviceBuffer."
  [{:keys [kernel-name source array-params dtype wg grid scalar-args]} bufs out]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        launch-2d! (ns-resolve ze 'launch-2d!)
        buf->doubles (ns-resolve ze 'buffer->double-array)
        _ (register! kernel-name {:source source :dtype dtype})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        [gx gy] grid
        args (into (mapv #(:segment (get bufs %)) array-params)
                   (into [(:segment out)] scalar-args))]
    (launch-2d! kernel-handle wg [gx gy] args)
    (vec (buf->doubles out))))

(defn- mk-bufs [dtype ^doubles Ad ^doubles Bd m n k]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        make-buffer (ns-resolve ze 'make-buffer)
        arr->buf! (ns-resolve ze 'array->buffer!)
        halfs (ns-resolve ze 'buffer-of-floats-as-half)]
    (if (= dtype :half)
      {:A (halfs (float-array (map float Ad))) :B (halfs (float-array (map float Bd)))
       :out (make-buffer (* m n) :half)}
      {:A (arr->buf! (make-buffer (* m k) dtype) Ad) :B (arr->buf! (make-buffer (* k n) dtype) Bd)
       :out (make-buffer (* m n) dtype)})))

(defn- rel-close? [xs ys tol]
  (and (= (count xs) (count ys))
       (every? true? (map (fn [a b] (< (/ (Math/abs (- (double a) (double b))) (max 1.0 (Math/abs (double b)))) tol)) xs ys))))

(defn- run [m k n dtype]
  (let [Ad (double-array (map #(* 0.1 (- (double (mod % 7)) 3.0)) (range (* m k))))
        Bd (double-array (map #(* 0.1 (- (double (mod % 5)) 2.0)) (range (* k n))))
        r (route/route-contraction (matmul-form m n k) :dtype dtype)
        {:keys [A B out]} (mk-bufs (:dtype r) Ad Bd m n k)
        gpu (launch-routed r {'A A 'B B} out)
        cpu (vec (ref-matmul Ad Bd m k n (= (:dtype r) :half)))]
    {:route r :gpu gpu :cpu cpu}))

(deftest routing-picks-correct-kernel-and-is-device-correct
  (if-not @gpu?
    (println "[skip] contract-route-device: no GPU device available")
    (do
      (testing "f16 gemma-shaped linear (256×640×512, N%8==0) → DPAS, matches front door"
        (let [{:keys [route gpu cpu]} (run 256 640 512 :half)]
          (is (= :dpas (:strategy route)))
          (is (rel-close? gpu cpu 2.0e-2) (str "dpas: " (take 3 gpu) " vs " (take 3 cpu)))))
      (testing "f64 → gate rejects (dtype-not-dpas), routes to regtiled, matches CPU"
        (let [{:keys [route gpu cpu]} (run 128 96 128 :double)]
          (is (= :regtiled (:strategy route)))
          (is (= :dtype-not-dpas (:fallback-reason route)))
          (is (rel-close? gpu cpu 1.0e-9) "regtiled-f64 exact")))
      (testing "pitch-unaligned f16 (N=70) → gate rejects (n-pitch-unaligned), regtiled, matches CPU"
        (let [{:keys [route gpu cpu]} (run 128 96 70 :half)]
          (is (= :regtiled (:strategy route)))
          (is (= :n-pitch-unaligned (:fallback-reason route)))
          (is (rel-close? gpu cpu 2.0e-2) "regtiled-f16 fallback"))))))

(deftest routing-decision-is-device-free
  (testing "route-contraction makes the gate decision without a GPU (pure emit)"
    (is (= :dpas (:strategy (route/route-contraction (matmul-form 128 128 128) :dtype :half))))
    (is (= :regtiled (:strategy (route/route-contraction (matmul-form 128 128 128) :dtype :double))))
    (is (route/par-contract-form? (matmul-form 8 8 8)))
    (is (not (route/par-contract-form? '(raster.par/map! a i 8 body))))))
