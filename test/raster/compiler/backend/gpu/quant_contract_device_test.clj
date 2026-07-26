(ns raster.compiler.backend.gpu.quant-contract-device-test
  "W1 quant spike: the SAME contraction abstraction spans int8. A par/contract over int8
   operands lowers (via the shared analyze-contraction + orientation gate) to a WIDENING
   kernel — int8 operands, int32 accumulate — with a dequant epilogue (out = scale·acc), and
   its f32 output matches a reference int8 matmul on device. Proves quantized matmul is the
   same SOAC contraction with different (operand/accumulate/output dtype, epilogue) facets;
   the dp4a / int8-DPAS packed dot is the tensorize leaf (perf layer). Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.backend.gpu.segop-opencl :as sco])
  (:import [java.lang.foreign MemorySegment]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- matmul-form [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(defn- ref-quant-matmul [^bytes A ^bytes B m k n scale a-zp b-zp]
  (let [scale (double scale) a-zp (int a-zp) b-zp (int b-zp)
        C (float-array (* m n))]
    (dotimes [i m] (dotimes [j n]
      (let [acc (loop [l 0 a 0]
                  (if (< l k)
                    (recur (inc l) (+ a (* (- (int (aget A (+ (* i k) l))) a-zp)
                                           (- (int (aget B (+ (* l n) j))) b-zp))))
                    a))]
        (aset C (+ (* i n) j) (float (* scale (double acc)))))))
    C))

(defn- run-quant [m k n scheme]
  (let [{:keys [scale a-zp b-zp] :or {scale 1.0 a-zp 0 b-zp 0}} scheme
        ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        make-buffer (ns-resolve ze 'make-buffer)
        arr->buf! (ns-resolve ze 'array->buffer!)
        buf->floats (ns-resolve ze 'buffer->float-array)
        launch-2d! (ns-resolve ze 'launch-2d!)
        A (byte-array (map #(byte (- (mod % 255) 127)) (range (* m k))))
        B (byte-array (map #(byte (- (mod (* 3 %) 255) 127)) (range (* k n))))
        ;; The scale and zero-points are ordinary FORM DATA now — an :epilogue and a per-operand
        ;; :decode — not a `:scheme` record with a private scale channel. Routed through
        ;; route-contraction because there is no int8-specific emitter left: this is the staged
        ;; emitter with one int32 contract level.
        form (concat (matmul-form m n k)
                     [:epilogue {:acc 'acc :expr '(raster.numeric/* acc s)
                                 :scalars [{:sym 's :dtype :float}]}]
                     (when-not (and (zero? a-zp) (zero? b-zp))
                       [:decode (cond-> {}
                                  (not (zero? a-zp)) (assoc 'A (list 'clojure.core/- 'x a-zp))
                                  (not (zero? b-zp)) (assoc 'B (list 'clojure.core/- 'x b-zp)))]))
        {:keys [kernel-name source array-params epilogue-scalars]}
        (route/route-contraction form :dtype :byte)
        _ (assert (= '[A B] array-params))
        _ (assert (= '[s] epilogue-scalars) "the scale is a surfaced epilogue scalar")
        _ (register! kernel-name {:source source :dtype :byte})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        abuf (arr->buf! (make-buffer (* m k) :byte) A)
        bbuf (arr->buf! (make-buffer (* k n) :byte) B)
        obuf (make-buffer (* m n) :float)
        nseg (* m n)
        gx (long (Math/ceil (/ (double nseg) 256.0)))
        args [(:segment abuf) (:segment bbuf) (:segment obuf)
              {:type :float :value (float scale)} {:type :int :value (int nseg)}]]
    (launch-2d! kernel-handle [256 1] [gx 1] args)
    {:gpu (vec (buf->floats obuf)) :cpu (vec (ref-quant-matmul A B m k n scale a-zp b-zp))}))

(defn- rel-close? [xs ys tol]
  (and (= (count xs) (count ys))
       (every? true? (map (fn [a b] (< (/ (Math/abs (- (double a) (double b))) (max 1.0 (Math/abs (double b)))) tol)) xs ys))))

(deftest quant-contraction-emits-generic-types-and-scheme
  (testing "types come from the GENERIC dtype facet map (not quant-specific); scheme carries semantics"
    (let [sr (cl/contract-form->segred (matmul-form 64 64 64) :dtype :byte)
          {:keys [source dtype out-dtype array-params epilogue-scalars]}
          (route/route-contraction (concat (matmul-form 4 6 8)
                                           [:epilogue {:acc 'acc :expr '(raster.numeric/* acc s)
                                                       :scalars [{:sym 's :dtype :float}]}])
                                   :dtype :byte)]
      (is (= :byte dtype))     ; int8 operand = the generic :byte dtype (int8_t)
      (is (= :float out-dtype))
      (is (= '[A B] array-params))
      (is (= '[s] epilogue-scalars))                         ; the scale, as an epilogue scalar
      (is (str/includes? source "const char* restrict A"))   ; :byte → opencl "char"
      (is (re-find #"int acc_0 = 0" source))                 ; :int widening accumulator
      ;; NB no explicit (int) cast: char*char integer-promotes to int in C, so the widening the old
      ;; kernel spelled out is implied by the dtype PAIR (:byte operands, :int accumulator)
      (is (re-find #"acc_0 \+= \(A\[" source))
      (is (re-find #"out\[seg\] = \(\(\(float\)acc_0\) \* s\)" source))))  ; dequant epilogue
  (testing "zero-point extends the decode scheme (asymmetric quant) — subtract in the load"
    (let [sr (cl/contract-form->segred (matmul-form 32 32 32) :dtype :byte)
          src (:source (route/route-contraction
                        (concat (matmul-form 4 6 8)
                                [:epilogue {:acc 'acc :expr '(raster.numeric/* acc s)
                                            :scalars [{:sym 's :dtype :float}]}
                                 :decode {'A '(clojure.core/- x 5) 'B '(clojure.core/- x -3)}])
                        :dtype :byte))]
      ;; the zero-point is a per-operand :decode on the LOAD path — exact, and needing no
      ;; correction reductions (a scale, which factors out of the sum, is the epilogue instead)
      (is (re-find #"\(A\[\(\(i \* 8\) \+ l\)\] - 5\)" src))
      (is (re-find #"\(B\[\(\(l \* 6\) \+ j\)\] - -3\)" src)))))

(deftest quant-contraction-matches-ref-on-device
  (if-not @gpu?
    (println "[skip] quant-contract-device: no GPU device available")
    (do
      (testing "symmetric int8 matmul (128×96×128) dequant == reference"
        (let [{:keys [gpu cpu]} (run-quant 128 96 128 {:scale 0.01})]
          (is (rel-close? gpu cpu 1.0e-6) (str "gpu " (take 3 gpu) " vs cpu " (take 3 cpu)))))
      (testing "non-square int8 (100×70×64)"
        (let [{:keys [gpu cpu]} (run-quant 100 70 64 {:scale 0.005})]
          (is (rel-close? gpu cpu 1.0e-6) "non-square")))
      (testing "ASYMMETRIC int8 with zero-points (96×80×64) == reference"
        (let [{:keys [gpu cpu]} (run-quant 96 80 64 {:scale 0.008 :a-zp 7 :b-zp -4})]
          (is (rel-close? gpu cpu 1.0e-6) "asymmetric zero-point"))))))
