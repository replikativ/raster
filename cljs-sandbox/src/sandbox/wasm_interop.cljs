(ns sandbox.wasm-interop
  "Exploration I (consumer half) — the cljs↔wasm CPU-runtime seam, in the browser.
   Mirrors raster's intended Model-A path: fetch a STATIC .wasm kernel, instantiate
   it, share a Float64Array view over its linear memory (zero-copy), call the kernel
   over (ptr,len), read results back. This is the fress boundary pattern minus
   fressian — flat f64 buffers, no serialization on the hot path.

   Validates: shadow-compiled cljs can ship + instantiate wasm, share memory, and
   drive a kernel with bit-identical results to a cljs reference.")

(defn log! [s]
  (when-let [el (.getElementById js/document "wasmlog")]
    (set! (.-textContent el) (str (.-textContent el) "\n" s)))
  (js/console.log s))

(defn ^:async run! []
  (-> (js/fetch "/kernels.wasm")
      (.then (fn [resp] (.arrayBuffer resp)))
      (.then (fn [buf] (js/WebAssembly.instantiate buf #js {})))
      (.then
       (fn [result]
         (let [exports (.. result -instance -exports)
               mem     (.-memory exports)
               saxpy   (.-saxpy exports)
               saxpy-s (.-saxpy_simd exports)
               n       1000
               a       3.0
               ;; lay out x at offset 0, y at N*8 bytes — a "buffer handle" = (offset,len)
               x-ptr   0
               y-ptr   (* n 8)
               view    (js/Float64Array. (.-buffer mem))
               x-idx   (/ x-ptr 8)
               y-idx   (/ y-ptr 8)]
           ;; fill shared memory directly through the view (zero-copy)
           (dotimes [i n]
             (aset view (+ x-idx i) (* 1.0 i))
             (aset view (+ y-idx i) (* 0.5 i)))
           ;; cljs reference
           (let [ref (js/Float64Array. n)]
             (dotimes [i n] (aset ref i (+ (* a (* 1.0 i)) (* 0.5 i))))
             ;; call the wasm kernel over shared memory (scalar)
             (saxpy x-ptr y-ptr n a)
             (let [max-err (reduce (fn [mx i] (js/Math.max mx (js/Math.abs (- (aget view (+ y-idx i)) (aget ref i)))))
                                   0 (range n))]
               (log! (str "cljs→wasm saxpy (scalar): n=" n
                          "  max|wasm-cljs|=" max-err
                          "  => " (if (< max-err 1e-9) "PASS" "FAIL")))
               (log! (str "  sample y[7]=" (aget view (+ y-idx 7)) " (ref " (aget ref 7) ")")))
             ;; reset y, run SIMD variant, check it matches too
             (dotimes [i n] (aset view (+ y-idx i) (* 0.5 i)))
             (saxpy-s x-ptr y-ptr n a)
             (let [max-err (reduce (fn [mx i] (js/Math.max mx (js/Math.abs (- (aget view (+ y-idx i)) (aget ref i)))))
                                   0 (range n))]
               (log! (str "cljs→wasm saxpy (simd f64x2): max|wasm-cljs|=" max-err
                          "  => " (if (< max-err 1e-9) "PASS" "FAIL"))))))))
      (.catch (fn [e] (log! (str "wasm-interop error: " e))))))
