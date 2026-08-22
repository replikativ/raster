(ns raster.gpu.range-transfer-test
  "RANGED TRANSFERS: move a sub-range of a session buffer, not the whole allocation.

   `upload!`/`download` copy the WHOLE buffer. A KV cache is allocated at `maxpos` positions and is
   position-major, so a continuation of `t` tokens is one contiguous prefix — exporting it should
   move `t` rows, not `maxpos`. For gemma-270m that is 9 MiB vs 72 MiB per 256-token continuation.

   The invariant that matters is not 'the range arrives' — it is 'NOTHING ELSE MOVES'. A ranged
   write that clobbers a neighbouring position looks identical to a correct one until a restored
   continuation diverges at the wrong token. So every test here checks the untouched regions too.

   The second invariant: out-of-range is an ERROR, never a clamp. `array->buffer!` clamps to
   `(min buf src)` bytes, which is fine for a whole-buffer copy and exactly wrong for a range."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as g])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

;; gemma-270m's real KV shape: 2048 positions x (1 kv-head x 256 head-dim)
(def ^:private maxpos 2048)
(def ^:private kvrow 256)
(def ^:private N (* maxpos kvrow))

(defn- with-filled-session
  "A session whose :kc0 buffer holds value=index at every element, so any disturbance is visible."
  [f]
  (let [s (g/make-session :ze:0)
        ze (find-ns 'raster.gpu.ze-runtime)
        buf ((ns-resolve ze 'make-buffer) N :float)]
    (try
      (swap! s assoc-in [:buffers :kc0] buf)
      (let [a (float-array N)] (dotimes [i N] (aset a i (float i))) (g/upload! s :kc0 a))
      (f s)
      (finally (g/close-session! s)))))

(defn- region-is-identity? [^floats arr from to]
  (every? (fn [i] (== (aget arr i) (float i))) (range from to)))

(deftest a-token-prefix-exports-as-one-contiguous-range
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ranged transfers: prefix export")
    (with-filled-session
      (fn [s]
        (let [tokens 128 n (* tokens kvrow) out (float-array n)]
          (g/download-range! s :kc0 out {:src-element 0 :elements n})
          (is (region-is-identity? out 0 n) "the first 128 positions, in order"))))))

(deftest a-ranged-import-touches-only-its-range
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ranged transfers: import from an off-heap segment")
    (with-filled-session
      (fn [s]
        (let [tokens 128 n (* tokens kvrow)
              at (* 64 kvrow)                                   ; import at token 64
              arena (Arena/ofShared)
              ;; an OFF-HEAP segment with the payload at a non-zero element offset — the mmap case
              seg (.allocate arena (* 4 (+ n 100)) 64)]
          (try
            (dotimes [i n] (.setAtIndex seg ValueLayout/JAVA_FLOAT (+ i 100) (float (- i))))
            (g/upload-range! s :kc0 seg {:src-element 100 :dst-element at :elements n})
            (let [back (g/download s :kc0)]
              (testing "the imported range is correct, from the segment's own offset"
                (is (every? (fn [i] (== (aget back (+ at i)) (float (- i)))) (range n))))
              (testing "NOTHING ELSE MOVED — the invariant a clobbering write would violate"
                (is (region-is-identity? back 0 at) "positions before the import")
                (is (region-is-identity? back (+ at n) N) "positions after the import")))
            (finally (.close arena))))))))

(deftest a-jvm-array-and-a-memory-segment-take-the-same-path
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ranged transfers: array/segment parity")
    (with-filled-session
      (fn [s]
        (let [n (* 8 kvrow)
              arr (float-array n) _ (dotimes [i n] (aset arr i (float (* 10 i))))
              seg (MemorySegment/ofArray (aclone arr))
              via-arr (float-array n) via-seg (float-array n)]
          (g/upload-range! s :kc0 arr {:dst-element 0 :elements n})
          (g/download-range! s :kc0 via-arr {:elements n})
          (g/upload-range! s :kc0 seg {:dst-element 0 :elements n})
          (g/download-range! s :kc0 via-seg {:elements n})
          (is (= (vec via-arr) (vec via-seg))))))))

(deftest out-of-range-is-an-error-not-a-clamp
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ranged transfers: bounds")
    (with-filled-session
      (fn [s]
        (testing "host side too small for the requested range"
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds the host-side"
                                (g/download-range! s :kc0 (float-array 16) {:elements 32}))))
        (testing "range runs past the end of the buffer"
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds the buffer"
                                (g/upload-range! s :kc0 (float-array 1024)
                                                 {:dst-element (- N 10) :elements 1024}))))
        (testing "negative offsets are refused, not wrapped"
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"negative"
                                (g/download-range! s :kc0 (float-array 16)
                                                   {:src-element -1 :elements 8}))))
        (testing "…and a refused transfer left the buffer untouched"
          (is (region-is-identity? (g/download s :kc0) 0 N)))))))

(deftest a-segment-source-does-not-materialize-a-jvm-array
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ranged transfers: zero-copy segment path")
    (with-filled-session
      (fn [s]
        ;; The claim that motivated this API: an mmap'd segment is copied straight into the shared
        ;; allocation. That was asserted, not measured; this MEASURES it. The threshold is an order
        ;; of magnitude below the payload, so a float[] materialization (= payload bytes) fails it.
        (let [n (* 128 kvrow) payload (* 4 n)
              arena (Arena/ofShared) seg (.allocate arena payload 64)
              bean (java.lang.management.ManagementFactory/getThreadMXBean)
              alloc #(.getCurrentThreadAllocatedBytes bean)]
          (try
            (dotimes [_ 3] (g/upload-range! s :kc0 seg {:elements n}))      ; warm
            (let [a0 (alloc) _ (dotimes [_ 20] (g/upload-range! s :kc0 seg {:elements n})) a1 (alloc)
                  per-call (quot (- a1 a0) 20)]
              (is (< per-call (quot payload 10))
                  (str "per-call heap " per-call " B vs payload " payload " B")))
            (finally (.close arena))))))))
