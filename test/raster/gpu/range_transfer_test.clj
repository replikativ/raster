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

(def ^:private opencl-available?
  (delay
    (try
      ((requiring-resolve 'raster.gpu.ocl-runtime/init!))
      true
      (catch Throwable _ false))))

(defn- with-filled-session
  "A session whose :kc0 buffer holds value=index at every element, so any disturbance is visible."
  [f]
  (let [s (g/make-session :ze:0)
        a (float-array N)]
    (try
      (dotimes [i N] (aset a i (float i)))
      (g/alloc! s {:kc0 [:float N a]})
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

(deftest a-resident-view-makes-range-offsets-relative
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ranged transfers: resident view")
    (with-filled-session
      (fn [s]
        (let [at (* 64 kvrow)
              n (* 8 kvrow)
              view (g/buffer-view s :kc0 {:byte-offset (* 4 at) :shape [n]})
              out (float-array n)]
          (g/download-range! s view out {:elements n})
          (is (every? (fn [i] (== (aget out i) (float (+ at i)))) (range n)))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds the buffer view"
                                (g/download-range! s view out
                                                   {:src-element (dec n) :elements 2}))))))))

(deftest resident-range-copy-stays-between-checked-views
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "resident range copy")
    (let [session (g/make-session :ze:0)
          values (float-array (map float (range 16)))]
      (try
        (g/alloc! session {:source [:float 16 values]
                           :destination [:float 16 nil]})
        (let [source (g/buffer-view session :source
                                    {:byte-offset (* 4 3) :shape [5]})
              destination (g/buffer-view session :destination
                                         {:byte-offset (* 4 7) :shape [5]})]
          (is (identical? destination
                          (g/copy-range! session source destination {:elements 5})))
          (is (= [0.0 0.0 0.0 0.0 0.0 0.0 0.0
                  3.0 4.0 5.0 6.0 7.0 0.0 0.0 0.0 0.0]
                 (vec (g/download session :destination))))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds the buffer view"
                                (g/copy-range! session source destination
                                               {:src-element 4 :elements 2}))))
        (finally
          (g/close-session! session))))))

(deftest opencl-resident-range-copy-uses-the-device-buffer-path
  (if-not @opencl-available?
    (is true "OpenCL device unavailable")
    (let [session (g/make-session :ocl:0)]
      (try
        (g/alloc! session {:source [:float 6 (float-array [1 2 3 4 5 6])]
                           :destination [:float 6 (float-array 6)]})
        (g/copy-range! session :source :destination
                       {:src-element 2 :dst-element 1 :elements 3})
        (is (= [0.0 3.0 4.0 5.0 0.0 0.0]
               (vec (g/download session :destination))))
        (finally
          (g/close-session! session))))))

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

;; ── batched: the whole KV cache in one call ─────────────────────────────────────────
(defn- with-kv-session
  "A session with `layers` kc/vc pairs, each holding value = (layer-tag + index) so a write to the
   wrong layer is as visible as a write to the wrong position."
  [layers f]
  (let [s (g/make-session :ze:0)]
    (try
      (doseq [l (range layers) kind ["kc" "vc"]]
        (let [k (keyword (str kind l)) tag (float (* 1e6 (+ 1 (* 2 l) (if (= kind "vc") 1 0))))
              a (float-array N)]
          (dotimes [i N] (aset a i (+ tag (float i))))
          (g/alloc! s {k [:float N a]})))
      (f s)
      (finally (g/close-session! s)))))

(deftest a-batch-moves-every-layer-in-one-call
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "batched ranged transfers: whole-cache export/import")
    (with-kv-session 3
      (fn [s]
        (let [tokens 64 n (* tokens kvrow)
              keys* (for [l (range 3) kind ["kc" "vc"]] (keyword (str kind l)))
              outs (zipmap keys* (repeatedly #(float-array n)))
              res (g/download-ranges! s (for [k keys*] [k (get outs k) {:elements n}]))]
          (is (= 6 (count res)) "one result per entry")
          (is (= (map #(get outs %) keys*) (seq res))
              "…and IN ENTRY ORDER — the i-th result is the i-th entry's destination, which is
               what lets a caller zip results back to layers without re-deriving the mapping")
          (doseq [[l kind] (for [l (range 3) kind ["kc" "vc"]] [l kind])]
            (let [k (keyword (str kind l)) tag (* 1e6 (+ 1 (* 2 l) (if (= kind "vc") 1 0)))]
              (is (every? (fn [i] (== (aget ^floats (get outs k) i) (float (+ tag i)))) (range n))
                  (str k " carries ITS OWN layer's prefix, not a neighbour's")))))))))

(deftest a-bad-entry-anywhere-leaves-the-whole-cache-untouched
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "batched ranged transfers: all-or-nothing")
    (with-kv-session 3
      (fn [s]
        ;; The failure a batched API newly makes possible: entries 1-5 valid, entry 6 out of range.
        ;; Without validate-all-first, five layers get overwritten before the throw and the cache
        ;; is half-restored — indistinguishable from correct until decode diverges.
        (let [n (* 64 kvrow)
              zeros (float-array n)
              entries (concat (for [l (range 3) kind ["kc"]] [(keyword (str kind l)) zeros {:elements n}])
                              (for [l (range 2)] [(keyword (str "vc" l)) zeros {:elements n}])
                              [[:vc2 zeros {:dst-element (- N 10) :elements n}]])]   ; the bad one, LAST
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds the buffer"
                                (g/upload-ranges! s entries)))
          (testing "NOT ONE of the five valid earlier entries was written"
            (doseq [[l kind] (for [l (range 3) kind ["kc" "vc"]] [l kind])]
              (let [k (keyword (str kind l)) tag (* 1e6 (+ 1 (* 2 l) (if (= kind "vc") 1 0)))
                    back (g/download s k)]
                (is (every? (fn [i] (== (aget ^floats back i) (float (+ tag i)))) (range n))
                    (str k " must be untouched"))))))))))

(deftest an-unknown-key-is-refused-before-anything-moves
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "batched ranged transfers: unknown key")
    (with-kv-session 1
      (fn [s]
        (let [n (* 8 kvrow) zeros (float-array n)]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"No buffer for key"
                                (g/upload-ranges! s [[:kc0 zeros {:elements n}]
                                                     [:kc99 zeros {:elements n}]])))
          (is (== (aget ^floats (g/download s :kc0) 0) (float 1e6)) ":kc0 untouched"))))))
