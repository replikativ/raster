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

;; ── A1a: 0-contract (outer product → SegMap) + n-free (batch → naive segred) ──────────
(defn- launch-1d-routed
  "Launch a routed 1-D contraction (segmap / naive-segred) with a trailing nseg count."
  [r bufs out-buf nseg]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        launch-2d! (ns-resolve ze 'launch-2d!)
        buf->doubles (ns-resolve ze 'buffer->double-array)
        _ (register! (:kernel-name r) {:source (:source r) :dtype (:dtype r)})
        {:keys [kernel-handle]} (ensure-loaded! (:kernel-name r))
        args (into (mapv #(:segment (get bufs %)) (:array-params r))
                   [(:segment out-buf) {:type :int :value (int nseg)}])]
    (launch-2d! kernel-handle (:wg r) (:grid r) args)
    (vec (buf->doubles out-buf))))

(defn- mk-f64 [xs] (let [ze (find-ns 'raster.gpu.ze-runtime)
                         mk (ns-resolve ze 'make-buffer) a->b (ns-resolve ze 'array->buffer!)]
                     (a->b (mk (count xs) :double) (double-array (map double xs)))))
(defn- out-f64 [n] ((ns-resolve (find-ns 'raster.gpu.ze-runtime) 'make-buffer) n :double))

(deftest a1a-zero-contract-routes-to-segmap-outer-product
  (if-not @gpu?
    (println "[skip] a1a-segmap: no GPU")
    (testing "0 contract axes → :segmap; C[i,j]=a[i]·b[j] == reference on device"
      (let [a [1 2 3 4] b [10 20 30]
            form (list 'raster.par/contract 'C [['i 4] ['j 3]] []
                       (list '* (list 'aget 'a 'i) (list 'aget 'b 'j)))
            r (route/route-contraction form :dtype :double)
            gpu (launch-1d-routed r {'a (mk-f64 a) 'b (mk-f64 b)} (out-f64 12) 12)
            cpu (vec (for [i (range 4) j (range 3)] (double (* (nth a i) (nth b j)))))]
        (is (= :segmap (:strategy r)))
        (is (= gpu cpu))))))

(deftest a1a-n-free-routes-to-naive-segred
  (if-not @gpu?
    (println "[skip] a1a-naive-segred: no GPU")
    (testing "3 free + 1 contract (batch matmul) → :naive-segred; == reference on device"
      (let [B 2 M 4 N 3 K 5
            Ad (double-array (map #(* 0.1 (double %)) (range (* B M K))))
            Bd (double-array (map #(* 0.2 (double %)) (range (* B K N))))
            form (list 'raster.par/contract 'C [['bb B] ['i M] ['j N]] [['l K]]
                       (list '* (list 'aget 'A (list '+ (list '* (list '+ (list '* 'bb M) 'i) K) 'l))
                             (list 'aget 'B (list '+ (list '* (list '+ (list '* 'bb K) 'l) N) 'j))))
            r (route/route-contraction form :dtype :double)
            ze (find-ns 'raster.gpu.ze-runtime)
            abuf ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) (* B M K) :double) Ad)
            bbuf ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) (* B K N) :double) Bd)
            gpu (launch-1d-routed r {'A abuf 'B bbuf} (out-f64 (* B M N)) (* B M N))
            cpu (vec (for [bb (range B) i (range M) j (range N)]
                       (reduce + (for [l (range K)]
                                   (* (aget Ad (+ (* (+ (* bb M) i) K) l))
                                      (aget Bd (+ (* (+ (* bb K) l) N) j)))))))]
        (is (= :naive-segred (:strategy r)))
        (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-9) gpu cpu)))))))

(deftest a1c-multi-contract-flattens-and-routes-to-naive-segred
  (if-not @gpu?
    (println "[skip] a1c-multi-contract: no GPU")
    (testing "2 contract axes → flattened → :naive-segred; C[i]=Σ_{l1,l2} A[i,l1,l2]·V[l1,l2] == ref"
      (let [I 4
            A (double-array (map #(* 0.1 (double %)) (range (* I 2 3))))
            V (double-array (map #(* 0.5 (double %)) (range 6)))
            form (list 'raster.par/contract 'C [['i I]] [['l1 2] ['l2 3]]
                       (list '* (list 'aget 'A (list '+ (list '* 'i 6) (list '* 'l1 3) 'l2))
                             (list 'aget 'V (list '+ (list '* 'l1 3) 'l2))))
            r (route/route-contraction form :dtype :double)
            ze (find-ns 'raster.gpu.ze-runtime)
            abuf ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) (* I 2 3) :double) A)
            vbuf ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) 6 :double) V)
            gpu (launch-1d-routed r {'A abuf 'V vbuf} (out-f64 I) I)
            cpu (vec (for [i (range I)]
                       (reduce + (for [l1 (range 2) l2 (range 3)]
                                   (* (aget A (+ (* i 6) (* l1 3) l2)) (aget V (+ (* l1 3) l2)))))))]
        (is (= :naive-segred (:strategy r)))
        (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-9) gpu cpu)))))))

;; ── A3: output-axis permutation falls out of free-axis DECLARATION ORDER ──────────────
(deftest a3-transposed-output-falls-out-of-free-axis-order
  (if-not @gpu?
    (println "[skip] a3-transposed-output: no GPU")
    (testing "einsum ij,jk->ki (transposed output) is correct by declaring free-axes [k i]"
      ;; A[M,K]·B[K,N] but output C[N,M] = (A·B)ᵀ: declare free in output order (kk outer, ii inner).
      ;; The emitter's row-major write over declared free axes yields C[k,i] with no store change.
      (let [M 3 K 4 N 2
            Ad (double-array (map #(* 0.1 (double %)) (range (* M K))))
            Bd (double-array (map #(* 0.2 (double %)) (range (* K N))))
            form (list 'raster.par/contract 'C [['kk N] ['ii M]] [['jj K]]
                       (list '* (list 'aget 'A (list '+ (list '* 'ii K) 'jj))
                             (list 'aget 'B (list '+ (list '* 'jj N) 'kk))))
            r (route/route-contraction form :dtype :double)
            ze (find-ns 'raster.gpu.ze-runtime)
            abuf ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) (* M K) :double) Ad)
            bbuf ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) (* K N) :double) Bd)
            gpu (launch-routed r {'A abuf 'B bbuf} (out-f64 (* N M)))
            cpu (vec (for [kk (range N) ii (range M)]
                       (reduce + (for [jj (range K)] (* (aget Ad (+ (* ii K) jj)) (aget Bd (+ (* jj N) kk)))))))]
        (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-9) gpu cpu)))))))

;; ── C1-nt: int8 routes to the quant leaves (dp4a for :nt, quant-naive for :nn) ─────────
(defn- launch-int8-routed
  "Launch a routed int8 descriptor. `scalars` supplies a value per `:epilogue-scalars`, bound in
   SLOT ORDER: operands…, out, epilogue scalars…, trailing count. (That order is the descriptor's
   contract; the ABI datum in the plan makes it data rather than a convention each caller repeats.)"
  ([r bufs] (launch-int8-routed r bufs {}))
  ([r bufs scalars]
   (let [ze (find-ns 'raster.gpu.ze-runtime)
         [gx gy] (:grid r)
         o ((ns-resolve ze 'make-buffer) (:out-elems r) :float)]
     ((ns-resolve ze 'register-kernel!) (:kernel-name r) {:source (:source r) :dtype :byte})
     (let [{:keys [kernel-handle]} ((ns-resolve ze 'ensure-kernel-loaded!) (:kernel-name r))
           args (-> (mapv #(:segment (get bufs %)) (:array-params r))
                    (conj (:segment o))
                    (into (mapv (fn [sym] {:type :float :value (float (get scalars sym))})
                                (:epilogue-scalars r)))
                    (into (:scalar-args r)))]
       ((ns-resolve ze 'launch-2d!) kernel-handle (:wg r) [gx gy] args)
       (vec ((ns-resolve ze 'buffer->float-array) o))))))

(defn- mk-i8 [xs] (let [ze (find-ns 'raster.gpu.ze-runtime)]
                    ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) (count xs) :byte)
                                                     (byte-array (map byte xs)))))

(deftest c1-int8-routes-to-quant-leaves
  (if-not @gpu?
    (println "[skip] c1-int8-quant-routing: no GPU")
    (let [M 4 K 8 N 4 scale 0.01
          Ab (mapv #(byte (- (mod % 255) 127)) (range (* M K)))
          Bv (mapv #(byte (- (mod (* 3 %) 255) 127)) (range (* N K)))  ; reused as [N,K] and [K,N]
          Aget (fn [i] (nth Ab i)) Bget (fn [i] (nth Bv i))
          close? (fn [xs ys] (every? true? (map #(< (/ (Math/abs (- (double %1) (double %2)))
                                                       (max 1.0 (Math/abs (double %2)))) 1.0e-6)
                                                xs ys)))]
      (testing ":nt (B stored [N,K]) → :dp4a peak leaf; int8 matmul dequant == reference"
        (let [form (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
                         (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
                               (list 'aget 'B (list '+ (list '* 'j K) 'l)))
                         :epilogue {:acc 'acc :expr '(raster.numeric/* acc s)
                                    :scalars [{:sym 's :dtype :float}]})
              r (route/route-contraction form :dtype :byte)
              gpu (launch-int8-routed r {'A (mk-i8 Ab) 'B (mk-i8 Bv)} {'s scale})
              cpu (for [i (range M) j (range N)]
                    (* scale (reduce + (for [l (range K)] (* (int (Aget (+ (* i K) l))) (int (Bget (+ (* j K) l))))))))]
          (is (= :dp4a (:strategy r)))
          (is (close? gpu cpu))))
      (testing ":nn (B stored [K,N]) → :quant-naive widening leaf; == reference"
        (let [form (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
                         (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
                               (list 'aget 'B (list '+ (list '* 'l N) 'j)))
                         :epilogue {:acc 'acc :expr '(raster.numeric/* acc s)
                                    :scalars [{:sym 's :dtype :float}]})
              r (route/route-contraction form :dtype :byte)
              gpu (launch-int8-routed r {'A (mk-i8 Ab) 'B (mk-i8 Bv)} {'s scale})
              cpu (for [i (range M) j (range N)]
                    (* scale (reduce + (for [l (range K)] (* (int (Aget (+ (* i K) l))) (int (Bget (+ (* l N) j))))))))]
          (is (= :quant-naive (:strategy r)))
          (is (close? gpu cpu)))))))

;; ── B3-insert (Option 1): :nn int8 reaches the dp4a PEAK leaf via an inserted transpose ──
(defn- exec-pre-step [bufs {:keys [src dst rows cols dtype]}]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        d ((ns-resolve ze 'make-buffer) (* rows cols) dtype)
        g ((ns-resolve ze 'record-graph!)
           [{:bound ((ns-resolve ze 'bind-registered-transpose!) (get bufs src) d rows cols dtype)
             :kernel-name "t"}])]
    ((ns-resolve ze 'replay-graph!) g)
    (assoc bufs dst d)))

(deftest b3insert-nn-int8-reaches-dp4a-via-transpose
  (if-not @gpu?
    (println "[skip] b3insert-nn-dp4a: no GPU")
    (testing ":nn int8 matmul + :prefer-peak? → transpose pre-step + dp4a; == reference on device"
      (let [M 4 K 8 N 4 scale 0.01
            Ab (mapv #(byte (- (mod % 255) 127)) (range (* M K)))
            Bnn (mapv #(byte (- (mod (* 3 %) 255) 127)) (range (* K N)))   ; [K,N] (:nn)
            form (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
                       (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
                             (list 'aget 'B (list '+ (list '* 'l N) 'j)))
                       :epilogue {:acc 'acc :expr '(raster.numeric/* acc s)
                                  :scalars [{:sym 's :dtype :float}]})
            r (route/route-contraction form :dtype :byte :prefer-peak? true)
            bufs (reduce exec-pre-step {'A (mk-i8 Ab) 'B (mk-i8 Bnn)} (:pre-steps r))
            gpu (launch-int8-routed r bufs {'s scale})
            cpu (for [i (range M) j (range N)]
                  (* scale (reduce + (for [l (range K)] (* (int (nth Ab (+ (* i K) l))) (int (nth Bnn (+ (* l N) j))))))))]
        (is (= :dp4a (:strategy r)))
        (is (= 1 (count (:pre-steps r))))
        (is (= :byte (:dtype (first (:pre-steps r)))))            ; byte-granularity transpose
        (is (every? true? (map #(< (/ (Math/abs (- (double %1) (double %2)))
                                      (max 1.0 (Math/abs (double %2)))) 1.0e-6) gpu cpu)))))))

(deftest b3insert-refuses-to-retarget-an-unverified-layout
  ;; device-free. The previous :nn→:nt rewrite substituted ASSUMED canonical strides having only
  ;; checked which axis SYMBOLS appeared, so an operand with different strides silently computed
  ;; the wrong result. The map-based retarget VERIFIES the actual index against the layout first.
  (testing "col operand with non-canonical strides is NOT retargeted (falls back)"
    (let [M 4 N 4 K 8
          ;; B indexed as (l + j*K*2) — same symbols, WRONG strides for a [K,N] operand
          bad (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
                    (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
                          (list 'aget 'B (list '+ (list '* 'j (* K 2)) 'l))))
          r (try (route/route-contraction bad :dtype :byte :prefer-peak? true)
                 (catch clojure.lang.ExceptionInfo _ {:strategy :rejected :pre-steps []}))]
      (is (not= :dp4a (:strategy r)) "must not claim the peak leaf on an unverified layout")
      (is (empty? (:pre-steps r)))))
  (testing "an operand layout no PEAK leaf can index now falls back to the scalar nest, which has
            no layout requirement at all — it emits the declared body verbatim, so the result is
            correct rather than refused. (It used to throw \"no quant leaf handles\"; that message
            belonged to a hand-written orientation gate on a leaf that no longer exists.)"
    (let [M 4 N 4 K 8
          bad (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
                    (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
                          (list 'aget 'B (list '+ (list '* 'j (* K 2)) 'l))))
          r (route/route-contraction bad :dtype :byte)]
      (is (= :quant-naive (:strategy r)))
      (is (not (:tensorized r)))
      ;; and the emitted body indexes B exactly as declared — stride 2K, not an assumed K
      (is (re-find #"B\[\(\(j \* 16\) \+ l\)\]" (:source r)))))
  (testing "the canonical :nn form IS retargeted (transpose pre-step + dp4a)"
    (let [M 4 N 4 K 8
          ok (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
                   (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
                         (list 'aget 'B (list '+ (list '* 'l N) 'j))))
          r (route/route-contraction ok :dtype :byte :prefer-peak? true)]
      (is (= :dp4a (:strategy r)))
      (is (= 1 (count (:pre-steps r)))))))

(deftest all-routing-decisions-are-device-free
  ;; The strategy choices are pure emit — they must be checkable without a GPU, so CI validates
  ;; the routing brain even where it cannot run kernels. (Previously every strategy assertion sat
  ;; inside an `if-not @gpu?` guard and was skipped entirely off-device.)
  (let [strategy (fn [form & opts] (:strategy (apply route/route-contraction form opts)))]
    (testing "0 contract axes → :segmap"
      (is (= :segmap (strategy '(raster.par/contract C [[i 4] [j 3]] []
                                                     (* (aget a i) (aget b j))) :dtype :double))))
    (testing "n free ≠ 2 → :naive-segred"
      (is (= :naive-segred (strategy '(raster.par/contract C [[b 2] [i 4] [j 3]] [[l 5]]
                                                           (* (aget A x) (aget B y))) :dtype :double))))
    (testing "n ≥ 2 contract axes → :naive-segred (flattened)"
      (is (= :naive-segred (strategy '(raster.par/contract C [[i 4]] [[l1 2] [l2 3]]
                                                           (* (aget A x) (aget V y))) :dtype :double))))
    (testing "2 free + 1 contract: f16 canonical → :dpas, f64 → :regtiled"
      (is (= :dpas (strategy (matmul-form 128 128 128) :dtype :half)))
      (is (= :regtiled (strategy (matmul-form 128 128 128) :dtype :double))))
    (testing "int8 :nt → :dp4a, :nn → :quant-naive, :nn + prefer-peak? → :dp4a + transpose"
      (let [nt '(raster.par/contract C [[i 4] [j 4]] [[l 8]]
                                     (* (aget A (+ (* i 8) l)) (aget B (+ (* j 8) l))))
            nn '(raster.par/contract C [[i 4] [j 4]] [[l 8]]
                                     (* (aget A (+ (* i 8) l)) (aget B (+ (* l 4) j))))]
        (is (= :dp4a (strategy nt :dtype :byte)))
        (is (= :quant-naive (strategy nn :dtype :byte)))
        (let [r (route/route-contraction nn :dtype :byte :prefer-peak? true)]
          (is (= :dp4a (:strategy r)))
          (is (= :byte (:dtype (first (:pre-steps r))))))))
    (testing "every descriptor carries the full launch contract"
      (doseq [[label r] [[:segmap (route/route-contraction
                                   '(raster.par/contract C [[i 4] [j 3]] [] (* (aget a i) (aget b j)))
                                   :dtype :double)]
                         [:dpas (route/route-contraction (matmul-form 128 128 128) :dtype :half)]
                         [:regtiled (route/route-contraction (matmul-form 128 128 128) :dtype :double)]]]
        (is (every? some? [(:kernel-name r) (:source r) (:array-params r) (:dtype r)
                           (:out-dtype r) (:out-elems r) (:wg r) (:grid r) (:scalar-args r)])
            (str label " descriptor incomplete"))))))

;; ── Q1: the emitted tile is DERIVED from the hardware descriptor, never hardcoded ─────
;; The DPAS leaf previously hardcoded {128 128 32 32 32, subgroup 16}. That constant happens to
;; equal the Arc 140V derivation, so it worked here and no test caught it — but on a part with a
;; different GRF budget / subgroup size / matrix shape it would emit a wrong-for-the-hardware
;; tile while the production GEMM path adapted correctly. These are the tests that catch that.
(deftest tile-is-derived-from-the-hardware-descriptor
  (let [mm (matmul-form 256 512 128)
        hw (requiring-resolve 'raster.compiler.core.hardware/derive-gemm-tile)]
    (testing "no descriptor ⇒ hw/derive-gemm-tile's own defaults (the Arc config) — a NO-OP swap"
      (let [r (route/route-contraction mm :dtype :half)]
        (is (= (hw {}) (:tile r)))
        (is (= {:block-m 128 :block-n 128 :sg-m 32 :sg-n 32 :block-k 32}
               (select-keys (:tile r) [:block-m :block-n :sg-m :sg-n :block-k]))
            "must still reproduce the hand-tuned Arc tile")
        (is (= [256 1] (:wg r)))
        (is (= [4 2] (:grid r)))))                     ; ceil(512/128), ceil(256/128)
    (testing "a part with HALF the GRF and subgroup 8 gets a rescaled tile AND launch geometry"
      (let [small {:matrix {:m 8 :n 8 :k 16 :subgroup 8} :grf-bytes-per-lane 128}
            r (route/route-contraction mm :dtype :half :desc small)]
        (is (= {:block-m 64 :block-n 64 :sg-m 16 :sg-n 16 :block-k 32}
               (select-keys (:tile r) [:block-m :block-n :sg-m :sg-n :block-k])))
        (is (= [128 1] (:wg r)) "workgroup = (bm/sgm)·(bn/sgn)·subgroup = 4·4·8")
        (is (= [8 4] (:grid r)) "grid follows the smaller block tile")))
    (testing "an explicit tile (e.g. an autotune result) overrides the derivation"
      (let [t (hw {} {:wg-subgroups 4})
            r (route/route-contraction mm :dtype :half :tile t)]
        (is (= t (:tile r)))
        (is (not= (:block-m (hw {})) (:block-m t)) "the override must actually differ")))
    (testing "the emitted kernel source carries the derived tile, not a constant"
      (let [small {:matrix {:m 8 :n 8 :k 16 :subgroup 8} :grf-bytes-per-lane 128}
            src (:source (route/route-contraction mm :dtype :half :desc small))]
        (is (re-find #"intel_reqd_sub_group_size\(8\)" src)
            "subgroup size must follow the descriptor into the kernel attribute")))))
