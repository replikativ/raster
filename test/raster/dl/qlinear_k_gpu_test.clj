(ns raster.dl.qlinear-k-gpu-test
  "The SAME composable K-quant GEMV deftms (qmatmul-q{4,6}k-gpu!) compile to OpenCL via the
   shared c_emit and run on the GPU (ze:0), byte-exact with the CPU composable kernel on
   identical quantized inputs. Proves the format registry reaches a working GPU kernel through
   the work-item-per-row par/map-void! twin — int8 weights + float scales + int bsums in one
   mixed-dtype kernel. Skipped when no GPU device is present."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.qlinear-k :as qk]
            [raster.dl.nn :as nn]
            [raster.compiler.backend.cpu.quant :as q]
            [raster.par :as par]
            [raster.gpu.core :as gpu]))

(defn- pack-i8 ^ints [^bytes b]
  (let [w (quot (alength b) 4) out (int-array w)]
    (dotimes [i w]
      (let [j (* i 4)]
        (aset out i (unchecked-int
                     (bit-or (bit-and (int (aget b j)) 0xFF)
                             (bit-shift-left (bit-and (int (aget b (+ j 1))) 0xFF) 8)
                             (bit-shift-left (bit-and (int (aget b (+ j 2))) 0xFF) 16)
                             (bit-shift-left (bit-and (int (aget b (+ j 3))) 0xFF) 24))))))
    out))

(defn- bytes->ints-le ^ints [^bytes b]      ; reinterpret bytes as int32 little-endian
  (let [w (quot (alength b) 4) out (int-array w)
        bb (.order (java.nio.ByteBuffer/wrap b) java.nio.ByteOrder/LITTLE_ENDIAN)]
    (dotimes [i w] (aset out i (.getInt bb (* i 4))))
    out))

(defn- rand-i8 ^bytes [n seed]
  (let [a (byte-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (byte (- (.nextInt r 255) 127)))) a))

(defn- gpu-available? []
  (try (require 'raster.gpu.ze-runtime)
       (let [qfn (resolve 'raster.gpu.ze-runtime/query-devices)]
         (boolean (and qfn (seq (qfn)))))
       (catch Throwable _ false)))

(defn- gen [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5)))) a))

(defn- maxerr [^floats a ^floats b]
  (reduce max 0.0 (map (fn [x y] (Math/abs (double (- x y)))) (seq a) (seq b))))

(deftest rms-norm-gpu-lowers
  (when (gpu-available?)
    (testing "the existing typed library deftm rms-norm! lowers to a correct OpenCL kernel"
      ;; Proves the GPU emitter reads the deftm's DECLARED types (Double eps/gain-offset stay
      ;; float not int-by-name-regex; Long features types the index integer), types loop recur
      ;; counters from var-types (int, not float), and enables fp64 when the body uses double.
      ;; A real library op (par/map-void! over rows, reduce+map inside), not a hand-shaped kernel.
      (let [rows 4 feat 256
            x (gen (* rows feat) 7) w (gen feat 8)
            ycpu (nn/rms-norm x w rows feat 1.0e-6 1.0)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :rn #'nn/rms-norm!)
          (gpu/alloc! sess {:x [:float (alength x) x] :w [:float feat w] :out [:float (* rows feat) nil]})
          (gpu/prepare! sess :rn {"x" :x "weight" :w "out" :out}
                        [{:type :float :value 1.0e-6} {:type :int :value feat} {:type :float :value 1.0}]
                        rows {:kernel-phase :rn})  ; scalars ordered by name: eps, features, gain-offset
          (gpu/invoke-bound! sess :rn)
          (is (< (maxerr ycpu (gpu/download sess :out)) 1e-3))
          (finally (gpu/close-session! sess)))))))

(deftest elementwise-ffin-ops-gpu-lower
  (when (gpu-available?)
    (testing "silu-mul! (SwiGLU) and residual-add! par/map-void! ops lower to correct OpenCL"
      (let [n 512 gate (gen n 1) up (gen n 2) a (gen n 3) b (gen n 4)
            sig (fn [^double v] (/ 1.0 (+ 1.0 (Math/exp (- v)))))
            sm-ref (let [o (float-array n)] (dotimes [i n] (aset o i (float (* (* (aget gate i) (sig (aget gate i))) (aget up i))))) o)
            ra-ref (let [o (float-array n)] (dotimes [i n] (aset o i (float (+ (aget a i) (aget b i))))) o)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :sm #'nn/silu-mul!)
          (gpu/compile! sess :ra #'nn/residual-add!)
          (gpu/alloc! sess {:gate [:float n gate] :up [:float n up] :smo [:float n nil]
                            :a [:float n a] :b [:float n b] :rao [:float n nil]})
          (gpu/prepare! sess :sm {"gate" :gate "up" :up "out" :smo} [] n {:kernel-phase :sm})
          (gpu/prepare! sess :ra {"a" :a "b" :b "out" :rao} [] n {:kernel-phase :ra})
          (gpu/invoke-bound! sess :sm) (gpu/invoke-bound! sess :ra) (gpu/sync! sess)
          (is (< (maxerr sm-ref (gpu/download sess :smo)) 1e-4) "silu-mul")
          (is (< (maxerr ra-ref (gpu/download sess :rao)) 1e-5) "residual-add")
          (finally (gpu/close-session! sess)))))))

(deftest dp4a-jvm-reference
  (testing "par/dp4a matches scalar 4-lane signed int8 dot (little-endian lanes)"
    ;; bytes (4,3,2,1)·(1,1,1,1) = 10
    (is (= 10 (int (par/dp4a (unchecked-int 0x01020304) (unchecked-int 0x01010101) 0))))
    ;; signed lanes + nonzero acc
    (is (= 100 (int (par/dp4a (unchecked-int 0x01000000) (unchecked-int 0x05000000) 95)))) ; 1*5+95
    (is (= -2 (int (par/dp4a (unchecked-int 0x000000FF) (unchecked-int 0x00000002) 0)))))) ; (-1)*2

(deftest i8gemv-dp4a-gpu-exact
  (when (gpu-available?)
    (testing "dp4a int8 GEMV core on ze:0 is exact vs scalar int8 dot"
      (let [out 16 in 256 kw (quot in 4)
            W (rand-i8 (* out in) 7) x (rand-i8 in 9)
            wp (pack-i8 W) xp (pack-i8 x)
            yref (int-array out)
            _ (dotimes [o out]
                (aset yref o (int (areduce x k s 0 (+ s (* (int (aget W (+ (* o in) k))) (int (aget x k))))))))
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :g #'qk/i8gemv-dp4a!)
          (gpu/alloc! sess {:wp [:int (alength wp) wp] :xp [:int (alength xp) xp] :y [:float out nil]})
          (gpu/invoke! sess :g {"wp" :wp "xp" :xp "y" :y} [{:type :int :value kw}] out)
          (let [ygpu (gpu/download sess :y)]
            (is (every? (fn [o] (= (int (aget yref o)) (int (aget ^floats ygpu o)))) (range out))))
          (finally (gpu/close-session! sess)))))))

(deftest q4k-gpu-matches-cpu
  (when (gpu-available?)
    (testing "Q4_K work-item-per-row kernel on ze:0 == CPU composable"
      (let [out 32 in 256
            W (gen (* out in) 11) x (gen in 22)
            {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
            {:keys [xq xs bsums]}    (q/quantize-act-q8k x in q/q4-K)
            ycpu (float-array out)
            _ (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq ycpu in out 0 out)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :q4k #'qk/qmatmul-q4k-gpu!)
          (gpu/alloc! sess {:xq [:byte (alength xq) xq] :xs [:float (alength xs) xs]
                            :bsums [:int (alength bsums) bsums] :wq [:byte (alength wq) wq]
                            :da [:float (alength da) da] :db [:float (alength db) db]
                            :aq [:byte (alength aq) aq] :bq [:byte (alength bq) bq]
                            :y [:float out nil]})
          (gpu/invoke! sess :q4k
                       {"xq" :xq "xs" :xs "bsums" :bsums "wq" :wq "da" :da "db" :db
                        "aq" :aq "bq" :bq "y" :y}
                       [{:type :int :value in}] out)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3))
          (finally (gpu/close-session! sess)))))))

(deftest q4k-dp4a-bound-dispatch
  (when (gpu-available?)
    (testing "prepare!/invoke-bound! (sync) and async prepare!+sync! match invoke! result"
      (let [out 32 in 256
            W (gen (* out in) 11) x (gen in 22)
            {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
            {:keys [xq xs bsums]}    (q/quantize-act-q8k x in q/q4-K)
            wp (bytes->ints-le wq) xp (pack-i8 xq)
            ycpu (float-array out)
            _ (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq ycpu in out 0 out)
            sym {"xp" :xp "xs" :xs "bsums" :bsums "wp" :wp "da" :da "db" :db "aq" :aq "bq" :bq "y" :y}
            scal [{:type :int :value in}]
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :q4kdp #'qk/qmatmul-q4k-dp4a!)
          (gpu/alloc! sess {:xp [:int (alength xp) xp] :xs [:float (alength xs) xs]
                            :bsums [:int (alength bsums) bsums] :wp [:int (alength wp) wp]
                            :da [:float (alength da) da] :db [:float (alength db) db]
                            :aq [:byte (alength aq) aq] :bq [:byte (alength bq) bq]
                            :y [:float out nil]})
          ;; synchronous bound dispatch
          (gpu/prepare! sess :q4kdp sym scal out)
          (gpu/invoke-bound! sess :q4kdp)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3) "sync bound")
          ;; async bound dispatch: zero result, batch-launch, sync, then read
          (gpu/prepare! sess :q4kdp sym scal out {:async? true})
          (dotimes [_ 3] (gpu/invoke-bound! sess :q4kdp))
          (gpu/sync! sess)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3) "async bound + sync!")
          (finally (gpu/close-session! sess)))))))

(deftest q4k-dp4a-multi-binding
  (when (gpu-available?)
    (testing "two bindings of the SAME kernel with different buffers don't clobber each other"
      ;; Regression: bind-kernel! sets args on the kernel handle (mutable); reusing one shared
      ;; handle made the last prepare! win, so every invoke ran the last binding's args. Each
      ;; binding must get its own handle. Two distinct weights → two y buffers, both must match.
      (let [out 32 in 256
            mk (fn [seed-w seed-x]
                 (let [W (gen (* out in) seed-w) x (gen in seed-x)
                       {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
                       {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q4-K)
                       ycpu (float-array out)]
                   (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq ycpu in out 0 out)
                   {:wp (bytes->ints-le wq) :da da :db db :aq aq :bq bq
                    :xp (pack-i8 xq) :xs xs :bsums bsums :ycpu ycpu}))
            a (mk 11 22) b (mk 99 88)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :mm #'qk/qmatmul-q4k-dp4a!)
          (gpu/alloc! sess {:axp [:int (alength ^ints (:xp a)) (:xp a)] :axs [:float (alength ^floats (:xs a)) (:xs a)] :abs [:int (alength ^ints (:bsums a)) (:bsums a)]
                            :awp [:int (alength ^ints (:wp a)) (:wp a)] :ada [:float (alength ^floats (:da a)) (:da a)] :adb [:float (alength ^floats (:db a)) (:db a)] :aaq [:byte (alength ^bytes (:aq a)) (:aq a)] :abq [:byte (alength ^bytes (:bq a)) (:bq a)] :ay [:float out nil]
                            :bxp [:int (alength ^ints (:xp b)) (:xp b)] :bxs [:float (alength ^floats (:xs b)) (:xs b)] :bbs [:int (alength ^ints (:bsums b)) (:bsums b)]
                            :bwp [:int (alength ^ints (:wp b)) (:wp b)] :bda [:float (alength ^floats (:da b)) (:da b)] :bdb [:float (alength ^floats (:db b)) (:db b)] :baq [:byte (alength ^bytes (:aq b)) (:aq b)] :bbq [:byte (alength ^bytes (:bq b)) (:bq b)] :by [:float out nil]})
          (gpu/prepare! sess :a {"xp" :axp "xs" :axs "bsums" :abs "wp" :awp "da" :ada "db" :adb "aq" :aaq "bq" :abq "y" :ay} [{:type :int :value in}] out {:kernel-phase :mm})
          (gpu/prepare! sess :b {"xp" :bxp "xs" :bxs "bsums" :bbs "wp" :bwp "da" :bda "db" :bdb "aq" :baq "bq" :bbq "y" :by} [{:type :int :value in}] out {:kernel-phase :mm})
          (gpu/invoke-bound! sess :a)
          (gpu/invoke-bound! sess :b)
          (is (< (maxerr (:ycpu a) (gpu/download sess :ay)) 1e-3) "binding A correct")
          (is (< (maxerr (:ycpu b) (gpu/download sess :by)) 1e-3) "binding B correct (not clobbered by A)")
          (finally (gpu/close-session! sess)))))))

(deftest q8k-quant-then-matmul-chain
  (when (gpu-available?)
    (testing "GPU q8_K activation quantizer feeds dp4a matmul, all on-device in one graph"
      ;; The decode enabler: quantize the float activation ON the GPU (quant-act-q8k-gpu!),
      ;; then the matmul consumes xp/xs/bsums with NO host round-trip. Recorded as a 2-op graph
      ;; (barrier enforces quant→matmul order). Result must match the CPU dequant reference.
      (let [out 32 in 256 nsb (quot in 256)
            W (gen (* out in) 5) x (gen in 6)
            {:keys [wq da db aq bq] :as ew} (q/quantize-weight-q4k W q/q4-K)
            wp (bytes->ints-le wq)
            ;; CPU reference: quantize act, dequant act + weights, plain matmul
            {:keys [xq xs]} (q/quantize-act-q8k x in q/q4-K)
            Wdq (q/dequant-q4k ew q/q4-K (* out in))
            xdq (let [d (float-array in) dact (aget ^floats xs 0)]
                  (dotimes [k in] (aset d k (float (* dact (aget ^bytes xq k))))) d)
            yref (let [y (float-array out)]
                   (dotimes [o out] (aset y o (float (areduce xdq k s 0.0 (+ s (* (aget Wdq (+ (* (long o) (long in)) k)) (aget xdq k)))))))
                   y)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :quant #'qk/quant-act-q8k-gpu!)
          (gpu/compile! sess :mm #'qk/qmatmul-q4k-dp4a!)
          (gpu/alloc! sess {:x [:float in x]
                            :xp [:int (quot in 4) nil] :xs [:float nsb nil] :bsums [:int (quot in 32) nil]
                            :wp [:int (alength wp) wp] :da [:float (alength da) da] :db [:float (alength db) db]
                            :aq [:byte (alength aq) aq] :bq [:byte (alength bq) bq] :y [:float out nil]})
          (gpu/prepare! sess :quant {"x" :x "xp" :xp "xs" :xs "bsums" :bsums} [] nsb {:kernel-phase :quant})
          (gpu/prepare! sess :mm {"xp" :xp "xs" :xs "bsums" :bsums "wp" :wp "da" :da "db" :db "aq" :aq "bq" :bq "y" :y}
                        [{:type :int :value in}] out {:kernel-phase :mm})
          (gpu/record-graph! sess [:quant :mm])
          (gpu/replay! sess)
          (let [ygpu (gpu/download sess :y)
                scale (reduce max 1e-9 (map (fn [v] (Math/abs (double v))) (seq yref)))]
            (is (< (/ (maxerr yref ygpu) scale) 1e-2) "on-device quant→matmul == CPU dequant ref"))
          (finally (gpu/close-session! sess)))))))

(deftest q4k-dp4a-command-graph
  (when (gpu-available?)
    (testing "record-graph!/replay! runs a fixed kernel sequence; each op reads its own buffers"
      (let [out 32 in 256
            mk (fn [sw sx]
                 (let [W (gen (* out in) sw) x (gen in sx)
                       {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
                       {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q4-K)
                       y (float-array out)]
                   (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq y in out 0 out)
                   {:wp (bytes->ints-le wq) :da da :db db :aq aq :bq bq :xp (pack-i8 xq) :xs xs :bsums bsums :y y}))
            a (mk 11 22) b (mk 77 88)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :mm #'qk/qmatmul-q4k-dp4a!)
          (gpu/alloc! sess {:axp [:int (alength ^ints (:xp a)) (:xp a)] :axs [:float (alength ^floats (:xs a)) (:xs a)] :abs [:int (alength ^ints (:bsums a)) (:bsums a)] :awp [:int (alength ^ints (:wp a)) (:wp a)] :ada [:float (alength ^floats (:da a)) (:da a)] :adb [:float (alength ^floats (:db a)) (:db a)] :aaq [:byte (alength ^bytes (:aq a)) (:aq a)] :abq [:byte (alength ^bytes (:bq a)) (:bq a)] :ay [:float out nil]
                            :bxp [:int (alength ^ints (:xp b)) (:xp b)] :bxs [:float (alength ^floats (:xs b)) (:xs b)] :bbs [:int (alength ^ints (:bsums b)) (:bsums b)] :bwp [:int (alength ^ints (:wp b)) (:wp b)] :bda [:float (alength ^floats (:da b)) (:da b)] :bdb [:float (alength ^floats (:db b)) (:db b)] :baq [:byte (alength ^bytes (:aq b)) (:aq b)] :bbq [:byte (alength ^bytes (:bq b)) (:bq b)] :by [:float out nil]})
          (gpu/prepare! sess :a {"xp" :axp "xs" :axs "bsums" :abs "wp" :awp "da" :ada "db" :adb "aq" :aaq "bq" :abq "y" :ay} [{:type :int :value in}] out {:kernel-phase :mm})
          (gpu/prepare! sess :b {"xp" :bxp "xs" :bxs "bsums" :bbs "wp" :bwp "da" :bda "db" :bdb "aq" :baq "bq" :bbq "y" :by} [{:type :int :value in}] out {:kernel-phase :mm})
          (gpu/record-graph! sess [:a :b])
          (gpu/replay! sess)
          (is (< (maxerr (:y a) (gpu/download sess :ay)) 1e-3) "graph op A")
          (is (< (maxerr (:y b) (gpu/download sess :by)) 1e-3) "graph op B")
          ;; replay again (idempotent, reads current buffers) — still correct
          (gpu/replay! sess)
          (is (< (maxerr (:y b) (gpu/download sess :by)) 1e-3) "graph replay idempotent")
          (finally (gpu/close-session! sess)))))))

(deftest q4k-dp4a-gpu-matches-cpu
  (when (gpu-available?)
    (testing "Q4_K dp4a kernel on ze:0 (int32-packed nibble-mask trick) == CPU composable"
      (let [out 32 in 256
            W (gen (* out in) 11) x (gen in 22)
            {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
            {:keys [xq xs bsums]}    (q/quantize-act-q8k x in q/q4-K)
            ycpu (float-array out)
            _ (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq ycpu in out 0 out)
            wp (bytes->ints-le wq) xp (pack-i8 xq)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :q4kdp #'qk/qmatmul-q4k-dp4a!)
          (gpu/alloc! sess {:xp [:int (alength xp) xp] :xs [:float (alength xs) xs]
                            :bsums [:int (alength bsums) bsums] :wp [:int (alength wp) wp]
                            :da [:float (alength da) da] :db [:float (alength db) db]
                            :aq [:byte (alength aq) aq] :bq [:byte (alength bq) bq]
                            :y [:float out nil]})
          (gpu/invoke! sess :q4kdp
                       {"xp" :xp "xs" :xs "bsums" :bsums "wp" :wp "da" :da "db" :db
                        "aq" :aq "bq" :bq "y" :y}
                       [{:type :int :value in}] out)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3))
          (finally (gpu/close-session! sess)))))))

(deftest q6k-dp4a-gpu-matches-cpu
  (when (gpu-available?)
    (testing "Q6_K dp4a kernel on ze:0 == CPU composable"
      (let [out 32 in 256
            W (gen (* out in) 33) x (gen in 44)
            {:keys [wq sc ds]}   (q/quantize-weight-q6k W q/q6-K)
            {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q6-K)
            ycpu (float-array out)
            _ (qk/qmatmul-q6k-composable! xq xs bsums wq sc ds ycpu in out 0 out)
            wp (bytes->ints-le wq) xp (pack-i8 xq)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :q6kdp #'qk/qmatmul-q6k-dp4a!)
          (gpu/alloc! sess {:xp [:int (alength xp) xp] :xs [:float (alength xs) xs]
                            :bsums [:int (alength bsums) bsums] :wp [:int (alength wp) wp]
                            :sc [:byte (alength sc) sc] :ds [:float (alength ds) ds]
                            :y [:float out nil]})
          (gpu/invoke! sess :q6kdp
                       {"xp" :xp "xs" :xs "bsums" :bsums "wp" :wp "sc" :sc "ds" :ds "y" :y}
                       [{:type :int :value in}] out)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3))
          (finally (gpu/close-session! sess)))))))

(deftest q6k-gpu-matches-cpu
  (when (gpu-available?)
    (testing "Q6_K work-item-per-row kernel on ze:0 == CPU composable"
      (let [out 32 in 256
            W (gen (* out in) 33) x (gen in 44)
            {:keys [wq sc ds]}   (q/quantize-weight-q6k W q/q6-K)
            {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q6-K)
            ycpu (float-array out)
            _ (qk/qmatmul-q6k-composable! xq xs bsums wq sc ds ycpu in out 0 out)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :q6k #'qk/qmatmul-q6k-gpu!)
          (gpu/alloc! sess {:xq [:byte (alength xq) xq] :xs [:float (alength xs) xs]
                            :bsums [:int (alength bsums) bsums] :wq [:byte (alength wq) wq]
                            :sc [:byte (alength sc) sc] :ds [:float (alength ds) ds]
                            :y [:float out nil]})
          (gpu/invoke! sess :q6k
                       {"xq" :xq "xs" :xs "bsums" :bsums "wq" :wq "sc" :sc "ds" :ds "y" :y}
                       [{:type :int :value in}] out)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3))
          (finally (gpu/close-session! sess)))))))
