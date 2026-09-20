(ns raster.quant.kernels-k-test
  "The composable K-quant GEMV deftms compile to C (compile-aot :target :c) and match the
   dequant-matmul reference — proving the registry's formats reach a working C kernel via
   the SAME composable path the legacy Q4_0 uses (and that the GPU/OpenCL path will reuse).
   Single-call correctness; no spin-pool, no Valhalla."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.report :as report]
            [raster.quant.kernels-k :as qk]
            [raster.quant.pack :as pack]
            [raster.compiler.backend.cpu.quant :as q]
            [raster.compiler.pipeline :as pipeline]))

(defn- clang-available? []
  (try
    (let [cc (or (System/getenv "RASTER_CC") "clang")
          p (-> (ProcessBuilder. ^java.util.List [cc "--version"])
                (.redirectErrorStream true) (.start))]
      (.waitFor p) (zero? (.exitValue p)))
    (catch Exception _ false)))

(defn- gen [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5)))) a))

(defn- bytes->ints-le [^bytes values]
  (let [result (int-array (quot (alength values) 4))
        buffer (.order (java.nio.ByteBuffer/wrap values) java.nio.ByteOrder/LITTLE_ENDIAN)]
    (dotimes [i (alength result)]
      (aset result i (.getInt buffer (* i 4))))
    result))

(defn- signed-packed-byte
  [^ints values index]
  (let [word (aget values (quot index 4))
        shift (* 8 (rem index 4))]
    (int (unchecked-byte (bit-and (unsigned-bit-shift-right word shift) 0xFF)))))

(defn- float-row [^floats values row width]
  (let [result (float-array width)]
    (System/arraycopy values (* row width) result 0 width)
    result))

(defn- pad-rows [^floats values nrows width padded-width]
  (let [result (float-array (* nrows padded-width))]
    (dotimes [row nrows]
      (System/arraycopy values (* row width) result (* row padded-width) width))
    result))

(defn- ref-matmul [^floats W-dq ^floats x-dq out in]
  (let [y (float-array out)]
    (dotimes [o out]
      (aset y o (float (areduce x-dq k s 0.0
                                (+ s (* (aget W-dq (+ (* (long o) (long in)) k)) (aget x-dq k)))))))
    y))

(defn- dequant-act [{:keys [xq xs]} in]
  (let [d (float-array in) dact (aget ^floats xs 0)]    ; in = one super-block
    (dotimes [k in] (aset d k (float (* dact (aget ^bytes xq k))))) d))

(deftest q4k-composable-c
  (when (clang-available?)
    (testing "composable Q4_K GEMV → C matches dequant-matmul"
      (let [out 8 in 256
            W (gen (* out in) 1) x (gen in 2)
            {:keys [wq da db aq bq] :as ew} (q/quantize-weight-q4k W q/q4-K)
            {:keys [xq xs bsums] :as ea} (q/quantize-act-q8k x in q/q4-K)
            cfn (pipeline/compile-aot #'qk/qmatmul-q4k-composable! :target :c)
            y (float-array out)]
        (cfn xq xs bsums wq da db aq bq y in out 0 out)
        (let [yref (ref-matmul (q/dequant-q4k ew q/q4-K (* out in)) (dequant-act ea in) out in)]
          (dotimes [o out]
            (is (< (Math/abs (- (aget y o) (aget yref o))) 1e-2)
                (str "Q4_K row " o ": C " (aget y o) " vs ref " (aget yref o)))))))))

(deftest q6k-composable-c
  (when (clang-available?)
    (testing "composable Q6_K GEMV → C matches dequant-matmul"
      (let [out 8 in 256
            W (gen (* out in) 3) x (gen in 4)
            {:keys [wq sc ds] :as ew} (q/quantize-weight-q6k W q/q6-K)
            {:keys [xq xs bsums] :as ea} (q/quantize-act-q8k x in q/q6-K)
            cfn (pipeline/compile-aot #'qk/qmatmul-q6k-composable! :target :c)
            y (float-array out)]
        (cfn xq xs bsums wq sc ds y in out 0 out)
        (let [yref (ref-matmul (q/dequant-q6k ew q/q6-K (* out in)) (dequant-act ea in) out in)]
          (dotimes [o out]
            (is (< (Math/abs (- (aget y o) (aget yref o))) 1e-2)
                (str "Q6_K row " o ": C " (aget y o) " vs ref " (aget yref o)))))))))

(deftest q6k-dp4a-matches-the-composable-kernel-without-a-device
  (let [out 7 in 256
        W (gen (* out in) 31) x (gen in 32)
        {:keys [wq sc ds]} (q/quantize-weight-q6k W q/q6-K)
        {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q6-K)
        wp (bytes->ints-le wq) xp (bytes->ints-le xq)
        expected (float-array out) actual (float-array out)]
    (qk/qmatmul-q6k-composable! xq xs bsums wq sc ds expected in out 0 out)
    (qk/qmatmul-q6k-dp4a! xp xs bsums wp sc ds actual in out)
    (dotimes [o out]
      (is (< (Math/abs (- (aget expected o) (aget actual o))) 1e-3)
          (str "Q6_K output row " o)))))

(deftest signed-i8-prefill-gemm-preserves-independent-dense-rows
  (let [nrows 3 in 64 out 5 nb (quot in 32)
        W (gen (* out in) 41) x (gen (* nrows in) 42)
        {:keys [wp ws]} (pack/quantize-one-q8 W in out "test-weight")
        xp (int-array (* nrows (quot in 4)))
        xs (float-array (* nrows nb))
        actual (float-array (* nrows out))]
    (qk/quant-act-i8-rows-gpu! x xp xs in nrows)
    (qk/qmatmul-i8-gemm! xp xs wp ws actual in out nrows)
    (dotimes [row nrows]
      (dotimes [o out]
        (let [expected
              (float
               (loop [b 0 acc 0.0]
                 (if (< b nb)
                   (let [dp (loop [k 0 dot 0]
                              (if (< k 32)
                                (recur (inc k)
                                       (+ dot
                                          (* (signed-packed-byte wp (+ (* o in) (* b 32) k))
                                             (signed-packed-byte xp (+ (* row in) (* b 32) k)))))
                                dot))]
                     (recur (inc b)
                            (+ acc (* (aget ws (+ (* o nb) b))
                                      (aget xs (+ (* row nb) b))
                                      dp))))
                   acc)))]
          (is (< (Math/abs (- expected (aget actual (+ (* row out) o)))) 1e-4)
              (str "signed-I8 row " row ", output " o)))))))

(deftest q8k-activation-quantization-has-an-ordinary-row-axis
  (let [nrows 3 in 512 nsb (quot in 256) nsub (quot in 32)
        x (gen (* nrows in) 71)
        xp (int-array (* nrows (quot in 4)))
        xs (float-array (* nrows nsb))
        bsums (int-array (* nrows nsub))
        submax (float-array (* nrows nsub))]
    (qk/quant-act-q8k-rows-gpu! x xp xs bsums submax in nrows)
    (dotimes [row nrows]
      (let [{expected-xq :xq expected-xs :xs expected-bsums :bsums}
            (q/quantize-act-q8k (float-row x row in) in q/q4-K)
            expected-xp (bytes->ints-le expected-xq)]
        (is (= (vec expected-xp)
               (subvec (vec xp) (* row (quot in 4)) (* (inc row) (quot in 4))))
            (str "packed activation row " row))
        (is (= (vec expected-xs)
               (subvec (vec xs) (* row nsb) (* (inc row) nsb)))
            (str "activation scales row " row))
        (is (= (vec expected-bsums)
               (subvec (vec bsums) (* row nsub) (* (inc row) nsub)))
            (str "activation block sums row " row))))))

(deftest cooperative-q8k-quantization-removes-reduction-scratch
  (let [nrows 3 in 512 n (* nrows in)
        x (gen n 171)
        expected-xp (int-array (quot n 4))
        expected-xs (float-array (* nrows (quot in 256)))
        expected-bsums (int-array (quot n 32))
        submax (float-array (quot n 32))
        actual-xp (int-array (alength expected-xp))
        actual-xs (float-array (alength expected-xs))
        actual-bsums (int-array (alength expected-bsums))]
    (qk/quant-act-q8k-rows-gpu! x expected-xp expected-xs expected-bsums submax in nrows)
    (qk/quant-act-q8k-cooperative-rows-gpu! x actual-xp actual-xs actual-bsums in nrows)
    (is (= (vec expected-xp) (vec actual-xp)))
    (is (= (vec expected-xs) (vec actual-xs)))
    (is (= (vec expected-bsums) (vec actual-bsums)))))

(deftest cooperative-padded-q8k-matches-the-compatibility-layout
  (let [nrows 3 width 640 padded-in 768 n (* nrows padded-in)
        dense (gen (* nrows width) 173)
        padded (pad-rows dense nrows width padded-in)
        expected-xp (int-array (quot n 4))
        expected-xs (float-array (* nrows (quot padded-in 256)))
        expected-bsums (int-array (quot n 32))
        submax (float-array (quot n 32))
        actual-xp (int-array (alength expected-xp))
        actual-xs (float-array (alength expected-xs))
        actual-bsums (int-array (alength expected-bsums))]
    (qk/quant-act-q8k-rows-gpu! padded expected-xp expected-xs expected-bsums submax
                                padded-in nrows)
    (qk/quant-act-q8k-cooperative-padded-rows-gpu!
     dense actual-xp actual-xs actual-bsums width padded-in nrows)
    (is (= (vec expected-xp) (vec actual-xp)))
    (is (= (vec expected-xs) (vec actual-xs)))
    (is (= (vec expected-bsums) (vec actual-bsums)))))

(deftest cooperative-q8k-retains-the-legacy-halfway-rounding-point
  ;; This seed contains a value for which recomputing id from the rounded public FP32 scale changes
  ;; one byte and its sub-block sum. Keep the exact private-maximum arithmetic contract explicit.
  (let [in 256 x (gen in 1238)
        expected-xp (int-array 64) expected-xs (float-array 1)
        expected-bsums (int-array 8) submax (float-array 8)
        actual-xp (int-array 64) actual-xs (float-array 1) actual-bsums (int-array 8)]
    (qk/quant-act-q8k-rows-gpu!
     x expected-xp expected-xs expected-bsums submax in 1)
    (qk/quant-act-q8k-cooperative-rows-gpu!
     x actual-xp actual-xs actual-bsums in 1)
    (is (= (vec expected-xp) (vec actual-xp)))
    (is (= (vec expected-xs) (vec actual-xs)))
    (is (= (vec expected-bsums) (vec actual-bsums)))))

(deftest q8k-padded-row-quantization-does-not-materialize-layout-padding
  (let [nrows 3 width 640 padded-in 768
        nsub (quot padded-in 32) nsb (quot padded-in 256)
        dense (gen (* nrows width) 73)
        padded (pad-rows dense nrows width padded-in)
        expected-xp (int-array (* nrows (quot padded-in 4)))
        expected-xs (float-array (* nrows nsb))
        expected-bsums (int-array (* nrows nsub))
        expected-submax (float-array (* nrows nsub))
        actual-xp (int-array (alength expected-xp))
        actual-xs (float-array (alength expected-xs))
        actual-bsums (int-array (alength expected-bsums))
        actual-submax (float-array (alength expected-submax))]
    (qk/quant-act-q8k-rows-gpu! padded expected-xp expected-xs expected-bsums expected-submax
                                padded-in nrows)
    (qk/quant-act-q8k-padded-rows-gpu! dense actual-xp actual-xs actual-bsums actual-submax
                                       width padded-in nrows)
    (is (= (vec expected-xp) (vec actual-xp)) "packed words equal explicit zero padding")
    (is (= (vec expected-xs) (vec actual-xs)) "super-block scales equal explicit zero padding")
    (is (= (vec expected-bsums) (vec actual-bsums)) "sub-block sums equal explicit zero padding")
    (is (= (vec expected-submax) (vec actual-submax)) "reduction scratch preserves row boundaries")
    (let [equal-xp (int-array (alength expected-xp))
          equal-xs (float-array (alength expected-xs))
          equal-bsums (int-array (alength expected-bsums))
          equal-submax (float-array (alength expected-submax))]
      (qk/quant-act-q8k-padded-rows-gpu! padded equal-xp equal-xs equal-bsums equal-submax
                                         padded-in padded-in nrows)
      (is (= (vec expected-xp) (vec equal-xp)) "equal-width adapter agrees with branch-free packing")
      (is (= (vec expected-xs) (vec equal-xs)) "equal-width adapter agrees on scales")
      (is (= (vec expected-bsums) (vec equal-bsums)) "equal-width adapter agrees on sums"))))

(deftest q4k-projection-shares-weights-across-activation-rows
  (let [nrows 3 in 512 out 11
        x (gen (* nrows in) 81)
        W (gen (* out in) 82)
        {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
        wp (bytes->ints-le wq)
        xp (int-array (* nrows (quot in 4)))
        xs (float-array (* nrows (quot in 256)))
        bsums (int-array (* nrows (quot in 32)))
        submax (float-array (* nrows (quot in 32)))
        actual (float-array (* nrows out))]
    (qk/quant-act-q8k-rows-gpu! x xp xs bsums submax in nrows)
    (qk/qmatmul-q4k-dp4a-rows! xp xs bsums wp da db aq bq actual in out nrows)
    (dotimes [row nrows]
      (let [{:keys [xq xs bsums]} (q/quantize-act-q8k (float-row x row in) in q/q4-K)
            expected (float-array out)]
        (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq expected in out 0 out)
        (dotimes [o out]
          (is (< (Math/abs (- (aget actual (+ (* row out) o)) (aget expected o))) 1e-3)
              (str "Q4_K row " row ", output " o)))))))

(deftest i8-activation-packing-uses-typed-kernel-body
  (let [compiled (pipeline/show-pipeline #'qk/quant-act-i8-rows-gpu!
                                         :target-device :ocl:0 :dtype :float)
        report (report/from-pipeline compiled)
        kernel (first (:kernels compiled))
        nodes (tree-seq coll? seq (get-in kernel [:attributes :kernel-body :operations]))]
    (is (= :typed-soac (get-in report [:route :source-dialect])))
    (is (true? (get-in report [:route :typed-validated])))
    (is (= {:kernel-body 1} (get-in report [:emission :routes])))
    (is (empty? (get-in report [:emission :declines])))
    (is (some #(and (map? %)
                    (= :cast (:op %))
                    (= :int (:result-type %))
                    (= :wrap (get-in % [:options :overflow])))
              nodes)
        "packed int8 lanes retain explicit bit-pattern narrowing")
    (is (not (re-find #"rstr_trap_cast_i64_i32" (:source kernel)))
        "packing must not emit a checked narrowing that traps when bit 31 is set")))

(deftest row-capable-q4k-path-lowers-through-the-shared-gpu-pipeline
  (let [quant-kernels (:kernels (pipeline/show-pipeline #'qk/quant-act-q8k-rows-gpu!
                                                        :target-device :ze:0 :dtype :float))
        padded-kernels (:kernels (pipeline/show-pipeline #'qk/quant-act-q8k-padded-rows-gpu!
                                                         :target-device :ze:0 :dtype :float))
        projection-pipeline (pipeline/show-pipeline #'qk/qmatmul-q4k-dp4a-rows!
                                                    :target-device :ze:0 :dtype :float)
        projection-kernels (:kernels projection-pipeline)
        projection-report (report/from-pipeline projection-pipeline)]
    (is (= 2 (count quant-kernels)) "Q8_K remains an ordered two-phase reduction")
    (is (= 2 (count padded-kernels)))
    (doseq [kernel (concat quant-kernels padded-kernels)]
      (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
      (is (nil? (get-in kernel [:attributes :kernel-body-decline])))
      (is (= :one-work-item-per-element (get-in kernel [:attributes :kernel-body :schedule :strategy])))
      (is (= [256] (get-in kernel [:launch :workgroup-size]))))
    (doseq [packing [(second quant-kernels) (second padded-kernels)]]
      (let [body (get-in packing [:attributes :kernel-body])
            ops (tree-seq coll? seq (:operations body))
            branch (first (filter #(and (= "raster.compiler.ir.kernel_body.IfRegion"
                                           (some-> % class .getName))
                                        (some (fn [op] (= 'xs (:buffer op)))
                                              (:then-operations %))) ops))
            condition (first (filter #(= (:condition branch) (get-in % [:result :id])) ops))]
        (is (= :independent (get-in body [:attributes :effect-iteration-order])))
        (is (= :eq (get-in condition [:expression :op])))
        (is (= 0 (get-in condition [:expression :arguments 1 :value]))
            "only the zero digit publishes the super-block scale")
        (is (not-any? #(= 'xs (:buffer %))
                      (tree-seq coll? seq (:else-operations branch))))))
    (is (= '[[[x :float] [submax :float] [_n_bound :long]]
             [[submax :float] [x :float] [bsums :int] [xp :int]
              [xs :float] [in :long] [_n_bound :long]]]
           (mapv #(mapv (juxt :name :dtype) (:abi %)) quant-kernels)))
    (is (= '[[[x :float] [submax :float] [padded-in :long] [width :long] [_n_bound :long]]
             [[submax :float] [x :float] [bsums :int] [xp :int] [xs :float]
              [padded-in :long] [width :long] [_n_bound :long]]]
           (mapv #(mapv (juxt :name :dtype) (:abi %)) padded-kernels))
        "the adapter changes layout indexing without changing the two-phase Q8_K representation")
    (doseq [kernel padded-kernels]
      (is (boolean (some #(and (= :lt (:op %)) (some #{'width} (:arguments %)))
                         (tree-seq coll? seq (get-in kernel [:attributes :kernel-body :operations]))))
          "both phases retain the width comparison guarding dense-row reads"))
    (is (= 1 (count projection-kernels)))
    (is (= {:backend :opencl
            :source-dialect :typed-soac
            :typed-validated true
            :declines []}
           (:route projection-report))
        "Q4_K uses the validated typed route rather than compatibility lowering")
    (is (= {:segops 1 :kernel-graphs 0 :structured-loops 0
            :typed-reused 1 :typed-scalar-equations 1
            :backend-reused 1 :backend-relowered 0 :fallback 0}
           (:lowering projection-report))
        "the scheduled SegMap is preserved through OpenCL emission")
    (is (= '[[aq :byte] [bq :byte] [bsums :int] [da :float] [db :float]
             [wp :int] [xp :int] [xs :float] [y :float]
             [in :long] [out :long] [_n_bound :long]]
           (mapv (juxt :name :dtype) (:abi (first projection-kernels)))))
    (is (re-find #"rstr_dp4a" (:source (first projection-kernels)))
        "Q4_K row projection reaches the target-neutral integer-dot intrinsic")
    (is (not (re-find #"\\bdouble\\b" (:source (first projection-kernels))))
        "the float projection does not accidentally promote accumulation to FP64")))

(deftest cooperative-q8k-is-four-typed-general-schedules
  (doseq [quantizer [#'qk/quant-act-q8k-cooperative-rows-gpu!
                     #'qk/quant-act-q8k-cooperative-padded-rows-gpu!]]
    (let [compiled (pipeline/show-pipeline quantizer :target-device :ze:0 :dtype :float)
          kernels (:kernels compiled)
          report (report/from-pipeline compiled)
          descriptor (pipeline/compile-gpu-program quantizer :ze:0 :dtype :float)]
      (is (= {:backend :opencl :source-dialect :typed-soac
              :typed-validated true :declines []}
             (:route report)))
      (is (= 4 (count kernels)))
      (is (= [:one-workgroup-per-segment
              :one-work-item-per-element
              :one-work-item-per-element
              :one-work-item-per-element]
             (mapv #(get-in % [:attributes :kernel-body :schedule :strategy]) kernels)))
      (is (= [256] (get-in (first kernels) [:launch :workgroup-size]))
          "the static 256-element max fold does not waste a 1024-lane workgroup")
      (is (= {:kernel-body 4} (get-in report [:emission :routes])))
      (is (empty? (get-in report [:emission :declines])))
      (is (= [{:sym 'maxes :dtype :float}]
             (mapv #(select-keys % [:sym :dtype]) (:allocs descriptor)))
          "one compiler-owned maximum per super-block replaces caller-managed 8x submax"))))

(deftest sibling-dp4a-contractions-preserve-typed-soac-through-opencl
  (let [q6-pipeline (pipeline/show-pipeline #'qk/qmatmul-q6k-dp4a!
                                            :target-device :ze:0 :dtype :float)
        i8-pipeline (pipeline/show-pipeline #'qk/qmatmul-i8-gemm!
                                            :target-device :ze:0 :dtype :float)
        q6-report (report/from-pipeline q6-pipeline)
        i8-report (report/from-pipeline i8-pipeline)
        expected-route {:backend :opencl :source-dialect :typed-soac
                        :typed-validated true :declines []}]
    (is (= expected-route (:route q6-report)))
    (is (= expected-route (:route i8-report)))
    (is (= {:segops 1 :kernel-graphs 0 :structured-loops 0
            :typed-reused 1 :typed-scalar-equations 0
            :backend-reused 1 :backend-relowered 0 :fallback 0}
           (:lowering q6-report)))
    (is (= {:segops 1 :kernel-graphs 0 :structured-loops 0
            :typed-reused 1 :typed-scalar-equations 1
            :backend-reused 1 :backend-relowered 0 :fallback 0}
           (:lowering i8-report)))
    (is (= '[[[bsums :int] [ds :float] [sc :byte] [wp :int] [xp :int]
              [xs :float] [y :float] [in :long] [_n_bound :long]]]
           (mapv #(mapv (juxt :name :dtype) (:abi %)) (:kernels q6-pipeline))))
    (is (= '[[[wp :int] [ws :float] [xp :int] [xs :float] [y :float]
              [in :long] [out :long] [_n_bound :long]]]
           (mapv #(mapv (juxt :name :dtype) (:abi %)) (:kernels i8-pipeline))))
    (is (every? #(re-find #"rstr_dp4a" (:source %))
                (concat (:kernels q6-pipeline) (:kernels i8-pipeline)))
        "both formats use the one target-neutral integer-dot intrinsic")))
