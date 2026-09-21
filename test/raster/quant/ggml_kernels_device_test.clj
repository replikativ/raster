(ns raster.quant.ggml-kernels-device-test
  "The GPU dot kernels against ggml's generic vec_dot, bit for bit.

  Weights and activations are quantized by raster.quant.ggml (itself checked
  byte for byte against llama.cpp), decoded with `kernel-layout`, and every
  GPU output must equal `ggml/vec-dot` without contraction exactly."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.value :as value]
            [raster.quant.ggml :as ggml]
            [raster.quant.ggml-kernels :as gk]))

(defn- student-t ^double [^java.util.Random r]
  (let [z (.nextGaussian r)
        chi (+ (Math/pow (.nextGaussian r) 2) (Math/pow (.nextGaussian r) 2)
               (Math/pow (.nextGaussian r) 2))]
    (/ z (Math/sqrt (/ chi 3.0)))))

(defn- values ^floats [n seed scale]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (* (double scale) (student-t r)))))
    a))

(defn- row-bytes [^bytes blocks ^long row ^long row-size]
  (java.util.Arrays/copyOfRange blocks (* row row-size) (* (inc row) row-size)))

(def ^:private cases
  "[weight-format activation-format kernel in out nrows]"
  [[:q8_0 :q8_0 #'gk/qdot-q8-0-rows! 640 64 2]
   [:q5_0 :q8_0 #'gk/qdot-q5-0-rows! 640 64 2]
   [:q4_K :q8_K #'gk/qdot-q4-K-rows! 1024 64 2]
   [:q6_K :q8_K #'gk/qdot-q6-K-rows! 1024 64 2]])

(defn- buffers
  "Session buffer specs and kernel parameter bindings for one case."
  [wf af w x]
  (let [wl (ggml/kernel-layout wf (:blocks w) (:n w) (:rows w))
        xl (ggml/kernel-layout af (:blocks x) (:n x) (:rows x))
        spec (fn [k ^objects arr]
               [k (if (instance? (Class/forName "[F") arr)
                    [:float (alength ^floats arr) arr]
                    [:int (alength ^ints arr) arr])])
        entries (cond-> {"xq" (:q xl) "xd" (:d xl) "wq" (:q wl) "wd" (:d wl)}
                  (= af :q8_K) (assoc "xbs" (:bsums xl))
                  (= wf :q4_K) (assoc "wdmin" (:dmin wl) "wsc" (:sc wl) "wm" (:m wl))
                  (= wf :q6_K) (assoc "wsc" (:sc wl)))]
    {:arrays entries
     :specs (into {} (map (fn [[param arr]] (spec (keyword param) arr))) entries)
     :bindings (into {} (map (fn [[param _]] [param (keyword param)])) entries)}))

(declare run-program)

(deftest q6-k-dot-is-a-typed-product-followed-by-an-ordered-map
  (let [descriptor (pipeline/compile-gpu-program #'gk/qdot-q6-K-rows!
                                                 :ze:0 :dtype :float)
        executable (get-in descriptor [:steps 0 :artifact])
        lowering (resident-plan/lower
                  {:id ::q6-resident-product
                   :target :ze:0
                   :descriptor descriptor
                   :arguments [(int-array 64) (float-array 1)
                               (int-array 64) (float-array 1) (int-array 16)
                               (float-array 1) 256 1 1]
                   :outputs ['y]})]
    (is (empty? (:allocs descriptor))
        "the certified graph refinement substitutes the semantic temporary")
    (is (= [:executable] (mapv :convention (:steps descriptor))))
    (is (contains? #{:product-tree-ordered-consumer
                     :subgroup-product-ordered-consumer}
                   (get-in executable [:attributes :strategy])))
    (is (resident-plan/certified-plan? lowering)
        "self extent contracts resolve from resident views without hidden ABI scalars")
    (is (= '[xq xd wq wd wsc y in out nrows] (:all-params descriptor)))))

(deftest q6-k-product-semantics-remain-bit-identical
  (let [in 256 out 3 nrows 2
        wfloats (values (* out in) 31 0.05)
        xfloats (values (* nrows in) 32 1.5)
        wblocks (ggml/quantize :q6_K wfloats in out)
        xblocks (ggml/quantize :q8_K xfloats in nrows)
        wl (ggml/kernel-layout :q6_K wblocks in out)
        xl (ggml/kernel-layout :q8_K xblocks in nrows)
        y (float-array (* nrows out))
        wrow (ggml/row-bytes :q6_K in)
        xrow (ggml/row-bytes :q8_K in)]
    (gk/qdot-q6-K-rows! (:q xl) (:d xl) (:q wl) (:d wl) (:sc wl)
                        y in out nrows)
    (is (= (mapv (fn [row o]
                   (Float/floatToRawIntBits
                    (float (ggml/vec-dot :q6_K (row-bytes wblocks o wrow)
                                         (row-bytes xblocks row xrow) in))))
                 (mapcat #(repeat out %) (range nrows))
                 (cycle (range out)))
           (mapv #(Float/floatToRawIntBits %) y)))))

(deftest q4-k-product-semantics-remain-bit-identical
  (let [in 256 out 3 nrows 2
        wfloats (values (* out in) 35 0.05)
        xfloats (values (* nrows in) 36 1.5)
        wblocks (ggml/quantize :q4_K wfloats in out)
        xblocks (ggml/quantize :q8_K xfloats in nrows)
        wl (ggml/kernel-layout :q4_K wblocks in out)
        xl (ggml/kernel-layout :q8_K xblocks in nrows)
        y (float-array (* nrows out))
        wrow (ggml/row-bytes :q4_K in)
        xrow (ggml/row-bytes :q8_K in)]
    (gk/qdot-q4-K-rows! (:q xl) (:d xl) (:bsums xl)
                        (:q wl) (:d wl) (:dmin wl) (:sc wl) (:m wl)
                        y in out nrows)
    (is (= (mapv (fn [row o]
                   (Float/floatToRawIntBits
                    (float (ggml/vec-dot :q4_K (row-bytes wblocks o wrow)
                                         (row-bytes xblocks row xrow) in))))
                 (mapcat #(repeat out %) (range nrows))
                 (cycle (range out)))
           (mapv #(Float/floatToRawIntBits %) y)))))

(deftest q6-k-product-subgroup-executes-bit-identically-through-equation-first
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "equation-first Q6_K subgroup product")
    (let [in 256 out 3 nrows 2
          wfloats (values (* out in) 41 0.05)
          xfloats (values (* nrows in) 42 1.5)
          wblocks (ggml/quantize :q6_K wfloats in out)
          xblocks (ggml/quantize :q8_K xfloats in nrows)
          wl (ggml/kernel-layout :q6_K wblocks in out)
          xl (ggml/kernel-layout :q8_K xblocks in nrows)
          output (float-array (* nrows out))
          wrow (ggml/row-bytes :q6_K in)
          xrow (ggml/row-bytes :q8_K in)
          expected (mapv (fn [row o]
                           (Float/floatToRawIntBits
                            (float (ggml/vec-dot
                                    :q6_K (row-bytes wblocks o wrow)
                                    (row-bytes xblocks row xrow) in))))
                         (mapcat #(repeat out %) (range nrows))
                         (cycle (range out)))
          arguments [(:q xl) (:d xl) (:q wl) (:d wl) (:sc wl)
                     output
                     (long in) (long out) (long nrows)]
          compilation (equation-first/compile #'gk/qdot-q6-K-rows!
                                              {:target :ze:0 :dtype :float})
          plan (equation-first/lower compilation arguments)
          output-node (some (fn [[id node]]
                              (when (identical? output (:source node)) id))
                            (:nodes plan))]
      (is (= :subgroup-product-ordered-consumer
             (get-in compilation [:kernels 0 :attributes :kernel-body :schedule :strategy])))
      (with-open [live (gpu-link/instantiate! plan)]
        (gpu-link/run! live)
        (is (= expected
               (mapv #(Float/floatToRawIntBits %)
                     (gpu-link/download live output-node))))))))

(deftest q4-k-product-subgroup-executes-bit-identically-through-public-compiled-artifact
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "equation-first Q4_K subgroup product")
    (let [in 256 out 3 nrows 2
          wfloats (values (* out in) 43 0.05)
          xfloats (values (* nrows in) 44 1.5)
          wblocks (ggml/quantize :q4_K wfloats in out)
          xblocks (ggml/quantize :q8_K xfloats in nrows)
          wl (ggml/kernel-layout :q4_K wblocks in out)
          xl (ggml/kernel-layout :q8_K xblocks in nrows)
          output (float-array (* nrows out))
          wrow (ggml/row-bytes :q4_K in)
          xrow (ggml/row-bytes :q8_K in)
          expected (mapv (fn [row o]
                           (Float/floatToRawIntBits
                            (float (ggml/vec-dot
                                    :q4_K (row-bytes wblocks o wrow)
                                    (row-bytes xblocks row xrow) in))))
                         (mapcat #(repeat out %) (range nrows))
                         (cycle (range out)))
          arguments [(:q xl) (:d xl) (:bsums xl)
                     (:q wl) (:d wl) (:dmin wl) (:sc wl) (:m wl)
                     output
                     (long in) (long out) (long nrows)]
          prepared (compiled/lower #'gk/qdot-q4-K-rows! arguments
                                   {:compiler :equation-first
                                    :target :ze:0 :dtype :float
                                    :outputs '[y]})
          live (compiled/instantiate! prepared)]
      (try
        (let [result (live {})]
          (is (invocation-link/certified-link? (:lowering prepared)))
          (is (= :kernel-body (get-in prepared [:descriptor :steps 0 :convention])))
          (is (= expected
                 (mapv #(Float/floatToRawIntBits %)
                       (value/->host (:y result))))))
        (finally
          (compiled/close! live))))))

(deftest dot-kernels-match-the-generic-reference
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ggml dot kernels")
    (doseq [[wf af kernel in out nrows] cases]
      (let [wfloats (values (* out in) 11 0.02)
            xfloats (values (* nrows in) 12 2.0)
            w {:blocks (ggml/quantize wf wfloats in out) :n in :rows out}
            x {:blocks (ggml/quantize af xfloats in nrows) :n in :rows nrows}
            {:keys [arrays]} (buffers wf af w x)
            wrow (ggml/row-bytes wf in)
            xrow (ggml/row-bytes af in)
            expected (for [row (range nrows) o (range out)]
                       (Float/floatToRawIntBits
                        (float (ggml/vec-dot wf (row-bytes (:blocks w) o wrow)
                                             (row-bytes (:blocks x) row xrow) in))))]
        (let [actual-values
              (:y (run-program kernel
                               (assoc arrays "y" (float-array (* nrows out))
                                      'in in 'out out 'nrows nrows)
                               [:y]))
              actual (map #(Float/floatToRawIntBits %) actual-values)
              mismatches (count (remove true? (map = expected actual)))]
          (testing (str (name wf) " x " (name af))
            (when (= wf :q4_K)
              (is (some false? (for [row (range nrows) o (range out)]
                                 (= (ggml/vec-dot wf (row-bytes (:blocks w) o wrow)
                                                  (row-bytes (:blocks x) row xrow) in)
                                    (ggml/vec-dot wf (row-bytes (:blocks w) o wrow)
                                                  (row-bytes (:blocks x) row xrow) in
                                                  {:contract? true}))))
                  "the data tells FMA contraction apart, so equality rules it out"))
            (is (zero? mismatches)
                (str mismatches " of " (* nrows out) " outputs differ; first "
                     (first (remove (fn [[e a]] (= e a)) (map vector expected actual)))))))))))

(defn- activation-inputs
  "[label floats nrows width] cases for the activation quantizers."
  [width]
  (let [r (java.util.Random. 21)
        n (* 2 width)
        f (fn [g] (let [a (float-array n)] (dotimes [i n] (aset a i (float (g i)))) a))]
    [["gaussian" (f (fn [_] (* 2.0 (.nextGaussian r)))) 2 width]
     ["ties" (f (fn [i] (if (zero? (mod i 32)) 127.0 (+ 0.5 (- (mod (* 7 i) 200) 100))))) 2 width]
     ["signed-tie" (f (fn [i] (case (mod i 256) 0 -127.0 1 127.0 (* 0.25 (.nextGaussian r))))) 2 width]
     ["negative-max" (f (fn [i] (if (zero? (mod i 32)) -3.0 (* 0.5 (.nextGaussian r))))) 2 width]
     ["tiny" (f (fn [_] (* 1.0e-7 (.nextGaussian r)))) 2 width]
     ["negative-zero" (f (fn [_] -0.0)) 2 width]
     ["zeros" (f (fn [_] 0.0)) 2 width]]))

(defn- run-program
  "Compile `kernel` as a resident program, run it once on `arguments`, and return
  the arrays of the parameters listed in `outputs`."
  [kernel arguments outputs]
  (let [descriptor (pipeline/compile-gpu-program kernel :ze:0 :dtype :float)
        sess (gpu/make-session :ze:0)]
    (try
      (let [by-name (into {} (map (fn [[k v]] [(name k) v])) arguments)
            args (mapv #(get by-name (name %)) (:all-params descriptor))
            program (fixture/instantiate! sess descriptor args
                                          (into {} (for [p (:all-params descriptor)
                                                         :when (some #(= (name p) (name %)) outputs)]
                                                     [p :output])))
            results (fixture/run! program args)
            by-result (into {} (map (fn [[k v]] [(name k) v])) results)]
        (into {} (map (fn [k] [k (get by-result (name k))])) outputs))
      (finally (gpu/close-session! sess)))))

(deftest activation-quantizers-have-hardware-free-kernelbody-emission
  (doseq [kernel [#'gk/quant-act-q8-0-rows! #'gk/quant-act-q8-K-rows!]]
    (let [compiled (pipeline/compile-gpu-program kernel :ze:debug :dtype :float)
          artifacts (mapv :artifact (:steps compiled))
          kernels (mapcat kexec/artifacts artifacts)]
      (is (seq artifacts))
      (is (every? #(= :kernel-body (get-in % [:attributes :emission-route]))
                  kernels)
          (str (:name (meta kernel)) " must not depend on the GPU execution gate")))))

(deftest cooperative-activation-quantizers-retain-reference-semantics-on-jvm
  (doseq [[fmt kernel width] [[:q8_0 gk/quant-act-q8-0-rows! 640]
                              [:q8_K gk/quant-act-q8-K-rows! 1024]]
          [label x nrows width] (activation-inputs width)]
    (let [expected (ggml/kernel-layout fmt (ggml/quantize fmt x width nrows) width nrows)
          nblocks (* nrows (quot width (if (= fmt :q8_K) 256 32)))
          xq (int-array (alength ^ints (:q expected)))
          xd (float-array nblocks)
          xbs (when (= fmt :q8_K) (int-array (* 16 nblocks)))]
      (if xbs
        (kernel x xq xd xbs nblocks)
        (kernel x xq xd nblocks))
      (is (= (vec ^ints (:q expected)) (vec xq)) (str fmt " " label " codes"))
      (is (= (vec ^floats (:d expected)) (vec xd)) (str fmt " " label " scales"))
      (when xbs
        (is (= (vec ^ints (:bsums expected)) (vec xbs))
            (str fmt " " label " block sums"))))))

(deftest activation-quantizers-match-the-reference
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ggml activation quantizers")
    (doseq [[fmt kernel width] [[:q8_0 #'gk/quant-act-q8-0-rows! 640]
                                [:q8_K #'gk/quant-act-q8-K-rows! 1024]]
            [label x nrows width] (activation-inputs width)]
      (let [expected (ggml/kernel-layout fmt (ggml/quantize fmt x width nrows) width nrows)
            nblocks (* nrows (quot width (if (= fmt :q8_K) 256 32)))
            arguments (cond-> {'x x 'xq (int-array (alength ^ints (:q expected)))
                               'xd (float-array nblocks) 'nblocks nblocks}
                        (= fmt :q8_K) (assoc 'xbs (int-array (* 16 nblocks))))
            result (run-program kernel arguments
                                (cond-> ['xq 'xd] (= fmt :q8_K) (conj 'xbs)))
            actual {:xq (get result 'xq) :xd (get result 'xd) :xbs (get result 'xbs)}
            float-bits (fn [xs] (mapv #(Float/floatToRawIntBits %) xs))]
        (testing (str (name fmt) " " label)
          (is (= (vec (:q expected)) (vec (:xq actual))) "codes")
          (is (= (float-bits (:d expected)) (float-bits (:xd actual))) "scales, bit for bit")
          (when (= fmt :q8_K)
            (is (= (vec (:bsums expected)) (vec (:xbs actual))) "block sums")))))))
