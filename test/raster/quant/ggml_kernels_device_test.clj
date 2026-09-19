(ns raster.quant.ggml-kernels-device-test
  "The GPU dot kernels against ggml's generic vec_dot, bit for bit.

  Weights and activations are quantized by raster.quant.ggml (itself checked
  byte for byte against llama.cpp), decoded with `kernel-layout`, and every
  GPU output must equal `ggml/vec-dot` without contraction exactly."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.core :as gpu]
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
    {:specs (into {} (map (fn [[param arr]] (spec (keyword param) arr))) entries)
     :bindings (into {} (map (fn [[param _]] [param (keyword param)])) entries)}))

(deftest dot-kernels-match-the-generic-reference
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "ggml dot kernels")
    (doseq [[wf af kernel in out nrows] cases]
      (let [wfloats (values (* out in) 11 0.02)
            xfloats (values (* nrows in) 12 2.0)
            w {:blocks (ggml/quantize wf wfloats in out) :n in :rows out}
            x {:blocks (ggml/quantize af xfloats in nrows) :n in :rows nrows}
            {:keys [specs bindings]} (buffers wf af w x)
            wrow (ggml/row-bytes wf in)
            xrow (ggml/row-bytes af in)
            expected (for [row (range nrows) o (range out)]
                       (Float/floatToRawIntBits
                        (float (ggml/vec-dot wf (row-bytes (:blocks w) o wrow)
                                             (row-bytes (:blocks x) row xrow) in))))
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :dot kernel)
          (gpu/alloc! sess (assoc specs :y [:float (* nrows out) nil]))
          (gpu/prepare! sess :dot (assoc bindings "y" :y) [in out] (* nrows out)
                        {:kernel-phase :dot})
          (gpu/invoke-bound! sess :dot)
          (gpu/sync! sess)
          (let [actual (map #(Float/floatToRawIntBits %) (gpu/download sess :y))
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
                       (first (remove (fn [[e a]] (= e a)) (map vector expected actual)))))))
          (finally (gpu/close-session! sess)))))))

(defn- activation-inputs
  "[label floats nrows width] cases for the activation quantizers."
  [width]
  (let [r (java.util.Random. 21)
        n (* 2 width)
        f (fn [g] (let [a (float-array n)] (dotimes [i n] (aset a i (float (g i)))) a))]
    [["gaussian" (f (fn [_] (* 2.0 (.nextGaussian r)))) 2 width]
     ["ties" (f (fn [i] (if (zero? (mod i 32)) 127.0 (+ 0.5 (- (mod (* 7 i) 200) 100))))) 2 width]
     ["negative-max" (f (fn [i] (if (zero? (mod i 32)) -3.0 (* 0.5 (.nextGaussian r))))) 2 width]
     ["tiny" (f (fn [_] (* 1.0e-7 (.nextGaussian r)))) 2 width]
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
