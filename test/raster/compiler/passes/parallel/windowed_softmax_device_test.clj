(ns raster.compiler.passes.parallel.windowed-softmax-device-test
  "GPU/JVM parity for the certified row-owned masked softmax effect map."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.attention :as attention]
            [raster.dl.gpu-grad-parity :as level-zero-probe]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl-probe]
            [raster.gpu.value :as value]))

(defn- run-windowed-softmax! [target]
  (let [descriptor (pipeline/compile-gpu-program
                    #'attention/attn-prefill-softmax-windowed-head-major!
                    target :dtype :float)]
    (is (= [:map-void] (mapv :convention (:steps descriptor))))
    (is (= [:kernel-body]
           (mapv #(get-in % [:artifact :attributes :emission-route])
                 (:steps descriptor))))
    (gpu/with-gpu-session [session target]
      (doseq [[rows heads left right] [[1 1 0 0] [4 2 2 2] [5 3 0 3] [5 2 4 0]]]
        (let [source (float-array
                      (map #(float (/ (- (mod (* 17 %) 29) 14) 3.0))
                           (range (* heads rows rows))))
              expected (aclone source)
              arguments [source (long rows) (long heads) (long left) (long right)]
              program (fixture/instantiate! session descriptor arguments {'sc :output})]
          (attention/attn-prefill-softmax-windowed-head-major!
           expected rows heads left right)
          (try
            (let [actual (get (fixture/run! program arguments) 'sc)]
              (is (= (alength expected) (count actual)))
              (is (every? #(< (Math/abs (double %)) 2.0e-5)
                          (map - (vec expected) actual))
                  (str "windowed softmax " [rows heads left right] " matches JVM")))
            (finally
              (fixture/close! program))))))))

(deftest opencl-windowed-softmax-preserves-jvm-results
  (if @opencl-probe/opencl-available?
    (run-windowed-softmax! :ocl:0)
    (opencl-probe/opencl-skip! "windowed softmax row ownership")))

(deftest level-zero-windowed-softmax-preserves-jvm-results
  (if @level-zero-probe/gpu-available?
    (run-windowed-softmax! :ze:0)
    (level-zero-probe/gpu-skip! "windowed softmax row ownership")))

(defn- run-equation-first-windowed-softmax! [target]
  (let [rows 4
        heads 2
        left 2
        right 2
        source (float-array
                (map #(float (/ (- (mod (* 17 %) 29) 14) 3.0))
                     (range (* heads rows rows))))
        expected (aclone source)
        arguments [source (long rows) (long heads) (long left) (long right)]
        prepared (compiled/lower
                  #'attention/attn-prefill-softmax-windowed-head-major! arguments
                  {:compiler :equation-first :target target :dtype :float
                   :donate '[sc]})
        artifact (compiled/instantiate! prepared)]
    (attention/attn-prefill-softmax-windowed-head-major!
     expected rows heads left right)
    (try
      (is (= :equation-first (get-in prepared [:schedule :compiler])))
      (is (= :none (get-in prepared [:schedule :stats :fallback])))
      (let [result (artifact)
            actual (vec (value/->host (:sc' result)))]
        (is (= (alength expected) (count actual)))
        (is (every? #(< (Math/abs (double %)) 2.0e-5)
                    (map - (vec expected) actual))))
      (finally (compiled/close! artifact)))))

(deftest opencl-windowed-softmax-runs-through-public-equation-first-artifact
  (if @opencl-probe/opencl-available?
    (run-equation-first-windowed-softmax! :ocl:0)
    (opencl-probe/opencl-skip! "public equation-first windowed softmax")))

(deftest level-zero-windowed-softmax-runs-through-public-equation-first-artifact
  (if @level-zero-probe/gpu-available?
    (run-equation-first-windowed-softmax! :ze:0)
    (level-zero-probe/gpu-skip! "public equation-first windowed softmax")))
