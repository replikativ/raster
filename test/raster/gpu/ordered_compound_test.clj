(ns raster.gpu.ordered-compound-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.equation-first-c-family-test :as fixture]
            [raster.compiler.ir.abstract-value :as av]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as program-runtime]))

(defn- compilation [target]
  (equation-first/compile
   #'fixture/c-family-broadcast-product-map!
   {:target target :dtype :int
    :values {'input (av/tensor {:dtype :int :shape [64]})
             'weights (av/tensor {:dtype :int :shape [64]})
             'output (av/tensor {:dtype :int :shape [1]})}}))

(defn- check-compound! [target]
  (let [compiled (compilation target)
        input (int-array (range 64))
        weights (int-array (repeat 64 1))
        expected (int-array 1)
        live (link/instantiate-ordered-invocation! compiled [input weights (int-array 1) 1] {})]
    (try
      (is (= 1 (count (:kernels compiled))))
      (doseq [delta [0 7]]
        (aset input 0 (int delta))
        (fixture/c-family-broadcast-product-map! input weights expected 1)
        (link/run! live)
        (let [state @(:execution-state live)
              ordered (get-in state [:ordered :current])
              token (get-in state [:resident-boundary :plan :attributes :public-buffer-bindings 'output])]
          (is (= (vec expected) (vec (link/download live token))))
          (is (= (count (:source-order ordered)) (:source-cursor ordered)))
          (is (= 1 (count (:equation-spans ordered))))
          (is (nil? (:pending ordered)))))
      (finally (link/close! live))))
  (let [compiled (compilation target)
        live (link/instantiate-ordered-invocation!
              compiled [(int-array 64) (int-array 64) (int-array 1) 1] {})
        failure (Exception. "injected compound execution failure")]
    (try
      (with-redefs [program-runtime/run-prepared! (fn [& _] (throw failure))]
        (is (identical? failure (try (link/run! live) nil (catch Exception error error)))))
      (let [state @(:execution-state live)
            ordered (get-in state [:ordered :current])
            start (first (keys (:equation-spans ordered)))]
        (is (= start (:source-cursor ordered)) "failed execution consumes neither constituent")
        (is (nil? (:resident-boundary state)) "failed stage publishes no completed boundary"))
      (finally (link/close! live))))
  (let [compiled (compilation target)
        live (link/instantiate-ordered-invocation!
              compiled [(int-array 64) (int-array 64) (int-array 1) -1] {})
        submissions (atom 0)]
    (try
      (with-redefs [program-runtime/run-prepared!
                    (fn [& _] (swap! submissions inc) (throw (AssertionError. "premature submission")))]
        (is (thrown? NegativeArraySizeException (link/run! live))))
      (is (zero? @submissions) "the original earlier allocation must fail before the fused stage")
      (is (nil? (:resident-boundary @(:execution-state live))))
      (finally (link/close! live)))))

(deftest ordered-compounds-complete-once-and-replay-on-local-devices
  (if @ocl-probe/opencl-available?
    (check-compound! :ocl:0)
    (ocl-probe/opencl-skip! "ordered compound source spans"))
  (if @ze-probe/gpu-available?
    (check-compound! :ze:0)
    (ze-probe/gpu-skip! "ordered compound source spans")))
