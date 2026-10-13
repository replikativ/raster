(ns raster.gpu.cuda-runtime-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.cuda-nvrtc :as nvrtc]
            [raster.gpu.resource-cleanup :as cleanup])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(defn- runtime-var [name]
  (try
    (require 'raster.gpu.cuda-runtime)
    (ns-resolve 'raster.gpu.cuda-runtime (symbol name))
    (catch java.io.FileNotFoundException _ nil)))

(defn- ordered-artifact []
  (artifact/make
   {:kernel-name "ordered_contract" :target :cuda-c
    ;; This synthetic source tests the transport ABI, not numerical CUDA coverage.
    :source "extern \"C\" __global__ void ordered_contract(float alpha, const float* input, long n, float* output) {}"
    :abi [(abi/slot 'alpha :scalar :float)
          (abi/slot 'input :input :float :aliasing :no-write-alias)
          (abi/slot 'n :scalar :long :role :bound)
          (abi/slot 'output :output :float :role :result)]
    :arguments '[alpha input n output]
    :launch (launch/spec {:workgroup-size [8 2 1] :group-count [3 4 5]})}))

(deftest ordered-call-retains-typed-offset-pointers-and-all-launch-axes
  ;; A missing API is an assertion failure in a working compiler/contract harness.
  ;; Import, FFM and native setup failures are never counted as behavioural RED.
  (let [a (ordered-artifact)
        bind! (runtime-var "bind-kernel-call")]
    (is (= '[alpha input n output] (:arguments a)))
    (is (some? bind!) "CUDA must bind a modern ordered KernelCall")
    (when bind!
      (let [operations (atom []) next-address (atom 4096)
            compile-count (atom 0)
            compilation-identity {:nvrtc-version [13 2] :virtual-architecture "compute_86"
                                  :include-paths ["/verified/include"] :virtual-header-version 2
                                  :compiler-options ["--gpu-architecture=compute_86" "--std=c++17"
                                                     "--fmad=false" "--include-path=/verified/include"]}
            fake (fn [op args]
                   (swap! operations conj [op args])
                   (case op
                     :initialize {:context (MemorySegment/ofAddress 1)
                                  :stream (MemorySegment/ofAddress 2)
                                  :device 0 :compute-capability [8 6]
                                  :max-workgroup-size 1024 :max-block-dimensions [1024 1024 64]
                                  :max-grid-dimensions [2147483647 65535 65535]
                                  :shared-local-memory 49152}
                     :allocate (swap! next-address + 4096)
                     :module-load (MemorySegment/ofAddress 3)
                     :module-function (MemorySegment/ofAddress 4)
                     nil))
            driver-var (do (require 'raster.gpu.cuda-driver)
                           (ns-resolve 'raster.gpu.cuda-driver '*invoke*))
            state-var (runtime-var "state") registry-var (runtime-var "kernel-registry")
            make-buffer (runtime-var "make-buffer") slice (runtime-var "slice-buffer")
            free! (runtime-var "free-buffer!") register! (runtime-var "register-kernel!")]
        (with-bindings {driver-var fake}
          (with-redefs-fn {state-var (atom {}) registry-var (atom {})
                           #'nvrtc/compile-artifact-ptx
                           (fn [& _]
                             (swap! compile-count inc)
                             (assoc compilation-identity :ptx (byte-array [1 0])))}
            (fn []
              (let [input (make-buffer 24 :float)
                    output (make-buffer 17 :float)
                    view (slice input 12 68 :float)
                    args [{:type :float :value (float 2)} view
                          {:type :long :value 17} output]]
                (try
                  (register! (:kernel-name a) a)
                  (let [prepared (bind! (call/make a args))]
                    (try
                      (is (= compilation-identity (:compilation-identity prepared)))
                      (let [reused (bind! (call/make a args))]
                        (try
                          (is (= compilation-identity (:compilation-identity reused)))
                          (is (= 1 @compile-count) "the existing per-registration result is reused")
                          (finally ((runtime-var "destroy-prepared!") reused))))
                      ((runtime-var "launch-registered-bound!") prepared)
                      (let [[_ [function geometry parameters]]
                            (first (filter #(= :launch (first %)) @operations))
                            addresses (mapv #(.get parameters ValueLayout/ADDRESS (long (* % 8))) (range 4))]
                        (is (= 4 (.address function)))
                        (is (= [8 2 1] (:workgroup-size geometry)))
                        (is (= [3 4 5] (:group-count geometry)))
                        (is (= (float 2) (.get (.reinterpret (nth addresses 0) 4) ValueLayout/JAVA_FLOAT 0)))
                        (is (= (+ (:device-pointer input) 12)
                               (.get (.reinterpret (nth addresses 1) 8) ValueLayout/JAVA_LONG 0)))
                        (is (= 17 (.get (.reinterpret (nth addresses 2) 8) ValueLayout/JAVA_LONG 0)))
                        (is (= (:device-pointer output)
                               (.get (.reinterpret (nth addresses 3) 8) ValueLayout/JAVA_LONG 0))))
                      (finally ((runtime-var "destroy-prepared!") prepared))))
                  (let [before (count @operations)]
                    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"storage dtype"
                                          (bind! (call/make a (assoc args 1 (assoc view :dtype :int))))))
                    (is (= before (count @operations)) "dtype rejection precedes driver contact"))
                  (finally
                    (free! input) (free! output)
                    ((runtime-var "close-kernel-arena!") nil)))))))))))
