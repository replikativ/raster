(ns raster.compiler.gpu-registration-test
  "Hardware-free failure propagation at the compiler/runtime registration boundary."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

(defn- error-of [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error error)))

(deftest registration-failures-are-contextual-and-preserve-the-cause
  (doseq [[target backend] [[:ocl:0 'raster.gpu.ocl-runtime]
                            [:ze:0 'raster.gpu.ze-runtime]]
          [helper registrar reason values]
          [['register-gpu-kernels! 'register-kernel! :gpu-kernel-registration
            [{:kernel-name "first"} {:kernel-name "second"} {:kernel-name "third"}]]
           ['register-gpu-dispatches! 'register-kernel-dispatch! :gpu-dispatch-registration
            [{:id "first"} {:id "second"} {:id "third"}]]]]
    (let [calls (atom []) failure (ex-info "injected native registration failure" {})
          register (fn [& args]
                     (swap! calls conj args)
                     (when (= 2 (count @calls)) (throw failure)))
          error (with-redefs-fn {(ns-resolve backend registrar) register}
                  #(error-of (fn [] ((ns-resolve 'raster.compiler.pipeline helper)
                                    values target))))]
      (is (= reason (:reason (ex-data error))))
      (is (= target (:device-id (ex-data error))))
      (is (= backend (:backend (ex-data error))))
      (is (identical? failure (.getCause error)))
      (is (= 2 (count @calls)) "never register the suffix after a failed prefix"))))

(deftest backend-emission-cannot-return-an-executable-after-registration-failure
  (doseq [phase [:kernel :dispatch]]
    (let [failure (ex-info "registration rejected" {})
          calls (atom [])
          register-kernel (fn [& _]
                            (swap! calls conj :kernel)
                            (when (= phase :kernel) (throw failure)))
          register-dispatch (fn [& _]
                              (swap! calls conj :dispatch)
                              (when (= phase :dispatch) (throw failure)))
          error (with-redefs [device/select-runtime-backend (constantly :opencl)
                             opencl-pass/opencl-pass
                             (fn [& _] {:form :must-not-be-returned
                                        :kernels [{:kernel-name "fixture"}]
                                        :dispatches [{:id "fixture"}]})
                             ocl/register-kernel! register-kernel
                             ocl/register-kernel-dispatch! register-dispatch]
                  (error-of #((ns-resolve 'raster.compiler.pipeline 'pass-backend)
                              '(do) {:target-device :ocl:0})))]
      (is (= (if (= phase :kernel) :gpu-kernel-registration :gpu-dispatch-registration)
             (:reason (ex-data error))))
      (is (identical? failure (.getCause error)))
      (is (= (if (= phase :kernel) [:kernel] [:kernel :dispatch]) @calls)))))
