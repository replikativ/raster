(ns raster.gpu.opencl-call-carrier-test
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [raster.gpu.ocl-runtime])
  (:import [java.io PushbackReader StringWriter]
           [java.lang.invoke MethodHandles MethodType]))

(defn- compiled-call-helper []
  ;; Compile the actual helper body, without redefining any runtime Var or opening a device.
  (with-open [reader (PushbackReader. (io/reader (io/resource "raster/gpu/ocl_runtime.clj")))]
    (loop []
      (let [form (read {:eof ::eof} reader)]
        (when (= ::eof form) (throw (AssertionError. "OpenCL call helper is missing")))
        (if (and (seq? form) (= 'defn- (first form)) (= 'cl-call! (second form)))
          (let [warnings (StringWriter.)
                helper (binding [*ns* (the-ns 'raster.gpu.ocl-runtime)
                                 *warn-on-reflection* true *err* warnings]
                         (eval (cons 'fn (drop-while #(not (vector? %)) (drop 2 form)))))]
            {:helper helper :warnings (str warnings)})
          (recur))))))

(deftest native-call-array-overload-is-nonreflective-and-preserves-errors
  (let [{:keys [helper warnings]} (compiled-call-helper)]
    (is (= "" warnings) "The actual Object[] call must resolve without reflective overload lookup")
    (testing "success and original OpenCL status diagnosis"
      (is (= 0 (helper "success" (MethodHandles/constant Integer/TYPE (int 0)) [])))
      (let [error (try (helper "failure" (MethodHandles/constant Integer/TYPE (int -30)) [])
                       (catch clojure.lang.ExceptionInfo error error))]
        (is (= {:function "failure" :error -30} (ex-data error)))))
    (testing "nonzero primitive argument count and order are unchanged"
      (let [^"[Ljava.lang.Class;" carriers (into-array Class [Integer/TYPE Integer/TYPE])
            handle (.findStatic (MethodHandles/lookup) Integer "compare"
                                (MethodType/methodType Integer/TYPE carriers))]
        (is (= 0 (helper "equal" handle [(int 7) (int 7)])))
        (doseq [[arguments status] [[[1 2] -1] [[2 1] 1]]]
          (let [error (try (helper "ordered" handle (mapv int arguments))
                           (catch clojure.lang.ExceptionInfo error error))]
            (is (= {:function "ordered" :error status} (ex-data error)))))))
    (testing "native boundary failures retain exact identity"
      (let [failure (IllegalStateException. "downcall failure")
            handle (.bindTo (MethodHandles/throwException Integer/TYPE IllegalStateException) failure)
            observed (try (helper "throw" handle []) (catch Throwable error error))]
        (is (identical? failure observed))))))
