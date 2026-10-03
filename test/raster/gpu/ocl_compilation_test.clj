(ns raster.gpu.ocl-compilation-test
  "Hardware-free checks at the native OpenCL compilation boundary."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ocl-runtime :as ocl])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.invoke MethodHandles MethodType]))

(defn- handle [arity f]
  (.bindTo (.findVirtual (MethodHandles/lookup) clojure.lang.IFn "invoke"
                         (MethodType/genericMethodType arity)) f))

(defn- runtime-var [sym] (ns-resolve 'raster.gpu.ocl-runtime sym))

(deftest device-info-distinguishes-driver-from-opencl-version
  (with-redefs-fn
    {(runtime-var 'query-device-info-string)
     (fn [_ ^long parameter] (case parameter 0x102D "driver-A" 0x102F "OpenCL 3.0" "fixture"))
     (runtime-var 'query-device-info-uint) (fn ^long [_ ^long _] 1)
     (runtime-var 'query-device-info-ulong) (fn ^long [_ ^long _] 1024)
     (runtime-var 'query-device-info-size-t) (fn ^long [_ ^long _] 256)
     (runtime-var 'device-type-keyword) (constantly :gpu)
     (runtime-var 'device-buffer-offset-alignment) (constantly 64)}
    (fn []
      (let [info ((runtime-var 'device-info) MemorySegment/NULL)]
        (is (= "driver-A" (:driver-version info)))
        (is (= "OpenCL 3.0" (:version info)))))))

(deftest compiler-options-and-failed-program-ownership
  (with-open [arena (Arena/ofConfined)]
    (let [calls (atom [])
          fail? (atom false)
          diagnostic-fail? (atom false)
          program (.allocate arena 8)
          compile! (fn [source compilation]
                     (let [name (str (gensym "compile_test_"))]
                       (ocl/register-kernel! name {:source source :target :opencl-c
                                                   :attributes {:compilation compilation}}
                                             :fixture)
                       (:program ((runtime-var 'ensure-kernel-loaded!) name))))
          native (fn [arity f] (delay (handle arity f)))
          replacements
          {(runtime-var 'ensure-init!) (fn [])
           (runtime-var 'kernel-registry) (atom {})
           (runtime-var 'kernel-dispatch-registry) (atom {})
           (runtime-var 'state) (atom {:arena arena :context MemorySegment/NULL
                                       :device MemorySegment/NULL :device-info {}})
           (runtime-var 'h-clCreateProgramWithSource)
           (native 5 (fn [_ _ _ _ err]
                       (swap! calls conj :create)
                       (.set ^MemorySegment err ValueLayout/JAVA_INT 0 (int 0))
                       program))
           (runtime-var 'h-clBuildProgram)
           (native 6 (fn [_ _ _ options _ _]
                       (swap! calls conj [:build (when-not (= MemorySegment/NULL options)
                                                   (.getString ^MemorySegment options 0))])
                       (if @fail? -11 0)))
           (runtime-var 'h-clCreateKernel)
           (native 3 (fn [_ _ err]
                       (.set ^MemorySegment err ValueLayout/JAVA_INT 0 (int 0)) program))
           (runtime-var 'h-clReleaseKernel) (native 1 (fn [_] 0))
           (runtime-var 'h-clGetProgramBuildInfo)
           (native 6 (fn [_ _ _ size buffer result-size]
                       (when @diagnostic-fail? (throw (ex-info "diagnostic failure" {})))
                       (.set ^MemorySegment result-size ValueLayout/JAVA_LONG 0 (long 5))
                       (when (pos? size) (.setString ^MemorySegment buffer 0 "oops"))
                       0))
           (runtime-var 'h-clReleaseProgram)
           (native 1 (fn [p] (swap! calls conj [:release p]) 0))}]
      (with-redefs-fn replacements
        (fn []
          (try
            (is (= program (compile! "source" {})))
            (is (= [:create [:build "-cl-fp32-correctly-rounded-divide-sqrt"]] @calls))
            (reset! calls [])
            (is (= program (compile! "source" {:language-standard "CL3.0"})))
            (is (= [:create [:build "-cl-fp32-correctly-rounded-divide-sqrt -cl-std=CL3.0"]]
                   @calls))
            (reset! calls [])
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lacks required"
                                  (compile! "source" {:language-standard "CL3.0"
                                                      :extensions #{"cl_missing"}})))
            (is (empty? @calls) "requirements fail before creating a native program")
            (reset! fail? true)
            (doseq [diagnostic-failure [false true]]
              (reset! diagnostic-fail? diagnostic-failure)
              (reset! calls [])
              (is (thrown-with-msg? clojure.lang.ExceptionInfo
                                    (if diagnostic-failure #"diagnostic failure" #"clBuildProgram failed: oops")
                                    (compile! "invalid source" {})))
              (is (= [:create [:build "-cl-fp32-correctly-rounded-divide-sqrt"]
                      [:release program]]
                     @calls)
                  "failed builds release exactly once, even when log retrieval throws"))
            (finally (ocl/close-kernel-arena! :fixture))))))))
