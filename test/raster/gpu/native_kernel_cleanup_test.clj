(ns raster.gpu.native-kernel-cleanup-test
  "Hardware-free fault injection through the production KernelCall binders/destructors."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze])
  (:import [java.lang.foreign MemorySegment]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(deftest both-production-binders-own-success-and-failed-native-acquisition
  (doseq [backend [:ze :ocl] fail-build? [false true] fail-release? [false true]]
    (let [namespace (if (= backend :ze) 'raster.gpu.ze-runtime 'raster.gpu.ocl-runtime)
          v #(ns-resolve namespace %)
          artifact (probe/emit-artifact :float :opencl-portable)
          handle (MemorySegment/ofArray (byte-array 8))
          output (MemorySegment/ofArray (float-array 2))
          kernel-call (call/make artifact [output])
          bind! (if (= backend :ze) ze/bind-kernel-call ocl/bind-kernel-call)
          destroy! (if (= backend :ze) ze/destroy-prepared! ocl/destroy-prepared!)
          primary (ex-info "argument binding failed" {})
          native-fault (ex-info "destruction outcome unknown" {})
          releases (atom []) adopted (atom []) acquired (atom 0)
          release-call (fn [context _ args]
                         (swap! releases conj [context (first args)])
                         (when fail-release? (throw native-fault)))
          redefs (merge
                   {(v 'kernel-registry) (atom {(:kernel-name artifact) artifact})
                    (v 'ensure-kernel-loaded!) (fn [_] {:module handle :program handle :entry-name "probe"})
                    (v 'create-kernel-fresh) (fn [& _] (swap! acquired inc) handle)}
                   (if (= backend :ze)
                     {(v 'h-zeKernelDestroy) (delay :fake)
                      (v 'ze-call!) release-call
                      (v 'bind-kernel!) (fn [& _]
                                         (when fail-build? (throw primary))
                                         {:kernel handle :gc-seg (MemorySegment/ofArray (int-array 3))})}
                     {(v 'h-clReleaseKernel) (delay :fake)
                      (v 'cl-call!) release-call
                      (v 'set-kernel-arg-buffer!) (fn [_handle ^long _index _buffer]
                                                   (when fail-build? (throw primary)))}))]
      (with-redefs-fn redefs
        (fn []
          (if fail-build?
            (do
              (is (identical? primary (error-of #(bind! kernel-call {:adopt-cleanup! (fn [owner] (swap! adopted conj owner))}))))
              (is (= (if fail-release? 1 0) (count @adopted)))
              (when fail-release?
                (is (identical? native-fault (error-of #(cleanup/release! (first @adopted)))))))
            (let [prepared (bind! kernel-call {:adopt-cleanup! (fn [owner] (swap! adopted conj owner))})]
              (is (some? (::cleanup/owner prepared)))
              (is (empty? @releases))
              (dotimes [_ 2]
                (is (if fail-release?
                      (identical? native-fault (error-of #(destroy! prepared)))
                      (nil? (destroy! prepared)))))
              (is (= :missing-cleanup-owner
                     (:reason (ex-data (error-of #(destroy! (dissoc prepared ::cleanup/owner)))))))))
          (is (= 1 @acquired))
          (is (= [[(if (= backend :ze) "zeKernelDestroy" "clReleaseKernel") handle]] @releases)))))))
