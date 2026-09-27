(ns raster.gpu.ocl-launch-test
  "Native launch marshalling without loading an OpenCL driver. Native layout hints are an
   interop contract: removing them reintroduces reflective overload search between launches."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ocl-runtime :as ocl])
  (:import [java.lang.foreign MemorySegment ValueLayout AddressLayout
            ValueLayout$OfInt ValueLayout$OfLong ValueLayout$OfFloat ValueLayout$OfDouble]))

(deftest native-layout-vars-retain-concrete-overload-types
  (doseq [[symbol expected] [['PTR AddressLayout] ['I32 ValueLayout$OfInt]
                            ['I64 ValueLayout$OfLong] ['F32 ValueLayout$OfFloat]
                            ['F64 ValueLayout$OfDouble]]]
    (let [v (ns-resolve 'raster.gpu.ocl-runtime symbol)
          tag (:tag (meta v))]
      (is (= expected (if (class? tag) tag (ns-resolve 'raster.gpu.ocl-runtime tag))))
      (is (instance? expected @v)))))

(deftest launch-geometry-marshals-one-through-three-dimensions
  (doseq [[wg groups local global] [[8 4 [8] [32]]
                                    [[2 4] [3 5] [2 4] [6 20]]
                                    [[2 4 8] [3 5 7] [2 4 8] [6 20 56]]]]
    (let [seen (atom nil)
          capture (fn [operation _ [_ _ dims _ g l wait-count _ event]]
                    (let [read-longs (fn [^MemorySegment segment]
                                       (mapv #(.get segment ValueLayout/JAVA_LONG (long (* 8 %)))
                                             (range dims)))]
                      (reset! seen {:operation operation :global (read-longs g)
                                    :local (read-longs l) :wait-count wait-count :event event})))]
      (with-redefs-fn {(ns-resolve 'raster.gpu.ocl-runtime 'cl-call!) capture
                      (ns-resolve 'raster.gpu.ocl-runtime 'h-clEnqueueNDRangeKernel) (delay nil)}
        #(#'ocl/enqueue-bound! {:kernel MemorySegment/NULL :wg wg} groups
                              MemorySegment/NULL MemorySegment/NULL))
      (is (= {:operation "clEnqueueNDRangeKernel" :global global :local local
              :wait-count 0 :event MemorySegment/NULL} @seen)))))
