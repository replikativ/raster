(ns raster.compiler.passes.parallel.recursive-packed-stage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]))

(defn- reference [a b super-scale sub-scale]
  (vec
   (for [i (range 2) j (range 3)]
     (reduce
      (fn [outer sb]
        (let [middle
              (reduce
               (fn [mid blk]
                 (let [dot (reduce + 0
                                   (for [t (range 4)]
                                     (* (long (aget ^bytes a (+ (* i 16) (* sb 8) (* blk 4) t)))
                                        (long (aget ^bytes b (+ (* j 16) (* sb 8) (* blk 4) t))))))]
                   (float (+ mid (float (* (float dot)
                                           (aget ^floats sub-scale (+ (* j 2) blk))))))))
               (float 0) (range 2))]
          (float (+ outer (float (* middle (aget ^floats super-scale (+ (* i 2) sb))))))))
      (float 0) (range 2)))))

(deftest public-three-stage-packed-fold-preserves-each-rounding-boundary
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "recursive packed staged contraction")
    (let [a (byte-array (take 32 (cycle [-128 127 3 -5 0 17])))
          b (byte-array (take 48 (cycle [127 -128 -11 0 9])))
          super-scale (float-array [0.13 0.71 0.93 0.27])
          sub-scale (float-array [0.19 0.37 0.59 0.83 0.41 0.67])
          out (float-array (repeat 6 Float/NaN))
          compiled (equation-first/compile #'fixtures/packed-three-stage!
                                          {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compiled [a b super-scale sub-scale out])
          output-id (some (fn [[id node]] (when (identical? out (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (is (= :none (get-in compiled [:stats :fallback])))
        (is (= :staged-scalar (get-in compiled [:kernels 0 :attributes :kernel-body :schedule :strategy])))
        (link/run! executable)
        (is (= (reference a b super-scale sub-scale) (vec (link/download executable output-id))))
        (finally (link/close! executable))))))
