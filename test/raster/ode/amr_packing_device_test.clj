(ns raster.ode.amr-packing-device-test
  "Hierarchy-derived resident packing through existing typed gather/scatter operations.
   This is not subcycling, reflux, or a new patch ownership/codec API."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.build-manifest :as build]
            [raster.dl.array-ops :as ops]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.completed-evidence-device-test :as producer]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.ode.amr-geometry :as geometry]
            [raster.ode.amr-geometry-test :as geometry-oracle]))

(deftm patch-packing-roundtrip!
  [base :- (Array double), fine :- (Array double),
   base-local :- (Array int), base-packed :- (Array int),
   fine-local :- (Array int), fine-packed :- (Array int),
   base-active :- (Array double), fine-active :- (Array double),
   packed :- (Array double), base-out :- (Array double), fine-out :- (Array double),
   nb :- Long, nf :- Long, base-size :- Long, fine-size :- Long, packed-size :- Long] :- Void
  (ops/gather-blocks! base base-local base-active nb 1 base-size)
  (ops/scatter-blocks! base-active base-packed packed nb 1 packed-size)
  (ops/gather-blocks! fine fine-local fine-active nf 1 fine-size)
  (ops/scatter-blocks! fine-active fine-packed packed nf 1 packed-size)
  (ops/gather-blocks! packed base-packed base-active nb 1 packed-size)
  (ops/scatter-blocks! base-active base-local base-out nb 1 base-size)
  (ops/gather-blocks! packed fine-packed fine-active nf 1 packed-size)
  (ops/scatter-blocks! fine-active fine-local fine-out nf 1 fine-size))

(defn- bits [values]
  (mapv #(Double/doubleToRawLongBits (double %)) values))

(defn- run-packing! [target]
  (with-redefs [build/current-identity #'producer/test-build]
    (let [hierarchy (#'geometry-oracle/hierarchy
                     [4 4] [{:ratio [2 2] :patches [{:offsets [2 2] :shape [4 4]}]}])
          projection (geometry/project-hierarchy hierarchy {:domain-lengths [1.0 1.0]})
          {:keys [patch-rows cell-count]} (geometry/materialize-connectivity projection)
          [base-row fine-row] patch-rows
          nb (count (:local-indices base-row)) nf (count (:local-indices fine-row))
          source (fn [offset]
                   (double-array (map #(if (zero? %) -0.0 (+ offset (* 0.125 %))) (range 16))))
          base (source 100.0) fine (source 200.0)
          base-out (double-array (repeat 16 -777.0)) fine-out (double-array (repeat 16 -888.0))
          packed (double-array cell-count)
          base-active (double-array nb) fine-active (double-array nf)
          args [base fine (:local-indices base-row) (:packed-indices base-row)
                (:local-indices fine-row) (:packed-indices fine-row)
                base-active fine-active packed base-out fine-out nb nf 16 16 cell-count]
          source-by-patch {(:patch base-row) base (:patch fine-row) fine}
          ;; Independent host indexing oracle, not the gather/scatter implementation.
          expected-packed (mapv (fn [{:keys [patch local-index]}]
                                  (aget ^doubles (get source-by-patch patch) local-index))
                                (:cells projection))
          active-base (set (:local-indices base-row))
          expected-base (mapv #(if (contains? active-base %)
                                 (aget base %) -777.0) (range 16))
          expected-fine (vec fine)
          cpu-args (mapv #(if (.isArray (class %)) (aclone %) %) args)]
      (is (= [12 16 28] [nb nf cell-count]))
      ;; Geometry-generated destinations are disjoint and cover all packed active cells.
      (is (= (set (range cell-count))
             (set (concat (:packed-indices base-row) (:packed-indices fine-row)))))
      (doseq [[src indices out n capacity direction]
              [[base (:local-indices base-row) base-active nb 16 :gather]
               [base-active (:packed-indices base-row) packed nb cell-count :scatter]
               [fine (:local-indices fine-row) fine-active nf 16 :gather]
               [fine-active (:packed-indices fine-row) packed nf cell-count :scatter]
               [packed (:packed-indices base-row) base-active nb cell-count :gather]
               [base-active (:local-indices base-row) base-out nb 16 :scatter]
               [packed (:packed-indices fine-row) fine-active nf cell-count :gather]
               [fine-active (:local-indices fine-row) fine-out nf 16 :scatter]]]
        (ops/validate-block-transfer! direction src indices out n 1 capacity))
      (apply patch-packing-roundtrip! cpu-args)
      (is (= (bits expected-packed) (bits (nth cpu-args 8))))
      (is (= (bits expected-base) (bits (nth cpu-args 9))))
      (is (= (bits expected-fine) (bits (nth cpu-args 10))))
      (let [prepared (compiled/lower #'patch-packing-roundtrip! args
                                     {:target target :compiler :equation-first :dtype :double
                                      :inline? true :outputs '[packed base-out fine-out]})
            c (compiled/instantiate! prepared)]
        (try
          (with-open [receipt (compiled/invoke-with-evidence c {})]
            (let [values (into {} (map (fn [{:keys [key node]}]
                                         [key (link/download (:executable c) node)]))
                               (:out-tree prepared))]
              (is (= (bits expected-packed) (bits (get values :packed))))
              (is (= (bits expected-base) (bits (get values :base-out))))
              (is (= (bits expected-fine) (bits (get values :fine-out))))
              (is (= 3 (count (:outputs @receipt))))))
          (finally (compiled/close! c)))))))

(deftest hierarchy-patch-packing-on-opencl
  (if @opencl/opencl-fp64-available?
    (run-packing! :ocl:0)
    (opencl/opencl-skip! "FP64 hierarchy patch packing" :fp64)))

(deftest hierarchy-patch-packing-on-level-zero
  (if @ze/gpu-available?
    (run-packing! :ze:0)
    (ze/gpu-skip! "FP64 hierarchy patch packing")))
