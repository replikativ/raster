(ns raster.compiler.passes.parallel.layout-transform-schedule-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.layout-transform-schedule :as schedule]))

(deftest cast-is-a-typed-unrolled-affine-kernel-body
  (let [kernel (schedule/cast-body
                {:id :cast :input 'x :output 'y :extent-id 'n
                 :source-dtype :float :destination-dtype :half :vector-width 4
                 :rounding :nearest-even :overflow :ieee})]
    (is (body/kernel-body? kernel))
    (is (= [:float :half :int] (mapv :dtype (:parameters kernel))))
    (is (= 4 (count (:masks kernel))))
    (is (= 12 (count (:operations kernel))))
    (is (= [2] (:group-count
                (launch/realize (:launch kernel) {'n 1025})))
        "the masked tail owns a work-item even when the extent is not divisible by four")
    (is (= {:strategy :affine-elementwise-cast
            :elements-per-work-item 4 :rounding :nearest-even :overflow :ieee}
           (:schedule kernel)))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (testing (name dialect)
        (let [source (emitter/emit-scalar-kernel "cast_rows" kernel
                                                 {:target-dialect dialect})]
          (is (str/includes? source "cast_rows"))
          (is (str/includes? source "layout_converted_3")))))))

(deftest transpose-is-an-affine-permutation-not-target-source
  (let [kernel (schedule/transpose-body
                {:id :transpose :input 'x :output 'y
                 :row-extent-id 'm :column-extent-id 'n
                 :element-dtype :half})
        source (emitter/emit-scalar-kernel "transpose_rows" kernel)]
    (is (body/kernel-body? kernel))
    (is (= [1 0] (get-in kernel [:schedule :permutation])))
    (is (= [:layout-row :layout-column]
           (get-in kernel [:operations 0 :coordinates])))
    (is (= [:layout-column :layout-row]
           (get-in kernel [:operations 1 :coordinates])))
    (is (str/includes? source "rstr_layout_column"))
    (is (str/includes? source "rstr_layout_row"))))
