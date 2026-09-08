(ns raster.compiler.passes.parallel.layout-transform-schedule-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.backend.gpu.layout-transform :as layout-emitter]
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

(deftest layout-extents-preserve-independent-integer-widths
  (doseq [extent-dtype [:int :long]]
    (let [kernel (layout-emitter/cast-body
                  {:id :wide-cast :input 'x :output 'y :extent-dtype extent-dtype
                   :source-dtype :float :destination-dtype :half :vector-width 4
                   :rounding :nearest-even :overflow :ieee})]
      (is (= extent-dtype (:dtype (last (:parameters kernel)))))
      (is (= [2] (:group-count (launch/realize (:launch kernel) {:layout-elements 1025}))))
      (doseq [dialect [:opencl-portable :cuda :hip]]
        (is (string? (:source (layout-emitter/emit-cast-kernel
                               {:kernel-name "wide_cast" :input 'x :output 'y
                                :extent-dtype extent-dtype
                                :source-dtype :float :destination-dtype :half
                                :vector-width 4 :rounding :nearest-even :overflow :ieee
                                :target-dialect dialect})))))))
  (doseq [row-dtype [:int :long] column-dtype [:int :long]]
    (let [options {:id :wide-transpose :input 'x :output 'y :element-dtype :float
                   :row-extent-dtype row-dtype :column-extent-dtype column-dtype}
          kernel (layout-emitter/transpose-body options)]
      (is (= [row-dtype column-dtype] (mapv :dtype (drop 2 (:parameters kernel)))))
      (doseq [dialect [:opencl-portable :cuda :hip]]
        (is (string? (:source (layout-emitter/emit-transpose-kernel
                               (assoc options :kernel-name "wide_transpose"
                                      :target-dialect dialect)))))))))

(deftest layout-extents-reject-non-index-representations
  (doseq [invalid [:float :double :half]
          make-kernel [(fn [] (schedule/cast-body
                               {:input 'x :output 'y :extent-dtype invalid
                                :source-dtype :float :destination-dtype :half
                                :rounding :nearest-even :overflow :ieee}))
                       (fn [] (schedule/transpose-body
                               {:input 'x :output 'y :element-dtype :float
                                :row-extent-dtype invalid}))
                       (fn [] (schedule/transpose-body
                               {:input 'x :output 'y :element-dtype :float
                                :column-extent-dtype invalid}))]]
    (try
      (make-kernel)
      (is false "non-index extent dtype must fail before emission")
      (catch clojure.lang.ExceptionInfo e
        (is (= :layout-transform-index-dtype (:reason (ex-data e))))))))

(deftest long-extents-have-checked-non-allocating-launch-boundaries
  (let [cast (schedule/cast-body
              {:id :large-cast :input 'x :output 'y :extent-dtype :long
               :source-dtype :float :destination-dtype :half :vector-width 4
               :rounding :nearest-even :overflow :ieee})
        transpose (schedule/transpose-body
                   {:id :large-transpose :input 'x :output 'y :element-dtype :float
                    :row-extent-dtype :long :column-extent-dtype :long})]
    (is (= [2097152]
           (:group-count (launch/realize (:launch cast) {:layout-elements 2147483648})))
        "a logical extent above int range is retained without allocating that buffer")
    (is (thrown? ArithmeticException
                 (launch/realize (:launch transpose)
                                 {:layout-rows 3037000500 :layout-cols 3037000500}))
        "shape products cannot wrap the checked host launch arithmetic")))
