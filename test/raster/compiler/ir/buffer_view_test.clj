(ns raster.compiler.ir.buffer-view-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as view]))

(def allocation
  (view/allocation {:id :kv-cache :byte-size 4096 :memory-space :device
                    :device :ze:0 :alignment 64 :coherence :host-coherent
                    :ownership :owned}))

(deftest shaped-views-have-stable-byte-ranges
  (let [whole (view/view allocation {:id :whole :dtype :float :shape [16 64]})
        prefix (view/subview whole {:id :prefix :dtype :float :shape [4 64]})
        tail (view/subview whole {:id :tail :byte-offset (* 4 4 64)
                                  :shape [12 64]})]
    (is (= [64 1] (:strides whole)))
    (is (= 4096 (:byte-length whole)))
    (is (view/overlaps? whole prefix))
    (is (view/disjoint? prefix tail))
    (is (view/contiguous? prefix))
    (is (view/same-range? whole
                          (view/view allocation {:id :other-name :dtype :float
                                                 :shape [1024]})))))

(deftest strided-views-state-their-physical-span
  (let [columns (view/view allocation {:id :columns :dtype :float
                                       :shape [4 8] :strides [64 2]})]
    (is (= (* 4 (inc (+ (* 3 64) (* 7 2)))) (:byte-length columns)))
    (is (not (view/contiguous? columns)))))

(deftest views-fail-loud-on-invalid-physical-claims
  (testing "shape is never inferred implicitly in the physical IR"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicit realized shape"
                          (view/view allocation {:dtype :float}))))
  (testing "shape cannot exceed allocation"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds its allocation"
                          (view/view allocation {:dtype :float :shape [1025]}))))
  (testing "byte length cannot contradict shape and strides"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"differs"
                          (view/view allocation {:dtype :float :shape [4]
                                                 :byte-length 12}))))
  (testing "subviews are contained by their base, not merely by the allocation"
    (let [prefix (view/view allocation {:id :prefix :dtype :float :shape [16]})]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds its base"
                            (view/subview prefix {:byte-offset 32 :dtype :float
                                                  :shape [16]}))))))
