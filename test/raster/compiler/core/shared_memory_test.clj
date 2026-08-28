(ns raster.compiler.core.shared-memory-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.shared-memory :as shared]))

(def banks32 {:count 32 :word-bytes 4 :provenance :test})

(deftest finite-xor-layouts-are-exact-bijections
  (is (= #{:identity :xor-2 :xor-4 :xor-8 :xor-16 :xor-32}
         layout/shared-memory-swizzles))
  (doseq [swizzle layout/shared-memory-swizzles]
    (let [descriptor (layout/shared-memory [32 64] :float swizzle)
          offsets (for [row (range 32) column (range 64)]
                    (layout/layout->offset descriptor [row column]))]
      (is (= (set (range (* 32 64))) (set offsets)) (name swizzle))))
  (testing "the family cannot be extended by arbitrary masks or invalid tile periods"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unknown shared-memory swizzle"
                          (layout/shared-memory [32 32] :float :xor-user)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid shared-memory layout"
                          (layout/shared-memory [8 24] :float :xor-16)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid shared-memory layout"
                          (layout/shared-memory [32] :float :xor-32)))))

(deftest xor-layout-removes-a-modeled-column-bank-conflict
  (let [column-access (mapv #(vector % 0) (range 32))
        dense (shared/analyze-access banks32
                                      (layout/shared-memory [32 32] :float :identity)
                                      column-access)
        swizzled (shared/analyze-access banks32
                                         (layout/shared-memory [32 32] :float :xor-32)
                                         column-access)]
    (is (= 32 (:max-conflict-degree dense)))
    (is (= 1 (:max-conflict-degree swizzled)))
    (is (= (set (range 32)) (set (:banks swizzled))))))

(deftest broadcasts-resource-charge-and-abstention-are-explicit
  (let [descriptor (layout/shared-memory [32 32] :float :xor-32)
        broadcast (shared/analyze-access banks32 descriptor (vec (repeat 32 [0 0])))
        report (shared/resource-model banks32 descriptor
                                      (mapv #(vector % 0) (range 32))
                                      :capacity 2048)]
    (is (= 1 (:max-conflict-degree broadcast)))
    (is (= {0 32} (:broadcast-address-groups broadcast)))
    (is (= 4096 (:physical-bytes report)))
    (is (= 0 (:padding-bytes report)))
    (is (false? (:feasible? report)))
    (is (= {:status :unavailable
            :reason :shared-memory-bank-topology-unavailable
            :lanes 1 :swizzle :xor-32}
           (shared/analyze-access nil descriptor [[0 0]])))))

(deftest wide-elements-account-for-every-bank-word-touched
  (let [descriptor (layout/shared-memory [2] :double :identity)
        report (shared/analyze-access banks32 descriptor [[0] [1]])]
    (is (= [[0 1] [2 3]] (:banks-per-lane report)))
    (is (nil? (:banks report)))
    (is (= {0 1, 1 1, 2 1, 3 1} (:transactions-by-bank report)))
    (is (= 1 (:max-conflict-degree report)))))
