(ns raster.compiler.core.calibration-admission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hw]
            [raster.runtime.hardware :as rt]))

(def baseline
  {:bandwidth-bytes-s 100.0 :peak-flops {:f32 200.0 :f64 50.0}
   :launch-overhead-ns 10.0
   :provenance {:bandwidth-bytes-s :derived :peak-flops :derived
                :launch-overhead-ns :derived}})

(defn overlay [measurement]
  (with-redefs [rt/measured-for (constantly measurement)]
    (#'hw/merge-measured baseline :test-device)))

(deftest raw-observations-do-not-become-planning-facts
  (is (= baseline (overlay nil)))
  (doseq [provenance [nil :measured-noisy :derived]
          value [900.0 Double/NaN Double/POSITIVE_INFINITY -1.0 0.0]]
    (let [m {:bandwidth-bytes-s value :provenance {:bandwidth-bytes-s provenance}}
          result (overlay m)]
      (is (= baseline (dissoc result :measured)))
      (is (identical? m (:measured result)))))
  (let [m {:bandwidth-bytes-s 900.0 :provenance {:bandwidth-bytes-s :measured}
           :bench {:bandwidth {:stationary? false}}}]
    (is (= baseline (dissoc (overlay m) :measured)))))

(deftest admission-is-per-field-and-finite
  (doseq [value [nil "fast" Double/NaN Double/NEGATIVE_INFINITY -1.0 0.0]]
    (let [m {:bandwidth-bytes-s value :launch-overhead-ns value
             :peak-flops {:f32 value}
             :provenance {:bandwidth-bytes-s :measured :peak-flops :measured
                          :launch-overhead-ns :measured}}]
      (is (= baseline (dissoc (overlay m) :measured)))))
  (let [m {:bandwidth-bytes-s 900.0 :launch-overhead-ns 20.0
           :peak-flops {:f32 700.0 :f64 Double/NaN}
           :provenance {:bandwidth-bytes-s :measured-noisy :peak-flops :measured
                        :launch-overhead-ns :measured}}
        result (overlay m)]
    (is (= 100.0 (:bandwidth-bytes-s result)))
    (is (= (:peak-flops baseline) (:peak-flops result)))
    (is (= 20.0 (:launch-overhead-ns result)))
    (is (= {:bandwidth-bytes-s :derived :peak-flops :derived
            :launch-overhead-ns :measured} (:provenance result)))
    (is (identical? m (:measured result))))
  (is (= 900.0 (:bandwidth-bytes-s
               (overlay {:bandwidth-bytes-s 900.0
                         :provenance {:bandwidth-bytes-s :measured}
                         :bench {:bandwidth {:stationary? true}}})))))

(deftest peak-families-are-atomic-and-alias-consistent
  (let [desc (assoc baseline :peak-flops {:f32 200.0 :float 200.0 :f64 50.0 :double 50.0})
        apply-peaks (fn [peaks]
                      (with-redefs [rt/measured-for (constantly
                                                     {:peak-flops peaks
                                                      :provenance {:peak-flops :measured}})]
                        (#'hw/merge-measured desc :test-device)))]
    (doseq [peaks [{:f32 700.0} {:f32 700.0 :float 701.0 :f64 80.0}
                  {:f32 700.0 :f64 Double/NaN} {:unknown 3.0}]]
      (is (= desc (dissoc (apply-peaks peaks) :measured))))
    (let [result (apply-peaks {:f32 700.0 :f64 80.0})]
      (is (= 700.0 (hw/peak-flops-for result :float)))
      (is (= 700.0 (hw/peak-flops-for result :f32)))
      (is (= 80.0 (hw/peak-flops-for result :double)))
      (is (= :measured (get-in result [:provenance :peak-flops])))))
  (let [result (overlay {:bandwidth-bytes-s 999.0
                         :provenance {:bandwidth-bytes-s :measured-noisy}})]
    (is (= (hw/evidence-signature baseline) (hw/evidence-signature result)))))
