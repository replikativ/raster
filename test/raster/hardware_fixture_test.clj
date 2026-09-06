(ns raster.hardware-fixture-test
  (:require [clojure.test :refer [deftest is]]
            [raster.hardware-fixture :as fixture]
            [raster.runtime.hardware :as hardware]))

(deftest registry-isolation-restores-identities-and-values
  (let [registries [#'raster.runtime.hardware/device-registry
                    #'raster.runtime.hardware/initialized?
                    #'raster.runtime.hardware/measured-registry]
        original-atoms (mapv deref registries)
        original-values (mapv deref original-atoms)]
    (doseq [fail? [false true]]
      (let [run #(fixture/isolated
                  (fn []
                    (is (= [{} false {}] (mapv (comp deref deref) registries)))
                    ;; Avoid probing hardware: this test checks fixture state, not discovery.
                    (reset! @#'raster.runtime.hardware/initialized? true)
                    (hardware/register-target-device! :ze:987 {:name "fixture-only"
                                                              :capabilities {}})
                    (hardware/set-measured! :ze:987 {:bandwidth 17})
                    (fixture/isolated
                     (fn [] (is (= [{} false {}] (mapv (comp deref deref) registries)))))
                    (is (= "fixture-only" (:name (hardware/device :ze:987))))
                    (is (= {:bandwidth 17} (hardware/measured-for :ze:987)))
                    (if fail? (throw (ex-info "fixture failure" {})) :returned)))]
        (if fail?
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fixture failure" (run)))
          (is (= :returned (run))))
        (is (every? true? (map identical? original-atoms (map deref registries))))
        (is (= original-values (mapv (comp deref deref) registries)))))))
