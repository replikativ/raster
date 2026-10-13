(ns raster.compiler.ir.host-device-storage-dependency-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]))

(deftm stored-scalar-continuation
  [input :- (Array float)] :- Double
  (let [out (float-array 1)
        written (par/map-void! i 1 (aset out i (aget input i)))
        result (double (aget out 0))]
    result))

(deftest physical-device-output-is-not-evaluated-on-the-host-during-preparation
  (let [compilation (equation-first/compile #'stored-scalar-continuation
                                           {:target :ocl:0 :dtype :float})
        calls (atom [])
        terminal-id (-> compilation :emitted :equations last :id)
        prior-logical-results (set (mapcat :results
                                          (butlast (get-in compilation [:emitted :equations]))))
        producer (some #(when-not (get-in % [:attributes :host-only]) %)
                       (get-in compilation [:emitted :equations]))
        operation (first (:operations producer))
        expected-storage (set (vals (emitted-equation/physical-results operation)))
        original scalar/evaluate-host-equation
        error
        (with-redefs [scalar/evaluate-host-equation
                      (fn [& arguments]
                        (swap! calls conj (:id (first arguments)))
                        (apply original arguments))]
          (try (equation-first/lower compilation [(float-array [1.2345679])])
               (catch clojure.lang.ExceptionInfo e e)))]
    (is (= :emitted-program-host-device-dependency (:reason (ex-data error))))
    (is (some? terminal-id))
    (is (emitted-equation/emitted-equation? operation) "this fixture exercises direct emission")
    (is (seq expected-storage))
    (is (= expected-storage (:values (ex-data error)))
        "the refusal identifies the exact independently projected physical destination")
    (is (not-any? prior-logical-results (:values (ex-data error)))
        "the regression specifically requires physical aliases, not the existing logical check")
    (is (not-any? #{terminal-id} @calls)
        "independent host prefix work is permitted; the device-dependent terminal is not")))
