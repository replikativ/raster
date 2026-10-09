(ns raster.compiler.core.hardware-scalar-capability-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.runtime.hardware :as runtime]))

(defn- project [target caps source]
  (with-redefs [runtime/device (fn [_] {:type target :capabilities caps :source source})]
    (#'hardware/build-descriptor :synthetic:scalar-capabilities)))

(deftest optional-scalar-support-is-not-inferred-from-performance
  (doseq [caps [{} {:peak-flops {:double 1.0e12}}
                  {:fp64? true :extensions "cl_khr_fp64"}]]
    (let [descriptor (project :ocl caps {:all :catalogued})]
      (is (= :unknown (hardware/scalar-dtype-support descriptor :double)))
      (is (= :unknown (hardware/scalar-dtype-support descriptor :half))))))

(deftest optional-scalar-support-retains-positive-negative-and-unknown-facts
  (let [ocl (project :ocl {:extensions "cl_khr_fp64 other_extension"} {:extensions :detected})
        amd (project :ocl {:extensions "cl_amd_fp64"} {:extensions :observed})
        ze (project :ze {:fp64? false :fp16? true} {:fp64? :detected :fp16? :detected})
        partial (project :ze {:fp64? true} {:fp64? :detected})]
    (is (= :supported (hardware/scalar-dtype-support ocl :f64)))
    (is (= :unsupported (hardware/scalar-dtype-support ocl :half)))
    (is (= :detected (get-in ocl [:execution :provenance :scalar-dtype-support :double])))
    (is (= :supported (hardware/scalar-dtype-support amd :double)))
    (is (= :unsupported (hardware/scalar-dtype-support ze :double)))
    (is (= :supported (hardware/scalar-dtype-support ze :half)))
    (is (= :supported (hardware/scalar-dtype-support partial :double)))
    (is (= :unknown (hardware/scalar-dtype-support partial :half)))))

(deftest explicit-cross-compiler-dtype-sets-are-canonical-and-closed
  (let [descriptor (project :cuda {:scalar-dtypes #{:f32 :f64 :i32}} {:scalar-dtypes :user})]
    (is (= :supported (hardware/scalar-dtype-support descriptor :double)))
    (is (= :unsupported (hardware/scalar-dtype-support descriptor :half)))
    (is (= :user (get-in descriptor [:execution :provenance :scalar-dtype-support :double]))))
  (doseq [[caps source reason]
          [[{:scalar-dtypes [:float :double]} {:scalar-dtypes :user} :hardware-scalar-dtypes-invalid]
           [{:scalar-dtypes #{:short}} {:scalar-dtypes :user} :hardware-scalar-dtypes-invalid]
           [{:fp64? 1} {:fp64? :detected} :hardware-scalar-flag-invalid]
           [{:extensions nil} {:extensions :detected} :hardware-scalar-extensions-invalid]
           [{:fp64? false :extensions "cl_khr_fp64"}
            {:fp64? :detected :extensions :detected} :hardware-scalar-dtype-conflict]]]
    (is (= reason (try (project :ocl caps source)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest frozen-evidence-distinguishes-supported-unsupported-and-unknown
  (let [descriptors [(project :ze {:fp64? true} {:fp64? :detected})
                     (project :ze {:fp64? false} {:fp64? :detected})
                     (project :ze {} {})]]
    (is (= [:supported :unsupported :unknown]
           (mapv #(hardware/scalar-dtype-support % :double) descriptors)))
    (is (= 3 (count (distinct (map hardware/evidence-signature descriptors)))))))
