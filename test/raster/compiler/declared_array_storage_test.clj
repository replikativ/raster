(ns raster.compiler.declared-array-storage-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is use-fixtures]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.core.types :as types]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.compiled :as compiled]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.runtime.hardware :as hardware]))

(def target :ze:declared-array-storage-test)
(def policy storage/policy)

(use-fixtures :each hardware-fixture/isolated
  (fn [f]
    (hardware/register-target-device!
     target {:name "Synthetic mixed storage target"
             :capabilities {:total-eus 32 :subgroup-sizes [16]
                            :max-workgroup-size 1024 :shared-local-memory 65536}})
    (f)))

(defn- pointer-types [artifact]
  (into {} (keep (fn [slot]
                  (when-not (= :scalar (:kind slot)) [(:name slot) (:dtype slot)])))
        (:abi artifact)))

(deftest one-derivation-separates-storage-from-scalar-computation
  (let [params '[weights state n scale] tags '[floats doubles long double]]
    (is (= (opencl-pass/derive-param-types params tags :double)
           (opencl-pass/derive-param-types params tags :double
                                          {:preserve-declared-array-storage? false})))
    (is (= {:scalar-types {'n :long 'scale :float}
            :array-types {'weights :float 'state :double}}
           (opencl-pass/derive-param-types params tags :float policy)))
    (is (= {:scalar-types {'n :long 'scale :double}
            :array-types {'weights :float 'state :double}}
           (opencl-pass/derive-param-types params tags :double policy)))))

(deftest both-public-verticals-retain-mixed-storage-and-typed-loads
  (let [expected {'weights :float 'state :double 'out :double}
        eq (equation-first/compile #'storage/mixed-storage! (merge policy {:target target :dtype :double}))
        resident (pipeline/compile-gpu-program #'storage/mixed-storage! target :dtype :double
                                               :preserve-declared-array-storage? true)
        artifacts (concat (:kernels eq) (map :artifact (:steps resident)))]
    (is (= expected (get-in eq [:options :array-types])))
    (is (= expected (into {} (map (fn [p] [p (get-in eq [:semantic :values p :dtype])]))
                         (keys expected))))
    (is (= 2 (count artifacts)))
    (doseq [artifact artifacts]
      (is (= expected (pointer-types artifact)))
      (is (str/includes? (:source artifact) "const float* restrict weights"))
      (is (str/includes? (:source artifact) "const double* restrict state"))
      (is (every? #(= (:dtype %) (:kernel-dtype %)) (:abi artifact))))
    (let [default (equation-first/compile #'storage/mixed-storage! {:target target :dtype :double})]
      (is (= {'weights :double 'state :double 'out :double}
             (pointer-types (first (:kernels default)))))
      (is (not= (:id default) (:id eq))))))

(deftest internal-scratch-allocation-retains-its-physical-dtype
  (let [eq (equation-first/compile #'storage/mixed-scratch!
                                  (merge policy {:target target :dtype :double}))
        resident (pipeline/compile-gpu-program #'storage/mixed-scratch! target :dtype :double
                                               :preserve-declared-array-storage? true)
        scratch (first (:allocs resident))]
    (is (= 1 (count (:allocs resident))))
    (is (= :double (:dtype scratch)))
    (is (= :double (get-in eq [:semantic :values (:sym scratch) :dtype])))
    (is (= 2 (count (:steps resident))))
    (is (= 2 (count (:kernels eq))))
    (doseq [artifact (concat (:kernels eq) (map :artifact (:steps resident)))]
      (is (= :double (get (pointer-types artifact) (:sym scratch)))))))

(deftest jvm-aot-uses-the-same-declared-storage-facts
  (let [f (pipeline/compile-aot #'storage/mixed-storage! :dtype :double :simd? false
                              :preserve-declared-array-storage? true)
        out (double-array 2)]
    (f (float-array [1.25 2.5]) (double-array [3.0 4.0]) out 2)
    (is (= [4.25 6.5] (vec out)))))

(deftest soa-fields-use-the-same-storage-policy
  (with-redefs [types/soa-registry
                (atom {'Particle {:fields [{:name "x" :element-tag 'float :array-tag 'floats}]}})
                types/soa-reverse-registry (atom {'ParticleSoA 'Particle})]
    (let [source '(raster.par/map-void! i n
                    (aset out i (double (.x (aget ^ParticleSoA particles i)))))
          emit (fn [preserve?]
                 (first (:kernels
                         (opencl-pass/opencl-pass source :device-id target :dtype :double
                                                  :min-elements 0 :array-types {'out :double}
                                                  :scalar-types {'n :long}
                                                  :preserve-declared-array-storage? preserve?))))]
      (is (= :double (get (pointer-types (emit false)) 'particles_x)))
      (is (= :float (get (pointer-types (emit true)) 'particles_x))))))

(deftest public-template-caches-do-not-cross-storage-policies
  (compiled/clear-compilation-cache!)
  (try
    (doseq [compiler [nil :equation-first]]
      (let [opts (cond-> {:target target :dtype :double :outputs '[out]}
                   compiler (assoc :compiler compiler))
            prepare (fn [preserve?]
                      (compiled/lower #'storage/mixed-storage!
                                      [(if preserve? (float-array 4) (double-array 4))
                                       (double-array 4) (double-array 4) 4]
                                      (cond-> opts preserve? (merge policy))))
            default (prepare false)
            mixed (prepare true)
            repeated (prepare true)]
        (is (false? (get-in (compiled/preparation-report default) [:template :cache-hit?])))
        (is (false? (get-in (compiled/preparation-report mixed) [:template :cache-hit?])))
        (is (true? (get-in (compiled/preparation-report repeated) [:template :cache-hit?])))
        (is (= :double (get-in default [:in-tree 0 :dtype])))
        (is (= :float (get-in mixed [:in-tree 0 :dtype])))))
    (finally (compiled/clear-compilation-cache!))))
