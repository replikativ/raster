(ns raster.compiler.declared-array-storage-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is use-fixtures]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.core.types :as types]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
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

(deftest mixed-scale-retains-independent-scalar-and-storage-dtypes
  (let [compiled (equation-first/compile
                  #'storage/mixed-scale (merge policy {:target target :dtype :double}))
        artifact (first (:kernels compiled))]
    (is (= 1 (count (:kernels compiled))))
    (is (= :float (get-in compiled [:options :array-types 'values])))
    (is (= :double (get-in compiled [:options :scalar-types 'coefficient])))
    (is (every? #(= :float (:dtype %))
                (remove #(= :scalar (:kind %)) (:abi artifact))))
    (is (= :double (:dtype (first (filter #(= 'coefficient (:name %))
                                        (:abi artifact))))))))

(deftest projected-scalar-reduction-declines-an-unrepresented-device-host-gap
  ;; Preserve this public reproduction while scalar-value projection joins the numerical
  ;; equation vertical. A successful JVM adjoint is not yet a whole-program GPU claim.
  (is (= (float 6.25)
         (storage/mixed-scale-energy-gradient (float 2.0) (double-array [1.5 -2.0]))))
  (let [failure (try
                  (compiled/lower
                   #'storage/mixed-scale-energy-gradient
                   [(float 2.0) (double-array [1.5 -2.0])]
                   (merge policy {:compiler :equation-first :target target :dtype :double}))
                  nil
                  (catch clojure.lang.ExceptionInfo error error))]
    (is (some? failure))
    (let [data (ex-data failure)
          region-gap? (and (= :scheduled-equation-region (:reason data))
                           (seq (:host-gap data)))
          ;; Cold emission can stop one boundary earlier: an untyped dynamic scalar
          ;; cotangent retains its nil-safe projection helper. A warm specialization
          ;; reaches the numerical region gap instead. Neither is GPU support.
          dynamic-projection? (and (= {'raster.ad.tangent/project-float 1}
                                       (:undevirtualized data))
                                   (= {} (:non-exempt-untagged data)))]
      (is (or region-gap? dynamic-projection?)
          (str (some-> failure .getMessage) " " data)))))

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

(deftest converted-double-fold-retains-parallel-row-ownership
  (let [eq (equation-first/compile #'storage/mixed-fold-storage!
                                  (merge policy {:target target :dtype :double}))
        resident (pipeline/compile-gpu-program #'storage/mixed-fold-storage! target :dtype :double
                                               :preserve-declared-array-storage? true)]
    (is (= 1 (count (:kernels eq))))
    (is (= 1 (count (:steps resident))))
    (doseq [artifact (concat (:kernels eq) (map :artifact (:steps resident)))]
      (is (= {'storage :float} (pointer-types artifact)))
      (is (= :independent (get-in artifact [:provenance :scheduled-operation :body :schedule :association]))))))

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

(deftest session-kernel-cache-does-not-cross-storage-policies
  (let [session (atom {:device-id target}) calls (atom [])]
    (with-redefs-fn {#'gpu/compile-deftm-internal!
                    (fn [_ _ opts]
                      (swap! calls conj opts)
                      {:kernels [] :dispatches []})}
      #(do
         (gpu/compile! session :default #'storage/mixed-storage! {:dtype :double})
         (gpu/compile! session :preserved #'storage/mixed-storage! (assoc policy :dtype :double))
         (gpu/compile! session :again #'storage/mixed-storage! (assoc policy :dtype :double))
         (is (= 2 (count @calls)))
         (is (= [nil true] (mapv :preserve-declared-array-storage? @calls)))))))
