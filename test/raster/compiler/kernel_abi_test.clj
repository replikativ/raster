(ns raster.compiler.kernel-abi-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.gpu.core]
            [raster.gpu.ze-runtime :as ze]))

(def map-abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'scale :scalar :float :role :parameter)
   (kabi/slot '_n_bound :scalar :int :role :bound)])

(deftest generic-abi-allows-ordered-multi-output
  (let [abi [(kabi/slot 'x :input :float)
             (kabi/slot 'side :output :float :role :secondary-result)
             (kabi/slot 'out :output :float :role :result)
             (kabi/slot '_n_bound :scalar :int :role :bound)]]
    (is (= abi (kabi/validate! abi)))
    (is (= '[x side out] (mapv :name (kabi/pointer-slots abi))))
    (is (= '[_n_bound] (mapv :name (kabi/scalar-slots abi))))))

(deftest emitted-signature-is-checked-structurally
  (is (= map-abi
         (kabi/validate-source-signature!
          "map_k" "__kernel void map_k(__global const float* x, __global float* out, float scale, int _n_bound) {}"
          map-abi)))
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"does not match"
       (kabi/validate-source-signature!
       "map_k" "__kernel void map_k(__global float* out, __global const float* x, float scale, int _n_bound) {}"
        map-abi)))
  (let [unicode-abi [(kabi/slot 'y_α_42 :input :float)
                     (kabi/slot 'out :output :float :role :result)]]
    (is (= unicode-abi
           (kabi/validate-source-signature!
            "unicode_k" "__kernel void unicode_k(__global const float* y_α_42, __global float* out) {}"
            unicode-abi)))))

(deftest split-resident-binding-is-checked-against-abi
  (is (= '[x out]
         (mapv :name (:pointer-slots (kabi/validate-split-binding!
                                     map-abi [:x :out] [{:type :float :value 2.0}])))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"pointer count"
                        (kabi/validate-split-binding! map-abi [:x] [{:type :float :value 2.0}])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar dtype"
                        (kabi/validate-split-binding! map-abi [:x :out]
                                                      [{:type :int :value 2}]))))

(deftest ordered-reduction-binding-is-derived-from-roles
  (let [abi [(kabi/slot 'x :input :float :role :operand)
             (kabi/slot 'out :output :float :role :result)
             (kabi/slot 'scale :scalar :float :role :parameter)
             (kabi/slot '_n_bound :scalar :int :role :bound)]
        binding (kabi/validate-reduction-arguments! abi [:x :out :scale :n])]
    (is (= '[x out] (mapv (comp :name first) (:pointer-pairs binding))))
    (is (= '[scale] (mapv (comp :name first) (:scalar-pairs binding))))
    (is (= 'out (-> binding :result-pair first :name)))
    (is (= '_n_bound (-> binding :bound-pair first :name))))
  (testing "missing or wrongly-kinded semantic roles fail loudly"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exactly one :result"
          (kabi/validate-reduction-arguments!
           [(kabi/slot 'x :input :float)
            (kabi/slot 'out :output :float)
            (kabi/slot '_n_bound :scalar :int :role :bound)]
           [:x :out :n])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #":bound slot must be scalar"
          (kabi/validate-reduction-arguments!
           [(kabi/slot 'x :input :float :role :bound)
            (kabi/slot 'out :output :float :role :result)]
           [:x :out])))))

(deftest session-resolution-consumes-abi-not-legacy-input-list
  (testing "door C resolves the functional output slot instead of binding only :array-params"
    (let [resolve-bufs @#'raster.gpu.core/resolve-kernel-bufs]
      (is (= [:x-buffer :out-buffer]
             (resolve-bufs {:abi map-abi :array-params '[x]}
                           {:x :x-buffer :out :out-buffer}
                           {}))))))

(deftest staging-rejects-marker-abi-mismatch-before-driver-loading
  (let [kernel-name (str "abi_mismatch_" (gensym))]
    (ze/register-kernel! kernel-name {:abi map-abi :dtype :float})
    (try
      ;; Missing x and scale: validation must fail before ensure-kernel-loaded! can initialize
      ;; Level Zero or try compiling a source-less registry entry.
      (let [e (try
                (ze/invoke-registered-kernel kernel-name [] (float-array 1) [] 1)
                nil
                (catch clojure.lang.ExceptionInfo e e))]
        (is (some? e))
        (is (= 4 (:expected (ex-data e))))
        (is (= 2 (:actual (ex-data e)))))
      (finally
        ;; Public registration is intentionally global; overwrite with a harmless tombstone so
        ;; this unique test entry cannot retain native resources (none were created here).
        (ze/register-kernel! kernel-name {:test-tombstone? true})))))

(deftest staging-rejects-pointer-dtype-mismatch-before-driver-loading
  (let [kernel-name (str "abi_dtype_mismatch_" (gensym))]
    (ze/register-kernel! kernel-name {:abi map-abi :dtype :float})
    (try
      (let [e (try
                (ze/invoke-registered-kernel kernel-name [(double-array 1)]
                                             (float-array 1) [1.0] 1)
                nil
                (catch clojure.lang.ExceptionInfo e e))]
        (is (some? e))
        (is (= :float (:expected (ex-data e))))
        (is (= :double (:actual (ex-data e)))))
      (finally
        (ze/register-kernel! kernel-name {:test-tombstone? true})))))

(deftest reduction-staging-rejects-abi-errors-before-driver-loading
  (let [kernel-name (str "reduction_abi_mismatch_" (gensym))
        abi [(kabi/slot 'x :input :float :role :operand)
             (kabi/slot 'out :output :float :role :result)
             (kabi/slot 'scale :scalar :float :role :parameter)
             (kabi/slot '_n_bound :scalar :int :role :bound)]]
    (ze/register-kernel! kernel-name {:abi abi :dtype :float})
    (try
      (testing "a caller-owned staging result is rejected before source-less kernel loading"
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"result value must be nil"
              (ze/invoke-registered-reduction-kernel
               kernel-name [(float-array 4) (float-array 1) 2.0 4]))))
      (testing "input storage dtype is checked before source-less kernel loading"
        (let [e (try
                  (ze/invoke-registered-reduction-kernel
                   kernel-name [(double-array 4) nil 2.0 4])
                  nil
                  (catch clojure.lang.ExceptionInfo e e))]
          (is (some? e))
          (is (= :float (:expected (ex-data e))))
          (is (= :double (:actual (ex-data e))))))
      (finally
        (ze/register-kernel! kernel-name {:test-tombstone? true})))))
