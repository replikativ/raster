(ns raster.compiler.kernel-abi-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.core]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

(def map-abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'scale :scalar :float :role :parameter)
   (kabi/slot '_n_bound :scalar :int :role :bound)])

(defn- stable-body []
  (body/make
   {:id :stable-abi
    :parameters [(body/->KernelParameter 'x :input :float [1] :global
                                         (layout/row-major [1] :float) :operand)
                 (body/->KernelParameter 'out :output :float [1] :global
                                         (layout/row-major [1] :float) :result)]
    :stable-reads [(body/stable-read 'x)]
    :launch (launch/spec {:workgroup-size [1] :group-count [1]})}))

(deftest stable-read-contracts-project-to-one-target-input
  (let [projected (body-abi/project-contracts
                   [(kabi/slot 'x :input :float :c-name "source")
                    (kabi/slot 'out :output :float :c-name "destination")]
                   (stable-body))]
    (is (= :no-write-alias (get-in projected [0 :aliasing])))
    (is (nil? (get-in projected [1 :aliasing])))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"exactly one target ABI input"
         (body-abi/project-contracts
          [(kabi/slot 'renamed :input :float)
           (kabi/slot 'out :output :float)]
          (stable-body)))))
  (testing "the contract is only legal on input pointers"
    (doseq [slot [(kabi/slot 'out :output :float :aliasing :no-write-alias)
                  (kabi/slot 'n :scalar :int :aliasing :no-write-alias)
                  (kabi/slot 'x :input :float :aliasing :unknown)]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"aliasing contract"
                            (kabi/validate! [slot (kabi/slot 'out :output :float)]))))))

(deftest required-async-overlap-projects-pointer-alignment
  (let [kernel-body
        (body/make
         {:id :async-alignment-abi
          :parameters [(body/->KernelParameter 'x :input :float [16] :global
                                               (layout/row-major [16] :float) :operand)
                       (body/->KernelParameter 'out :output :float [16] :global
                                               (layout/row-major [16] :float) :result)]
          :stable-reads [(body/stable-read 'x)]
          :allocations [(body/->WorkgroupAllocation
                         'scratch :float [16] (layout/row-major [16] :float) 16)]
          :operations [(body/->AsyncWorkgroupCopy
                        'copy 'x [0] 'scratch [0] 16 16 :cached :required
                        (body/full-participation))
                       (body/->AsyncCommit 'group ['copy])
                       (body/->AsyncWait ['group] 0 :acquire (body/full-participation))
                       (body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                                                (body/full-participation))]
          :launch (launch/spec {:workgroup-size [16] :group-count [1]
                                :shared-memory-bytes 64})})
        projected (body-abi/project-contracts
                   [(kabi/slot 'x :input :float)
                    (kabi/slot 'out :output :float)]
                   kernel-body)]
    (is (= 16 (get-in projected [0 :alignment])))
    (is (= :no-write-alias (get-in projected [0 :aliasing])))
    (is (nil? (get-in projected [1 :alignment])))
    (testing "alignment is a pointer-only power-of-two ABI contract"
      (doseq [slot [(kabi/slot 'n :scalar :int :alignment 16)
                    (kabi/slot 'x :input :float :alignment 3)]]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"positive power of two"
             (kabi/validate! [slot (kabi/slot 'out :output :float)])))))))

(deftest stable-input-aliases-are-checked-against-every-output
  (let [abi [(kabi/slot 'x :input :float :aliasing :no-write-alias)
             (kabi/slot 'side :output :float)
             (kabi/slot 'out :output :float)]]
    (is (= [:x :side :out]
           (kabi/validate-alias-contracts! abi [:x :side :out] =)))
    (let [e (try
              (kabi/validate-alias-contracts! abi [:same :side :same] =)
              nil
              (catch clojure.lang.ExceptionInfo e e))]
      (is (= :kernel-abi-no-write-alias (:reason (ex-data e)))))))

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

(deftest soa-physical-slots-collapse-to-one-logical-binding
  (let [abi [(kabi/slot 'particles_x :output :float :binding 'particles :field :x :role :inout)
             (kabi/slot 'particles_id :output :int :binding 'particles :field :id :role :inout)
             (kabi/slot '_n_bound :scalar :int :role :bound)]]
    (is (= '[particles] (kabi/pointer-binding-names abi)))
    (is (= '[particles]
           (:pointer-bindings (kabi/validate-split-binding! abi [:gpu-soa] []))))
    (is (= [:float :int]
           (kabi/validate-physical-pointer-dtypes! abi [:float :int])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"physical pointer count"
                          (kabi/validate-physical-pointer-dtypes! abi [:float])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"storage dtype"
                          (kabi/validate-physical-pointer-dtypes! abi [:float :float]))))
  (testing "a logical SoA cannot be split around another physical binding"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be contiguous"
                          (kabi/validate!
                           [(kabi/slot 'particles_x :output :float :binding 'particles :field :x)
                            (kabi/slot 'other :input :float)
                            (kabi/slot 'particles_id :output :int :binding 'particles
                                       :field :id)])))))

(deftest repeated-unbound-name-remains-positional
  (testing "an in-place generic map may bind the same value as an input and the result"
    (let [abi [(kabi/slot 'x :input :float :role :inout)
               (kabi/slot 'other :input :float :role :operand)
               (kabi/slot 'x :output :float :c-name "out" :role :result)
               (kabi/slot '_n_bound :scalar :int :role :bound)]]
      (is (= abi (kabi/validate! abi)))
      (is (= '[x other x] (kabi/pointer-binding-names abi)))
      (is (= '[x other x]
             (:pointer-bindings
              (kabi/validate-split-binding! abi [:x-in :other :x-out] [])))))))

(deftest access-kind-and-semantic-role-are-orthogonal
  (let [slot (kabi/slot 'state :inout :float :c-name "out" :role :result)
        abi [slot (kabi/slot '_n_bound :scalar :int :role :bound)]]
    (is (= abi (kabi/validate! abi)))
    (is (= :read-write (kabi/slot-access slot)))
    (is (kabi/readable? slot))
    (is (kabi/writable? slot))
    (is (= '[state] (kabi/pointer-binding-names abi)))
    (is (= '[state]
           (:pointer-bindings (kabi/validate-split-binding! abi [:state] []))))))

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

(deftest staging-refuses-to-invent-a-map-abi
  (let [kernel-name (str "missing_abi_" (gensym))]
    (ze/register-kernel! kernel-name {:dtype :float})
    (try
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"has no ordered ABI"
                            (ze/invoke-registered-kernel
                             kernel-name [] (float-array 1) [] 1)))
      (finally
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

(deftest map-void-staging-rejects-abi-errors-before-driver-loading
  (let [kernel-name (str "map_void_abi_mismatch_" (gensym))
        abi [(kabi/slot 'x :input :float :role :operand)
             (kabi/slot 'out :output :int :role :effect)
             (kabi/slot 'limit :scalar :int :role :parameter)
             (kabi/slot '_n_bound :scalar :int :role :bound)]]
    (ze/register-kernel! kernel-name {:abi abi :dtype :float})
    (try
      (testing "typed scalar mismatch is rejected before source-less kernel loading"
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar dtype"
              (ze/invoke-registered-map-void-kernel
               kernel-name [(float-array 1) (int-array 1)]
               [{:type :float :value 1.0}] 1))))
      (testing "mixed pointer storage mismatch is rejected before source-less kernel loading"
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"storage dtype"
              (ze/invoke-registered-map-void-kernel
               kernel-name [(float-array 1) (float-array 1)]
               [{:type :int :value 1}] 1))))
      (finally
        (ze/register-kernel! kernel-name {:test-tombstone? true})))))

(deftest opencl-map-void-staging-rejects-abi-errors-before-driver-loading
  (let [kernel-name (str "ocl_map_void_abi_mismatch_" (gensym))
        abi [(kabi/slot 'x :input :float :role :operand)
             (kabi/slot 'out :output :int :role :effect)
             (kabi/slot 'limit :scalar :int :role :parameter)
             (kabi/slot '_n_bound :scalar :int :role :bound)]]
    (ocl/register-kernel! kernel-name {:abi abi :dtype :float})
    (try
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"storage dtype"
            (ocl/invoke-registered-map-void-kernel
             kernel-name [(float-array 1) (float-array 1)]
             [{:type :int :value 1}] 1)))
      (finally
        (ocl/register-kernel! kernel-name {:test-tombstone? true})))))

(deftest direct-compiled-map-staging-range-checks-every-physical-int-before-driver-loading
  (let [abi [(kabi/slot 'x :input :float :role :operand)
             (kabi/slot 'out :output :float :role :result)
             (kabi/slot 'offset :scalar :int :role :parameter)
             (kabi/slot '_n_bound :scalar :long :kernel-dtype :int :role :bound)]
        too-large (inc (long Integer/MAX_VALUE))]
    (doseq [[runtime register! invoke!]
            [[:level-zero ze/register-kernel! ze/invoke-registered-kernel]
             [:opencl ocl/register-kernel! ocl/invoke-registered-kernel]]]
      (let [kernel-name (str "direct_int_range_" (name runtime) "_" (gensym))]
        (register! kernel-name {:abi abi :dtype :float})
        (try
          (testing (str (name runtime) " captured scalar")
            (is (thrown? ArithmeticException
                  (invoke! kernel-name [(float-array 1)] (float-array 1)
                           [too-large] 1))))
          (testing (str (name runtime) " logical-long/physical-int bound")
            (is (thrown? ArithmeticException
                  (invoke! kernel-name [(float-array 1)] (float-array 1)
                           [0] too-large))))
          (finally
            (register! kernel-name {:test-tombstone? true})))))))

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
