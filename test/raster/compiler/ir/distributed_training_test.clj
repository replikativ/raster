(ns raster.compiler.ir.distributed-training-test
  "Hardware-free AD/collective/SGD composition and independent numerical oracles.
   Retained in the OpenCL acceptance job as a compiler companion to native tests."
  {:raster.test/opencl-gate true}
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.normal-return :as normal-return]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.compiler.fixtures.distributed-training
             :refer [batches initial-theta local-loss local-gradient-sum multi-step-training-plan
                     value-node rounded-gradient-sum max-error expected-updates
                     training-plan analytic-gradient-sum]]))

(deftest finite-training-oracle-rejects-incomplete-and-nonfinite-results
  (is (zero? (max-error [] [])))
  (is (zero? (max-error (float-array [1 2]) (double-array [1 2]))))
  (is (= 0.25 (max-error [1.0 2.0] [1.25 2.0])))
  (doseq [[expected actual] [[[] [0]] [[0] []] [[1 2] [1]] [[1] [1 2]]]]
    (is (= {:reason :training-oracle-length
            :expected (count expected) :actual (count actual)}
           (try (max-error expected actual) nil
                (catch clojure.lang.ExceptionInfo error (ex-data error))))))
  (doseq [value [Double/NaN Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]
          side [:expected :actual]]
    (is (= {:reason :training-oracle-nonfinite :side side :index 1}
           (try (max-error [0.0 (if (= side :expected) value 1.0)]
                           [0.0 (if (= side :actual) value 1.0)])
                nil
                (catch clojure.lang.ExceptionInfo error (ex-data error)))))))

(deftest repeated-local-ad-matches-rounded-independent-oracle
  (let [prepared (reverse/prepare-value+grad #'local-loss [0])
        execute ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared)]
    (let [body (:walked-body prepared)
          emission (:emission (reverse/prepared-bytecode-entry prepared))
          completions (filter completion/canonical-form? (tree-seq coll? seq body))]
      (is (seq completions))
      (doseq [expression completions]
        (is (some? (bytecode/retained-input-completion-context emission body expression))
            "every physical completion in the unchanged training module retains its emitted context"))
      (let [method (:raster.compiler.backend.jvm.generated-body/entry-method prepared)
            ;; The generated Method, not desired dtypes, supplies entry carriers.
            parameter-facts
            (into {} (keep (fn [[parameter ^Class carrier]]
                             (when (.isPrimitive carrier)
                               [parameter {:carrier carrier :presence :present
                                           :builtin-class (get types/primitive-info
                                                               (symbol (.getName carrier)))}])))
                  (map vector (:params prepared) (.getParameterTypes ^java.lang.reflect.Method method)))
            facts (normal-return/completed-scalar-bindings
                   (mapv vec (partition 2 (second (first body)))) parameter-facts (:source-ns prepared)
                   (fn [expression]
                     (:result-fact (bytecode/retained-input-completion-context emission body expression))))
            loss (first (last (first body)))]
        (is (= {:carrier Object :presence :present :builtin-class Double} (get facts loss))
            "the actual boxed loss composes through shared scalar analysis without erasing its evaluation")))
    (doseq [row-counts [[1 3] [1 2 4]]
            theta (cons (initial-theta) (expected-updates row-counts 2))
            {:keys [rows x y] :as batch} (batches row-counts)]
      (let [expected (rounded-gradient-sum theta batch)
            [primal gradient & inactive] (execute theta x y rows 17)]
        (is (< (max-error expected (local-gradient-sum theta x y rows 17)) 1.0e-7))
        ;; Reuse the same prepared runtime across different rows, data and
        ;; parameters. This qualifies preparation, not resident GPU execution.
        (is (= primal (local-loss theta x y rows 17)))
        (is (= [nil nil nil nil] (vec inactive)))
        (is (< (max-error expected
                          (mapv #(float (* (double rows) (double %))) gradient))
               1.0e-7))))))

(deftest finite-training-composition-retains-parameters-and-orders-updates
  (let [device :ocl:analytic
        descriptor {:device-id device :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}]
    (with-redefs [hardware/descriptor-for (constantly descriptor)
                  gpu/make-session (fn [& _] (throw (ex-info "assembly opened a session" {})))
                  gpu/alloc! (fn [& _] (throw (ex-info "assembly allocated device storage" {})))]
      (doseq [row-counts [[1 3] [1 2 4]]]
        (let [plan (multi-step-training-plan device row-counts 3)
              workers (get-in plan [:mesh :devices])
              parameter-allocations
              (into #{} (for [worker workers
                              :let [step [[:epoch 0] [worker :produce]]
                                    local (get-in plan [:device-plans worker :entries step :link-plan])]
                              [id binding] (get-in plan [:device-plans worker :steps step :bindings])
                              :when (= [worker :parameters] (:value binding))]
                          (get-in (value-node local id) [:view :allocation :id])))
              scratch-by-epoch
              (mapv (fn [epoch]
                      (set/difference
                       (into #{} (for [[_ worker-plan] (:device-plans plan)
                                       [[scope _] entry] (:entries worker-plan)
                                       :when (= [:epoch epoch] scope)
                                       [_ node] (get-in entry [:link-plan :nodes])
                                       :let [id (get-in node [:view :allocation :id])]
                                       :when id] id))
                       parameter-allocations)) (range 3))]
          (is (= 3 (count (:refinements plan))))
          (is (= (mapv #(vector [:epoch 2] [% :update]) workers) (:outputs plan)))
          (is (= (distributed/certify plan) (distributed/verify! (distributed/certify plan))))
          (is (= (count workers) (count parameter-allocations)))
          (is (every? seq scratch-by-epoch))
          (doseq [left (range 3) right (range (inc left) 3)]
            (is (empty? (set/intersection (nth scratch-by-epoch left) (nth scratch-by-epoch right)))
                "private scratch and immutable contribution storage are fresh across epochs"))
          (is (thrown? clojure.lang.ExceptionInfo
                       (distributed/check-readiness
                        (update plan :steps
                                #(mapv (fn [step]
                                         (if (= [[:epoch 1] [(first workers) :produce]] (:id step))
                                           (assoc step :dependencies []) step)) %))))
              "an epoch cannot read parameters unordered with respect to the preceding update")
          (doseq [worker workers]
            (let [parameters (for [epoch (range 3) kind [:produce :update]
                                   :let [step [[:epoch epoch] [worker kind]]
                                         local (get-in plan [:device-plans worker :entries step :link-plan])]
                                   [id binding] (get-in plan [:device-plans worker :steps step :bindings])
                                   :when (= [worker :parameters] (:value binding))]
                               (value-node local id))]
              (is (= 6 (count parameters)))
              (is (apply = (map #(dissoc (:view %) :id) parameters)))
              (is (every? #(identical? (:source (first parameters)) (:source %)) parameters)))
            (doseq [epoch [1 2]]
              (is (= [[[:epoch (dec epoch)] [worker :update]]]
                     (:dependencies (first (filter #(= [[:epoch epoch] [worker :produce]] (:id %))
                                                   (:steps plan)))))))))))))

(deftest unequal-batch-local-ad-producers-match-independent-analytic-gradients
  (doseq [rows [[1 3] [1 2 4]]
          {:keys [x y rows] :as batch} (batches rows)]
    (let [actual (local-gradient-sum (initial-theta) x y rows 17)]
      (is (= 17 (alength actual)))
      (is (< (max-error (analytic-gradient-sum batch) actual) 1.0e-7)))))

(deftest differentiated-training-assembly-certifies-without-device-resources
  (let [device :ocl:analytic
        descriptor {:device-id device :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}]
    (with-redefs [hardware/descriptor-for (fn [target]
                                         (assert (= device target))
                                         descriptor)
                  gpu/make-session (fn [& _] (throw (ex-info "hardware-free assembly opened a session" {})))
                  gpu/alloc! (fn [& _] (throw (ex-info "hardware-free assembly allocated device storage" {})))
                  compiled/instantiate! (fn [& _] (throw (ex-info "hardware-free assembly realized a program" {})))]
      (doseq [row-counts [[1 3] [1 2 4]]]
        (let [n (count row-counts)
              plan (training-plan device row-counts)
              certified (distributed/certify plan)]
          (is (= certified (distributed/verify! certified)))
          (is (= n (count (filter #(= :produce (second (:id %))) (:steps plan)))))
          (is (= n (count (filter #(= :update (second (:id %))) (:steps plan)))))
          (is (= (* 2 (dec n)) (count (filter #(= :transfer (:kind %)) (:steps plan)))))
          (is (= (dec n) (count (filter #(= :combined (second (:id %))) (:steps plan)))))
          (is (= (mapv #(vector % :update) (get-in plan [:mesh :devices])) (:outputs plan)))
          (doseq [[worker ssa] (get-in plan [:refinements :all-reduce :refinement :outputs])
                  :let [local (get-in plan [:device-plans worker :entries [worker :update] :link-plan])
                        matches (for [[value binding]
                                      (get-in plan [:device-plans worker :steps [worker :update] :bindings])
                                      :when (= ssa (:value binding))] value)]]
            (is (= 1 (count matches)))
            (is (nil? (:source (value-node local (first matches))))
                "host gradient initialization cannot stand in for the actual collective producer")))))))

