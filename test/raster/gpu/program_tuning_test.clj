(ns raster.gpu.program-tuning-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.dispatch-benchmark :as benchmark]
            [raster.gpu.program-tuning :as program-tuning]))

(def ^:private abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'width :scalar :long :role :shape)])

(defn- emitted
  [kernel-name strategy workgroup]
  (artifact/make
   {:kernel-name kernel-name
    :source (str "__kernel void " kernel-name
                 "(__global const float* x, __global float* out, long width) {}")
    :abi abi
    :arguments '[x out width]
    :launch (launch/spec {:workgroup-size [workgroup]
                          :group-count [(launch/ceil-div 'width workgroup)]})
    :effects {:kind :elementwise-map :reads ['x] :writes ['out]}
    :attributes {:strategy strategy}}))

(def ^:private alternatives
  [(emitted "program_tuning_reference" :reference 64)
   (emitted "program_tuning_subgroup" :subgroup 16)])

(defn- dispatch
  [id]
  (kdispatch/make
   {:id id
    :alternatives alternatives
    :default-strategy :reference
    :selector {:kind :runtime-scalar-threshold
               :argument 'width :threshold 256
               :at-least :subgroup :otherwise :reference}
    :attributes
    {:tuning {:schedule-path [:generic-reduction :measured-selectors]
              :schedule-key id
              :numerical-mode {:input :f32 :accumulate :f32 :output :f32}
              :layout {:input :contiguous :output :contiguous}}}}))

(def ^:private dispatch-a (dispatch "dispatch-a"))
(def ^:private dispatch-b (dispatch "dispatch-b"))

(def ^:private descriptor
  {:steps [{:phase :first-a :dispatch dispatch-a}
           {:phase :only-b :dispatch dispatch-b}
           {:phase :second-a :dispatch dispatch-a}
           {:phase :ordinary-kernel}]})

(deftest manifest-groups-equivalent-sites-in-program-order
  (let [manifest (program-tuning/manifest descriptor)]
    (is (= 3 (:site-count manifest)))
    (is (= 2 (:group-count manifest)))
    (is (= ["dispatch-a" "dispatch-b"] (mapv :id (:groups manifest))))
    (is (= [0 2] (get-in manifest [:groups 0 :step-indices])))
    (is (= [:first-a :second-a] (get-in manifest [:groups 0 :phases])))
    (is (= 2 (get-in manifest [:groups 0 :site-count])))
    (is (= 1 (get-in manifest [:groups 1 :representative-step-index])))))

(deftest manifest-rejects-a-reused-id-with-a-different-contract
  (let [different (assoc-in dispatch-a [:attributes :tuning :layout]
                            {:input :strided :output :contiguous})]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"incompatible tuning sites"
         (program-tuning/manifest
          {:steps [{:phase :first :dispatch dispatch-a}
                   {:phase :different :dispatch different}]})))))

(deftest tuning-plan-is-deterministic-and-explicitly-bounded
  (let [manifest (program-tuning/manifest descriptor)
        plan (program-tuning/tuning-plan manifest {:max-groups 1})]
    (is (= ["dispatch-a"] (mapv :id (:selected-groups plan))))
    (is (= ["dispatch-b"] (mapv :id (:deferred-groups plan))))
    (is (= 1 (get-in plan [:budget :selected-groups])))
    (testing "manifest order wins over caller collection order"
      (is (= ["dispatch-a" "dispatch-b"]
             (mapv :id (:selected-groups
                        (program-tuning/tuning-plan
                         manifest {:group-ids ["dispatch-b" "dispatch-a"]}))))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"unknown dispatch groups"
         (program-tuning/tuning-plan manifest {:group-ids ["absent"]})))
    (is (empty? (:selected-groups
                 (program-tuning/tuning-plan manifest {:max-groups 0}))))))

(deftest schedule-fragments-merge-by-dispatch-key-and-conflicts-fail
  (is (= {:generic-reduction {:measured-selectors
                              {"dispatch-a" :selector-a
                               "dispatch-b" :selector-b}}}
         (program-tuning/merge-schedule-overrides
          [{:generic-reduction {:measured-selectors {"dispatch-a" :selector-a}}}
           {:generic-reduction {:measured-selectors {"dispatch-b" :selector-b}}}])))
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"conflict"
       (program-tuning/merge-schedule-overrides
        [{:generic-reduction {:measured-selectors {"dispatch-a" :selector-a}}}
         {:generic-reduction {:measured-selectors {"dispatch-a" :selector-b}}}]))))

(deftest program-runner-checks-budget-before-tuning-and-merges-results
  (let [manifest (program-tuning/manifest descriptor)
        plan (program-tuning/tuning-plan manifest)
        session (atom {:programs {:program {:descriptor descriptor}}})
        calls (atom [])
        fake-tune
        (fn [_ _ _ runtime-values _ & options]
          (let [step (:step (apply hash-map options))
                dispatch (get-in descriptor [:steps step :dispatch])
                id (:id dispatch)
                selector {:kind :runtime-scalar-ranges
                          :argument 'width :below :reference
                          :ranges [{:at-least 256 :strategy :subgroup}]}]
            (swap! calls conj [id runtime-values step])
            {:tuning :fake
             :selector selector
             :schedule-override
             {:generic-reduction {:measured-selectors {id selector}}}
             :step-index step
             :phase (get-in descriptor [:steps step :phase])}))]
    (with-redefs [benchmark/tune-program-dispatch! fake-tune]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"exceeds its physical measurement budget"
           (benchmark/tune-program-dispatches!
            session descriptor {} plan
            (constantly [256 128 256])
            (fn [& _] {})
            :max-measurements 7)))
      (is (empty? @calls))
      (let [result (benchmark/tune-program-dispatches!
                    session descriptor {} plan
                    (constantly [256 128 256])
                    (fn [& _] {})
                    :max-measurements 8)]
        (is (= 8 (:planned-measurements result)))
        (is (= [["dispatch-a" [128 256] 0]
                ["dispatch-b" [128 256] 1]]
               @calls))
        (is (= #{"dispatch-a" "dispatch-b"}
               (set (keys (get-in result
                                  [:schedule-override :generic-reduction
                                   :measured-selectors])))))))))
