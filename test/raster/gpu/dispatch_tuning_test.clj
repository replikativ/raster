(ns raster.gpu.dispatch-tuning-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.tuning-cache :as cache])
  (:import [java.nio.file Files]))

(def ^:private abi
  [(kabi/slot 'x :input :float :role :operand)
   (kabi/slot 'out :output :float :role :result)
   (kabi/slot 'width :scalar :long :role :shape)])

(defn- artifact
  [name strategy workgroup]
  (kart/make
   {:kernel-name name
    :source (str "__kernel void " name
                 "(__global const float* x, __global float* out, long width) {}")
    :abi abi
    :arguments '[x out width]
    :launch (klaunch/spec {:workgroup-size [workgroup]
                           :group-count [(klaunch/ceil-div 'width workgroup)]})
    :effects {:kind :map :reads ['x] :writes ['out]}
    :attributes {:strategy strategy}}))

(def ^:private reference (artifact "tune_reference" :reference 64))
(def ^:private subgroup (artifact "tune_subgroup" :subgroup 16))

(def ^:private dispatch
  (kdispatch/make
   {:id "dispatch-tuning-test"
    :alternatives [reference subgroup]
    :default-strategy :reference
    :selector {:kind :runtime-scalar-threshold
               :argument 'width
               :threshold 256
               :at-least :subgroup
               :otherwise :reference}}))

(def ^:private fixed-dispatch
  (kdispatch/make
   {:id "fixed-dispatch-tuning-test"
    :alternatives [reference subgroup]
    :default-strategy :reference
    :selector {:kind :fixed-strategy :strategy :reference}}))

(def ^:private descriptor
  {:device-id :ze:0 :vendor "Intel" :arch "xe2" :subgroup-size 16
   :machine-lanes 8192 :driver-version "test-driver"})

(def ^:private numerical-mode
  {:input :f32 :accumulate :f32 :output :f32})

(def ^:private layout
  {:input :row-major :output :row-major})

(defn- stable-measurement
  [cost]
  (measurement/summarize [cost cost cost]
                         :timing-source :device-event
                         :cv-threshold 0.01))

(defn- validated-result
  [artifact cost]
  {:measurement (stable-measurement cost)
   :validation {:passed? true
                :oracle-hash "oracle-v1"
                :candidate-hash (:source-hash (tuning/executable-signature artifact))
                :max-error 0.0}})

(defn- temporary-cache-root
  []
  (.toFile (Files/createTempDirectory "raster-dispatch-tuning-"
                                      (make-array java.nio.file.attribute.FileAttribute 0))))

(deftest static-schedule-axis-produces-a-fixed-measured-selector
  (binding [cache/*cache-root* (temporary-cache-root)]
    (let [calls (atom [])
          result (tuning/tune-fixed!
                  fixed-dispatch descriptor
                  (fn [artifact]
                    (let [strategy (kdispatch/alternative-strategy artifact)]
                      (swap! calls conj strategy)
                      (validated-result artifact (if (= :subgroup strategy) 4.0 10.0))))
                  :numerical-mode numerical-mode :layout layout)
          tuned (tuning/apply-tuning fixed-dispatch result descriptor numerical-mode layout)]
      (is (= {:kind :fixed-strategy :strategy :subgroup} (:selector result)))
      (is (= [:reference :subgroup] @calls))
      (is (= :subgroup
             (kdispatch/alternative-strategy
              (kdispatch/select-alternative tuned [:x :out {:type :long :value 17}]))))
      (is (= [] (get-in result [:identity :policy :runtime-values]))
          "static tuning adds no synthetic value to the executable ABI"))))

(deftest validated-device-measurements-produce-and-cache-piecewise-selection
  (binding [cache/*cache-root* (temporary-cache-root)]
    (let [calls (atom 0)
          costs {:reference {128 10.0 256 10.0 768 5.0}
                 :subgroup {128 12.0 256 7.0 768 6.0}}
          benchmark (fn [artifact runtime-value]
                      (swap! calls inc)
                      (validated-result artifact
                                        (get-in costs [(kdispatch/alternative-strategy artifact)
                                                       runtime-value])))
          result (tuning/tune! dispatch descriptor [768 128 256] benchmark
                               :numerical-mode numerical-mode :layout layout)
          tuned (tuning/apply-tuning dispatch result descriptor numerical-mode layout)
          select #(kdispatch/alternative-strategy
                   (kdispatch/select-alternative tuned [:x :out {:type :long :value %}]))]
      (is (tuning/dispatch-tuning? result))
      (is (= {:kind :runtime-scalar-ranges
              :argument 'width
              :below :reference
              :ranges [{:at-least 256 :strategy :subgroup}
                       {:at-least 768 :strategy :reference}]}
             (:selector result)))
      (is (= :reference (select 128)))
      (is (= :subgroup (select 512)))
      (is (= :reference (select 768)))
      (is (= 6 @calls))
      (testing "an exact cache hit performs no benchmark calls"
        (let [cached (tuning/tune! dispatch descriptor [128 256 768]
                                   (fn [& _] (throw (ex-info "must not benchmark" {})))
                                   :numerical-mode numerical-mode :layout layout)]
          (is (= (:selector result) (:selector cached)))))
      (testing "shape samples and noise policy participate in cache identity"
        (let [base (tuning/tuning-identity dispatch descriptor [128 256 768]
                                           numerical-mode layout 0.001)
              other-shapes (tuning/tuning-identity dispatch descriptor [128 256]
                                                   numerical-mode layout 0.001)
              other-policy (tuning/tuning-identity dispatch descriptor [128 256 768]
                                                   numerical-mode layout 0.01)]
          (is (not= (tuning/cache-key base) (tuning/cache-key other-shapes)))
          (is (not= (tuning/cache-key base) (tuning/cache-key other-policy)))
          (is (nil? (tuning/cache-get dispatch other-shapes)))))
      (testing "layout/numerical/device identity is checked again when applying"
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"identity differs"
                              (tuning/apply-tuning dispatch result descriptor numerical-mode
                                                   {:input :column-major})))))))

(deftest default-wins-inside-the-noise-floor
  (binding [cache/*cache-root* (temporary-cache-root)]
    (let [result (tuning/tune!
                  dispatch descriptor [256]
                  (fn [artifact _]
                    (validated-result artifact
                                      (if (= :reference
                                             (kdispatch/alternative-strategy artifact))
                                        100.0 99.95)))
                  :numerical-mode numerical-mode :layout layout :force? true)]
      (is (= [] (get-in result [:selector :ranges])))
      (is (= :reference (kdispatch/alternative-strategy
                         (kdispatch/select-alternative
                          (tuning/apply-tuning dispatch result descriptor numerical-mode layout)
                          [:x :out {:type :long :value 4096}])))))))

(deftest invalid-or-noisy-candidates-are-never-cached
  (doseq [[label benchmark message]
          [[:wrong
            (fn [artifact _]
              (assoc (validated-result artifact 1.0)
                     :validation {:passed? false
                                  :oracle-hash "oracle-v1"
                                  :candidate-hash (:source-hash
                                                   (tuning/executable-signature artifact))}))
            #"must pass an oracle"]
           [:noisy
            (fn [artifact _]
              {:measurement (measurement/summarize [1.0 100.0]
                                                   :timing-source :device-event
                                                   :cv-threshold 0.01)
               :validation {:passed? true
                            :oracle-hash "oracle-v1"
                            :candidate-hash (:source-hash
                                             (tuning/executable-signature artifact))}})
            #"non-stationary"]]]
    (testing (name label)
      (binding [cache/*cache-root* (temporary-cache-root)]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo message
             (tuning/tune! dispatch descriptor [256] benchmark
                           :numerical-mode numerical-mode :layout layout :force? true)))
        (is (nil? (tuning/cache-get
                   dispatch
                   (tuning/tuning-identity dispatch descriptor [256] numerical-mode layout
                                           0.001))))))))
