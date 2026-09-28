(ns raster.quant.train-device-test
  "Q8 forward and pullback must use the same equation-first GPU path as adapter training."
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as rev]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.dl.loss :as loss]
            [raster.dl.nn :as nn]
            [raster.dl.optim :as optim]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.quant.train :as qt]))

(deftm q8-lora-loss
  [A :- (Array float) B :- (Array float) x :- (Array float)
   codes :- (Array byte) scales :- (Array float) tgt :- (Array float)
   rows :- Long in :- Long rank :- Long out :- Long] :- Double
  (let [base (qt/qlinear-q8 x codes scales rows in out)
        hidden (nn/linear-nb x A rows in rank)
        delta (nn/linear-nb hidden B rows rank out)
        prediction (nn/residual-add base delta (clojure.core/* rows out))]
    (loss/mse-loss prediction tgt (clojure.core/* rows out))))

(deftm q8-lora-train-step
  [A :- (Array float) B :- (Array float) x :- (Array float)
   codes :- (Array byte) scales :- (Array float) tgt :- (Array float)
   rows :- Long in :- Long rank :- Long out :- Long lr :- Double] :- (Array float)
  (let [vg ((rev/value+grad #'raster.quant.train-device-test/q8-lora-loss)
            A B x codes scales tgt rows in rank out)
        dA (clojure.core/nth vg 1)
        dB (clojure.core/nth vg 2)]
    (optim/sgd-step! A dA (arrays/alength A) (float lr))
    (optim/sgd-step! B dB (arrays/alength B) (float lr))
    A))

(defn- q8-lora-fixture []
  (let [rows 2 in 32 rank 2 out 2
        weights (float-array (map #(float (/ (inc (mod % 9)) 64.0))
                                  (range (* in out))))
        {:keys [codes scales]} (qt/q8-quantize weights out in)]
    {:A (float-array (map #(float (/ (inc (mod % 7)) 32.0))
                          (range (* rank in))))
     :B (float-array (map #(float (/ (inc (mod % 5)) 40.0))
                          (range (* out rank))))
     :x (float-array (map #(float (/ (inc (mod % 11)) 16.0))
                          (range (* rows in))))
     :codes codes :scales scales
     :tgt (float-array [0.2 -0.3 0.4 0.1])
     :rows rows :in in :rank rank :out out :lr 0.02}))

(defn- prepare-q8-lora [target {:keys [A B x codes scales tgt rows in rank out lr]}]
  (compiled/lower #'q8-lora-train-step
                  [A B x codes scales tgt rows in rank out lr]
                  {:compiler :equation-first :target target :dtype :float
                   :donate '[A B] :constants '[x codes scales tgt]
                   :schedule {:precision :f32-scalar}
                   :on-non-resident :throw}))

(deftest q8-lora-train-step-has-a-certified-resident-artifact
  (let [prepared (prepare-q8-lora :cuda:0 (q8-lora-fixture))]
    (is (compiled/prepared? prepared))
    (is (= {:A :A' :B :B'} (:donated prepared)))
    (is (= 0 (get-in prepared [:lowering :plan :attributes :driver-allocations])))
    (is (seq (get-in prepared [:lowering :plan :instances])))))

(deftest q8-lora-train-step-replays-two-updates-with-frozen-base
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "equation-first Q8 LoRA resident training")
    (let [{:keys [A B x codes scales tgt rows in rank out lr] :as fixture}
          (q8-lora-fixture)
          expected-A (aclone ^floats A)
          expected-B (aclone ^floats B)
          initial-codes (vec codes)
          initial-scales (vec scales)
          prepared (prepare-q8-lora :ze:0 fixture)
          artifact (compiled/instantiate! prepared)
          prior (volatile! nil)]
      (try
        (dotimes [_ 2]
          (q8-lora-train-step expected-A expected-B x codes scales tgt
                              rows in rank out lr)
          (let [old @prior
                result (with-redefs [link/write!
                                     (fn [& _]
                                       (throw (ex-info "resident replay uploaded input" {})))]
                         (artifact {}))
                next-A (:A' result)
                actual-A (value/->host next-A)
                actual-B (value/->host (:B' result))]
            (when old
              (is (not (value/live? old)) "prior donated value is invalid after replay"))
            (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) 1.0e-3)
                                   expected-A actual-A)))
            (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) 1.0e-3)
                                   expected-B actual-B)))
            (vreset! prior next-A)))
        (is (= initial-codes (vec codes)))
        (is (= initial-scales (vec scales)))
        (finally (compiled/close! artifact)))
      (is (not (value/live? @prior)) "closing the owner invalidates projected state"))))

(deftest q8-forward-and-backward-lower-through-typed-soac
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "equation-first Q8 forward and pullback")
    (let [in 32 out 2 rows 2
          weights (float-array (map #(float (/ (inc (mod % 9)) 64.0))
                                    (range (* in out))))
          {:keys [codes scales]} (qt/q8-quantize weights out in)
          x (float-array (map #(float (/ (inc (mod % 7)) 16.0))
                              (range (* rows in))))
          dy (float-array [0.2 -0.3 0.4 0.1])]
      (doseq [[kernel input expected input-key]
              [[#'qt/qlinear-q8 x (qt/qlinear-q8 x codes scales rows in out) 'x]
               [#'qt/qlinear-q8-dx dy (qt/qlinear-q8-dx dy codes scales rows in out) 'dy]]]
        (let [prepared (compiled/lower kernel
                                       [input codes scales (long rows) (long in) (long out)]
                                       {:compiler :equation-first :target :ze:0 :dtype :float
                                        :constants [input-key 'codes 'scales]
                                        :on-non-resident :throw})
              artifact (compiled/instantiate! prepared)]
          (try
            (let [actual (value/->host (:result (artifact {})))]
              (is (= (alength ^floats expected) (alength ^floats actual)))
              (is (every? true?
                          (map #(<= (Math/abs (- (double %1) (double %2))) 1.0e-5)
                               expected actual))))
            (finally (compiled/close! artifact))))))))
