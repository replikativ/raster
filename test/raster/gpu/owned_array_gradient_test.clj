(ns raster.gpu.owned-array-gradient-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ad.value-grad-position-test :as fixture]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.value :as value]))

(defn- analytic-gradient [weights x target]
  (let [prediction (fn [row output]
                     (reduce + (for [k (range 4)]
                                 (* (aget ^floats x (+ (* row 4) k))
                                    (aget ^floats weights (+ (* output 4) k))))))]
    (vec (for [output (range 3) feature (range 4)]
           (* (/ 2.0 6.0)
              (reduce + (for [row (range 2)]
                          (* (aget ^floats x (+ (* row 4) feature))
                             (- (prediction row output)
                                (aget ^floats target (+ (* row 3) output)))))))))))

(defn- check-gradient! [device source constant-target?]
  (let [weights (float-array (map #(/ (double %) 10.0) (range 1 13)))
        x (float-array [0.1 0.2 -0.3 0.4 0.5 -0.6 0.7 0.8])
        target (float-array [0.2 -0.1 0.4 0.3 0.6 -0.2])
        captured-target (aclone target)
        prepared (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "preparation opened a session")))
                              gpu/alloc! (fn [& _] (throw (AssertionError. "preparation allocated")))]
                   (compiled/lower source [weights x target 2 4 3]
                                   (cond-> {:compiler :equation-first :target device :dtype :float}
                                     constant-target? (assoc :constants '[tgt]))))
        artifact (compiled/instantiate! prepared)]
    (try
      (is (some? (:ordered-invocation prepared)))
      (is (nil? (:lowering prepared)) "no physical boundary is fabricated before source work")
      (let [first-result (atom nil)]
        (doseq [offset [0.0 0.25]]
          (aset x 0 (float (+ 0.1 offset)))
          (aset weights 0 (float (+ 0.1 (* 0.5 offset))))
          (when constant-target?
            (aset target 0 (float (+ 0.2 (* 10.0 offset)))))
          (let [expected (analytic-gradient weights x (if constant-target? captured-target target))
                actual (vec (value/->host (:result (artifact {}))))]
            (is (= 12 (count actual)))
            (is (< (apply max (map #(Math/abs (- %1 %2)) expected actual)) 1.0e-6))
            (if @first-result
              (is (not= @first-result actual) "replay refreshes the changed host input")
              (reset! first-result actual)))))
      (let [replacement (aclone captured-target)]
        (aset replacement 0 (float 1.5))
        (if constant-target?
          (is (= :compiled-ordered-input-role
                 (try (artifact {:tgt replacement}) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
              "a late logical alias cannot turn its captured root into a replaceable input")
          (let [expected (analytic-gradient weights x replacement)
                actual (vec (value/->host (:result (artifact {:tgt replacement}))))]
            (is (< (apply max (map #(Math/abs (- %1 %2)) expected actual)) 1.0e-6)
                "the public target override reaches its original physical parameter root")
            (let [expected (analytic-gradient weights x target)
                  actual (vec (value/->host (:result (artifact {}))))]
              (is (< (apply max (map #(Math/abs (- %1 %2)) expected actual)) 1.0e-6)
                  "omission restores the original default, not the preceding replacement")))))
      (finally (compiled/close! artifact)))))

(deftest direct-and-nested-array-gradients-use-the-owned-public-path
  (doseq [source [#'fixture/vgp-train-direct #'fixture/vgp-train-nested]
          constant-target? [false true]]
    (if @ocl-probe/opencl-available?
      (check-gradient! :ocl:0 source constant-target?)
      (ocl-probe/opencl-skip! "owned array-gradient source preparation"))
    (if @ze-probe/gpu-available?
      (check-gradient! :ze:0 source constant-target?)
      (ze-probe/gpu-skip! "owned array-gradient source preparation"))))
