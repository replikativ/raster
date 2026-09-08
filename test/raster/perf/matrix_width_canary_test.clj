(ns raster.perf.matrix-width-canary-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.perf.matrix-width-canary :as canary]))

(deftest each-candidate-must-write-its-own-output
  (let [resident (atom (float-array [1.0 2.0]))
        case (canary/candidate-case :fake-session [1 2 16] (float-array [1.0 2.0]))]
    (with-redefs [gpu/upload! (fn [_ _ values] (reset! resident (aclone ^floats values)))
                  gpu/download (fn [_ _] @resident)]
      ((:before-run! case) {})
      (is (false? (:passed? ((:validate! case) {}))) "a no-op cannot inherit correct output")
      (aset ^floats @resident 0 (float 1.0))
      (is (false? (:passed? ((:validate! case) {}))) "partial writes leave poison")
      (aset ^floats @resident 1 (float 2.0))
      (is (= {:passed? true :relative-l1 0.0}
             (select-keys ((:validate! case) {}) [:passed? :relative-l1])))
      ((:before-run! case) {})
      (is (false? (:passed? ((:validate! case) {})))))))

(deftest canary-rejects-unbounded-work-before-device-access
  (doseq [shapes [[] [[1024 1024 1024]] [[0 16 16]] [[16 17 16]]
                  [[16 16 32]] [[16 32 16]]]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"bounded positive aligned"
                          (canary/run! :missing-device shapes)))))
