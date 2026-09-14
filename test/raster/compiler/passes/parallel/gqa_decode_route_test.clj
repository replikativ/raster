(ns raster.compiler.passes.parallel.gqa-decode-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.coverage :as coverage]
            [raster.dl.attention :as attention]))

(def ^:private compiled-route
  (delay (coverage/report-var #'attention/gqa-decode-attention-buf!
                              {:target-device :ocl:0 :dtype :float})))

(deftest public-buffered-decode-attention-is-entirely-typed
  (let [report @compiled-route]
    (is (= :typed-soac (:route report)))
    (is (:typed-validated report))
    (is (empty? (:declines report)))
    (is (= 3 (get-in report [:emission :kernel-count])))
    (is (= {:kernel-body 3} (get-in report [:emission :routes])))
    (is (= {:independent 3} (:effect-orders report)))))

(deftest cache-length-is-clamped-to-the-physical-row-capacity
  (let [n-q 4 n-kv 2 group 2 head-dim 4 maxpos 5 scale 0.5
        q (double-array (map #(/ (inc %) 16.0) (range (* n-q head-dim))))
        k (double-array (map #(/ (- (mod % 9) 4) 8.0)
                             (range (* maxpos n-kv head-dim))))
        v (double-array (map #(/ (- (mod % 7) 3) 4.0)
                             (range (* maxpos n-kv head-dim))))]
    (doseq [cache-length [0 1 3 5 7]]
      (testing (str "cache length " cache-length)
        (let [effective (min cache-length maxpos)
              expected (attention/gqa-decode-attention
                        q k v effective n-q n-kv head-dim scale)
              out (double-array (* n-q head-dim))
              scratch (double-array (repeat (* n-q maxpos) -77.0))]
          (is (nil? (attention/gqa-decode-attention-buf!
                     q k v out scratch (long-array [cache-length])
                     n-q group n-kv head-dim maxpos scale)))
          (is (= (vec expected) (vec out)))
          (doseq [head (range n-q)
                  position (range effective maxpos)]
            (is (= -77.0 (aget scratch (+ (* head maxpos) position)))
                "inactive physical capacity remains untouched")))))))
