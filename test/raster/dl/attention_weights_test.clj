(ns raster.dl.attention-weights-test
  "ASR alignment contracts are mandatory host tests; they need neither BLAS nor a GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.attention :as attn]
            [raster.dl.attention-reference :as reference]
            [raster.gpu.compiled :as compiled]
            [raster.runtime.hardware :as hardware]))

(defn- mass [values n]
  (reduce + 0.0 (take n values)))

(deftest gqa-decode-attention-weights-test
  ;; Moved from attention-test, whose namespace-wide BLAS fixture skipped this non-BLAS oracle.
  (let [n 5 hd 8
        rng (java.util.Random. 11)
        frand (fn [k] (float-array (repeatedly k #(.nextGaussian rng))))
        scale (/ 1.0 (Math/sqrt (double hd)))]
    (doseq [[n-q n-kv] [[4 4] [4 2]]]
      (let [q (frand (* n-q hd)) k (frand (* n n-kv hd)) v (frand (* n n-kv hd))
            sink (float-array n)
            expected (reference/gqa-decode q k v n n-q n-kv hd scale)
            actual (attn/gqa-decode-attention-weights! q k v n n-q n-kv hd scale sink)]
        (is (java.util.Arrays/equals ^floats expected ^floats actual)
            "output matches the independent sequential oracle bit-for-bit")
        (is (< (Math/abs (- 1.0 (mass sink n))) 1e-5))
        (attn/gqa-decode-attention-weights! q k v n n-q n-kv hd scale sink)
        (is (< (Math/abs (- 2.0 (mass sink n))) 1e-5))))))

(deftest alignment-preserves-existing-sink-tail-and-empty-history
  (doseq [make-array [float-array double-array]
          n [0 1 3]]
    (let [q (make-array [1.0 -0.5 0.25 1.0])
          k (make-array [1.0 0.5 -1.0 2.0 0.5 -0.75])
          v (make-array [1.0 2.0 3.0 4.0 5.0 6.0])
          sink (make-array [0.25 -0.5 0.75 17.0 19.0])
          before (vec sink)
          inputs (mapv vec [q k v])
          expected (reference/gqa-decode q k v n 2 1 2 0.5)
          actual (attn/gqa-decode-attention-weights! q k v n 2 1 2 0.5 sink)]
      (is (= (vec expected) (vec actual)))
      (is (= inputs (mapv vec [q k v])) "query and cache inputs are not donated")
      (is (= (subvec before n) (subvec (vec sink) n)) "the untouched parent tail is retained")
      (is (< (Math/abs (- (if (zero? n) 0.0 1.0)
                          (- (mass sink n) (mass before n)))) 1e-5)))))

(deftest head-average-preserves-each-sink-materialization
  ;; Each head contributes 1/3. At this Float magnitude each separate update rounds away,
  ;; whereas accumulating heads in Double and storing once would incorrectly add one.
  (let [sink (float-array [8388608.0])
        result (attn/gqa-decode-attention-weights!
                (float-array [0.0 0.0 0.0]) (float-array [0.0]) (float-array [7.0])
                1 3 1 1 1.0 sink)]
    (is (= [7.0 7.0 7.0] (vec result)))
    (is (= 8388608.0 (double (aget sink 0)))))
  (let [sink (double-array [8388608.0])]
    (attn/gqa-decode-attention-weights!
     (double-array [0.0 0.0 0.0]) (double-array [0.0]) (double-array [7.0])
     1 3 1 1 1.0 sink)
    (is (< (Math/abs (- 8388609.0 (aget sink 0))) 1e-8))))

(deftest allocating-alignment-is-not-a-resident-gpu-program
  (hardware/register-target-device!
   :ze:asr-alignment-contract
   {:name "Synthetic ASR boundary target"
    :capabilities {:total-eus 32 :subgroup-sizes [16] :max-workgroup-size 1024
                   :shared-local-memory 65536}})
  (doseq [dtype [:float :double]
          :let [make-array (if (= dtype :float) float-array double-array)
                sink (make-array [0.25 0.5 0.75])
                before (vec sink)
                args [(make-array 4) (make-array 6) (make-array 6) 3 2 1 2 0.5 sink]]]
    (testing "a scalar-route corpus row is not public resident coverage"
      (is (= :equation-first-coverage
             (try
               (compiled/lower #'attn/gqa-decode-attention-weights! args
                               {:compiler :equation-first :target :ze:asr-alignment-contract :dtype dtype
                                :outputs '[wsink]})
               nil
               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (= before (vec sink)) "declining compilation must not execute alignment effects")))
