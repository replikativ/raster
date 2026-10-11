(ns raster.compiler.passes.parallel.zero-contract-epilogue-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]))

(deftm closed-zero-contract
  [A :- (Array double) C :- (Array double)] :- (Array double)
  (par/contract C [[i 2] [j 5]] []
    (clojure.core/+ (clojure.core/aget A i) (clojure.core/aget A j))
    :epilogue {:acc acc :expr (clojure.core/+ acc 3.0)
               :operands [] :scalars [] :dtype :double})
  C)

(deftest ordinary-deftm-zero-contract-matches-independent-coordinate-oracle
  (doseq [values [[0.5 -1.0 7.25 2.0 -3.0] [10.0 0.25 -7.0 6.0 1.0]]]
    (let [input (double-array values)
          output (double-array 10)
          actual (closed-zero-contract input output)
          expected (vec (for [i (range 2) j (range 5)]
                          (+ (nth values i) (nth values j) 3.0)))]
      (is (identical? output actual))
      (is (= expected (vec actual)))
      (is (= values (vec input))))))

(deftest surface-zero-contract-evaluates-body-and-epilogue-once-not-initializer
  (let [body-calls (atom 0) epilogue-calls (atom 0)
        output (double-array 6)]
    (par/contract output [[i 2] [j 3]] []
      (do (swap! body-calls inc) (double (+ (* i 3) j)))
      :init (throw (Exception. "zero-axis initializer must not run"))
      :combine (throw (Exception. "zero-axis combine must not run"))
      :epilogue {:acc acc :expr (do (swap! epilogue-calls inc) (+ acc 0.5))
                 :operands [] :scalars [] :dtype :double})
    (is (= 6 @body-calls))
    (is (= 6 @epilogue-calls))
    (is (= [0.5 1.5 2.5 3.5 4.5 5.5] (vec output)))))

(deftm closed-zero-contract-rounded
  [A :- (Array double) C :- (Array double)] :- (Array double)
  (par/contract C [[i 2]] [] (clojure.core/aget A i)
    :epilogue {:acc acc :expr (clojure.core/double (clojure.core/float acc))
               :operands [] :scalars [] :dtype :double})
  C)

(deftest completed-result-preserves-an-observable-float-checkpoint
  (let [input (double-array [16777217.0 -16777217.0])
        output (double-array 2)]
    (is (identical? output (closed-zero-contract-rounded input output)))
    (is (= [16777216.0 -16777216.0] (vec output)))
    (is (not= (vec input) (vec output)))))
