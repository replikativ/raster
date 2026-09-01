(ns raster.compiler.ir.scan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.scan :as scan]))

(deftest associative-scan-certification
  (let [facts (scan/certify {:acc 'acc :init 0.0
                             :lambda '(+ acc (clojure.core/aget values i))}
                            :double)]
    (is (scan/associative-scan? facts))
    (is (= '+ (:combine facts)))
    (is (= '(clojure.core/aget values i) (:element facts)))
    (is (= 0.0 (:identity facts))))
  (testing "an accumulator cast does not hide the one accumulator position"
    (is (scan/associative-scan?
         (scan/certify {:acc 'acc :init 0.0
                        :lambda '(raster.numeric/+ (float acc) (aget values i))}
                       :float)))))

(deftest reassociation-certification-uses-the-shared-pure-let-rewrite
  (let [facts (scan/certify-reassociation
               {:acc 'acc :init 0.0
                :lambda '(let* [difference (- (aget x i) (aget target i))]
                           (+ acc (* difference difference)))}
               :double)]
    (is (scan/associative-scan? facts))
    (is (= '+ (:combine facts)))
    (is (= '(* (- (aget x i) (aget target i))
               (- (aget x i) (aget target i)))
           (:element facts)))))

(deftest general-recurrences-are-not-relabelled-parallel-scans
  (testing "an RNN recurrence is sequential unless it is explicitly lifted to an associative algebra"
    (try
      (scan/certify {:acc 'h :init 0.0
                     :lambda '(Math/tanh (+ (* w h) (aget x i)))}
                    :double)
      (is false "general recurrence must decline")
      (catch clojure.lang.ExceptionInfo e
        (is (= :scan-not-associative (:reason (ex-data e)))))))
  (testing "an unregistered/non-associative combine declines"
    (try
      (scan/certify {:acc 'acc :init 0.0 :lambda '(- acc (aget x i))} :double)
      (is false "subtraction is not an associative scan combine")
      (catch clojure.lang.ExceptionInfo e
        (is (= :scan-not-associative (:reason (ex-data e))))))))

(deftest block-parallel-scan-requires-the-monoid-identity
  (try
    (scan/certify {:acc 'acc :init 2.0 :lambda '(+ acc (aget x i))} :double)
    (is false "a non-identity init must not be injected independently into every block")
    (catch clojure.lang.ExceptionInfo e
      (is (= :scan-nonidentity-init (:reason (ex-data e))))
      (is (= 0.0 (:identity (ex-data e)))))))

(deftest scan-elements-must-be-proven-pure
  (try
    (scan/certify {:acc 'acc :init 0.0
                   :lambda '(+ acc (mystery-effect! values i))}
                  :double)
    (is false "unknown/effectful element calls must not be reordered")
    (catch clojure.lang.ExceptionInfo e
      (is (= :scan-element-impure-or-unknown (:reason (ex-data e)))))))
