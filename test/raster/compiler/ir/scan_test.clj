(ns raster.compiler.ir.scan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.scan :as scan]))

(deftest rounded-or-failing-casts-do-not-prove-integer-identities
  (doseq [[dtype init] [[:int '(float 2147483647)]
                       [:long (list 'double Long/MAX_VALUE)]
                       [:long (double Long/MAX_VALUE)]
                       [:int '(int Double/POSITIVE_INFINITY)]]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-identity init"
          (scan/certify {:acc 'acc :init init :lambda '(min acc x)} dtype)))))

(deftest checked-nested-casts-retain-the-original-init
  (let [init '(double (float 0.0))
        certificate (scan/certify {:acc 'acc :init init :lambda '(+ acc x)} :double)]
    (is (= init (:init certificate)))
    (is (scan/associative-scan? certificate))))

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
                       :float))))
  (testing "only a cast to the reduction dtype denotes the accumulator"
    (doseq [body '[(+ (int acc) (aget values i))
                   (+ (float acc) (aget values i))]]
      (try
        (scan/certify {:acc 'acc :init 0.0 :lambda body} :double)
        (is false (str "cross-dtype accumulator cast must decline: " body))
        (catch clojure.lang.ExceptionInfo exception
          (is (= :scan-not-elementwise (:reason (ex-data exception))))))))
  (testing "a frontend numeric-literal cast does not change the exact identity"
    (is (scan/associative-scan?
         (scan/certify {:acc 'acc :init '(double 0.0)
                        :lambda '(+ acc (aget values i))}
                       :float)))))

(deftest integral-min-max-use-the-exact-bounded-domain-identities
  (is (= Integer/MAX_VALUE (descriptor/typed-reduce-identity 'min :int)))
  (is (= Integer/MIN_VALUE (descriptor/typed-reduce-identity 'max :int)))
  (is (= Long/MAX_VALUE (descriptor/typed-reduce-identity 'min :long)))
  (is (= Long/MIN_VALUE (descriptor/typed-reduce-identity 'max :long)))
  (is (scan/associative-scan?
       (scan/certify {:acc 'acc :init Integer/MIN_VALUE
                      :lambda '(max acc (aget values i))}
                     :int)))
  (try
    (scan/certify {:acc 'acc :init 'Double/NEGATIVE_INFINITY
                   :lambda '(max acc (aget values i))}
                  :int)
    (is false "an IEEE infinity is not an integer-domain identity")
    (catch clojure.lang.ExceptionInfo exception
      (is (= :scan-nonidentity-init (:reason (ex-data exception)))))))

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

(deftest reassociation-certification-uses-the-central-effect-analysis
  (let [facts (scan/certify-reassociation
               {:acc 'acc :init 0.0
                :lambda '(raster.numeric/+
                          acc
                          (raster.numeric/* (aget a i) (aget b i)))}
               :half)]
    (is (scan/associative-scan? facts))
    (is (= 'raster.numeric/* (first (:element facts))))))

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
