(ns raster.compiler.ir.scan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]))

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

(deftest wrapping-integral-arithmetic-is-an-explicit-monoid
  (doseq [[combine identity]
          [['unchecked-add 0] ['unchecked-add-int 0]
           ['clojure.core/unchecked-add 0] ['clojure.core/unchecked-add-int 0]
           ['unchecked-multiply 1] ['unchecked-multiply-int 1]
           ['clojure.core/unchecked-multiply 1]
           ['clojure.core/unchecked-multiply-int 1]]]
    (let [certificate (scan/certify-reassociation
                       {:acc 'acc :init identity :lambda (list combine 'acc 'element)} :int)]
      (is (scan/associative-scan? certificate))
      (is (= combine (:combine certificate)))
      (is (= identity (:identity certificate))))))

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

(deftest floating-min-max-certificates-retain-their-exceptional-value-policy
  (let [maximum (scan/certify-reassociation
                 {:acc 'acc :init '(float Double/NEGATIVE_INFINITY)
                  :lambda '(raster.numeric/max acc element)}
                 :float)
        minimum (scan/certify-reassociation
                 {:acc 'acc :init 'Double/POSITIVE_INFINITY
                  :lambda '(raster.numeric/min acc element)}
                 :double)]
    (is (= {:nan-policy :propagate :signed-zero-policy :prefer-positive}
           (select-keys maximum [:nan-policy :signed-zero-policy])))
    (is (= {:nan-policy :propagate :signed-zero-policy :prefer-negative}
           (select-keys minimum [:nan-policy :signed-zero-policy])))))

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

(deftest scan-certification-retains-the-element-behind-an-attested-projection
  (let [conversion (dialect/scalar-convert
                    {:source-dtype :float :target-dtype :float
                     :rounding :exact :overflow :exact
                     :source-op 'clojure.core/float}
                    '(clojure.core/aget values i))
        step (list 'clojure.core/+ 'acc conversion)
        projection (dialect/scalar-converts->source step)
        proof (scan/certify-projected-scan
               {:acc 'acc :init 0.0 :lambda step} :float projection)]
    (is (= conversion (:element proof))
        "the executable element retains its typed conversion policy")
    (is (= (nth projection 2) (:element (:certificate proof)))
        "the algebra certificate retains only the shared source vocabulary"))
  (try
    (scan/certify-projected-scan
     {:acc 'acc :init 0.0 :lambda '(- acc x)} :double '(+ acc x))
    (is false "a scan cannot certify a different projected operator")
    (catch clojure.lang.ExceptionInfo exception
      (is (= :scan-certificate-projection (:reason (ex-data exception)))))))

(deftest projected-certification-normalizes-only-removable-let-bindings
  (let [step '(let* [term (* (clojure.core/aget values i)
                             (clojure.core/aget values i))]
                     (+ acc term))
        proof (scan/certify-projected-scan {:acc 'acc :init 0.0 :lambda step} :double)]
    (is (= '(+ acc (* (clojure.core/aget values i)
                      (clojure.core/aget values i)))
           (:normalized-step-result proof)))
    (is (= '(* (clojure.core/aget values i) (clojure.core/aget values i))
           (:element proof))))
  (try
    (scan/certify-projected-scan
     {:acc 'acc :init 0 :lambda '(let* [term (int x)] (+ acc term))} :int)
    (is false "a checked let initializer cannot be duplicated or delayed by certification")
    (catch clojure.lang.ExceptionInfo exception
      (is (= :scan-certificate-projection (:reason (ex-data exception))))
      (is (= :impure-binding
             (get-in (ex-data exception) [:normalization-error :reason]))))))
