(ns raster.compiler.ir.scan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]))

(deftest retained-associative-certificates-are-independently-validated
  (doseq [[dtype op init]
          [[:float '+ 0.0] [:double '* 1.0]
           [:int 'unchecked-add-int 0] [:long 'unchecked-multiply 1]
           [:float 'min Float/POSITIVE_INFINITY] [:double 'max Double/NEGATIVE_INFINITY]]]
    (let [certificate (scan/certify-reassociation
                       {:acc 'acc :init init :lambda (list op 'acc 'element)} dtype)]
      (is (identical? certificate (scan/validate! certificate)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (scan/validate! (assoc certificate :identity :forged))))))
  (let [certificate (scan/certify-reassociation
                     {:acc 'acc :init 0.0
                      :lambda '(raster.numeric/+ (float acc) (raster.numeric/* x y))} :float)]
    (is (identical? certificate (scan/validate! certificate)))))

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
      (is (= :wrap (:overflow certificate)))
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

(deftest reassociation-certification-retains-the-local-element-region
  (let [facts (scan/certify-reassociation
               {:acc 'acc :init 0.0
                :lambda '(let* [difference (- (aget x i) (aget target i))]
                               (+ acc (* difference difference)))}
               :double)]
    (is (scan/associative-scan? facts))
    (is (= '+ (:combine facts)))
    (is (= '(let* [difference (- (aget x i) (aget target i))]
              (* difference difference))
           (:element facts)))
    (is (identical? facts (scan/validate! facts)))))

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

(deftest projected-certification-preserves-once-only-element-evaluation
  (let [step '(let* [term (* (clojure.core/aget values i)
                             (clojure.core/aget values i))]
                     (+ acc term))
        proof (scan/certify-projected-scan {:acc 'acc :init 0.0 :lambda step} :double)]
    (is (= '(+ acc (let* [term (* (clojure.core/aget values i)
                                  (clojure.core/aget values i))] term))
           (:normalized-step-result proof)))
    (is (= '(let* [term (* (clojure.core/aget values i) (clojure.core/aget values i))] term)
           (:element proof))))
  (let [proof (scan/certify-projected-scan
               {:acc 'acc :init 0 :lambda '(let* [term (int x)] (+ acc term))} :int)]
    (is (= '(let* [term (int x)] term) (:element proof)))
    (is (= '(+ acc (let* [term (int x)] term)) (:normalized-step-result proof)))))

(deftest retained-element-regions-preserve-values-checks-and-unused-bindings
  (doseq [step ['(let* [term (clojure.core/aget values i)] (+ acc (* term term)))
               '(let* [unused (int x) term (clojure.core/aget values i)] (+ acc term))
               '(let* [term (clojure.core/aget values i)]
                  (let* [term (+ term 1.0)] (+ acc (* term term))))
               '(do (int x) (+ acc (clojure.core/aget values i)))]]
    (let [proof (scan/certify-projected-scan {:acc 'acc :init 0.0 :lambda step} :double)
          compile-step (fn [body] (eval (list 'fn '[acc values i x] body)))
          source (compile-step step)
          normalized (compile-step (:normalized-step-result proof))]
      (doseq [[values i x] [[(double-array [2.0 3.0]) 0 7]
                            [(double-array 0) 0 7]
                            [(double-array 0) 0 Long/MAX_VALUE]
                            [(double-array [2.0]) 0 Long/MAX_VALUE]
                            [nil 0 7]]]
        (let [observe (fn [f]
                        (try [:value (f 1.0 values i x)]
                             (catch Throwable error [:error (class error)])))]
          (is (= (observe source) (observe normalized)))))))
  (doseq [step ['(let* [term acc] (+ acc term))
               '(let* [acc 1.0] (+ acc x))
               '(let* [+ custom] (+ acc x))
               '(let* [term (do (println "effect") x)] (+ acc term))]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (scan/certify-projected-scan {:acc 'acc :init 0.0 :lambda step} :double)))))

(deftest element-factoring-does-not-rebind-an-accumulator-conversion
  (doseq [step ['(let* [double (fn [_] 7.0)] (+ (double acc) x))
               (list 'let* ['callee 'unknown]
                     (with-meta '(.invk callee acc x) {:raster.op/original 'clojure.core/+}))
               (list 'clojure.core/+
                     (with-meta '(double acc) {:raster.op/lexical-callee true}) 'x)]
          certify [scan/certify-reassociation scan/certify-projected-scan]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (certify {:acc 'acc :init 0.0 :lambda step} :double)))))
