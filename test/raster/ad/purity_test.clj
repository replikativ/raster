(ns raster.ad.purity-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.purity :as purity]))

;; Initialize registry once
(defonce ^:private init-done
  (do (purity/init-raster-registry!) true))

;; ================================================================
;; S-expression purity (via tools.analyzer)
;; ================================================================

(deftest pure-sexp-test
  (testing "Pure arithmetic is pure"
    (is (true? (purity/pure-sexp? '(+ 1 2))))
    (is (true? (purity/pure-sexp? '(Math/sin 1.0))))
    (is (true? (purity/pure-sexp? '(Math/pow 2.0 3.0)))))

  (testing "Constant expressions are pure"
    (is (true? (purity/pure-sexp? 42)))
    (is (true? (purity/pure-sexp? "hello")))
    (is (true? (purity/pure-sexp? :keyword)))))

;; ================================================================
;; Op-level purity check (symbol-based)
;; ================================================================

(deftest pure-op-test
  (testing "Math ops are pure"
    (is (= :pure (purity/pure-op? 'Math/sin)))
    (is (= :pure (purity/pure-op? 'Math/cos)))
    (is (= :pure (purity/pure-op? 'Math/exp)))
    (is (= :pure (purity/pure-op? 'Math/log)))
    (is (= :pure (purity/pure-op? 'Math/sqrt))))

  (testing "Arithmetic ops are pure"
    (is (= :pure (purity/pure-op? '+)))
    (is (= :pure (purity/pure-op? '-)))
    (is (= :pure (purity/pure-op? '*)))
    (is (= :pure (purity/pure-op? '/))))

  (testing "Array reads are pure"
    (is (= :pure (purity/pure-op? 'aget)))
    (is (= :pure (purity/pure-op? 'alength)))
    (is (= :pure (purity/pure-op? 'aclone))))

  (testing "IO ops are impure"
    (is (= :impure (purity/pure-op? 'println)))
    (is (= :impure (purity/pure-op? 'clojure.core/println)))
    (is (= :impure (purity/pure-op? 'print))))

  (testing "Mutation ops are impure"
    (is (= :impure (purity/pure-op? 'swap!)))
    (is (= :impure (purity/pure-op? 'reset!)))
    (is (= :impure (purity/pure-op? 'aset))))

  (testing ".invk dispatch is pure"
    (is (= :pure (purity/pure-op? '.invk))))

  (testing "Unknown ops return :unknown"
    (is (= :unknown (purity/pure-op? 'some-random-fn)))))

;; ================================================================
;; Beichte effect analysis
;; ================================================================

(deftest beichte-effect-test
  (testing "pure vars"
    (is (true? (purity/pure-var? #'clojure.core/+)))
    (is (true? (purity/pure-var? #'clojure.core/identity))))

  (testing "impure vars"
    (is (not (purity/pure-var? #'clojure.core/println)))))

;; ================================================================
;; Validate for AD
;; ================================================================

(deftest validate-for-ad-test
  (testing "Pure body returns nil (no warnings)"
    (let [body '(let* [a (Math/sin x)
                       b (Math/cos x)]
                      (+ a b))]
      (is (nil? (purity/validate-for-ad! body #{'x})))))

  (testing "Impure body returns warnings"
    (let [body '(let* [_ (println "debug")
                       a (Math/sin x)]
                      a)
          warnings (purity/validate-for-ad! body #{'x})]
      (is (seq warnings))
      (is (= :impure-op-in-ad-path (:warning (first warnings))))
      (is (= 'println (:op (first warnings)))))))
