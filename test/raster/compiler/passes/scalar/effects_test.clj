(ns raster.compiler.passes.scalar.effects-test
  (:require [clojure.test :refer [deftest is]]
            [beichte.core :as beichte]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.core.op-descriptor :as descriptor]))

(deftest consumed-value-proofs-retain-source-evaluation
  (doseq [source ['(clojure.core/+ x y) '(clojure.core// x y)
                  (with-meta '(.invk raster.numeric/_plus__m_long_long-impl x y)
                    {:raster.op/original 'raster.numeric/+})]]
    (let [expression (vary-meta source assoc :raster.effect/preserve-evaluation true)]
      (is (contains? (:flags (effects/descriptor expression)) :preserve-source-evaluation))
      (is (false? (effects/removable-expr? expression)))
      (is (false? (effects/cse-safe-expr? expression))))))

(deftest optional-purity-abstention-cannot-drop-source-obligations
  (doseq [failure [(IllegalArgumentException. "unsupported analysis")
                   (StackOverflowError. "bounded analyzer refusal")]
          [expression flag]
          [[(with-meta '(unknown/effect x) {:raster.effect/preserve-evaluation true})
            :preserve-source-evaluation]
           [(with-meta '(unknown/conversion x) {:raster.effect/source-reference-cast true})
            :source-reference-cast]
           ['(clojure.core/int x) :checked-source-cast]]]
    (with-redefs [beichte/analyze-full (fn [& _] (throw failure))]
      ;; Call the analyzer directly so an unrelated cached purity result cannot
      ;; hide the abstention path under test.
      (let [result (#'effects/analyze-descriptor expression)]
        (is (= :io (:effect result)))
        (is (contains? (:flags result) flag))))))

(deftest optional-obligation-abstention-retains-a-generic-denial
  (doseq [failure [(IllegalArgumentException. "unsupported obligation analysis")
                   (StackOverflowError. "bounded obligation refusal")]]
    (with-redefs-fn {#'effects/source-cast-flags (fn [& _] (throw failure))}
      #(let [result (#'effects/analyze-descriptor '(clojure.core/double 1.0))]
         (is (= :io (:effect result)))
         (is (contains? (:flags result) :unclosed-source-obligations))))))

(deftest storage-contracts-deny-cse-without-body-analysis
  (doseq [contract [{:buffer-write {:mode :overwrite :buf-arg-idx 0}}
                    {:buffer-write {:mode :accumulate :buf-arg-idx 0}}
                    {:effects {:mutating? true}}
                    {:buffer {:allocates? true}}]
          expression ['(fixture/write! buffer value)
                      '(fixture/write!_m_doubles_double buffer value)
                      (with-meta '(.invk fixture/write-impl buffer value)
                        {:raster.op/original 'fixture/write!})]]
    (let [resolved (atom [])]
      (with-redefs [descriptor/resolve-op-descriptor
                    (fn [op] (swap! resolved conj op) [contract op])
                    effects/removable-expr?
                    (fn [_] (throw (ex-info "unnecessary body analysis" {})))]
        (is (false? (effects/cse-safe-expr? expression)))
        (is (= [(descriptor/semantic-op expression)] @resolved))))))

(deftest storage-contracts-never-grant-cse-permission
  (doseq [contract [nil {} {:effects {:pure? true}}
                    {:effects {:mutating? false} :buffer {:allocates? false}}]
          removable? [false true]]
    (let [analyzed (atom []) expression '(fixture/operation buffer)]
      (with-redefs [descriptor/resolve-op-descriptor (fn [_] [contract 'fixture/operation])
                    effects/removable-expr? (fn [expr] (swap! analyzed conj expr) removable?)]
        (is (= removable? (effects/cse-safe-expr? expression)))
        (is (= [expression] @analyzed))))))

(deftest lexical-and-unidentified-calls-do-not-borrow-storage-contracts
  (doseq [expression [(with-meta '(fixture/write! buffer value)
                       {:raster.op/lexical-callee true})
                     (with-meta '(.invk fixture/write-impl buffer value)
                       {:raster.op/lexical-callee true
                        :raster.op/original 'fixture/write!})
                     '(.invk fixture/write-impl buffer value)]]
    (let [analyzed (atom [])]
      (with-redefs [descriptor/resolve-op-descriptor
                    (fn [_] (throw (ex-info "no global semantic identity" {})))
                    effects/removable-expr? (fn [expr] (swap! analyzed conj expr) false)]
        (is (false? (effects/cse-safe-expr? expression)))
        (is (= [expression] @analyzed))))))

(deftest declared-storage-resolution-preserves-qualified-identity
  (let [resolve-contract descriptor/resolve-op-descriptor
        operation 'raster.nn/dense-backward-dW-into!
        contract {:buffer-write {:mode :accumulate :buf-arg-idx 2}}]
    ;; Exercise the real resolver, including its mangled-name path, without
    ;; requiring the training library or mutating its process-wide registry.
    (with-redefs [descriptor/get-op-descriptor (fn [op] (when (= op operation) contract))]
      (doseq [expression ['(raster.nn/dense-backward-dW-into! dy x out)
                          '(raster.nn/dense-backward-dW-into!_m_doubles_doubles_doubles dy x out)
                          (with-meta '(.invk generated/dw-impl dy x out)
                            {:raster.op/original operation})]]
        (with-redefs [effects/removable-expr?
                      (fn [_] (throw (ex-info "declared writer was analyzed" {})))]
          (is (false? (effects/cse-safe-expr? expression)))))
      (is (nil? (resolve-contract 'foreign/dense-backward-dW-into!_m_doubles_doubles_doubles)))
      (let [analyzed (atom 0)]
        (with-redefs [effects/removable-expr? (fn [_] (swap! analyzed inc) false)]
          (is (false? (effects/cse-safe-expr?
                       '(foreign/dense-backward-dW-into!_m_doubles_doubles_doubles dy x out))))
          (is (= 1 @analyzed)))))))

(deftest beichte-analyzes-the-semantic-operation-behind-invk
  (let [pure-call
        (with-meta
          '(.invk raster.numeric/_star__m_long_long-impl rows width)
          {:raster.op/original 'raster.numeric/*
           :raster.type/tag 'long :tag 'long})
        unknown-call '(.invk user/unknown-impl rows width)]
    (is (= {:effect :pure :flags #{}} (effects/descriptor pure-call)))
    (is (effects/removable-expr? pure-call))
    (is (not (effects/removable-expr? unknown-call))
        "an invk without certified semantic metadata remains conservative")))

(deftest allocation-purity-does-not-imply-commonable-identity
  (let [allocation
        (with-meta
          '(.invk raster.arrays/zeros-like_m_floats_long-impl exemplar n)
          {:raster.op/original 'raster.arrays/zeros-like
           :raster.type/tag 'floats :tag 'floats})]
    (is (effects/removable-expr? allocation)
        "an unused local allocation may be eliminated")
    (is (not (effects/cse-safe-expr? allocation))
        "two live allocations may not share mutable identity")))

(deftest total-conversion-evidence-does-not-invent-an-exception
  (let [int-value (with-meta 'x {:raster.type/tag 'int})
        long-value (with-meta 'x {:raster.type/tag 'long})]
    (doseq [expression [(list 'clojure.core/long int-value)
                        (list 'clojure.core/int int-value)
                        '(clojure.core/int 2147483647)]]
      (is (effects/removable-expr? expression)))
    (doseq [expression [(list 'clojure.core/int long-value)
                        '(clojure.core/int 2147483648)
                        '(clojure.core/long x)]]
      (is (contains? (:flags (effects/descriptor expression)) :checked-source-cast)))))

(deftest checked-source-casts-carry-exceptional-control
  (doseq [expression ['(clojure.core/int x)
                      '(clojure.core/byte x)
                      '(clojure.core/long x)
                      '(clojure.core/* (clojure.core/int x) 0)]]
    (is (= :pure (:effect (effects/descriptor expression)))
        "a possible throw is distinct from an external mutation or IO effect")
    (is (contains? (:flags (effects/descriptor expression)) :checked-source-cast))
    (is (not (effects/removable-expr? expression))))
  (is (not (contains? (:flags (effects/descriptor '(clojure.core/unchecked-int x)))
                      :checked-source-cast))
      "the op descriptor's explicit wrapping cast has no checked exception obligation"))

(deftest checked-floating-casts-retain-source-evaluation-before-canonicalization
  (let [wide (with-meta 'wide {:tag 'double})
        narrow (with-meta 'narrow {:tag 'float})]
    (doseq [expression [(list 'clojure.core/float wide)
                        '(clojure.core/float Double/MAX_VALUE)
                        '(clojure.core/float Double/POSITIVE_INFINITY)
                        '(let* [unused (clojure.core/float wide)] 7)]]
      (is (contains? (:flags (effects/descriptor expression)) :checked-source-cast))
      (is (not (effects/removable-expr? expression))))
    (doseq [expression [(list 'clojure.core/float narrow)
                        '(clojure.core/float 1.0)
                        '(clojure.core/unchecked-float wide)]]
      (is (not (contains? (:flags (effects/descriptor expression)) :checked-source-cast))))))

(deftest lexical-float-callees-do-not-acquire-a-core-cast-contract
  (doseq [expression [(with-meta '(float x) {:raster.op/lexical-callee true})
                      (with-meta '(.invk local/float-impl x)
                        {:raster.op/lexical-callee true :raster.op/original 'clojure.core/float})]]
    (is (nil? (descriptor/semantic-op expression)))
    (is (not (contains? (:flags (effects/descriptor expression)) :checked-source-cast)))))

(deftest generated-reference-conversions-retain-their-evaluation-obligation
  (let [cast '(clojure.core/double x)
        retained (with-meta cast {:raster.effect/source-reference-cast true})
        selected (with-meta '(.invk generated/cast-impl x)
                   {:raster.op/original 'clojure.core/double
                    :raster.effect/source-reference-cast true})]
    (doseq [expression [retained selected]]
      (is (contains? (:flags (effects/descriptor expression)) :source-reference-cast))
      (is (not (effects/removable-expr? expression)))
      (is (not (effects/cse-safe-expr? expression))))
    (is (not (contains? (:flags (effects/descriptor cast)) :source-reference-cast))
        "missing scoped type information alone does not declare a reference cast")))

(deftest cached-descriptors-distinguish-metadata
  ;; Descriptors are memoized. The source operation of a devirtualized call
  ;; lives in metadata, which equality ignores, so the cache must not hand one
  ;; form's descriptor to an equal form with different metadata.
  (let [form '(.invk raster.numeric/_star__m_long_long-impl rows width)
        certified (with-meta form {:raster.op/original 'raster.numeric/*
                                   :raster.type/tag 'long :tag 'long})]
    (is (= form certified))
    (is (not (effects/removable-expr? form)))
    (is (effects/removable-expr? certified))
    (is (not (effects/removable-expr? form)) "a cached certified result is not reused")))
