(ns raster.compiler.passes.scalar.effects-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.op-descriptor :as descriptor]))

(deftest object-context-completion-inherits-evaluation-not-sharing-permission
  (doseq [operand [nil 1.0
                   (with-meta '(raster.numeric/+ x y) {:raster.type/tag 'long})
                   '(clojure.core/double-array -1)
                   '(clojure.core/double-array n)
                   '(fixture/unknown x)]]
    (let [expression (completion/complete-value completion/object-input-contract operand)]
      (is (= (effects/descriptor operand) (effects/descriptor expression)))
      (is (= (effects/removable-expr? operand) (effects/removable-expr? expression)))
      (is (false? (effects/cse-safe-expr? expression))
          "separate Object materializations may have distinct boxing identities")))
  (doseq [expression [(completion/complete-value
                      (completion/primitive-input-contract Long/TYPE {:unchecked-math false}) nil)
                     (list completion/completion-symbol {:policy :unknown} nil)]]
    (is (not (effects/removable-expr? expression)))))

(deftest checked-integral-evaluation-is-not-external-impurity
  (let [checked (with-meta '(raster.numeric/+ x y) {:raster.type/tag 'long})
        wrapped (with-meta '(clojure.core/unchecked-add x y) {:raster.type/tag 'long})
        floating (with-meta '(raster.numeric/+ x y) {:raster.type/tag 'double})
        source (effects/descriptor checked)]
    (is (= :pure (:effect source)))
    (is (contains? (:flags source) :checked-source-arithmetic))
    (is (not (effects/removable-expr? checked)))
    (is (not (contains? (:flags (effects/descriptor wrapped)) :checked-source-arithmetic)))
    (is (not (contains? (:flags (effects/descriptor floating)) :checked-source-arithmetic)))
    (is (contains? (:flags (effects/descriptor (list 'clojure.core/float checked)))
                   :checked-source-arithmetic)
        "a floating consumer does not erase the original checked integral operation")))

(deftest literal-range-proof-discharge-is-separate-from-source-wrapping
  (doseq [[operation arguments trapped?]
          [['clojure.core/+ [1 2] false]
           ['clojure.core/+ [Long/MAX_VALUE 1] true]
           ['clojure.core/* [2 3] false]
           ['clojure.core/* [Long/MAX_VALUE 2] true]
           ['clojure.core/unchecked-add [Long/MAX_VALUE 1] false]]]
    (let [expression (with-meta (list* operation arguments) {:raster.type/tag 'long})]
      (is (= trapped? (contains? (:flags (effects/descriptor expression)) :checked-source-arithmetic))))))

(deftest shared-ordering-obligations-distinguish-total-values
  (doseq [expression [(with-meta '(raster.numeric/* rows width) {:raster.type/tag 'long})
                      '(clojure.core/int n)]]
    (is (effects/ordered-scalar-evaluation-required? expression)))
  (doseq [expression [nil 1.0 '(clojure.core/double-array 3)
                      (with-meta '(clojure.core/unchecked-multiply rows width)
                        {:raster.type/tag 'long})]]
    (is (not (effects/ordered-scalar-evaluation-required? expression)))))

(deftest partial-divisor-evaluation-is-not-external-impurity
  (doseq [expression [(with-meta '(clojure.core/quot x y) {:raster.type/tag 'long})
                      '(clojure.core/quot x y)
                      (with-meta '(clojure.core/rem x y) {:raster.type/tag 'double})
                      '(clojure.core/quot 1 0)]]
    (is (= :pure (effects/analyze-effect expression)))
    (is (effects/ordered-scalar-evaluation-required? expression))
    (is (not (effects/removable-expr? expression))))
  (doseq [expression [(with-meta '(clojure.core/quot x 2) {:raster.type/tag 'long})
                      (with-meta '(clojure.core// x y) {:raster.type/tag 'double})]]
    (is (not (effects/ordered-scalar-evaluation-required? expression)))))

(deftest array-extent-obligations-use-the-source-number-narrowing
  (doseq [[extent checked?] [[0 false] [3 false] [-1 true] [2147483648 true]
                             [4294967296 false] [4294967299 false] [-4294967296 false] ['n true]]]
    (let [expression (list 'clojure.core/double-array extent)]
      (is (= checked? (contains? (:flags (effects/descriptor expression)) :source-array-extent)))))
  (is (not (contains? (:flags (effects/descriptor '(quote (clojure.core/double-array -1))))
                      :source-array-extent))))

(deftest source-semantic-flags-are-not-stale-effect-cache-entries
  (let [expression (with-meta '(raster.numeric/+ x y) {:raster.type/tag 'long})]
    (is (contains? (:flags (effects/descriptor expression)) :checked-source-arithmetic))
    (with-redefs [intrinsics/source-overflow-policy (constantly :wrap)]
      (is (not (contains? (:flags (effects/descriptor expression)) :checked-source-arithmetic))))
    (is (contains? (:flags (effects/descriptor expression)) :checked-source-arithmetic))))

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
    (is (= :pure (:effect (effects/descriptor pure-call))))
    (is (contains? (:flags (effects/descriptor pure-call)) :checked-source-arithmetic))
    (is (not (effects/removable-expr? pure-call))
        "external purity does not erase a checked Long multiplication")
    (is (not (effects/removable-expr? unknown-call))
        "an invk without certified semantic metadata remains conservative")))

(deftest allocation-purity-does-not-imply-commonable-identity
  (let [allocation
        (with-meta
          '(.invk raster.arrays/zeros-like_m_floats_long-impl exemplar n)
          {:raster.op/original 'raster.arrays/zeros-like
           :raster.type/tag 'floats :tag 'floats})]
    (is (= :pure (:effect (effects/descriptor allocation))))
    (is (contains? (:flags (effects/descriptor allocation)) :source-allocation-evaluation))
    (is (not (effects/removable-expr? allocation))
        "an allocation facet alone does not prove that a dynamic extent is valid")
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
    (is (= :pure (:effect (effects/descriptor certified))))
    (is (contains? (:flags (effects/descriptor certified)) :checked-source-arithmetic))
    (is (not (effects/removable-expr? form)) "a cached certified result is not reused")))
