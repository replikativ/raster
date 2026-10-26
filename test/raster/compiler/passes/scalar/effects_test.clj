(ns raster.compiler.passes.scalar.effects-test
  (:require [clojure.test :refer [deftest is]]
            [beichte.analyze :as beichte-analyze]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.core.op-descriptor :as descriptor]))

(defn unowned-effect-source [value] value)

(deftest sharing-and-observation-invalidation-are-distinct
  (is (= {:shareable? true :invalidates-observations? false}
         (effects/cse-properties '(clojure.core/aget input 0))))
  (is (= {:shareable? false :invalidates-observations? true}
         (effects/cse-properties '(clojure.core/aset input 0 9.0))))
  (is (= {:shareable? false :invalidates-observations? true}
         (effects/cse-properties '(clojure.core/int unknown))))
  (with-redefs [effects/descriptor (constantly {:effect :pure :flags #{}})
                descriptor/resolve-op-descriptor
                (fn [_] [{:buffer-write {:buf-arg-idx 0}} 'fixture/write])]
    (is (= {:shareable? false :invalidates-observations? true}
           (effects/cse-properties '(fixture/write input))))))

(deftest collection-elements-have-execution-effects-but-quoted-data-does-not
  (doseq [expression ['[(clojure.core/aset input 0 9.0)]
                      '{:write (clojure.core/aset input 0 9.0)}
                      '{(clojure.core/aset input 0 9.0) :write}
                      '#{(clojure.core/aset input 0 9.0)}]]
    (is (not (effects/removable-expr? expression)))
    (is (= {:shareable? false :invalidates-observations? true}
           (effects/cse-properties expression))))
  (is (effects/removable-expr? '(quote [(clojure.core/aset input 0 9.0)]))))

(deftest optional-effect-analysis-never-replays-unowned-var-source
  (let [lookups (atom [])
        operation 'raster.compiler.passes.scalar.effects-test/unowned-effect-source]
    (with-redefs [beichte-analyze/source-var
                  (fn [v]
                    (swap! lookups conj v)
                    (throw (ex-info "compiler must not expand this source" {})))]
      (is (= :io (effects/analyze-var-effect #'unowned-effect-source)))
      (is (= :io (:effect (effects/descriptor (list operation 1)))))
      (is (not (effects/removable-expr? (list operation 1))))
      (is (= :pure (:effect (effects/descriptor '(clojure.core/+ 1 2)))))
      (is (empty? @lookups)))))

(deftest producer-denials-do-not-reinterpret-ordinary-arithmetic-contracts
  (let [ordinary '(clojure.core/+ x 1)
        retained (with-meta ordinary {:raster.effect/preserve-evaluation true})]
    (is (effects/removable-expr? ordinary) "the existing general arithmetic API is unchanged")
    (doseq [expression [retained [retained] (list 'do retained 2.0)]]
      (is (contains? (:flags (effects/descriptor expression)) :preserve-source-evaluation))
      (is (not (effects/removable-expr? expression))))
    (is (not (contains? (:flags (effects/descriptor (list 'quote retained)))
                       :preserve-source-evaluation)))
    (is (effects/removable-expr? ordinary) "metadata-distinct cache observations stay separate"))
  (let [attributes {:source-dtype :double :target-dtype :float
                    :rounding :nearest-even :overflow :ieee :source-op 'clojure.core/float}]
    (is (conversion/canonical-attributes? attributes))
    (is (not (conversion/canonical-attributes? (assoc attributes :overflow :trap)))
        "existing numerical IR policy is not a new checked JVM source-cast proof")))

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
                    effects/descriptor
                    (fn [_] (throw (ex-info "unnecessary body analysis" {})))]
        (is (false? (effects/cse-safe-expr? expression)))
        (is (= [(descriptor/semantic-op expression)] @resolved))))))

(deftest storage-contracts-never-grant-cse-permission
  (doseq [contract [nil {} {:effects {:pure? true}}
                    {:effects {:mutating? false} :buffer {:allocates? false}}]
          removable? [false true]]
    (let [analyzed (atom []) expression '(fixture/operation buffer)]
      (with-redefs [descriptor/resolve-op-descriptor (fn [_] [contract 'fixture/operation])
                    effects/descriptor (fn [expr] (swap! analyzed conj expr)
                                         {:effect (if removable? :pure :io) :flags #{}})]
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
                    effects/descriptor (fn [expr] (swap! analyzed conj expr)
                                         {:effect :io :flags #{}})]
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
        (with-redefs [effects/descriptor
                      (fn [_] (throw (ex-info "declared writer was analyzed" {})))]
          (is (false? (effects/cse-safe-expr? expression)))))
      (is (nil? (resolve-contract 'foreign/dense-backward-dW-into!_m_doubles_doubles_doubles)))
      (let [analyzed (atom 0)]
        (with-redefs [effects/descriptor (fn [_] (swap! analyzed inc)
                                         {:effect :io :flags #{}})]
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
    (is (not (effects/removable-expr? allocation))
        "external purity does not discharge a dynamic constructor's failures")
    (is (empty? (:flags (effects/descriptor '(clojure.core/double-array 8))))
        "closed valid size adds no denial; DCE still checks its effect policy")
    (is (not (effects/cse-safe-expr? allocation))
        "two live allocations may not share mutable identity")))

(deftest allocation-evaluation-requires-a-closed-valid-size
  (doseq [expression ['(clojure.core/double-array -1)
                      '(clojure.core/double-array n)
                      '(clojure.core/double-array 2147483648)
                      '(clojure.core/double-array (clojure.core/int 2147483648))
                      '(clojure.core/long-array (clojure.core/int 8))
                      '(clojure.core/double-array [1.0 2.0])
                      '(clojure.core/double-array 2 (effectful-initializer))
                      '(clojure.core/aclone source)
                      (with-meta '(clojure.core/double-array 2)
                        {:raster.op/lexical-callee true})]]
    (is (contains? (:flags (effects/descriptor expression)) :source-allocation-evaluation)
        (pr-str expression))
    (is (not (effects/removable-expr? expression))))
  (doseq [expression ['(clojure.core/double-array 0)
                      '(clojure.core/float-array 8)
                      '(clojure.core/long-array 8)]]
    (is (not (contains? (:flags (effects/descriptor expression))
                       :source-allocation-evaluation))))
  (is (effects/removable-expr? '(quote (clojure.core/double-array -1)))
      "constructor-shaped quoted data is not an evaluation"))

(deftest constructor-permission-does-not-survive-replaced-core-inline
  (let [constructor #'clojure.core/double-array
        original (meta constructor)
        expression '(clojure.core/double-array 17)
        denied? #(contains? (:flags (effects/descriptor expression))
                            :source-allocation-evaluation)]
    ;; Exercise a cache hit independently of Beichte registry version/details.
    ;; The expression and resolved Var remain identical throughout the test.
    (with-redefs-fn {#'effects/closed-registry-proof (constantly [constructor])}
      (fn []
        (is (not (denied?)))
        (is (not (denied?)))
        (doseq [[key replacement] [[:inline (fn [& _] '(throw (Exception. "replaced inline")))]
                                   [:inline-arities (constantly false)]]]
          (try
            (alter-meta! constructor assoc key replacement)
            (is (denied?))
            (finally (reset-meta! constructor original))))
        (with-redefs-fn {constructor (fn [& _] (throw (Exception. "replacement must not run")))}
          #(is (denied?)))
        (is (not (denied?)) "restoration reinstates the original constructor contract")))))

(deftest constructor-omission-retains-the-existing-source-owner
  (let [inputs (atom {})]
    (binding [method-entry/*emission-inputs* inputs]
      (is (empty? (:flags (effects/descriptor '(clojure.core/double-array 8)))))
      (is (contains? (:definition-proofs @inputs) #'clojure.core/double-array))
      (is (true? (method-entry/assert-definition-inputs-current! @inputs))))))

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
    (is (effects/removable-expr? certified))
    (is (not (effects/removable-expr? form)) "a cached certified result is not reused")))
