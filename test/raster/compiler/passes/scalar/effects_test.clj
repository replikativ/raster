(ns raster.compiler.passes.scalar.effects-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.scalar.effects :as effects]))

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
