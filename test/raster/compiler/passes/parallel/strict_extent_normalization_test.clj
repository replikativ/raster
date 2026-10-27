(ns raster.compiler.passes.parallel.strict-extent-normalization-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defn- long-call [operation & arguments]
  (with-meta (apply list operation arguments) {:raster.type/tag 'long}))

(defn- extract [expression]
  (#'frontend/name-strict-extent-subexpressions
   {:normalized [] :local-scalar-types {'n :long 'm :long 'blocks :long 'zero :long}}
   expression #{} 0))

(defn- execute-extracted [state expression environment]
  (let [environment
        (reduce (fn [environment [id init]]
                  (assoc environment id (long (scalar/evaluate-expression 'clojure.core environment init))))
                environment (:normalized state))]
    (scalar/evaluate-expression 'clojure.core environment expression)))

(defn- outcome [f]
  (try {:value (f)}
       (catch ArithmeticException error
         {:exception (class error) :message (.getMessage error)})))

(deftest strict-long-subexpressions-are-named-in-original-evaluation-order
  (let [product (long-call 'clojure.core/* 'n 'm)
        blocked (long-call 'clojure.core/* product 'blocks)
        extent (long-call 'clojure.core/* blocked 16)
        [state rewritten] (extract extent)
        [[product-id product-init] [blocked-id blocked-init]] (:normalized state)]
    (is (= 2 (count (:normalized state))))
    (is (semantic/retained-source-equivalent? product product-init))
    (is (= (list 'clojure.core/* product-id 'blocks) blocked-init))
    (is (= (list 'clojure.core/* blocked-id 16) rewritten))
    (is (every? #(= :long (get-in state [:local-scalar-types (first %)]))
                (:normalized state)))
    (is (= 960 (execute-extracted state rewritten {'n 3 'm 4 'blocks 5})))
    (let [later (with-meta product (assoc (meta product) :line 100 :column 4))
          [next-state reused]
          (#'frontend/name-strict-extent-subexpressions state later #{} 1)]
      (is (= product-id reused) "source locations do not prevent exact semantic reuse")
      (is (= (:normalized state) (:normalized next-state))))
    (let [different-policy (with-meta product (assoc (meta product) :source-overflow :wrap))
          [_ retained]
          (#'frontend/name-strict-extent-subexpressions state different-policy #{} 1)]
      (is (semantic/retained-source-equivalent? different-policy retained)
          "semantic policy metadata is not ignored when matching a prior computation"))))

(deftest strict-extraction-preserves-first-failing-operation
  (let [division (long-call 'clojure.core/quot 'n 'zero)
        overflow (long-call 'clojure.core/* 'n 'n)
        environment {'n Long/MAX_VALUE 'zero 0}]
    (doseq [[left right] [[division overflow] [overflow division]]]
      (let [expression (long-call 'clojure.core/+ left right)
            [state rewritten] (extract expression)]
        (is (= 2 (count (:normalized state))))
        (is (= (outcome #(scalar/evaluate-expression 'clojure.core environment expression))
               (outcome #(execute-extracted state rewritten environment))))))))

(deftest rejected-trees-do-not-extract-even-an-admissible-child
  (let [good (long-call 'clojure.core/* 'n 'm)
        int-product (with-meta (list 'clojure.core/* 'n 'm) {:raster.type/tag 'int})
        narrowing (with-meta (list 'clojure.core/int 'n) {:raster.type/tag 'int})
        read (long-call 'clojure.core/aget 'array 0)]
    (doseq [[reason expression]
            [[:int-arithmetic (long-call 'clojure.core/+ good int-product)]
             [:narrowing (long-call 'clojure.core/+ good narrowing)]
             [:untyped (long-call 'clojure.core/+ good '(clojure.core/* n m))]
             [:unqualified-operation (long-call '+ good (long-call '* 'n 'm))]
             [:conditional (long-call 'if true good (long-call 'clojure.core/quot 'n 'zero))]
             [:array-read (long-call 'clojure.core/+ good read)]]]
      (testing (name reason)
        (let [[state retained] (extract expression)]
          (is (empty? (:normalized state)))
          (is (semantic/retained-source-equivalent? expression retained)))))
    (let [expression (long-call 'clojure.core/+ good 'blocks)
          [state retained] (binding [util/*shadowing-locals* #{'clojure.core/*}]
                             (extract expression))]
      (is (empty? (:normalized state)))
      (is (semantic/retained-source-equivalent? expression retained)))))

(deftest allocation-extent-subexpression-dominates-a-later-map-extent
  (let [product (long-call 'clojure.core/* 'n 'm)
        allocation-extent (long-call 'clojure.core/*
                                    (long-call 'clojure.core/* product 'blocks) 16)
        source (list 'let*
                     ['partials (list 'clojure.core/int-array allocation-extent)
                      'producer '(raster.par/map! partials i blocks int 1)
                      'consumer (list 'raster.par/map! 'out 'i product 'int 2)]
                     'consumer)
        options {:scalar-types {'n :long 'm :long 'blocks :long}
                 :array-types {'out :int}}
        normalized (frontend/normalize-source source options)
        pairs (mapv vec (partition 2 (second normalized)))
        product-pair (first (filter #(semantic/retained-source-equivalent? product (second %)) pairs))
        by-id (into {} pairs)]
    (is (some? product-pair))
    (is (= (first product-pair) (nth (by-id 'consumer) 3))
        "the later map uses the already evaluated scalar, not a hoisted new check")
    (is (= 1 (count (filter #(semantic/retained-source-equivalent? product (second %)) pairs))))
    (is (= normalized (frontend/normalize-source normalized options)))))
