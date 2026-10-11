(ns raster.compiler.passes.parallel.functional-fresh-write-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defn- typed [form tag] (with-meta form {:raster.type/tag tag}))
(defn- square [] (typed '(raster.numeric/* x x) 'double))
(defn- effect [body] (list 'raster.par/map-void! 'i 'size body))
(defn- store [value] (list 'clojure.core/aset 'out 'i value))
(defn- source [pairs body] (list 'let* (vec (mapcat identity pairs)) body))
(def ^:private facts {:array-types {} :scalar-types {'x :double 'size :long}})
(defn- project [source] (frontend/canonicalize-fresh-full-write-maps source facts))

(deftest single-write-projection-preserves-allocation-effect-site-and-nil-statement
  (let [original (source [['out '(clojure.core/double-array size)]
                          ['effect (effect (store (square)))]] 'out)
        result (project original)
        pairs (vec (partition 2 (second (:form result))))]
    (is (= 1 (count (:rewrites result))))
    (is (empty? (:declines result)))
    (is (= '[out (clojure.core/double-array size)] (first pairs)))
    (is (= 'raster.par/map! (first (second (second pairs)))))
    (is (= '[out i size nil] (vec (take 4 (rest (second (second pairs)))))))
    (is (= (square) (last (second (second pairs)))))
    (is (= '[effect nil] (last pairs)))
    (is (= 'out (last (:form result))))
    (is (= :double (get-in result [:rewrites 0 :allocation :dtype])))
    (is (= 'doubles (:raster.type/tag (meta (ffirst pairs)))))))

(deftest missing-proof-does-not-use-a-default-dtype-or-change-source
  (let [allocation ['out '(clojure.core/double-array size)]
        good-effect ['effect (effect (store (square)))]]
    (doseq [[pairs tail]
            [[[allocation ['effect (effect (store '(unknown-helper x)))]] 'out]
             [[[ 'out '(clojure.core/float-array size)] good-effect] 'out]
             [[['out '(user-defined/double-array size)] good-effect] 'out]
             [[['out '(clojure.core/double-array 4)] good-effect] 'out]
             [[allocation ['before '(clojure.core/aget out 0)] good-effect] 'out]
             [[allocation ['alias 'out] good-effect] 'out]
             [[allocation ['effect (effect (store (typed '(clojure.core/aget out i) 'double)))]] 'out]
             [[allocation ['effect (effect (list 'do (store (square)) (store (square))))]] 'out]
             [[allocation ['effect (effect (list 'if 'flag (store (square)) nil))]] 'out]
             [[allocation good-effect] '[effect out]]
             [[allocation ['effect (effect (store (typed '(clojure.core/float x) 'float)))]] 'out]]]
      (let [original (source pairs tail)
            result (project original)]
        (is (empty? (:rewrites result)))
        (is (seq (:declines result)))
        (is (= original (:form result)))))))

(deftest final-double-value-does-not-hide-intermediate-float-rounding
  (let [original (source [['out '(clojure.core/double-array size)]
                          ['effect (effect (store (typed
                                                   '(clojure.core/double (clojure.core/float x))
                                                   'double)))]] 'out)
        result (project original)]
    (is (empty? (:rewrites result)))
    (is (= :fresh-write-unsupported-conversion (get-in result [:declines 0 :reason])))
    (is (= original (:form result)))))

(deftest lexical-constructor-alias-is-not-a-fresh-builtin-allocation
  (doseq [allocation ['(double-array size)
                      (with-meta '(double-array size) {:raster.op/lexical-callee true})
                      (with-meta '(.invk double-array size)
                        {:raster.op/lexical-callee true
                         :raster.op/original 'clojure.core/double-array})]]
    (let [original (source [['double-array 'allocator]
                            ['out allocation]
                            ['effect (effect (store (square)))]] 'out)
          result (project original)]
      (is (empty? (:rewrites result)))
      (is (= :fresh-write-unproved-allocation (get-in result [:declines 0 :reason])))
      (is (= original (:form result)))))
  ;; A retained callable parameter's lexical dispatch marker is equally decisive,
  ;; even if legacy metadata records the builtin spelling.
  (let [allocation (with-meta '(.invk double-array size)
                     {:raster.op/lexical-callee true
                      :raster.op/original 'clojure.core/double-array})
        original (source [['out allocation]
                          ['effect (effect (store (square)))]] 'out)
        result (project original)]
    (is (empty? (:rewrites result)))
    (is (= :fresh-write-unproved-allocation (get-in result [:declines 0 :reason])))))

(deftest immutable-extent-needs-positive-integral-evidence
  (let [original (source [['out '(clojure.core/double-array size)]
                          ['effect (effect (store (square)))]] 'out)]
    (doseq [scalar-types [{'x :double} {'x :double 'size :double}]]
      (let [result (frontend/canonicalize-fresh-full-write-maps
                    original {:array-types {} :scalar-types scalar-types})]
        (is (empty? (:rewrites result)))
        (is (seq (:declines result)))))))

(deftest surrounding-effects-and-source-metadata-stay-at-their-original-sites
  (let [statement (with-meta 'effect {:line 17 :raster.effect/effectful true})
        original (with-meta
                   (source [['before '(notify! :before)]
                            ['out '(clojure.core/double-array size)]
                            [statement (effect (store (square)))]
                            ['after '(notify! :after)]] 'out)
                   {:line 11})
        result (project original)
        pairs (vec (partition 2 (second (:form result))))]
    (is (= 1 (count (:rewrites result))))
    (is (= '[before (notify! :before)] (first pairs)))
    (is (= '[after (notify! :after)] (last pairs)))
    (is (= [statement nil] (nth pairs 3)))
    (is (= (meta statement) (meta (first (nth pairs 3)))))
    (is (= (meta original) (meta (:form result))))))

(deftest public-prepared-tail-and-integral-identity-extent-use-shared-normalizers
  ;; Actual public preparation shape: allocation receives ANF's proved identity
  ;; long cast while the effect launch still names the original Long parameter.
  (let [allocation '(clojure.core/double-array anf__1)
        final '(raster.par/reduce total 0.0 i size
                 (raster.numeric/+ total (clojure.core/aget out_alpha i)))
        original (with-meta
                   (list 'let* (with-meta ['anf__1 '(clojure.core/long size)
                                          'out_alpha allocation] {:line 19})
                         '(notify! :before)
                         (list 'raster.par/map-void! 'i 'size
                               (list 'clojure.core/aset 'out_alpha 'i (square)))
                         '(notify! :after) final)
                   {:line 17})
        result (project original)
        pairs (vec (partition 2 (second (:form result))))]
    (is (= 1 (count (:rewrites result))))
    (is (empty? (:declines result)))
    ;; Do not erase or reevaluate the cast/allocation; canonical equality is proof
    ;; only. The original executable allocation remains before all statements.
    (is (= '[anf__1 (clojure.core/long size)] (first pairs)))
    (is (= allocation (second (second pairs))))
    (is (= '(notify! :before) (second (nth pairs 2))))
    (is (= 'raster.par/map! (first (second (nth pairs 3)))))
    (is (nil? (second (nth pairs 4))))
    (is (= '(notify! :after) (second (nth pairs 5))))
    (is (= final (last (:form result))))
    (is (= {:line 19} (meta (second (:form result)))))
    (is (= (meta original) (meta (:form result))))))

(deftest extent-equality-does-not-erase-narrowing-arithmetic-or-lexical-conversions
  (doseq [initializer ['(clojure.core/int size)
                       '(clojure.core/unchecked-add size 1)
                       (typed '(long size) 'long)]]
    (let [pairs (cond-> []
                  (= initializer (typed '(long size) 'long))
                  (conj ['long 'converter]))
          original (list 'let* (vec (mapcat identity
                                            (into pairs [['extent initializer]
                                                         ['out '(clojure.core/double-array extent)]])))
                         (effect (store (square))) 'out)
          result (project original)]
      (is (empty? (:rewrites result)))
      (is (seq (:declines result)))
      ;; No effect normalization or metadata changes leak out on denial.
      (is (= original (:form result))))))
