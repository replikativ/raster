(ns raster.compiler.passes.parallel.auxiliary-source-match-cost-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.passes.parallel.auxiliary-semantic-witness-test :as witness]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- fixture []
  (let [program (#'witness/functional-program)
        retained (first (mapcat #(get-in % [:attributes :realization-bindings])
                               (:equations program)))]
    [program retained]))

(defn- append-pairs [program pairs]
  (update program :source
          (fn [[head bindings & body :as source]]
            (with-meta (list* head (into bindings (mapcat identity pairs)) body) (meta source)))))

(deftest binding-search-does-not-encode-unrelated-source-expressions
  (let [[program {:keys [binding expression]}] (fixture)
        unrelated (mapv #(vector (symbol (str "unrelated_" %)) (Object.)) (range 32))
        program (append-pairs program unrelated)
        original semantic/equivalent?
        pairs (atom [])]
    (with-redefs [semantic/equivalent?
                  (fn [left right]
                    (when (and (vector? left) (vector? right)
                               (= 2 (count left) (count right))
                               (symbol? (first left)) (symbol? (first right)))
                      (swap! pairs conj [(first left) (first right)]))
                    (original left right))]
      (is (route/realization-binding? program binding expression)))
    (is (seq @pairs))
    (is (every? (fn [[left right]] (= left right)) @pairs))
    (is (not-any? (fn [[left right]]
                    (some #(or (= left (first %)) (= right (first %))) unrelated)) @pairs))
    ;; This is a binding-scoped witness predicate, not admission of the unrelated
    ;; runtime objects as a complete source program.
    (is (<= (count @pairs) 4) "only candidate pairs and independent allocation evidence")))

(deftest same-binding-candidates-retain-exact-comparison-and-uniqueness
  (let [[program {:keys [binding expression]}] (fixture)]
    (testing "an unsupported value under the matching binder still fails loudly"
      (let [changed (#'witness/replace-binding program binding (Object.))]
        (is (= :semantic-fingerprint-unsupported
               (try (route/realization-binding? changed binding expression) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
    (testing "all exact same-binding pairs count, not merely the first"
      (is (false? (route/realization-binding?
                   (append-pairs program [[binding expression]]) binding expression))))
    (testing "ordinary binder equality cannot authorize changed type metadata"
      (let [forged (with-meta binding (assoc (meta binding) :raster.type/tag 'float))
            changed (update program :source
                            (fn [[head bindings & body]]
                              (list* head
                                     (vec (mapcat (fn [[id value]]
                                                    [(if (= binding id) forged id) value])
                                                  (partition 2 bindings))) body)))]
        (is (= binding forged))
        (is (false? (route/realization-binding? changed binding expression)))))))
