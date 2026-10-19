(ns raster.ad.product-ports-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.product-ports :as ports]))

(deftest product-ports-retain-fieldwise-activity-and-owning-identity
  (let [x (with-meta 'x {:raster.type/tag 'double})
        plan (ports/literal-plan 'p [x [nil x] false] {'x {:value x}} {'x true})
        fact (select-keys plan [:schema :active-paths])
        copy (ports/literal-plan 'copy ['p] {'p {:product fact}} {'p true})]
    (is (= #{[0] [1 1]} (:active-paths plan)))
    (is (= [(ports/field-port 'p [0]) (ports/field-port 'p [1 0])
            (ports/field-port 'p [1 1]) (ports/field-port 'p [2])]
           (mapv :port (:edges plan))))
    (is (= #{[0 0] [0 1 1]} (:active-paths copy)))
    (is (= (mapv :port (:edges plan)) (mapv :source (:edges copy))))
    (is (not-any? #(= 'x %) (tree-seq coll? seq fact)))
    (is (every? vector? (map :port (:edges plan))))))

(deftest projected-products-rebase-ports-without-importing-child-value-ids
  (let [x (with-meta 'private-child {:raster.type/tag 'double})
        plan (ports/literal-plan 'outer [[x] nil] {x {:value x}} {x true})
        facts {'outer {:product (select-keys plan [:schema :active-paths])}}
        projected (ports/projection-plan 'selected '(clojure.core/nth outer 0) facts)]
    (is (= [{:port (ports/field-port 'selected [0]) :path [0] :tag 'double
             :source (ports/field-port 'outer [0 0]) :active? true}]
           (:edges projected)))
    (is (not-any? #(= x %) (tree-seq coll? seq projected)))
    (is (nil? (ports/projection-plan 'bad '(clojure.core/nth outer index) facts)))
    (is (nil? (ports/projection-plan 'bad '(nth outer 0) facts)))
    (is (nil? (ports/projection-plan 'bad 'outer facts)))
    (is (nil? (ports/projection-plan
               'bad (with-meta '(clojure.core/first outer) {:raster.op/lexical-callee true}) facts)))))

(deftest unknown-and-compound-active-fields-do-not-become-constants
  (is (= :ad-product-unknown-tangent
         (try (ports/literal-plan 'p ['x] {} {'x true}) nil
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
  (is (= :ad-product-unclosed-field
         (try (ports/literal-plan 'p ['(raster.numeric/* x x)] {} {'x true}) nil
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

(deftest projection-consumes-an-owner-proved-index-not-an-index-type
  (let [x (with-meta 'x {:raster.type/tag 'double})
        plan (ports/literal-plan 'p [x nil] {x {:value x}} {x true})
        facts {'p {:product (select-keys plan [:schema :active-paths])}
               'selected {:value 'selected :constant-index 0}}]
    (is (= [{:port 'result :path [] :tag 'double
             :source (ports/field-port 'p [0]) :active? true}]
           (:edges (ports/projection-plan 'result '(clojure.core/nth p selected) facts))))
    (is (nil? (ports/projection-plan 'result '(clojure.core/nth p selected)
                                    (assoc facts 'selected {:value 'selected :tag 'long}))))))

(deftest cyclic-alias-facts-do-not-invent-a-tangent-declaration
  (is (= :ad-product-unknown-tangent
         (try (ports/literal-plan 'p ['a] {'a {:alias 'b} 'b {:alias 'a}} {'a true}) nil
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

(deftest explicit-absence-joins-only-declared-scalar-tangents
  (let [x (with-meta 'x {:raster.type/tag 'double})
        fact #(select-keys % [:schema :active-paths])
        absent (fact (ports/literal-plan 'absent [nil [nil]] {} {}))
        scalar (fact (ports/literal-plan 'scalar [x [x]] {} {'x true}))]
    (doseq [[a b] [[absent scalar] [scalar absent]]]
      (let [joined (ports/join-facts a b)]
        (is (= scalar joined))
        (is (= scalar (ports/join-facts joined absent)))
        (is (not-any? #(and (map? %) (contains? % :presence))
                      (tree-seq coll? seq joined)))))
    (let [unknown (fact (ports/literal-plan 'unknown ['untyped ['untyped]] {} {}))]
      (is (= :ad-product-choice-schema
             (try (ports/join-facts unknown scalar) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest absence-does-not-follow-activity-or-unproved-facts
  (is (= {:kind :leaf :tag 'double}
         (get-in (ports/literal-plan 'p [1.0] {1.0 {:presence :absent}} {}) [:schema :fields 0])))
  (let [fact #(select-keys (ports/literal-plan 'p ['field] {'field %} {'field true})
                           [:schema :active-paths])
        declared (with-meta 'field {:raster.type/tag 'double})]
    (is (= #{} (:active-paths (fact {:value declared :presence :absent}))))
    (is (= #{[0]} (:active-paths (fact {:value declared :presence :unknown}))))
    (is (= #{[0]} (:active-paths (fact {:value declared :presence :no-return}))))))

(deftest absence-join-refuses-unsupported-or-malformed-schema-relaxation
  (let [product #(hash-map :schema {:kind :product :fields [%]} :active-paths #{})
        absent {:kind :leaf :tag nil :presence :absent}]
    (is (= :ad-product-choice-schema
           (try (ports/join-facts (assoc (product absent) :active-paths #{[0]})
                                  (product {:kind :leaf :tag 'double})) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (doseq [[a b] [[{:kind :leaf :tag nil} {:kind :leaf :tag 'double}]
                   [absent {:kind :leaf :tag 'doubles}]
                   [{:kind :leaf :tag 'float} {:kind :leaf :tag 'double}]
                   [(assoc absent :tag 'float) {:kind :leaf :tag 'double}]
                   [(assoc absent :unexpected true) {:kind :leaf :tag 'double}]
                   [absent {:kind :leaf :tag 'double :presence :present}]
                   [absent {:kind :product :fields []}]]]
      (is (= :ad-product-choice-schema
             (try (ports/join-facts (product a) (product b)) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))
