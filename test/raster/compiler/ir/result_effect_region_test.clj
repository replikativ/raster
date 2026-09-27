(ns raster.compiler.ir.result-effect-region-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn- result-region []
  (dialect/effect-result-region
   [{:result 'chosen :dtype :int} {:result 'other :dtype :int}]
   [(dialect/local-value 'local :int 'seed)]
   ['(effect out :unique i true local)]
   ['local 'seed]))

(defn- spine []
  (list 'effect-region []
        [(result-region) '(effect out :unique chosen true other)]))

(deftest result-region-retains-typed-yields-and-effect-inventory
  (let [scheduled (dialect/scheduled-effect (result-region))]
    (is (= [{:result 'chosen :dtype :int} {:result 'other :dtype :int}]
           (get-in scheduled [:region :results])))
    (is (= ['local 'seed] (get-in scheduled [:region :yields])))
    (is (= [scheduled]
           (dialect/validate-scheduled-effect-carries! [scheduled] '#{out i seed})))
    (is (= ['out] (mapv :destination (dialect/effect-leaves [scheduled]))))
    (is (dialect/strict-effect-scalar-policy? [scheduled]))
    (is (dialect/scheduled-effect-carries? [scheduled]))))

(deftest result-regions-share-sequential-scope-authority
  (let [r (spine) scope (form/scope-info r)
        renamed (util/alpha-convert r)
        substituted (util/subst-syms {'seed 'chosen} r)]
    (is (= ['chosen 'other nil] (get-in scope [:scopes 0 :binders])))
    (is (= '#{out i seed} (util/free-syms r)))
    (is (= r ((:rebuild scope) (:scopes scope) (:outer scope))))
    (is (= (util/free-syms r) (util/free-syms renamed)))
    (is (= (util/alpha-normalize r) (util/alpha-normalize renamed)))
    (is (= '#{out i chosen} (util/free-syms substituted)))
    (let [[region store] (nth substituted 2)
          results (:results (second region))]
      (is (not= 'chosen (:result (first results))))
      (is (= (:result (first results)) (nth store 3)))
      (is (= (:result (second results)) (nth store 5))))))

(deftest result-region-rejects-missing-ambiguous-and-escaping-values
  (let [scheduled (dialect/scheduled-effect (result-region))
        continuation {:destination 'out :destination-index 'chosen
                      :predicate true :value 'other}]
    (is (= [scheduled continuation]
           (dialect/validate-scheduled-effect-carries!
            [scheduled continuation] '#{out i seed})))
    (doseq [bad [(assoc-in scheduled [:region :yields] ['local])
                 (assoc-in scheduled [:region :results] [])
                 (assoc-in scheduled [:region :results 1 :result] 'chosen)
                 (assoc-in scheduled [:region :results 0 :result] 'seed)
                 (assoc-in scheduled [:region :results 0 :dtype] :double)
                 (assoc-in scheduled [:region :predicate] true)
                 (assoc-in scheduled [:region :yields] ['chosen 'seed])
                 (assoc-in scheduled [:region :locals 0 :init] 'chosen)
                 (assoc-in scheduled [:region :yields]
                           ['(effect out :unique i true 1) 'seed])]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries! [bad] '#{out i seed}))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (dialect/validate-scheduled-effect-carries!
                  [scheduled (assoc continuation :value 'local)] '#{out i seed})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (dialect/validate-scheduled-effect-carries!
                  [continuation scheduled] '#{out i seed})))))

(deftest malformed-canonical-result-regions-do-not-become-valid-by-rebinding
  (let [bad '(effect-region {:results [{:result chosen :dtype :int}
                                       {:result other :dtype :int}]}
                           [] [] [seed])]
    (is (not (dialect/effect-form? bad)))
    (doseq [transformed [(util/alpha-convert bad)
                         (util/subst-syms {'seed 'chosen} bad)]]
      (is (= 2 (count (:results (second transformed)))))
      (is (= 1 (count (last transformed))))
      (is (not (dialect/effect-form? transformed))))))
