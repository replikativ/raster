(ns raster.compiler.ir.nested-effect-region-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- program [region access effects]
  (let [tensor (av/tensor {:dtype :float :shape '[n]})
        facts (dialect/default-program-facts
               {:values {'out tensor 'result tensor
                         'n (av/tensor {:dtype :long :shape []})}
                :inputs '[n]
                :effects effects
                :equations {'e (assoc (dialect/default-equation-facts)
                                      :effects effects :aliases {'result 'out}
                                      :attributes {:result-storage
                                                   [{:destination 'out :access access
                                                     :host-return :effect}]})}})]
    (dialect/make
     facts
     [(list '= 'e '[result]
            (list 'effect-map {:index 'i :extent 'n :dtypes [:float]
                               :iteration-order :independent}
                  [] [] '[out] (dialect/effect-lambda-form '[dst] [region])))]
     [])))

(deftest canonical-nested-regions-validate-storage-and-local-contracts
  (let [region '(effect-region [(let-value v :float (clojure.core/aget dst i))]
                              [(effect dst :unique i 1 v)])
        p (program region :read-write #{:memory/read :memory/write})]
    (is (= p (dialect/validate! p)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (program region :write #{:memory/read :memory/write})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (program region :read-write #{:memory/write})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (program
                  (list 'effect-loop {:index 'k :lower 0} 2
                        (dialect/effect-lambda-form '[k] [region]))
                  :write #{:memory/read :memory/write})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (program '(effect-region [(let-value v :unknown 1)]
                                          [(effect dst :unique i 1 v)])
                          :write #{:memory/write})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (program
                  '(effect-loop {:index k :lower 0} 2
                     (lambda [k]
                       (effect-region []
                         [(effect-region []
                            [(effect-loop {:index j :lower 0} 2
                               (lambda [j]
                                 (effect-region [] [(effect dst :unique i 1 0.0)])))])])))
                  :write #{:memory/write})))))

(def nested
  '(effect-region [(let-value inv :double (/ 1.0 sum))]
                  [(effect out :unique i 1 inv)]))

(deftest production-continuation-gate-descends-through-ordinary-loops
  (let [region '(effect-region [] [(effect dst :unique i 1 1.0)])
        p (program (list 'effect-loop {:index 'k :lower 0} 2
                         (dialect/effect-lambda-form '[k] [region]))
                   :write #{:memory/write})
        sequential (walk/postwalk #(if (and (map? %) (contains? % :iteration-order))
                                     (assoc % :iteration-order :sequential) %) p)]
    (is (= sequential (dialect/validate! sequential)))
    (is (nil? (#'route/sequential-continuation-decline p)))
    (is (= :sequential-effect-continuation
           (:reason (#'route/sequential-continuation-decline sequential))))))

(deftest nested-regions-project-and-retain-lexical-scope
  (let [scheduled (dialect/scheduled-effect nested)]
    (is (= {:region {:locals [{:id 'inv :dtype :double :init '(/ 1.0 sum)}]
                       :effects [{:destination 'out :conflict :unique
                                  :destination-index 'i :predicate 1 :value 'inv}]}}
           scheduled))
    (is (= [scheduled]
           (dialect/validate-scheduled-effect-carries! [scheduled] '#{sum out i})))
    (is (= '[out] (mapv :destination (dialect/effect-leaves [scheduled]))))
    (is (= '[out] (mapv :destination
                       (dialect/effect-part-leaves [(dialect/effect-parts nested)]))))
    (is (= '#{sum out i} (util/free-syms nested)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (dialect/validate-scheduled-effect-carries!
                  [scheduled {:destination 'out :destination-index 'i :predicate 1 :value 'inv}]
                  '#{sum out i}))))
  (doseq [locals [[{:id 'v :dtype :double :init 'later}
                   {:id 'later :dtype :double :init 1.0}]
                  [{:id 'sum :dtype :double :init 1.0}]
                  [{:id 'v :dtype :unknown :init 1.0}]
                  [{:id 'v :dtype :double :init '(clojure.core/aset out i 1.0)}]]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (dialect/validate-scheduled-effect-carries!
                  [{:region {:locals locals :effects []}}] '#{sum out i})))))

(deftest regions-cannot-hide-loops-inside-carried-bodies
  (let [inner {:loop {:index 'j :lower 0 :extent 2 :locals []
                      :effects [{:destination 'out :destination-index 'j
                                 :predicate 1 :value 0.0}]}}
        carried {:loop {:index 'k :lower 0 :extent 2 :locals []
                        :carry {:parameter 'acc :result 'sum :dtype :double
                                :init 0.0 :update 'acc}
                        :effects [{:region {:locals [] :effects [inner]}}]}}]
    (is (dialect/scheduled-effect-carries?
         [{:region {:locals [] :effects [carried]}}]))
    (is (thrown? clojure.lang.ExceptionInfo
                 (dialect/validate-scheduled-effect-carries! [carried] '#{out})))))
