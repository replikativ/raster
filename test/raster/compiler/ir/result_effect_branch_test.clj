(ns raster.compiler.ir.result-effect-branch-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.backend.jvm.segop-simd :as jvm]
            [raster.compiler.passes.parallel.soac-lower :as lower]))

(def result-specs [{:result 'selected :dtype :float}])

(def canonical-branch
  (dialect/effect-branch
   result-specs '(= i 0)
   (dialect/result-effect-region
    [(dialect/local-value 'then-value :float 1.0)] [] ['then-value])
   (dialect/result-effect-region
    [(dialect/local-value 'else-value :float 2.0)] [] ['else-value])))

(defn- program
  [branch]
  (let [tensor (av/tensor {:dtype :float :shape '[n]})
        effects #{:memory/write}
        facts (dialect/default-program-facts
               {:values {'out tensor 'result tensor
                         'n (av/tensor {:dtype :long :shape []})}
                :inputs '[n]
                :effects effects
                :equations
                {'e (assoc (dialect/default-equation-facts)
                           :effects effects
                           :aliases {'result 'out}
                           :attributes
                           {:result-storage [{:destination 'out :access :write
                                              :host-return :effect}]})}})]
    (list 'soac-program facts
          [(list '= 'e '[result]
                 (list 'effect-map {:index 'i :extent 'n :dtypes [:float]
                                    :iteration-order :independent}
                       [] [] '[out]
                       (dialect/effect-lambda-form
                        '[dst]
                        [branch '(effect dst :unique i true selected)])))]
          [])))

(deftest canonical-result-branch-projects-without-recovering-source
  (let [parts (dialect/effect-parts canonical-branch)
        scheduled (dialect/scheduled-effect canonical-branch)]
    (is (= {:branch true
            :predicate '(= i 0)
            :results result-specs
            :then {:locals [{:id 'then-value :dtype :float :init 1.0}]
                   :effects [] :yields ['then-value]}
            :else {:locals [{:id 'else-value :dtype :float :init 2.0}]
                   :effects [] :yields ['else-value]}}
           parts))
    (is (= {:branch {:predicate '(= i 0)
                     :results result-specs
                     :then {:locals [{:id 'then-value :dtype :float :init 1.0}]
                            :effects [] :yields ['then-value]}
                     :else {:locals [{:id 'else-value :dtype :float :init 2.0}]
                            :effects [] :yields ['else-value]}}}
           scheduled))
    (is (dialect/strict-effect-scalar-policy? [scheduled]))
    (is (dialect/scheduled-effect-carries? [scheduled]))
    (is (empty? (dialect/effect-leaves [scheduled])))
    (is (empty? (dialect/effect-part-leaves [parts])))
    (is (= (program canonical-branch)
           (dialect/validate! (program canonical-branch))))
    (let [remapped (dialect/remap-values
                    (program canonical-branch)
                    {'n :length 'out :output 'result :logical-result})
          lambda (:lambda (dialect/operation-parts (first (dialect/equations remapped))))]
      (is (= canonical-branch (first (:body-results (dialect/lambda-parts lambda))))
          "stable value remapping leaves branch-local identities lexical")
      (is (= :length (dialect/operation-extent (first (dialect/equations remapped))))))))

(deftest branch-arms-have-independent-lexical-effect-scopes
  (let [atomic {:destination 'counts :destination-index 'i :predicate true :value 1
                :result 'ticket :result-dtype :int}
        branch {:branch {:predicate 'pred
                         :results [{:result 'chosen :dtype :int}]
                         :then {:locals [] :effects [atomic] :yields ['ticket]}
                         :else {:locals [{:id 'fallback :dtype :int :init 0}]
                                :effects [] :yields ['fallback]}}}
        continuation {:destination 'out :destination-index 'chosen
                      :predicate true :value 'i}]
    (is (= [branch continuation]
           (dialect/validate-scheduled-effect-carries!
            [branch continuation] '#{counts out i pred})))
    (is (= '[counts]
           (mapv :destination (dialect/effect-leaves [branch]))))
    (testing "arm-local effect results do not escape the branch"
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries!
                    [branch (assoc continuation :destination-index 'ticket)]
                    '#{counts out i pred}))))
    (testing "one arm cannot consume the other arm's local result"
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries!
                    [(assoc-in branch [:branch :else :yields] ['ticket])]
                    '#{counts out i pred}))))))

(deftest malformed-result-branches-are-rejected
  (let [scheduled (dialect/scheduled-effect canonical-branch)]
    (testing "the predicate sees only the entry scope"
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries!
                    [(assoc-in scheduled [:branch :predicate] 'selected)] '#{i}))))
    (testing "result binders are fresh at the merge point"
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries!
                    [(assoc-in scheduled [:branch :results 0 :result] 'i)] '#{i}))))
    (testing "both arms yield the complete merge tuple"
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries!
                    [(assoc-in scheduled [:branch :then :yields] [])] '#{i}))))
    (testing "yields cannot hide writes"
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries!
                    [(assoc-in scheduled [:branch :then :yields]
                               ['(clojure.core/aset out i 1.0)])]
                    '#{i out}))))
    (testing "predicates and yields cannot hide canonical effect terms"
      (doseq [candidate [(assoc-in scheduled [:branch :predicate]
                                   '(effect out :unique i true 1.0))
                         (assoc-in scheduled [:branch :then :yields]
                                   ['(effect out :unique i true 1.0)])]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (dialect/validate-scheduled-effect-carries! [candidate] '#{i out})))))
    (testing "known lexical yield dtypes agree with the declared merge result"
      (is (thrown? clojure.lang.ExceptionInfo
                   (dialect/validate-scheduled-effect-carries!
                    [(assoc-in scheduled [:branch :results 0 :dtype] :double)] '#{i}))))
    (testing "canonical result declarations are nonempty, typed and distinct"
      (doseq [attributes [{:results []}
                          {:results [{:result 'x :dtype :float}] :extra true}
                          {:results [{:result 'x :dtype :half}]}
                          {:results [{:result 'x :dtype :float}
                                     {:result 'x :dtype :float}]}]]
        (is (not (dialect/effect-form?
                     (list 'effect-if attributes true
                           '(effect-region [] [] [0.0])
                           '(effect-region [] [] [0.0])))))))))

(deftest lexical-result-reuse-lowers-to-unique-target-identities
  (doseq [kind [:atomic :loop :branch :plain-loop]
          result-name ['ticket 'selected]]
    (let [arm (fn [value]
                (dialect/result-effect-region
                 []
                 [(case kind
                    :atomic (list 'effect 'scratch (dialect/reducing-scatter-conflict '+ :float)
                                  'i true value {:result result-name :dtype :float})
                    :loop (list 'effect-loop
                                {:index 'k :lower 0
                                 :carries [{:parameter 'acc :result result-name :dtype :float}]}
                                1 [value]
                                '(lambda [k acc]
                                   (effect-region [] [(effect scratch :unique i true acc)] [acc])))
                    :plain-loop '(effect-loop {:index k :lower 0} 1
                                   (lambda [k] (effect-region [] [(effect scratch :unique i true 1.0)])))
                    :branch (dialect/effect-branch
                             [{:result result-name :dtype :float}] true
                             (dialect/result-effect-region [] [(list 'effect 'scratch :unique 'i true value)] [value])
                             (dialect/result-effect-region [] [(list 'effect 'scratch :unique 'i true value)] [value])))]
                 [(if (= kind :plain-loop) value result-name)]))
          raw (program (dialect/effect-branch result-specs '(= i 0) (arm 1.0) (arm 2.0)))
          [eq id outputs [op attrs inputs captures destinations [_ _ region]]]
          (first (dialect/equations raw))
          tensor (av/tensor {:dtype :float :shape '[n]})
          facts (-> (dialect/facts raw)
                    (update :values assoc 'side tensor 'side-result tensor)
                    (assoc-in [:equations 'e :aliases 'side-result] 'side)
                    (update-in [:equations 'e :attributes :result-storage]
                               conj {:destination 'side :access :read-write :host-return :effect}))
          typed (dialect/make facts
                              [(list eq id (conj outputs 'side-result)
                                     (list op (assoc attrs :dtypes [:float :float]) inputs captures
                                           (conj destinations 'side)
                                           (list 'lambda '[dst scratch] region)))] [])
          operation (first (lower/lower-typed-effect-map typed :ze:0))
          execute (eval (list 'fn '[out side n] (jvm/compile-effect-segmap operation)))
          out (float-array [10 20])]
      (execute out (float-array [10 20]) 2)
      (is (= (if (= kind :atomic) [10.0 20.0] [1.0 2.0]) (vec out)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (is (= :kernel-body
               (get-in (emit/generate-scheduled-segmap-kernel
                        operation :dtype :float :target-dialect target
                        :array-types {'out :float 'side :float} :scalar-types {'n :long})
                       [:attributes :emission-route]))
            (str kind " " result-name " " target))))))
