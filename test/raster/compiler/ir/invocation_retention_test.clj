(ns raster.compiler.ir.invocation-retention-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.invocation-link :as invocation]))

(deftest invocation-boundary-inspection-is-shared-only-within-one-lowering
  ;; Isolated scan mechanics. Public equation-first integration tests still certify
  ;; real graphs and materialized invocations; these maps are not proof artifacts.
  (let [read-operation {:id :read}
        write-operation {:id :write}
        graphs {:read {:inputs [{:id 'A :role :input}]}
                :write {:outputs [{:id 'A :role :output} {:id 'B :role :output}]}}
        program {:equations [{:operations [read-operation]}
                             {:operations []}
                             {:operations [write-operation]}]
                 :values {'A {:shape [4]} 'B {:shape [4]}}
                 :outputs ['B]}
        inspected (atom [])
        materialized {:program-buffers {'A {:id :a :shape [4]}
                                       'B {:id :b :shape [4]}}}]
    (with-redefs [equation-dispatch/boundary-graph
                  (fn [operation]
                    (swap! inspected conj (:id operation))
                    (get graphs (:id operation)))
                  invocation/complete-write? (fn [& _] true)]
      (let [boundaries (#'invocation/equation-boundaries program)]
        (is (= [:read :write] @inspected))
        (is (identical? read-operation (:operation (first boundaries))))
        (is (= #{'A 'B} (#'invocation/required-materialized-buffers program boundaries)))
        (is (= #{'B} (#'invocation/write-before-read-inputs boundaries materialized {}))
            "first-read order must not turn A into an overwrite input")
        (is (= [:read :write] @inspected) "buffer scans must not reconstruct boundaries"))
      (#'invocation/equation-boundaries program)
      (is (= [:read :write :read :write] @inspected)
          "a subsequent lowering must inspect fresh boundaries"))))

(defn- fixture []
  {:plan {:instances [{:id :program}] :outputs []}
   :certificate
   {:effect-evidence
    {:step-facts (mapv (fn [step]
                         {:step step :phase (nth [:produce :replace] step)
                          :facts [{:node :buffer :access :write}]
                          :complete-writes #{:buffer}})
                       [0 1])}}
   :memory {:values {:storage {:leaves [{:node :buffer}]}}
            :nodes {:buffer {:value :storage :allocation :allocation :role :internal}}}
   :emitted {:inputs [] :outputs []
             :equations [{:id :produce} {:id :replace} {:id :consume}]}
   :bindings {'a :storage 'b :storage 'physical-destination :storage}
   :definitions {'a {:kind :equation :id :produce}
                 'b {:kind :equation :id :replace}}
   :uses {'a [:replace] 'b [:consume]}})

(defn- witness [{:keys [plan certificate memory emitted bindings definitions uses storage-references]
                 :or {storage-references #{}}}]
  ;; Pure report logic only. Public integration tests independently exercise verified invocation
  ;; construction, real complete-write evidence, output escapes, and resident AD replay.
  (let [classification (#'invocation/compiler-value-classification
                        emitted bindings definitions storage-references)]
    (#'invocation/value-retention-witness
     plan certificate memory emitted bindings classification uses)))

(deftest semantic-retention-is-not-storage-permission-or-completion
  (let [base (fixture)
        report (witness base)]
    (is (= :witnessed (:status report)))
    (is (= ['a 'b] (get-in report [:storage :storage :versions])))
    (is (not (contains? (:values report) 'physical-destination)))
    (is (= :unproven (:completion report)))
    (is (= {:equation-index 1 :equation :replace :phase :read}
           (get-in report [:values 'a :retained-through])))
    (doseq [[reason changed]
            [[:overlapping-value-versions (assoc-in base [:uses 'a] [:consume])]
             [:overlapping-value-versions
              (assoc-in base [:definitions 'b :id] :produce)]
             [:escaped-old-version (assoc-in base [:emitted :outputs] ['a])]
             [:ambiguous-physical-escape (assoc-in base [:plan :outputs] [:buffer])]
             [:multi-view-allocation
              (assoc-in base [:memory :nodes :alias]
                        {:allocation :allocation :value :other :role :internal})]
             [:partial-write
              (assoc-in base [:certificate :effect-evidence :step-facts 1 :complete-writes] #{})]
             [:read-write-mutation
              (assoc-in base [:certificate :effect-evidence :step-facts 1 :facts 0 :access]
                        :read-write)]
             [:read-write-mutation
              (assoc-in base [:certificate :effect-evidence :step-facts 1 :facts]
                        [{:node :buffer :symbol 'a :access :read}
                         {:node :buffer :symbol 'b :access :write}])]
             [:unattributed-mutation
              (update-in base [:certificate :effect-evidence :step-facts]
                         conj {:step 2 :phase :consume
                               :facts [{:node :buffer :access :write}]
                               :complete-writes #{:buffer}})]]]
      (testing (str reason)
        (let [report (witness changed)]
          (is (= :unknown (:status report)))
          (is (some #(= reason (:reason %)) (:unknown report)))
          (is (= :unknown (get-in report [:storage :storage :status]))))))
    (let [report (witness (update-in base [:plan :instances] conj {:id :other}))]
      (is (= :unknown (:status report)))
      (is (some #(= :multiple-program-instances (:reason %)) (:unknown report)))
      (is (= :unproven (:completion report))))
    (let [report (witness
                  (assoc-in base [:emitted :equations 0 :operations]
                            [(emitted-loop/map->EmittedStructuredLoop {})]))]
      (is (= :unknown (:status report)))
      (is (some #(= :structured-replay (:reason %)) (:unknown report))))))

(deftest exact-write-destinations-are-storage-references-not-entry-versions
  (let [base {:plan {:instances [{:id :program}] :outputs [:buffer]}
              :certificate {:effect-evidence
                            {:step-facts [{:step 0 :phase :produce
                                           :facts [{:node :buffer :symbol 'result :access :write}]
                                           :complete-writes #{:buffer}}]}}
              :memory {:values {:storage {:leaves [{:node :buffer}]}}
                       :nodes {:buffer {:value :storage :allocation :allocation :role :output}}}
              :emitted {:inputs '[destination] :outputs '[destination]
                        :equations [{:id :produce :operands [] :results '[result]}]}
              :bindings {'destination :storage 'result :storage}
              :definitions {'result {:kind :equation :id :produce}}
              :uses {}}
        boundaries [{:buffer-accesses {'destination :write}}]
        relation {:boundary-index 0 :equation :produce :result 'result
                  :destination 'destination :access :write
                  :host-return :buffer :host-read? false
                  :physical-destination 'destination}
        references (fn [fixture relations boundaries]
                     (#'invocation/storage-reference-values-from-relations
                      (:emitted fixture) (:bindings fixture) (:definitions fixture) (:uses fixture)
                      boundaries relations))
        storage-references (references base [relation] boundaries)
        classification (#'invocation/compiler-value-classification
                        (:emitted base) (:bindings base) (:definitions base) storage-references)
        report (witness (assoc base :storage-references storage-references))]
    (is (= #{'destination} storage-references))
    (is (= {:definition {:kind :storage-only} :semantic-output? false}
           (get classification 'destination)))
    (is (= ['result] (get-in report [:storage :storage :versions])))
    (is (not (contains? (:values report) 'destination)))
    (is (= :witnessed (:status report)))
    (doseq [[label changed relations changed-boundaries]
            [[:missing-producer (assoc base :definitions {}) [relation] boundaries]
             [:not-a-public-physical-output (assoc-in base [:emitted :outputs] [])
              [relation] boundaries]
             [:independent-producer
              (assoc-in base [:definitions 'destination] {:kind :equation :id :prior})
              [relation] boundaries]
             [:semantic-use (assoc base :uses {'destination [:observe]}) [relation] boundaries]
             [:outer-host-read (assoc-in base [:emitted :attributes :host-read-values]
                                         ['destination])
              [relation] boundaries]
             [:host-read base [(assoc relation :host-read? true)] boundaries]
             [:effect-return base [(assoc relation :host-return :effect)] boundaries]
             [:read-write-contract base [(assoc relation :access :read-write)]
              [{:buffer-accesses {'destination :read-write}}]]
             [:read-write-graph base [relation]
              [{:buffer-accesses {'destination :read-write}}]]
             [:missing-storage base [] boundaries]
             [:mismatched-physical base [(assoc relation :physical-destination 'other)] boundaries]
             [:different-binding (assoc-in base [:bindings 'result] :other-storage)
              [relation] boundaries]
             [:additional-write-without-storage-relation base [relation]
              (conj boundaries {:buffer-accesses {'destination :write}})]]]
      (testing (name label)
        (is (empty? (references changed relations changed-boundaries)))))
    (testing "a physical input/output without storage authority remains a semantic entry escape"
      (is (= {:definition {:kind :program-input} :semantic-output? true}
             (get (#'invocation/compiler-value-classification
                   (:emitted base) (:bindings base) (:definitions base) #{})
                  'destination))))))

(deftest storage-reference-classification-does-not-grant-write-completion
  (let [base (-> (fixture)
                 (assoc :emitted {:inputs '[physical-destination]
                                  :outputs '[physical-destination]
                                  :equations [{:id :produce} {:id :replace} {:id :consume}]}
                        :storage-references #{'physical-destination})
                 (assoc-in [:certificate :effect-evidence :step-facts 0 :complete-writes] #{}))
        report (witness base)]
    (is (= :unknown (:status report)))
    (is (some #(= :partial-write (:reason %)) (:unknown report)))
    (is (not (contains? (:values report) 'physical-destination)))
    (is (= ['a 'b] (get-in report [:storage :storage :versions])))))
