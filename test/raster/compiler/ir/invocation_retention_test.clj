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

(defn- witness [{:keys [plan certificate memory emitted bindings definitions uses]}]
  ;; Pure report logic only. Public integration tests independently exercise verified invocation
  ;; construction, real complete-write evidence, output escapes, and resident AD replay.
  (#'invocation/value-retention-witness
   plan certificate memory emitted bindings definitions uses))

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
