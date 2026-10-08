(ns raster.compiler.ir.distributed-refinement-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.scan :as scan]))

(defn- direct-refinement [n]
  (let [devices (mapv #(keyword (str "worker-" %)) (range n))
        root (first devices)
        operation (distributed/collective-operation
                   {:id :all-reduce :kind :all-reduce :group :workers :value :gradient
                    :reduction (scan/certify-reassociation
                                {:acc 'acc :init 0.0 :lambda '(+ acc element)} :float)})
        inputs (into {} (map (fn [worker] [[worker :input] worker])) devices)
        gathered
        (reduce (fn [{:keys [nodes result]} worker]
                  (let [received [worker :received]
                        combined [worker :combined]]
                    {:nodes (into nodes
                                  [{:id received :kind :copy :input [worker :input]
                                    :target root :route [[worker root]]}
                                   {:id combined :kind :combine :left result :right received
                                    :device root}])
                     :result combined}))
                {:nodes [] :result [root :input]} (rest devices))
        outputs (into {root (:result gathered)}
                      (map (fn [worker] [worker [worker :output]])) (rest devices))
        nodes (into (:nodes gathered)
                    (map (fn [worker]
                           {:id [worker :output] :kind :copy :input (:result gathered)
                            :target worker :route [[root worker]]})) (rest devices))]
    (distributed/collective-refinement
     {:operation operation :group (distributed/collective-group :workers devices)
      :value (av/tensor {:dtype :float :shape [17]
                        :sharding {:kind :replicated :devices devices}})
      :numerical {:mode :reassociated :policy :direct-tree
                  :rounding :nearest-even :accumulator-dtype :float}
      :inputs inputs :nodes nodes :outputs outputs})))

(defn- reason [f]
  (:reason (ex-data (try (f) (catch clojure.lang.ExceptionInfo e e)))))

(deftest all-reduce-contribution-proofs-support-unequal-tree-sizes-and-fanout
  (doseq [n [2 3 4 7]]
    (let [refinement (direct-refinement n)
          facts (distributed/refinement-facts refinement)
          participants (set (get-in refinement [:group :devices]))
          ;; Independent scalar interpretation of the declared copy/combine nodes.
          initial (into {} (map-indexed (fn [i device] [[device :input] (inc i)])
                                       (get-in refinement [:group :devices])))
          results (reduce (fn [values {:keys [id kind input left right]}]
                            (assoc values id (case kind :copy (get values input)
                                                  :combine (+ (get values left) (get values right)))))
                          initial (:nodes refinement))]
      (is (= (* n (inc n) 1/2) (get results (get-in refinement [:outputs :worker-0]))))
      (doseq [[worker output] (:outputs refinement)]
        (is (= participants (get-in facts [:values output :contributors])))
        (is (= worker (get-in facts [:values output :device])))
        (is (= (* n (inc n) 1/2) (get results output))))
      (is (= [] (get-in facts [:dependencies [:worker-1 :received]])))
      (is (= [[:worker-1 :received]] (get-in facts [:dependencies [:worker-1 :combined]])))
      (when (> n 2)
        (is (= [[:worker-1 :combined] [:worker-2 :received]]
               (get-in facts [:dependencies [:worker-2 :combined]]))))
      (is (= #{(:producer (get-in facts [:outputs :worker-0]))}
             (set (get-in facts [:dependencies [:worker-1 :output]])))))))

(deftest refinement-uses-the-retained-monoid-not-a-sum-only-registry
  (let [refinement (assoc-in (direct-refinement 3) [:operation :reduction]
                             (scan/certify-reassociation
                              {:acc 'acc :init 1.0 :lambda '(* acc element)} :float))
        facts (distributed/refinement-facts refinement)
        expression (get-in facts [:outputs :worker-0 :expression])
        evaluate (fn evaluate [expr]
                   (if (= :input (first expr))
                     (get {[:worker-0 :input] 2 [:worker-1 :input] 3
                           [:worker-2 :input] 5} (second expr))
                     (do (is (= '* (first expr)))
                         (* (evaluate (second expr)) (evaluate (nth expr 2))))))]
    (is (= 30 (evaluate expression)))
    (is (apply = (map :expression (vals (:outputs facts)))))))

(deftest a-single-participant-needs-neither-a-copy-nor-a-combine
  (let [refinement (direct-refinement 1)
        facts (distributed/refinement-facts refinement)]
    (is (empty? (:nodes refinement)))
    (is (empty? (:dependencies facts)))
    (is (= {:worker-0 {:device :worker-0 :contributors #{:worker-0}
                       :expression [:input [:worker-0 :input]] :producer nil}}
           (:outputs facts)))))

(deftest refinement-rejects-incomplete-duplicated-and-ill-scoped-contributions
  (let [refinement (direct-refinement 3)]
    (doseq [[path replacement expected]
            [[[:nodes 1 :right] [:worker-0 :input] :distributed-refinement-combine]
             [[:nodes 1 :device] :worker-2 :distributed-refinement-combine]
             [[:nodes 0 :target] :worker-1 :distributed-refinement-copy]
             [[:nodes 0 :route] [] :distributed-refinement-copy]
             [[:nodes 0 :input] :missing :distributed-refinement-node]
             [[:nodes 0 :id] [:worker-0 :input] :distributed-refinement-node]
             [[:nodes 0 :unknown] true :distributed-refinement-node]
             [[:outputs :worker-0] [:worker-0 :input] :distributed-refinement-output]
             [[:outputs :worker-1] [:worker-2 :output] :distributed-refinement-output]
             [[:inputs [:worker-1 :input]] :worker-0 :distributed-refinement-boundary]]]
      (is (= expected (reason #(distributed/refinement-facts
                                (assoc-in refinement path replacement))))))
    (is (= :distributed-refinement-boundary
           (reason #(distributed/refinement-facts (update refinement :outputs dissoc :worker-2)))))
    (is (= :distributed-refinement-type
           (reason #(distributed/refinement-facts (assoc refinement :contributors :trusted)))))))

(deftest refinement-revalidates-monoid-value-and-numerical-declarations
  (let [refinement (direct-refinement 2)]
    (doseq [[path replacement expected]
            [[[:numerical] {} :distributed-refinement-numerical]
             [[:numerical] {:mode :exact :policy :claimed-bitwise} :distributed-refinement-numerical]
             [[:numerical :accumulator-dtype] :double :distributed-refinement-numerical]
             [[:value :dtype] :double :distributed-refinement-value]
             [[:value :representation] {:kind :quantized} :distributed-refinement-value]
             [[:value :shape] ['n] :distributed-refinement-value]
             [[:value :sharding :devices] [:worker-0] :distributed-refinement-value]
             [[:operation :kind] :all-gather :distributed-collective-reduction]]]
      (is (= expected (reason #(distributed/refinement-facts
                                (assoc-in refinement path replacement))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (distributed/refinement-facts
                  (assoc-in refinement [:operation :reduction :combine] '*))))))
