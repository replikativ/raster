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

(deftest refinement-preserves-the-shared-multi-participant-group-contract
  (is (= :distributed-collective-group (reason #(direct-refinement 1)))))

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

(defn- projection-inputs [n]
  (let [refinement (direct-refinement n)
        devices (get-in refinement [:group :devices])
        links (distinct (mapcat :route (filter #(= :copy (:kind %)) (:nodes refinement))))]
    [refinement
     (distributed/topology
      (mapv #(distributed/device {:id % :memory-capacity-bytes 1048576}) devices)
      (mapv (fn [[source target :as id]]
              (distributed/link {:id id :source source :target target
                                 :bandwidth-bytes-s 1.0e9 :latency-ns 1})) links))
     (into {} (map (fn [[id worker]]
                     [id (distributed/compute-step {:id [worker :produce] :device worker
                                                    :duration-ns 1})])) (:inputs refinement))
     (into {} (keep #(when (= :combine (:kind %))
                      [(:id %) {:duration-ns 2 :peak-memory-bytes 204}])) (:nodes refinement))]))

(deftest topology-projection-preserves-arithmetic-dependencies-and-broadcast-completion
  (doseq [n [2 3 4 7]]
    (let [[refinement cluster inputs costs] (projection-inputs n)
          projected (distributed/project-refinement refinement cluster inputs costs)
          first-copy (first (:steps projected))
          final-combine [(keyword (str "worker-" (dec n))) :combined]
          plan (distributed/plan
                {:id :projected-tree
                 :mesh (distributed/mesh [{:name :workers :size n}]
                                         (get-in refinement [:group :devices]))
                 :topology cluster :values (:values projected) :shards (:shards projected)
                 :steps (into (mapv inputs (keys (:inputs refinement))) (:steps projected))
                 :outputs (:completions projected)})]
      (is (= [[:worker-1 :produce]] (:dependencies first-copy)))
      (is (= 68 (:bytes first-copy)))
      (is (= :compute (:kind (second (:steps projected)))))
      (is (= [[:worker-0 :produce] [:worker-1 :received]]
             (:dependencies (second (:steps projected)))))
      (is (= final-combine (first (:completions projected))))
      (is (= [final-combine] (:dependencies (last (:steps projected)))))
      (is (= (set (keys (:values (distributed/refinement-facts refinement))))
             (set (keys (:values projected)))))
      (is (= plan (distributed/validate! plan)))
      (is (pos? (:makespan-ns (distributed/simulate plan)))))))

(deftest projection-rejects-route-cost-and-producer-forgeries
  (let [[refinement cluster inputs costs] (projection-inputs 3)
        project #(distributed/project-refinement %1 %2 %3 %4)]
    (is (= :distributed-refinement-input-producers
           (reason #(project refinement cluster (dissoc inputs [:worker-1 :input]) costs))))
    (is (= :distributed-refinement-input-producers
           (reason #(project refinement cluster
                             (assoc-in inputs [[:worker-1 :input] :device] :worker-0) costs))))
    (is (= :distributed-refinement-step-identities
           (reason #(project refinement cluster
                             (assoc-in inputs [[:worker-1 :input] :id] [:worker-1 :received]) costs))))
    (is (= :distributed-refinement-combine-costs
           (reason #(project refinement cluster inputs {}))))
    (is (= :distributed-refinement-combine-costs
           (reason #(project refinement cluster inputs
                             (assoc-in costs [[:worker-1 :combined] :duration-ns] 0)))))
    (is (= :distributed-refinement-bytes
           (reason #(project (assoc-in refinement [:value :shape]
                                      [Integer/MAX_VALUE Integer/MAX_VALUE])
                             cluster inputs costs))))
    (is (= :distributed-transfer-link
           (reason #(project (assoc-in refinement [:nodes 0 :route] [:missing]) cluster inputs costs))))
    (is (= :distributed-transfer-continuity
           (reason #(project (assoc-in refinement [:nodes 0 :route] [[:worker-0 :worker-1]])
                             cluster inputs costs))))))
