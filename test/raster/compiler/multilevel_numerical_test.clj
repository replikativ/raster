(ns raster.compiler.multilevel-numerical-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ode.multilevel :as multilevel]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as gpu-distributed]
            [raster.gpu.link :as link]))

(deftm transfer-cycle!
  [out :- (Array double), fine :- (Array double), coarse :- (Array double),
   nx :- Long, ny :- Long]
  (multilevel/prolong-constant-2d! fine coarse nx ny)
  (multilevel/restrict-average-2d! out fine nx ny))

(deftest cell-centred-transfer-oracles
  (doseq [[nx ny] [[1 1] [3 5] [4 2]]]
    (let [coarse (double-array (map #(- (* 0.25 %) 2.0) (range (* nx ny))))
          fine (double-array (* 4 nx ny))
          restored (double-array (* nx ny))]
      (multilevel/prolong-constant-2d! fine coarse nx ny)
      (let [expected (vec (for [i (range (* 2 nx)) j (range (* 2 ny))]
                            (aget coarse (+ (* (quot i 2) ny) (quot j 2)))))]
        (is (= expected (vec fine))))
      (multilevel/restrict-average-2d! restored fine nx ny)
      (is (= (vec coarse) (vec restored)))
      ;; Non-piecewise-constant fine data tests averaging independently of prolongation.
      (let [fine (double-array (map #(* 0.125 %) (range (* 4 nx ny))))]
        (multilevel/restrict-average-2d! restored fine nx ny)
        (is (= (* 0.25 (reduce + (vec fine))) (reduce + (vec restored))))))
    (let [constant (double-array (repeat (* 4 nx ny) 3.25))
          coarse (double-array (* nx ny))]
      (multilevel/restrict-average-2d! coarse constant nx ny)
      (is (every? #(= 3.25 %) coarse)))))

(deftest generated-coarse-fine-transfers-match-reference
  (doseq [operator [#'multilevel/prolong-constant-2d! #'multilevel/restrict-average-2d!]]
    (let [prolong? (= operator #'multilevel/prolong-constant-2d!)
          source (double-array (map #(- (* 0.125 %) 1.0) (range (if prolong? 15 60))))
          destination (double-array (repeat (if prolong? 60 15) -99.0))
          expected (aclone destination)
          _ (operator expected source 3 5)
          compiled (equation/compile operator {:target :ze:0 :dtype :double})
          plan (equation/lower compiled [destination source 3 5])]
      (is (seq (:outputs plan)))
      (if-not @gp/gpu-available?
        (gp/gpu-skip! "generated-coarse-fine-transfer")
        (with-open [executable (link/instantiate! plan)]
          (link/run! executable)
          (is (= (vec expected) (vec (link/download executable (first (:outputs plan)))))))))))

(deftest composed-transfer-cycle-keeps-intermediate-resident
  (let [coarse (double-array (map #(- (* 0.125 %) 1.0) (range 15)))
        fine (double-array 60) out (double-array 15)
        compiled (equation/compile #'transfer-cycle! {:target :ze:0 :dtype :double})
        plan (equation/lower compiled [out fine coarse 3 5])]
    (is (= 3 (count (:nodes plan))) "only the three caller buffers are materialized")
    (if-not @gp/gpu-available?
      (gp/gpu-skip! "generated-coarse-fine-cycle")
      (with-open [executable (link/instantiate! plan)]
        (link/run! executable)
        (let [actual (link/download executable (first (:outputs plan)))]
          (is (= (vec coarse) (vec actual))))))))

(defn- scheduled-transfer-cycle [coarse fine]
  (let [shapes {:coarse [3 5] :fine [6 10]}
        fields {coarse :coarse fine :fine}
        hierarchy (amr/hierarchy
                   {:id :transfer-cycle :base-shape [3 5] :proper-nesting-width 0
                    :levels (mapv (fn [[index field ratio]]
                                    (amr/level
                                     {:id index :index index :ratio-to-parent ratio
                                      :patches [(amr/patch {:id field :level index :device :ze:0
                                                           :offsets [0 0] :shape (shapes field)
                                                           :field field})]}))
                                  [[0 :coarse nil] [1 :fine [2 2]]])})
        values (update-vals shapes #(av/tensor {:dtype :double :shape %
                                               :sharding {:kind :partitioned :axis 0 :devices [:ze:0]}}))
        scheduled (reduce (fn [result [kind source target method]]
                            (conj result
                                  (amr/schedule-coarse-fine
                                   hierarchy values
                                   (amr/coarse-fine-operation
                                    {:id kind :kind kind :source-patch source :target-patch target
                                     :source-region {:offsets [0 0] :shape (shapes source)}
                                     :target-region {:offsets [0 0] :shape (shapes target)}
                                     :operator {:method method :required-invariants #{:constant-preserving}}})
                                   {:duration-ns 1 :dependencies (if (seq result)
                                                                 [(:completion (peek result))] [])})))
                          [] [[:prolongation :coarse :fine :piecewise-constant]
                              [:restriction :fine :coarse :cell-average]])
        entries (into {}
                      (for [[kind operator args]
                            [[:prolongation #'multilevel/prolong-constant-2d! [fine coarse 3 5]]
                             [:restriction #'multilevel/restrict-average-2d! [coarse fine 3 5]]]
                            :let [plan (equation/lower (equation/compile operator {:target :ze:0 :dtype :double}) args)
                                  plan (update plan :nodes
                                               #(update-vals % (fn [node]
                                                                  (assoc-in node [:view :allocation :id]
                                                                            (fields (:source node))))))]]
                        [kind {:link-plan plan}]))
        calls (into {}
                    (for [scheduled scheduled
                          :let [kind (get-in scheduled [:operation :kind])
                                plan (get-in entries [kind :link-plan])]]
                      [(:completion scheduled)
                       {:entry kind
                        :bindings (into {}
                                        (for [[id value] (:values plan)
                                              :let [node (get-in value [:leaves 0 :node])
                                                    field (fields (get-in plan [:nodes node :source]))]
                                              :when field]
                                          [id {:local-shape (shapes field)
                                               :placements [{:kind :owned :value field :shard field
                                                             :local-offsets [0 0]}]}]))}]))]
    (distributed/plan
     {:id :generated-multilevel-cycle
      :mesh (distributed/mesh [{:name :worker :size 1}] [:ze:0])
      :topology (distributed/topology [(distributed/device {:id :ze:0 :memory-capacity-bytes 1048576})] [])
      :values values
      :shards (into {} (for [[field shape] shapes]
                        [field [(distributed/shard {:id field :value field :device :ze:0
                                                    :offsets [0 0] :shape shape})]]))
      :device-plans {:ze:0 {:entries entries :steps calls}}
      :steps (vec (mapcat :steps scheduled)) :outputs [(:completion (peek scheduled))]})))

(deftest coarse-fine-schedule-executes-its-generated-local-entries
  (let [coarse (double-array (map #(* 0.25 %) (range 15)))
        fine (double-array 60)
        plan (scheduled-transfer-cycle coarse fine)]
    (is (= 2 (count (:actions (distributed/check-readiness plan)))))
    (is (= :distributed-readiness-race
           (try (distributed/check-readiness (assoc-in plan [:steps 1 :dependencies] []))
                nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
        "the transfer cycle cannot observe a fine field without its producer dependency")
    (if-not @gp/gpu-available?
      (gp/gpu-skip! "scheduled-coarse-fine-cycle")
      (with-open [executable (gpu-distributed/instantiate! plan)]
        (gpu-distributed/run! executable)
        (let [result (first (vals (get (gpu-distributed/output-values executable) [:restriction :apply])))
              actual (double-array 15)]
          (gpu/download-range! (get (:sessions executable) :ze:0) result actual {:elements 15})
          (is (= (vec coarse) (vec actual))))))))
