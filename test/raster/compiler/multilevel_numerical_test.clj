(ns raster.compiler.multilevel-numerical-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.ode.multilevel :as multilevel]
            [raster.ode.multilevel-compiler :as multilevel-compiler]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.amr-execution :as amr-execution]
            [raster.compiler.ir.numerical-state :as numerical-state]
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
        (is (= (vec (for [i (range nx) j (range ny)]
                      (/ (reduce + (for [di [0 1] dj [0 1]]
                                     (aget fine (+ (* (+ (* 2 i) di) (* 2 ny))
                                                   (* 2 j) dj)))) 4.0)))
               (vec restored)))
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

(defn- bilinear-cell-averages [nx ny]
  ;; Exact unit-square cell averages of 1 + x + 2y + xy, not point samples of a
  ;; general function. Its domain integral is 11/4 on every resolution.
  (double-array (for [i (range nx) j (range ny)
                      :let [x (/ (+ i 0.5) nx) y (/ (+ j 0.5) ny)]]
                  (+ 1.0 x (* 2.0 y) (* x y)))))

(deftest smooth-cell-averages-preserve-integrals-and-expose-transfer-error
  (let [errors
        (mapv (fn [nx]
                (let [ny (* 2 nx)
                      coarse (bilinear-cell-averages nx ny)
                      exact-fine (bilinear-cell-averages (* 2 nx) (* 2 ny))
                      prolonged (double-array (* 4 nx ny))
                      restricted (double-array (* nx ny))]
                  (multilevel/prolong-constant-2d! prolonged coarse nx ny)
                  (multilevel/restrict-average-2d! restricted exact-fine nx ny)
                  (is (< (reduce max 0.0 (map #(Math/abs (- %1 %2)) coarse restricted)) 1.0e-12))
                  (is (< (Math/abs (- 2.75 (/ (reduce + (vec prolonged)) (* 4 nx ny)))) 1.0e-12))
                  (is (< (Math/abs (- 2.75 (/ (reduce + (vec restricted)) (* nx ny)))) 1.0e-12))
                  (Math/sqrt (/ (reduce + (map (fn [a b] (let [d (- a b)] (* d d)))
                                              prolonged exact-fine))
                                (* 4 nx ny)))))
              [8 16 32])]
    ;; Piecewise-constant prolongation is conservative, not an exact smooth-field
    ;; refinement: its volume-weighted RMS discrepancy must decrease at first order.
    (is (every? pos? errors))
    (doseq [[coarser finer] (partition 2 1 errors)]
      (is (< 1.95 (/ coarser finer) 2.05)))))

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
    (with-meta (distributed/plan
     {:id :generated-multilevel-cycle
      :mesh (distributed/mesh [{:name :worker :size 1}] [:ze:0])
      :topology (distributed/topology [(distributed/device {:id :ze:0 :memory-capacity-bytes 1048576})] [])
      :values values
      :shards (into {} (for [[field shape] shapes]
                        [field [(distributed/shard {:id field :value field :device :ze:0
                                                    :offsets [0 0] :shape shape})]]))
      :device-plans {:ze:0 {:entries entries :steps calls}}
      :steps (vec (mapcat :steps scheduled)) :outputs [(:completion (peek scheduled))]})
      {:hierarchy hierarchy :scheduled scheduled})))

(defn- certified-transfer-workload [plan]
  (let [{:keys [hierarchy scheduled]} (meta plan)
        fields (mapv
                (fn [patch]
                  (let [field (:field patch) shape (:shape patch) n (reduce * shape)]
                    (numerical-state/field
                     {:id field :value (get-in plan [:values field]) :chunk-shape shape
                      :coordinate-space {:hierarchy (:id hierarchy) :level (:level patch) :patch (:id patch)
                                         :axes [{:name :x :centering :cell} {:name :y :centering :cell}]}
                      ;; Structural manifest fixture only: no durable publication is claimed.
                      :chunks [(numerical-state/chunk
                                {:id field :offsets [0 0] :shape shape
                                 :logical-byte-length (* 8 n) :stored-byte-length (* 8 n)
                                 :content (numerical-state/content-address :sha-256 (format "%064x" n))
                                 :storage {:format :raw-array :byte-order :little-endian}})]})))
                (mapcat :patches (:levels hierarchy)))
        state (numerical-state/certify
               (numerical-state/manifest
                {:id :transfer-state :parents [] :logical-coordinate {:step 0}
                 :fields fields :numerical-contract {:mode :ieee-fp64 :determinism :reproducible-order
                                                    :compatibility-id "multilevel-f64"}
                 :provenance {:program-fingerprint "multilevel-structural-fixture"}}))]
    (amr/certify (amr/plan {:id :transfer-workload :mode :transfer-cycle
                           :hierarchy hierarchy :state state :distributed-plan (distributed/certify plan)
                           :coarse-fine scheduled}))))

(deftest implementation-witnesses-bind-exact-plans-and-field-effects
  (let [coarse (double-array (map #(* 0.25 %) (range 15)))
        fine (double-array 60)
        plan (scheduled-transfer-cycle coarse fine)
        workload (certified-transfer-workload plan)
        bindings (:bindings (distributed/compute-bindings plan))
        implementations
        (into {} (for [[kind source-field target-field]
                       [[:prolongation :coarse :fine] [:restriction :fine :coarse]]
                       :let [binding (bindings [kind :apply])
                             local-id (fn [field] (some (fn [[id value]] (when (= field (:value value)) id))
                                                       (:values binding)))]]
                   [kind (amr-execution/attest-implementation
                          workload kind (:link-plan binding)
                          {:source (local-id source-field) :target (local-id target-field)
                           :producer :test/multilevel-compiler-provider :invariants #{:constant-preserving}
                           :numerical {:mode :exact :policy :typed-fp64-evaluation-order}})]))
        execution (amr-execution/certify workload implementations)
        failure (fn [changed]
                  (try (amr-execution/certify workload changed) nil
                       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= execution (amr-execution/verify! execution)))
    (is (= :amr-execution-implementation
           (failure (assoc-in implementations [:prolongation :link-plan]
                              (get-in implementations [:restriction :link-plan])))))
    (is (= :amr-execution-effects
           (failure (assoc-in implementations [:prolongation :source]
                              (get-in implementations [:prolongation :target])))))
    (is (= :amr-execution-invariants
           (failure (assoc-in implementations [:restriction :invariants] #{}))))
    (is (= :amr-execution-implementations (failure (dissoc implementations :restriction))))
    (let [partial (equation/lower (equation/compile #'arrays/acopy! {:target :ze:0 :dtype :double})
                                  [coarse 0 fine 0 3])
          partial (update partial :nodes
                          #(update-vals % (fn [node]
                                            (assoc-in node [:view :allocation :id]
                                                      (if (identical? coarse (:source node)) :coarse :fine)))))
          ids (into {} (for [[id value] (:values partial)
                             :let [source (get-in partial [:nodes (get-in value [:leaves 0 :node]) :source])]]
                         [(if (identical? coarse source) :coarse :fine) id]))
          changed (-> plan
                      (assoc-in [:device-plans :ze:0 :entries :prolongation :link-plan] partial)
                      (assoc-in [:device-plans :ze:0 :steps [:prolongation :apply] :bindings]
                                (into {} (for [[field id] ids]
                                           [id {:local-shape (get-in plan [:values field :shape])
                                                :placements [{:kind :owned :value field :shard field
                                                              :local-offsets [0 0]}]}]))))
          changed-workload (amr/certify (amr/plan (assoc (:plan workload)
                                                       :distributed-plan (distributed/certify changed))))
          witness (amr-execution/attest-implementation
                   changed-workload :prolongation partial
                   {:source (ids :coarse) :target (ids :fine) :producer :test/incorrect-partial-provider
                    :invariants #{:constant-preserving} :numerical {:mode :exact :policy :test}})]
      (is (= :read-write (get (link-plan/value-accesses partial) (ids :fine))))
      (is (= :amr-execution-effects
             (try (amr-execution/certify changed-workload (assoc implementations :prolongation witness)) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
          "an initialized partial-copy result is not a pure whole-patch transfer"))
    (is (= :amr-execution-invariants (failure (assoc-in implementations [:restriction :producer] false))))
    (is (= :numerical-contract (failure (assoc-in implementations [:restriction :numerical] {}))))
    (is (= :amr-execution-implementation
           (failure (assoc-in implementations [:restriction :contract :ratio] [4 4]))))
    (let [compute-bindings distributed/compute-bindings]
      (with-redefs [distributed/compute-bindings
                    (fn [p] (assoc-in (compute-bindings p)
                                      [:bindings [:prolongation :apply] :boundary-outputs :extra]
                                      {:access :write}))]
        (is (= :amr-execution-effects (failure implementations))
            "boundary-output effects cannot bypass the exact external field set")))
    (let [compute-bindings distributed/compute-bindings
          target-id (get-in implementations [:prolongation :target])]
      (with-redefs [distributed/compute-bindings
                    (fn [p] (assoc-in (compute-bindings p)
                                      [:bindings [:prolongation :apply] :values target-id :access]
                                      :read-write))]
        (is (= :amr-execution-effects (failure implementations))
            "overwrite transfers may not depend on the previous target contents")))
    (let [compute-bindings distributed/compute-bindings
          source-id (get-in implementations [:prolongation :source])]
      (with-redefs [distributed/compute-bindings
                    (fn [p] (assoc-in (compute-bindings p)
                                      [:bindings [:prolongation :apply] :values source-id :domain] nil))]
        (is (= :amr-execution-shard (failure implementations))
            "logical shape alone cannot prove a whole dense physical patch")))
    (if-not @gp/gpu-available?
      (gp/gpu-skip! "certified-amr-execution-binding")
      (with-open [executable (gpu-distributed/instantiate!
                             (get-in (amr-execution/verify! execution)
                                     [:workload :plan :distributed-plan :plan]))]
        (gpu-distributed/run! executable)
        (let [result (first (vals (get (gpu-distributed/output-values executable) [:restriction :apply])))
              actual (double-array 15)]
          (gpu/download-range! (get (:sessions executable) :ze:0) result actual {:elements 15})
          (is (= (vec coarse) (vec actual))))))))

(deftest numerical-providers-compile-attest-and-execute-with-semantic-roles
  (let [coarse (double-array (map #(* 0.25 %) (range 15))) fine (double-array 60)
        original (scheduled-transfer-cycle coarse fine)
        workload (certified-transfer-workload original)
        implementations {:prolongation (multilevel-compiler/compile-prolongation
                                        workload :prolongation coarse fine {})
                         :restriction (multilevel-compiler/compile-restriction
                                       workload :restriction fine coarse {})}
        local {:entries (update-vals implementations #(hash-map :link-plan (:link-plan %)))
               :steps (into {}
                            (for [[id implementation] implementations
                                  :let [contract (:contract implementation)]]
                              [(:completion contract)
                               {:entry id
                                :bindings (into {}
                                                (for [role [:source :target]
                                                      :let [patch (get contract role)]]
                                                  [(get implementation role)
                                                   {:local-shape (:shape patch)
                                                    :placements [{:kind :owned :value (:field patch)
                                                                  :shard (:field patch) :local-offsets [0 0]}]}]))}]))}
        plan (distributed/plan (assoc original :device-plans {:ze:0 local}))
        workload (amr/certify (amr/plan (assoc (:plan workload) :distributed-plan (distributed/certify plan))))
        execution (amr-execution/certify workload implementations)]
    (is (= execution (amr-execution/verify! execution)))
    (is (= :bounded-error (get-in implementations [:restriction :numerical :mode])))
    (is (contains? (get-in implementations [:restriction :numerical :error-model :assumptions])
                   :no-intermediate-overflow))
    (with-redefs [equation/compile (fn [& _] (throw (ex-info "unexpected compilation" {})))]
      (is (= :multilevel-provider-contract
             (try (multilevel-compiler/compile-prolongation workload :restriction fine coarse {}) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (if-not @gp/gpu-available?
      (gp/gpu-skip! "provider-attested-multilevel-cycle")
      (with-open [executable (gpu-distributed/instantiate! plan)]
        (gpu-distributed/run! executable)
        (let [result (first (vals (get (gpu-distributed/output-values executable) [:restriction :apply])))
              actual (double-array 15)]
          (gpu/download-range! (get (:sessions executable) :ze:0) result actual {:elements 15})
          (is (= (vec coarse) (vec actual))))))))

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
