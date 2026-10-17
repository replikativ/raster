(ns raster.compiler.passes.parallel.source-free-invocation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-map-element-test :as map-fixture]
            [raster.ad.normalized-indexed-map-test :as indexed-fixture]
            [raster.ad.normalized-map-sum-test :as sum-fixture]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.kernel-body]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.structured-control-route :as route]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- change-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

(defn- direct-flat-plan [id parameters values]
  (invocation/from-prefix
   {:id id :parameters parameters :parameter-values (select-keys values parameters)
    :bindings [] :binding-values {} :program-values values
    :program-inputs parameters :program-outputs []}))

(defn- pure-map
  ([] (pure-map {}))
  ([{:keys [shape output-shape] :or {shape '[n]}}]
   (let [array (av/tensor {:dtype :double :shape shape})
         output-array (av/tensor {:dtype :double :shape (or output-shape shape)})
         n (av/tensor {:dtype :long :shape []})
         equation (list '= 'map-0 '[y]
                        (list 'map {:index 'i :extent 'n} '[x] []
                              (soac/lambda-form
                               '[element] '[(raster.numeric/* element element)])))]
     (soac/make
      (soac/default-program-facts
       {:values {'x array 'n n 'y output-array}
        :inputs '[x n]
        :equations {'map-0 (soac/default-equation-facts)}})
      [equation] '[y]))))

(defn- ordered-effect-program
  ([] (ordered-effect-program 3 4))
  ([gradient-extent] (ordered-effect-program 3 gradient-extent))
  ([extent gradient-extent]
   (let [source (#'indexed-fixture/indexed-map :float :long extent gradient-extent)
         bundle (#'indexed-fixture/build source)]
     (get-in bundle [:gather-programs :transpose]))))

(defn- private-shape-map
  ([] (private-shape-map {}))
  ([{:keys [dimension-value data-shape extra-source?]
     :or {dimension-value (av/tensor {:dtype :long :shape []})
          data-shape '[m]}}]
   (let [indices (av/tensor {:dtype :long :shape [3]})
         data (av/tensor {:dtype :float :shape data-shape})
         output (av/tensor {:dtype :float :shape [3] :ownership :owned})
         values (cond-> {'indices indices 'data data 'm dimension-value 'out output}
                  extra-source? (assoc 'other data))
         inputs (cond-> '[indices data] extra-source? (conj 'other))
         captures (if extra-source? '[data other] '[data])
         parameters (if extra-source?
                      '[index data-array other-array]
                      '[index data-array])]
     (soac/make
      (soac/default-program-facts
       {:values values :inputs inputs
        :equations {'gather (soac/default-equation-facts)}})
      [(list '= 'gather '[out]
             (list 'map {:index 'i :extent 3
                         :attributes {:stable-array-captures captures}}
                   '[indices] captures
                   (soac/lambda-form
                    parameters []
                    '[(clojure.core/aget data-array index)])))]
      '[out]))))

(defn- nested-records
  [value]
  (lazy-seq
   (cons value
         (cond
           (map? value) (mapcat nested-records (vals value))
           (coll? value) (mapcat nested-records value)
           :else []))))

(deftest source-free-ordered-effect-promotion-retains-unresolved-storage-only
  (let [typed (ordered-effect-program)
        promoted (route/promote-typed-effect-algorithm typed)
        plan (get-in promoted [:attributes :invocation-plan])
        [storage] (:storage-bindings plan)
        destination (:program-value storage)
        [equation] (soac/equations typed)
        result (first (soac/outputs typed))
        touched (atom [])
        hostile-arguments (reify clojure.lang.Seqable
                            (seq [_] (swap! touched conj :arguments) nil))]
    (is (= {:program-value destination :external true} storage))
    (is (not (some #{destination} (:program-inputs plan))))
    (is (not= destination result))
    (is (= destination (get-in (soac/facts typed)
                               [:equations (second equation) :aliases result])))
    (is (= plan (invocation/validate-against! plan promoted)))
    (is (= :resident-premise (get-in plan [:attributes :external-storage])))
    (is (= :invocation-materialization-external-storage
           (reason #(materialization/materialize
                     plan hostile-arguments
                     (fn [& _] (swap! touched conj :evaluate))))))
    (is (empty? @touched)
        "external storage refuses before argument projection or scalar evaluation")
    (testing "the unchanged pure entry never accepts the external effect boundary"
      (is (= :structured-control-source-free-effects
             (reason #(route/promote-typed-algorithm typed)))))
    (testing "a dynamic private destination is retained as a closed premise"
      (let [dynamic (route/promote-typed-effect-algorithm
                     (ordered-effect-program 'm))
            dynamic-plan (get-in dynamic [:attributes :invocation-plan])]
        (is (= [{:id 'm :dtype :long}]
               (get-in dynamic-plan [:attributes :external-dimension-values])))
        (is (= :resident-premise
               (get-in dynamic-plan [:attributes :external-dimensions])))
        (is (= :resident-premise
               (get-in dynamic-plan [:attributes :external-storage])))
        (is (= :invocation-materialization-external-dimensions
               (reason #(materialization/materialize dynamic-plan [] nil))))))
    (testing "external premise maps and their rederived destination are exact"
      (is (= :invocation-storage-binding-form
             (reason #(invocation/validate!
                       (assoc-in plan [:storage-bindings 0 :forged] true)))))
      (is (= :invocation-external-storage-mode
             (reason #(invocation/validate!
                       (update plan :attributes dissoc :external-storage)))))
      (is (= :invocation-external-storage-mode
             (reason #(invocation/validate!
                       (update plan :storage-bindings conj
                               {:program-value (first (:program-inputs plan))
                                :invocation-value (get-in plan [:bindings 0 :invocation-value])})))))
      (is (= :invocation-storage-bindings
             (reason #(invocation/validate-against!
                       (assoc-in plan [:storage-bindings 0 :program-value]
                                 (first (:program-inputs plan)))
                       promoted))))
      (is (= :invocation-external-storage-boundary
             (reason #(invocation/validate!
                       (assoc-in plan [:storage-bindings 0 :program-value] result)))))
      (is (= :invocation-external-storage-mode
             (reason #(invocation/validate-against!
                       (assoc plan :storage-bindings []) promoted))))
      (let [rogue 'unowned-external-candidate
            abstract (get-in plan [:values destination])
            changed-plan (-> plan
                             (assoc-in [:values rogue] abstract)
                             (assoc-in [:storage-bindings 0 :program-value] rogue))
            changed-program (assoc-in promoted [:values rogue] abstract)]
        (is (= changed-plan (invocation/validate! changed-plan)))
        (is (= changed-program (route/validate-typed-program! changed-program)))
        (is (= :invocation-program-mismatch
               (reason #(invocation/validate-against! changed-plan changed-program)))))
      (let [algorithm (get-in promoted [:equations 0 :algorithm])
            equation-id (second (first (soac/equations algorithm)))
            result (first (soac/outputs algorithm))
            rogue 'forged-alias-destination
            facts (-> (soac/facts algorithm)
                      (assoc-in [:values rogue] (get-in plan [:values destination]))
                      (assoc-in [:equations equation-id :aliases result] rogue))
            malformed (apply list (assoc (vec algorithm) 1 facts))
            changed-program (assoc-in promoted [:equations 0 :algorithm] malformed)]
        (is (= :typed-soac-result-storage-alias
               (reason #(invocation/validate-against! plan changed-program)))))
      (let [pure-plan (get-in (route/promote-typed-algorithm (pure-map))
                              [:attributes :invocation-plan])]
        (is (= :invocation-external-storage-mode
               (reason #(invocation/validate!
                         (assoc-in pure-plan [:attributes :external-storage]
                                   :resident-premise)))))
        (is (= :invocation-external-storage-mode
               (reason #(invocation/validate!
                         (assoc-in pure-plan [:attributes :external-storage]
                                   :forged)))))))))

(deftest source-free-indexed-transpose-retains-one-closed-external-dimension
  (let [source (#'indexed-fixture/indexed-map :float :long 3 'm)
        bundle (#'indexed-fixture/build source)
        transpose (get-in bundle [:gather-programs :transpose])
        promoted (route/promote-typed-effect-algorithm transpose)
        plan (get-in promoted [:attributes :invocation-plan])
        dimensions (get-in plan [:attributes :external-dimension-values])
        touched (atom [])
        hostile (reify clojure.lang.Seqable
                  (seq [_] (swap! touched conj :arguments) nil))]
    (is (= [{:id 'm :dtype :long}] dimensions))
    (is (= :resident-premise (get-in plan [:attributes :external-dimensions])))
    (is (not (some #{'m} (:program-inputs plan))))
    (is (not (some #{'m} (:program-outputs plan))))
    (is (not (some #{'m} (mapv :symbol (:parameters plan)))))
    (is (= '[m] (get-in (soac/facts transpose)
                         [:values (get-in bundle [:gather-programs :interface
                                                  :gradient-output]) :shape])))
    (is (= :invocation-materialization-external-dimensions
           (reason #(materialization/materialize
                     plan hostile (fn [& _] (swap! touched conj :evaluate))))))
    (is (empty? @touched))
    (is (= :invocation-program-mismatch
           (reason #(invocation/validate-against!
                     (update plan :attributes dissoc
                             :external-dimensions :external-dimension-values)
                     promoted))))
    (is (= :invocation-external-dimension-mode
           (reason #(invocation/validate!
                     (assoc-in plan [:attributes :external-dimension-values 0 :dtype]
                               :int)))))
    (is (= :invocation-external-dimension-mode
           (reason #(invocation/validate!
                     (update-in plan [:attributes :external-dimension-values]
                                conj {:id 'forged :dtype :long})))))
    (let [changed (assoc-in plan [:values 'm :attributes :forged] true)]
      (is (= changed (invocation/validate! changed)))
      (is (= :invocation-program-mismatch
             (reason #(invocation/validate-against! changed promoted)))))
    (let [scheduled (route/schedule-program promoted
                                            {:target-device :ocl:external-dimension
                                             :dtype :float})]
      (doseq [target [:opencl-intel :opencl-portable :cuda :hip]]
        (let [{:keys [program kernels]}
              (c-family/emit-program scheduled {:target-dialect target :dtype :float})]
          (is (some? program) (str target))
          (is (seq kernels) (str target))))))
  (testing "an external dimension cannot become executable or native authority"
    (let [source (#'indexed-fixture/indexed-map :float :long 3 'm)
          transpose (get-in (#'indexed-fixture/build source) [:gather-programs :transpose])
          promoted (route/promote-typed-effect-algorithm transpose)]
      (is (= :invocation-source-free-external-dimension
             (reason #(invocation/source-free-external-dimensions
                       (update-in promoted [:equations 0 :operands] conj 'm) []))))
      (is (= :invocation-source-free-external-dimension
             (reason #(route/promote-typed-effect-algorithm
                       (change-facts
                        transpose
                        (fn [facts]
                          (assoc-in facts [:attributes :native-initialization-providers]
                                    ['m]))))))
      (doseq [host-attributes [{:host-read-values ['m]}
                               {:host-read-sites [{:values #{'m}}]}]]
        (let [changed-algorithm
              (change-facts transpose #(update % :attributes merge host-attributes))
              changed (assoc-in promoted [:equations 0 :algorithm] changed-algorithm)]
          (is (identical? changed-algorithm (soac/validate! changed-algorithm)))
          (is (= changed (route/validate-typed-program! changed)))
          (is (= :invocation-source-free-external-dimension
                 (reason #(invocation/source-free-external-dimensions changed []))))))))))

(deftest source-free-ordered-effect-refuses-native-initialization-and-retains-zero-extent
  (let [base (ordered-effect-program)
        native (change-facts
                base #(assoc-in % [:attributes :native-initialization-providers]
                                [(-> base soac/equations first soac/operation-parts
                                     :destinations first)]))]
    (is (identical? base (soac/validate! base)))
    (is (= :structured-control-source-free-effects
           (reason #(route/promote-source-free-algorithm native))))
    (let [zero (ordered-effect-program 0 4)
          promoted (route/promote-source-free-algorithm zero)
          scheduled (route/schedule-program promoted {:target-device :ocl:zero-effect
                                                       :dtype :float})
          storage (get-in promoted [:attributes :invocation-plan :storage-bindings])]
      (is (= 1 (count storage)))
      (is (true? (get-in storage [0 :external])))
      (is (= 0 (get-in (-> zero soac/equations first soac/operation-parts)
                       [:attributes :extent])))
      (doseq [target [:opencl-intel :opencl-portable :cuda :hip]]
        (let [emitted (c-family/emit-program scheduled {:target-dialect target :dtype :float})
              kernel (first (:kernels emitted))
              body (get-in kernel [:attributes :kernel-body])
              inputs (filter #(= :input (:kind %)) (:parameters body))
              bindings (zipmap (map :name (:abi kernel)) (:arguments kernel))
              loops (distinct (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                                      (nested-records body)))]
          (is (= 1 (count (:kernels emitted))) (str target))
          (is (= [[0] [0]] (mapv :shape inputs)) "empty inputs are not padded to fake capacity")
          (is (= (mapv :shape inputs) (mapv #(get-in % [:layout :shape]) inputs)))
          (is (= [[1] [1]] [(:workgroup-size (:launch body)) (:group-count (:launch body))]))
          (is (= 1 (count loops)))
          (is (= 0 (get-in (first loops) [:lower :argument])))
          (is (= 1 (:step (first loops))))
          (is (= '_n_bound (get-in (first loops) [:upper :argument])))
          (is (= 0 (get bindings '_n_bound)) "the ordered loop performs no input reads"))))))

(deftest source-free-ordered-effect-schedules-one-ordered-loop-without-atomics
  (let [promoted (route/promote-typed-effect-algorithm (ordered-effect-program))
        scheduled (route/schedule-program promoted {:target-device :ocl:source-free-effect
                                                     :dtype :float})
        emitted (c-family/emit-program scheduled {:target-dialect :opencl-portable
                                                  :dtype :float})
        body (get-in emitted [:kernels 0 :attributes :kernel-body])
        records (nested-records body)
        loops (distinct (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %) records))
        atomics (distinct (filter #(instance? raster.compiler.ir.kernel_body.AtomicRMW %) records))]
    (is (= :scheduled-parallel (:dialect scheduled)))
    (is (= :one-work-item-ordered-loop (get-in body [:schedule :strategy])))
    (is (= [[1] [1]] [(:workgroup-size (:launch body))
                      (:group-count (:launch body))]))
    (is (= 1 (count loops)))
    (is (empty? atomics))
    (is (= :ordered (get-in (first loops) [:attributes :association])))
    (doseq [target [:opencl-intel :opencl-portable :cuda :hip]]
      (let [{:keys [program kernels]}
            (c-family/emit-program scheduled {:target-dialect target :dtype :float})]
        (is (some? program) (str target))
        (is (= 1 (count kernels)) (str target))))))

(deftest source-free-promotion-retains-the-exact-zero-prefix-interface
  (let [typed (pure-map)
        promoted (route/promote-typed-algorithm typed)
        plan (get-in promoted [:attributes :invocation-plan])
        values (:values (soac/facts typed))]
    (is (= :typed-parallel (:dialect promoted)))
    (is (nil? (:source promoted)))
    (is (= '[x n] (:inputs promoted) (mapv :symbol (:parameters plan))))
    (is (= '[y] (:outputs promoted) (:program-outputs plan)))
    (is (empty? (:steps plan)))
    (is (empty? (:storage-bindings plan)))
    (is (= '[x n] (mapv :program-value (:bindings plan))))
    (is (= (select-keys values '[x n])
           (into {} (map (juxt :symbol :value)) (:parameters plan))))
    (is (every? true?
                (map #(identical? (get values (:symbol %)) (:value %)) (:parameters plan))))
    (is (= values (:values promoted)))
    (is (= plan (invocation/validate-against! plan promoted)))
    (is (= :typed-soac (get-in plan [:attributes :source-dialect])))
    (is (= :explicit-typed-algorithm (get-in promoted [:provenance :front-end])))))

(deftest source-free-private-dimension-uses-one-exact-self-shape-projection
  (let [typed (private-shape-map)
        original-values (:values (soac/facts typed))
        promoted (route/promote-typed-algorithm typed)
        plan (get-in promoted [:attributes :invocation-plan])
        marker (get-in plan [:attributes :source-free-shape-projections])
        [projection] (filter invocation/shape-projection? (:steps plan))
        data-parameter (some #(when (= 'data (:symbol %)) %) (:parameters plan))
        materialized
        (materialization/materialize
         plan
         (mapv (fn [{:keys [symbol]}]
                 (case symbol
                   indices (long-array [0 1 2])
                   data (float-array [4.0 5.0 6.0 7.0])))
               (:parameters plan))
         nil)]
    (is (= [{:dimension 'm :source 'data :axis 0}] marker))
    (is (= 'm (:symbol projection)))
    (is (= (:id data-parameter) (:source projection)))
    (is (= '[data] (mapv :symbol (:operands projection))))
    (is (= '[(extent data)] (get-in data-parameter [:value :shape])))
    (is (= (dissoc (get original-values 'data) :shape)
           (dissoc (:value data-parameter) :shape)))
    (is (= '[m] (get-in (soac/facts typed) [:values 'data :shape])
           (get-in promoted [:values 'data :shape])))
    (is (identical? (get original-values 'data) (get-in promoted [:values 'data])))
    (is (= {:type :long :value 4} (get-in materialized [:values (:id projection)])))
    (is (= [4] (get-in materialized [:program-buffers 'data :shape])))
    (is (= plan (invocation/validate-against! plan promoted)))
    (testing "marker, source and parameter facets participate in fresh correspondence"
      (is (= :invocation-program-mismatch
             (reason #(invocation/validate-against!
                       (update plan :attributes dissoc :source-free-shape-projections)
                       promoted))))
      (is (= :invocation-program-mismatch
             (reason #(invocation/validate-against!
                       (update plan :attributes dissoc
                               :source-dialect :source-free-shape-projections)
                       promoted))))
      (is (= :invocation-program-mismatch
             (reason #(invocation/validate-against!
                       plan (assoc-in promoted [:attributes :host-control]
                                      :explicit-typed-algorithm)))))
      (doseq [tampered [(assoc-in plan [:attributes :source-free-shape-projections 0 :axis] 1)
                        (assoc-in plan [:attributes :source-free-shape-projections 0 :source]
                                  'indices)
                        (assoc-in plan [:parameters 1 :value :ownership] :borrowed)
                        (assoc plan :steps [])]]
        (is (some? (reason #(invocation/validate-against! tampered promoted)))))))
  (testing "a storage-compatible parameter cannot replace the projection's exact source"
    (let [base (private-shape-map)
          facts (-> (soac/facts base)
                    (assoc-in [:values 'other]
                              (av/tensor {:dtype :float :shape [4]}))
                    (update :inputs conj 'other))
          typed (soac/make
                 facts
                 [(list '= 'gather '[out]
                        (list 'map {:index 'i :extent 3
                                    :attributes {:stable-array-captures '[data other]}}
                              '[indices] '[data other]
                              (soac/lambda-form
                               '[index data-array other-array] []
                               '[(clojure.core/aget data-array index)])))]
                 '[out])
          promoted (route/promote-typed-algorithm typed)
          plan (get-in promoted [:attributes :invocation-plan])
          data-binding (first (keep-indexed
                               #(when (= 'data (:program-value %2)) %1)
                               (:bindings plan)))
          other-parameter (some #(when (= 'other (:symbol %)) %) (:parameters plan))
          changed (assoc-in plan [:bindings data-binding :invocation-value]
                            (:id other-parameter))]
      (is (= changed (invocation/validate! changed)))
      (is (= :invocation-program-mismatch
             (reason #(invocation/validate-against! changed promoted))))))
  (testing "the current canonical indexed primal uses the same projection"
    (let [source (#'indexed-fixture/indexed-map :float :long 3 'm)
          primal (get-in (#'indexed-fixture/build source) [:gather-programs :primal])
          promoted (route/promote-typed-algorithm primal)]
      (is (= [{:dimension 'm :source 'data :axis 0}]
             (get-in promoted [:attributes :invocation-plan :attributes
                               :source-free-shape-projections])))
      (is (= '[m] (get-in (soac/facts primal) [:values 'data :shape])
             (get-in promoted [:values 'data :shape]))))))

(deftest source-free-private-dimension-materializes-and-emits-through-common-boundaries
  (let [typed (private-shape-map)
        compilation (equation-first/compile-typed
                     typed {:target :ocl:shape-projection
                            :schedule {:precision :f32-scalar}})
        plan (get-in compilation [:semantic :attributes :invocation-plan])
        arguments (mapv (fn [{:keys [symbol]}]
                          (case symbol
                            indices (long-array [0 1 2])
                            data (float-array [4.0 5.0 6.0 7.0])))
                        (:parameters plan))
        materialized (materialization/materialize plan arguments nil)
        layout (invocation-link/project-resident-layout
                materialized (:emitted compilation) (:target compilation))
        scheduled (route/schedule-program
                   (route/promote-typed-algorithm typed)
                   {:target-device :ocl:shape-projection :dtype :float})]
    (is (= {:type :long :value 4} (get-in layout [:scalar-values 'm])))
    (is (= [4] (get-in layout [:values (get-in layout [:buffer-bindings 'data])
                                :abstract :shape])))
    (doseq [target [:opencl-intel :opencl-portable :cuda :hip]]
      (let [{:keys [program kernels]}
            (c-family/emit-program scheduled {:target-dialect target :dtype :float})]
        (is (some? program) (str target))
        (is (seq kernels) (str target))))))

(deftest source-free-private-dimension-refuses-ambiguous-or-expanded-authority
  (is (= :invocation-source-free-shape-projection
         (reason #(route/promote-typed-algorithm
                   (private-shape-map {:extra-source? true})))))
  (is (= :invocation-source-free-shape-projection
         (reason #(route/promote-typed-algorithm
                   (private-shape-map
                    {:dimension-value (av/tensor {:dtype :float :shape []})})))))
  (is (= :structured-control-source-free-dimension
         (reason #(route/promote-typed-algorithm
                   (private-shape-map {:data-shape '[(clojure.core/+ m 0)]})))))
  (let [promoted (route/promote-typed-algorithm (private-shape-map))
        executable (update-in promoted [:equations 0 :operands] conj 'm)]
    (is (= :invocation-source-free-shape-projection
           (reason #(invocation/source-free-shape-projections executable)))))
  (testing "an explicit public dimension retains the old zero-prefix authority"
    (let [base (private-shape-map)
          facts (update (soac/facts base) :inputs conj 'm)
          typed (soac/make
                 facts
                 [(list '= 'gather '[out]
                        (list 'map {:index 'i :extent 3
                                    :attributes {:stable-array-captures '[data]}}
                              '[indices] '[data m]
                              (soac/lambda-form
                               '[index data-array data-size] []
                               '[(clojure.core/aget
                                  data-array (clojure.core/rem index data-size))])))]
                 '[out])
          promoted (route/promote-typed-algorithm typed)]
      (is (nil? (get-in promoted [:attributes :invocation-plan :attributes
                                  :source-free-shape-projections])))
      (is (empty? (get-in promoted [:attributes :invocation-plan :steps]))))))

(deftest source-free-promotion-materializes-through-the-existing-flat-boundary
  (let [promoted (route/promote-typed-algorithm (pure-map))
        plan (get-in promoted [:attributes :invocation-plan])
        input (double-array [1.0 2.0 3.0])
        materialized (materialization/materialize plan [input 3] nil)
        x-id (:invocation-value (first (:bindings plan)))
        n-id (:invocation-value (second (:bindings plan)))]
    (is (= [3] (get-in materialized [:program-buffers 'x :shape])))
    (is (identical? input (get-in materialized [:program-buffers 'x :source])))
    (is (= {:type :long :value 3} (get-in materialized [:program-scalars 'n])))
    (is (= (get-in materialized [:values x-id])
           (get-in materialized [:program-buffers 'x])))
    (is (= (get-in materialized [:values n-id])
           (get-in materialized [:program-scalars 'n])))
    (is (= :invocation-materialization-arity
           (reason #(materialization/materialize plan [input] nil))))
    (is (= :invocation-materialization-shape-value
           (reason #(materialization/materialize plan [input 3.0] nil))))
    (is (= '[x n] (mapv :symbol (:parameters plan)))
        "buffer-before-dimension order is retained and materializes from the complete scalar set"))
  (testing "the existing materializer executes its canonical value and self-extent forms"
    (let [n (av/tensor {:dtype :long :shape []})
          by-value (av/tensor {:dtype :double :shape '[(value n)]})
          self-sized (av/tensor {:dtype :double :shape '[(extent self-sized)]})
          value-plan (direct-flat-plan :value-dimension '[buffer n]
                                       {'buffer by-value 'n n})
          self-plan (direct-flat-plan :self-extent '[self-sized]
                                      {'self-sized self-sized})]
      (is (= [3] (get-in (materialization/materialize
                           value-plan [(double-array 3) 3] nil)
                         [:program-buffers 'buffer :shape])))
      (is (= [4] (get-in (materialization/materialize
                           self-plan [(double-array 4)] nil)
                         [:program-buffers 'self-sized :shape]))))))

(deftest source-free-promotion-refuses-unsupported-physical-boundaries
  (testing "rank, missing facts and compound dimensions remain outside the flat boundary"
    (is (some? (reason #(route/promote-typed-algorithm (pure-map {:shape '[2 n]})))))
    (let [missing (change-facts
                   (pure-map)
                   #(assoc-in % [:values 'unsupported]
                              (av/tensor {:dtype :double :shape '[private-size]})))]
      (is (= :invocation-source-free-external-dimension
             (reason #(route/promote-typed-algorithm missing)))))
    (doseq [shape '[[(value private-size)] [(extent private-buffer)]
                     [(clojure.core/+ n 1)]]]
      (let [typed (change-facts
                   (pure-map)
                   #(assoc-in % [:values 'unsupported]
                              (av/tensor {:dtype :double :shape shape})))]
        (is (= :structured-control-source-free-dimension
               (reason #(route/promote-typed-algorithm typed)))))))
  (testing "a present private integral dimension becomes an explicit resident premise"
    (let [base (pure-map)
          private-scalar (av/tensor {:dtype :long :shape []})
          private-shaped (change-facts
                          base
                          #(-> %
                               (assoc-in [:values 'private-size] private-scalar)
                               (assoc-in [:values 'private-buffer]
                                         (av/tensor {:dtype :double :shape '[private-size]}))))
          promoted (route/promote-typed-algorithm private-shaped)
          plan (get-in promoted [:attributes :invocation-plan])]
      (is (= [{:id 'private-size :dtype :long}]
             (get-in plan [:attributes :external-dimension-values])))
      (is (= :resident-premise (get-in plan [:attributes :external-dimensions])))
      (is (not (some #{'private-size} (:program-inputs plan))))
      (is (= :invocation-materialization-external-dimensions
             (reason #(materialization/materialize plan [] nil))))
      (testing "a wrongly typed public dimension remains unavailable"
        (is (= :structured-control-source-free-dimension
               (reason #(#'route/source-free-dimension!
                          {'scale (av/tensor {:dtype :double :shape []})}
                          #{} false 'buffer 'scale)))))))
  (testing "an input may inspect only its own extent; produced values may use a public buffer"
    (let [base (pure-map)
          other (av/tensor {:dtype :double :shape '[(extent x)]})
          produced (change-facts base #(assoc-in % [:values 'scratch] other))]
      (is (= :structured-control-source-free-dimension
             (reason #(#'route/source-free-dimension!
                        {'x (av/tensor {:dtype :double :shape '[(extent x)]})
                         'other other}
                        #{'x 'other} true 'other '(extent x)))))
      (is (= '(value n)
             (#'route/source-free-dimension!
              {'n (av/tensor {:dtype :long :shape []})} #{} false 'scratch '(value n))))
      (is (some? (route/promote-typed-algorithm produced)))))
  (testing "effects, aliases, storage and native initialization never become invocation authority"
    (let [base (pure-map)]
      (is (= :structured-control-source-free-effects
             (reason #(route/promote-typed-algorithm
                       (change-facts base (fn [facts] (assoc facts :aliases {'y 'x})))))))
      (is (= :structured-control-source-free-effects
             (reason #(route/promote-typed-algorithm
                       (change-facts
                        base
                        (fn [facts]
                          (assoc-in facts [:attributes :native-initialization-providers]
                                    ['x])))))))
      (is (some? (reason #(route/promote-typed-algorithm
                           (change-facts
                            base
                            (fn [facts]
                              (-> facts
                                  (assoc-in [:equations 'map-0 :aliases] {'y 'x})
                                  (assoc-in [:equations 'map-0 :attributes :result-storage]
                                            [{:destination 'x :access :read-write
                                              :host-return :buffer}]))))))))
      (is (some? (reason #(route/promote-typed-algorithm
                           (change-facts
                            base
                            (fn [facts]
                              (-> facts
                                  (assoc :effects #{:memory/write})
                                  (assoc-in [:equations 'map-0 :effects]
                                            #{:memory/write})))))))))))

(deftest source-free-promotion-does-not-relax-the-source-route
  (let [typed (assoc (route/promote-typed-algorithm (pure-map))
                     :source '(let* [x (double-array 2)] x))]
    (is (some? (reason #(route/promote-program typed {}))))))

(defn- canonical-programs []
  (let [map-bundle (normalized/linearize-map-program
                    (#'map-fixture/map-program :double {})
                    (the-ns 'raster.compiler.passes.parallel.source-free-invocation-test))
        sum-bundle (#'sum-fixture/build (#'sum-fixture/source :double) 'xs)]
    (for [[family bundle facet] [[:map map-bundle :map-programs]
                                 [:sum sum-bundle :sum-programs]]
          phase [:primal :linear :transpose]]
      {:family family :phase phase :program (get-in bundle [facet phase])})))

(deftest canonical-derivatives-use-the-common-source-free-schedule-and-emitters
  (doseq [{:keys [family phase] typed :program} (canonical-programs)]
    (let [promoted (route/promote-typed-algorithm typed)
          scheduled (route/schedule-program promoted {:target-device :ocl:0 :dtype :double})]
      (is (= :typed-parallel (:dialect promoted)))
      (is (= :scheduled-parallel (:dialect scheduled)))
      (doseq [target [:opencl-intel :opencl-portable :cuda :hip]]
        (let [{:keys [program kernels]}
              (c-family/emit-program scheduled {:target-dialect target :dtype :double})]
          (is (seq kernels) (str family " " phase " " target " kernels"))
          (is (some? program) (str family " " phase " " target " program")))))))
