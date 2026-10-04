(ns raster.compiler.passes.parallel.scheduled-equation-graph-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.extent-expression :as extent]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.indexed-attention-recognize :as indexed-recognize]
            [raster.compiler.passes.parallel.indexed-weighted-reduction-body :as indexed-body]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.map-read-requirements :as map-reads]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.dl.nn :as nn]))

(def ^:private indexed-reduction-source
  '(let* [raw (raster.dl.array-ops/indexed-dot
               Q K dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          weights (raster.dl.array-ops/scale-clamp-exp
                   raw (raster.numeric// 1.0 (raster.numeric/sqrt dk))
                   5.0 (clojure.core/* n-edges n-heads))
          denominator (raster.dl.array-ops/scatter-add
                       weights dst n-nodes n-edges n-heads)
          weighted (raster.dl.array-ops/scatter-mul-add
                    weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
          normalized (raster.dl.array-ops/segment-div
                      weighted denominator n-nodes emb-dim n-heads 1.0e-6)]
     normalized))

(defn- indexed-plan-program
  []
  (let [plan (first (indexed-recognize/recognize
                     indexed-reduction-source :dtype :float :accumulator-dtype :float))
        runtime-values (vec (distinct (filter symbol? (:runtime-parameters plan))))
        input-values (into {}
                           (map (fn [{:keys [id dtype shape]}]
                                  [id (av/tensor {:dtype dtype :shape shape})]))
                           (:operands plan))
        output-description (:output plan)
        storage-id (:id output-description)
        result 'semantic-result
        values (merge input-values
                      (into {} (map (fn [id] [id (av/tensor {:dtype :long :shape []})]))
                            runtime-values)
                      {storage-id (av/tensor {:dtype (:dtype output-description)
                                              :shape [(:elements output-description)]})
                       result (av/tensor {:dtype (:dtype output-description)
                                          :shape (:shape output-description)})})
        operands (vec (distinct (concat (mapv :id (:operands plan)) runtime-values)))
        equation (program/->ProgramEquation
                  :indexed-reduction [:binding 'semantic-result] nil operands [result]
                  plan [plan] #{:memory/read :memory/write}
                  {:source :synthetic-indexed-reduction}
                  {:algorithm-dialect :segmented-weighted-reduction
                   :result-storage [{:destination storage-id :access :write
                                     :host-return :buffer}]})]
    (program/->ParallelProgram
     :typed-parallel nil values operands [equation] [result]
     #{:memory/read :memory/write} [] {:source :synthetic-indexed-reduction} {})))

(def ^:private three-map-source
  '(let* [first-effect
          (raster.par/map! tmp i n float (clojure.core/aget x i))
          second-effect
          (raster.par/map! middle j n float
                           (clojure.core/+ (clojure.core/aget tmp j) 1.0))
          third-effect
          (raster.par/map! out k n float
                           (clojure.core/* (clojure.core/aget middle k) 2.0))]
     third-effect))

(def ^:private scalar-gap-source
  '(let* [first-effect
          (raster.par/map! tmp i n float (clojure.core/aget x i))
          ^{:raster.type/tag long} doubled
          (clojure.core/* (clojure.core/long n) (clojure.core/long 2))
          second-effect
          (raster.par/map! out j n float
                           (clojure.core/+ (clojure.core/aget tmp j)
                                           (clojure.core/float doubled)))]
     second-effect))

(defn- scheduled-three-maps []
  (let [options {:dtype :float :target-device :ocl:0
                 :array-types {'x :float 'tmp :float 'middle :float 'out :float}
                 :scalar-types {'n :long}}
        typed (frontend/form->program three-map-source options)
        envelope (route/program-envelope typed)]
    (:form (segop-lower/segop-lower-pass envelope options))))

(defn- scheduled-scalar-gap []
  (let [options {:dtype :float :target-device :ocl:0
                 :array-types {'x :float 'tmp :float 'out :float}
                 :scalar-types {'n :long}}
        typed (frontend/form->program scalar-gap-source options)
        envelope (route/program-envelope typed)]
    (:form (segop-lower/segop-lower-pass envelope options))))

(defn- reason-of [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest contiguous-equations-form-one-exact-semantic-source-graph
  (let [scheduled (scheduled-three-maps)
        equations (subvec (:equations scheduled) 0 2)
        {:keys [algorithm body graph]} (equation-graph/make-for-equations scheduled equations)]
    (is (= [0 1] (get-in body [:attributes :equation-region])))
    (is (= [0 1] (get-in (soac/facts algorithm) [:attributes :equation-region])))
    (is (= 2 (count (soac/equations algorithm))))
    (is (= (:results (peek equations)) (soac/outputs algorithm)))
    (is (= 2 (count (:nodes graph))))
    (is (= #{'x} (set (map :id (:inputs graph)))))
    (is (= #{'middle} (set (map :id (:outputs graph)))))
    (is (= #{'tmp} (set (map :id (:temporaries graph)))))))

(deftest symbolic-map-read-refines-an-opaque-caller-buffer-capacity
  (let [scheduled (scheduled-three-maps)
        {:keys [graph]} (equation-graph/make-for-equation
                         scheduled (first (:equations scheduled)))
        input (first (filter #(= 'x (:id %)) (:inputs graph)))
        certificate (get-in graph [:nodes 0 :read-capacity-certificate])]
    (is (= 'n (:elements input))
        "the active map domain proves x[0..n), and graph binding enforces that capacity")
    (is (= :zero-based-dense-read-spans (:kind certificate)))
    (is (= {'x 'n} (:requirements certificate))
        "the checked graph retains the exact proof target for later address projection")))

(deftest known-symbolic-map-capacity-retains-a-checked-range-boundary
  (let [source '(let* [result (raster.par/pmap i n float
                                            (float (+ (aget x i) (float capacity))))]
                     result)
        options {:dtype :float :target-device :ocl:0 :array-types {'x :float}
                 :scalar-types {'n :long 'capacity :long}
                 :values {'x (av/tensor {:dtype :float :shape '[capacity]})
                          'capacity (av/tensor {:dtype :long :shape []})}}
        typed (frontend/form->program source options)
        scheduled (:form (segop-lower/segop-lower-pass (route/program-envelope typed) options))
        {:keys [graph]} (equation-graph/make-for-equation scheduled (first (:equations scheduled)))
        node (first (:nodes graph))
        check #(precondition/check! (:preconditions graph)
                                    (partial graph-call/resolve-integer
                                             {'n {:type :long :value 3}
                                              'capacity {:type :long :value %}}))]
    (is (= 'capacity (:elements (first (:inputs graph)))))
    (is (= [{:expression 'capacity :op :>= :value 'n}] (:preconditions graph)))
    (is (= :certified-index-expression
           (get-in (map-reads/validate-and-project-addresses (:operation node) node graph)
                   [:address-projection :kind])))
    (is (= :kernel-precondition-failed (reason-of #(check 2))))
    (is (true? (check 4)))))

(deftest scalar-fold-graphs-derive-core-read-minima-without-matrix-scheduling
  (let [options {:dtype :float :target-device :ocl:0
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m :long 'n :long 'k :long 'batch :long}}
        graph-for (fn [axes body extra-options]
                    (let [source (list 'let* ['result (list 'raster.par/contract 'C axes [['l 'k]] body)]
                                       'result)
                          options (merge options extra-options)
                          typed (frontend/form->program source options)
                          scheduled (:form (segop-lower/segop-lower-pass
                                            (route/program-envelope typed) options))]
                      (:graph (equation-graph/make-for-equation
                               scheduled (first (:equations scheduled))))))]
    (doseq [body ['(* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))
                 '(* (aget A (+ (* i k) l)) (aget B (+ (* j k) l)))
                 '(* (aget A (+ (* l m) i)) (aget B (+ (* l n) j)))
                 '(* (aget A (+ (* l m) i)) (aget B (+ (* j k) l)))]]
      (let [graph (graph-for [['i 'm] ['j 'n]] body {})
            extents (into {} (map (juxt :id :elements)) (:inputs graph))]
        (is (extent/equivalent? (launch/product 'm 'k) (extents 'A)))
        (is (extent/equivalent? (launch/product 'n 'k) (extents 'B)))
        (is (nil? (:abi graph)) "target-neutral read minima must not invent a callable ABI")))
    (let [graph (graph-for [['i 'm] ['j 'n]]
                           '(* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))
                           {:values {'A (av/tensor {:dtype :float :shape [2]})}})
          values {'m {:type :long :value 3} 'n {:type :long :value 5}
                  'k {:type :long :value 7}}]
      (is (= 2 (:elements (first (filter #(= 'A (:id %)) (:inputs graph))))))
      (is (= :kernel-precondition-failed
             (reason-of #(precondition/check! (:preconditions graph)
                                               (partial graph-call/resolve-integer values))))))
    (let [typed (frontend/form->program
                 '(let* [result (raster.par/contract C [[i m] [j n]] [[l k]]
                                                        (* (aget A (+ (* i k) l))
                                                           (aget B (+ (* l n) j))))]
                    result) options)
          equation (first (soac/equations typed))]
      (is (some? (projection/segmented-reduce-core-read-requirements typed equation)))
      (doseq [[facet value] [[:representation {:kind :quantized :scheme :q4-k}]
                             [:logical-layout {:order [0]}]
                             [:sharding {:axis 0}]]]
        (is (nil? (projection/segmented-reduce-core-read-requirements
                   (with-meta
                     (list* (first typed) (assoc-in (soac/facts typed) [:values 'A facet] value)
                            (nnext typed)) (meta typed)) equation)))))))

(deftest known-static-map-capacity-is-not-increased-by-a-read-requirement
  (let [options {:dtype :float :target-device :ocl:0 :array-types {'x :float}
                 :scalar-types {'n :long}
                 :values {'x (av/tensor {:dtype :float :shape [3]})}}
        typed (frontend/form->program
               '(let* [result (raster.par/pmap i n float (aget x i))] result) options)
        scheduled (:form (segop-lower/segop-lower-pass (route/program-envelope typed) options))
        graph (:graph (equation-graph/make-for-equation scheduled (first (:equations scheduled))))
        check #(precondition/check! (:preconditions graph)
                                    (partial graph-call/resolve-integer {'n {:type :long :value %}}))]
    (is (= 3 (:elements (first (:inputs graph)))))
    (is (= [{:expression 3 :op :>= :value 'n}] (:preconditions graph)))
    (is (true? (check 3)))
    (is (= :kernel-precondition-failed (reason-of #(check 4))))))

(deftest address-projection-recomputes-the-proof-and-requires-graph-capacity
  (let [scheduled (scheduled-three-maps)
        {:keys [graph]} (equation-graph/make-for-equation
                         scheduled (first (:equations scheduled)))
        node (first (:nodes graph))
        operation (:operation node)
        projected (map-reads/validate-and-project-addresses operation node graph)
        forged-node (assoc-in node [:read-capacity-certificate :requirements 'x] 1)
        forged-graph (assoc graph :nodes [forged-node])
        undersized (assoc-in graph [:inputs 0 :elements] 1)
        runtime-guarded (assoc undersized :preconditions
                               [{:expression 1 :op :>= :value 'n}])]
    (is (= :certified-index-expression (get-in projected [:address-projection :kind])))
    (is (= :map-address-certificate-mismatch
           (reason-of #(map-reads/validate-and-project-addresses
                        operation forged-node forged-graph))))
    (is (= :map-address-certificate-capacity
           (reason-of #(map-reads/validate-and-project-addresses
                        operation node undersized))))
    (is (= :certified-index-expression
           (get-in (map-reads/validate-and-project-addresses
                    operation node runtime-guarded)
                   [:address-projection :kind]))
        "a checked graph-boundary capacity contract can discharge a dynamic read proof")))

(deftest equation-region-must-be-an-exact-contiguous-slice
  (let [scheduled (scheduled-three-maps)
        equations (:equations scheduled)]
    (is (= :scheduled-equation-region
           (reason-of #(equation-graph/make-for-equations
                        scheduled [(first equations) (peek equations)]))))
    (is (= :scheduled-equation-region
           (reason-of #(equation-graph/make-for-equations scheduled []))))))

(deftest pure-scalar-gap-is-hoisted-into-the-region-proof-prefix
  (let [scheduled (scheduled-scalar-gap)
        numerical (filterv (comp seq :operations) (:equations scheduled))
        {:keys [body graph]} (equation-graph/make-for-equations scheduled numerical)]
    (is (= 2 (count numerical)))
    (is (= 3 (count (:equations body))))
    (is (true? (get-in body [:equations 0 :attributes :host-only])))
    (is (= (mapv :id numerical) (mapv :id (subvec (:equations body) 1))))
    (is (= 2 (count (:nodes graph))))))

(deftest scheduled-graph-checks-each-retained-host-algorithm-once
  (let [scheduled (scheduled-scalar-gap)
        numerical (filterv (comp seq :operations) (:equations scheduled))
        {:keys [algorithm body graph]} (equation-graph/make-for-equations scheduled numerical)
        host-equation (first (:equations body))
        host-algorithm (:algorithm host-equation)
        result (first (:results host-equation))
        checks (atom 0)
        original soac/validate!]
    (with-redefs [soac/validate!
                  (fn [candidate]
                    (when (identical? host-algorithm candidate) (swap! checks inc))
                    (original candidate))]
      (is (= graph (equation-graph/make algorithm body)))
      (is (= 1 @checks) "the checked program supplies the prefix's semantic proof")
      (is (= graph (equation-graph/make algorithm body)))
      (is (= 2 @checks) "a later graph construction independently checks the prefix"))
    (is (thrown? clojure.lang.ExceptionInfo
                 (equation-graph/make
                  algorithm
                  (assoc-in body [:equations 0 :algorithm]
                            (apply list (assoc (vec host-algorithm) 1
                                               (assoc (soac/facts host-algorithm)
                                                      :inputs ['missing]))))))
        "an invalid retained host algorithm still fails during program validation")
    (is (= :scheduled-equation-prefix
           (reason-of #(equation-graph/make
                        algorithm (assoc-in body [:values result :shape] [1]))))
        "prefix-specific scalar shape checks are not covered by a valid algorithm alone")))

(deftest later-global-extent-becomes-a-bindable-buffer-capacity
  (let [scheduled (scheduled-three-maps)
        scheduled (-> scheduled
                      (assoc-in [:values 'later-extent] (get-in scheduled [:values 'n]))
                      (assoc-in [:values 'x :shape] ['later-extent]))
        {:keys [body graph]} (equation-graph/make-for-equation
                              scheduled (first (:equations scheduled)))
        graph-input (first (filter #(= 'x (:id %)) (:inputs graph)))]
    (is (= '[(extent x)] (get-in body [:values 'x :shape])))
    (is (= 'n (:elements graph-input))
        "the local access proof replaces a later global shape with this graph's bindable minimum")))

(deftest earlier-equation-does-not-capture-a-later-global-buffer-extent
  ;; group-norm's final dense map normalizes `batch*channel-span` after three product equations.
  ;; The program-wide value table once leaked that later extent into every earlier graph ABI,
  ;; producing a forward reference at backend reconstruction time.
  (let [report (pipeline/compile-report #'nn/group-norm-jvp-dx
                                        :target-device :ocl:0 :dtype :float)]
    (is (= :typed-soac (get-in report [:route :source-dialect])))
    (is (= {:kernel-body 4} (get-in report [:emission :routes])))
    (is (empty? (get-in report [:route :declines])))))

(deftest segmented-plan-public-capacity-keeps-descriptor-owned-minima
  (let [program (indexed-plan-program)
        equation (first (:equations program))
        plan (:algorithm equation)
        descriptors (conj (:operands plan) (:output plan))
        capacity-program (reduce (fn [program {:keys [id]}]
                                   (assoc-in program [:values id :shape] [(list 'extent id)]))
                                 program descriptors)
        graph (:graph (equation-graph/make-for-plan-equation capacity-program equation))]
    (is (= (mapv swr/descriptor-launch-elements (:operands plan))
           (mapv :elements (:inputs graph))))
    (is (= (swr/descriptor-launch-elements (:output plan))
           (get-in graph [:outputs 0 :elements])))
    (is (= :segmented-plan-equation-boundary
           (reason-of #(equation-graph/make-for-plan-equation
                        (assoc-in capacity-program [:values (first (:results equation)) :shape]
                                  [(list 'extent (first (:results equation)))])
                        equation)))
        "only backing storage, never the logical result, may use self-capacity")
    (doseq [damaged [(assoc-in capacity-program [:values (get-in plan [:operands 0 :id]) :dtype] :double)
                     (assoc-in capacity-program [:values (get-in plan [:operands 0 :id]) :representation]
                               {:kind :quantized :scheme :q4-k})
                     (assoc-in capacity-program [:values (get-in plan [:operands 0 :id]) :shape]
                               '[(extent unrelated)])]]
      (is (= :segmented-plan-equation-boundary
             (reason-of #(equation-graph/make-for-plan-equation damaged equation)))))))

(deftest segmented-plan-forms-one-exact-schedule-neutral-source-graph
  (let [parallel-program (indexed-plan-program)
        equation (first (:equations parallel-program))
        plan (:algorithm equation)
        {:keys [body graph]} (equation-graph/make-for-plan-equation
                              parallel-program equation)
        node (first (:nodes graph))]
    (is (= [equation] (:equations body)))
    (is (= (:dialect parallel-program) (:dialect body)))
    (is (= plan (:operation node)))
    (is (= (mapv :id (:operands plan)) (mapv :id (:inputs graph))))
    (is (= [(get-in plan [:output :id])] (mapv :id (:outputs graph))))
    (is (= (mapv swr/descriptor-launch-elements (:operands plan))
           (mapv :elements (:inputs graph))))
    (is (= (swr/descriptor-launch-elements (:output plan))
           (get-in graph [:outputs 0 :elements])))
    (is (= (set (map :id (:scalars graph))) (:scalar-uses node)))
    (is (every? #(= :long (:dtype %)) (:scalars graph)))
    (is (nil? (:abi graph)))
    (is (nil? (:arguments graph)))))

(deftest segmented-plan-graph-reconstructs-from-its-exact-scheduled-body
  (let [parallel-program (indexed-plan-program)
        equation (first (:equations parallel-program))
        plan (:algorithm equation)
        semantic (equation-graph/make-for-plan-equation parallel-program equation)
        source-graph (:graph semantic)
        scheduled-operation
        (indexed-body/schedule-reference-for-node
         plan (first (:nodes source-graph)) source-graph {:subgroup-size 16})
        scheduled-equation (assoc equation :operations [scheduled-operation])
        scheduled-program (assoc parallel-program :dialect :scheduled-parallel
                                  :equations [scheduled-equation])
        reconstructed (equation-graph/make-for-plan-equation
                       scheduled-program scheduled-equation)]
    (is (= [scheduled-operation]
           (get-in reconstructed [:body :equations 0 :operations])))
    (is (= source-graph (:graph reconstructed)))
    (is (= :segmented-weighted-reduction
           (get-in reconstructed [:graph :provenance :source-dialect])))))

(deftest segmented-plan-graph-preserves-public-scalar-width
  (let [parallel-program (indexed-plan-program)
        runtime-id (first (filter symbol?
                                  (get-in parallel-program
                                          [:equations 0 :algorithm :runtime-parameters])))
        parallel-program (assoc-in parallel-program [:values runtime-id :dtype] :int)
        {:keys [graph]} (equation-graph/make-for-plan-equation
                         parallel-program (first (:equations parallel-program)))]
    (is (= :int (:dtype (first (filter #(= runtime-id (:id %)) (:scalars graph))))))))

(deftest segmented-plan-graph-retains-only-an-exact-output-allocation-contract
  (let [parallel-program (indexed-plan-program)
        equation (first (:equations parallel-program))
        descriptor (get-in equation [:algorithm :output])
        allocation {:destination (:id descriptor) :source-binding-id 4
                    :extent (:elements descriptor) :initialization :zero
                    :dtype (:dtype descriptor)}
        initialized (assoc-in parallel-program [:attributes :allocations] [allocation])
        {:keys [body graph]} (equation-graph/make-for-plan-equation initialized equation)]
    (is (= [allocation] (get-in body [:attributes :allocations])))
    (is (= allocation (get-in graph [:attributes :output-allocation])))
    (is (= :segmented-plan-output-allocation
           (reason-of #(equation-graph/make-for-plan-equation
                        (assoc-in initialized [:attributes :allocations 0 :extent] 1)
                        equation))))
    (is (= :segmented-plan-output-allocation
           (reason-of #(equation-graph/make-for-plan-equation
                        (update-in initialized [:attributes :allocations] conj allocation)
                        equation))))
    (is (= :segmented-plan-equation-boundary
           (reason-of #(equation-graph/make-for-plan-equation
                        parallel-program (assoc equation :operations [:not-the-plan])))))))
