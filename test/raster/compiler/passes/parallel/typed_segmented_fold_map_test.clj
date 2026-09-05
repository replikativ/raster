(ns raster.compiler.passes.parallel.typed-segmented-fold-map-test
  (:require [clojure.string :as str]
            [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.passes.parallel.segfoldmap-body :as fold-body]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.par :as par]))

(def ^:private source
  '(let* [effect
          (raster.par/segmented-fold-map!
           [out] [[segment nsegments]] index width
           [[maximum -1.0E38 :float width
             (clojure.core/max
              maximum
              (clojure.core/aget values
                                 (clojure.core/+ (clojure.core/* segment width) index)))]
            [denominator 0.0 :float width
             (clojure.core/+
              denominator
              (Math/exp
               (clojure.core/-
                (clojure.core/aget values
                                   (clojure.core/+ (clojure.core/* segment width) index))
                maximum)))]]
           [(clojure.core//
             (Math/exp
              (clojure.core/-
               (clojure.core/aget values
                                  (clojure.core/+ (clojure.core/* segment width) index))
               maximum))
             denominator)])]
         effect))

(defn- scheduled-operation []
  (let [program (:program
                 (route/attempt source :float {'values :float 'out :float}
                                {:scalar-types {'nsegments :int 'width :int}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'values :float 'out :float}
                             :scalar-types {'nsegments :int 'width :int}}))]
    (first (:operations (first (:equations scheduled))))))

(defn- decline-rule
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (when (fold-body/declined? exception)
        (:missing-rule (ex-data exception))))))

(defn- reason-of
  [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception
         (:reason (ex-data exception)))))

(defn- operation-tree
  [operations]
  (mapcat (fn [operation]
            (cons operation
                  (cond
                    (instance? raster.compiler.ir.kernel_body.ForLoop operation)
                    (operation-tree (:operations operation))
                    (instance? raster.compiler.ir.kernel_body.IfRegion operation)
                    (concat (operation-tree (:then-operations operation))
                            (operation-tree (:else-operations operation)))
                    :else [])))
          operations))

(deftest interpreted-segmented-fold-map-preserves-dependent-fold-order
  (let [values (float-array [1.0 3.0, 2.0 2.0])
        out (float-array 4)]
    (par/segmented-fold-map!
     [out] [[segment 2]] index 2
     [[maximum -1.0E38 :float 2
       (max maximum (aget values (+ (* segment 2) index)))]
      [denominator 0.0 :float 2
       (+ denominator (Math/exp (- (aget values (+ (* segment 2) index)) maximum)))]]
     [(/ (Math/exp (- (aget values (+ (* segment 2) index)) maximum)) denominator)])
    (is (< (Math/abs (- (aget out 0) 0.11920292)) 1.0e-6))
    (is (< (Math/abs (- (aget out 1) 0.8807971)) 1.0e-6))
    (is (< (Math/abs (- (aget out 2) 0.5)) 1.0e-6))
    (is (< (Math/abs (- (aget out 3) 0.5)) 1.0e-6))))

(deftest dependent-folds-are-first-class-typed-soac
  (let [program (frontend/form->program
                 source {:dtype :float :array-types {'values :float 'out :float}
                         :scalar-types {'nsegments :int 'width :int}})
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)]
    (is (= 'segmented-fold-map (:kind operation)))
    (is (= 2 (count (:folds operation))))
    (is (= '[nsegments values width] (:inputs (dialect/facts program))))
    (is (= '[out] (dialect/physical-results program equation)))
    (is (= :int (:dtype (get-in (dialect/facts program) [:values 'width])))
        "launch dimensions retain their declared ABI representation")))

(deftest portable-schedule-is-a-capped-grid-stride-loop-over-three-ordered-loops
  (let [operation (scheduled-operation)
        {:keys [kernel-body]} (fold-body/lower
                               operation {:array-types {'values :float 'out :float}
                                          :scalar-types {'nsegments :int 'width :int}})
        segment-loop (first (:operations kernel-body))
        first-segment (-> kernel-body :indices last :expression)
        segment-axis-computes (take (count (segop/seg-space-segment-dims (:space operation)))
                                    (:operations segment-loop))
        source-workgroup (get-in operation [:grid :block-size])
        source-cap (first (rest (get-in operation [:grid :num-blocks])))
        loops (filterv #(and (instance? raster.compiler.ir.kernel_body.ForLoop %)
                             (not= :segment-grid-stride (get-in % [:attributes :role])))
                       (operation-tree (:operations kernel-body)))]
    (is (= :segment-grid-stride (get-in segment-loop [:attributes :role])))
    (is (every? #(instance? raster.compiler.ir.kernel_body.IndexCompute %)
                segment-axis-computes)
        "grid-stride segment axes are lexically recomputed as typed index SSA")
    (is (= (mapv :name (segop/seg-space-segment-dims (:space operation)))
           (mapv :id segment-axis-computes)))
    (is (instance? raster.compiler.ir.kernel_body.IndexExpr (:step segment-loop)))
    (is (= :add (:op first-segment)))
    (is (= :mul (:op (first (:arguments first-segment)))))
    (is (every? #(and (instance? raster.compiler.ir.kernel_body.IndexCast %)
                      (= :long (:dtype %))
                      (= :exact (:overflow %)))
                (concat (:arguments (first (:arguments first-segment)))
                        [(second (:arguments first-segment))]))
        "group, workgroup, and lane indices widen before address arithmetic")
    (is (= [source-workgroup] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [(launch/minimum
             (body/index-cast source-cap :long :exact)
             (launch/ceil-div
              (body/index-cast (launch/runtime-value 'nsegments) :long :exact)
              (body/index-cast source-workgroup :long :exact)))]
           (get-in kernel-body [:launch :group-count])))
    (is (zero? (get-in kernel-body [:launch :shared-memory-bytes])))
    (is (= 3 (count loops)) "maximum, denominator, then the final dense map")
    (is (= [:ordered :ordered :ordered]
           (mapv #(get-in % [:attributes :association]) loops)))
    (is (str/includes? (pr-str (second loops)) "maximum")
        "the second fold consumes the completed first fold")
    (is (empty? (:iter-args (last loops))))
    (is (= :final-map (get-in (last loops) [:attributes :role])))
    (is (= ['values] (mapv :buffer (:stable-reads kernel-body))))))

(deftest ordered-fold-map-has-one-complete-scheduled-body-certificate
  (let [operation (scheduled-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'nsegments :int 'width :int}}
        lowered (fold-body/lower operation options)
        scheduled (fold-body/schedule operation options)
        artifact (segop-opencl/generate-segfoldmap-kernel
                  operation :target-dialect :opencl-portable
                  :array-types (:array-types options) :scalar-types (:scalar-types options))]
    (is (scheduled-body/scheduled-kernel-body? scheduled))
    (is (identical? operation (:source scheduled)))
    (is (= (vec (concat (:arrays lowered) (:outputs lowered) (:scalars lowered)
                        [(:segment-count lowered)]))
           (:arguments scheduled))
        "the private _nseg parameter binds to the derived segment count")
    (is (= (scheduled-body/derive-scalar-bindings (:body scheduled) (:arguments scheduled))
           (:scalar-bindings scheduled)))
    (is (every? #(= :identity (:conversion %)) (:scalar-bindings scheduled)))
    (is (= {:kind :segmented-fold-map
            :uses [{:value 'values :access :read} {:value 'out :access :write}]
            :association :ordered}
           (:effects scheduled)))
    (is (= {:kind :segfoldmap-body-lowering
            :launch-rank 1
            :segment-parallelism :grid-stride-independent
            :association :ordered
            :source-grid (:grid operation)
            :aliasing :no-write-alias}
           (:legality scheduled)))
    (is (= {:mode :exact :policy :declaration-order :reassociation :none}
           (:numerics scheduled)))
    (is (= :scheduled-kernel-body (get-in artifact [:provenance :lowering])))
    (let [certificate (get-in artifact [:provenance :scheduled-operation])]
      (is (identical? operation (:source certificate)))
      (is (= '[values out nsegments width _nseg] (mapv :name (:abi artifact))))
      (is (= [:input :output :scalar :scalar :scalar] (mapv :kind (:abi artifact))))
      (is (= [:float :float :int :int :long] (mapv :dtype (:abi artifact))))
      (is (= (:arguments certificate) (:arguments artifact)))
      (is (= (scheduled-body/realized-launch certificate) (:launch artifact)))
      (is (= artifact
             (scheduled-body/validate-artifact-projection! certificate artifact))))))

(deftest one-certified-schedule-emits-for-all-c-family-targets
  (let [operation (scheduled-operation)]
    (doseq [[target expected]
            [[:opencl-portable :opencl-c]
             [:cuda :cuda-c]
             [:hip :hip-cpp]]]
      (testing (name target)
        (let [kernel (segop-opencl/generate-segfoldmap-kernel
                      operation :target-dialect target
                      :array-types {'values :float 'out :float}
                      :scalar-types {'nsegments :int 'width :int})]
          (is (artifact/kernel-artifact? kernel))
          (is (= expected (:target kernel)))
          (is (= :scheduled-kernel-body (get-in kernel [:provenance :lowering])))
          (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
          (is (= kernel
                 (scheduled-body/validate-artifact-projection!
                  (get-in kernel [:provenance :scheduled-operation]) kernel)))
          (is (= :no-write-alias (get-in kernel [:abi 0 :aliasing])))
          (is (str/includes? (:source kernel) "foldmap_segment +="))
          (is (if (contains? #{:cuda :hip} target)
                (str/includes? (:source kernel) "gridDim.x")
                (str/includes? (:source kernel) "get_num_groups(0)")))
          (is (every? #(<= (int %) 127) (:source kernel))))))))

(deftest graph-emission-closes-the-fold-map-certificate-over-node-scalars-and-effects
  (let [operation (scheduled-operation)
        scalar-types {'nsegments :int 'width :int}
        graph (kgraph/from-segops
               [operation]
               {:inputs #{'values} :outputs #{'out} :dtype :float
                :buffer-specs {'values {:dtype :float
                                        :elements (launch/product 'nsegments 'width)}
                               'out {:dtype :float
                                     :elements (launch/product 'nsegments 'width)}}
                :scalars [(kgraph/scalar 'nsegments :int)
                          (kgraph/scalar 'width :int)]})
        emitted (segop-opencl/generate-kernel-graph
                 graph :target-dialect :opencl-portable
                 :array-types {'values :float 'out :float}
                 :scalar-types scalar-types)
        artifact (get-in emitted [:nodes 0 :operation])
        certificate (get-in artifact [:provenance :scheduled-operation])]
    (is (identical? operation (:source certificate)))
    (is (= certificate (get-in artifact [:attributes :scheduled-kernel-body])))
    (is (= certificate
           (fold-body/validate-against-node! certificate (first (:nodes graph)) graph)))
    (is (= #{'nsegments 'width} (get-in graph [:nodes 0 :scalar-uses])))
    (is (= artifact
           (scheduled-body/validate-artifact-projection! certificate artifact)))))

(deftest malformed-fold-map-contracts-decline-before-a-refinement-witness
  (let [operation (scheduled-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'nsegments :int 'width :int}}
        result (first (:map-results operation))
        cases
        [[:aliasing-contract (assoc operation :aliasing :may-alias)]
         [:storage-contract (update operation :inputs conj 'out)]
         [:result-contract (assoc operation :outputs [] :dtypes [] :map-results [])]
         [:result-contract (assoc operation :dtypes [])]
         [:result-contract (assoc operation :map-results [])]
         [:mapped-extent (assoc operation :extent 'other-width)]
         [:fold-association (assoc-in operation [:folds 0 :association] :associative)]
         [:distinct-outputs (assoc operation
                                   :outputs ['out 'out]
                                   :dtypes [:float :float]
                                   :map-results [result result])]]]
    (doseq [[rule malformed] cases]
      (testing (name rule)
        (is (= rule (decline-rule #(fold-body/schedule malformed options))))))))

(deftest source-kernel-grid-is-an-exact-capped-zero-shared-memory-contract
  (let [operation (scheduled-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'nsegments :int 'width :int}}
        source-workgroup (get-in operation [:grid :block-size])]
    (is (= :source-grid
           (decline-rule #(fold-body/schedule
                           operation (assoc options :workgroup-size (inc source-workgroup))))))
    (is (= :source-grid-count
           (decline-rule #(fold-body/schedule
                           (assoc-in operation [:grid :num-blocks] 1) options))))
    (is (= :source-grid-shared-memory
           (decline-rule #(fold-body/schedule
                           (assoc-in operation [:grid :shared-mem-bytes] 4) options))))
    (let [large-cap-operation
          (assoc-in operation [:grid :num-blocks]
                    (list 'min Integer/MAX_VALUE
                          (nth (get-in operation [:grid :num-blocks]) 2)))
          first-segment (-> (fold-body/schedule large-cap-operation options)
                            :body :indices last :expression)
          multiplied (first (:arguments first-segment))]
      (is (= :mul (:op multiplied)))
      (is (every? #(= :long (:dtype %)) (:arguments multiplied))
          "even an adversarial positive source cap widens grid operands before multiplication"))))

(deftest graph-certification-closes-fold-map-pointer-dtypes-extents-and-schedule
  (let [operation (scheduled-operation)
        scalar-types {'nsegments :int 'width :int}
        elements (launch/product 'nsegments 'width)
        make-graph
        (fn [input-dtype input-elements]
          (kgraph/from-segops
           [operation]
           {:inputs #{'values} :outputs #{'out} :dtype :float
            :buffer-specs {'values {:dtype input-dtype :elements input-elements}
                           'out {:dtype :float :elements elements}}
            :scalars [(kgraph/scalar 'nsegments :int)
                      (kgraph/scalar 'width :int)]}))
        graph (make-graph :float elements)
        scheduled (fold-body/schedule
                   operation {:array-types {'values :float 'out :float}
                              :scalar-types scalar-types})
        node (first (:nodes graph))
        first-parameter (first (get-in scheduled [:body :parameters]))
        forged-shape
        (-> scheduled
            (assoc-in [:body :parameters 0 :shape] [1])
            (assoc-in [:body :parameters 0 :layout]
                      (layout/row-major [1] (:dtype first-parameter))))
        forged-launch
        (assoc-in scheduled [:body :launch :group-count] [1])
        forged-workgroup
        (assoc-in scheduled [:body :launch :workgroup-size] [1])
        forged-shared
        (assoc-in scheduled [:body :launch :shared-memory-bytes] 4)
        forged-schedule
        (assoc-in scheduled [:body :schedule :workgroup-size] 1)]
    (is (= scheduled (fold-body/validate-against-node! scheduled node graph)))
    (is (= :segfoldmap-storage-extent
           (reason-of #(fold-body/validate-against-node! forged-shape node graph))))
    (let [wrong-extent (make-graph :float (launch/sum elements 1))]
      (is (= :segfoldmap-storage-extent
             (reason-of #(fold-body/validate-against-node!
                          scheduled (first (:nodes wrong-extent)) wrong-extent)))))
    (let [forged-attribute-graph
          (kgraph/from-segops
           [operation]
           {:inputs #{'values} :outputs #{'out} :dtype :float
            :buffer-specs {'values {:dtype :float :elements 1}
                           'out {:dtype :float :elements 1}}
            :scalars [(kgraph/scalar 'nsegments :int)
                      (kgraph/scalar 'width :int)]
            :attributes {:derived-storage-scalars {'nsegments 1 'width 1}}})]
      (is (= :segfoldmap-storage-extent
             (reason-of #(fold-body/validate-against-node!
                          scheduled (first (:nodes forged-attribute-graph))
                          forged-attribute-graph)))
          "descriptive graph attributes cannot forge the storage proof"))
    (let [wrong-dtype (make-graph :double elements)]
      (is (= :segfoldmap-storage-dtype
             (reason-of #(fold-body/validate-against-node!
                          scheduled (first (:nodes wrong-dtype)) wrong-dtype)))))
    (is (= :segfoldmap-schedule-source
           (reason-of #(fold-body/validate-against-node! forged-launch node graph))))
    (doseq [forged [forged-workgroup forged-shared forged-schedule]]
      (is (= :segfoldmap-schedule-source
           (reason-of #(fold-body/validate-against-node! forged node graph)))))))

(deftest graph-storage-replays-the-closed-host-scalar-extent-prefix
  (let [typed (:program (route/attempt source :float {'values :float 'out :float}
                                      {:scalar-types {'nsegments :int 'width :int}}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0
                                 :array-types {'values :float 'out :float}
                                 :scalar-types {'nsegments :int 'width :int}}))
        numerical (some #(when (seq (:operations %)) %) (:equations scheduled))
        extent 'derived-extent
        extent-expression (with-meta '(clojure.core/* segments elements)
                            {:raster.type/tag 'long})
        values (-> (:values scheduled)
                   (assoc extent (av/tensor {:dtype :long :shape []}))
                   (update 'values assoc :shape [(list 'value extent)])
                   ;; Opaque graph extent identities are resolver-owned leaves, not nested
                   ;; graph-scalar expressions.  Keep one alongside the expanded input extent.
                   (update 'out assoc :shape [(list 'extent 'input)]))
        scalar-algorithm
        (dialect/make
         (dialect/default-program-facts
          {:values values :inputs '[nsegments width]
           :equations {extent (dialect/default-equation-facts)}})
         [(list '= extent [extent]
                (list 'scalar {:dtypes [:long]} '[nsegments width]
                      (dialect/lambda-form '[segments elements]
                                           [extent-expression])))]
         [extent])
        scalar-equation
        (parallel-program/->ProgramEquation
         extent [:test :derived-extent] nil '[nsegments width] [extent]
         scalar-algorithm [] #{} {:source :test} {:host-only true})
        equations [scalar-equation numerical]
        closed (assoc scheduled
                      :values values
                      :inputs (parallel-program/infer-inputs equations)
                      :equations equations
                      :outputs (:results numerical)
                      :effects (reduce set/union #{} (map :effects equations)))
        graph (equation-graph/make (:algorithm numerical) closed)
        values-buffer (some #(when (= 'values (:id %)) %) (:inputs graph))
        graph-scalar-references
        (reduce set/union #{}
                (map #(into #{}
                             (filter (fn [value] (or (symbol? value) (keyword? value))))
                             (launch/expression-references (:elements %)))
                     (concat (:inputs graph) (:outputs graph) (:temporaries graph))))]
    (is values-buffer)
    (is (= #{'nsegments 'width}
           (launch/expression-references (:elements values-buffer)))
        "the graph allocation replays the preceding typed scalar definition")
    (is (not-any? #{extent}
                  (launch/expression-references (:elements values-buffer))))
    (is (= #{'nsegments 'width}
           (set (map :id (:scalars graph)))))
    (is (= (list 'extent 'input)
           (:elements (some #(when (= 'out (:id %)) %) (:outputs graph))))
        "a compound resolver identity remains opaque without inventing a GraphScalar")
    (is (set/subset? graph-scalar-references (set (map :id (:scalars graph))))
        "every graph-storage extent closes over an explicit integral graph scalar")
    (is (nil? (get-in graph [:attributes :derived-storage-scalars]))
        "the derivation remains a retained program proof, never a graph assertion")
    (let [proof-values (update values 'out assoc :shape [(list 'value extent)])
          proof-closed (assoc closed :values proof-values)
          proof-graph (equation-graph/make (:algorithm numerical) proof-closed)
          operation (first (:operations numerical))
          certificate (fold-body/schedule
                       operation {:array-types {'values :float 'out :float}
                                  :scalar-types {'nsegments :int 'width :int}})
          emitted (segop-opencl/generate-kernel-graph
                   proof-graph
                   :target-dialect :opencl-portable
                   :array-types {'values :float 'out :float}
                   :scalar-types {'nsegments :int 'width :int}
                   :scheduled-equation-body proof-closed)]
      (is (= certificate
             (fold-body/validate-against-node!
              certificate (first (:nodes proof-graph)) proof-graph proof-closed)))
      (is (artifact/kernel-artifact? (get-in emitted [:nodes 0 :operation]))
          "graph emission threads the exact retained equation proof into FoldMap"))))

(deftest host-prefix-does-not-hide-later-numerical-inputs
  (let [two-map-source
        '(let* [y (raster.par/pmap i n float
                                   (clojure.core/* 2.0 (clojure.core/aget x i)))
                z (raster.par/pmap j n float
                                   (clojure.core/+ (clojure.core/aget y j)
                                                   (clojure.core/aget b j)))
                _effect (dotimes [k n]
                          (clojure.core/aset
                           acc 0 (clojure.core/+ (clojure.core/aget acc 0)
                                                 (clojure.core/aget y k))))]
               z)
        options {:dtype :float :target-device :ocl:0
                 :array-types {'x :float 'b :float 'acc :float}
                 :scalar-types {'n :long}}
        typed (:program (route/attempt two-map-source :float (:array-types options)
                                       {:scalar-types (:scalar-types options)}))
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        numerical-equations (:equations scheduled)
        host-id 'graph-proof-host
        values (assoc (:values scheduled) host-id (av/tensor {:dtype :long :shape []}))
        host-algorithm
        (dialect/make
         (dialect/default-program-facts
          {:values values :inputs []
           :equations {host-id (dialect/default-equation-facts)}})
         [(list '= host-id [host-id]
                (list 'scalar {:dtypes [:long]} []
                      (dialect/lambda-form [] [1])))]
         [host-id])
        host-equation
        (parallel-program/->ProgramEquation
         host-id [:test :graph-proof-host] nil [] [host-id]
         host-algorithm [] #{} {:source :test} {:host-only true})
        algorithms (mapv :algorithm numerical-equations)
        numerical-inputs (parallel-program/infer-inputs numerical-equations)
        combined-algorithm
        (dialect/make
         (dialect/default-program-facts
          {:values values
           :inputs numerical-inputs
           :equations (apply merge (map (comp :equations dialect/facts) algorithms))
           :effects (reduce set/union #{} (map (comp :effects dialect/facts) algorithms))})
         (vec (mapcat dialect/equations algorithms))
         (:outputs scheduled))
        equations (into [host-equation] numerical-equations)
        closed (assoc scheduled
                      :values values
                      :inputs (parallel-program/infer-inputs equations)
                      :equations equations)
        graph (equation-graph/make combined-algorithm closed)]
    (is (= 2 (count numerical-equations)))
    (is (some #{'b} numerical-inputs)
        "the second equation introduces an external input absent from the first")
    (is (= 2 (count (:nodes graph))))
    (is (some #{'b} (map :id (:inputs graph))))))

(deftest scalar-captures-retain-their-typed-abi
  (let [scaled-source
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[segment nsegments]] index width
                 [[sum 0.0 :float width
                   (clojure.core/+ sum
                                   (clojure.core/aget values
                                                      (clojure.core/+ (clojure.core/* segment width)
                                                                      index)))]]
                 [(clojure.core/* scale sum)])]
               effect)
        program (:program
                 (route/attempt scaled-source :float {'values :float 'out :float}
                                {:scalar-types {'nsegments :int 'width :int 'scale :float}}))
        operation (-> (segop-lower/segop-lower-pass
                       program {:dtype :float :target-device :ocl:0
                                :array-types {'values :float 'out :float}
                                :scalar-types {'nsegments :int 'width :int 'scale :float}})
                      :form :equations first :operations first)
        kernel (segop-opencl/generate-segfoldmap-kernel
                operation :target-dialect :opencl-portable
                :array-types {'values :float 'out :float}
                :scalar-types {'nsegments :int 'width :int 'scale :float})
        scale-parameter (some #(when (= 'scale (:name %)) %) (:abi kernel))]
    (is (= :float (:dtype scale-parameter)))
    (is (str/includes? (:source kernel) "float scale"))))

(deftest a-fold-cannot-reference-a-future-accumulator
  (let [bad
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[segment nsegments]] index width
                 [[first 0.0 :float width (clojure.core/+ first future)]
                  [future 0.0 :float width (clojure.core/+ future 1.0)]]
                 [first])]
               effect)]
    (try
      (frontend/form->program
       bad {:dtype :float :array-types {'out :float}
            :scalar-types {'nsegments :int 'width :int}})
      (is false "future fold dependencies must fail validation")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-segmented-fold-map-fold-closure
               (:reason (ex-data exception))))))))
