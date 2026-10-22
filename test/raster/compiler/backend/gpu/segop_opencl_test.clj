(ns raster.compiler.backend.gpu.segop-opencl-test
  "SegRed OpenCL kernel generation — pins #55: aget lowering must NOT be
   ns-sensitive. A parametric-array kernel's reads arrive DEVIRTUALIZED
   ((.invk aget-impl arr i) with :raster.op/original), a typed-array kernel's
   as bare clojure.core/aget — both must lower to the subscript `arr[i]`.
   Before the fix SegRed skipped normalize-array-prims (SegMap didn't) and its
   let*-rewrap dropped the metadata, so the devirtualized shape fell through
   to a broken gpufn_aget helper call (qlinear-k) while the typed shape
   emitted x[i] (decoder-gpu)."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kernel-call]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.backend.gpu.kernel-body-target :as kernel-target]
            [raster.compiler.backend.gpu.emitted-graph-interface :as emitted-interface]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.equation-first :as equation-first]
            [raster.dl.nn :as dl-nn]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.compiler.passes.parallel.map-read-requirements :as map-reads]
            [raster.compiler.passes.parallel.segmap-capacity-fixture :as capacity-fixture]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.passes.parallel.segscan-body :as segscan-body]
            [raster.compiler.passes.parallel.segstencil-body :as segstencil-body]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.backend.gpu.segop-opencl :as sg]
            [raster.compiler.backend.gpu.segmap-retirement-fixture :as retirement]))

(deftest graph-array-shapes-retain-exact-zero-without-inventing-other-capacities
  (let [project (ns-resolve 'raster.compiler.backend.gpu.segop-opencl
                            'graph-array-shapes)
        graph (fn [id elements]
                (kgraph/make
                 {:inputs [(kgraph/buffer id :float elements :device :input)]
                  :outputs [] :nodes []}))]
    (is (= {'zero [0]} (project (graph 'zero 0))))
    (is (= {} (project (graph 'negative -1))))
    (is (= {} (project (graph 'self 'self))))
    (is (= {} (project (graph 'unknown '(opaque-capacity n)))))))

(deftest scalar-reduction-storage-alias-requires-the-exact-closed-projection
  (let [source '(let* [^long extent (clojure.core/long
                                    ^long (clojure.core/* (clojure.core/long n) (clojure.core/long 1)))
                       result (raster.par/reduce acc 0.0 i extent
                                                (+ acc (clojure.core/aget a i)))] result)
        options {:dtype :double :array-types {'a :double} :scalar-types {'n :long}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (:algorithm (last (:equations typed)))
        body (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm body)
        node (first (:nodes graph))
        operation (:operation node)
        ;; This fixture schedules a graph node directly, so supply the retained long type
        ;; of the host-prefix extent as the production graph emitter does.
        certificate (segred-body/schedule operation (first (:outputs operation))
                                          (assoc-in options [:scalar-types 'extent] :long))]
    (is (seq (equation-graph/validated-storage-scalars graph algorithm body)))
    (is (= certificate (segred-body/validate-against-node! certificate node graph algorithm body)))
    (is (not (contains? (:attributes graph) :scalar-definitions)))
    (doseq [[candidate closed-algorithm closed-body]
            [[(update graph :inputs #(mapv (fn [b] (assoc b :elements 7)) %)) algorithm body]
             [graph algorithm nil]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (segred-body/validate-against-node!
                    certificate node candidate closed-algorithm closed-body))
          "an independent graph capacity or incomplete proof context cannot authorize the alias"))))

(deftest map-storage-retains-exact-shape-and-closes-its-scalar-abi
  (let [source '(let* [^long extent (clojure.core/long
                                    ^long (clojure.core/* (clojure.core/long rows)
                                                         (clojure.core/long width)))
                       result (raster.par/pmap i extent float (clojure.core/aget a i))] result)
        options {:dtype :float :array-types {'a :float}
                 :scalar-types {'rows :long 'width :long}
                 :values {'a (av/tensor {:dtype :float :shape ['rows 'width]})}}
        typed (:program (typed-route/attempt source :float {'a :float} options))
        body (:form (segop-lower/segop-lower-pass typed options))
        equation (last (:equations body))
        algorithm (:algorithm equation)
        graph (equation-graph/make algorithm body)
        emit #(get-in (sg/generate-kernel-graph
                       % :scheduled-equation-algorithm algorithm :scheduled-equation-body body)
                      [:nodes 0 :operation])
        artifact (emit graph)
        parameters (get-in artifact [:attributes :kernel-body :parameters])
        input (first (filter #(= :input (:kind %)) parameters))
        scheduled (get-in artifact [:attributes :scheduled-kernel-body])
        node (first (:nodes graph))
        context (equation-graph/input-storage-context body equation (first (:operations equation)))]
    (is (= [(get-in graph [:inputs 0 :elements])] (:shape input)))
    (is (= (layout/row-major (:shape input) :float) (:layout input)))
    (is (= '[a result rows width extent] (:arguments artifact)))
    (is (= #{'rows 'width 'extent}
           (scheduled-body/graph-scalar-dependencies scheduled node graph)))
    (doseq [tamper [(fn [parameter]
                     (let [shape [(klaunch/product 2 (first (:shape parameter)))]]
                       (assoc parameter :shape shape :layout (layout/row-major shape :float))))
                   #(assoc-in % [:layout :strides] [2])]]
      (let [candidate (update-in scheduled [:body :parameters]
                                 #(mapv (fn [parameter]
                                          (if (= :output (:kind parameter))
                                            (tamper parameter) parameter)) %))]
        (is (= :scheduled-kernel-body-output-storage
               (try (scheduled-body/graph-scalar-dependencies candidate node graph) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
    (is (contains? (:scalar-definitions context) 'extent))
    (is (thrown? clojure.lang.ExceptionInfo
                 (equation-graph/input-storage-context
                  body equation (assoc (first (:operations equation)) :id :foreign))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (emit (assoc-in graph [:inputs 0 :elements] 1)))
        "a stale graph capacity cannot be certified by descriptive alias metadata")
    (is (thrown? clojure.lang.ExceptionInfo
                 (sg/generate-kernel-graph graph :scheduled-equation-algorithm algorithm))
        "the alias needs the complete exact closed context")
    (let [unrelated (assoc-in graph [:attributes :derived-storage-scalars] {'extent 1})
          ignored (emit unrelated)]
      (is (= parameters (get-in ignored [:attributes :kernel-body :parameters])))
      (is (= (:arguments artifact) (:arguments ignored)))
      (is (= (get-in artifact [:attributes :kernel-body :operations])
             (get-in ignored [:attributes :kernel-body :operations]))
          "descriptive aliases cannot change retained storage or its computation"))))

(deftest partial-storage-extent-is-an-honest-scalar-slot-under-a-compound-bound
  (let [source '(let* [^long extent (clojure.core/long
                                    ^long (clojure.core/* (clojure.core/long rows)
                                                         (clojure.core/long width)))
                       result (raster.par/pmap i extent float
                                (clojure.core/aget a (clojure.core/quot i width)))] result)
        options {:dtype :float :array-types {'a :float}
                 :scalar-types {'rows :long 'width :long}
                 :values {'a (av/tensor {:dtype :float :shape ['rows]})}}
        typed (:program (typed-route/attempt source :float {'a :float} options))
        program (:form (segop-lower/segop-lower-pass typed options))
        equation (last (:equations program))
        graph (equation-graph/make (:algorithm equation) program)
        emitted (sg/generate-kernel-graph
                 graph :scheduled-equation-algorithm (:algorithm equation)
                 :scheduled-equation-body program)
        artifact (get-in emitted [:nodes 0 :operation])
        scheduled (get-in artifact [:attributes :scheduled-kernel-body])
        input (first (filter #(= :input (:kind %))
                             (get-in artifact [:attributes :kernel-body :parameters])))]
    (is (= ['rows] (:shape input)))
    (is (= :long (:dtype (first (filter #(= 'rows (:id %))
                                       (get-in artifact [:attributes :kernel-body :parameters]))))))
    (is (= '[a result rows width extent] (:arguments artifact)))
    (is (emitted-equation/emitted-equation?
         (emitted-equation/make (:algorithm equation) program emitted)))
    (is (= 'rows (:value (first (filter #(= 'rows (:parameter %))
                                       (:scalar-bindings scheduled))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (sg/generate-kernel-graph
                  (update graph :scalars #(vec (remove (fn [scalar] (= 'rows (:id scalar))) %)))
                  :scheduled-equation-algorithm (:algorithm equation)
                  :scheduled-equation-body program)))))

(deftest output-only-storage-scalars-require-exact-independent-graph-storage
  (let [operation (segop/->SegMap
                   1903 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(clojure.core/aget input i) nil #{'input} #{'output} #{}
                   (segop/->KernelGrid 1 32 0) :float 'output nil)
        elements (klaunch/product (klaunch/runtime-value 'rows) (klaunch/runtime-value 'chunks))
        scheduled (segmap-body/schedule
                   operation {:workgroup-size 32 :scalar-types {'n :long 'rows :long 'chunks :long}
                              :array-types {'input :float 'output :float}
                              :array-shapes {'input ['n] 'output [elements]}})
        node (kgraph/->ScheduledKernel
              :storage-only-output operation
              (mapv #(kgraph/->ValueUse (:value %) (:access %)) (get-in scheduled [:effects :uses]))
              #{'n} [])
        graph (kgraph/make
               {:inputs [(kgraph/buffer 'input :float 'n :device :input)]
                :outputs [(kgraph/buffer 'output :float elements :device :output)]
                :scalars (mapv #(kgraph/scalar % :long) '[n rows chunks]) :nodes [node]})
        check #(scheduled-body/graph-scalar-dependencies %1 node %2)
        artifact (kernel-target/emit-artifact "storage_output" scheduled :opencl-portable)
        emitted (kgraph/map-operations graph (constantly artifact))
        finalize #(emitted-interface/finalize! %1 :opencl-c {} %2)]
    (is (= #{'n}
           (scheduled-body/certified-physical-scalar-dependencies
            node (assoc graph :scalars nil) artifact))
        "a legacy graph cannot authorize represented storage-only dependencies")
    (is (nil? (scheduled-body/certified-physical-scalar-dependencies
               (assoc node :scalar-uses nil) (assoc graph :scalars nil) artifact))
        "legacy ABI-only projection does not manufacture a source dependency proof")
    (doseq [scalars [(:scalars graph) []]]
      (is (= :scheduled-kernel-body-node-scalars
             (try (scheduled-body/certified-physical-scalar-dependencies
                   (assoc node :scalar-uses nil) (assoc graph :scalars scalars) artifact) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (let [same-deps-source (assoc operation :scalars #{'rows 'chunks})
          same-deps-node (assoc node :operation same-deps-source :scalar-uses #{'n 'rows 'chunks})
          same-deps-graph (assoc graph :nodes [same-deps-node])
          same-deps-artifact (kernel-target/emit-artifact
                              "same_deps_storage_output" (assoc scheduled :source same-deps-source)
                              :opencl-portable)]
      (is (= #{'n 'rows 'chunks}
             (scheduled-body/certified-physical-scalar-dependencies
              same-deps-node same-deps-graph same-deps-artifact)))
      (doseq [changed [(assoc-in same-deps-node [:operation :id] :foreign)
                       (assoc-in same-deps-node [:operation :lambda]
                                 '(clojure.core/+ 1.0 (clojure.core/aget input i)))]]
        (is (= :scheduled-kernel-body-source
               (try (scheduled-body/certified-physical-scalar-dependencies
                     changed (assoc same-deps-graph :nodes [changed]) same-deps-artifact) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
    (is (= #{'n 'rows 'chunks} (check scheduled graph)))
    (is (= #{'n 'rows 'chunks}
           (check scheduled (assoc-in graph [:outputs 0 :elements] (klaunch/product 1 elements)))))
    (is (= scheduled (segmap-body/validate-static-graph-capacities! scheduled node graph)))
    (is (= #{'n 'rows 'chunks} (get-in (finalize emitted graph) [:nodes 0 :scalar-uses])))
    (is (= #{'n 'rows 'chunks}
           (get-in (sg/generate-kernel-graph
                    graph :scheduled-bodies {(:id node) scheduled}
                    :target-dialect :opencl-portable) [:nodes 0 :scalar-uses])))
    (is (= :scheduled-kernel-body-source
           (try (finalize emitted (assoc-in graph [:nodes 0 :operation :id] :foreign)) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= :scheduled-kernel-body-source
           (try (sg/generate-kernel-graph
                 (assoc-in graph [:nodes 0 :operation :id] :foreign)
                 :scheduled-bodies {(:id node) scheduled} :target-dialect :opencl-portable) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (let [rogue (-> scheduled
                    (update-in [:body :parameters] conj
                               (kernel-body/->KernelParameter 'rogue :scalar :long [] nil nil :parameter))
                    (update :arguments conj 'rogue)
                    (update :scalar-bindings conj {:parameter 'rogue :value 'rogue :dtype :long
                                                  :kernel-dtype :long :conversion :identity}))
          rogue-artifact (kernel-target/emit-artifact "rogue_storage_output" rogue :opencl-portable)
          source (update graph :scalars conj (kgraph/scalar 'rogue :long))]
      (is (= :scheduled-kernel-body-node-scalars
             (try (finalize (kgraph/map-operations source (constantly rogue-artifact)) source) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= :scheduled-kernel-body-node-scalars
             (try (sg/generate-kernel-graph
                   source :scheduled-bodies {(:id node) rogue} :target-dialect :opencl-portable) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (doseq [candidate [(assoc-in graph [:outputs 0 :elements] (klaunch/product 2 elements))
                      (assoc-in graph [:outputs 0 :dtype] :double)]]
      (is (= :scheduled-kernel-body-output-storage
             (try (check scheduled candidate) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (let [candidate (update-in scheduled [:body :parameters]
                               #(mapv (fn [parameter]
                                        (if (= :output (:kind parameter))
                                          (assoc-in parameter [:layout :strides] [2]) parameter)) %))]
      (is (= :scheduled-kernel-body-output-storage
             (try (check candidate graph) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (segmap-body/validate-static-graph-capacities!
                  scheduled node (update graph :scalars #(vec (remove (fn [scalar] (= 'rows (:id scalar))) %))))))))

(deftest portable-map-retains-selected-math-consent
  (let [operation (segop/->SegMap
                   903 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   (with-meta '(raster.math/tanh (clojure.core/aget input i))
                     {:raster.type/tag 'float})
                   nil #{'input} #{'output} #{}
                   (segop/->KernelGrid 1 32 0) :float 'output nil)
        options {:array-types {'input :float 'output :float} :workgroup-size 32}
        policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        ordinary (segmap-body/schedule operation options)
        selected (segmap-body/schedule operation (assoc options :scalar-math policy))]
    (is (not (contains? (:numerics ordinary) :scalar-math)))
    (is (= policy (get-in selected [:numerics :scalar-math])))
    (is (= #{:float} (kernel-body/required-scalar-dtypes
                      (get-in ordinary [:body :operations]))))
    (is (= #{:float :double} (kernel-body/required-scalar-dtypes
                              (get-in selected [:body :operations]))))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (let [descriptor {:execution {:scalar-dtype-support {:double :supported}}}
            artifact (sg/generate-scheduled-segmap-kernel
                      operation :array-types (:array-types options) :scalar-math policy
                      :target-descriptor descriptor :target-dialect dialect)]
        (is (= policy (get-in artifact [:provenance :scheduled-operation :numerics :scalar-math]))))
      (try
        (sg/generate-scheduled-segmap-kernel
         operation :array-types (:array-types options) :scalar-math policy
         :target-descriptor {} :target-dialect dialect)
        (is false "map wrapper must preserve caller hardware admission context")
        (catch clojure.lang.ExceptionInfo error
          (is (= :kernel-body-target-math-capability (:reason (ex-data error)))))))))

(deftest portable-map-empty-extent-has-a-masked-valid-launch
  (let [operation (segop/->SegMap
                   902 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(clojure.core/aget input i) nil #{'input} #{'output} #{}
                   (segop/->KernelGrid 1 32 0) :float 'output nil)
        kernel-body (:kernel-body
                     (segmap-body/lower operation
                                        {:array-types {'input :float 'output :float}
                                         :workgroup-size 32}))
        spec (:launch kernel-body)]
    (doseq [[extent groups] [[0 1] [1 1] [32 1] [33 2]]]
      (is (= [groups] (:group-count (klaunch/realize spec
                                                    (fn [_] extent))))))
    (is (= :map-active (:id (first (:masks kernel-body)))))
    (is (= (kernel-body/predicate :lt 'i
                                 (kernel-body/index-cast '_n_bound :long :exact))
           (first (:predicates (first (:masks kernel-body))))))))

(deftest exact-map-bound-storage-reuses-its-existing-typed-slot
  (let [operation (segop/->SegMap
                   2902 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(clojure.core/aget input i) nil #{'input} #{'output} #{}
                   (segop/->KernelGrid 1 32 0) :float 'output nil)
        candidate (segmap-body/schedule operation
                                        {:array-types {'input :float 'output :float}
                                         :workgroup-size 32})
        parameters (get-in candidate [:body :parameters])]
    (is (= ['_n_bound] (:shape (first parameters))) "n is a proved read span, not a default capacity")
    (is (= ['_n_bound] (mapv :id (filter #(= :scalar (:kind %)) parameters))))
    (is (= '[input output n] (:arguments candidate)))
    (is (= :int (:dtype (last parameters))))))

(deftest map-access-proof-declines-negative-and-indirect-coordinates
  (doseq [expression ['(aget a (- i 1))
                      '(aget a (int (aget b j)))]]
    (let [graph (capacity-fixture/inferred-graph expression)
          operation (get-in graph [:nodes 0 :operation])]
      (is (nil? (segmap-body/static-read-requirements
                 operation {:dtype :float :array-types {'a :float 'b :float 'C :float}}))))))

(deftest public-outer-product-derives-unknown-input-capacities
  (doseq [[expression expected]
          [['(* (aget a i) (aget b j)) {'a 4 'b 3}]
           ['(+ (aget a i) (aget a j)) {'a 4}]
           ['(let* [k (+ i j)] (aget a k)) {'a 6}]]]
    (let [graph (capacity-fixture/inferred-graph expression)
          emitted (sg/generate-kernel-graph graph)
          artifact (get-in emitted [:nodes 0 :operation])]
      (is (= expected (into {} (map (juxt :id :elements)) (:inputs graph))))
      (is (= :kernel-body (kart/emission-route artifact)))
      (is (= (into {} (map (fn [[id n]] [id [n]])) expected)
             (into {} (comp (filter #(= :input (:kind %))) (map (juxt :id :shape)))
                   (get-in artifact [:attributes :kernel-body :parameters])))))))

(deftest public-symbolic-map-retains-read-minima-after-address-projection
  (let [compilation (equation-first/compile #'dl-nn/batch-norm
                                           {:target :cuda:0 :dtype :float})
        parameters (mapcat #(get-in % [:attributes :kernel-body :parameters])
                           (:kernels compilation))
        beta (filter #(= 'beta (:id %)) parameters)]
    (is (= 4 (count (:kernels compilation))))
    (is (= 1 (count beta)))
    (is (= '[features] (:shape (first beta))))
    (is (= 'features (:minimum-elements (first beta))))
    (is (every? #(= :kernel-body (kart/emission-route %)) (:kernels compilation)))))

(deftest read-minima-reject-an-unrelated-source-address-projection
  (let [initial (capacity-fixture/graph)
        index (get-in initial [:nodes 0 :operation :space :dims 0 :name])
        operation (assoc (get-in initial [:nodes 0 :operation])
                         :space (segop/make-seg-space index 'n) :scalars #{'n}
                         :scalar-region
                         {:locals [] :result (list '* (list 'aget 'a index) (list 'aget 'b index))})
        certificate (map-reads/symbolic-read-certificate operation {})
        node (assoc (first (:nodes initial)) :operation operation
                    :scalar-uses #{'n} :read-capacity-certificate certificate)
        graph (kgraph/make {:inputs [(kgraph/buffer 'a :float 'n :device :input)
                                     (kgraph/buffer 'b :float 'n :device :input)]
                            :outputs [(kgraph/buffer 'C :float 'n :device :output)]
                            :scalars [(kgraph/scalar 'n :long)] :nodes [node]})
        projected (map-reads/validate-and-project-addresses (:operation node) node graph)
        altered (assoc-in projected [:scalar-region :result] (float 0.0))]
    (is (= :zero-based-dense-read-spans (:kind certificate)))
    (is (= :read-capacity-source
           (try (segmap-body/schedule
                 altered {:array-types {'a :float 'b :float 'C :float}
                          :array-shapes {'a '[n] 'b '[n] 'C '[n]}
                          :scalar-types {'n :long}
                          :graph-node node :kernel-graph graph})
                nil
                (catch clojure.lang.ExceptionInfo error (:missing-rule (ex-data error))))))))

(deftest typed-map-preserves-independent-static-graph-capacities
  (doseq [a-capacity [4 8]
          dialect [:opencl-intel :cuda :hip]]
    (let [graph (capacity-fixture/graph a-capacity)
          emitted (sg/generate-kernel-graph graph :target-dialect dialect)
          artifact (get-in emitted [:nodes 0 :operation])
          body (get-in artifact [:attributes :kernel-body])
          shapes (into {} (map (juxt :id :shape)) (:parameters body))
          certificate (get-in artifact [:attributes :scheduled-kernel-body])]
      (is (= :kernel-body (kart/emission-route artifact)))
      (is (= [a-capacity] (shapes 'a)))
      (is (= [3] (shapes 'b)))
      (is (= [12] (shapes 'C)))
      (is (= [a-capacity 3] (mapv :elements (:inputs emitted))))
      (is (= {'a 4 'b 3}
             (into {} (comp (filter #(= :input (:kind %)))
                            (map (juxt :id :minimum-elements))) (:parameters body))))
      (is (= 12 (get-in body [:attributes :extent])))
      (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact))))))

(deftest typed-map-transports-symbolic-storage-instead-of-iteration-width
  (let [extent (klaunch/product 2 2)
        graph (assoc-in (capacity-fixture/graph) [:inputs 0 :elements] extent)
        artifact (get-in (sg/generate-kernel-graph graph) [:nodes 0 :operation])
        shapes (into {} (map (juxt :id :shape))
                     (get-in artifact [:attributes :kernel-body :parameters]))
        candidate (get-in artifact [:attributes :scheduled-kernel-body])
        node (first (:nodes graph))]
    (is (= [extent] (shapes 'a)))
    (is (= [3] (shapes 'b)))
    (is (= [12] (shapes 'C)))
    (is (identical? candidate
                    (scheduled-body/validate-dense-input-storage! candidate node graph))
        "the shared helper rebinds each expression through parameter-to-argument identities")
    (doseq [corrupted [(-> candidate
                          (assoc-in [:body :parameters 0 :shape] [1])
                          (assoc-in [:body :parameters 0 :layout] (layout/row-major [1] :float)))
                      (assoc-in candidate [:body :parameters 0 :layout :strides] [2])]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"capacity differs"
                           (segmap-body/validate-static-graph-capacities! corrupted node graph))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no retained storage capacity"
                         (sg/generate-kernel-graph
                          (assoc-in graph [:inputs 0 :elements] 'a))))
    (let [failure (try (sg/generate-kernel-graph
                        (assoc-in graph [:inputs 0 :elements] 0))
                       nil
                       (catch clojure.lang.ExceptionInfo error error))]
      (is (instance? clojure.lang.ExceptionInfo failure))
      (is (= [0] (:shape (ex-data failure)))))))

(deftest map-storage-capacity-does-not-replace-its-read-obligation
  (doseq [capacity [0 1 2 3]]
    (try
      (sg/generate-kernel-graph (assoc-in (capacity-fixture/graph) [:inputs 0 :elements] capacity))
      (is false "a quotient-index input needs four elements, independently of the output width")
      (catch clojure.lang.ExceptionInfo failure
        (is (= :input-capacity (:missing-rule (ex-data failure))))
        (is (= 4 (:required-elements (ex-data failure))))
        (is (= capacity (:buffer-elements (ex-data failure)))))))
  (let [graph (assoc-in (capacity-fixture/graph) [:inputs 0 :elements] '(extent a))
        artifact (get-in (sg/generate-kernel-graph graph) [:nodes 0 :operation])
        candidate (get-in artifact [:attributes :scheduled-kernel-body])
        bind (fn [capacity]
               (let [buffers {'a (float-array capacity) 'b (float-array 3) 'C (float-array 12)}
                     arguments (mapv (fn [slot argument]
                                       (if (= :scalar (:kind slot))
                                         {:type (:kernel-dtype slot) :value argument}
                                         (get buffers argument)))
                                     (:abi artifact) (:arguments artifact))]
                 (kernel-call/validate-retained-input-capacities!
                  artifact arguments (fn [value _] (java.lang.reflect.Array/getLength value)))))]
    (is (identical? artifact (bind 4)))
    (is (identical? artifact (bind 8)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"smaller than" (bind 3)))
    (doseq [tamper [#(dissoc % :minimum-elements)
                    #(assoc % :minimum-elements 0) #(assoc % :minimum-elements 1)]]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"read minimum differs"
           (segmap-body/validate-static-graph-capacities!
            (update-in candidate [:body :parameters 0] tamper) (first (:nodes graph)) graph)))))
  (let [operation (segop/->SegMap
                   :unknown-empty-read (segop/make-seg-space 'i 4)
                   (segop/->SegLevel :thread :virtual)
                   '(clojure.core/aget a (clojure.core/aget indices i))
                   nil #{'a 'indices} #{'out} #{}
                   (segop/->KernelGrid 1 32 0) :float 'out nil)]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"requires a proved zero read domain"
         (sg/generate-scheduled-segmap-kernel
          operation :array-types {'a :float 'indices :int 'out :float}
          :array-shapes {'a [0] 'indices [4] 'out [4]})))
    (let [empty (sg/generate-scheduled-segmap-kernel
                 (assoc operation :space (segop/make-seg-space 'i 0))
                 :array-types {'a :float 'indices :int 'out :float}
                 :array-shapes {'a [0] 'indices [0] 'out [0]})]
      (is (= [0 0] (mapv :minimum-elements
                         (filter #(= :input (:kind %))
                                 (get-in empty [:attributes :kernel-body :parameters]))))))))

(deftest raw-primary-inout-keeps-its-known-written-prefix
  (let [operation (segop/->SegMap
                   :primary-inout (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(inc (clojure.core/aget out i)) nil #{'out} #{'out} #{'n}
                   (segop/->KernelGrid 1 32 0) :float 'out nil)
        emit #(sg/generate-scheduled-segmap-kernel
               % :array-types {'out :float} :scalar-types {'n :int})
        ;; Exercise precedence independently: even a smaller input-read requirement is not
        ;; permission to discard the primary output's exact per-index write domain.
        artifact (with-redefs [map-reads/independent-read-requirements (fn [& _] {'out 1})]
                   (emit operation))]
    (is (= '[out n _n_bound] (mapv :name (:abi artifact))))
    (is (= '[out n n] (:arguments artifact)))
    (is (= ['_n_bound] (get-in artifact [:attributes :kernel-body :parameters 0 :shape])))
    (is (= '_n_bound (get-in artifact [:attributes :kernel-body :parameters 0 :minimum-elements])))
    (let [physical (sg/generate-scheduled-segmap-kernel
                    operation :array-types {'out :float} :scalar-types {'n :int}
                    :array-shapes {'out [8]})]
      (is (= [8] (get-in physical [:attributes :kernel-body :parameters 0 :shape])))
      (is (= (layout/row-major [8] :float)
             (get-in physical [:attributes :kernel-body :parameters 0 :layout])))
      (is (identical? physical
                      (kernel-call/validate-retained-input-capacities!
                       physical [(float-array 8) {:type :int :value 4} {:type :int :value 4}]
                       (fn [value _] (java.lang.reflect.Array/getLength value)))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"smaller than"
                           (kernel-call/validate-retained-input-capacities!
                            physical [(float-array 8) {:type :int :value 9} {:type :int :value 9}]
                            (fn [value _] (java.lang.reflect.Array/getLength value))))))
    (try
      (kernel-call/validate-retained-input-capacities!
       artifact [(float-array 3) {:type :int :value 4} {:type :int :value 4}]
       (fn [value _] (java.lang.reflect.Array/getLength value)))
      (is false "three backing elements cannot cover four primary writes")
      (catch clojure.lang.ExceptionInfo failure
        (is (= :kernel-body-buffer-capacity (:reason (ex-data failure))))
        (is (= 4 (:required-elements (ex-data failure))))
        (is (= 3 (:buffer-elements (ex-data failure))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"writable results may only read"
                         (emit (assoc operation :lambda '(inc (clojure.core/aget out 0)))))
        "constant cross-lane reads do not bypass the independent alias contract")))

(deftest ordered-effects-cannot-smuggle-a-primary-cross-lane-read
  (let [operation (assoc
                   (segop/->SegMap
                    :ordered-primary-read (segop/make-seg-space 'i 4)
                    (segop/->SegLevel :thread :virtual)
                    nil nil #{'out} #{'out} #{}
                    (segop/->KernelGrid 1 32 0) :float 'out nil)
                   :scalar-region
                   {:locals [] :iteration-order :sequential
                    :effects ['(clojure.core/aset out i
                                (clojure.core/aget out (clojure.core/+ i 1)))]})
        emit (fn [capacity]
               (sg/generate-scheduled-segmap-kernel
                operation :array-types {'out :float} :array-shapes {'out [capacity]}))
        result-operation (-> operation
                             (assoc :scalar-region nil
                                    :lambda '(clojure.core/aget out (clojure.core/+ i 1))))]
    ;; Ordered effects have explicit stores and no primary result. The malformed mixed
    ;; representation must decline before constructing a pointer contract (or popping forms).
    (doseq [capacity [4 5]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"map result has no scalar expression"
                           (emit capacity))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"writable results may only read"
                         (sg/generate-scheduled-segmap-kernel
                          result-operation :array-types {'out :float} :array-shapes {'out [5]})))))

(deftest secondary-indexed-store-does-not-inherit-the-primary-write-minimum
  (let [operation (segop/->SegMap
                   :secondary-indexed-store (segop/make-seg-space 'i 'n)
                   (segop/->SegLevel :thread :virtual)
                   '(do (clojure.core/aset histogram (clojure.core/aget bins i) (float 1))
                        (clojure.core/aget values i))
                   nil #{'bins 'values} #{'histogram 'result} #{'n}
                   (segop/->KernelGrid 1 32 0) :float 'result nil)
        artifact (sg/generate-scheduled-segmap-kernel
                  operation :array-types {'bins :int 'values :float 'histogram :float 'result :float}
                  :array-shapes {'bins [12] 'values [12] 'histogram [3] 'result [12]}
                  :scalar-types {'n :int})
        parameters (into {} (map (juxt :id identity))
                         (get-in artifact [:attributes :kernel-body :parameters]))]
    (is (= '_n_bound (:minimum-elements (parameters 'result))))
    (is (not (contains? (parameters 'histogram) :minimum-elements)))
    (is (= [3] (:shape (parameters 'histogram))))))

(deftest graph-input-backing-extent-preserves-exact-storage-identity
  (let [graph (assoc-in (capacity-fixture/graph) [:inputs 0 :elements] '(extent a))
        artifact (get-in (sg/generate-kernel-graph graph) [:nodes 0 :operation])
        candidate (get-in artifact [:attributes :scheduled-kernel-body])
        node (first (:nodes graph))
        shape [(klaunch/runtime-value '(extent a))]]
    (is (= shape (get-in candidate [:body :parameters 0 :shape])))
    (is (identical? candidate (scheduled-body/validate-dense-input-storage! candidate node graph)))
    (doseq [wrong-shape [[(klaunch/runtime-value '(extent b))] [1]]]
      (let [wrong (-> candidate
                      (assoc-in [:body :parameters 0 :shape] wrong-shape)
                      (assoc-in [:body :parameters 0 :layout] (layout/row-major wrong-shape :float)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"capacity differs"
                             (scheduled-body/validate-dense-input-storage! wrong node graph)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no retained storage capacity"
                         (sg/generate-kernel-graph
                          (assoc-in graph [:inputs 0 :elements] '(extent b)))))))

(deftest map-capacity-projection-rejects-drift-without-changing-the-algorithm
  (let [graph (capacity-fixture/graph)
        emit #(get-in (sg/generate-kernel-graph %) [:nodes 0 :operation])
        small (emit graph)
        larger (emit (capacity-fixture/graph 8))
        source #(str/replace (:source %) (:kernel-name %) "map_capacity")
        candidate (get-in small [:attributes :scheduled-kernel-body])
        corrupted (-> candidate
                      (assoc-in [:body :parameters 0 :shape] [12])
                      (assoc-in [:body :parameters 0 :layout] (layout/row-major [12] :float)))]
    (is (= (source small) (source larger)))
    (is (= (:arguments small) (:arguments larger)))
    (is (= (:launch small) (:launch larger)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"capacity differs"
                         (segmap-body/validate-static-graph-capacities!
                          corrupted (first (:nodes graph)) graph)))
    (try
      (segmap-body/validate-static-graph-capacities!
       (assoc-in candidate [:body :parameters 0 :layout] nil) (first (:nodes graph)) graph)
      (is false "a missing layout must fail structural admission")
      (catch clojure.lang.ExceptionInfo e
        (is (= "kernel buffer parameter requires a named layout" (.getMessage e)))
        (is (= {:layout nil} (ex-data e)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exact graph node"
                         (segmap-body/validate-static-graph-capacities!
                          candidate (assoc (first (:nodes graph)) :id :foreign-node) graph)))
    (let [index (first (keep-indexed #(when (= :result (:role %2)) %1)
                                   (get-in candidate [:body :parameters])))]
      (is (some? index))
      (doseq [tamper [#(dissoc % :minimum-elements) #(assoc % :minimum-elements 1)]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"write minimum differs"
                             (segmap-body/validate-static-graph-capacities!
                              (update-in candidate [:body :parameters index] tamper)
                              (first (:nodes graph)) graph)))))))

(deftest emitted-graph-interface-preserves-in-place-buffer-direction
  (let [operation (segop/->SegMap
                   901 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(inc (clojure.core/aget output i)) nil #{'output} #{'output} #{'n}
                   (segop/->KernelGrid 1 32 0) :float 'output nil)
        scheduled (kgraph/from-segops [operation]
                                      {:inputs #{'output} :outputs #{'output} :dtype :float
                                       ;; This fixture proves direction, not unknown-capacity admission.
                                       :buffer-specs {'output {:dtype :float :elements 'n}}
                                       :scalars [(kgraph/scalar 'n :int)]})
        emitted (sg/generate-kernel-graph
                 scheduled :array-types {'output :float} :scalar-types {'n :int})]
    (is (= ['output 'n] (:arguments emitted)))
    (is (= [:inout :scalar] (mapv :kind (:abi emitted))))
    (let [parameters (get-in emitted [:nodes 0 :operation :attributes :kernel-body :parameters])]
      (is (= 1 (count (filter #(= :inout (:kind %)) parameters))))
      (is (= ['n] (:shape (first parameters)))
          "graph storage expression remains independent of its private iteration-bound slot"))
    (is (= :inout (get-in emitted [:inputs 0 :role])))))

(deftest two-phase-reduction-graph-emits-its-explicit-scheduled-dataflow
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double}
                 :scalar-types {'n :long}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled)
        emitted (sg/generate-kernel-graph
                 graph :array-types {'a :double} :scalar-types {'n :long})
        [phase-one phase-two] (mapv :operation (:nodes emitted))
        source-nodes (:nodes graph)
        certificates (mapv #(get-in % [:provenance :scheduled-operation])
                           [phase-one phase-two])
        partials (first (:inputs (second (mapv :operation (:nodes graph)))))
        result-id (:id (first (:outputs graph)))
        temporary-specs (graph-call/temporary-specs
                         emitted {'n {:type :long :value 1025}})]
    (is (= [(kgraph/scalar 'n :long)] (:scalars graph)))
    (is (= (:scalars graph) (:scalars emitted)))
    (let [slot (->> (map vector (:abi emitted) (:arguments emitted))
                    (some (fn [[slot argument]] (when (= 'n argument) slot))))]
      (is (= :long (:dtype slot))
          "the scheduled graph supplies the public logical dtype")
      (is (= :long (:kernel-dtype slot))
          "the public graph carrier retains its declared width independently of node specialization"))
    (doseq [artifact [phase-one phase-two]
            :let [[slot argument] (last (map vector (:abi artifact) (:arguments artifact)))]]
      (is (= :int (:kernel-dtype slot))
          "the target-private bound retains its physical scalar representation")
      (is (= #{'n} (klaunch/expression-references argument))
          "a target-private derived bound closes over the public logical scalar"))
    (is (= 2 (count (:nodes emitted))))
    (is (every? kart/kernel-artifact? [phase-one phase-two]))
    (is (= [] (:dependencies (first source-nodes))))
    (is (= [(:id (first source-nodes))] (:dependencies (second source-nodes))))
    (is (= [:block-local :cross-block]
           (mapv #(get-in % [:effects :phase]) [phase-one phase-two])))
    (is (= [[{:value 'a :access :read} {:value partials :access :write}]
            [{:value partials :access :read} {:value result-id :access :write}]]
           (mapv #(get-in % [:effects :uses]) [phase-one phase-two])))
    (is (= (mapv :operation source-nodes) (mapv :source certificates))
        "each scheduled body retains its exact scalar SegRed phase")
    (doseq [[certificate node] (map vector certificates source-nodes)]
      (is (= certificate (segred-body/validate-against-node! certificate node graph))))
    (is (= [(first (get-in phase-one [:attributes :kernel-body :launch :group-count]))]
           (-> phase-one :attributes :kernel-body :parameters second :shape))
        "the partial output is closed over the actual checked group-count expression")
    (is (= [1] (-> phase-two :attributes :kernel-body :parameters second :shape)))
    (is (= [:checked-range :checked-range]
           (mapv #(->> % :scalar-bindings
                       (some (fn [binding]
                               (when (= '_n_bound (:parameter binding))
                                 (:conversion binding)))))
                 certificates)))
    (is (str/includes? (:source phase-two) (str (name partials) "[")))
    (is (not (re-find #"\ba\[" (:source phase-two)))
        "the cross-block target kernel must not resurrect the original reduction body")
    (is (= 2 (second (get temporary-specs partials)))
        "dynamic two-phase scratch resolves through KernelLaunch IR at call time")))

(deftest scheduled-reduction-graph-has-no-source-shaped-target-fallback
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled)
        decline (fn [& _]
                  (throw (ex-info "simulated portable coverage gap"
                                  {:reason :segred-kernel-body-declined
                                   :missing-rule :simulated
                                   :fallback :none})))]
    (doseq [target-dialect [:opencl-intel :cuda]]
      (testing (name target-dialect)
        (let [exception
              (try
                (with-redefs [segred-body/lower decline]
                  (sg/generate-kernel-graph graph :array-types {'a :double}
                                            :target-dialect target-dialect))
                nil
                (catch clojure.lang.ExceptionInfo exception exception))]
          (is (= :kernel-graph-target-lowering-missing (:reason (ex-data exception))))
          (is (= :simulated (get-in (ex-data exception)
                                    [:kernel-body-decline :missing-rule])))
          (is (= :none (:fallback (ex-data exception)))))))))

(deftest stencil-retains-selected-math-consent
  (let [operation (segop/->SegStencil
                   905 (segop/make-seg-space 'i 32) (segop/->SegLevel :thread :virtual)
                   (with-meta '(raster.math/tanh (clojure.core/aget input i))
                     {:raster.type/tag 'float})
                   #{'input} #{'output} #{} (segop/->KernelGrid 1 32 0)
                   :float 'output 1 :dirichlet nil :no-write-alias)
        options {:array-types {'input :float 'output :float} :workgroup-size 32}
        policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        ordinary (segstencil-body/schedule operation options)
        selected (segstencil-body/schedule operation (assoc options :scalar-math policy))]
    (is (not (contains? (:numerics ordinary) :scalar-math)))
    (is (= policy (get-in selected [:numerics :scalar-math])))
    (is (= (get-in ordinary [:body :launch]) (get-in selected [:body :launch])))
    (is (contains? (kernel-body/required-scalar-dtypes (get-in selected [:body :operations]))
                   :double))
    (is (thrown? clojure.lang.ExceptionInfo
                 (scheduled-body/validate! (update selected :numerics dissoc :scalar-math))))))

(deftest typed-stencil-emits-a-guarded-typed-artifact
  (let [source '(let* [result
                       (raster.par/stencil!
                        du [u] 1 :dirichlet double i n
                        (* alpha inv-dx2
                           (+ (clojure.core/aget u (clojure.core/- i 1))
                              (* -2.0 (clojure.core/aget u i))
                              (clojure.core/aget u (clojure.core/+ i 1)))))]
                      result)
        typed (-> (typed-route/attempt
                   source :double {'du :double 'u :double}
                   {:scalar-types {'n :long 'alpha :double 'inv-dx2 :double}})
                  :program :equations first :algorithm)
        stencil (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        artifact (sg/generate-segstencil-kernel-body
                  stencil
                  :array-types {'du :double 'u :double}
                  :scalar-types {'n :long 'alpha :double 'inv-dx2 :double})
        certificate (get-in artifact [:provenance :scheduled-operation])
        kernel-body (get-in artifact [:attributes :kernel-body])
        if-region (some #(when (= "IfRegion" (some-> % class .getSimpleName)) %)
                        (:operations kernel-body))
        interior-region (some #(when (= "IfRegion" (some-> % class .getSimpleName)) %)
                              (:then-operations if-region))
        loads (filterv #(= "ScalarLoad" (some-> % class .getSimpleName))
                       (:then-operations interior-region))]
    (is (kart/kernel-artifact? artifact))
    (is (scheduled-body/scheduled-kernel-body? certificate))
    (is (identical? stencil (:source certificate)))
    (is (= :scheduled-kernel-body (get-in artifact [:provenance :lowering])))
    (is (= certificate (get-in artifact [:attributes :scheduled-kernel-body])))
    (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
    (is (= '[u du alpha inv-dx2 _n_bound] (mapv :name (:abi artifact))))
    (is (= [:input :output :scalar :scalar :scalar] (mapv :kind (:abi artifact))))
    (is (= [:double :double :double :double :long] (mapv :dtype (:abi artifact))))
    (is (= [:double :double :double :double :int]
           (mapv :kernel-dtype (:abi artifact))))
    (is (= '[u du alpha inv-dx2 n] (:arguments artifact)))
    (is (= :no-write-alias (get-in artifact [:abi 0 :aliasing])))
    (is (= [{:value 'u :access :read} {:value 'du :access :write}]
           (get-in artifact [:effects :uses])))
    (is (= {:kind :stencil :boundary :dirichlet :radius 1}
           (dissoc (:effects artifact) :uses)))
    (is (= [{:parameter 'alpha :value 'alpha :dtype :double :kernel-dtype :double
             :conversion :identity}
            {:parameter 'inv-dx2 :value 'inv-dx2 :dtype :double :kernel-dtype :double
             :conversion :identity}
            {:parameter '_n_bound :value 'n :dtype :long :kernel-dtype :int
             :conversion :checked-range}]
           (:scalar-bindings certificate)))
    (is (= :portable-segstencil (get-in kernel-body [:attributes :kind])))
    (is (= 3 (count loads))
        "the interior control region must dominate every neighborhood read")))

(deftest typed-stencil-graph-preserves-its-scheduled-certificate
  (let [source '(let* [result
                       (raster.par/stencil!
                        du [u] 1 :dirichlet double i n
                        (* alpha (clojure.core/aget u i)))]
                      result)
        scalar-types {'n :long 'alpha :double}
        typed (-> (typed-route/attempt source :double {'du :double 'u :double}
                                      {:scalar-types scalar-types})
                  :program :equations first :algorithm)
        stencil (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        graph (kgraph/from-segops
               [stencil]
               {:inputs #{'u} :outputs #{'du} :dtype :double
                :buffer-specs {'u {:dtype :double :elements 'n}
                               'du {:dtype :double :elements 'n}}
                :scalars [(kgraph/scalar 'alpha :double)
                          (kgraph/scalar 'n :long)]})
        emitted (sg/generate-kernel-graph
                 graph :array-types {'du :double 'u :double}
                 :scalar-types scalar-types)
        artifact (get-in emitted [:nodes 0 :operation])
        certificate (get-in artifact [:provenance :scheduled-operation])]
    (is (kgraph/kernel-graph? emitted))
    (is (identical? stencil (:source certificate)))
    (is (= certificate (get-in artifact [:attributes :scheduled-kernel-body])))
    (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
    (is (= [:double :long] (mapv :dtype (:scalars emitted))))
    (is (= :long (some #(when (= '_n_bound (:name %)) (:dtype %)) (:abi artifact))))
    (is (= :int (some #(when (= '_n_bound (:name %)) (:kernel-dtype %)) (:abi artifact))))
    (is (= certificate
           (scheduled-body/validate-against-node! certificate (first (:nodes graph)) graph)))))

(deftest stencil-bound-conversion-is-proof-carrying-and-range-checked
  (let [source '(let* [result
                       (raster.par/stencil!
                        du [u] 1 :dirichlet double i n
                        (clojure.core/aget u i))]
                      result)
        scalar-types {'n :long}
        typed (-> (typed-route/attempt source :double {'du :double 'u :double}
                                      {:scalar-types scalar-types})
                  :program :equations first :algorithm)
        stencil (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        scheduled (segstencil-body/schedule
                   stencil {:array-types {'du :double 'u :double}
                            :scalar-types scalar-types})
        missing-proof (update scheduled :scalar-bindings
                              (fn [bindings]
                                (mapv #(if (= '_n_bound (:parameter %))
                                         (assoc % :conversion :identity)
                                         %)
                                      bindings)))
        graph (kgraph/from-segops
               [stencil]
               {:inputs #{'u} :outputs #{'du} :dtype :double
                :buffer-specs {'u {:dtype :double :elements 'n}
                               'du {:dtype :double :elements 'n}}
                :scalars [(kgraph/scalar 'n :long)]})
        emitted (sg/generate-kernel-graph
                 graph :array-types {'du :double 'u :double}
                 :scalar-types scalar-types)]
    (try
      (scheduled-body/validate! missing-proof)
      (is false "logical long to physical int requires an explicit checked-range proof")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :scheduled-kernel-body-scalar-bindings (:reason (ex-data exception))))))
    (try
      (graph-call/make emitted {'u :resident-u 'du :resident-du}
                       {'n {:type :long :value (inc (long Integer/MAX_VALUE))}})
      (is false "a proved narrowing still rejects an out-of-range runtime extent")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-scalar-range (:reason (ex-data exception))))))))

(defn- segred-source [body-expr]
  (let [form (list 'raster.par/reduce 'acc 0.0 'j 'n body-expr)
        s (soac/par-form->soac 'result form 0 :dtype :double)
        segops (lower/lower-reduce s nil)
        operation (first segops)]
    (:source (sg/generate-segred-kernel
              operation (first (:outputs operation)) :dtype :double
              :array-types {'a :double} :scalar-types {'n :int}))))

(defn- emitted-scan-graph
  ([form] (emitted-scan-graph form {}))
  ([form opts]
   (let [result-type (or (:dtype opts) :double)
         node (soac/par-form->soac 'scan-result form 71 :dtype result-type)
         operations (lower/lower-scan node nil :dtype result-type)
         graph (lower/scan-kernel-graph node operations
                                        (update opts :scalar-types #(merge {'n :int} %)))]
     (sg/generate-scan-kernel-graph graph
                                    :scalar-types (:scalar-types opts)))))

(deftest scan-stages-use-shared-certificates-and-independent-reconstruction
  (let [form '(raster.par/scan out acc 0.0 i n double (+ acc (aget values i)))
        node (soac/par-form->soac 'scan-result form 906 :dtype :double)
        operations (lower/lower-scan node nil :dtype :double)
        graph (lower/scan-kernel-graph node operations {:scalar-types {'n :int}})
        emitted (sg/generate-scan-kernel-graph graph)]
    (doseq [[source-node emitted-node] (map vector (:nodes graph) (:nodes emitted))
            :let [artifact (:operation emitted-node)
                  certificate (get-in artifact [:provenance :scheduled-operation])]]
      (is (scheduled-body/scheduled-kernel-body? certificate))
      (is (= certificate (segscan-body/validate-against-node! certificate source-node graph)))
      (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (segscan-body/validate-against-node!
                    (assoc-in certificate [:numerics :policy] :unrelated-policy) source-node graph))))
    (is (= :inout (get-in emitted [:nodes 1 :operation :abi 0 :kind])))
    (doseq [[field value] [[:num-blocks 1] [:shared-mem-bytes 0]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (sg/generate-scan-kernel-graph
                    (assoc-in graph [:nodes 0 :operation :grid field] value)))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (sg/generate-scan-kernel-graph graph :array-types {'values :float})))))

(deftest scan-math-consent-is-retained-and-independently-reconstructed
  (let [element (with-meta '(Math/tanh (clojure.core/aget values i))
                  {:raster.type/tag 'float})
        form (list 'raster.par/scan 'out 'acc 0.0 'i 'n 'float (list '+ 'acc element))
        source (soac/par-form->soac 'scan-result form 909 :dtype :float)
        graph (lower/scan-kernel-graph source (lower/lower-scan source nil :dtype :float)
                                      {:scalar-types {'n :int}})
        node (first (:nodes graph))
        policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        ordinary (segscan-body/schedule-for-node node graph {})
        selected (segscan-body/schedule-for-node node graph {:scalar-math policy})]
    (is (not (contains? (:numerics ordinary) :scalar-math)))
    (is (= policy (get-in selected [:numerics :scalar-math])))
    (is (= (get-in ordinary [:body :launch]) (get-in selected [:body :launch])))
    (is (contains? (kernel-body/required-scalar-dtypes (get-in selected [:body :operations]))
                   :double))
    (is (= selected (segscan-body/validate-against-node!
                     selected node graph {:scalar-math policy})))
    (doseq [dialect [:opencl-portable :cuda :hip]
            :let [emitted (sg/generate-kernel-graph graph :target-dialect dialect :scalar-math policy)
                  bodies (map #(get-in % [:operation :provenance :scheduled-operation]) (:nodes emitted))]]
      (is (= (count (:nodes graph)) (count bodies)))
      (is (every? #(= policy (get-in % [:numerics :scalar-math])) bodies))
      (is (some #(contains? (kernel-body/required-scalar-dtypes (get-in % [:body :operations]))
                           :double) bodies)))
    (doseq [options [{} {:scalar-math {:overrides {}}}]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (segscan-body/validate-against-node! selected node graph options))))
    (doseq [stage (:nodes graph)
            :let [baseline (segscan-body/schedule-for-node stage graph {})
                  chosen (segscan-body/schedule-for-node stage graph {:scalar-math policy})]]
      (is (= policy (get-in chosen [:numerics :scalar-math])))
      (is (= (select-keys baseline [:arguments :scalar-bindings :legality :effects])
             (select-keys chosen [:arguments :scalar-bindings :legality :effects])))
      (is (= chosen (segscan-body/validate-against-node!
                     chosen stage graph {:scalar-math policy}))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (scheduled-body/validate! (update selected :numerics dissoc :scalar-math))))))

(deftest legacy-scan-scalars-still-require-type-evidence
  (let [form '(raster.par/scan out acc 0.0 i n double
                               (+ acc (* scale (aget values i))))
        node (soac/par-form->soac 'scan-result form 907 :dtype :double)
        operations (lower/lower-scan node nil :dtype :double)]
    (is (= :scan-source-scalar-type
           (try (lower/scan-kernel-graph node operations {:scalar-types {'n :int}})
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (kgraph/kernel-graph?
         (sg/generate-scan-kernel-graph
          (lower/scan-kernel-graph node operations {:scalar-types {'n :int 'scale :double}}))))))

(defn- kernel-body-operations
  [artifact]
  (letfn [(walk [operations]
            (mapcat
             (fn [operation]
               (cons operation
                     (concat (walk (or (:operations operation) []))
                             (walk (or (:then-operations operation) []))
                             (walk (or (:else-operations operation) [])))))
             operations))]
    (walk (get-in artifact [:attributes :kernel-body :operations]))))

(defn- operation-kind [operation]
  (some-> operation class .getSimpleName))

(deftest segscan-graph-emits-one-verified-artifact-per-scheduled-node
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i n double
                                   (+ acc (* scale (clojure.core/aget values i))))
                 {:array-types {'values :double 'out :double}
                  :scalar-types {'scale :double}})
        [intra block carry] (mapv :operation (:nodes emitted))]
    (is (kgraph/kernel-graph? emitted))
    (is (every? kart/kernel-artifact? [intra block carry]))
    (is (= [:intra-block :block-scan :carry-in]
           (mapv #(get-in % [:attributes :phase]) [intra block carry])))
    (is (every? #(= :portable-segscan
                    (get-in % [:attributes :kernel-body :attributes :kind]))
                [intra block carry]))
    (testing "stage 1 owns both the result and graph temporary in one exact ABI"
      (is (= [:temporary :result :operand :parameter :bound]
             (mapv :role (:abi intra))))
      (is (= '[scale n] (take-last 2 (:arguments intra))))
      (let [operations (kernel-body-operations intra)]
        (is (some #(and (= "ScalarLoad" (operation-kind %))
                        (= 'values (:buffer %)))
                  operations))
        (is (some #(and (= "ScalarCompute" (operation-kind %))
                        (= :* (get-in % [:expression :op])))
                  operations))))
    (testing "block totals are scanned in one chunked workgroup for unbounded symbolic sizes"
      (is (= [1] (get-in block [:launch :group-count])))
      (let [operations (kernel-body-operations block)
            loop (some #(when (= "ForLoop" (operation-kind %)) %) operations)
            safe-bound (some #(when (and (= "ScalarCompute" (operation-kind %))
                                         (= 'scan-safe-bound (get-in % [:result :id])))
                                %)
                             operations)
            safe-expression (:expression safe-bound)]
        (is (= [0 'scan-safe-bound 256 :ordered]
               [(:lower loop) (:upper loop) (:step loop)
                (get-in loop [:attributes :association])]))
        (is (= :min (:op safe-expression)))
        (is (= :max (get-in safe-expression [:arguments 0 :op])))
        (is (= '_n_bound (get-in safe-expression [:arguments 0 :arguments 0]))
            "the in-body clamp, rather than an ABI role assertion, proves the loop extent")))
    (testing "carry consumes the scanned total of the preceding block"
      (let [loads (filter #(= "ScalarLoad" (operation-kind %))
                          (kernel-body-operations carry))]
        (is (= #{(first (map :id (:temporaries emitted))) 'out}
               (set (map :buffer loads))))))
    (testing "entry-point names are valid C identifiers"
      (is (every? #(re-matches #"[A-Za-z_][A-Za-z0-9_]*" (:kernel-name %))
                  [intra block carry])))
    (let [[intra-node block-node carry-node] (:nodes emitted)]
      (is (= [(:id intra-node)] (:dependencies block-node)))
      (is (= (mapv :id [intra-node block-node]) (:dependencies carry-node))
          "target lowering preserves the verified schedule"))))

(deftest typed-scan-schedule-target-lowers-without-a-legacy-soac-node
  (let [source '(let* [result (raster.par/scan out acc 0.0 i n float
                                               (+ acc (clojure.core/aget values i)))]
                      result)
        typed (:program (typed-route/attempt source :float
                                             {'values :float 'out :float}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0
                                 :array-types {'values :float 'out :float}}))
        graph (get-in scheduled [:equations 0 :attributes :kernel-graph])
        emitted (sg/generate-kernel-graph graph)]
    (is (= :typed-soac (get-in scheduled [:provenance :source-dialect])))
    (is (= 3 (count (:nodes emitted))))
    (is (every? kart/kernel-artifact? (map :operation (:nodes emitted))))
    (is (= [:intra-block :block-scan :carry-in]
           (mapv #(get-in % [:operation :attributes :phase]) (:nodes emitted))))))

(deftest segscan-artifact-abi-preserves-mixed-input-storage
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i n double
                                   (+ acc (double (clojure.core/aget values i))))
                 {:array-types {'values :int 'out :double}})
        intra (get-in emitted [:nodes 0 :operation])
        values-slot (first (filter #(= 'values (:name %)) (:abi intra)))]
    (is (= :int (:dtype values-slot)))
    (is (= :int (:kernel-dtype values-slot)))
    (is (re-find #"__global const int\* restrict values" (:source intra)))))

(deftest small-segscan-emits-the-degenerate-one-artifact-graph
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i 64 double (+ acc (aget values i))))
        artifact (get-in emitted [:nodes 0 :operation])]
    (is (= 1 (count (:nodes emitted))))
    (is (= :single (get-in artifact [:attributes :phase])))
    (is (= [1] (get-in artifact [:launch :group-count])))
    (is (not (re-find #"block_totals" (:source artifact))))))

(deftest segscan-carry-emits-the-certified-combine
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 1.0 i n double (* acc (aget values i))))
        carry (get-in emitted [:nodes 2 :operation])
        combines (for [operation (kernel-body-operations carry)
                       :when (= "ScalarCompute" (operation-kind operation))]
                   (get-in operation [:expression :op]))]
    (is (= [:*] (vec combines))
        "carry propagation must use the certified monoid, not a hard-coded addition")))

(deftest integral-segscan-carries-certified-wrapping-through-every-combine
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc (int 0) i n int
                                   (unchecked-add-int acc (aget values i)))
                 {:dtype :int
                  :array-types {'values :int 'out :int}
                  :scalar-types {'n :int}})
        combines (for [node (:nodes emitted)
                       operation (kernel-body-operations (:operation node))
                       :when (and (= "ScalarCompute" (operation-kind operation))
                                  (= :+ (get-in operation [:expression :op]))
                                  (= :int (get-in operation [:expression :result-type])))]
                   (:expression operation))]
    (is (seq combines))
    (is (every? #(= :wrap (get-in % [:options :overflow])) combines)
        "every reassociated integer combine must retain the certified modulo-2^N policy")))

(deftest segscan-carry-separates-its-launch-width-from-the-scanned-block-width
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              73)
        operations (lower/lower-scan node nil :dtype :float)
        scan-width (get-in operations [0 :grid :block-size])
        carry-width (quot scan-width 2)
        operations (assoc-in operations [2 :grid]
                             (segop/->KernelGrid
                              (klaunch/ceil-div 'n carry-width)
                              carry-width 0))
        graph (lower/scan-kernel-graph
               node operations {:scalar-types {'n :int} :array-types {'values :float 'out :float}})
        carry (get-in (sg/generate-scan-kernel-graph graph) [:nodes 2 :operation])
        scan-block (some #(when (= 'scan-block (:id %)) %)
                         (get-in carry [:attributes :kernel-body :indices]))]
    (is (= [carry-width] (get-in carry [:launch :workgroup-size])))
    (is (= scan-width
           (get-in carry [:attributes :kernel-body :schedule :scan-block-size])))
    (is (= (kernel-body/expression :floor-div 'scan-index scan-width)
           (:expression scan-block)))))

;; ================================================================
;; Silently-ignored-information family: a SegRed combine that the
;; extractor only partially models must FAIL LOUD, not miscompile.
;; ================================================================

(deftest segred-nary-combine-rejected
  (testing "a variadic (+ acc x y) combine is rejected, not silently summing only x"
    ;; Before the fix, op-args extraction took a0=acc a1=x and DROPPED y — emitting a
    ;; kernel that reduced with just x. A segmented reduce combine is binary (op acc elem).
    (is (thrown-with-msg?
         Exception #"certified associative reduction"
         (segred-source '(+ acc (clojure.core/aget a j) (clojure.core/aget b j))))))
  (testing "the ordinary binary combine still lowers unchanged"
    (let [source (segred-source '(+ acc (clojure.core/aget a j)))]
      (is (re-find #"tree_combined_.* = .* \+ " source))
      (is (re-find #"output\[" source)))))

(deftest segred-multistatement-lambda-rejected
  (testing "a reduce lambda with a multi-statement let body is rejected, not (last bdy)-dropped"
    ;; The single-aset-void store-drop shape on the reduce side: (last bdy) silently dropped
    ;; the earlier body forms.
    (let [failure (try
                    (segred-source '(let* [t (clojure.core/aget a j)] (+ acc t) (+ acc t)))
                    nil
                    (catch clojure.lang.ExceptionInfo error (ex-data error)))]
      (is (= :scalar-reduction-certificate-projection (:reason failure)))
      (is (= :reduction-not-elementwise (get-in failure [:normalization-error :reason]))))))

(deftest segred-devirtualized-aget-lowers-to-subscript
  (testing "the parametric (.invk aget-impl …) shape — the qlinear-k side of #55"
    (let [aget-invk (with-meta (list '.invk 'raster.arrays/aget_m_doubles_long-impl 'a 'j)
                      {:raster.op/original 'raster.arrays/aget
                       :raster.type/tag 'double})
          plus-invk (with-meta (list '.invk 'raster.numeric/_plus__m_double_double-impl
                                     'acc aget-invk)
                      {:raster.op/original 'raster.numeric/+
                       :raster.type/tag 'double})
          src (segred-source plus-invk)]
      (is (not (re-find #"gpufn_aget" src))
          "devirtualized aget must normalize to a subscript, not a helper call")
      (is (re-find #"a\[" src)))))

(deftest segred-bare-aget-lowers-to-subscript
  (testing "the typed clojure.core/aget shape — the decoder-gpu side of #55"
    (let [src (segred-source '(+ acc (clojure.core/aget a j)))]
      (is (not (re-find #"gpufn_aget" src)))
      (is (re-find #"a\[" src)))))

(deftest segred-emits-complete-ordered-typed-abi
  (let [product (with-meta '(* scale (clojure.core/aget a i))
                  {:raster.type/tag 'float})
        form (with-meta (list 'raster.par/reduce 'acc 0.0 'i 'n
                              (list '+ (list 'float 'acc) product))
               {:raster.type/elem-type :float})
        s (soac/par-form->soac 'result form 0 :dtype :float)
        segred (first (lower/lower-reduce s nil))
        physical-output (first (:outputs segred))
        k (sg/generate-segred-kernel segred (first (:outputs segred)) :dtype :float
                                     :scalar-types {'scale :float 'n :int})]
    (is (kart/kernel-artifact? k))
    (testing "signature, ABI and compiler values have one identical order"
      (is (= ['a physical-output 'scale '_n_bound] (mapv :name (:abi k))))
      (is (= [:input :output :scalar :scalar] (mapv :kind (:abi k))))
      (is (= [:float :float :float :int] (mapv :kernel-dtype (:abi k))))
      (is (= [:operand :result :parameter :bound] (mapv :role (:abi k))))
      (is (= ['a physical-output 'scale 'n] (:arguments k)))
      (is (= (kabi/signature-shape (:abi k))
             (kabi/source-signature-shape (:kernel-name k) (:source k))))
      (is (= :no-write-alias (get-in k [:abi 0 :aliasing]))))
    (testing "launch and semantic origin are part of the value"
      (is (klaunch/launch-spec? (:launch k)))
      (is (= [256] (get-in k [:launch :workgroup-size])))
      (is (= 1024 (get-in k [:launch :shared-memory-bytes])))
      (is (= :kernel-body (get-in k [:provenance :dialect])))
      (is (= (:id segred) (get-in k [:provenance :segop-id])))
      (is (= :scheduled-kernel-body (get-in k [:provenance :lowering])))
      (is (= :float (get-in k [:attributes :dtype])))
      (is (= :kernel-body (get-in k [:attributes :emission-route])))
      (is (some? (get-in k [:attributes :kernel-body])))
      (is (nil? (get-in k [:attributes :n-phases]))
          "one phase artifact does not claim that it owns two phases")))
  (testing "the artifact retains a physical result identity; only the host marker substitutes nil"
    (let [form '(raster.par/reduce acc 0.0 i n (+ acc (clojure.core/aget a i)))
          s (soac/par-form->soac 'result form 0 :dtype :float)
          operation (first (lower/lower-reduce s nil))
          physical-output (first (:outputs operation))
          k (sg/generate-segred-kernel operation nil :dtype :float
                                       :scalar-types {'n :int})]
      (is (= ['a physical-output '_n_bound] (mapv :name (:abi k))))
      (is (= ['a physical-output 'n] (:arguments k))))))

(deftest exact-empty-scalar-fold-retains-seed-bits-without-demanding-element-storage
  (doseq [dtype [:float :double]
          negative? [false true]]
    (let [seed (if (= :float dtype) (float (if negative? -0.0 0.0))
                   (if negative? -0.0 0.0))
          source (list 'let* ['result (list 'raster.par/reduce 'acc seed 'i 0
                                           '(+ acc (clojure.core/aget a i)))] 'result)
          options {:dtype dtype :array-types {'a dtype}}
          typed (:program (typed-route/attempt source dtype {'a dtype} options))
          program (:form (segop-lower/segop-lower-pass typed options))
          algorithm (equation-graph/algorithm-for-equations program (:equations program))
          graph (equation-graph/make algorithm program)
          node (first (:nodes graph))
          operation (:operation node)
          certificate (segred-body/schedule operation nil options)
          literal (get-in certificate [:body :operations 0 :expression :arguments 0])
          bits (if (= :float dtype)
                 #(Float/floatToRawIntBits (float %))
                 #(Double/doubleToRawLongBits (double %)))]
      (is (segop/empty-scalar-reduction? operation))
      (is (= #{'a} (:inputs operation)) "semantic element inputs remain retained for audit")
      (is (= #{} (segop/operation-inputs operation)))
      (is (= {} (map-reads/fold-read-requirements operation options)))
      (is (= [] (:inputs graph)))
      (is (= dtype (:type literal)))
      (is (= (class seed) (class (:value literal))))
      (is (= (bits seed) (bits (:value literal))))
      (is (= [] (get-in certificate [:body :allocations])))
      (is (= :exact-empty-fold (get-in certificate [:numerics :policy])))
      (is (= :exact (get-in certificate [:numerics :mode])))
      (is (= [1] (get-in certificate [:body :launch :workgroup-size])))
      (is (= certificate (segred-body/validate-against-node!
                         certificate node graph algorithm program)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [emitted (sg/generate-kernel-graph
                       graph :target-dialect target
                       :scheduled-equation-algorithm algorithm :scheduled-equation-body program)
              artifact (get-in emitted [:nodes 0 :operation])]
          (is (emitted-equation/emitted-equation?
               (emitted-equation/make algorithm program emitted)))
          (is (= '[result 0] (:arguments artifact)))
          (is (not (str/includes? (:source artifact) "__local")))
          (is (not (str/includes? (:source artifact) "barrier(")))
          (is (str/includes? (:source artifact) (if negative? "-0.0" "0.0")))))
      (doseq [forged [(assoc-in certificate [:body :operations 0 :expression :arguments 0 :value]
                                (if (= :float dtype) (float (if negative? 0.0 -0.0))
                                    (if negative? 0.0 -0.0)))
                      (assoc-in certificate [:source :reduction :components 0 :neutral]
                                (if (= :float dtype) (float (if negative? 0.0 -0.0))
                                    (if negative? 0.0 -0.0)))
                      (assoc-in certificate [:body :launch :workgroup-size] [2])
                      (assoc-in certificate [:body :attributes :identity] 1.0)]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (segred-body/validate-against-node! forged node graph algorithm program))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (segred-body/validate-against-node!
                    certificate node (assoc-in graph [:outputs 0 :elements] 2) algorithm program)))
      (doseq [forged [(assoc-in operation [:space :dims 0 :bound] 1)
                      (assoc operation :phase :cross-block)
                      (assoc-in operation [:grid :block-size] 2)
                      (assoc-in operation [:schedule :attributes :group-count] 2)]]
        (is (thrown? clojure.lang.ExceptionInfo (segred-body/schedule forged nil options)))))))

(deftest empty-scalar-fold-keeps-terminal-captures-but-not-element-captures
  (let [base (first (lower/lower-reduce
                    (soac/par-form->soac
                     'result '(raster.par/reduce acc 0.0 i 0
                                                (+ acc (* fold-scale (clojure.core/aget a i))))
                     94 :dtype :double) nil :dtype :double))
        transform (kernel-body/->ScalarRegion
                   '[completed resident scale]
                   '(clojure.core/* scale
                                    (clojure.core/+ completed
                                                    (clojure.core/double
                                                     (clojure.core/aget resident 0))))
                   [{:sym 'resident :dtype :float :map (axis-map/singleton-broadcast)}] :double)
        operation (-> base
                      (update :inputs conj 'resident)
                      (assoc :scalars #{'fold-scale 'scale})
                      (assoc-in [:reduction :attributes :resident-scalar-captures] #{'resident})
                      (assoc-in [:reduction :attributes :result-region] transform))
        options {:array-types {'a :double 'resident :float 'result :double}
                 :scalar-types {'fold-scale :double 'scale :double}}
        certificate (segred-body/schedule operation nil options)
        graph (kgraph/from-segops
               [operation] {:inputs #{'resident} :outputs #{'result} :dtype :double
                            :scalars [(kgraph/scalar 'scale :double)]
                            :buffer-specs {'resident {:dtype :float :elements 1}
                                           'result {:dtype :double :elements 1}}})]
    (is (= #{'a 'resident} (:inputs operation)))
    (is (= #{'resident} (segop/operation-inputs operation)))
    (is (= #{'scale} (segop/operation-scalars operation)))
    (is (= {'resident 1} (map-reads/fold-read-requirements operation options)))
    (is (= '[resident result scale 0] (:arguments certificate)))
    (is (= 1 (count (filter #(instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                           (get-in certificate [:body :operations])))))
    (is (= certificate (segred-body/validate-against-node! certificate (first (:nodes graph)) graph)))
    (doseq [target [:opencl-portable :cuda :hip]]
      (let [artifact (get-in (sg/generate-kernel-graph graph :target-dialect target)
                             [:nodes 0 :operation])]
        (is (= '[resident result scale 0] (:arguments artifact)))
        (is (str/includes? (:source artifact) "resident["))))))

(deftest empty-fold-terminal-conversion-and-scalar-capture-retain-their-own-contracts
  (doseq [[accumulator-dtype result-dtype] [[:double :float] [:float :double]]]
    (let [form (with-meta
                 '(raster.par/reduce acc 0.0 i 0
                                     (+ acc (* fold-scale (clojure.core/aget a i))))
                 {:raster.type/elem-type accumulator-dtype})
          base (first (lower/lower-reduce
                       (soac/par-form->soac 'result form 96 :dtype accumulator-dtype)
                       nil :dtype accumulator-dtype))
          cast (if (= [:double :float] [accumulator-dtype result-dtype])
                 'clojure.core/unchecked-float 'clojure.core/double)
          transform (kernel-body/->ScalarRegion
                     '[completed scale] (list cast '(clojure.core/* completed scale))
                     [] result-dtype)
          operation (-> base (assoc :scalars #{'fold-scale 'scale})
                        (assoc-in [:reduction :attributes :result-region] transform))
          options {:array-types {'a accumulator-dtype 'result result-dtype}
                   :scalar-types {'fold-scale accumulator-dtype 'scale accumulator-dtype}}
          certificate (segred-body/schedule operation nil options)
          graph (kgraph/from-segops
                 [operation] {:inputs #{} :outputs #{'result} :dtype result-dtype
                              :scalars [(kgraph/scalar 'scale accumulator-dtype)]
                              :buffer-specs {'result {:dtype result-dtype :elements 1}}})]
      (is (= #{'scale} (segop/operation-scalars operation)))
      (is (= '[result scale 0] (:arguments certificate)))
      (is (= accumulator-dtype (get-in certificate [:numerics :accumulator-dtype])))
      (is (= result-dtype (get-in certificate [:numerics :result-transform :result-dtype])))
      (is (= result-dtype (get-in certificate [:body :parameters 0 :dtype])))
      (is (= certificate (segred-body/validate-against-node! certificate (first (:nodes graph)) graph)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (is (= '[result scale 0]
               (get-in (sg/generate-kernel-graph graph :target-dialect target)
                       [:nodes 0 :operation :arguments]))))
      (when (= :float result-dtype)
        (let [checked (assoc-in operation [:reduction :attributes :result-region :expression]
                                '(clojure.core/float (clojure.core/* completed scale)))]
          (is (thrown? clojure.lang.ExceptionInfo
                       (segred-body/schedule checked nil options)))
          (is (thrown? clojure.lang.ExceptionInfo
                       (sg/generate-kernel-graph
                        (assoc-in graph [:nodes 0 :operation] checked)
                        :target-dialect :opencl-portable))))))))

(deftest nonempty-scalar-fold-does-not-elide-element-inputs
  (doseq [bound [1 'n]]
    (let [source (list 'raster.par/reduce 'acc 0.0 'i bound
                       '(+ acc (clojure.core/aget a i)))
          operation (first (lower/lower-reduce
                            (soac/par-form->soac 'result source 95 :dtype :double)
                            nil :dtype :double))
          certificate (segred-body/schedule operation nil {:scalar-types {'n :int}})]
      (is (not (segop/empty-scalar-reduction? operation)))
      (is (= :scalar-workgroup-tree (get-in operation [:schedule :strategy])))
      (is (= #{'a} (segop/operation-inputs operation)))
      (is (= 'a (first (:arguments certificate))))
      (is (seq (get-in certificate [:body :allocations]))))))

(deftest one-scalar-segred-schedule-emits-through-every-c-family-fixture
  (let [operation (first (lower/lower-reduce
                          (soac/par-form->soac
                           'result
                           '(raster.par/reduce acc 0.0 i n
                                               (+ acc (clojure.core/aget a i)))
                           90 :dtype :float)
                          nil :dtype :float))]
    (doseq [[target expected]
            [[:opencl-portable :opencl-c]
             [:cuda :cuda-c]
             [:hip :hip-cpp]]]
      (testing (name target)
        (let [artifact (sg/generate-segred-kernel
                        operation nil :dtype :float :target-dialect target
                        :scalar-types {'n :int})
              certificate (get-in artifact [:provenance :scheduled-operation])]
          (is (= expected (:target artifact)))
          (is (identical? operation (:source certificate)))
          (is (= artifact
                 (scheduled-body/validate-artifact-projection!
                  certificate artifact))))))))

(deftest scalar-reduction-separates-physical-capacity-from-read-minimum
  (let [source '(let* [a (raster.par/pmap j 8 double (clojure.core/double j))
                       result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))] result)
        options {:dtype :double :scalar-types {'n :long}}
        typed (:program (typed-route/attempt source :double {} options))
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        algorithm (equation-graph/algorithm-for-equations scheduled (:equations scheduled))
        graph (equation-graph/make algorithm scheduled)
        node-index (first (keep-indexed #(when (some #{'a} (:inputs (:operation %2))) %1)
                                       (:nodes graph)))
        node (nth (:nodes graph) node-index)
        operation (:operation node)
        artifact (sg/generate-segred-kernel operation (first (:outputs operation))
                                           :dtype :double :scalar-types {'n :long}
                                           :graph-node node :kernel-graph graph)
        certificate (get-in artifact [:attributes :scheduled-kernel-body])
        parameter (first (get-in artifact [:attributes :kernel-body :parameters]))
        arguments (fn [capacity n]
                    (mapv (fn [slot]
                            (case (:kind slot)
                              :input (double-array capacity)
                              :output (double-array 64)
                              :scalar {:type (:dtype slot) :value n}))
                          (:abi artifact)))
        admit (fn [capacity n]
                (kernel-call/validate-retained-input-capacities!
                 artifact (arguments capacity n)
                 (fn [buffer _] (java.lang.reflect.Array/getLength buffer))))]
    (is (some #(and (= 'a (:id %)) (= 8 (:elements %))) (:temporaries graph))
        "the reduction consumes genuinely internal storage, not a caller input")
    (is (= [8] (:shape parameter)))
    (is (= '_n_bound (:minimum-elements parameter)))
    (is (identical? artifact (admit 8 4)))
    (is (identical? artifact (admit 8 0)))
    (doseq [[capacity n] [[7 4] [8 9]]]
      (is (= :kernel-body-buffer-capacity
             (try (admit capacity n) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (doseq [tamper [#(dissoc % :minimum-elements)
                    #(assoc % :minimum-elements 0) #(assoc % :minimum-elements 1)]]
      (is (= :storage-extent
             (try (segred-body/validate-against-node!
                   (update-in certificate [:body :parameters 0] tamper)
                   (nth (:nodes graph) node-index) graph)
                  nil
                  (catch clojure.lang.ExceptionInfo error (:missing-rule (ex-data error)))))))))

(deftest scalar-segred-schedule-rejects-a-forged-graph-source
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double} :scalar-types {'n :long}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled-program (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled-program)
        node (last (:nodes graph))
        operation (:operation node)
        certificate (segred-body/schedule operation (first (:outputs operation)) options)
        forged-node (assoc node :operation (assoc operation :id :forged-reduction))
        forged-storage (update graph :outputs
                               #(mapv (fn [buffer] (assoc buffer :elements 2)) %))
        forged-certificates
        [(assoc-in certificate [:body :attributes :identity] 1.0)
         (assoc-in certificate [:body :schedule :reduction-operator] :*)
         (assoc-in certificate [:numerics :policy] :forged-tree)]]
    (is (identical? operation (:source certificate)))
    (try
      (sg/generate-segred-kernel operation 'forged-result
                                 :dtype :double :array-types {'a :double})
      (is false "production scalar emission cannot replace the semantic output identity")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :physical-output-remap
               (get-in (ex-data exception)
                       [:kernel-body-decline :missing-rule])))))
    (try
      (segred-body/validate-against-node! certificate forged-node graph)
      (is false "a boundary-compatible but different SegRed must not reuse the certificate")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :scheduled-kernel-body-source (:reason (ex-data exception))))))
    (try
      (segred-body/validate-against-node! certificate node forged-storage)
      (is false "the graph result allocation must equal the scheduled output group count")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :storage-extent (:missing-rule (ex-data exception))))))
    (try
      (sg/generate-kernel-graph forged-storage
                                :array-types {'a :double}
                                :scalar-types {'n :long})
      (is false "production graph emission must apply the SegRed/node storage validator")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :storage-extent
               (get-in (ex-data exception) [:kernel-body-decline :missing-rule])))))
    (doseq [forged forged-certificates]
      (try
        (segred-body/validate-against-node! forged node graph)
        (is false "a body or numerical certificate cannot be changed independently of source")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :schedule-source (:missing-rule (ex-data exception)))))))))

(deftest scalar-segred-schedule-rejects-forged-phase-and-tree-facts
  (let [operation (first (lower/lower-reduce
                          (soac/par-form->soac
                           'result
                           '(raster.par/reduce acc 0.0 i 4096
                                               (+ acc (clojure.core/aget a i)))
                           92 :dtype :float)
                          nil :dtype :float))
        options {:array-types {'a :float}}
        missing-rule
        (fn [candidate]
          (try
            (segred-body/schedule candidate nil options)
            nil
            (catch clojure.lang.ExceptionInfo exception
              (:missing-rule (ex-data exception)))))
        cases
        [[:fused-map-representation (assoc operation :lambda '(identity element))]
         [:phase-level (assoc operation :level (segop/->SegLevel :block :none))]
         [:grid-shared-memory (assoc-in operation [:grid :shared-mem-bytes] 0)]
         [:schedule-grid (assoc-in operation [:schedule :workgroup-size] 128)]
         [:schedule-grid (assoc-in operation [:schedule :attributes :group-count] 1)]
         [:schedule-grid (assoc-in operation [:schedule :numerical-mode :overflow] :wrap)]
         [:physical-phase (assoc-in operation [:reduction :attributes :physical-phase]
                                    :cross-block)]]]
    (doseq [[expected candidate] cases]
      (testing (name expected)
        (is (= expected (missing-rule candidate)))))
    (let [grid (assoc (:grid operation) :num-blocks (klaunch/sum 1 1))
          forged (assoc operation :grid grid
                          :schedule (segred-body/scalar-workgroup-tree-schedule
                                     (:reduction operation) grid (:phase operation)))]
      (is (= :launch-grid (missing-rule forged))
          "an internally self-consistent but unrelated launch expression is not a proof"))))

(deftest scalar-segred-max-int-bound-uses-wide-overflow-free-schedule-arithmetic
  (let [operation (first (lower/lower-reduce
                          (soac/par-form->soac
                           'result
                           (list 'raster.par/reduce 'acc 0.0 'i Integer/MAX_VALUE
                                 '(+ acc (clojure.core/aget a i)))
                           93 :dtype :float)
                          nil :dtype :float))
        kernel-body (:kernel-body
                     (segred-body/lower operation nil
                                        :dtype :float :array-types {'a :float}))
        casts (filterv #(instance? raster.compiler.ir.kernel_body.IndexCast
                                   (:expression %))
                       (:indices kernel-body))
        loop (first (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                            (:operations kernel-body)))
        index-expressions (into {} (map (juxt :id :expression) (:indices kernel-body)))]
    (is (= '[wide-bound wide-group-index wide-group-count wide-local-index]
           (mapv :id casts)))
    (is (every? #(= :long (get-in % [:expression :dtype])) casts))
    (is (= :long (get-in loop [:index :type])))
    (is (= :ceil-div (get-in index-expressions ['group-chunk :op]))
        "the checked index algebra handles zero without forming n-1 or n+groups-1")
    (is (= :add (:op (get index-expressions 'group-end))))))

(deftest reduction-math-consent-is-reconstructed-from-independent-options
  (let [form (with-meta
               '(raster.par/reduce acc 0.0 i 32 (+ acc (clojure.core/aget a i)))
               {:raster.type/elem-type :float})
        base (first (lower/lower-reduce (soac/par-form->soac 'result form 904 :dtype :float)
                                        nil :dtype :float))
        transform (kernel-body/->ScalarRegion
                   '[completed] (with-meta '(raster.math/tanh completed)
                                   {:raster.type/tag 'float}) [] :float)
        operation (assoc-in base [:reduction :attributes :result-region] transform)
        node (kgraph/->ScheduledKernel
               :reduction operation [(kgraph/->ValueUse 'a :read)
                                     (kgraph/->ValueUse 'result :write)] #{} [])
        graph (kgraph/make {:inputs [(kgraph/buffer 'a :float 32 :global :input)]
                           :outputs [(kgraph/buffer 'result :float 1 :global :output)]
                           :scalars [] :nodes [node]})
        policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        options {:array-types {'a :float 'result :float} :scalar-math policy}
        selected (segred-body/schedule operation nil options)]
    (is (= policy (get-in selected [:numerics :scalar-math])))
    (is (= :float (get-in selected [:numerics :accumulator-dtype])))
    (is (contains? (kernel-body/required-scalar-dtypes (get-in selected [:body :operations]))
                   :double))
    (is (= selected (segred-body/validate-against-node!
                    selected node graph nil nil {:scalar-math policy})))
    (let [emitted (sg/generate-kernel-graph graph :scheduled-bodies {:reduction selected}
                                           :scalar-math policy :target-dialect :opencl-portable)]
      (is (= policy (get-in emitted [:nodes 0 :operation :provenance :scheduled-operation :numerics :scalar-math]))))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (let [supported {:device-id :synthetic-test-device
                       :execution {:scalar-dtype-support {:double :supported}}}
            emitted (sg/generate-kernel-graph
                     graph :scheduled-bodies {:reduction selected} :scalar-math policy
                     :target-descriptor supported :target-dialect dialect)]
        (is (= policy (get-in emitted [:nodes 0 :operation :provenance
                                      :scheduled-operation :numerics :scalar-math]))))
      (doseq [descriptor [{} {:execution {:scalar-dtype-support {:double :unsupported}}}]]
        (try
          (sg/generate-kernel-graph graph :scheduled-bodies {:reduction selected}
                                   :scalar-math policy :target-descriptor descriptor
                                   :target-dialect dialect)
          (is false "certified graph must admit selected evaluation against supplied hardware")
          (catch clojure.lang.ExceptionInfo error
            (is (= :kernel-body-target-math-capability (:reason (ex-data error))))))))
    (doseq [candidate [graph (assoc-in graph [:attributes :scalar-math] policy)]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (sg/generate-kernel-graph candidate :scheduled-bodies {:reduction selected}
                                             :target-dialect :opencl-portable))))
    (doseq [dialect [:opencl-portable :cuda :hip]
            :let [emitted (sg/generate-kernel-graph graph :target-dialect dialect :scalar-math policy)
                  body (get-in emitted [:nodes 0 :operation :provenance :scheduled-operation])]]
      (is (= policy (get-in body [:numerics :scalar-math])))
      (is (= :float (get-in body [:numerics :accumulator-dtype])))
      (is (contains? (kernel-body/required-scalar-dtypes (get-in body [:body :operations])) :double)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (segred-body/validate-against-node! selected node graph)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (segred-body/validate-against-node!
                  selected node graph nil nil {:scalar-math {:overrides {}}})))))

(deftest completed-scalar-reduction-transform-is-terminal-and-numerically-explicit
  (let [form (with-meta
               '(raster.par/reduce acc 0.0 i 32 (+ acc (clojure.core/aget a i)))
               {:raster.type/elem-type :float})
        base (first (lower/lower-reduce (soac/par-form->soac 'result form 91 :dtype :float) nil
                                        :dtype :float))
        transform (kernel-body/->ScalarRegion
                   '[completed scale] '(clojure.core/* completed scale) [] :float)
        terminal (-> base
                     (assoc-in [:reduction :attributes :result-region] transform)
                     (assoc :scalars #{'scale}))
        certificate (segred-body/schedule
                     terminal nil {:array-types {'a :float}
                                   :scalar-types {'scale :float}})]
    (is (= :single (:phase terminal)))
    (is (= {:kind :typed-scalar-region
            :policy :same-typed-ssa-evaluation-order
            :input-dtype :float :result-dtype :float}
           (get-in certificate [:numerics :result-transform])))
    (is (= :certified-workgroup-tree (get-in certificate [:numerics :policy])))
    (let [nonterminal (assoc terminal
                             :phase :block-local
                             :level (segop/->SegLevel :block :virtual)
                             :schedule (segred-body/scalar-workgroup-tree-schedule
                                        (:reduction terminal) (:grid terminal) :block-local))]
      (try
        (segred-body/schedule nonterminal nil
                             {:array-types {'a :float}
                              :scalar-types {'scale :float}})
        (is false "a block-local phase must not apply the completed transform")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :nonterminal-result-transform
                 (:missing-rule (ex-data exception)))))))))

(deftest completed-scalar-reduction-loads-a-mixed-typed-resident-singleton
  (let [step '(clojure.core/+ acc element)
        transform (dialect/make-result-transform
                   {:accumulator 'acc
                    :expression '(clojure.core/+ acc
                                                 (clojure.core/double
                                                  (clojure.core/aget resident 0)))
                    :operands [{:value 'resident :dtype :float
                                :map (axis-map/singleton-broadcast)}]
                    :scalars [] :result-dtype :double})
        attributes {:index 'i :extent 32 :accumulators '[acc]
                    :identities [0.0] :dtypes [:double]
                    :algebra [(scan/certify-reassociation
                               {:acc 'acc :init 0.0 :lambda step} :double)]
                    :attributes {:stable-array-captures '[resident]}
                    :result-transform transform}
        resident-value (assoc (av/tensor {:dtype :float :shape []})
                              :representation {:kind :resident-scalar-buffer :elements 1}
                              :memory-space :device)
        facts (dialect/default-program-facts
               {:values {'a (av/tensor {:dtype :double :shape [32]})
                         'resident resident-value
                         'sum (av/tensor {:dtype :double :shape []})}
                :inputs '[a resident]
                :equations {'reduction (dialect/default-equation-facts)}})
        equation (list '= 'reduction '[sum]
                       (list 'reduce attributes '[a] '[resident]
                             (dialect/lambda-form '[acc element resident-capture] [step])))
        program (dialect/make facts [equation] '[sum])
        terminal (first (lower/lower-typed-reduce program nil :dtype :double))
        options {:array-types {'a :double 'resident :float 'sum :double}}
        certificate (segred-body/schedule terminal nil options)
        parameters (get-in certificate [:body :parameters])
        resident-parameter (first (filter #(= 'resident (:id %)) parameters))
        resident-load (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                           (= 'resident (:buffer %)))
                                     (get-in certificate [:body :operations])))]
    (is (= :float (:dtype resident-parameter)))
    (is (= [1] (:shape resident-parameter)))
    (is (= 1 (:minimum-elements resident-parameter)))
    (is (= #{'resident}
           (get-in terminal [:reduction :attributes :resident-scalar-captures])))
    (is (= (axis-map/singleton-broadcast)
           (get-in terminal [:reduction :attributes :result-region :operands 0 :map])))
    (is (contains? (:inputs terminal) 'resident))
    (is (not (contains? (:scalars terminal) 'resident)))
    (is (= [0] (:coordinates resident-load)))
    (is (= :float (get-in resident-load [:result :type])))
    (is (contains? (set (map :buffer (get-in certificate [:body :stable-reads])))
                   'resident))
    (let [missing-rule
          (fn [candidate candidate-options]
            (try
              (segred-body/schedule candidate nil candidate-options)
              nil
              (catch clojure.lang.ExceptionInfo exception
                (:missing-rule (ex-data exception)))))]
      (is (= :full-reduction-result-operands
             (missing-rule terminal (assoc-in options [:array-types 'resident] :double))))
      (is (= :full-reduction-result-operands
             (missing-rule (assoc-in terminal
                                     [:reduction :attributes :resident-scalar-captures] #{})
                           options)))
      (is (= :full-reduction-result-operands
             (missing-rule (assoc-in terminal
                                     [:reduction :attributes :result-region :operands 0 :map]
                                     (axis-map/of-axes [['i 32]]))
                           options)))
      (is (= :uniform-scalar-storage
             (missing-rule terminal (assoc-in options [:array-types 'a] :float))))
      (let [nonterminal (assoc terminal
                               :phase :block-local
                               :level (segop/->SegLevel :block :virtual)
                               :schedule (segred-body/scalar-workgroup-tree-schedule
                                          (:reduction terminal) (:grid terminal) :block-local))]
        (is (= :nonterminal-result-transform (missing-rule nonterminal options)))))))

(deftest completed-conversion-separates-terminal-storage-from-reduction-state
  (doseq [[source target] [[:float :double] [:double :float]]]
    (let [form (with-meta
                 '(raster.par/reduce acc 0.0 i 32 (+ acc (clojure.core/aget a i)))
                 {:raster.type/elem-type source})
          base (first (lower/lower-reduce (soac/par-form->soac 'result form 91 :dtype source)
                                          nil :dtype source))
          transform (kernel-body/->ScalarRegion
                     '[completed] (list (symbol (name target)) 'completed) [] target)
          terminal (assoc-in base [:reduction :attributes :result-region] transform)
          options {:array-types {'a source 'result target}}
          certificate (segred-body/schedule terminal nil options)
          parameters (get-in certificate [:body :parameters])]
      (is (= target (:dtype (first (filter #(= :output (:kind %)) parameters)))))
      (is (= source (:dtype (first (filter #(= :input (:kind %)) parameters)))))
      (is (= source (get-in certificate [:numerics :accumulator-dtype])))
      (is (= target (get-in certificate [:numerics :result-transform :result-dtype])))
      (is (= source (:dtype (first (get-in certificate [:body :allocations])))))
      (let [arithmetic (assoc-in terminal [:reduction :attributes :result-region :expression]
                                 (list (symbol (name target)) '(+ completed 0.25)))
            arithmetic-certificate (segred-body/schedule arithmetic nil options)]
        (is (= source (get-in arithmetic-certificate [:numerics :accumulator-dtype])))
        (is (= target (get-in arithmetic-certificate [:numerics :result-transform :result-dtype]))))
      (doseq [candidate [base]]
        (try
          (segred-body/schedule candidate nil options)
          (is false "mixed output storage requires a certified floating terminal region")
          (catch clojure.lang.ExceptionInfo error
            (is (= :uniform-scalar-storage (:missing-rule (ex-data error))))))))))

(deftest unproved-affine-loads-decline-the-pointwise-reduction-route
  (let [form '(raster.par/reduce acc 0.0 i n
                                 (+ acc (clojure.core/aget a (+ i offset))))
        node (soac/par-form->soac 'result form 22 :dtype :float)
        operation (first (lower/lower-reduce node nil :dtype :float))]
    (try
      (sg/generate-segred-kernel
       operation nil :dtype :float :scalar-types {'offset :int 'n :int})
      (is false "an n-element buffer does not prove i+offset is in bounds")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-graph-target-lowering-missing
               (:reason (ex-data exception))))
        (is (= :indexed-load
               (get-in (ex-data exception)
                       [:kernel-body-decline :missing-rule])))))))

(deftest mixed-floating-reduction-arithmetic-is-explicit-typed-ssa
  (let [product (with-meta
                  '(* (double scale) (clojure.core/aget a i))
                  {:raster.type/tag 'double})
        form (with-meta
               (list 'raster.par/reduce 'acc 0.0 'i 'n
                     (list '+ 'acc (list 'float product)))
               {:raster.type/elem-type :float})
        node (soac/par-form->soac 'result form 88 :dtype :float)
        operation (first (lower/lower-reduce node :ze:0 :dtype :float))
        artifact (sg/generate-segred-kernel
                  operation nil :dtype :float
                  :scalar-types {'scale :double 'n :long})
        loop-body (:operations (first (:operations
                                       (get-in artifact [:attributes :kernel-body]))))
        casts (keep (fn [operation]
                      (let [expression (:expression operation)]
                        (when (= :cast (:op expression))
                          [(:result-type expression) (:options expression)])))
                    loop-body)]
    (is (= [[:double {:rounding :exact :overflow :exact}]
            [:float {:rounding :nearest-even :overflow :ieee}]]
           casts))
    (is (= :double (some #(when (= 'scale (:name %)) (:kernel-dtype %))
                         (:abi artifact))))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))))

(deftest reduction-scalar-typing-declines-uncertified-language-semantics
  (let [options {:index 'i :coordinate 'element-index :dtype :float
                 :arrays #{'a} :scalars #{'scale 'counter}
                 :scalar-types {'scale :double 'counter :long}}
        decline (fn [expression]
                  (try
                    (segred-body/lower-element-operations expression options)
                    nil
                    (catch clojure.lang.ExceptionInfo exception
                      (ex-data exception))))]
    (testing "mixed arithmetic needs its retained walker/TypedClojure result fact"
      (is (= :scalar-result-dtype
             (:missing-rule
              (decline '(* scale (clojure.core/aget a i)))))))
    (testing "even uniform Float operands do not identify clojure.core versus typed arithmetic"
      (is (= :scalar-result-dtype
             (:missing-rule
              (decline '(* (clojure.core/aget a i)
                           (clojure.core/aget a i)))))))
    (testing "checked integral casts are not reinterpreted as wrap or saturation"
      (is (= :checked-scalar-cast (:missing-rule (decline '(int scale)))))
      (is (= :checked-scalar-cast (:missing-rule (decline '(byte 128))))))
    (testing "integral value arithmetic cannot become a floating element implicitly"
      (is (= :element-result-dtype
             (:missing-rule
              (decline (with-meta '(+ counter 1) {:raster.type/tag 'long}))))))
    (testing "literal and expression results cannot be silently narrowed to the accumulator"
      (is (= {:missing-rule :element-result-dtype :element-dtype :double}
             (select-keys (decline 0.1) [:missing-rule :element-dtype])))
      (is (= :element-result-dtype
             (:missing-rule
              (decline (with-meta '(* scale (clojure.core/aget a i))
                         {:raster.type/tag 'double}))))))
    (testing "retained local conversions cannot silently narrow the region result"
      (let [local (with-meta 'local {:raster.type/tag 'float})]
        (is (= :element-result-dtype
               (:missing-rule
                (decline (list 'let* [local 0.1] (list 'double local))))))))))

;; ================================================================
;; Horizontally-fused multi-output SegMap: the SECONDARY output (written
;; only via a side-effect aset in the fused lambda) must be a NON-const
;; __global array param — never a scalar. Before the fix it was declared
;; `float hfuse_out__N` (a scalar) while the body indexed it as an array:
;; a broken kernel, and the extraction layer then eval'd the bare buffer
;; symbol on the host (`Unable to resolve symbol: hfuse_out__N`).
;; ================================================================

(deftest segmap-fused-secondary-output-is-array-param
  (let [{:keys [artifact operation admission]} (retirement/emit :secondary-output :ocl:0 false)
        slots (:abi artifact)
        pointers (filterv #(not= :scalar (:kind %)) slots)
        outputs (filterv #(= :output (:kind %)) pointers)]
    (is (= 1 (get-in admission [:stats :horizontal])))
    (is (= #{'hout1 'hout2} (:outputs operation)))
    (is (= #{'hout1 'hout2} (set (map :name outputs))))
    (is (= #{'a 'b 'd} (set (map :name (filter #(= :input (:kind %)) pointers)))))
    (is (every? #(= :float (:dtype %)) outputs))
    (is (= :kernel-body (kart/attribute artifact :emission-route)))
    (doseq [slot outputs]
      (let [name (or (:c-name slot) (str (:name slot)))]
        (is (str/includes? (:source artifact) (str name "[")))
        (is (re-find (re-pattern (str "__global float\\*\\s+" name)) (:source artifact)))
        (is (not (re-find (re-pattern (str "__global const float\\*[^,)]*" name)) (:source artifact))))))
    (is (= (mapv :name (remove #(= :bound (:role %)) slots))
           (vec (butlast (:arguments artifact)))))
    (is (= 'n (last (:arguments artifact))))
    (is (= artifact (kart/validate! artifact)))))

(deftest canonical-secondary-output-alias-contract-survives-retirement
  (let [{:keys [artifact]} (retirement/emit :secondary-output :ocl:0 false)
        pointers (filterv #(not= :scalar (:kind %)) (:abi artifact))
        arrays (mapv (fn [_] (float-array 1)) pointers)
        output-indices (keep-indexed #(when (= :output (:kind %2)) %1) pointers)]
    (is (= arrays (kabi/validate-logical-pointer-aliases!
                  (:abi artifact) arrays kernel-call/pointer-overlaps?)))
    (doseq [index output-indices]
      (is (thrown? clojure.lang.ExceptionInfo
                   (kabi/validate-logical-pointer-aliases!
                    (:abi artifact) (assoc arrays index (first arrays)) kernel-call/pointer-overlaps?))))
    ;; This ABI proves input-vs-write disjointness, not pairwise output disjointness.
    ;; Fresh pure-map output identity belongs to realization and its numerical/native controls.
    (let [shared-outputs (assoc arrays (second output-indices) (nth arrays (first output-indices)))]
      (is (= shared-outputs
             (kabi/validate-logical-pointer-aliases!
              (:abi artifact) shared-outputs kernel-call/pointer-overlaps?))))))

(deftest segmap-abi-preserves-integer-scalar-type
  (let [{:keys [artifact]} (retirement/emit :integer-offset :ocl:0 false)
        offset (some #(when (= 'offset (:name %)) %) (:abi artifact))]
    (is (= :scalar (:kind offset)))
    (is (= :int (:dtype offset) (:kernel-dtype offset)))
    (is (re-find #"int offset" (:source artifact)))
    (is (some #{'offset} (:arguments artifact)))
    (is (= :kernel-body (kart/attribute artifact :emission-route)))
    (is (= artifact (kart/validate! artifact)))))

(deftest retired-segmap-fixtures-preserve-independent-source-values
  (doseq [kind [:secondary-output :integer-offset]
          n [0 1 3]
          seed [1.0 -2.0]]
    (let [{:keys [source packet]} (retirement/emit kind :ocl:0 false)
          execute #(eval (list 'fn '[a b d n offset] %))
          a (float-array (map #(+ seed %) (range 4)))
          b (float-array [2.0 -4.0 8.0])
          d (float-array [0.5 0.25 -0.5])
          expected (if (= kind :secondary-output)
                     [(mapv #(float (* %1 %2)) (take n d) (take n b))
                      (mapv #(float (* %1 %2)) (take n d) (take n a))]
                     (mapv float (take n (drop 1 a))))
          values #(if (= kind :secondary-output) (mapv vec %) (vec %))]
      (is (= expected
             (values ((execute source) a b d (long n) (int 1)))
             (values ((execute (:source packet)) a b d (long n) (int 1))))))))

(deftest scheduled-segmap-fails-closed-at-the-kernel-body-boundary
  (let [form '(raster.par/map! out i n float
                               (clojure.core/aget a i))
        operation (-> (soac/par-form->soac 'out form 2)
                      (lower/lower-map nil :dtype :float)
                      first)
        failure
        (with-redefs [segmap-body/lower
                      (fn [& _]
                        (throw (ex-info "simulated portable coverage gap"
                                        {:reason :segmap-kernel-body-declined
                                         :missing-rule :simulated
                                         :fallback :none})))]
          (try
            (sg/generate-scheduled-segmap-kernel operation :dtype :float)
            nil
            (catch clojure.lang.ExceptionInfo exception
              (ex-data exception))))]
    (is (= :segmap-kernel-body-declined (:reason failure)))
    (is (= :simulated (:missing-rule failure)))
    (is (= :none (:fallback failure)))))
