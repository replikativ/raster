(ns raster.compiler.passes.parallel.typed-soac-fusion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.soac-dialect-adapter :as adapter]
            [raster.compiler.passes.parallel.soac-graph :as graph]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as typed-fusion]))

(def ^:private map-map-pairs
  [['y '(raster.par/map! y i n nil (* (aget x i) (aget x i)))]
   ['z '(raster.par/map! z j n nil (+ (aget y j) 1.0))]])

(def ^:private map-reduce-pairs
  [['y '(raster.par/map! y i n nil (* (aget x i) (aget x i)))]
   ['sum '(raster.par/reduce acc 0.0 j n (+ acc (aget y j)))]])

(def ^:private horizontal-map-pairs
  [['u '(raster.par/map! u i n nil (* (aget a i) 2.0))]
   ['v '(raster.par/map! v j n nil (+ (aget b j) 1.0))]])

(def ^:private captured-map-map-pairs
  [['y '(raster.par/map! y i n nil (* (aget x i) scale))]
   ['z '(raster.par/map! z j n nil (+ (aget y j) bias))]])

(def ^:private expensive-fanout-pairs
  [['a '(raster.par/map! a i n nil (Math/exp (Math/exp (Math/exp (aget x i)))))]
   ['b '(raster.par/map! b j n nil (* (aget a j) 2.0))]
   ['c '(raster.par/reduce acc 0.0 k n (+ acc (aget a k)))]])

(defn- differential-fusion
  [pairs outputs]
  (let [legacy-graph (graph/build-fusion-graph (legacy/let-bindings->nodes pairs))
        typed-input (adapter/legacy-nodes->program (:nodes legacy-graph)
                                                   {:outputs outputs :dtype :float})
        [legacy-fused legacy-stats] (graph/fusion-fixpoint legacy-graph)
        legacy-result (adapter/legacy-nodes->program (:nodes legacy-fused)
                                                     {:outputs outputs :dtype :float})
        [typed-result typed-stats] (typed-fusion/fusion-fixpoint typed-input)]
    {:legacy-result legacy-result
     :typed-result typed-result
     :legacy-stats legacy-stats
     :typed-stats typed-stats}))

(deftest map-map-fusion-matches-current-graph
  (let [{:keys [legacy-result typed-result typed-stats]}
        (differential-fusion map-map-pairs '[z])]
    (is (= (dialect/equations legacy-result) (dialect/equations typed-result)))
    (is (= {:vertical 1 :horizontal 0 :iterations 2} typed-stats))
    (is (= '[n x] (:inputs (dialect/facts typed-result))))
    (is (not (contains? (:values (dialect/facts typed-result)) 'y)))))

(deftest map-reduce-fusion-matches-current-graph
  (let [{:keys [legacy-result typed-result typed-stats]}
        (differential-fusion map-reduce-pairs '[sum])]
    (is (= (dialect/equations legacy-result) (dialect/equations typed-result)))
    (is (= {:vertical 1 :horizontal 0 :iterations 2} typed-stats))
    (is (= 'reduce (first (nth (first (dialect/equations typed-result)) 3))))
    (is (not-any? #{'y} (flatten (dialect/equations typed-result))))))

(deftest map-scan-fusion-preserves-the-certified-resident-boundary
  (doseq [[source mode result-shape]
          [['(let* [mapped (raster.par/pmap i n float
                                             (* (clojure.core/aget x i) 2.0))
                    result (raster.par/scan out acc 0.0 j n float
                                            (+ acc (clojure.core/aget mapped j)))]
                   result)
            :inclusive '[n]]
           ['(let* [mapped (raster.par/pmap i n float
                                             (* (clojure.core/aget x i) 2.0))
                    result (raster.par/scan-exclusive out acc 0.0 j n float
                                                      (+ acc (clojure.core/aget mapped j)))]
                   result)
            :exclusive '[(clojure.core/inc n)]]]]
    (let [program (frontend/form->program source
                                          {:dtype :float
                                           :array-types {'x :float 'out :float}})
          mapped-id (first (nth (first (dialect/equations program)) 2))
          [result stats] (typed-fusion/fusion-fixpoint program)
          equation (first (dialect/equations result))
          operation (dialect/operation-parts equation)
          result-id (first (nth equation 2))
          equation-facts (get-in (dialect/facts result) [:equations (second equation)])]
      (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
      (is (= 1 (count (dialect/equations result))))
      (is (= 'scan (:kind operation)))
      (is (= mode (get-in operation [:attributes :mode])))
      (is (= '[x] (:arrays operation)))
      (is (= result-shape (get-in (dialect/facts result) [:values result-id :shape])))
      (is (= #{:memory/write} (:effects equation-facts)))
      (is (= {result-id 'out} (:aliases equation-facts)))
      (is (= [{:destination 'out :access :write :host-return :buffer}]
             (get-in equation-facts [:attributes :result-storage])))
      (is (not-any? #{mapped-id} (flatten (dialect/equations result))))
      (is (= result (dialect/validate! result))))))

(deftest map-scan-fusion-declines-an-aliased-destination-read
  (let [source '(let* [mapped (raster.par/pmap i n float
                                                (* (clojure.core/aget x i) 2.0))
                       result (raster.par/scan x acc 0.0 j n float
                                               (+ acc (clojure.core/aget mapped j)))]
                      result)
        program (frontend/form->program source {:dtype :float :array-types {'x :float}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations result)))
        "materializing the map preserves its complete read before the scan mutates x")))

(deftest horizontal-map-fusion-matches-current-graph
  (let [{:keys [legacy-result typed-result typed-stats]}
        (differential-fusion horizontal-map-pairs '[u v])
        equation (first (dialect/equations typed-result))
        lambda (nth (nth equation 3) 4)
        region (dialect/lambda-parts lambda)]
    (is (= (dialect/equations legacy-result) (dialect/equations typed-result)))
    (is (= {:vertical 0 :horizontal 1 :iterations 2} typed-stats))
    (is (= '[u v] (nth equation 2)))
    (is (= '[%element0 %element1] (second lambda)))
    (is (= '[(* %element0 2.0) (+ %element1 1.0)] (:body-results region)))))

(deftest captured-map-fusion-matches-current-graph
  (let [{:keys [legacy-result typed-result typed-stats]}
        (differential-fusion captured-map-map-pairs '[z])
        equation (first (dialect/equations typed-result))
        operation (nth equation 3)
        lambda (nth operation 4)
        region (dialect/lambda-parts lambda)]
    (is (= (dialect/equations legacy-result) (dialect/equations typed-result)))
    (is (= {:vertical 1 :horizontal 0 :iterations 2} typed-stats))
    (is (= '[bias scale] (nth operation 3)))
    (is (= '[%element0 %capture0 %capture1] (second lambda)))
    (is (= '[(+ (* %element0 %capture1) %capture0)] (:body-results region)))))

(deftest typed-multi-consumer-placement-is-hardware-costed
  (let [program (-> expensive-fanout-pairs
                    legacy/let-bindings->nodes
                    graph/build-fusion-graph
                    :nodes
                    (adapter/legacy-nodes->program {:outputs '[b c] :dtype :float}))
        poor-am {:ridge {:float 2.0}}
        rich-am {:ridge {:float 100.0}}
        [materialized materialized-stats] (typed-fusion/fusion-fixpoint program poor-am)
        [recomputed recomputed-stats] (typed-fusion/fusion-fixpoint program rich-am)
        materialized-witness (-> materialized-stats :placements first)
        recomputed-witness (-> recomputed-stats :placements first)]
    (is (= 0 (:vertical materialized-stats)))
    (is (= 3 (count (dialect/equations materialized))))
    (is (= :materialize (:decision materialized-witness)))
    (is (= '[1 2] (:consumers materialized-witness)))
    (is (= (:placements materialized-stats)
           (get-in (dialect/facts materialized) [:attributes :fusion/placements])))

    (is (= 2 (:vertical recomputed-stats)))
    (is (= 2 (count (dialect/equations recomputed))))
    (is (= :recompute (:decision recomputed-witness)))
    (is (= 2 (:consumer-count recomputed-witness)))
    (is (not-any? #{'a} (mapcat #(nth % 2) (dialect/equations recomputed))))
    (is (= recomputed (dialect/validate! recomputed)))))

(deftest current-graph-rewrites-the-canonical-reduction-region
  (let [pairs [['tmp '(raster.par/map! tmp i n float (* (aget x i) 2.0))]
               ['sum '(raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]]
        source-graph (graph/build-fusion-graph (legacy/let-bindings->nodes pairs))
        [fused _] (graph/fusion-fixpoint source-graph)
        reduce-node (first (filter legacy/soac-reduce? (vals (:nodes fused))))
        reconstructed (legacy/soac->par-form reduce-node)]
    (is (not-any? #{'tmp} (flatten reconstructed))
        "the eliminated map result cannot survive in ProductReduction")
    (is (some #{'float} (flatten reconstructed))
        "the materialized map cast is part of the inlined value semantics")))

(deftest effectful-equations-do-not-fuse
  (let [legacy-graph (graph/build-fusion-graph (legacy/let-bindings->nodes map-map-pairs))
        program (adapter/legacy-nodes->program (:nodes legacy-graph)
                                               {:outputs '[z] :dtype :float})
        facts (-> (dialect/facts program)
                  (update-in [:equations 0 :effects] conj :memory/write)
                  (assoc :effects #{:memory/write}))
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

(deftest vertical-fusion-does-not-recompute-a-read-after-an-intervening-write
  (let [source '(let* [producer (raster.par/pmap i n float
                                                  (clojure.core/aget x i))
                       middle (raster.par/map! x j n float
                                               (+ (clojure.core/aget x j) 1.0))
                       consumer (raster.par/pmap k n float
                                                  (* (clojure.core/aget producer k) 2.0))]
                      consumer)
        program (frontend/form->program source {:dtype :float :array-types {'x :float}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 3 (count (dialect/equations result))))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (some #{'producer} (flatten (dialect/equations result)))
        "the consumer must read the value captured before x is mutated")))

(deftest horizontal-fusion-does-not-move-a-read-before-an-intervening-write
  (let [source '(let* [left (raster.par/pmap i n float
                                              (+ (clojure.core/aget a i) 1.0))
                       middle (raster.par/map! x j n float
                                               (+ (clojure.core/aget x j) 1.0))
                       right (raster.par/pmap k n float
                                               (* (clojure.core/aget x k) 2.0))]
                      [left right])
        program (frontend/form->program
                 source {:dtype :float :array-types {'a :float 'x :float}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 3 (count (dialect/equations result))))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= '[left middle right] (dialect/outputs result))
        "the effect result remains observable and ordered between the two reads")))

(deftest aliased-equations-decline-unproved-fusion
  (let [legacy-graph (graph/build-fusion-graph (legacy/let-bindings->nodes map-map-pairs))
        program (adapter/legacy-nodes->program (:nodes legacy-graph)
                                               {:outputs '[z] :dtype :float})
        facts (assoc-in (dialect/facts program) [:equations 0 :aliases] {'y 'x})
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

(deftest vertical-fusion-composes-local-ssa-regions
  (let [legacy-graph (graph/build-fusion-graph (legacy/let-bindings->nodes map-map-pairs))
        program (adapter/legacy-nodes->program (:nodes legacy-graph)
                                               {:outputs '[z] :dtype :float})
        equations (dialect/equations program)
        equations (mapv (fn [equation]
                          (let [operation (dialect/operation-parts equation)
                                {:keys [parameters body-results]}
                                (dialect/lambda-parts (:lambda operation))]
                            (list '= (second equation) (nth equation 2)
                                  (list 'map (:attributes operation) (:arrays operation)
                                        (:captures operation)
                                        (dialect/lambda-form
                                         parameters
                                         [(dialect/local-value
                                           'shared :float (first body-results))]
                                         '[shared])))))
                        equations)
        program (dialect/make (dialect/facts program)
                              equations
                              (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 1 (count (dialect/equations result))))
    (is (= [{:id 'rstr_producer_local_0
             :dtype :float
             :init '(* %element0 %element0)}
            {:id 'rstr_consumer_local_0
             :dtype :float
             :init '(+ rstr_producer_local_0 1.0)}]
           locals))
    (is (= '[rstr_consumer_local_0] body-results))
    (is (= result (dialect/validate! result)))))

(deftest horizontal-fusion-alpha-renames-and-composes-local-ssa-regions
  (let [legacy-graph (graph/build-fusion-graph
                      (legacy/let-bindings->nodes horizontal-map-pairs))
        program (adapter/legacy-nodes->program (:nodes legacy-graph)
                                               {:outputs '[u v] :dtype :float})
        equations
        (mapv (fn [equation]
                (let [operation (dialect/operation-parts equation)
                      {:keys [parameters body-results]}
                      (dialect/lambda-parts (:lambda operation))]
                  (list '= (second equation) (nth equation 2)
                        (list 'map (:attributes operation) (:arrays operation)
                              (:captures operation)
                              (dialect/lambda-form
                               parameters
                               [(dialect/local-value 'shared :float (first body-results))]
                               '[shared])))))
              (dialect/equations program))
        program (dialect/make (dialect/facts program) equations (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= {:vertical 0 :horizontal 1 :iterations 2} stats))
    (is (= ['rstr_left_local_0 'rstr_right_local_0] (mapv :id locals)))
    (is (= '[(* %element0 2.0) (+ %element1 1.0)] (mapv :init locals)))
    (is (= '[rstr_left_local_0 rstr_right_local_0] body-results))
    (is (= result (dialect/validate! result)))))

(deftest fusion-preserves-constituent-equation-facts
  (let [legacy-graph (graph/build-fusion-graph (legacy/let-bindings->nodes map-map-pairs))
        program (adapter/legacy-nodes->program (:nodes legacy-graph)
                                               {:outputs '[z] :dtype :float})
        facts (-> (dialect/facts program)
                  (assoc-in [:equations 0 :attributes] {:source :producer})
                  (assoc-in [:equations 1 :attributes] {:source :consumer}))
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result _] (typed-fusion/fusion-fixpoint program)
        constituents (get-in (dialect/facts result)
                             [:equations 1 :attributes :fusion/constituents])]
    (is (= #{0 1} (set (keys constituents))))
    (is (= {:source :producer} (get-in constituents [0 :attributes])))
    (is (= {:source :consumer} (get-in constituents [1 :attributes])))))

(deftest adapter-declines-unsupported-soacs
  (let [scan (legacy/par-form->soac
              'out '(raster.par/scan out acc 0.0 i n nil (+ acc (aget x i))) 0)]
    (try
      (adapter/legacy-nodes->program [scan] {:outputs '[out]})
      (is false "scan must not be silently projected through the differential subset")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :unsupported-legacy-node (:reason (ex-data exception))))))))

(deftest adapter-declines-arbitrary-scalar-bindings
  (let [binding (legacy/->ScalarBinding 0 'scale 2.0)]
    (try
      (adapter/legacy-nodes->program [binding] {})
      (is false "the differential adapter must not erase arbitrary scalar computation")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :unsupported-legacy-node (:reason (ex-data exception))))))))

(deftest horizontal-fusion-does-not-move-across-a-late-producer
  (let [tensor (av/tensor {:dtype :float :shape '[n]})
        extent (av/tensor {:dtype :long :shape []})
        equation-facts {'left (dialect/default-equation-facts)
                        'middle (assoc (dialect/default-equation-facts)
                                       :effects #{:memory/read})
                        'right (dialect/default-equation-facts)}
        program
        (dialect/make
         (dialect/default-program-facts
          {:values {'x tensor 'z tensor 'left-value tensor 'middle-value tensor
                    'right-value tensor 'n extent}
           :inputs '[n x z]
           :equations equation-facts
           :effects #{:memory/read}})
         [(list '= 'left '[left-value]
                (list 'map {:index 'i :extent 'n} '[x] []
                      (dialect/lambda-form '[x-element] '[(* x-element 2.0)])))
          (list '= 'middle '[middle-value]
                (list 'map {:index 'i :extent 'n} '[z] []
                      (dialect/lambda-form '[z-element] '[(+ z-element 1.0)])))
          (list '= 'right '[right-value]
                (list 'map {:index 'i :extent 'n} '[middle-value] []
                      (dialect/lambda-form '[middle-element]
                                           '[(* middle-element 3.0)])))]
         '[left-value right-value])
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

(deftest legacy-vertical-fusion-declines-ambiguous-multi-result-map
  (let [producer (-> (legacy/par-form->soac
                      'u '(raster.par/map! u i n nil (* (aget x i) 2.0)) 0)
                     (assoc :outputs #{'u 'v}))
        consumer (legacy/par-form->soac
                  'z '(raster.par/map! z j n nil (+ (aget u j) 1.0)) 1)
        graph (graph/build-fusion-graph [producer consumer])]
    (is (false? (boolean (graph/can-fuse-vertically? graph 0 1)))
        "legacy dependency edges do not identify which tuple component was consumed")))

(defn- contract-result-map-program
  [map-expression]
  (frontend/form->program
   (list 'let*
         ['contract-step
          '(raster.par/contract
            C [[i 4] [j 8]] [[l 16]]
            (* (clojure.core/aget A (+ (* i 16) l))
               (clojure.core/aget B (+ (* l 8) j))))
          'map-step
          (list 'raster.par/map! 'D 't 32 nil map-expression)]
         'map-step)
   {:dtype :float
    :array-types '{A :float B :float C :float D :float bias :float residual :float}
    :scalar-types '{scale :float}}))

(deftest segmented-reduce-result-map-becomes-a-typed-result-transform
  (let [program (contract-result-map-program
                 '(* (+ (clojure.core/aget C t)
                        (clojure.core/aget bias (mod t 8))
                        (clojure.core/aget residual t))
                     scale))
        fold-before (-> program dialect/equations first typed-fusion/equation-info :body-results)
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        operation (dialect/operation-parts equation)
        transform (get-in operation [:attributes :result-transform])
        transform-region (dialect/lambda-parts (:lambda transform))
        facts (dialect/facts result)]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 1 (count (dialect/equations result))))
    (is (= '[map-step] (nth equation 2)))
    (is (= '[map-step] (dialect/outputs result)))
    (is (= '[A B bias residual scale] (:inputs facts)))
    (is (= 'D (get-in facts [:equations 0 :attributes :result-storage 0 :destination])))
    (is (= {'map-step 'D} (get-in facts [:equations 0 :aliases])))
    (is (not-any? #{'C [:effect-map 0 0]} (keys (:values facts))))
    (is (= '[4 8] (get-in facts [:values 'map-step :shape])))
    (is (= fold-before (:body-results (dialect/lambda-parts (:lambda operation))))
        "post-reduction fusion must never rewrite the reduction fold")
    (is (= '[residual bias] (mapv :value (:operands transform))))
    (is (= '#{i j}
           (-> transform :operands first :map axis-map/axes set)))
    (is (= '#{j}
           (-> transform :operands second :map axis-map/axes set)))
    (is (= '[scale] (mapv :value (:scalars transform))))
    (is (not-any? #{'C 't} (flatten (:body-results transform-region))))
    (is (= result (dialect/validate! result)))))

(deftest segmented-reduce-result-map-refuses-an-unproved-operand-index
  (let [program (contract-result-map-program
                 '(+ (clojure.core/aget C t)
                     (clojure.core/aget bias (+ t 1))))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations result))))))

(deftest segmented-reduce-result-map-retains-an-externally-live-intermediate
  (let [program (contract-result-map-program
                 '(+ (clojure.core/aget C t) 1.0))
        produced (-> program dialect/equations first (nth 2) first)
        program (dialect/make (dialect/facts program)
                              (dialect/equations program)
                              [produced 'map-step])
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations result))))))

(deftest completed-reduction-fuses-a-pure-scalar-consumer
  (let [program
        (frontend/form->program
         '(let* [total (raster.par/reduce acc 0.0 i n
                                           (+ acc (clojure.core/aget x i)))
                 ^double scaled (* ^double total ^double scale)]
            scaled)
         {:dtype :double
          :array-types {'x :double}
          :scalar-types {'scale :double}})
        fold-before (-> program dialect/equations first typed-fusion/equation-info :body-results)
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        operation (dialect/operation-parts equation)
        transform (get-in operation [:attributes :result-transform])
        transform-region (dialect/lambda-parts (:lambda transform))
        remapped (dialect/remap-values result {'scale 'gain})
        remapped-attributes (:attributes (dialect/operation-parts
                                          (first (dialect/equations remapped))))]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 1 (count (dialect/equations result))))
    (is (= 'reduce (:kind operation)))
    (is (= '[scaled] (nth equation 2)))
    (is (= '[scaled] (dialect/outputs result)))
    (is (= '#{n scale x} (set (:inputs (dialect/facts result)))))
    (is (= fold-before (:body-results (dialect/lambda-parts (:lambda operation))))
        "the scalar epilogue must never enter the element or partial-reduction fold")
    (is (= [] (:operands transform)))
    (is (= '[scale] (mapv :value (:scalars transform))))
    (is (not-any? #{'total} (flatten (:body-results transform-region))))
    (is (= '[gain] (mapv :value (get-in remapped-attributes
                                        [:result-transform :scalars]))))
    (is (not (contains? remapped-attributes :segment-axes))
        "remapping a full reduction must not invent segmented axes")
    (is (= result (dialect/validate! result)))))
