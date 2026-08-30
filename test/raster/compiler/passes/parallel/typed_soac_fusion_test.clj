(ns raster.compiler.passes.parallel.typed-soac-fusion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.soac-dialect-adapter :as adapter]
            [raster.compiler.passes.parallel.soac-graph :as graph]
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

(deftest aliased-equations-decline-unproved-fusion
  (let [legacy-graph (graph/build-fusion-graph (legacy/let-bindings->nodes map-map-pairs))
        program (adapter/legacy-nodes->program (:nodes legacy-graph)
                                               {:outputs '[z] :dtype :float})
        facts (assoc-in (dialect/facts program) [:equations 0 :aliases] {'y 'x})
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

(deftest local-ssa-equations-decline-fusion-until-region-composition-is-proved
  (let [legacy-graph (graph/build-fusion-graph (legacy/let-bindings->nodes map-map-pairs))
        program (adapter/legacy-nodes->program (:nodes legacy-graph)
                                               {:outputs '[z] :dtype :float})
        equations (dialect/equations program)
        producer (first equations)
        operation (dialect/operation-parts producer)
        {:keys [parameters body-results]} (dialect/lambda-parts (:lambda operation))
        producer (list '= (second producer) (nth producer 2)
                       (list 'map (:attributes operation) (:arrays operation)
                             (:captures operation)
                             (dialect/lambda-form
                              parameters
                              [(dialect/local-value 'shared :float (first body-results))]
                              '[shared])))
        program (dialect/make (dialect/facts program)
                              (assoc equations 0 producer)
                              (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

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
