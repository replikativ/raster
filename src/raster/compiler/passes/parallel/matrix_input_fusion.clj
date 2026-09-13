(ns raster.compiler.passes.parallel.matrix-input-fusion
  "Fuse a private representation cast into a matrix input, before target emission.
   This constructs a candidate, not a universally legal replacement: physical input/output
   disjointness is required by its emitted ABI and must be proved before automatic selection."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.layout-stage :as layout]
            [raster.compiler.ir.matrix-stage :as matrix]))

(defn- same-extent? [a b]
  (and (some? a) (some? b)
       (or (= a b)
           (and (empty? (launch/expression-references a))
                (empty? (launch/expression-references b))
                (= (launch/resolve-expression {} a) (launch/resolve-expression {} b))))))

(defn fuse-lhs-cast
  "Return an eligible candidate graph, or nil when this rewrite cannot prove its obligations.
   Accepts only an exact, full-reduction, unbatched matrix consumer of a private FP32→FP16 cast.
   No source expressions or operator inference participate. Target support is checked later."
  [g producer-id consumer-id]
  (let [g (graph/validate! g)
        nodes (:nodes g)
        by-id (into {} (map (juxt :id identity)) nodes)
        producer (get by-id producer-id)
        consumer (get by-id consumer-id)
        cast (:operation producer)
        stage (:operation consumer)]
    (when (and (layout/layout-stage? cast) (matrix/matrix-stage? stage))
      (layout/validate! cast)
      (matrix/validate! stage)
      (let [input (:input cast) temporary (:output cast)
            input-buffer (first (filter #(= input (:id %)) (:inputs g)))
            temp-buffer (first (filter #(= temporary (:id %)) (:temporaries g)))
            uses-of (fn [id] (for [node nodes use (:uses node) :when (= id (:buffer use))]
                              [(:id node) (:access use)]))
            [m _ k] (:dimensions stage)
            extent (launch/product m k)]
        (when (and (= :cast (:operation cast))
                   (= [:float :half] ((juxt :input-dtype :output-dtype) cast))
                   (= {:rounding :nearest-even :overflow :ieee}
                      (dissoc (:policy cast) :vector-width))
                   (= :half (:operand-dtype stage))
                   (= temporary (:lhs stage))
                   (not= temporary (:rhs stage))
                   (not-any? #(= temporary (:sym %)) (get-in stage [:epilogue :operands]))
                   (empty? (:input-value-regions stage))
                   (nil? (:batching stage))
                   (= {:kind :full :range [0 k]} (:reduction stage))
                   (= 1 (count (:input-shape cast)))
                   (same-extent? extent (first (:input-shape cast)))
                   (= :float (:dtype input-buffer)) (= :half (:dtype temp-buffer))
                   (same-extent? extent (:elements input-buffer))
                   (same-extent? extent (:elements temp-buffer))
                   (= [[input :read] [temporary :write]]
                      (mapv (juxt :buffer :access) (:uses producer)))
                   (= #{[producer-id :write] [consumer-id :read]}
                      (set (uses-of temporary)))
                   (= 2 (count (uses-of temporary)))
                   (not-any? #(= input (:buffer %)) (:uses consumer))
                   (every? #(= :read (second %)) (uses-of input))
                   (some #{producer-id} (:dependencies consumer))
                   (every? #(or (= consumer-id (:id %))
                                (not (some #{producer-id} (:dependencies %)))) nodes))
          (let [reserved (set (concat (map :id (concat (:inputs g) (:outputs g)
                                                      (:temporaries g) (:scalars g)))
                                     (keys (:epilogue stage))))
                local (fn [prefix]
                        (first (remove reserved (map #(symbol (str prefix %)) (range)))))
                loaded (local "tile_input_") converted (local "tile_cast_")
                region (body/->ScalarSSARegion
                        [loaded] [] [] :float
                        [(body/->ScalarCompute
                          (body/value converted :half)
                          (body/cast-expression loaded :half :nearest-even :ieee))]
                        converted :half)
                replacement (matrix/validate!
                             (assoc stage :lhs input :input-value-regions {input region}))
                consumer (-> consumer
                             (assoc :operation replacement)
                             (update :uses (fn [uses] (mapv #(if (= temporary (:buffer %))
                                                             (assoc % :buffer input) %) uses)))
                             (update :scalar-uses set/union (:scalar-uses producer))
                             (update :dependencies
                                     #(vec (distinct (concat (remove #{producer-id} %)
                                                             (:dependencies producer))))))
                candidate (-> g
                              (assoc :nodes (into [] (comp (remove #(= producer-id (:id %)))
                                                          (map #(if (= consumer-id (:id %)) consumer %))) nodes))
                              (update :temporaries #(filterv (fn [b] (not= temporary (:id b))) %))
                              (assoc-in [:attributes :input-fusion]
                                        {:producer producer :consumer (:operation (get by-id consumer-id))
                                         :physical-precondition :input-output-disjoint
                                         :selection :explicit-only}))]
            (graph/validate! candidate)
            (when-not (= (graph/boundary-contract g) (graph/boundary-contract candidate))
              (throw (ex-info "matrix input fusion changed the public boundary"
                              {:reason :matrix-input-fusion-boundary})))
            candidate))))))
