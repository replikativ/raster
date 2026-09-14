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

(defn fuse-input-cast
  "Fuse one exact private FP32→FP16 cast into a matrix operand load region.

   `operand-role` is `:lhs` or `:rhs`. The matrix may be unbatched or carry a leading batch axis;
   its declared batching contract determines whether that operand owns the batch extent. Returns
   nil unless the cast covers exactly the selected physical operand and its temporary has no other
   observer. No source expression or target inference participates."
  [g producer-id consumer-id operand-role]
  (when-not (contains? #{:lhs :rhs} operand-role)
    (throw (ex-info "matrix input fusion requires an explicit operand role"
                    {:reason :raster/bug :operand-role operand-role})))
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
            [m n k] (:dimensions stage)
            selected (get stage operand-role)
            other (get stage (if (= :lhs operand-role) :rhs :lhs))
            batch-extent (when (get (:batching stage) operand-role)
                           (get-in stage [:batching :extent]))
            extent (apply launch/product
                          (cond-> []
                            batch-extent (conj batch-extent)
                            true (into (if (= :lhs operand-role) [m k] [k n]))))]
        (when (and (= :cast (:operation cast))
                   (= [:float :half] ((juxt :input-dtype :output-dtype) cast))
                   (= {:rounding :nearest-even :overflow :ieee}
                      (dissoc (:policy cast) :vector-width))
                   (= :half (:operand-dtype stage))
                   (= temporary selected)
                   (not= temporary other)
                   (not-any? #(= temporary (:sym %)) (get-in stage [:epilogue :operands]))
                   (not (contains? (:input-value-regions stage) temporary))
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
                             (-> stage
                                 (assoc operand-role input)
                                 (update :input-value-regions assoc input region)))
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
                              (update-in [:attributes :input-fusions] (fnil conj [])
                                         {:operand operand-role
                                          :producer producer
                                          :consumer (:operation (get by-id consumer-id))
                                          :physical-precondition :input-output-disjoint
                                          :selection :binding-admission-required})
                              ;; Retain the singular diagnostic key for callers that inspect a
                              ;; graph with one fused input.
                              (assoc-in [:attributes :input-fusion]
                                        {:operand operand-role
                                         :producer producer
                                         :consumer (:operation (get by-id consumer-id))
                                         :physical-precondition :input-output-disjoint
                                         :selection :binding-admission-required}))]
            (graph/validate! candidate)
            (when-not (= (graph/boundary-contract g) (graph/boundary-contract candidate))
              (throw (ex-info "matrix input fusion changed the public boundary"
                              {:reason :matrix-input-fusion-boundary})))
            candidate))))))

(defn fuse-lhs-cast
  "Compatibility spelling for fusing an exact cast into the left matrix operand."
  [g producer-id consumer-id]
  (fuse-input-cast g producer-id consumer-id :lhs))

(defn fuse-rhs-cast
  "Fuse an exact cast into the right matrix operand."
  [g producer-id consumer-id]
  (fuse-input-cast g producer-id consumer-id :rhs))
