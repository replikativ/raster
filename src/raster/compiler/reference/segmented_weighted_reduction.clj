(ns raster.compiler.reference.segmented-weighted-reduction
  "Schedule-free double-precision interpreter for segmented weighted-reduction plans.

   The first executable membership/storage pair is indexed graph attention. This evaluator is the
   oracle for recognition and later fused leaves: it executes the plan's scalar regions and
   reduction/normalization fields rather than calling the source array operators. Unsupported
   providers fail explicitly instead of being approximated."
  (:require [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defn- fail
  [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

(defn- extent
  [value scalars]
  (let [resolved (if (symbol? value) (get scalars value ::missing) value)]
    (when (= ::missing resolved)
      (fail "segmented reduction is missing a scalar extent"
            :segmented-reduction-missing-scalar {:extent value}))
    (when-not (and (integer? resolved) (pos? resolved))
      (fail "segmented reduction extent must resolve to a positive integer"
            :segmented-reduction-invalid-runtime-extent
            {:extent value :resolved resolved}))
    (long resolved)))

(defn- scalar-call
  [operator arguments]
  (cond
    (contains? '#{raster.numeric/+ clojure.core/+} operator) (reduce + 0.0 arguments)
    (contains? '#{raster.numeric/* clojure.core/*} operator) (reduce * 1.0 arguments)
    (contains? '#{raster.numeric/- clojure.core/-} operator)
    (if (= 1 (count arguments)) (- (double (first arguments))) (reduce - arguments))
    (contains? '#{raster.numeric// clojure.core//} operator) (reduce / arguments)
    (contains? '#{raster.numeric/min clojure.core/min} operator)
    (reduce #(Math/min (double %1) (double %2)) arguments)
    (contains? '#{raster.numeric/max clojure.core/max} operator)
    (reduce #(Math/max (double %1) (double %2)) arguments)
    (contains? '#{raster.numeric/abs clojure.core/abs} operator)
    (Math/abs (double (first arguments)))
    (contains? '#{raster.numeric/sqrt Math/sqrt} operator)
    (Math/sqrt (double (first arguments)))
    (contains? '#{raster.math/exp Math/exp} operator)
    (Math/exp (double (first arguments)))
    (contains? '#{raster.math/log Math/log} operator)
    (Math/log (double (first arguments)))
    :else
    (fail "reference evaluator encountered an unsupported scalar operator"
          :segmented-reduction-reference-unsupported-scalar-op
          {:operator operator})))

(defn- scalar-expression
  [expression environment]
  (cond
    (number? expression) (double expression)
    (symbol? expression)
    (if (contains? environment expression)
      (get environment expression)
      (fail "scalar region references an unbound parameter"
            :segmented-reduction-reference-unbound-parameter
            {:parameter expression :available (set (keys environment))}))
    (seq? expression)
    (scalar-call (first expression)
                 (mapv #(scalar-expression % environment) (rest expression)))
    :else
    (fail "reference evaluator encountered an unsupported scalar value"
          :segmented-reduction-reference-unsupported-scalar-value
          {:expression expression})))

(defn- region-value
  [region values]
  (when-not (= (count (:parameters region)) (count values))
    (fail "scalar region received the wrong number of positional values"
          :segmented-reduction-reference-region-arity
          {:parameters (:parameters region) :values values}))
  (scalar-expression (:body region) (zipmap (:parameters region) values)))

(defn- argument-value
  [argument scalars]
  (case (:kind argument)
    :literal (double (:value argument))
    :inverse-sqrt (let [n (extent (:extent argument) scalars)]
                    (/ 1.0 (Math/sqrt (double n))))
    (fail "reference evaluator encountered an unsupported score argument"
          :segmented-reduction-reference-unsupported-score-argument
          {:argument argument})))

(defn- array-length
  [values]
  (cond
    (nil? values) nil
    (.isArray (class values)) (java.lang.reflect.Array/getLength values)
    (counted? values) (count values)
    :else nil))

(defn- element
  [buffers id index]
  (let [values (get buffers id ::missing)]
    (when (= ::missing values)
      (fail "segmented reduction is missing a buffer"
            :segmented-reduction-reference-missing-buffer {:buffer id}))
    (let [length (array-length values)]
      (when-not (and length (<= 0 index) (< index length))
        (fail "segmented reduction buffer index is out of bounds"
              :segmented-reduction-reference-buffer-bounds
              {:buffer id :index index :length length}))
      (double (if (.isArray (class values))
                (java.lang.reflect.Array/get values (int index))
                (nth values (int index)))))))

(defn- index-element
  [buffers id index]
  (let [value (element buffers id index)
        integer-value (long value)]
    (when-not (= value (double integer-value))
      (fail "segmented reduction index buffer contains a non-integral value"
            :segmented-reduction-reference-nonintegral-index
            {:buffer id :index index :value value}))
    integer-value))

(defn- indexed-provider!
  [plan]
  (when-not (and (= :edge-list-by-destination (get-in plan [:membership :kind]))
                 (= :indexed-dense-values (get-in plan [:storage :kind]))
                 (= :dot (get-in plan [:score :kind]))
                 (= :indexed-query (get-in plan [:score :left :kind]))
                 (= :indexed-key (get-in plan [:score :right :kind]))
                 (= :indexed-value (get-in plan [:value :kind])))
    (fail "reference evaluator does not implement this membership/storage provider"
          :segmented-reduction-reference-unsupported-provider
          {:membership (get-in plan [:membership :kind])
           :storage (get-in plan [:storage :kind])
           :score-left (get-in plan [:score :left :kind])
           :score-right (get-in plan [:score :right :kind])
           :value (get-in plan [:value :kind])}))
  plan)

(defn evaluate
  "Evaluate a checked indexed weighted-reduction plan.

   `buffers` maps compiler buffer identities to primitive arrays or indexed collections.
   `scalars` resolves symbolic plan extents such as n-nodes, n-edges, emb-dim and n-heads.
   Returns a double-array in the plan's flat output layout."
  [plan {:keys [buffers scalars] :or {buffers {} scalars {}}}]
  (let [{:keys [segment-axes membership score weight value numerator denominator normalization]}
        (-> plan swr/validate! indexed-provider!)
        axis-extents (into {} (map (juxt :name #(extent (:extent %) scalars))) segment-axes)
        n-destinations (get axis-extents :destination)
        n-heads (get axis-extents :head)
        n-edges (extent (:edges membership) scalars)
        components (extent (:components value) scalars)
        total-dim (extent (:total-dim value) scalars)
        destination-id (:destination-indices membership)
        source-id (:source-indices membership)
        score-arguments (mapv #(argument-value % scalars) (:arguments score))
        edges-by-destination (vec (repeat n-destinations []))
        edges-by-destination
        (reduce (fn [rows edge]
                  (let [destination (index-element buffers destination-id edge)
                        source (index-element buffers source-id edge)]
                    (when-not (and (< -1 destination n-destinations)
                                   (< -1 source n-destinations))
                      (fail "indexed attention edge endpoint is out of bounds"
                            :segmented-reduction-reference-edge-bounds
                            {:edge edge :source source :destination destination
                             :entities n-destinations}))
                    (update rows destination conj [edge source])))
                edges-by-destination (range n-edges))
        output (double-array (* n-destinations total-dim))
        score-components (extent (get-in score [:axis :extent]) scalars)
        left-id (get-in score [:left :buffer])
        right-id (get-in score [:right :buffer])
        value-id (:buffer value)
        epsilon (double (:epsilon normalization))]
    (when-not (= components score-components)
      (fail "indexed reference currently requires equal score and value head slices"
            :segmented-reduction-reference-unequal-head-components
            {:score-components score-components :value-components components}))
    (when (> (* n-heads components) total-dim)
      (fail "indexed head slices exceed the declared row width"
            :segmented-reduction-reference-head-layout-overflow
            {:heads n-heads :components components :total-dim total-dim}))
    (doseq [destination (range n-destinations)
            head (range n-heads)]
      (let [members (nth edges-by-destination destination)
            weighted-members
            (mapv
             (fn [[edge source]]
               (let [head-offset (* head components)
                     dot (reduce
                          (fn [acc component]
                            (+ acc
                               (region-value
                                (:combine score)
                                [(element buffers left-id
                                          (+ (* destination total-dim)
                                             head-offset component))
                                 (element buffers right-id
                                          (+ (* source total-dim)
                                             head-offset component))])))
                          (double (:identity denominator))
                          (range score-components))
                     score-value (region-value (:finalize score)
                                               (into [dot] score-arguments))
                     weight-value (region-value weight [score-value])]
                 {:edge edge :source source :weight weight-value}))
             members)
            denominator-value
            (reduce (fn [acc member]
                      (+ acc (region-value (:map-region denominator)
                                           [(:weight member)])))
                    (double (:identity denominator)) weighted-members)]
        (doseq [component (range components)]
          (let [numerator-value
                (reduce
                 (fn [acc {:keys [source weight]}]
                   (+ acc
                      (region-value
                       (:map-region numerator)
                       [weight
                        (element buffers value-id
                                 (+ (* source total-dim) (* head components) component))])))
                 (double (:identity numerator)) weighted-members)
                result (if (zero? denominator-value)
                         (double (:empty-result normalization))
                         (/ numerator-value (+ denominator-value epsilon)))
                output-index (+ (* destination total-dim) (* head components) component)]
            (aset output output-index result)))))
    output))
