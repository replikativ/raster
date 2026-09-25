(ns raster.compiler.passes.parallel.typed-soac-resident
  "Realize non-escaping TypedSOAC reduction scalars as resident one-element buffers.

   Logical reduction results remain rank-zero AbstractValues.  Their representation records the
   physical resident realization, scheduled reductions own the buffer output role, and consuming
   scalar regions load element zero through an explicit stable capture.  Pure scalar equations
   depending on a resident reduction are beta-reduced into their parallel consumers, so no device
   value is reconstructed or synchronized through host scalar control."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]))

(def resident-representation :resident-scalar-buffer)

(defn resident-scalar-value?
  [value]
  (= resident-representation (get-in value [:representation :kind])))

(defn- operation-info
  [equation]
  (or (fusion/equation-info equation)
      (throw (ex-info "resident realization received an unknown TypedSOAC equation"
                      {:reason :typed-soac-resident-equation :equation equation}))))

(defn- emit-equation
  [{:keys [kind id results attributes arrays captures destinations parameters locals body-results]}]
  (list '= id (vec results)
        (case kind
          :contract (list 'contract attributes (vec arrays) (vec captures))
          :scalar (list 'scalar attributes (vec captures)
                        (dialect/lambda-form (vec parameters) (dialect/emit-locals locals)
                                             (vec body-results)))
          :effect-map (list 'effect-map attributes (vec arrays) (vec captures)
                            (vec destinations)
                            (dialect/effect-lambda-form
                             (vec parameters) (dialect/emit-locals locals)
                             (vec body-results)))
          (list (symbol (name kind)) attributes (vec arrays) (vec captures)
                (dialect/lambda-form (vec parameters) (dialect/emit-locals locals)
                                     (vec body-results))))))

(defn- parameter-parts
  [info]
  (dialect/parameter-layout (emit-equation info)))

(defn- result-transform-inputs
  [attributes]
  (let [{:keys [operands scalars]} (:result-transform attributes)]
    (map :value (concat operands scalars))))

(defn- use-sites
  [equations]
  (reduce
   (fn [uses equation]
     (let [{:keys [id kind arrays captures attributes]} (operation-info equation)]
       (-> uses
           (into (map (fn [value] [value {:equation id :role :array}]) arrays))
           (into (map (fn [value] [value {:equation id :role (if (= :contract kind)
                                                             :contract-capture :capture)}]) captures))
           ;; The transform has its own typed scalar boundary. Rewriting the primary
           ;; lambda cannot turn a transform scalar into a resident buffer load.
           (into (map (fn [value] [value {:equation id :role :result-transform}])
                      (result-transform-inputs attributes)))
           (cond-> (dialect/value-id? (:extent attributes))
             (conj [(:extent attributes) {:equation id :role :extent}])))))
   [] equations))

(defn- scalar-definitions
  [equations]
  (into {}
        (keep (fn [equation]
                (let [info (operation-info equation)]
                  (when (and (= :scalar (:kind info))
                             (= 1 (count (:results info)))
                             (= 1 (count (:body-results info))))
                    [(first (:results info)) info]))))
        equations))

(defn- dependent-scalars
  [scalar-defs roots]
  (loop [dependent #{}]
    (let [known (set/union roots dependent)
          dependent'
          (into dependent
                (keep (fn [[result {:keys [captures]}]]
                        (when (some known captures) result)))
                scalar-defs)]
      (if (= dependent dependent') dependent (recur dependent')))))

(defn- scalar-expression
  [scalar-defs dependent roots id]
  (letfn [(expand [value visiting]
            (cond
              (contains? roots value)
              (list 'clojure.core/aget value 0)

              (contains? dependent value)
              (do
                (when (contains? visiting value)
                  (throw (ex-info "resident scalar equations contain a dependency cycle"
                                  {:reason :typed-soac-resident-scalar-cycle :value value})))
                (let [{:keys [captures parameters body-results]} (get scalar-defs value)
                      expression (util/subst-syms (zipmap parameters captures)
                                                  (first body-results))
                      substitutions (into {}
                                          (keep (fn [capture]
                                                  (when (or (contains? roots capture)
                                                            (contains? dependent capture))
                                                    [capture (expand capture (conj visiting value))])))
                                          captures)]
                  (util/subst-syms substitutions expression)))

              :else value))]
    (expand id #{})))

(defn- rewrite-lambda-captures
  "Substitute captured scalar equations without reconstructing a source-shaped kernel.
   `stable-captures` are newly introduced array reads that must keep the no-write-alias contract."
  [info values replacements stable-captures]
  (let [{:keys [accumulators elements capture-parameters destination-parameters]}
        (parameter-parts info)
        capture-substitutions
        (into {}
              (map (fn [[parameter capture]]
                     [parameter (get replacements capture capture)]))
              (map vector capture-parameters (:captures info)))
        global-locals (mapv #(update % :init
                                     (fn [init]
                                       (util/subst-syms capture-substitutions init)))
                            (:locals info))
        global-bodies (mapv #(util/subst-syms capture-substitutions %) (:body-results info))
        bound (set (concat accumulators elements destination-parameters
                           [(get-in info [:attributes :index])]
                           (map first (get-in info [:attributes :segment-axes]))))
        stable-before (set (get-in info [:attributes :attributes :stable-array-captures]))
        referenced-values (->> (concat (mapcat #(util/free-syms (:init %) bound) global-locals)
                                       (mapcat #(util/free-syms % bound) global-bodies)
                                       stable-before
                                       (result-transform-inputs (:attributes info)))
                               (filter #(contains? values %)) distinct (sort-by pr-str) vec)
        new-parameters (mapv #(symbol (str "%capture" %)) (range (count referenced-values)))
        body-substitutions (zipmap referenced-values new-parameters)
        stable-after (set (filter #(or (contains? stable-before %)
                                       (contains? stable-captures %)
                                       (resident-scalar-value? (get values %)))
                                  referenced-values))]
    (assoc info
           :captures referenced-values
           :parameters (vec (concat accumulators elements new-parameters
                                    destination-parameters))
           :locals (mapv #(update % :init
                                  (fn [init]
                                    (util/subst-syms body-substitutions init)))
                         global-locals)
           :body-results (mapv #(util/subst-syms body-substitutions %) global-bodies)
           :attributes (assoc-in (:attributes info) [:attributes :stable-array-captures]
                                 (vec (filter stable-after referenced-values))))))

(defn- rewrite-lambda-consumer
  [info values scalar-defs dependent roots]
  (let [replacements
        (into {}
              (keep (fn [capture]
                      (cond
                        (contains? roots capture)
                        [capture (list 'clojure.core/aget capture 0)]
                        (contains? dependent capture)
                        [capture (scalar-expression scalar-defs dependent roots capture)])))
              (:captures info))]
    (rewrite-lambda-captures info values replacements #{})))

(defn- rewrite-consumer
  [info values scalar-defs dependent roots]
  ;; Only a lexical consumer of a realized/dependent scalar needs a new buffer load. Besides
  ;; avoiding needless alpha-renaming, this keeps unrelated host scalar equations entirely out
  ;; of the parallel parameter-layout machinery (their lambda may intentionally have no element
  ;; or accumulator partition). A contraction's scalar closure is likewise not a resident-buffer
  ;; lambda; its capture uses are explicit escape sites above.
  (let [affected (set/union dependent roots)]
    (if (or (= :contract (:kind info))
            (empty? (set/intersection affected (set (:captures info)))))
      info
      (rewrite-lambda-consumer info values scalar-defs dependent roots))))

(defn- uniform-input-load
  "Recognize an ordered scalar equation that reads one immutable program input.
   A launch extent or host-visible result is never a candidate for this transformation."
  [info input-values values]
  (let [{:keys [kind results captures parameters locals body-results]} info
        expression (first body-results)
        arguments (when (descriptor/aget-call? expression)
                    (vec (descriptor/call-args expression)))
        array (first captures)
        index (second arguments)]
    (when (and (= :scalar kind) (= 1 (count results)) (= 1 (count captures))
               (= 1 (count parameters)) (empty? locals) (= 1 (count body-results))
               (= 2 (count arguments)) (= (first arguments) (first parameters))
               (integer? index) (not (neg? index))
               (contains? input-values array)
               (seq (:shape (get values array))))
      {:result (first results) :array array :index index})))

(defn inline-uniform-input-loads
  "Move a single-use uniform input load into its resident map/effect-map consumer.

   The input remains a stable, no-write-alias array capture. This is a device-side read of the
   currently bound buffer, not a host-side download or a cached copy of allocation-time data.
   Extent values, host-visible scalar results, non-input storage and arrays written anywhere in
   this program remain ordered scalar equations. The JVM route does not apply this pass."
  [program]
  (let [program (dialect/validate! program)
        facts (dialect/facts program)
        equations (dialect/equations program)
        infos (mapv operation-info equations)
        input-values (set (:inputs facts))
        outputs (set (dialect/outputs program))
        uses (group-by first (use-sites equations))
        kinds (into {} (map (juxt :id :kind)) infos)
        written (set (concat (mapcat :destinations infos)
                             (mapcat (fn [equation]
                                       (map :destination
                                            (or (dialect/result-storage facts (second equation)) [])))
                                     equations)))
        candidates
        (into {}
              (keep (fn [info]
                      (when-let [{:keys [result array] :as load}
                                 (uniform-input-load info input-values (:values facts))]
                        (let [sites (get uses result)]
                          (when (and (not (contains? outputs result))
                                     (not (contains? written array))
                                     (= 1 (count sites))
                                     (= :capture (get-in (first sites) [1 :role]))
                                     (contains? #{:map :effect-map}
                                                (get kinds (get-in (first sites) [1 :equation]))))
                            [result load])))))
              infos)]
    (if (empty? candidates)
      [program {:resident-uniform-input-loads 0}]
      (let [removed (set (keys candidates))
            rewritten
            (->> infos
                 (remove #(and (= :scalar (:kind %)) (some removed (:results %))))
                 (mapv (fn [info]
                         (if (and (contains? #{:map :effect-map} (:kind info))
                                  (some removed (:captures info)))
                           (let [loads (keep candidates (:captures info))
                                 replacements (into {}
                                                    (map (fn [{:keys [result array index]}]
                                                           [result (list 'clojure.core/aget array index)]))
                                                    loads)
                                 stable (set (map :array loads))]
                             (rewrite-lambda-captures info (:values facts) replacements stable))
                           info)))
                 (mapv emit-equation))
            equation-ids (set (map second rewritten))
            definitions (set (mapcat #(nth % 2) rewritten))
            references (set (mapcat (fn [equation]
                                      (cond-> (dialect/operation-inputs equation)
                                        (dialect/value-id? (dialect/operation-extent equation))
                                        (conj (dialect/operation-extent equation))))
                                    rewritten))
            inputs (vec (sort-by pr-str (set/difference references definitions)))
            storage (set (mapcat (fn [equation]
                                   (map :destination
                                        (or (dialect/result-storage facts (second equation)) [])))
                                 rewritten))
            live-values (set/union definitions references storage outputs)
            facts (-> facts
                      (assoc :values (select-keys (:values facts) live-values) :inputs inputs)
                      (update :equations select-keys equation-ids)
                      (assoc-in [:attributes :resident-uniform-input-loads]
                                (vec (sort-by pr-str removed))))]
        [(dialect/make facts rewritten (dialect/outputs program))
         {:resident-uniform-input-loads (count removed)}]))))

(defn realize
  "Return `[program stats]`, realizing every eligible non-escaping scalar reduction.

   A root declines realization when it is a program result or is used as an element array,
   extent, or result-transform input (which requires a separate scalar-load schedule).
   Scalar chains depending on eligible roots must likewise remain internal and capture-only."
  [program]
  (let [program (dialect/validate! program)
        facts (dialect/facts program)
        equations (dialect/equations program)
        infos (mapv operation-info equations)
        outputs (set (dialect/outputs program))
        uses (group-by first (use-sites equations))
        stored-reduction-destinations
        (into {}
              (mapcat (fn [equation]
                        (when (= 'reduce (dialect/operation-kind equation))
                          (map (fn [result storage] [result (:destination storage)])
                               (nth equation 2)
                               (or (dialect/result-storage facts (second equation)) [])))))
              equations)
        stored-reduction-results (set (keys stored-reduction-destinations))
        candidate-roots
        (set (mapcat (fn [{:keys [kind results]}]
                       (when (= :reduce kind)
                         (filter (fn [result]
                                   (and (or (not (contains? outputs result))
                                            (contains? stored-reduction-results result))
                                        (every? #(= :capture (get-in % [1 :role]))
                                                (get uses result []))))
                                 results)))
                     infos))
        scalar-defs (scalar-definitions equations)
        dependent (dependent-scalars scalar-defs candidate-roots)
        escaping-dependent
        (set (filter (fn [value]
                       (or (contains? outputs value)
                           (not-every? #(= :capture (get-in % [1 :role]))
                                       (get uses value []))))
                     dependent))
        blocked-roots
        (set (filter (fn [root]
                       (some (fn [value]
                               (let [expression (scalar-expression scalar-defs dependent
                                                                   candidate-roots value)]
                                 (contains? (util/free-syms expression) root)))
                             escaping-dependent))
                     candidate-roots))
        roots (set/difference candidate-roots blocked-roots)
        dependent (dependent-scalars scalar-defs roots)
        removed-scalars (set/difference dependent escaping-dependent)]
    (if (empty? roots)
      [program {:resident-reductions 0 :inlined-scalars 0}]
      (let [values (reduce (fn [vs root]
                             (let [destination (get stored-reduction-destinations root)
                                   mark-resident
                                   (fn [values id]
                                     (cond-> values
                                       id (assoc-in [id :representation]
                                                    {:kind resident-representation :elements 1})
                                       id (assoc-in [id :memory-space] :device)))]
                               (-> vs
                                   (mark-resident root)
                                   (mark-resident destination))))
                           (:values facts) roots)
            rewritten
            (->> infos
                 (remove #(and (= :scalar (:kind %))
                               (some removed-scalars (:results %))))
                 (mapv #(rewrite-consumer % values scalar-defs removed-scalars roots))
                 (mapv emit-equation))
            equation-ids (set (map second rewritten))
            definitions (set (mapcat #(nth % 2) rewritten))
            references (set (mapcat (fn [equation]
                                      (cond-> (dialect/operation-inputs equation)
                                        (dialect/value-id? (dialect/operation-extent equation))
                                        (conj (dialect/operation-extent equation))))
                                    rewritten))
            inputs (vec (sort-by pr-str (set/difference references definitions)))
            storage-destinations
            (set (mapcat (fn [equation]
                           (map :destination
                                (or (dialect/result-storage facts (second equation)) [])))
                         rewritten))
            live-values (set/union definitions references storage-destinations outputs)
            facts (-> facts
                      (assoc :values (select-keys values live-values)
                             :inputs inputs)
                      (update :equations select-keys equation-ids)
                      (update :equations
                              (fn [equation-facts]
                                (reduce (fn [m equation]
                                          (let [id (second equation)
                                                results (set (nth equation 2))]
                                            (if (seq (set/intersection roots results))
                                              (assoc-in m [id :attributes :resident-realization]
                                                        {:kind resident-representation :elements 1})
                                              m)))
                                        equation-facts rewritten)))
                      (assoc-in [:attributes :resident-reductions] (vec (sort-by pr-str roots))))
            result (dialect/make facts rewritten (dialect/outputs program))]
        [result {:resident-reductions (count roots)
                 :inlined-scalars (count removed-scalars)}]))))
