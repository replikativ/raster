(ns raster.compiler.passes.parallel.typed-soac-resident
  "Realize non-escaping TypedSOAC reduction scalars as resident one-element buffers.

   Logical reduction results remain rank-zero AbstractValues.  Their representation records the
   physical resident realization, scheduled reductions own the buffer output role, and consuming
   scalar regions load element zero through an explicit stable capture.  Pure scalar equations
   depending on a resident reduction are beta-reduced into their parallel consumers, so no device
   value is reconstructed or synchronized through host scalar control."
  (:require [clojure.set :as set]
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
  [{:keys [kind id results attributes arrays captures parameters locals body-results]}]
  (list '= id (vec results)
        (list (symbol (name kind)) attributes (vec arrays) (vec captures)
              (dialect/lambda-form (vec parameters) (dialect/emit-locals locals)
                                   (vec body-results)))))

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
     (let [{:keys [id arrays captures attributes]} (operation-info equation)]
       (-> uses
           (into (map (fn [value] [value {:equation id :role :array}]) arrays))
           (into (map (fn [value] [value {:equation id :role :capture}]) captures))
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

(defn- rewrite-consumer
  [info values scalar-defs dependent roots]
  (let [{:keys [accumulators elements capture-parameters]} (parameter-parts info)
        capture-substitutions
        (into {}
              (map (fn [[parameter capture]]
                     [parameter
                      (cond
                        (contains? roots capture) (list 'clojure.core/aget capture 0)
                        (contains? dependent capture)
                        (scalar-expression scalar-defs dependent roots capture)
                        :else capture)]))
              (map vector capture-parameters (:captures info)))
        global-locals (mapv #(update % :init
                                     (fn [init]
                                       (util/subst-syms capture-substitutions init)))
                            (:locals info))
        global-bodies (mapv #(util/subst-syms capture-substitutions %) (:body-results info))
        bound (set (concat accumulators elements [(get-in info [:attributes :index])]
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
                                       (resident-scalar-value? (get values %)))
                                  referenced-values))]
    (assoc info
           :captures referenced-values
           :parameters (vec (concat accumulators elements new-parameters))
           :locals (mapv #(update % :init
                                  (fn [init]
                                    (util/subst-syms body-substitutions init)))
                         global-locals)
           :body-results (mapv #(util/subst-syms body-substitutions %) global-bodies)
           :attributes (assoc-in (:attributes info) [:attributes :stable-array-captures]
                                 (vec (filter stable-after referenced-values))))))

(defn realize
  "Return `[program stats]`, realizing every eligible non-escaping scalar reduction.

   A root declines realization when it is a program result or is used as an element array,
   extent, or result-transform input (which requires a separate scalar-load schedule).
   Scalar chains depending on eligible roots must likewise remain internal and capture-only."
  [program]
  (let [program (dialect/validate! program)
        equations (dialect/equations program)
        infos (mapv operation-info equations)
        outputs (set (dialect/outputs program))
        uses (group-by first (use-sites equations))
        candidate-roots
        (set (mapcat (fn [{:keys [kind results]}]
                       (when (= :reduce kind)
                         (filter (fn [result]
                                   (and (not (contains? outputs result))
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
      (let [facts (dialect/facts program)
            values (reduce (fn [vs root]
                             (-> vs
                                 (assoc-in [root :representation]
                                           {:kind resident-representation :elements 1})
                                 (assoc-in [root :memory-space] :device)))
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
