(ns raster.compiler.passes.parallel.staged-scalar-admission
  "Non-emitting contracts shared by floating staged schedule selection and construction."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.ir.contraction-closure :as closure]
            [raster.compiler.ir.contraction-facts :as facts]))

(defn decline! [rule message data]
  (throw (ex-info message (assoc data :reason :staged-scalar-body-declined :missing-rule rule))))

(defn analyze!
  "Analyze static floating-stage storage and scalar capture contracts without emission.
   Integer stage arithmetic needs its own overflow proof; packed Int32 already has one."
  [source & {:keys [scalar-types workgroup-size] :or {scalar-types {} workgroup-size 64}}]
  (when-not (facts/facts? source)
    (throw (ex-info "scalar staged body requires verified contraction facts" {:reason :raster/bug})))
  ;; A valid source can exceed this schedule's domain. Keep these capability misses
  ;; distinct from malformed lexical closures, which validate! must still reject.
  (when-not (every? #(and (integer? (second %)) (pos? (second %)))
                    (concat (:free-axes source) (:contract-axes source)))
    (decline! :static-domain "scalar staged schedule requires static extents" {:source source}))
  (when-not (and (contains? '#{+ clojure.core/+ raster.numeric/+} (:combine source))
                 (constant/zero-value? (:init source))
                 (empty? (get-in source [:opts :decode]))
                 (not-any? :decode (:operands source)))
    (decline! :numerical-contract "scalar staged schedule does not implement these numerical extensions"
              {:source source}))
  (let [{:keys [reads scalars]} (facts/dependencies source)
        _ (when (some #(= (:out source) (:sym %)) (get-in source [:epilogue :operands]))
            (decline! :result-transform-inout
                      "destination-reading result transforms require an inout storage proof"
                      {:source source}))
        attributes {:contraction source :array-parameters (vec (sort-by pr-str reads))
                    :capture-parameters (vec (sort-by pr-str scalars))}
        _ (closure/validate! attributes)
        _ (closure/validate-result-scalar-types! source scalar-types)
        requirements (closure/storage-requirements attributes)
        stage-list (:stages source)
        axes (vec (concat (:free-axes source) (:contract-axes source)))
        n (reduce *' 1 (map second (:free-axes source)))
        out-type (dtype/canon (:dtype (first stage-list)))
        _ (when-not (and (= out-type (dtype/canon (or (:out-dtype source) out-type)))
                         (every? #(= 1 (count (set (map :dtype %))))
                                 (vals (group-by :parameter requirements))))
            (decline! :storage-types "scalar staged storage requires one declared dtype per buffer"
                      {:requirements requirements :out-dtype (:out-dtype source)}))
        _ (when-not (and (every? #(contains? #{:float :double} (dtype/canon %))
                                 (concat [(:dtype source)] (map :dtype stage-list)))
                         (every? #(contains? scalar-types %) scalars)
                         (integer? workgroup-size) (<= 1 workgroup-size 256)
                         (every? #(<= (second %) Integer/MAX_VALUE) axes)
                         (<= (*' workgroup-size (quot (+ n (dec workgroup-size)) workgroup-size))
                             Integer/MAX_VALUE)
                         (every? #(<= (:elements %) Integer/MAX_VALUE) requirements))
            (decline! :static-floating-domain "scalar staged domain is not proved"
                      {:source source :scalar-types scalar-types}))
        sizes (into {} (map (fn [[id rs]] [id (apply max (map :elements rs))]))
                    (group-by :parameter requirements))
        array-types (into {} (map (juxt :parameter :dtype)) requirements)
]
    {:reads reads
     :attributes attributes
     :stage-list stage-list
     :axes axes
     :n n
     :out-type out-type
     :sizes sizes
     :array-types array-types}))
