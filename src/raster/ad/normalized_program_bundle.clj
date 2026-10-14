(ns raster.ad.normalized-program-bundle
  "Canonical TypedSOAC program envelope for checked maps, ordered map sums and indexed maps.
   Scalar derivation evidence remains nested; no rules, evaluator, cache or source importer
   are added. This migration adapter does not compose equations or add runtime support."
  (:require [raster.ad.normalized-scalar :as scalar]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]))

(defn- fail! [reason data]
  (throw (ex-info "Invalid normalized program bundle" (assoc data :reason reason))))

(defn- evidence-kind! [bundle]
  (let [facets (select-keys bundle [:map-programs :sum-programs :gather-programs])
        kind (case (set (keys facets))
               #{:map-programs} :checked-scalar-map
               #{:map-programs :sum-programs} :checked-scalar-map-sum
               #{:gather-programs} :checked-scalar-indexed-map
               nil)]
    (when-not (and kind (every? map? (vals facets)))
      (fail! :ad-program-bundle-scope {:facets (set (keys facets))}))
    kind))

(defn- checked-evidence! [bundle kind]
  (when-not (= kind (evidence-kind! bundle))
    (fail! :ad-program-bundle-scope {:kind kind}))
  (scalar/validate! bundle)
  bundle)

(defn- value-interface [program id source]
  {:id id :source source :value (get-in (soac/facts program) [:values id])})

(defn- program-envelope
  "Pure fixed-ID projection of already checked evidence. It generates no derivative code."
  [bundle kind]
  (when (and (= :checked-scalar-map kind)
             (not= 1 (count (soac/equations (get-in bundle [:map-programs :source])))))
    ;; Map-chain evidence is intentionally construction-only until canonical residual origins
    ;; can identify the exact original producer equation. Do not let the single-map projection
    ;; attribute every fused-stage residual to the first source equation.
    (fail! :ad-program-bundle-composition
           {:equations (count (soac/equations (get-in bundle [:map-programs :source])))}))
  (let [sum? (= :checked-scalar-map-sum kind)
        indexed? (= :checked-scalar-indexed-map kind)
        programs (cond indexed? (:gather-programs bundle)
                       sum? (:sum-programs bundle) :else (:map-programs bundle))
        {:keys [source primal linear transpose required-zero-gradient]} programs
        {:keys [active-input tangent-input tangent-output cotangent-input gradient-output
                residual-arrays scalar-captures]} (get-in bundle [:map-programs :interface])
        indexed-interface (:interface programs)
        source-parts (when indexed? (soac/operation-parts (first (soac/equations source))))
        index-input (when indexed? (first (:arrays source-parts)))
        active-input (if indexed? (first (:captures source-parts)) active-input)
        tangent-input (if indexed? (:tangent-input indexed-interface) tangent-input)
        gradient-output (if indexed? (:gradient-output indexed-interface) gradient-output)
        gradient-output (if sum? (get-in programs [:interface :gradient-result]) gradient-output)
        tangent-output (if sum? (get-in bundle [:sum-programs :interface :tangent-result]) tangent-output)
        tangent-output (if indexed? (:tangent-output indexed-interface) tangent-output)
        cotangent-input (if sum? (get-in bundle [:sum-programs :interface :seed]) cotangent-input)
        cotangent-input (if indexed? (:cotangent-input indexed-interface) cotangent-input)
        inputs (:inputs (soac/facts source))
        result (first (soac/outputs source))
        producer (second (first (soac/equations primal)))
        source-equation (second (first (soac/equations source)))
        residual-bindings
        (if indexed?
          {index-input {:id (:index-snapshot indexed-interface) :storage :owned-array
                        :producer {:equation producer :result (:index-snapshot indexed-interface)}
                        :origin {:equation source-equation :input index-input}
                        :value (get-in (soac/facts primal) [:values (:index-snapshot indexed-interface)])}}
          (into {}
              (concat
               (map (fn [[coefficient id]]
                      [coefficient {:id id :storage :owned-array
                                    :producer {:equation producer :result id}
                                    :origin {:equation source-equation :scalar-value coefficient}
                                    :value (get-in (soac/facts primal) [:values id])}])
                    residual-arrays)
               (map (fn [[coefficient id]]
                      [coefficient {:id id :storage :immutable-scalar
                                    :producer {:input id}
                                    :origin {:equation source-equation :scalar-value coefficient}
                                    :value (get-in (soac/facts source) [:values id])}])
                    scalar-captures))))
        physical-bindings
        (if indexed?
          (let [id (:value required-zero-gradient)
                equation (second (first (soac/equations transpose)))]
            {id {:value (get-in (soac/facts transpose) [:values id])
                 :requirement required-zero-gradient
                 :consumer {:equation equation :destination id :result gradient-output
                            :alias (get-in (soac/facts transpose) [:equations equation :aliases gradient-output])}}})
          {})]
    {:source source :primal primal :linear linear :transpose transpose
     :interface {:inputs (mapv #(value-interface source % %) inputs)
                 :active-inputs [active-input]
                 :primal-result (value-interface primal result result)
                 :tangent-input (value-interface linear tangent-input active-input)
                 :tangent-result (value-interface linear tangent-output result)
                 :cotangent-input (value-interface transpose cotangent-input result)
                 :cotangent-result (value-interface transpose gradient-output active-input)}
     :residual-bindings residual-bindings :physical-bindings physical-bindings
     :derivation {:kind kind :bundle bundle}
     :source-ns (:source-ns bundle)}))

(defn- pure-envelope [bundle]
  ;; Namespace instances are runtime identity, not process-independent semantic fingerprint data.
  ;; Both identities are checked separately before this projection is used for validation.
  (-> bundle
      (dissoc :certificate)
      (assoc :source-ns (ns-name (:source-ns bundle)))
      (assoc-in [:derivation :bundle :source-ns]
                (ns-name (get-in bundle [:derivation :bundle :source-ns])))))

(defn- namespace! [{:keys [source-ns derivation]}]
  (when-not (and (instance? clojure.lang.Namespace source-ns)
                 (identical? source-ns (get-in derivation [:bundle :source-ns])))
    (fail! :ad-program-bundle-source-namespace {})))

(defn- certificate [bundle]
  (namespace! bundle)
  (fingerprint/fingerprint (pure-envelope bundle)))

(defn validate!
  "Check the original scalar authority and fixed-ID source/program/interface correspondence.
   This is not independent proof of custom-rule mathematics or floating reassociation."
  [{:keys [source-ns derivation] :as bundle}]
  (when-not (and (contains? #{:checked-scalar-map :checked-scalar-map-sum :checked-scalar-indexed-map}
                            (:kind derivation))
                 (map? (:bundle derivation)))
    (fail! :ad-program-bundle-derivation {}))
  (namespace! bundle)
  (let [kind (:kind derivation)
        evidence (checked-evidence! (:bundle derivation) kind)
        expected (program-envelope evidence kind)]
    ;; Existing scalar validation already reconstructs all admitted lifts with fixed interface IDs and
    ;; validates each SOAC. Compare the canonical projection, not a newly generated derivative.
    (when-not (fingerprint/equivalent? (pure-envelope expected) (pure-envelope bundle))
      (fail! :ad-program-bundle-correspondence {}))
    (when-not (= (:certificate bundle) (certificate bundle))
      (fail! :ad-program-bundle-certificate {})))
  bundle)

(defn- adapt-kind [bundle kind]
  (let [result (program-envelope (checked-evidence! bundle kind) kind)]
    ;; Retain independent final validation and epoch/definition checks. Consolidating
    ;; the boundary is not permission to skip derivation replay during preparation.
    (validate! (assoc result :certificate (certificate result)))))

(defn adapt
  "Admit exact checked evidence through the one canonical owner boundary.
   Missing, nil or mixed family facets refuse before materialization. No execution occurs."
  [bundle]
  (adapt-kind bundle (evidence-kind! bundle)))

(defn adapt-map
  "Adapt one existing admitted map to canonical TypedSOAC primal/linear/transpose programs.
   Existing map APIs/runtimes remain unchanged; sum/indexed/composed programs are not admitted.
   No allocation or execution occurs. The evidence is transitional and may be retired only
   after a common program validator owns the same source/derivative/ownership obligations."
  [bundle]
  (adapt-kind bundle :checked-scalar-map))

(defn adapt-map-sum
  "Adapt an existing checked map→ordered sum without changing its reduction or execution.
   Tangent input is an array, tangent result and explicit cotangent seed are scalars; snapshots
   retain their original tuple-map producers. Indexed/ambiguous evidence refuses. No new
   composition, rules, reassociation permission, physical obligations or runtime mode is added."
  [bundle]
  (adapt-kind bundle :checked-scalar-map-sum))

(defn adapt-indexed-map
  "Project existing checked functional indexed-map programs without changing execution.
   Index snapshots originate in the excluded index input, not a scalar derivative coefficient.
   Physical bindings preserve a required fresh/exclusive typed-zero gradient and its ordered
   destination/result alias; they are obligations, never allocation or initialization receipts.
   Data and result extents remain distinct. Map/sum mixtures refuse."
  [bundle]
  (adapt-kind bundle :checked-scalar-indexed-map))
