(ns raster.ad.normalized-program-bundle
  "Canonical TypedSOAC program envelope, initially adapting one checked homogeneous map.
   Scalar derivation evidence remains nested; no rules, evaluator, cache or source importer
   are added. This migration adapter does not compose equations or add runtime support."
  (:require [raster.ad.normalized-scalar :as scalar]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]))

(defn- fail! [reason data]
  (throw (ex-info "Invalid normalized program bundle" (assoc data :reason reason))))

(defn- checked-map! [bundle]
  (scalar/validate! bundle)
  (when-not (and (:map-programs bundle)
                 (not (contains? bundle :sum-programs))
                 (not (contains? bundle :gather-programs)))
    (fail! :ad-program-bundle-scope {}))
  bundle)

(defn- value-interface [program id source]
  {:id id :source source :value (get-in (soac/facts program) [:values id])})

(defn- map-envelope
  "Pure fixed-ID projection of already checked evidence. It generates no derivative code."
  [bundle]
  (let [{:keys [source primal linear transpose interface]} (:map-programs bundle)
        {:keys [active-array tangent-input tangent-output cotangent-input gradient-output
                residual-arrays scalar-captures]} interface
        inputs (:inputs (soac/facts source))
        result (first (soac/outputs source))
        producer (second (first (soac/equations primal)))
        source-equation (second (first (soac/equations source)))
        residual-bindings
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
                    scalar-captures)))]
    {:source source :primal primal :linear linear :transpose transpose
     :interface {:inputs (mapv #(value-interface source % %) inputs)
                 :active-inputs [active-array]
                 :primal-result (value-interface primal result result)
                 :tangent-input (value-interface linear tangent-input active-array)
                 :tangent-result (value-interface linear tangent-output result)
                 :cotangent-input (value-interface transpose cotangent-input result)
                 :cotangent-result (value-interface transpose gradient-output active-array)}
     :residual-bindings residual-bindings :physical-bindings {}
     :derivation {:kind :checked-scalar-map :bundle bundle}
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
  (when-not (and (= :checked-scalar-map (:kind derivation)) (map? (:bundle derivation)))
    (fail! :ad-program-bundle-derivation {}))
  (namespace! bundle)
  (let [evidence (checked-map! (:bundle derivation))
        expected (map-envelope evidence)]
    ;; Existing scalar validation already reconstructs lift-map with fixed interface IDs and
    ;; validates each SOAC. Compare the canonical projection, not a newly generated derivative.
    (when-not (fingerprint/equivalent? (pure-envelope expected) (pure-envelope bundle))
      (fail! :ad-program-bundle-correspondence {}))
    (when-not (= (:certificate bundle) (certificate bundle))
      (fail! :ad-program-bundle-certificate {})))
  bundle)

(defn adapt-map
  "Adapt one existing admitted map to canonical TypedSOAC primal/linear/transpose programs.
   Existing map APIs/runtimes remain unchanged; sum/indexed/composed programs are not admitted.
   No allocation or execution occurs. The evidence is transitional and may be retired only
   after a common program validator owns the same source/derivative/ownership obligations."
  [bundle]
  (let [result (map-envelope (checked-map! bundle))]
    (validate! (assoc result :certificate (certificate result)))))
