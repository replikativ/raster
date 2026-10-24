(ns raster.compiler.core.source-policy
  "Definition-owned Clojure compilation policy, independent of value dtype or target ABI.
   Recording this context does not attest how an existing bytecode method was emitted.")

(defn valid? [policy]
  (and (map? policy) (= #{:unchecked-math} (set (keys policy)))
       (contains? #{false true :warn-on-boxed} (:unchecked-math policy))))

(defn capture
  "Capture at the actual source producer, never retrospectively at a consumer."
  []
  {:unchecked-math *unchecked-math*})

(defn unchecked-math
  "Validate an explicitly retained policy and project its compiler binding.
   Missing policy is not the checked policy."
  [policy]
  (when-not (valid? policy)
    (throw (ex-info "source compilation requires its retained policy"
                    {:reason :unclosed-source-policy :source-policy policy})))
  (:unchecked-math policy))

(defn call-with-policy [policy compile!]
  (binding [*unchecked-math* (unchecked-math policy)]
    (compile!)))

(defn call-with-owner
  "Use an owner's retained policy when present. Legacy policy-less walking keeps
   its prior behavior but gains NO policy evidence for a new owned projection."
  [metadata compile!]
  (if (contains? metadata :raster.core/source-policy)
    (call-with-policy (:raster.core/source-policy metadata) compile!)
    (compile!)))
