(ns raster.compiler.core.scalar-conversion
  "Conversion policies derived from authoritative scalar dtypes, not source-expression inference.
   This is a shared lowering helper, not another IR or a target/type registry."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]))

(def scalar-convert-symbol
  "The canonical typed scalar-conversion term head. Kept below the IR dialect so proof layers can
   inspect the shared scalar vocabulary without introducing an IR namespace cycle."
  'raster.compiler.ir.soac-dialect/scalar-convert)

(defn policy
  "Return [rounding overflow], or nil when this policy cannot represent the conversion.

   Integral narrowing is rejected unless the caller explicitly requests :wrap or :trap. Those
   options distinguish representation wrapping from a checked source conversion; the default
   remains rejection. Floating-to-integral source casts use the JVM/Clojure boundary contract:
   truncate toward zero, map NaN to zero, and saturate out-of-range values. Identity callers may
   elide the returned exact conversion. Unknown dtypes or policy options fail loudly."
  ([source target] (policy source target :reject))
  ([source target integral-narrowing]
   (when-not (contains? #{:reject :wrap :trap} integral-narrowing)
     (throw (ex-info "unsupported integral narrowing policy"
                     {:reason :unsupported-scalar-conversion-policy
                      :integral-narrowing integral-narrowing})))
   (let [source (dtype/canon source)
         target (dtype/canon target)
         fp-source? (dtype/fp-dtype? source)
         fp-target? (dtype/fp-dtype? target)
         widening? (<= (dtype/bytes-of source) (dtype/bytes-of target))]
     (cond
       (= source target) [:exact :exact]
       (and fp-source? fp-target?)
       (if widening? [:exact :exact] [:nearest-even :ieee])

       (and (not fp-source?) (not fp-target?))
       (cond widening? [:exact :exact]
             (= :wrap integral-narrowing) [:exact :wrap]
             (= :trap integral-narrowing) [:exact :trap])

       (and (not fp-source?) fp-target?)
       (if (and (= :double target) (<= (dtype/bytes-of source) 4))
         [:exact :exact]
         [:nearest-even (if (= :half target) :ieee :exact)])

       (and fp-source? (not fp-target?))
       [:toward-zero :saturate]

       :else nil))))

(defn canonical-attributes?
  "Whether attributes fully and consistently describe one canonical scalar conversion."
  [{:keys [source-dtype target-dtype rounding overflow source-op] :as attributes}]
  (let [narrowing (descriptor/cast-integral-narrowing source-op)
        requested (case narrowing :reject :trap :wrap :wrap nil)]
    (and (map? attributes)
         (= #{:source-dtype :target-dtype :rounding :overflow :source-op}
            (set (keys attributes)))
         (every? #(and (keyword? %) (dtype/known? %) (= % (dtype/canon %)))
                 [source-dtype target-dtype])
         (descriptor/cast-op? source-op)
         (= target-dtype
            (some-> source-op descriptor/cast-result-tag dtype/dtype-for-scalar-tag dtype/canon))
         (= (policy source-dtype target-dtype requested) [rounding overflow]))))

(defn canonical-form?
  "Whether value has the canonical scalar-conversion term shape and a valid contract."
  [value]
  (and (seq? value)
       (= scalar-convert-symbol (first value))
       (= 3 (count value))
       (canonical-attributes? (second value))))

(defn canonical-parts
  "Project a validated canonical conversion into its attributes and operand."
  [value]
  (when (canonical-form? value)
    {:attributes (second value) :operand (nth value 2)}))

(defn project-canonical-to-source
  "Replace only validated canonical conversions with their descriptor-owned source casts.

   Every other node and operand tree is retained, so this projection cannot erase effects or
   substitute a different scalar recurrence. Invalid terms using the reserved head fail loudly."
  [expression]
  (util/postwalk-preserving-meta
   (fn [form]
     (if (and (seq? form) (= scalar-convert-symbol (first form)))
       (let [{:keys [attributes operand]} (canonical-parts form)]
         (when-not attributes
           (throw (ex-info "invalid canonical scalar conversion"
                           {:reason :invalid-canonical-scalar-conversion
                            :conversion form})))
         (with-meta (list (:source-op attributes) operand) (meta form)))
       form))
   expression))

(defn verified-source-projection
  "Return the canonical source projection, rejecting different caller-supplied evidence."
  ([expression]
   (project-canonical-to-source expression))
  ([expression supplied]
   (let [verified (project-canonical-to-source expression)]
     (when (and supplied (not= supplied verified))
       (throw (ex-info "supplied scalar projection is not the canonical source projection"
                       {:reason :scalar-conversion-projection-mismatch
                        :expression expression :supplied supplied :verified verified})))
     verified)))
