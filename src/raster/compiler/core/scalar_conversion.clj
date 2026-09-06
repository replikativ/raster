(ns raster.compiler.core.scalar-conversion
  "Conversion policies derived from authoritative scalar dtypes, not source-expression inference.
   This is a shared lowering helper, not another IR or a target/type registry."
  (:require [raster.compiler.core.dtype :as dtype]))

(defn policy
  "Return [rounding overflow], or nil when this policy cannot represent the conversion.

   Integral narrowing is rejected unless the owner explicitly requests :wrap. That opt-in
   preserves an existing device representation policy; it does not prove equivalence to a
   checked Clojure cast. Floating-to-integral conversions remain unsupported. Identity callers
   may elide the returned exact conversion. Unknown dtypes or policy options fail loudly."
  ([source target] (policy source target :reject))
  ([source target integral-narrowing]
   (when-not (contains? #{:reject :wrap} integral-narrowing)
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
             (= :wrap integral-narrowing) [:exact :wrap])

       (and (not fp-source?) fp-target?)
       (if (and (= :double target) (<= (dtype/bytes-of source) 4))
         [:exact :exact]
         [:nearest-even (if (= :half target) :ieee :exact)])

       :else nil))))
