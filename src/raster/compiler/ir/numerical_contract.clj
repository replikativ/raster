(ns raster.compiler.ir.numerical-contract
  "Shared numerical attestation for verified schedule refinements.

   This value is producer evidence, not an inferred floating-point proof.  Scheduling passes state
   whether they preserve evaluation exactly, reassociate a known accumulator, or accept a named
   error model.  IR boundaries consume the same validator so numerical policy cannot drift between
   graph and single-kernel refinements."
  (:require [raster.compiler.core.dtype :as dtype]))

(def modes #{:exact :reassociated :bounded-error})
(def rounding-policies
  #{:nearest-even :toward-zero :up :down :exact :implementation-defined})

(def cast-overflow-policies
  #{:wrap :saturate :trap :exact :ieee})

(def retained-source-arithmetic {:kind :retained-typed-ssa})

(def target-library-math
  "A typed target math call, not a JVM-bitwise or correctly-rounded attestation.
   Implementation-defined accuracy is descriptive evidence, never a bounded-error proof."
  {:kind :target-library :accuracy :implementation-defined})

(defn target-library-math?
  [value]
  (= target-library-math value))

(defn blas-source-arithmetic
  "Describe a resolved BLAS product, not permission for a numerical refinement.
   Operand conversion and the typed result transform are outside the implementation-defined
   association/rounding of the product reduction. The overload's element dtype fixes the floor."
  [element-dtype]
  (when-not (contains? #{:float :double} element-dtype)
    (throw (ex-info "BLAS source arithmetic requires a resolved Float or Double overload"
                    {:reason :source-arithmetic :dtype element-dtype})))
  {:kind :abstract-blas-product
   :operands {:dtype element-dtype :conversion :identity}
   :accumulation {:scope :reduction-intermediates :minimum-dtype element-dtype
                  :source-order :implementation-defined
                  :rounding-points :implementation-defined}
   :result-transform :retained-typed-ssa})

(defn source-arithmetic?
  "Closed descriptive source schema. It never grants schedule or mixed-precision consent."
  [value]
  (or (= retained-source-arithmetic value)
      (and (map? value)
           (contains? #{:float :double} (get-in value [:operands :dtype]))
           (= value (blas-source-arithmetic (get-in value [:operands :dtype]))))))

(defn validate-source-arithmetic!
  [value]
  (when-not (source-arithmetic? value)
    (throw (ex-info "invalid source arithmetic contract"
                    {:reason :source-arithmetic :value value})))
  value)

(defn- declared-accumulation-preserves-source-floor?
  "A restrictive check on declared product accumulators, never an admission predicate.
   Exact ordered schedules retain their component types in SSA rather than inventing a global
   accumulator dtype. Approximate physical operand conversions have separate operational models."
  [source contract]
  (let [floor (get-in source [:accumulation :minimum-dtype])
        accepted (case floor :float #{:float :double} :double #{:double})
        declared (cond-> (mapv :dtype (:accumulators contract))
                   (contains? contract :accumulator-dtype) (conj (:accumulator-dtype contract)))]
    (or (empty? declared) (every? accepted declared))))

(defn rounding-policy?
  [value]
  (contains? rounding-policies value))

(defn cast-overflow-policy?
  [value]
  (contains? cast-overflow-policies value))

(defn- canonical-dtype?
  [value]
  (and (dtype/known? value) (= value (dtype/canon value))))

(defn- accumulator-component?
  [component]
  (and (map? component)
       (some? (:value component))
       (canonical-dtype? (:dtype component))
       (contains? rounding-policies (:rounding component))
       (keyword? (:policy component))))

(defn- component-accumulators?
  [contract]
  (let [components (:accumulators contract)]
    (and (vector? components)
         (seq components)
         (every? accumulator-component? components)
         (= (count components) (count (set (map :value components)))))))

(defn- result-transform?
  [transform]
  (and (map? transform)
       (= :typed-scalar-region (:kind transform))
       (= :same-typed-ssa-evaluation-order (:policy transform))
       (canonical-dtype? (:input-dtype transform))
       (canonical-dtype? (:result-dtype transform))))

(defn validate!
  "Validate and return a numerical contract.

   Exact evaluation order preserves the typed operations, not bitwise equivalence between
   different target math libraries. Covered scalar math operations (currently exp) retain their
   realization separately in KernelBody; unclassified transcendentals have no such attestation.

   `context` lets an owning IR preserve its public diagnostic identity while sharing this one
   contract."
  ([contract] (validate! contract {}))
  ([contract {:keys [reason ir]
              :or {reason :numerical-contract ir :numerical-contract}}]
   (letfn [(fail! [message data]
             (throw (ex-info message (assoc data :reason reason :ir ir))))]
     (when-not (and (map? contract)
                    (contains? modes (:mode contract))
                    (keyword? (:policy contract)))
       (fail! "numerical contract requires a supported mode and named policy"
              {:value contract :supported modes}))
     (case (:mode contract)
       :exact nil
       :reassociated
       (when-not (or (and (contains? rounding-policies (:rounding contract))
                          (canonical-dtype? (:accumulator-dtype contract)))
                     (component-accumulators? contract))
         (fail! "reassociated numerical contract requires one accumulator or checked components"
                {:value contract}))
       :bounded-error
       (when-not (and (contains? rounding-policies (:rounding contract))
                      (canonical-dtype? (:accumulator-dtype contract))
                      (map? (:error-model contract))
                      (keyword? (get-in contract [:error-model :kind])))
         (fail! "bounded-error numerical contract requires rounding, accumulator dtype, and an error model"
                {:value contract})))
     (when (and (contains? contract :result-transform)
                (not (result-transform? (:result-transform contract))))
       (fail! "numerical result transform requires a checked typed scalar-region policy"
              {:value (:result-transform contract)}))
     (when (contains? contract :source-arithmetic)
       (let [source (validate-source-arithmetic! (:source-arithmetic contract))]
         (when (and (= :abstract-blas-product (:kind source))
                    (not (declared-accumulation-preserves-source-floor? source contract)))
           (fail! "declared product accumulation is below the source arithmetic precision floor"
                  {:source-arithmetic source
                   :accumulator-dtype (:accumulator-dtype contract)
                   :accumulators (:accumulators contract)}))))
     contract)))
