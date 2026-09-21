(ns raster.compiler.ir.scan
  "Certified scalar algebra for parallel scans and reductions.

   Raster's surface `par/scan` is also a general left-to-right recurrence primitive. Such a
   recurrence is not automatically a parallel scan: Blelloch/Hillis-Steele scheduling is sound
   only when the body is an associative combine of the prior accumulator and an accumulator-free
   element expression. This boundary proves that property before SegScan scheduling."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.scalar-conversion :as scalar-conversion]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.core.util :as util]))

(defrecord AssociativeScan [acc init combine element identity dtype])

(defn associative-scan? [x]
  (and x (= "raster.compiler.ir.scan.AssociativeScan" (.getName (class x)))))

(defn- acc-ref?
  [expr acc reduction-dtype]
  (or (= expr acc)
      (and (seq? expr)
           (= 2 (count expr))
           (= (dtype/canon reduction-dtype)
              (some-> (descriptor/cast-result-tag (first expr)) keyword dtype/canon))
           (= acc (second expr)))))

(defn- contains-symbol?
  [expr target]
  (cond
    (= expr target) true
    (coll? expr) (boolean (some #(contains-symbol? % target) expr))
    :else false))

(defn- pure-element?
  [expr]
  ;; Effect analysis is deliberately centralized in the Beichte-backed scalar-effects
  ;; boundary.  A second syntactic operator allow-list here would disagree with the
  ;; canonical registry for valid Raster operations (for example raster.numeric/*).
  (= :pure (effects/analyze-effect expr)))

(defn- reason
  [operation suffix]
  (keyword (str (name operation) "-" suffix)))

(defn- certify*
  [reduction-op dtype operation]
  (let [{:keys [acc init lambda]} reduction-op
        lambda (try
                 (util/inline-pure-lets lambda)
                 (catch clojure.lang.ExceptionInfo exception
                   (if (= :impure-binding (:reason (ex-data exception)))
                     (throw (ex-info "parallel reduction element contains an impure binding"
                                     {:reason (reason operation "element-impure-or-unknown")
                                      :body lambda :reduction-op reduction-op}
                                     exception))
                     (throw exception))))
        combine (descriptor/semantic-op lambda)
        args (vec (descriptor/call-args lambda))
        dtype (dtype/canon (or dtype :double))
        supported-dtypes (if (= :scan operation)
                           #{:int :long :float :double}
                           #{:byte :int :long :half :float :double})]
    (when-not (contains? supported-dtypes dtype)
      (throw (ex-info "parallel reduction dtype is not supported by the current kernel dialect"
                      {:reason (reason operation "dtype-unsupported")
                       :dtype dtype :reduction-op reduction-op})))
    (when-not (and combine (= 2 (count args))
                   (descriptor/commutative-monoid-op? combine))
      (throw (ex-info "parallel recurrence is not a certified associative reduction"
                      {:reason (reason operation "not-associative")
                       :combine combine :body lambda :reduction-op reduction-op})))
    (let [[left right] args
          element (cond
                    (and (acc-ref? left acc dtype) (not (contains-symbol? right acc))) right
                    (and (acc-ref? right acc dtype) (not (contains-symbol? left acc))) left
                    :else nil)]
      (when-not element
        (throw (ex-info "parallel reduction must combine one accumulator with an accumulator-free element"
                        {:reason (reason operation "not-elementwise")
                         :acc acc :body lambda :reduction-op reduction-op})))
      (when-not (pure-element? element)
        (throw (ex-info "parallel reduction element expression is impure or has unknown effects"
                        {:reason (reason operation "element-impure-or-unknown")
                         :element element :body lambda :reduction-op reduction-op})))
      (let [identity (descriptor/typed-reduce-identity combine dtype)
            algebra (descriptor/algebra-facet combine)]
        (when-not (constant/equivalent? init identity)
          (throw (ex-info "parallel reduction with a non-identity init requires a distinct schedule"
                          {:reason (reason operation "nonidentity-init")
                           :combine combine :init init :identity identity :dtype dtype})))
        (cond-> (->AssociativeScan acc init combine element identity dtype)
          (:nan-policy algebra) (assoc :nan-policy (:nan-policy algebra))
          (:signed-zero-policy algebra)
          (assoc :signed-zero-policy (:signed-zero-policy algebra)))))))

(defn certify-reassociation
  "Certify a scalar recurrence for parallel reassociation.

   This is the shared proof boundary for reductions, scans, reducing scatters and nested folds.
   Pure `let` regions are beta-reduced by the compiler's one capture-safe implementation before
   the registered monoid and exact typed identity are checked."
  [reduction-op dtype]
  (certify* reduction-op dtype :reduction))

(defn compatible-certificate?
  "Whether two independently derived certificates prove the same typed monoid contract.

   Element expressions may differ across a mechanical parameter-to-load projection, so the
   comparison deliberately covers only the reassociation facts; callers must independently
   certify each concrete scalar region before using this predicate."
  [declared derived]
  (and (associative-scan? declared)
       (associative-scan? derived)
       ;; Accumulators and element operands are lexical binders that may be alpha-renamed or
       ;; projected from captures between independently certified regions. The certificate is the
       ;; typed monoid contract; each concrete region has already been certified on its own.
       (= (:dtype declared) (:dtype derived))
       (= (some-> (:combine declared) name symbol)
          (some-> (:combine derived) name symbol))
       (= (:nan-policy declared) (:nan-policy derived))
       (= (:signed-zero-policy declared) (:signed-zero-policy derived))
       (constant/equivalent? (:init declared) (:init derived))
       (constant/equivalent? (:identity declared) (:identity derived))))

(defn certify
  "Certify `scan-op` as a parallel associative scan or throw a structured conversion decline.

   The accepted core is `(combine acc element)` or `(combine element acc)`, where `combine` is a
   registered commutative monoid and `element` does not mention `acc`. Commutativity is stricter
   than the mathematical minimum but matches the current shared-memory emitters' reassociation.
   The initial value must be the registered identity: injecting a non-identity once globally needs
   a distinct schedule and must not be duplicated independently in every block."
  [scan-op dtype]
  (certify* scan-op dtype :scan))

(defn- certify-projected*
  [{:keys [acc init lambda] :as operation} dtype supplied-projection kind]
  (let [reason-key (if (= :reduction kind)
                     :scalar-reduction-certificate-projection
                     :scan-certificate-projection)
        projected
        (try
          (scalar-conversion/verified-source-projection lambda supplied-projection)
          (catch clojure.lang.ExceptionInfo exception
            (throw (ex-info "parallel scalar projection is not structurally attested"
                            {:reason reason-key :operation operation
                             :projection-error (ex-data exception)}
                            exception))))
        removable-projection?
        (fn [initializer]
          (try
            (effects/removable-expr?
             (scalar-conversion/project-canonical-to-source initializer))
            (catch clojure.lang.ExceptionInfo _ false)))
        normalize
        (fn [expression]
          (try
            (util/inline-pure-lets expression :pure? removable-projection?)
            (catch clojure.lang.ExceptionInfo exception
              (throw (ex-info "parallel scalar projection contains a non-removable binding"
                              {:reason reason-key :operation operation
                               :normalization-error (ex-data exception)}
                              exception)))))
        normalized-lambda (normalize lambda)
        normalized-projected (normalize projected)
        certificate (certify* (assoc operation :lambda normalized-projected) dtype kind)
        arguments (vec (descriptor/call-args normalized-lambda))
        projected-arguments (vec (descriptor/call-args normalized-projected))
        element-position (.indexOf ^java.util.List projected-arguments (:element certificate))
        accumulator-position (when (= 2 (count projected-arguments))
                               (- 1 element-position))]
    (when-not (and (= (descriptor/semantic-op normalized-lambda)
                      (descriptor/semantic-op normalized-projected))
                   (= (count arguments) (count projected-arguments) 2)
                   (<= 0 element-position 1)
                   (acc-ref? (nth arguments accumulator-position) acc dtype)
                   (acc-ref? (nth projected-arguments accumulator-position) acc dtype))
      (throw (ex-info "parallel scalar projection changed its recurrence structure"
                      {:reason reason-key :operation operation :projected normalized-projected
                       :element-position element-position})))
    {:certificate certificate
     :element (nth arguments element-position)
     :projected-step-result normalized-projected
     :normalized-step-result normalized-lambda}))

(defn certify-projected-reassociation
  "Certify a reduction projection while retaining the corresponding typed element."
  ([operation dtype]
   (certify-projected* operation dtype nil :reduction))
  ([operation dtype supplied-projection]
   (certify-projected* operation dtype supplied-projection :reduction)))

(defn certify-projected-scan
  "Certify a scan projection while retaining the corresponding typed element."
  ([operation dtype]
   (certify-projected* operation dtype nil :scan))
  ([operation dtype supplied-projection]
   (certify-projected* operation dtype supplied-projection :scan)))
