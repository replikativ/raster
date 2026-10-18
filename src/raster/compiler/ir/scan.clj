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
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.form :as form]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.core.util :as util]))

(defrecord AssociativeScan [acc init combine element identity dtype])

(defn associative-scan? [x]
  (and x (= "raster.compiler.ir.scan.AssociativeScan" (.getName (class x)))))

(defn- lexical-call?
  [expression locals]
  (let [{:keys [semantic-op implementation-op lexical-callee]} (descriptor/call-description expression)]
    (or lexical-callee (contains? locals semantic-op) (contains? locals implementation-op)
        (contains? locals (when (seq? expression) (first expression))))))

(defn- acc-ref?
  [expr acc reduction-dtype]
  (or (= expr acc)
      (and (seq? expr) (not (lexical-call? expr util/*shadowing-locals*))
           (= 1 (count (descriptor/call-args expr)))
           (= (dtype/canon reduction-dtype)
              (some-> (descriptor/cast-result-tag (descriptor/semantic-op expr)) keyword dtype/canon))
           (= acc (first (descriptor/call-args expr))))))

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

(defn- recurrence-form
  "Expose the combine without beta-reducing its accumulator-independent evaluation region.
   Prefix bindings/checks stay once per element, in source order and with their lexical scope.
   The accumulator reference belongs to the certified numeric carry contract, not an arbitrary
   capture. Local callees and carry-dependent prefixes cannot acquire monoid semantics."
  [expression acc dtype operation]
  (loop [expression expression contexts [] locals util/*shadowing-locals*]
    (let [binding? (and (seq? expression) (form/let-head? (first expression)))
          do? (and (seq? expression) (contains? '#{do clojure.core/do} (first expression)))
          bindings (when binding? (second expression))
          body (cond binding? (nnext expression) do? (next expression))]
      (if (or binding? do?)
        (do
          (when-not (and (seq body)
                         (or do? (and (vector? bindings) (even? (count bindings))
                                      (every? symbol? (take-nth 2 bindings))))
                         (not (some #{acc} (take-nth 2 bindings)))
                         (not-any? #(contains-symbol? % acc)
                                   (concat (take-nth 2 (rest bindings)) (butlast body))))
            (throw (ex-info "parallel recurrence prefix must be closed and accumulator-independent"
                            {:reason (reason operation "not-elementwise") :body expression :acc acc})))
          (recur (last body) (conj contexts expression)
                 (into locals (take-nth 2 bindings))))
        (let [operator (descriptor/semantic-op expression)
              args (vec (descriptor/call-args expression))
              position (binding [util/*shadowing-locals* locals]
                         (when (= 2 (count args))
                         (cond (and (acc-ref? (first args) acc dtype)
                                    (not (contains-symbol? (second args) acc))) 1
                               (and (acc-ref? (second args) acc dtype)
                                    (not (contains-symbol? (first args) acc))) 0)))]
          (when (lexical-call? expression locals)
            (throw (ex-info "a lexical callee is not a declared monoid operation"
                            {:reason (reason operation "not-associative") :body expression})))
          (if (and (seq contexts) (some? position))
            (let [element (reduce (fn [element context]
                                    (let [bindings (when (form/let-head? (first context)) (second context))
                                          binder (when (symbol? element)
                                                   (first (last (filter #(= element (first %))
                                                                        (partition 2 bindings)))))
                                          retained (if (symbol? element)
                                                     (get (util/binding-env bindings) element element)
                                                     element)
                                          type-keys (conj types/preserved-type-keys :tag)
                                          ;; The old region returned the combine, not this element.
                                          ;; Transfer only actual retained result/binder type facts;
                                          ;; preserve initializer conversions and executable scope.
                                          result-meta (merge (select-keys (meta retained) type-keys)
                                                             (select-keys (meta element) type-keys)
                                                             (select-keys (meta binder) type-keys))]
                                      (with-meta
                                        (util/remake-from context (concat (butlast context) [element]))
                                        (merge (apply dissoc (meta context) type-keys) result-meta))))
                                  (nth args position) (rseq contexts))
                  prefix (if (:dispatch? (descriptor/call-description expression))
                           (take 2 expression) (take 1 expression))]
              (util/remake-from expression (concat prefix (assoc args position element))))
            (if (seq contexts)
              (throw (ex-info "parallel recurrence prefix has no independent element"
                              {:reason (reason operation "not-elementwise") :body expression :acc acc}))
              expression)))))))

(defn- certify*
  [reduction-op dtype operation]
  (let [{:keys [acc init lambda]} reduction-op
        dtype (dtype/canon (or dtype :double))
        lambda (recurrence-form lambda acc dtype operation)
        combine (descriptor/semantic-op lambda)
        args (vec (descriptor/call-args lambda))
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
          (:overflow algebra) (assoc :overflow (:overflow algebra))
          (:nan-policy algebra) (assoc :nan-policy (:nan-policy algebra))
          (:signed-zero-policy algebra)
          (assoc :signed-zero-policy (:signed-zero-policy algebra)))))))

(defn certify-reassociation
  "Certify a scalar recurrence for parallel reassociation.

   This is the shared proof boundary for reductions, scans, reducing scatters and nested folds.
   Accumulator-independent local regions remain once-only element computations while the
   registered monoid and exact typed identity are checked. This grants no read/check removal."
  [reduction-op dtype]
  (certify* reduction-op dtype :reduction))

(defn validate!
  "Independently rederive a retained associative certificate; record identity is not proof.
   The existing scalar algebra authority checks purity, typed identity and numerical facets."
  [certificate]
  (when-not (associative-scan? certificate)
    (throw (ex-info "expected an associative scalar certificate"
                    {:reason :associative-scan-certificate})))
  (let [{:keys [acc init combine element dtype]} certificate
        derived (certify-reassociation {:acc acc :init init
                                       :lambda (list combine acc element)} dtype)]
    (when-not (= derived certificate)
      (throw (ex-info "associative scalar certificate differs from its rederived algebra"
                      {:reason :associative-scan-certificate
                       :expected derived :actual certificate})))
    certificate))

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
       (= (:overflow declared) (:overflow derived))
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
  (let [dtype (dtype/canon (or dtype :double))
        reason-key (if (= :reduction kind)
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
        normalize
        (fn [expression]
          (try
            (recurrence-form expression acc dtype kind)
            (catch clojure.lang.ExceptionInfo exception
              (throw (ex-info "parallel scalar projection has no closed element region"
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
