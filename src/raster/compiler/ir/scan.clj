(ns raster.compiler.ir.scan
  "Certified scalar algebra for parallel scans and reductions.

   Raster's surface `par/scan` is also a general left-to-right recurrence primitive. Such a
   recurrence is not automatically a parallel scan: Blelloch/Hillis-Steele scheduling is sound
   only when the body is an associative combine of the prior accumulator and an accumulator-free
   element expression. This boundary proves that property before SegScan scheduling."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.core.util :as util]))

(defrecord AssociativeScan [acc init combine element identity dtype])

(defn associative-scan? [x]
  (and x (= "raster.compiler.ir.scan.AssociativeScan" (.getName (class x)))))

(defn- acc-ref?
  [expr acc]
  (or (= expr acc)
      (and (seq? expr)
           (= 2 (count expr))
           (contains? #{'float 'double 'int 'long
                        'clojure.core/float 'clojure.core/double
                        'clojure.core/int 'clojure.core/long}
                      (first expr))
           (= acc (second expr)))))

(defn- contains-symbol?
  [expr target]
  (cond
    (= expr target) true
    (coll? expr) (boolean (some #(contains-symbol? % target) expr))
    :else false))

(defn- literal-value
  [expr]
  (if (and (seq? expr)
           (= 2 (count expr))
           (contains? #{'float 'double 'int 'long
                        'clojure.core/float 'clojure.core/double
                        'clojure.core/int 'clojure.core/long}
                      (first expr))
           (number? (second expr)))
    (second expr)
    expr))

(defn- identity-equivalent?
  [left right]
  (let [left (literal-value left)
        right (literal-value right)]
    (if (and (number? left) (number? right))
      (== (double left) (double right))
      (= left right))))

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
        dtype (or dtype :double)
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
                    (and (acc-ref? left acc) (not (contains-symbol? right acc))) right
                    (and (acc-ref? right acc) (not (contains-symbol? left acc))) left
                    :else nil)]
      (when-not element
        (throw (ex-info "parallel reduction must combine one accumulator with an accumulator-free element"
                        {:reason (reason operation "not-elementwise")
                         :acc acc :body lambda :reduction-op reduction-op})))
      (when-not (pure-element? element)
        (throw (ex-info "parallel reduction element expression is impure or has unknown effects"
                        {:reason (reason operation "element-impure-or-unknown")
                         :element element :body lambda :reduction-op reduction-op})))
      (let [identity (descriptor/typed-reduce-identity combine dtype)]
        (when-not (identity-equivalent? init identity)
          (throw (ex-info "parallel reduction with a non-identity init requires a distinct schedule"
                          {:reason (reason operation "nonidentity-init")
                           :combine combine :init init :identity identity :dtype dtype})))
        (->AssociativeScan acc init combine element identity dtype)))))

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
       (= (some-> (:combine declared) name symbol)
          (some-> (:combine derived) name symbol))
       (identity-equivalent? (:init declared) (:init derived))
       (identity-equivalent? (:identity declared) (:identity derived))
       (= (:dtype declared) (:dtype derived))))

(defn certify
  "Certify `scan-op` as a parallel associative scan or throw a structured conversion decline.

   The accepted core is `(combine acc element)` or `(combine element acc)`, where `combine` is a
   registered commutative monoid and `element` does not mention `acc`. Commutativity is stricter
   than the mathematical minimum but matches the current shared-memory emitters' reassociation.
   The initial value must be the registered identity: injecting a non-identity once globally needs
   a distinct schedule and must not be duplicated independently in every block."
  [scan-op dtype]
  (certify* scan-op dtype :scan))
