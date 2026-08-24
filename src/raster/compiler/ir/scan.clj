(ns raster.compiler.ir.scan
  "Certified algebra for parallel prefix scans.

   Raster's surface `par/scan` is also a general left-to-right recurrence primitive. Such a
   recurrence is not automatically a parallel scan: Blelloch/Hillis-Steele scheduling is sound
   only when the body is an associative combine of the prior accumulator and an accumulator-free
   element expression. This boundary proves that property before SegScan scheduling."
  (:require [raster.ad.purity :as purity]
            [raster.compiler.core.op-descriptor :as descriptor]))

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
  (cond
    (seq? expr) (let [op (descriptor/semantic-op expr)]
                  (and op
                       (= :pure (purity/pure-op? op))
                       (every? pure-element? (descriptor/call-args expr))))
    (coll? expr) (every? pure-element? expr)
    :else true))

(defn certify
  "Certify `scan-op` as a parallel associative scan or throw a structured conversion decline.

   The accepted core is `(combine acc element)` or `(combine element acc)`, where `combine` is a
   registered commutative monoid and `element` does not mention `acc`. Commutativity is stricter
   than the mathematical minimum but matches the current shared-memory emitters' reassociation.
   The initial value must be the registered identity: injecting a non-identity once globally needs
   a distinct schedule and must not be duplicated independently in every block."
  [scan-op dtype]
  (let [{:keys [acc init lambda]} scan-op
        combine (descriptor/semantic-op lambda)
        args (vec (descriptor/call-args lambda))
        dtype (or dtype :double)]
    (when-not (contains? #{:int :long :float :double} dtype)
      (throw (ex-info "parallel scan dtype is not supported by the current kernel dialect"
                      {:reason :scan-dtype-unsupported :dtype dtype :scan-op scan-op})))
    (when-not (and combine (= 2 (count args))
                   (descriptor/commutative-monoid-op? combine))
      (throw (ex-info "par/scan is a general recurrence, not a certified associative scan"
                      {:reason :scan-not-associative
                       :combine combine :body lambda :scan-op scan-op})))
    (let [[left right] args
          element (cond
                    (and (acc-ref? left acc) (not (contains-symbol? right acc))) right
                    (and (acc-ref? right acc) (not (contains-symbol? left acc))) left
                    :else nil)]
      (when-not element
        (throw (ex-info "parallel scan body must combine one accumulator with an accumulator-free element"
                        {:reason :scan-not-elementwise :acc acc :body lambda :scan-op scan-op})))
      (when-not (pure-element? element)
        (throw (ex-info "parallel scan element expression is impure or has unknown effects"
                        {:reason :scan-element-impure-or-unknown
                         :element element :body lambda :scan-op scan-op})))
      (let [identity (descriptor/typed-reduce-identity combine dtype)]
        (when-not (identity-equivalent? init identity)
          (throw (ex-info "parallel scan with a non-identity init requires a distinct global-prefix schedule"
                          {:reason :scan-nonidentity-init
                           :combine combine :init init :identity identity :dtype dtype})))
        (->AssociativeScan acc init combine element identity dtype)))))
