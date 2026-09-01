(ns raster.compiler.ir.kernel-launch
  "Backend-neutral launch contracts for emitted and scheduled kernels.

   A LaunchSpec belongs to compiler IR: its dimensions may refer to compiler values.  A
   LaunchGeometry is the concrete 1-3D value a backend submits.  Keeping the two distinct stops
   emitters, extractors and runtimes from inventing incompatible scalar/2-D marker conventions.")

(defrecord RuntimeValue [value])
(defrecord CeilDiv [value divisor])
(defrecord FloorDiv [value divisor])
(defrecord Sum [terms])
(defrecord Product [factors])
(defrecord AlignUp [value alignment])
(defrecord Minimum [values])
(defrecord LaunchSpec [workgroup-size group-count shared-memory-bytes])
(defrecord LaunchGeometry [workgroup-size group-count shared-memory-bytes])

(defn- record-kind?
  "Recognize compiler IR records across Typed Clojure's child DynamicClassLoaders."
  [record-class value]
  (and value (= record-class (.getName (class value)))))

(defn runtime-value
  "A launch dimension whose value is resolved when a call is bound."
  [value]
  (when (nil? value)
    (throw (ex-info "runtime launch value cannot be nil" {:value value})))
  (->RuntimeValue value))

(defn ceil-div
  "A runtime ceil-div dimension, normally `(ceil-div bound workgroup-size)`."
  [value divisor]
  (when (nil? value)
    (throw (ex-info "ceil-div launch value cannot be nil" {:value value :divisor divisor})))
  (when-not (or (not (integer? divisor)) (pos? divisor))
    (throw (ex-info "ceil-div launch divisor must be positive"
                    {:value value :divisor divisor})))
  (->CeilDiv value divisor))

(defn floor-div
  "A checked floor division of non-negative integer expressions. The divisor may itself be a
   symbolic integer expression and is proved positive when the expression is realized."
  [value divisor]
  (when (or (nil? value) (nil? divisor))
    (throw (ex-info "floor-div operands cannot be nil" {:value value :divisor divisor})))
  (when-not (or (not (integer? divisor)) (pos? divisor))
    (throw (ex-info "floor-div divisor must be positive" {:value value :divisor divisor})))
  (->FloorDiv value divisor))

(defn sum
  "A checked sum of integer launch/storage expressions. Sums are explicit IR rather than
   arbitrary Clojure forms so symbolic view extents remain inspectable and safely realizable."
  [& terms]
  (when-not (and (seq terms) (every? some? terms))
    (throw (ex-info "sum requires one or more non-nil terms" {:terms terms})))
  (->Sum (vec terms)))

(defn product
  "A checked product of integer launch/storage expressions. Products are explicit IR rather than
   arbitrary Clojure forms so graph scratch sizes can remain inspectable and safely realizable."
  [& factors]
  (when-not (and (seq factors) (every? some? factors))
    (throw (ex-info "product requires one or more non-nil factors" {:factors factors})))
  (->Product (vec factors)))

(defn align-up
  "Round an integer launch/storage expression up to a positive static alignment."
  [value alignment]
  (when (nil? value)
    (throw (ex-info "align-up value cannot be nil" {:value value :alignment alignment})))
  (when-not (and (integer? alignment) (pos? alignment))
    (throw (ex-info "align-up requires a positive integer alignment"
                    {:value value :alignment alignment})))
  (->AlignUp value alignment))

(defn minimum
  "The minimum of one or more integer expressions."
  [& values]
  (when-not (and (seq values) (every? some? values))
    (throw (ex-info "minimum requires one or more non-nil values" {:values values})))
  (->Minimum (vec values)))

(defn launch-spec? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.LaunchSpec" x))

(defn launch-geometry? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.LaunchGeometry" x))

(defn- runtime-value? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.RuntimeValue" x))

(defn- ceil-div? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.CeilDiv" x))

(defn- floor-div? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.FloorDiv" x))

(defn- sum? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.Sum" x))

(defn- product? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.Product" x))

(defn- align-up? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.AlignUp" x))

(defn- minimum? [x]
  (record-kind? "raster.compiler.ir.kernel_launch.Minimum" x))

(defn expression?
  "True for explicit symbolic integer-expression nodes and literal integers. Opaque compiler
   values such as symbols are leaves and are therefore also accepted. Sequential Clojure forms
   are deliberately rejected: callers must construct arithmetic with this namespace."
  [x]
  (cond
    (integer? x) true
    (runtime-value? x) (some? (:value x))
    (ceil-div? x) (and (expression? (:value x)) (expression? (:divisor x)))
    (floor-div? x) (and (expression? (:value x)) (expression? (:divisor x)))
    (sum? x) (and (seq (:terms x)) (every? expression? (:terms x)))
    (product? x) (and (seq (:factors x)) (every? expression? (:factors x)))
    (align-up? x) (and (expression? (:value x))
                       (integer? (:alignment x)) (pos? (:alignment x)))
    (minimum? x) (and (seq (:values x)) (every? expression? (:values x)))
    (sequential? x) false
    :else (some? x)))

(defn expression-references
  "Return the opaque compiler-value leaves read by an integer expression."
  [expression]
  (cond
    (integer? expression) #{}
    (runtime-value? expression) #{(:value expression)}
    (ceil-div? expression) (into (expression-references (:value expression))
                                 (expression-references (:divisor expression)))
    (floor-div? expression) (into (expression-references (:value expression))
                                  (expression-references (:divisor expression)))
    (sum? expression) (reduce into #{} (map expression-references (:terms expression)))
    (product? expression) (reduce into #{} (map expression-references (:factors expression)))
    (align-up? expression) (expression-references (:value expression))
    (minimum? expression) (reduce into #{} (map expression-references (:values expression)))
    :else #{expression}))

(defn- dimension-vector!
  [owner field dimensions pred expected]
  (when-not (vector? dimensions)
    (throw (ex-info (str owner " " (name field) " must be a vector")
                    {:field field :value dimensions})))
  (when-not (<= 1 (count dimensions) 3)
    (throw (ex-info (str owner " must have one to three dimensions")
                    {:field field :value dimensions :dimensions (count dimensions)})))
  (doseq [[axis value] (map-indexed vector dimensions)]
    (when-not (pred value)
      (throw (ex-info (str owner " " (name field) " dimension must be " expected)
                      {:field field :axis axis :value value :dimensions dimensions}))))
  dimensions)

(defn dimension-expression?
  "Whether `x` is legal as one symbolic launch dimension.

   Unlike `expression?`, this excludes arbitrary opaque leaves: callers must wrap runtime compiler
   values with `runtime-value` so scheduled extents cannot smuggle source S-expressions into an
   emitted ABI."
  [x]
  (or (and (integer? x) (pos? x))
      (runtime-value? x)
      (ceil-div? x)
      (product? x)
      (align-up? x)
      (floor-div? x)
      (sum? x)
      (minimum? x)))

(defn validate-spec!
  "Validate and return a symbolic launch specification."
  [spec]
  (when-not (launch-spec? spec)
    (throw (ex-info "launch contract must be a LaunchSpec" {:launch spec :actual (type spec)})))
  (let [{:keys [workgroup-size group-count shared-memory-bytes]} spec]
    (dimension-vector! "launch specification" :workgroup-size workgroup-size
                       dimension-expression?
                       "a positive integer or explicit runtime expression")
    (dimension-vector! "launch specification" :group-count group-count
                       dimension-expression?
                       "a positive integer or explicit runtime expression")
    (when-not (= (count workgroup-size) (count group-count))
      (throw (ex-info "launch workgroup and group-count dimensionality must match"
                      {:workgroup-size workgroup-size :group-count group-count})))
    (when-not (and (integer? shared-memory-bytes) (not (neg? shared-memory-bytes)))
      (throw (ex-info "launch shared-memory-bytes must be a non-negative integer"
                      {:shared-memory-bytes shared-memory-bytes}))))
  spec)

(defn spec
  "Construct a checked symbolic 1-3D launch contract."
  [{:keys [workgroup-size group-count shared-memory-bytes]
    :or {shared-memory-bytes 0}}]
  (validate-spec! (->LaunchSpec workgroup-size group-count shared-memory-bytes)))

(defn validate-geometry!
  "Validate and return concrete backend launch geometry."
  [geometry]
  (when-not (launch-geometry? geometry)
    (throw (ex-info "concrete launch must be a LaunchGeometry"
                    {:launch geometry :actual (type geometry)})))
  (let [{:keys [workgroup-size group-count shared-memory-bytes]} geometry
        positive-integer? #(and (integer? %) (pos? %))]
    (dimension-vector! "launch geometry" :workgroup-size workgroup-size
                       positive-integer? "a positive integer")
    (dimension-vector! "launch geometry" :group-count group-count
                       positive-integer? "a positive integer")
    (when-not (= (count workgroup-size) (count group-count))
      (throw (ex-info "launch workgroup and group-count dimensionality must match"
                      {:workgroup-size workgroup-size :group-count group-count})))
    (when-not (and (integer? shared-memory-bytes) (not (neg? shared-memory-bytes)))
      (throw (ex-info "launch shared-memory-bytes must be a non-negative integer"
                      {:shared-memory-bytes shared-memory-bytes}))))
  geometry)

(defn geometry
  "Construct checked concrete 1-3D launch geometry."
  [{:keys [workgroup-size group-count shared-memory-bytes]
    :or {shared-memory-bytes 0}}]
  (let [checked (validate-geometry!
                 (->LaunchGeometry workgroup-size group-count shared-memory-bytes))]
    (->LaunchGeometry (mapv long (:workgroup-size checked))
                      (mapv long (:group-count checked))
                      (long (:shared-memory-bytes checked)))))

(defn dimensions
  "Dimensionality of a checked launch spec or geometry."
  [launch]
  (cond
    (launch-spec? launch) (count (:workgroup-size (validate-spec! launch)))
    (launch-geometry? launch) (count (:workgroup-size (validate-geometry! launch)))
    :else (throw (ex-info "not a launch spec or geometry" {:launch launch :actual (type launch)}))))

(defn static-workgroup-size
  "Return the checked workgroup vector of a LaunchSpec when every dimension is compile-time
   integer. Runtime binders that cannot yet resolve dynamic workgroup expressions use this instead
   of guessing a backend default."
  [launch-spec]
  (let [workgroup-size (:workgroup-size (validate-spec! launch-spec))]
    (when-not (every? #(and (integer? %) (pos? %)) workgroup-size)
      (throw (ex-info "launch workgroup size is not statically known"
                      {:workgroup-size workgroup-size :launch launch-spec})))
    (mapv long workgroup-size)))

(defn- resolve-integer!
  [resolve-value expression value]
  (let [resolved (resolve-value value)]
    (when-not (integer? resolved)
      (throw (ex-info "resolved launch dimension must be an integer"
                      {:expression expression :value value :resolved resolved})))
    resolved))

(defn resolve-expression
  "Resolve one integer launch/storage expression.

   Besides literal integers this understands the explicit symbolic records declared above and used
   by launch and graph-buffer contracts. `resolve-value` supplies an integer for an opaque compiler
   value such as a bound symbol. Keeping this evaluator here prevents graph runners from
   interpreting arbitrary compiler S-expressions."
  [resolve-value dimension]
  (letfn [(resolve* [expression]
            (long
             (cond
               (integer? expression) expression
               (runtime-value? expression)
               (resolve-integer! resolve-value expression (:value expression))
               (ceil-div? expression)
               (let [value (resolve* (:value expression))
                     divisor (resolve* (:divisor expression))]
                 (when-not (pos? divisor)
                   (throw (ex-info "ceil-div resolved divisor must be positive"
                                   {:expression expression :divisor divisor})))
                 ;; This form avoids overflowing `value + divisor - 1` for a large positive value.
                 (+ (quot value divisor) (if (zero? (rem value divisor)) 0 1)))
               (floor-div? expression)
               (let [value (resolve* (:value expression))
                     divisor (resolve* (:divisor expression))]
                 (when-not (pos? divisor)
                   (throw (ex-info "floor-div resolved divisor must be positive"
                                   {:expression expression :divisor divisor})))
                 (quot value divisor))
               (sum? expression)
               (reduce (fn [acc term]
                         (Math/addExact (long acc) (long term)))
                       0 (map resolve* (:terms expression)))
               (product? expression)
               (reduce (fn [acc factor]
                         (Math/multiplyExact (long acc) (long factor)))
                       1 (map resolve* (:factors expression)))
               (align-up? expression)
               (let [value (resolve* (:value expression))
                     alignment (:alignment expression)
                     quotient (quot value alignment)]
                 (Math/multiplyExact
                  (long (+ quotient (if (zero? (rem value alignment)) 0 1)))
                  (long alignment)))
               (minimum? expression)
               (reduce min (map resolve* (:values expression)))
               :else (resolve-integer! resolve-value expression expression))))]
    (resolve* dimension)))

(defn realize
  "Resolve a LaunchSpec into concrete geometry. `resolve-value` maps a compiler value to a number."
  [launch-spec resolve-value]
  (let [{:keys [workgroup-size group-count shared-memory-bytes]}
        (validate-spec! launch-spec)]
    (geometry {:workgroup-size (mapv #(resolve-expression resolve-value %) workgroup-size)
               :group-count (mapv #(resolve-expression resolve-value %) group-count)
               :shared-memory-bytes shared-memory-bytes})))
