(ns raster.compiler.ir.kernel-launch
  "Backend-neutral launch contracts for emitted and scheduled kernels.

   A LaunchSpec belongs to compiler IR: its dimensions may refer to compiler values.  A
   LaunchGeometry is the concrete 1-3D value a backend submits.  Keeping the two distinct stops
   emitters, extractors and runtimes from inventing incompatible scalar/2-D marker conventions.")

(defrecord RuntimeValue [value])
(defrecord CeilDiv [value divisor])
(defrecord LaunchSpec [workgroup-size group-count shared-memory-bytes])
(defrecord LaunchGeometry [workgroup-size group-count shared-memory-bytes])

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
  (when-not (and (integer? divisor) (pos? divisor))
    (throw (ex-info "ceil-div launch divisor must be a positive integer"
                    {:value value :divisor divisor})))
  (->CeilDiv value divisor))

(defn launch-spec? [x] (instance? LaunchSpec x))
(defn launch-geometry? [x] (instance? LaunchGeometry x))

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

(defn- launch-expression?
  [x]
  (or (and (integer? x) (pos? x))
      (instance? RuntimeValue x)
      (instance? CeilDiv x)))

(defn validate-spec!
  "Validate and return a symbolic launch specification."
  [spec]
  (when-not (launch-spec? spec)
    (throw (ex-info "launch contract must be a LaunchSpec" {:launch spec :actual (type spec)})))
  (let [{:keys [workgroup-size group-count shared-memory-bytes]} spec]
    (dimension-vector! "launch specification" :workgroup-size workgroup-size
                       launch-expression? "a positive integer or explicit runtime expression")
    (dimension-vector! "launch specification" :group-count group-count
                       launch-expression? "a positive integer or explicit runtime expression")
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

(defn- resolved-dimension
  [resolve-value dimension]
  (long
   (cond
     (integer? dimension) dimension
     (instance? RuntimeValue dimension)
     (resolve-integer! resolve-value dimension (:value dimension))
     (instance? CeilDiv dimension)
     (let [value (resolve-integer! resolve-value dimension (:value dimension))
           divisor (:divisor dimension)]
       ;; This form avoids overflowing `value + divisor - 1` for a large positive value.
       (+ (quot value divisor) (if (zero? (rem value divisor)) 0 1)))
     :else (throw (ex-info "unsupported launch dimension expression"
                           {:dimension dimension :actual (type dimension)})))))

(defn realize
  "Resolve a LaunchSpec into concrete geometry. `resolve-value` maps a compiler value to a number."
  [launch-spec resolve-value]
  (let [{:keys [workgroup-size group-count shared-memory-bytes]}
        (validate-spec! launch-spec)]
    (geometry {:workgroup-size (mapv #(resolved-dimension resolve-value %) workgroup-size)
               :group-count (mapv #(resolved-dimension resolve-value %) group-count)
               :shared-memory-bytes shared-memory-bytes})))
