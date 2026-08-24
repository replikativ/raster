(ns raster.gpu.measurement
  "Backend-neutral device measurement values and sampling discipline.

   This namespace never launches a kernel and never chooses a schedule. A backend-facing caller
   supplies `sample-fn`, which must return one DEVICE duration in nanoseconds. Keeping sampling,
   selection, and execution separate makes autotuning an explicit offline operation and prevents
   compilation from acquiring hidden device side effects.")

(defrecord Measurement
           [min-ns
            median-ns
            p75-ns
            mean-ns
            cv
            stationary?
            n
            warmup-iterations
            budget-ms
            cold-warm
            timing-source
            compile-ms
            hashes
            samples-ns])

(defn measurement?
  "Recognize measurements across Typed Clojure child classloaders."
  [x]
  (and x (= "raster.gpu.measurement.Measurement" (.getName (class x)))))

(defn- finite-nonnegative?
  [x]
  (and (number? x)
       (Double/isFinite (double x))
       (not (neg? (double x)))))

(defn- percentile
  "Nearest-rank percentile over an already sorted, non-empty vector."
  [sorted-samples p]
  (let [n (count sorted-samples)
        index (dec (long (Math/ceil (* (double p) n))))]
    (double (nth sorted-samples (max 0 (min (dec n) index))))))

(defn summarize
  "Summarize non-empty DEVICE-duration samples (nanoseconds) as a Measurement.

   Options are metadata required to interpret/cache a result. `timing-source` must remain
   explicit; production autotuning uses `:device-event`, while tests may use `:synthetic`.
   A result is stationary when population coefficient-of-variation is below `cv-threshold`."
  [samples-ns & {:keys [cv-threshold warmup-iterations budget-ms cold-warm timing-source
                        compile-ms hashes]
                 :or {cv-threshold 0.05
                      warmup-iterations 0
                      budget-ms 0
                      cold-warm :warm
                      timing-source :device-event
                      compile-ms 0.0
                      hashes {}}}]
  (let [samples (mapv double samples-ns)]
    (when-not (seq samples)
      (throw (ex-info "measurement requires at least one device-duration sample" {})))
    (when-let [invalid (first (remove finite-nonnegative? samples))]
      (throw (ex-info "measurement samples must be finite, non-negative nanoseconds"
                      {:sample invalid})))
    (when-not (and (number? cv-threshold) (not (neg? (double cv-threshold))))
      (throw (ex-info "measurement cv-threshold must be non-negative"
                      {:cv-threshold cv-threshold})))
    (when-not (#{:warm :cold} cold-warm)
      (throw (ex-info "measurement cold-warm must be :warm or :cold"
                      {:cold-warm cold-warm})))
    (when-not (keyword? timing-source)
      (throw (ex-info "measurement timing-source must be a keyword"
                      {:timing-source timing-source})))
    (when-not (map? hashes)
      (throw (ex-info "measurement hashes must be a map" {:hashes hashes})))
    (let [n (count samples)
          sorted (vec (sort samples))
          mean (/ (reduce + 0.0 samples) (double n))
          variance (/ (reduce (fn [acc sample]
                                (let [delta (- sample mean)]
                                  (+ acc (* delta delta))))
                              0.0 samples)
                      (double n))
          cv (if (pos? mean) (/ (Math/sqrt variance) mean) 0.0)]
      (->Measurement (first sorted)
                     (percentile sorted 0.5)
                     (percentile sorted 0.75)
                     mean
                     cv
                     (<= cv (double cv-threshold))
                     n
                     (long warmup-iterations)
                     (double budget-ms)
                     cold-warm
                     timing-source
                     (double compile-ms)
                     hashes
                     samples))))

(defn measure!
  "Measure an explicit device-sample function under a bounded, do_bench-style discipline.

   `sample-fn` returns one DEVICE duration in nanoseconds; this function never substitutes host
   wall time. Warmups and five probe samples establish steady state and estimate the iteration
   count for `budget-ms`. Probe samples are not included in the reported distribution. `flush-fn`,
   when supplied, runs before every reported sample and owns any synchronization it requires.

   Options: :warmup-iterations (3), :budget-ms (100), :min-samples (3), :max-samples (10000),
   :cv-threshold (0.05), :flush-fn, :cold-warm, :compile-ms, :hashes, :timing-source."
  [sample-fn & {:keys [warmup-iterations budget-ms min-samples max-samples cv-threshold flush-fn
                       cold-warm compile-ms hashes timing-source]
                :or {warmup-iterations 3
                     budget-ms 100
                     min-samples 3
                     max-samples 10000
                     cv-threshold 0.05
                     cold-warm :warm
                     compile-ms 0.0
                     hashes {}
                     timing-source :device-event}}]
  (doseq [[field value] [[:warmup-iterations warmup-iterations]
                         [:min-samples min-samples]
                         [:max-samples max-samples]]]
    (when-not (and (integer? value) (not (neg? (long value))))
      (throw (ex-info "measurement iteration bounds must be non-negative integers"
                      {:field field :value value}))))
  (when-not (and (number? budget-ms) (pos? (double budget-ms)))
    (throw (ex-info "measurement budget-ms must be positive" {:budget-ms budget-ms})))
  (when (or (zero? (long min-samples)) (< (long max-samples) (long min-samples)))
    (throw (ex-info "measurement requires 0 < min-samples <= max-samples"
                    {:min-samples min-samples :max-samples max-samples})))
  (dotimes [_ (long warmup-iterations)]
    (sample-fn))
  (let [probe (mapv (fn [_] (double (sample-fn))) (range 5))]
    (when-let [invalid (first (remove finite-nonnegative? probe))]
      (throw (ex-info "device sample must be finite, non-negative nanoseconds"
                      {:sample invalid :phase :probe})))
    (let [estimate (max 1.0 (/ (reduce + 0.0 probe) (double (count probe))))
          wanted (long (/ (* (double budget-ms) 1.0e6) estimate))
          n (max (long min-samples) (min (long max-samples) wanted))
          samples (mapv (fn [_]
                          (when flush-fn (flush-fn))
                          (double (sample-fn)))
                        (range n))]
      (summarize samples
                 :cv-threshold cv-threshold
                 :warmup-iterations warmup-iterations
                 :budget-ms budget-ms
                 :cold-warm cold-warm
                 :timing-source timing-source
                 :compile-ms compile-ms
                 :hashes hashes))))

