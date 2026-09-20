(ns raster.perf.production-canary
  "Opt-in canaries of the public Raster compiler and resident replay paths.
   No timing assertion runs in ordinary CI and no run creates/updates its own baseline."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [raster.arrays :as arrays]
            [raster.compiler.core.hardware :as compiler-hardware]
            [raster.core :refer [deftm]]
            [raster.dl.nn :as dl-nn]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.link :as link]
            [raster.runtime.hardware :as hardware]
            [raster.runtime.microbench :as microbench]))

(deftm sumsq [x :- (Array double) n :- Long] :- Double
  (raster.par/reduce acc 0.0 i n
    (+ acc (* (arrays/aget x i) (arrays/aget x i)))))

(deftm gemm64! [A :- (Array float) B :- (Array float) C :- (Array float)] :- (Array float)
  (raster.par/contract C [[i 64] [j 64]] [[k 64]]
    (* (arrays/aget A (+ (* i 64) k)) (arrays/aget B (+ (* k 64) j)))
    :init (float 0.0)))

(deftm gemm-mnk! [A :- (Array float) B :- (Array float) C :- (Array float)
                   m :- Long n :- Long k :- Long] :- (Array float)
  (raster.par/contract C [[i m] [j n]] [[p k]]
    (* (arrays/aget A (+ (* i k) p)) (arrays/aget B (+ (* p n) j)))
    :init (float 0.0)))

(def gemm-shapes
  "Small opt-in shape ladder, including decode, multi-row projection and awkward tails.
   These cases are not a claim of representative frontier-scale throughput."
  [[64 64 64] [1 256 256] [8 256 256] [127 65 33] [256 256 256]])

(deftm gemm-relu-composed! [A :- (Array float) B :- (Array float) C :- (Array float)
                    m :- Long n :- Long k :- Long] :- (Array float)
  (let [product (raster.par/contract C [[i m] [j n]] [[p k]]
                 (* (arrays/aget A (+ (* i k) p)) (arrays/aget B (+ (* p n) j)))
                 :init (float 0.0))]
    (raster.par/map! C q (* m n) nil (max (float 0.0) (arrays/aget product q)))))

(deftm gemm-relu-prebound! [A :- (Array float) B :- (Array float) C :- (Array float)
                           m :- Long n :- Long k :- Long] :- (Array float)
  (let [size (* m n)
        product (raster.par/contract C [[i m] [j n]] [[p k]]
                  (* (arrays/aget A (+ (* i k) p)) (arrays/aget B (+ (* p n) j)))
                  :init (float 0.0))]
    (raster.par/map! C q size nil (max (float 0.0) (arrays/aget product q)))))

(deftm gemm-relu! [A :- (Array float) B :- (Array float) C :- (Array float)
                    m :- Long n :- Long k :- Long] :- (Array float)
  (raster.par/contract C [[i m] [j n]] [[p k]]
    (* (arrays/aget A (+ (* i k) p)) (arrays/aget B (+ (* p n) j)))
    :init (float 0.0)
    :epilogue {:acc acc :expr (max (float 0.0) acc)
               :operands [] :scalars [] :dtype :float}))

(defn- valid-measurement? [result]
  (let [median (get-in result [:measurement :median-ns])]
    (and (true? (:validated? result)) (number? median)
         (Double/isFinite (double median)) (pos? median))))

(defn verdict
  "Compare like-for-like stationary measurements. Source/schedule changes stay comparable;
   they belong in evidence, not in an identity that could silently bypass a regression."
  [baseline result]
  (cond
    (not (valid-measurement? result)) :invalid-measurement
    (not (true? (get-in result [:measurement :stationary?]))) :nonstationary
    (= :regression (get-in result [:performance-contract :status])) :roofline-regression
    (nil? baseline) :unbaselined
    (not= (:identity baseline) (:identity result)) :incomparable
    (not (and (valid-measurement? baseline)
              (true? (get-in baseline [:measurement :stationary?])))) :invalid-baseline
    (<= (get-in result [:measurement :median-ns])
        (* 1.15 (get-in baseline [:measurement :median-ns]))) :pass
    :else :regression))

(defn roofline-assessment
  "Compare a device-event measurement with a launch-aware analytic lower bound.

   This is deliberately a broad cliff detector, not an autotuner or a claim that the roofline is
   attainable. A descriptor without measured/catalogued bandwidth and peak FLOPS abstains."
  [descriptor work measurement max-slowdown]
  (when-not (and (number? max-slowdown) (Double/isFinite (double max-slowdown))
                 (pos? (double max-slowdown)))
    (throw (ex-info "roofline slowdown limit must be finite and positive"
                    {:max-slowdown max-slowdown})))
  (let [expected (compiler-hardware/roofline-time-ns descriptor work)
        observed (:median-ns measurement)]
    (cond
      (nil? expected)
      {:status :unmodelled :reason :missing-hardware-throughput
       :max-slowdown (double max-slowdown) :work work}

      (not (and (number? observed) (Double/isFinite (double observed)) (pos? (double observed))))
      {:status :invalid-measurement :expected-ns expected :observed-ns observed
       :max-slowdown (double max-slowdown) :work work}

      :else
      (let [slowdown (/ (double observed) (double expected))]
        {:status (if (<= slowdown (double max-slowdown)) :pass :regression)
         :expected-ns expected :observed-ns (double observed) :slowdown slowdown
         :max-slowdown (double max-slowdown) :work work}))))

(defn- identity-for [workload target dtype shape timing-source environment-tag]
  (when-not (and (string? environment-tag) (not-empty environment-tag))
    (throw (ex-info "canary requires an explicit stable-machine/driver :environment-tag" {})))
  (hardware/init!)
  {:version 1 :workload workload :target target :dtype dtype :shape shape
   :timing-source timing-source :cache-state :warm-resident
   :environment-tag environment-tag
   :device (hardware/device-signature target)
   :host (hardware/device-signature :cpu:0)
   :java-version (System/getProperty "java.vm.version")})

(defn- measure [thunk]
  (microbench/do-bench thunk :warmup-ms 500 :budget-ms 800 :cv-threshold 0.08))

(defn cpu! [{:keys [environment-tag compiler-revision]}]
  (let [identity (identity-for :sumsq-aot :cpu:0 :double [1048576] :host-call environment-tag)
        x (double-array (map #(/ (double (mod % 16)) 16.0) (range 1048576)))
        expected (reduce (fn [s v] (+ s (* v v))) 0.0 x)
        start (System/nanoTime)
        f (pipeline/compile-aot #'sumsq :dtype :double)
        compile-ns (- (System/nanoTime) start)
        sink (volatile! nil)
        thunk #(vreset! sink (f x (long (alength x))))]
    (when-not (= expected (double (thunk)))
      (throw (ex-info "production CPU canary failed independent numerical reference" {})))
    {:identity (assoc identity :numerical-policy :double)
     :compiler-revision compiler-revision :compile-ns compile-ns :validated? true
     :measurement (measure thunk)}))

(defn- checked-shape [shape]
  (when-not (and (vector? shape) (= 3 (count shape))
                 (every? #(and (integer? %) (<= 1 % Integer/MAX_VALUE)) shape)
                 (every? #(<= % Integer/MAX_VALUE)
                         (let [[m n k] shape] [(*' m n) (*' m k) (*' k n)])))
    (throw (ex-info "GEMM canary requires positive [m n k] with int-sized buffers" {:shape shape})))
  shape)

(defn gemm-arguments
  ([] (gemm-arguments [64 64 64]))
  ([shape]
   (let [[m n k] (checked-shape shape)]
     [(float-array (map #(float (/ (- (mod % 13) 6) 8.0)) (range (* m k))))
      (float-array (map #(float (/ (- (mod % 11) 5) 8.0)) (range (* k n))))
      (float-array (* m n))])))

(defn gemm-reference
  ([a b] (gemm-reference a b [64 64 64]))
  ([^floats a ^floats b shape]
   (let [[m n k] (checked-shape shape)]
     (float-array
      (for [i (range m) j (range n)]
        (float (reduce + 0.0
                       (for [p (range k)]
                         (* (double (aget a (+ (* i k) p)))
                            (double (aget b (+ (* p n) j))))))))))))

(defn- checked-rmsnorm-shape [shape]
  (when-not (and (vector? shape) (= 2 (count shape))
                 (every? #(and (integer? %) (<= 1 % Integer/MAX_VALUE)) shape)
                 (<= (apply *' shape) Integer/MAX_VALUE))
    (throw (ex-info "RMSNorm canary requires positive [rows features] with int-sized buffers"
                    {:shape shape})))
  shape)

(defn rmsnorm-arguments
  "Deterministic [x weight out rows features eps gain] arguments for the public RMSNorm canary."
  [shape]
  (let [[rows features] (checked-rmsnorm-shape shape)]
    [(float-array (map #(float (/ (- (mod % 17) 8) 9.0)) (range (* rows features))))
     (float-array (map #(float (/ (inc (mod % 11)) 16.0)) (range features)))
     (float-array (repeat (* rows features) Float/NaN))
     (long rows) (long features) 1.0e-6 1.0]))

(defn rmsnorm-reference
  "Independent host reference for `rms-norm-reassociated!`; comparison permits float
   reassociation rather than requiring declaration-order bit identity."
  [[x weight _out rows features eps gain]]
  (float-array
   (mapcat
    (fn [row]
      (let [base (* row features)
            ss (reduce + 0.0
                       (map (fn [i]
                              (let [value (double (aget ^floats x (+ base i)))]
                                (* value value)))
                            (range features)))
            scale (/ 1.0 (Math/sqrt (+ (/ ss features) eps)))]
        (map (fn [i]
               (float (* (aget ^floats x (+ base i)) scale
                         (+ gain (aget ^floats weight i)))))
             (range features))))
    (range rows))))

(defn rmsnorm!
  "Validate and device-event measure the public equation-first cooperative RMSNorm path.

   The launch-aware roofline contract is intentionally loose: it catches scalar serialization
   cliffs while leaving schedule selection to measured tuning. Timing never includes transfers."
  [{:keys [environment-tag target compiler-revision shape max-roofline-slowdown budget-ms]
    :or {target :ze:0 shape [1 640] max-roofline-slowdown 30.0 budget-ms 800.0}}]
  (let [[rows features :as dimensions] (checked-rmsnorm-shape shape)
        identity (identity-for :rmsnorm-reassociated-resident target :float dimensions
                               :device-event environment-tag)
        args (rmsnorm-arguments dimensions)
        expected (rmsnorm-reference args)
        started (System/nanoTime)
        prepared (compiled/lower #'dl-nn/rms-norm-reassociated! args
                                 {:compiler :equation-first :target target :dtype :float
                                  :constants '[weight] :outputs '[out]})
        compile-ns (- (System/nanoTime) started)
        started (System/nanoTime)
        c (compiled/instantiate! prepared {:profile? true})
        bind-ns (- (System/nanoTime) started)]
    (try
      (let [profile (compiled/profile c)
            actual ^floats (get-in profile [:result :out])
            _ (when-not (and actual (= (alength expected) (alength actual))
                             (every? (fn [[want got]]
                                       (<= (Math/abs (- (double want) (double got)))
                                           (* 2.0e-5 (max 1.0 (Math/abs (double want))))))
                                     (map vector expected actual)))
                (throw (ex-info "production RMSNorm canary failed independent numerical reference"
                                {:expected-head (take 8 expected) :actual-head (take 8 actual)})))
            measurement (compiled/measure c :budget-ms budget-ms :cv-threshold 0.08)
            ;; One fold read of x, then x + weight reads and one out write. All are warm resident.
            ;; The conservative scalar flop estimate matters less than launch cost for B=1.
            work {:flops (* 8 rows features)
                  :warm-bytes (* 4 4 rows features)
                  :dtype :f32 :n-kernels 1}
            performance (roofline-assessment (compiler-hardware/descriptor-for target)
                                             work measurement max-roofline-slowdown)]
        {:identity (assoc identity :numerical-policy :implementation-defined-association)
         :compiler-revision compiler-revision :compile-ns compile-ns :bind-ns bind-ns
         :validated? true :schedule (:schedule prepared) :execution (compiled/ir c)
         :profile (dissoc profile :result) :measurement measurement
         :performance-contract performance})
      (finally
        (compiled/close! c)))))

(defn prepare-gemm
  ([target args]
   (compiled/lower #'gemm64! args {:target target :dtype :float :on-non-resident :throw
                                  :constants ['A 'B]}))
  ([target args shape]
   (prepare-gemm target args shape {}))
  ([target args shape {:keys [variant gemm-precision constants]
                      :or {variant :plain constants ['A 'B]}}]
   (let [entry (case variant
                 :plain #'gemm-mnk!
                 :relu #'gemm-relu!
                 :relu-composed #'gemm-relu-composed!
                 :relu-prebound #'gemm-relu-prebound!
                 (throw (ex-info "unknown GEMM canary variant" {:variant variant})))]
     (compiled/lower entry (into args (map long (checked-shape shape)))
                     (cond-> {:target target :dtype :float :on-non-resident :throw :constants constants}
                       gemm-precision (assoc :gemm-precision gemm-precision))))))

(defn compilation-evidence
  "Describe alternatives from the exact prepared program, without recompiling it.
   Entry-point counts include candidate graphs; they are not measured launch counts. Descriptor
   scratch excludes graph-owned temporaries and is not a peak-memory measurement."
  [prepared]
  (let [descriptor (:descriptor prepared)]
    {:version 1
     :resident-step-count (count (:steps descriptor))
     :descriptor-scratch-count (count (:allocs descriptor))
     :steps
     (mapv (fn [step]
             {:convention (:convention step)
              :dispatch-diagnostics
              (when-let [dispatch (:dispatch step)]
                (select-keys (:attributes dispatch)
                             [:selection :declines :matrix-graph-decline]))
              :alternatives
              (mapv (fn [candidate]
                      (let [artifacts (executable/artifacts candidate)]
                        {:strategy (executable/strategy candidate)
                         :signature (tuning/executable-signature candidate)
                         :entry-point-count (count artifacts)
                         :emission-routes (frequencies (map artifact/emission-route artifacts))}))
                    (if-let [choice (:dispatch step)]
                      (:alternatives choice)
                      (when (executable/kernel-executable? (:artifact step))
                        [(:artifact step)])))})
           (:steps descriptor))}))

(defn gemm! [{:keys [environment-tag target compiler-revision shape variant gemm-precision]
              :or {target :ocl:0 variant :plain}}]
  (let [dimensions (checked-shape (or shape [64 64 64]))
        parameterized? (or shape gemm-precision (not= :plain variant))
        workload (case variant
                   :plain (if parameterized? :gemm-mnk-resident :gemm64-resident)
                   :relu :gemm-relu-resident
                   :relu-composed :gemm-relu-composed-resident
                   :relu-prebound :gemm-relu-prebound-resident
                   (throw (ex-info "unknown GEMM canary variant" {:variant variant})))
        identity (identity-for workload target :float dimensions
                               :host-synchronized-replay environment-tag)
        args (gemm-arguments dimensions)
        product (gemm-reference (first args) (second args) dimensions)
        expected (if (= :plain variant) (vec product) (mapv #(max (float 0.0) %) product))
        started (System/nanoTime)
        prepared (if parameterized?
                   (prepare-gemm target args dimensions {:variant variant :gemm-precision gemm-precision})
                   (prepare-gemm target args))
        compile-ns (- (System/nanoTime) started)
        started (System/nanoTime)
        c (compiled/instantiate! prepared)
        bind-ns (- (System/nanoTime) started)]
    (try
      (let [resident (:executable c)
            _ (link/run! resident)
            output (some #(when (= 'C (:sym %)) %) (:out-tree c))
            _ (when-not output
                (throw (ex-info "GEMM canary has no semantic C output" {})))
            actual (vec (link/download resident (:node output)))
            _ (when-not (= expected actual)
                (throw (ex-info "production GEMM canary failed independent numerical reference"
                                {:expected-head (take 8 expected) :actual-head (take 8 actual)})))
            evidence (compilation-evidence prepared)]
        {:identity (assoc identity :numerical-policy (get-in prepared [:schedule :precision]))
         :compiler-revision compiler-revision
         :compile-ns compile-ns :bind-ns bind-ns :validated? true
         :schedule (:schedule prepared)
         :execution (compiled/ir c)
         :compilation evidence
         :measurement (measure #(link/run! resident))})
      (finally (compiled/close! c)))))

(defn -main [options-file]
  (let [{:keys [case baseline output] :as opts} (edn/read-string (slurp options-file))]
    (when-not (and (string? (:compiler-revision opts)) (not-empty (:compiler-revision opts)))
      (throw (ex-info "canary requires :compiler-revision for auditable results" {})))
    (when (and baseline output
               (= (.getCanonicalPath (io/file baseline)) (.getCanonicalPath (io/file output))))
      (throw (ex-info "canary output must not overwrite its baseline" {})))
    (let [result ((clojure.core/case case :cpu cpu! :gemm gemm! :rmsnorm rmsnorm!
                       (throw (ex-info "canary :case must be :cpu, :gemm or :rmsnorm" {}))) opts)
          baseline (when (and baseline (.exists (io/file baseline)))
                     (edn/read-string (slurp baseline)))
          result (assoc result :verdict (verdict baseline result))]
      (when output (io/make-parents output) (spit output (pr-str result)))
      (prn result)
      (System/exit (if (= :pass (:verdict result)) 0 2)))))
