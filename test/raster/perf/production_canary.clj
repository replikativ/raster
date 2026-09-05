(ns raster.perf.production-canary
  "Opt-in canaries of the public Raster compiler and resident replay paths.
   No timing assertion runs in ordinary CI and no run creates/updates its own baseline."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
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
    (nil? baseline) :unbaselined
    (not= (:identity baseline) (:identity result)) :incomparable
    (not (and (valid-measurement? baseline)
              (true? (get-in baseline [:measurement :stationary?])))) :invalid-baseline
    (<= (get-in result [:measurement :median-ns])
        (* 1.15 (get-in baseline [:measurement :median-ns]))) :pass
    :else :regression))

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

(defn gemm-arguments []
  [(float-array (map #(float (/ (- (mod % 13) 6) 8.0)) (range 4096)))
   (float-array (map #(float (/ (- (mod % 11) 5) 8.0)) (range 4096)))
   (float-array 4096)])

(defn gemm-reference [^floats a ^floats b]
  (float-array
   (for [i (range 64) j (range 64)]
     (float (reduce + 0.0
                    (for [k (range 64)]
                      (* (double (aget a (+ (* i 64) k))) (double (aget b (+ (* k 64) j))))))))))

(defn prepare-gemm [target args]
  (compiled/lower #'gemm64! args {:target target :dtype :float :on-non-resident :throw
                                 :constants ['A 'B]}))

(defn gemm! [{:keys [environment-tag target compiler-revision] :or {target :ocl:0}}]
  (let [identity (identity-for :gemm64-resident target :float [64 64 64]
                               :host-synchronized-replay environment-tag)
        args (gemm-arguments)
        expected (vec (gemm-reference (first args) (second args)))
        started (System/nanoTime)
        prepared (prepare-gemm target args)
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
            alternatives (keep :dispatch (get-in prepared [:descriptor :steps]))]
        {:identity (assoc identity :numerical-policy (get-in prepared [:schedule :precision]))
         :compiler-revision compiler-revision
         :compile-ns compile-ns :bind-ns bind-ns :validated? true
         :schedule (:schedule prepared)
         :execution (compiled/ir c)
         :candidate-strategies (mapv #(mapv dispatch/alternative-strategy (:alternatives %))
                                     alternatives)
         :candidate-signatures (mapv #(mapv tuning/executable-signature (:alternatives %))
                                     alternatives)
         :measurement (measure #(link/run! resident))})
      (finally (compiled/close! c)))))

(defn -main [options-file]
  (let [{:keys [case baseline output] :as opts} (edn/read-string (slurp options-file))]
    (when-not (and (string? (:compiler-revision opts)) (not-empty (:compiler-revision opts)))
      (throw (ex-info "canary requires :compiler-revision for auditable results" {})))
    (when (and baseline output
               (= (.getCanonicalPath (io/file baseline)) (.getCanonicalPath (io/file output))))
      (throw (ex-info "canary output must not overwrite its baseline" {})))
    (let [result ((clojure.core/case case :cpu cpu! :gemm gemm!
                       (throw (ex-info "canary :case must be :cpu or :gemm" {}))) opts)
          baseline (when (and baseline (.exists (io/file baseline)))
                     (edn/read-string (slurp baseline)))
          result (assoc result :verdict (verdict baseline result))]
      (when output (io/make-parents output) (spit output (pr-str result)))
      (prn result)
      (System/exit (if (= :pass (:verdict result)) 0 2)))))
