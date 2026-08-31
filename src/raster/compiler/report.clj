(ns raster.compiler.report
  "Pure normalization of compiler diagnostics.

   This namespace does not compile, inspect source forms, or infer operations. It turns the
   authoritative records produced by the pipeline and its emitters into a small, stable report
   suitable for compatibility ledgers and tooling."
  (:require [clojure.string :as str]))

(def schema-version 1)

(defn- counter
  [m k]
  (long (or (get m k) 0)))

(defn- decline-key?
  [k]
  (and (keyword? k) (str/includes? (name k) "declin")))

(defn- decline-values
  [value]
  (cond
    (nil? value) []
    (map? value) [value]
    (sequential? value) value
    :else [{:value value}]))

(defn- normalize-decline
  [stage kind decline]
  (cond-> {:stage stage :kind kind}
    (:leaf decline) (assoc :leaf (:leaf decline))
    (:reason decline) (assoc :reason (:reason decline))
    (:message decline) (assoc :message (:message decline))
    (and (nil? (:leaf decline))
         (nil? (:reason decline))
         (nil? (:message decline)))
    (assoc :value (if (contains? decline :value) (:value decline) decline))))

(defn- pass-stats
  [pipeline]
  (into (sorted-map)
        (filter (fn [[k v]]
                  (and (keyword? k)
                       (str/ends-with? (name k) "-stats")
                       (map? v))))
        pipeline))

(defn- pass-declines
  [stats]
  (->> stats
       (mapcat (fn [[stage values]]
                 (mapcat (fn [[kind value]]
                           (when (and (decline-key? kind) (some? value))
                             (map #(normalize-decline stage kind %)
                                  (decline-values value))))
                         values)))
       (remove nil?)
       vec))

(defn- kernel-attribute
  [kernel k]
  (or (get kernel k) (get-in kernel [:attributes k])))

(defn- kernel-strategy
  [kernel]
  (or (kernel-attribute kernel :strategy)
      (get-in kernel [:provenance :strategy])))

(defn- emitted-declines
  [kernels]
  (->> kernels
       (mapcat (fn [kernel]
                 (map #(normalize-decline :emission :candidate %)
                      (decline-values (kernel-attribute kernel :declines)))))
       vec))

(defn- source-dialect
  [soac-stats segop-stats backend-stats]
  (or (:route soac-stats)
      (when (pos? (counter segop-stats :typed-soac-reused)) :typed-soac)
      (when (or (pos? (counter segop-stats :segops-lowered))
                (pos? (counter segop-stats :kernel-graphs-lowered))
                (pos? (counter backend-stats :segop-relowered)))
        :compatibility)
      :scalar))

(defn from-pipeline
  "Normalize one `raster.compiler.pipeline/show-pipeline` result.

   Residency is deliberately reported as unassessed. A successful backend emission does not prove
   that the computation is a straight-line resident program, and `:allocs` on a resident descriptor
   means device scratch rather than host allocation. The resident compiler will enrich this same
   schema when it owns those facts."
  [pipeline]
  (let [stats (pass-stats pipeline)
        soac-stats (:soac-fused-stats pipeline)
        segop-stats (:segop-lowered-stats pipeline)
        backend-stats (:backend-applied-stats pipeline)
        kernels (vec (:kernels pipeline))]
    {:schema-version schema-version
     :route {:backend (:backend pipeline)
             :source-dialect (source-dialect soac-stats segop-stats backend-stats)
             :typed-validated (true? (:typed-validated soac-stats))
             :declines (pass-declines stats)}
     :fusion {:vertical (counter soac-stats :vertical)
              :horizontal (counter soac-stats :horizontal)
              :iterations (counter soac-stats :iterations)
              :placements (counter soac-stats :placements)}
     :lowering {:segops (counter segop-stats :segops-lowered)
                :kernel-graphs (counter segop-stats :kernel-graphs-lowered)
                :typed-reused (counter segop-stats :typed-soac-reused)
                :typed-scalar-equations (counter segop-stats :typed-scalar-equations)
                :backend-reused (counter backend-stats :segop-reused)
                :backend-relowered (counter backend-stats :segop-relowered)
                :fallback (counter backend-stats :fallback)}
     :emission {:kernel-count (count kernels)
                :targets (->> kernels (map :target) (remove nil?) distinct vec)
                :strategies (->> kernels (map kernel-strategy) (remove nil?) distinct vec)
                :fallback-reasons (->> kernels
                                       (map #(kernel-attribute % :fallback-reason))
                                       (remove nil?) distinct vec)
                :declines (emitted-declines kernels)}
     :residency {:assessed? false
                 :resident? nil
                 :device-scratch-count nil
                 :host-array-allocs-in-compute nil
                 :internal-host-roundtrips nil}
     :pass-stats stats}))
