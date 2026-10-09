(ns raster.compiler.ir.distributed-cost-test
  "Supplied synthetic observations test admission, not measured fabric performance."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.distributed-plan :as plan]
            [raster.compiler.ir.distributed-plan-test :as fixture]))

(defn- context []
  {:devices (into {} (for [device [:gpu-0 :gpu-1]]
                      [device {:device {:driver "synthetic"}
                               :hardware-evidence {:device-id device :driver-version "synthetic"}}]))
   :transport :host-staged :max-staging-bytes 1024
   :transfer-layouts
   {:send-gradient
    (into {} (for [[role device] [[:source :gpu-0] [:target :gpu-1]]]
               [role {:dtype :float :shape [250000] :strides [1] :byte-offset 0 :byte-length 1000000
                      :allocation {:device device :byte-size 1000000 :memory-space :device
                                   :alignment 16 :coherence :device-only}}]))}})

(defn- options [p]
  (let [context (context)]
    {:route-context context :route-policy {:now-ms 2000 :cold-warm :warm}
     :profiles
     (mapv (fn [i duration]
             (let [devices (update-vals (:devices context) #(assoc % :session-id [:fresh i]))]
               {:plan p :execution-model :synchronous-serialized
                :transport :host-staged :max-staging-bytes 1024
                :observed-at {:clock :unix-epoch-ms :value 1000}
                :devices-before devices :devices-after devices :route-cost-context context
                :steps [{:step :send-gradient :kind :transfer :transport :host-staged
                         :route [:gpu-0->gpu-1] :bytes 1000000
                         :source-view (get-in context [:transfer-layouts :send-gradient :source])
                         :target-view (get-in context [:transfer-layouts :send-gradient :target])
                         :host-wall-ns duration :route-timing-source :host-monotonic
                         :host-timing-scope :binding-execution-and-release}]}))
           (range 3) [1000 1010 990])}))

(deftest matching-whole-route-cost-influences-simulation-not-certification
  (let [p (#'fixture/training-plan) baseline (plan/simulate p)
        certified (plan/certify p) result (plan/simulate p (options p))
        evidence (get-in result [:route-cost-evidence :send-gradient])]
    (is (= 1300 (:makespan-ns result)))
    (is (= 1000 (:duration-ns evidence)))
    (is (= :empirical-host-step (:source evidence)))
    (is (= [1000.0 1010.0 990.0] (get-in evidence [:measurement :samples-ns])))
    (is (= 3 (get-in evidence [:measurement :n])))
    (is (false? (:empirical-costs-certified? result)))
    (is (= (:transferred-bytes baseline) (:transferred-bytes result)))
    (is (= baseline (plan/simulate p)))
    (is (= certified (plan/verify! certified)))
    (is (= (get-in certified [:certificate :route-costs :send-gradient :duration-ns])
           (get-in baseline [:timeline :send-gradient :duration-ns])))))

(deftest mismatched-stale-noisy-and-repeated-evidence-falls-back-visibly
  (let [p (#'fixture/training-plan) baseline (plan/simulate p) opts (options p)]
    (doseq [[changed reason]
            [[(assoc-in opts [:profiles 0 :route-cost-context :transport] :resident-copy) :context-mismatch]
             [(assoc-in opts [:profiles 0 :route-cost-context :max-staging-bytes] 2048) :context-mismatch]
             [(assoc-in opts [:profiles 0 :route-cost-context :devices :gpu-0 :hardware-evidence :driver-version] "other") :context-mismatch]
             [(assoc-in opts [:profiles 0 :route-cost-context :transfer-layouts :send-gradient :source :dtype] :double) :context-mismatch]
             [(assoc-in opts [:profiles 0 :route-cost-context :transfer-layouts :send-gradient :source :strides] [2]) :context-mismatch]
             [(assoc-in opts [:profiles 0 :steps 0 :bytes] 4) :context-mismatch]
             [(assoc-in opts [:profiles 0 :steps 0 :route] []) :context-mismatch]
             [(assoc-in opts [:profiles 0 :steps 0 :route-timing-source] :device-event) :context-mismatch]
             [(assoc-in opts [:profiles 0 :steps 0 :host-timing-scope] :kernel-only) :context-mismatch]
             [(-> opts
                  (assoc-in [:profiles 0 :devices-before :gpu-0 :hardware-evidence :driver-version] "other")
                  (assoc-in [:profiles 0 :devices-after :gpu-0 :hardware-evidence :driver-version] "other")) :context-mismatch]
             [(assoc-in opts [:profiles 0 :transport] :resident-copy) :context-mismatch]
             [(assoc-in opts [:profiles 0 :max-staging-bytes] 2048) :context-mismatch]
             [(assoc-in opts [:profiles 0 :steps 0 :source-view :byte-offset] 4) :context-mismatch]
             [(assoc-in opts [:profiles 0 :devices-after :gpu-0 :session-id] :changed) :context-mismatch]
             [(assoc-in opts [:profiles 0 :observed-at :clock] :process-monotonic) :stale-or-clock-mismatch]
             [(assoc-in opts [:profiles 0 :observed-at :value] 2001) :stale-or-clock-mismatch]
             [(assoc-in opts [:profiles 0 :observed-at :value] Long/MIN_VALUE) :stale-or-clock-mismatch]
             [(assoc-in opts [:route-policy :max-age-ms] 999) :stale-or-clock-mismatch]
             [(assoc opts :profiles (subvec (:profiles opts) 0 2)) :insufficient-samples]
             [(assoc opts :profiles (vec (repeat 3 (first (:profiles opts))))) :repeated-owner]
             [(-> opts
                  (assoc-in [:profiles 1 :devices-before :gpu-0 :session-id] [:fresh 0])
                  (assoc-in [:profiles 1 :devices-after :gpu-0 :session-id] [:fresh 0])) :repeated-owner]
             [(assoc-in opts [:profiles 0 :steps 0 :host-wall-ns] 10000) :noisy-samples]
             [(assoc-in opts [:profiles 0 :steps 0 :host-wall-ns] Double/NaN) :invalid-duration]
             [(assoc-in opts [:profiles 0 :steps 0 :host-wall-ns] 0) :invalid-duration]
             [(update-in opts [:route-context :transfer-layouts] dissoc :send-gradient) :missing-context]
             [(assoc-in opts [:route-context :devices :gpu-0 :hardware-evidence] 42) :missing-context]]]
      (let [result (plan/simulate p changed) evidence (get-in result [:route-cost-evidence :send-gradient])]
        (is (= reason (:reason evidence)))
        (is (= :analytical (:source evidence)))
        (is (= (:makespan-ns baseline) (:makespan-ns result)))))))

(deftest malformed-caller-policy-is-not-silent-fallback
  (let [p (#'fixture/training-plan) opts (options p)]
    (doseq [invalid [nil {:unknown true} (dissoc opts :route-context)
                     (assoc opts :route-policy true)
                     (assoc-in opts [:route-policy :min-samples] 1)
                     (assoc-in opts [:route-policy :cv-threshold] Double/NaN)
                     (assoc-in opts [:route-policy :now-ms] -1)
                     (update opts :route-policy dissoc :cold-warm)]]
      (is (= :distributed-cost-options
             (try (plan/simulate p invalid) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest shared-omissions-cannot-become-matching-physical-facts
  (let [p (#'fixture/training-plan) opts (options p) baseline (plan/simulate p)]
    (doseq [path [[:transport] [:max-staging-bytes]
                  [:devices :gpu-0 :device] [:devices :gpu-0 :hardware-evidence :device-id]
                  [:transfer-layouts :send-gradient :source :dtype]
                  [:transfer-layouts :send-gradient :source :shape]
                  [:transfer-layouts :send-gradient :source :strides]
                  [:transfer-layouts :send-gradient :source :byte-length]
                  [:transfer-layouts :send-gradient :source :allocation :alignment]]]
      (let [remove-field (fn [ctx] (if (= 1 (count path)) (dissoc ctx (first path))
                                      (update-in ctx (pop path) dissoc (peek path))))
            changed (-> opts (update :route-context remove-field)
                        (update :profiles #(mapv (fn [report]
                                                   (update report :route-cost-context remove-field)) %)))
            result (plan/simulate p changed)]
        (is (= :missing-context (get-in result [:route-cost-evidence :send-gradient :reason])))
        (is (= (:makespan-ns baseline) (:makespan-ns result)))))))

(deftest unsupported-shared-transport-and-placement-are-not-evidence
  (let [p (#'fixture/training-plan) opts (options p)]
    (doseq [[field value] [[:transport :unknown] [:transport :resident-copy]
                          [:max-staging-bytes 0]]]
      (let [changed (-> opts (assoc-in [:route-context field] value)
                        (update :profiles #(mapv (fn [report]
                                                   (-> report (assoc field value)
                                                       (assoc-in [:route-cost-context field] value))) %)))]
        (is (= :missing-context
               (get-in (plan/simulate p changed) [:route-cost-evidence :send-gradient :reason])))))))
