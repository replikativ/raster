(ns raster.perf.gemm-comparison-test
  (:require [clojure.test :refer [deftest is]]
            [raster.perf.gemm-comparison :as comparison]
            [raster.perf.production-canary :as canary]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.runtime.hardware :as hardware]))

(def options {:shape [2 2 2] :rounds 2 :warmup-rounds 2
              :compiler-revision "fixture" :environment-tag "synthetic-not-a-device-result"})

(defn- with-fake-runtime [mode f]
  (let [events (atom [])
        buffers (atom {})
        args (canary/gemm-arguments (:shape options))
        expected (mapv #(max (float 0) %)
                       (canary/gemm-reference (first args) (second args) (:shape options)))
        negative (float-array (map #(float (- %)) (first args)))
        negative-expected (mapv #(max (float 0) %)
                                (canary/gemm-reference negative (second args) (:shape options)))
        result-for (fn [id]
                     (if (and (not= :stale-activation mode)
                              (= (vec negative) (get @buffers [id :A])))
                       negative-expected expected))]
    (with-redefs [hardware/init! (constantly nil)
                  hardware/device-signature (constantly {:fixture true})
                  canary/prepare-gemm
                  (fn [_ _ _ {:keys [variant constants]}]
                    (swap! events conj [:compile variant])
                    {:id variant :constants constants :schedule {:precision :f32-scalar}})
                  canary/compilation-evidence (fn [p] {:prepared-id (:id p)})
                  compiled/instantiate!
                  (fn [p opts]
                    (swap! events conj [:bind (:id p) opts])
                    (when (and (= mode :bind-failure) (= :relu (:id p)))
                      (throw (ex-info "injected binding failure" {})))
                    {:executable (:id p) :out-tree [{:sym 'C :node :C}]
                     :in-tree [{:sym 'A :node :A
                                :role (if (or (= mode :constant-activation)
                                              (some #{'A} (:constants p))) :constant :input)}]})
                  compiled/ir (constantly {:fixture true})
                  compiled/close! #(swap! events conj [:close (:executable %)])
                  link/upload! (fn [id node values]
                                 (if (= node :A)
                                   (do (swap! events conj [:activation id (vec values)])
                                       (swap! buffers assoc [id :A] (vec values)))
                                   (do (swap! events conj [:reset id])
                                       (swap! buffers assoc id (vec values)))))
                  link/run! (fn [id]
                              (swap! events conj [:run id])
                              (when-not (= mode :missing-write)
                                (swap! buffers assoc id (result-for id))))
                  link/profile! (fn [id]
                                  (swap! events conj [:profile id])
                                  (when-not (= mode :missing-write)
                                    (swap! buffers assoc id (result-for id)))
                                  {:device-wall-ms (if (map? mode) (:duration mode) 0.0125)
                                   :host-wall-ms 99.0 :kernel-total-ms 0.01
                                   :profile [{:kernel-name "observed-entry" :ms 0.01}]})
                  link/download (fn [id _] (get @buffers id))]
      (f events))))

(deftest interleaved-public-probe-retains-evidence-and-cleans-up
  (with-fake-runtime :success
    (fn [events]
      (let [result (comparison/run! options)]
        (is (= [:relu-composed :relu] (mapv :id (:candidates result))))
        (is (= 2 (count (filter #(= :compile (first %)) @events))))
        (is (= 10 (count (filter #(= :reset (first %)) @events))))
        (is (= 10 (count (filter #(= :run (first %)) @events))))
        (is (= [[:close :relu] [:close :relu-composed]] (take-last 2 @events)))
        (is (get-in result [:validation :every-replay?]))
        (is (= :host-synchronized-replay (get-in result [:scope :timing-source])))
        (is (= [:relu-composed :relu :relu :relu-composed]
               (mapv :candidate (get-in result [:comparison :samples]))))
        (is (every? #(= :host-synchronized-replay (:timing-source %))
                    (vals (get-in result [:comparison :measurements]))))))))

(deftest failures-release-every-successfully-bound-candidate
  (doseq [[mode closed] [[:bind-failure [[:close :relu-composed]]]
                         [:missing-write [[:close :relu] [:close :relu-composed]]]]]
    (with-fake-runtime mode
      (fn [events]
        (is (thrown? clojure.lang.ExceptionInfo (comparison/run! options)))
        (is (= closed (filter #(= :close (first %)) @events)))))))

(deftest device-events-are-not-host-time-or-kernel-sums
  (with-fake-runtime :success
    (fn [events]
      (let [result (comparison/run! (assoc options :timing-source :device-event))]
        (is (every? #(= 12500.0 (:ns %)) (get-in result [:comparison :samples])))
        (is (= :device-event (get-in result [:scope :timing-source])))
        (is (= 10 (count (:replay-profiles result))))
        (is (every? #(= "observed-entry" (get-in % [:profile :profile 0 :kernel-name]))
                    (:replay-profiles result)))
        (is (not-any? #(= :run (first %)) @events))
        (is (every? #(= {:profile? true} (nth % 2))
                    (filter #(= :bind (first %)) @events)))))))

(deftest device-profile-failures-never-fall-back-and-release-resources
  (doseq [mode [{:duration nil} {:duration Double/NaN} {:duration Double/POSITIVE_INFINITY}
               {:duration -1} :missing-write]]
    (with-fake-runtime mode
      (fn [events]
        (is (thrown? clojure.lang.ExceptionInfo
                     (comparison/run! (assoc options :timing-source :device-event))))
        (is (not-any? #(= :run (first %)) @events))
        (is (= [[:close :relu] [:close :relu-composed]] (take-last 2 @events)))))))

(deftest prebound-comparison-retains-the-selected-public-source-identity
  (with-fake-runtime :success
    (fn [events]
      (let [result (comparison/run! (assoc options :composed-variant :relu-prebound
                                                 :timing-source :device-event))]
        (is (= [:relu-prebound :relu] (mapv :id (:candidates result))))
        (is (= [:relu-prebound :relu :relu :relu-prebound]
               (mapv :candidate (get-in result [:comparison :samples]))))
        (is (= [[:close :relu] [:close :relu-prebound]] (take-last 2 @events)))))))

(deftest changing-activation-is-refreshed-and-validated-on-every-replay
  (doseq [clock [:device-event :host-synchronized-replay]]
    (with-fake-runtime :success
      (fn [events]
        (let [result (comparison/run! (assoc options :input-policy :changing-activation
                                                   :timing-source clock))
              updates (filter #(= :activation (first %)) @events)]
          (is (= ['B] (get-in result [:scope :constant-operands])))
          (is (= :alternating-sign (get-in result [:input-recipe :activation-pattern])))
          (is (= 10 (count updates)))
          (doseq [id [:relu-composed :relu]]
            (let [values (mapv #(nth % 2) (filter #(= id (second %)) updates))]
              (is (= [(first values) (second values) (first values) (second values) (first values)]
                     values))
              (is (not= (first values) (second values)))))
          (is (get-in result [:validation :every-replay?])))))))

(deftest changing-activation-declines-stale-results-and-constant-inputs
  (doseq [mode [:stale-activation :constant-activation]]
    (with-fake-runtime mode
      (fn [events]
        (is (thrown? clojure.lang.ExceptionInfo
                     (comparison/run! (assoc options :input-policy :changing-activation
                                                    :timing-source :device-event))))
        (is (= [[:close :relu] [:close :relu-composed]] (take-last 2 @events)))))))

(deftest probe-budgets-decline-before-runtime-initialization
  (with-redefs [hardware/init! #(throw (AssertionError. "runtime initialized before admission"))]
    (doseq [overrides [{:shape [0 2 2]} {:shape [2048 2048 2048]}
                       {:shape [1 1]} {:rounds 1} {:rounds 121} {:warmup-rounds -1}
                       {:warmup-rounds 1} {:environment-tag ""} {:compiler-revision nil}
                       {:timing-source :unknown} {:composed-variant :relu} {:input-policy :unknown}]]
      (is (thrown? clojure.lang.ExceptionInfo (comparison/run! (merge options overrides)))))))
