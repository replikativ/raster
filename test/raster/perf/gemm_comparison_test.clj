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
                       (canary/gemm-reference (first args) (second args) (:shape options)))]
    (with-redefs [hardware/init! (constantly nil)
                  hardware/device-signature (constantly {:fixture true})
                  canary/prepare-gemm
                  (fn [_ _ _ {:keys [variant]}]
                    (swap! events conj [:compile variant])
                    {:id variant :schedule {:precision :f32-scalar}})
                  canary/compilation-evidence (fn [p] {:prepared-id (:id p)})
                  compiled/instantiate!
                  (fn [p]
                    (when (and (= mode :bind-failure) (= :relu (:id p)))
                      (throw (ex-info "injected binding failure" {})))
                    {:executable (:id p) :out-tree [{:sym 'C :node :C}]})
                  compiled/ir (constantly {:fixture true})
                  compiled/close! #(swap! events conj [:close (:executable %)])
                  link/upload! (fn [id _ values]
                                 (swap! events conj [:reset id])
                                 (swap! buffers assoc id (vec values)))
                  link/run! (fn [id]
                              (swap! events conj [:run id])
                              (when-not (= mode :missing-write)
                                (swap! buffers assoc id expected)))
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

(deftest probe-budgets-decline-before-runtime-initialization
  (with-redefs [hardware/init! #(throw (AssertionError. "runtime initialized before admission"))]
    (doseq [overrides [{:shape [0 2 2]} {:shape [2048 2048 2048]}
                       {:shape [1 1]} {:rounds 1} {:rounds 121} {:warmup-rounds -1}
                       {:warmup-rounds 1} {:environment-tag ""} {:compiler-revision nil}]]
      (is (thrown? clojure.lang.ExceptionInfo (comparison/run! (merge options overrides)))))))
