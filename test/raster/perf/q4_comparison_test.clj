(ns raster.perf.q4-comparison-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.perf.q4-comparison :as comparison]
            [raster.runtime.hardware :as hardware]))

(def options {:shape [1 256 2] :rounds 2 :warmup-rounds 2
              :compiler-revision "fixture" :environment-tag "synthetic-not-a-device-result"})

(defn- with-fake-runtime [mode f]
  (let [events (atom [])
        buffers (atom {})
        expected (:expected-bits (comparison/problem (:shape options)))]
    (with-redefs [hardware/init! (constantly nil)
                  hardware/device-signature (constantly {:fixture true})
                  comparison/prepare-candidate
                  (fn [_ {:keys [id compiler]} _]
                    (swap! events conj [:compile id compiler])
                    {:id id :compiler compiler :descriptor {:steps [{:convention :fixture}]}})
                  compiled/instantiate!
                  (fn [prepared opts]
                    (swap! events conj [:bind (:id prepared) opts])
                    (when (and (= mode :bind-failure) (= :serial-source (:id prepared)))
                      (throw (ex-info "injected bind failure" {})))
                    {:id (:id prepared) :executable (:id prepared)
                     :out-tree [{:sym 'y :node :y}]})
                  compiled/profile
                  (fn [instance]
                    (swap! events conj [:profile (:id instance)])
                    {:device-wall-ms (if (= mode :missing-time) nil 0.0125)
                     :profile [{:kernel-name (name (:id instance)) :ms 0.01}]
                     :result {:y (float-array
                                  (map #(Float/intBitsToFloat
                                         (if (= mode :wrong-result) (inc %) %)) expected))}})
                  compiled/ir (fn [instance] [{:convention :fixture :id (:id instance)}])
                  compiled/close! #(swap! events conj [:close (:id %)])
                  link/upload! (fn [resident node values]
                                 (swap! buffers assoc [resident node] (vec values))
                                 (swap! events conj [:poison resident node]))]
      (f events buffers))))

(deftest public-comparison-validates-every-replay-and-cleans-up
  (with-fake-runtime :success
    (fn [events buffers]
      (let [result (comparison/run! options)]
        (is (= [:generated-product :serial-source] (mapv :id (:candidates result))))
        (is (= [:equation-first :resident-descriptor] (mapv :compiler (:candidates result))))
        (is (= 10 (count (filter #(= :profile (first %)) @events))))
        (is (= 10 (count (filter #(= :poison (first %)) @events))))
        (is (every? #(every? Float/isNaN %) (vals @buffers)))
        (is (= [[:close :serial-source] [:close :generated-product]] (take-last 2 @events)))
        (is (= :device-event (get-in result [:scope :timing-source])))
        (is (= :raw-float-bits (get-in result [:validation :comparison])))
        (is (= [:generated-product :serial-source :serial-source :generated-product]
               (mapv :candidate (get-in result [:comparison :samples]))))))))

(deftest comparison-fails-closed-and-releases-bound-candidates
  (doseq [[mode closed] [[:bind-failure [[:close :generated-product]]]
                         [:wrong-result [[:close :serial-source] [:close :generated-product]]]
                         [:missing-time [[:close :serial-source] [:close :generated-product]]]]]
    (with-fake-runtime mode
      (fn [events _]
        (is (thrown? clojure.lang.ExceptionInfo (comparison/run! options)))
        (is (= closed (filter #(= :close (first %)) @events)))))))

(deftest options-decline-before-runtime-initialization
  (with-redefs [hardware/init! #(throw (AssertionError. "runtime initialized before admission"))]
    (doseq [overrides [{:shape [1 255 2]} {:shape [0 256 2]} {:shape [1 256]}
                       {:shape [1 256 1000001]} {:rounds 1} {:rounds 3}
                       {:warmup-rounds -1} {:warmup-rounds 1}
                       {:environment-tag ""} {:compiler-revision nil}]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (comparison/run! (merge options overrides)))))))

(deftest deterministic-problem-has-the-public-abi-and-ggml-oracle
  (let [{:keys [arguments expected-bits]} (comparison/problem [1 256 3])]
    (is (= 12 (count arguments)))
    (is (= 3 (count expected-bits)))
    (is (every? integer? expected-bits))))
