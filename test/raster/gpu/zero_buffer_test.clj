(ns raster.gpu.zero-buffer-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.gpu.core :as gpu]))

(defn- fixture []
  (let [allocation (bview/allocation {:id :allocation :byte-size 4 :device :ocl:0
                                     :memory-space :device :ownership :owned})
        buffer {:dtype :byte :n-elements 4 :byte-size 4}]
    (atom {:device-id :ocl:0 :session-id (random-uuid) :buffers {:buffer buffer}
           :allocations {:buffer allocation} :events {}})))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest whole-root-zeroing-preserves-the-synchronous-owner-boundary
  (let [session (fixture)
        calls (atom 0)]
    (with-redefs-fn
      {#'gpu/rt-resolve (fn [_ operation]
                         (is (= "zero-buffer!" operation))
                         (fn [buffer]
                           (is (identical? buffer (get-in @session [:buffers :buffer])))
                           (swap! calls inc)))}
      (fn []
        (is (= :buffer (gpu/zero-buffer! session :buffer)))
        (is (= 1 @calls))
        (is (= :gpu-zero-partial-view
               (reason #(gpu/zero-buffer! session (gpu/buffer-view session :buffer {:shape [2]})))))
        (is (= 1 @calls))))))

(deftest zeroing-never-assumes-unawaited-work-is-safe
  (let [session (fixture)
        calls (atom 0)
        known {:buffer-keys #{:other-key} :allocation-ids #{:allocation}
               :resident-buffers []}]
    (with-redefs-fn
      {#'gpu/rt-resolve (fn [& _] (fn [_] (swap! calls inc)))}
      (fn []
        (doseq [event [(assoc known :status :pending)
                       (assoc known :status :awaited)
                       {:status :pending}
                       {:status :pending :buffer-keys #{:other-key}}
                       {:status :pending :buffer-keys #{} :allocation-ids nil :resident-buffers nil}]]
          (swap! session assoc :events {:event event})
          (is (= :gpu-zero-pending-events (reason #(gpu/zero-buffer! session :buffer))))
          (is (zero? @calls)))
        (doseq [event [(assoc known :status :complete)
                       {:status :pending :buffer-keys #{:other-key}
                        :allocation-ids #{:other-allocation} :resident-buffers []}]]
          (swap! session assoc :events {:event event})
          (is (= :buffer (gpu/zero-buffer! session :buffer))))
        (is (= 2 @calls))))))

(deftest failed-zeroing-does-not-manufacture-completion-or-initialization
  (let [session (fixture)
        before @session
        failure (ex-info "fill failed" {})]
    (with-redefs-fn {#'gpu/rt-resolve (fn [& _] (fn [_] (throw failure)))}
      (fn []
        (is (identical? failure (try (gpu/zero-buffer! session :buffer)
                                     (catch Throwable error error))))
        (is (= before @session))))))
