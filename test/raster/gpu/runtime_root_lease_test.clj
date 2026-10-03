(ns raster.gpu.runtime-root-lease-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))
(defn- live-state []
  (let [state (atom {:initialized? false})]
    (root/initialize! state [] (fn [_] {:context (Object.)}))
    state))

(deftest leases-pin-exact-generations-and-release-idempotently
  (let [state (live-state) other (live-state)
        lease (root/lease! state) owner (::cleanup/owner lease)]
    (is (= 1 (root/lease-count state)))
    (is (identical? lease (root/assert-lease-live! state lease)))
    (is (= :runtime-generation-mismatch
           (:reason (ex-data (error-of #(root/assert-lease-live! other lease))))))
    (is (= :runtime-generation-mismatch
           (:reason (ex-data (error-of #(root/assert-lease-live! state
                                                                 (assoc lease ::root/lease-token (random-uuid))))))))
    (cleanup/release! owner)
    (cleanup/release! owner)
    (is (zero? (root/lease-count state)))
    (is (some? (error-of #(root/assert-lease-live! state lease))))))

(deftest concurrent-leases-have-distinct-owners-and-balanced-retirement
  (let [state (live-state)
        leases (mapv deref (mapv (fn [_] (future (root/lease! state))) (range 12)))]
    (is (= 12 (root/lease-count state)))
    (is (= 12 (count (set (map ::root/lease-token leases)))))
    (doseq [job (mapv (fn [lease] (future (cleanup/release! (::cleanup/owner lease)))) leases)]
      @job)
    (is (zero? (root/lease-count state)))))

(deftest failed-native-child-keeps-lease-but-permits-known-resource-retirement
  (let [state (atom {:initialized? false})
        failure (ex-info "unknown native child acquisition" {})]
    (root/initialize! state [{:id :async :release (fn [_])}] (fn [_] {}))
    (let [lease (root/lease! state) entry (::root/entry @state)]
      (is (identical? failure
                      (error-of #(root/acquire-child! state :async
                                                      (fn [_] (fn [] (throw failure)))))))
      (is (= :runtime-root-unavailable
             (:reason (ex-data (error-of #(root/assert-lease-live! state lease))))))
      (is (= 1 (count @(:leases entry))))
      (cleanup/release! (::cleanup/owner lease))
      (is (empty? @(:leases entry)))
      (is (= :runtime-root-leases-incomplete
             (:reason (ex-data (error-of #(root/shutdown-construction! state)))))))))

(deftest lost-root-keeps-exact-lease-owner-as-cleanup-debt
  (let [state (live-state) lease (root/lease! state)
        entry (::root/entry @state) owner (::cleanup/owner lease)]
    ;; Deliberate authority-loss injection, not a supported caller mutation API.
    (swap! state dissoc ::root/entry)
    (is (= :runtime-generation-mismatch
           (:reason (ex-data (error-of #(cleanup/release! owner))))))
    (is (identical? owner (get @(:leases entry) (::root/lease-token lease))))
    (is (= [:runtime-root-lease] (cleanup/pending owner)))))
