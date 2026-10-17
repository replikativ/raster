(ns raster.gpu.compilation-arena-test
  "Hardware-free controls for the existing staged compiler arena owner."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.method-entry :as entry]
            [raster.gpu.ze-runtime :as ze]))

(deftest rejected-completed-attempt-releases-only-its-own-arena
  (let [scope (create-ns (gensym "raster.arena-proof."))
        old (fn [] :old) replacement (fn [] :new)
        v (intern scope 'input old)
        attempts (atom 0) closed (atom [])]
    (try
      (with-redefs [ze/close-kernel-arena! #(swap! closed conj %)]
        (let [compiled
              (entry/call-with-definition-inputs
               #(ze/make-gpu-fn
                 (fn []
                   (swap! attempts inc)
                   (entry/observe-definition-input! v scope 'input)
                   (let [body @v]
                     (when (= 1 @attempts)
                       (alter-var-root v (constantly replacement)))
                     (fn [] (body)))))
               ze/close!)]
          (try
            (is (= 2 @attempts))
            (is (= :new (compiled)))
            (is (= 1 (count @closed)))
            (is (not= (:arena-id (meta compiled)) (first @closed)))
            (finally (ze/close! compiled))))
        (is (= 2 (count (distinct @closed)))))
      (finally (remove-ns (ns-name scope))))))

(deftest partial-attempt-releases-arena-before-retry
  (let [attempts (atom 0) closed (atom [])]
    (with-redefs [ze/close-kernel-arena! #(swap! closed conj %)]
      (let [compiled
            (entry/call-with-definition-inputs
             #(ze/make-gpu-fn
               (fn []
                 (when (= 1 (swap! attempts inc))
                   (throw (ex-info "lazy compilation invalidated proof"
                                   {:reason :stale-definition-proof-input})))
                 (fn [] :ready)))
             ze/close!)]
        (try
          (is (= 2 @attempts))
          (is (= :ready (compiled)))
          (is (= 1 (count @closed)))
          (finally (ze/close! compiled))))
      (is (= 2 (count (distinct @closed)))))))

(deftest failed-discard-stops-retry-and-preserves-both-errors
  (let [scope (create-ns (gensym "raster.discard-proof."))
        v (intern scope 'input :before)
        attempts (atom 0)
        cleanup-failure (ex-info "discard failed" {:reason :cleanup-failure})]
    (try
      (let [failure
            (try
              (entry/call-with-definition-inputs
               #(do
                  (swap! attempts inc)
                  (entry/observe-definition-input! v scope 'input)
                  (alter-var-root v (constantly :after))
                  :resource)
               (fn [_] (throw cleanup-failure)))
              nil
              (catch clojure.lang.ExceptionInfo error error))]
        (is (= :definition-proof-discard-failed (:reason (ex-data failure))))
        (is (= 1 @attempts))
        (is (identical? cleanup-failure (ex-cause failure)))
        (is (= [:stale-definition-proof-input]
               (mapv #(-> % ex-data :reason) (.getSuppressed cleanup-failure)))))
      (finally (remove-ns (ns-name scope))))))

(deftest partial-compilation-cleanup-failure-stops-retry
  (let [attempts (atom 0)
        primary (ex-info "stale during emission" {:reason :stale-definition-proof-input})
        cleanup-failure (ex-info "arena cleanup failed" {:reason :cleanup-failure})]
    (with-redefs [ze/close-kernel-arena! (fn [_] (throw cleanup-failure))]
      (let [failure (try
                      (entry/call-with-definition-inputs
                       #(ze/make-gpu-fn (fn [] (swap! attempts inc) (throw primary)))
                       ze/close!)
                      nil
                      (catch clojure.lang.ExceptionInfo error error))]
        (is (= :gpu-compilation-cleanup-failed (:reason (ex-data failure))))
        (is (= 1 @attempts))
        (is (identical? cleanup-failure (ex-cause failure)))
        (is (= [primary] (vec (.getSuppressed cleanup-failure))))))))
