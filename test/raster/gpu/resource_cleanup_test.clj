(ns raster.gpu.resource-cleanup-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.resource-cleanup :as cleanup]))

(defn- error-of [f] (try (f) nil (catch Throwable e e)))

(deftest validates-the-entire-plan-before-release
  (let [calls (atom 0) release #(swap! calls inc)]
    (doseq [plan [nil (list {:id :a :release release})
                  [{:id :a :release release} {:id :a :release release}]
                  [{:id :a :release release :after #{:b}} {:id :b :release release}]
                  [{:id :a :release release :after [:b]}]
                  [{:id :a :release release :failure (ex-info "forged state" {})}]
                  [{:id :a :release release :retry-safe? true}]
                  [{:id :a :release {}}]
                  [{:id :a :release nil}]]]
      (is (= :invalid-cleanup-plan (:reason (ex-data (error-of #(cleanup/owner plan)))))))
      (is (zero? @calls))))

(deftest retains-failures-and-blocked-dependants-without-double-release
  (let [calls (atom []) fail? (atom true)
        fault (ex-info "kernel destruction failed" {:cleanup-retry-safe? true})
        owner (cleanup/owner
               [{:id :kernel :release #(do (swap! calls conj :kernel)
                                           (when @fail? (throw fault)))}
                {:id :independent :release #(swap! calls conj :independent)}
                {:id :view :after #{:kernel} :release #(swap! calls conj :view)}
                {:id :buffer :after #{:view} :release #(swap! calls conj :buffer)}])]
    (is (identical? owner (cleanup/assert-live! owner)))
    (is (identical? fault (error-of #(cleanup/release! owner))))
    (is (= [:kernel :independent] @calls))
    (is (= [:kernel :view :buffer] (cleanup/pending owner)))
    (is (= {:phase :failed
            :remaining [{:id :kernel :disposition :retryable}
                        {:id :view :disposition :pending}
                        {:id :buffer :disposition :pending}]}
           (cleanup/status owner)))
    (is (= :owner-releasing (:reason (ex-data (error-of #(cleanup/assert-live! owner))))))
    (reset! fail? false)
    (is (nil? (cleanup/release! owner)))
    (is (= [:kernel :independent :kernel :view :buffer] @calls))
    (is (= [] (cleanup/pending owner)))
    (is (= {:phase :released :remaining []} (cleanup/status owner)))
    (cleanup/release! owner)
    (is (= [:kernel :independent :kernel :view :buffer] @calls))
    (is (= :owner-releasing (:reason (ex-data (error-of #(cleanup/assert-live! owner))))))))

(deftest preserves-primary-and-independent-suppressed-errors
  (let [a (ex-info "a" {}) b (ex-info "b" {})
        owner (cleanup/owner [{:id :a :release #(throw a)}
                              {:id :b :release #(throw b)}
                              {:id :same-error :release #(throw a)}])]
    (is (identical? a (error-of #(cleanup/release! owner))))
    (is (= [b] (vec (.getSuppressed a))))
    (is (= [:a :b :same-error] (cleanup/pending owner)))))

(deftest recursive-release-is-rejected-and-remains-owned
  (let [owner-ref (atom nil)
        owner (cleanup/owner [{:id :kernel :release #(cleanup/release! @owner-ref)}])]
    (reset! owner-ref owner)
    (is (= :recursive-cleanup (:reason (ex-data (error-of #(cleanup/release! owner))))))
    (is (= [:kernel] (cleanup/pending owner)))))

(deftest indeterminate-failures-are-never-automatically-retried
  (doseq [error [(ex-info "unknown native outcome" {})
                 (ex-info "explicitly unsafe" {:cleanup-retry-safe? false})
                 (ex-info "truthy is not a guarantee" {:cleanup-retry-safe? :yes})
                 (AssertionError. "native outcome unknown")
                 (IllegalStateException. "downcall outcome unknown")]]
    (let [calls (atom [])
          owner (cleanup/owner [{:id :kernel :release #(do (swap! calls conj :kernel)
                                                           (throw error))}
                                {:id :buffer :after #{:kernel} :release #(swap! calls conj :buffer)}
                                {:id :independent :release #(swap! calls conj :independent)}])]
      (is (identical? error (error-of #(cleanup/release! owner))))
      (is (identical? error (error-of #(cleanup/release! owner))))
      (is (= [:kernel :independent] @calls))
      (is (= [:kernel :buffer] (cleanup/pending owner)))
      (is (= {:phase :poisoned
              :remaining [{:id :kernel :disposition :indeterminate}
                          {:id :buffer :disposition :pending}]}
             (cleanup/status owner))))))

(deftest poisoning-does-not-prevent-independent-safe-cleanup
  (let [calls (atom []) fail? (atom true)
        unknown (ex-info "unknown" {})
        retryable (ex-info "retry safe" {:cleanup-retry-safe? true})
        owner (cleanup/owner
               [{:id :unknown :release #(do (swap! calls conj :unknown) (throw unknown))}
                {:id :retryable :release #(do (swap! calls conj :retryable)
                                              (when @fail? (throw retryable)))}
                {:id :blocked :after #{:unknown} :release #(swap! calls conj :blocked)}
                {:id :independent-buffer :after #{:retryable}
                 :release #(swap! calls conj :independent-buffer)}])]
    (is (identical? unknown (error-of #(cleanup/release! owner))))
    (is (= [retryable] (vec (.getSuppressed unknown))))
    (reset! fail? false)
    (is (identical? unknown (error-of #(cleanup/release! owner))))
    (is (= [:unknown :retryable :retryable :independent-buffer] @calls))
    (is (= [:unknown :blocked] (cleanup/pending owner)))
    (is (= :poisoned (:phase (cleanup/status owner))))))

(deftest concurrent-release-calls-do-not-double-destroy
  (let [started (promise) proceed (promise) calls (atom 0)
        owner (cleanup/owner [{:id :kernel :release #(do (swap! calls inc)
                                                        (deliver started true)
                                                        (when (= :timeout (deref proceed 5000 :timeout))
                                                          (throw (ex-info "test timed out" {}))))}])
        first-call (future (cleanup/release! owner))]
    (try
      (is (= true (deref started 5000 :timeout)))
      (is (= :releasing (:phase (cleanup/status owner))))
      (is (= :owner-releasing (:reason (ex-data (error-of #(cleanup/assert-live! owner))))))
      (let [second-call (future (cleanup/release! owner))]
        (deliver proceed true)
        (is (nil? (deref first-call 5000 :timeout)))
        (is (nil? (deref second-call 5000 :timeout))))
      (is (= 1 @calls))
      (is (empty? (cleanup/pending owner)))
      (finally (deliver proceed true)))))
