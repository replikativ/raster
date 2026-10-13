(ns raster.gpu.cuda-prepared-execution
  "Existing-stream pending use, shared by synchronous copies and prepared launches.
   Cleanup/root owners remain the sole resource retirement authority."
  (:require [raster.gpu.resource-cleanup :as cleanup]))

(defn ledger [root-lock]
  ;; Use the root's monitor so construction and native admission have one lock order.
  {:lock root-lock :pending (atom #{}) :completion-error (atom nil) :debts (atom {})})
(defn binding-state [] {:pending? (atom false) :launch-error (atom nil)})

(defn assert-admitted! [ledger]
  (when-let [failure (some-> ledger :completion-error deref)] (throw failure)))

(defn assert-binding-admitted! [ledger binding]
  (assert-admitted! ledger)
  (when-let [failure @(:launch-error binding)] (throw failure)))

(defn complete! [ledger wait!]
  (locking (:lock ledger)
    (assert-admitted! ledger)
    (try (wait!)
         (catch Throwable failure
           (when-not (true? (:cleanup-retry-safe? (ex-data failure)))
             (compare-and-set! (:completion-error ledger) nil failure))
           (throw failure)))))

(defn- synchronize-under-lock! [ledger stream native!]
  (complete! ledger #(native! :stream-sync [stream]))
  (doseq [binding @(:pending ledger)]
    (reset! (:pending? binding) false)
    (reset! (:launch-error binding) nil))
  (reset! (:pending ledger) #{}))

(defn drain! [ledger binding stream native!]
  (locking (:lock ledger)
    (when @(:pending? binding) (synchronize-under-lock! ledger stream native!))))

(defn submit! [ledger binding stream native! assert-live! operation arguments async?]
  (locking (:lock ledger)
    (assert-live!)
    (assert-binding-admitted! ledger binding)
    ;; Reserve use before enqueue: a throwing driver call does not prove absence of work.
    (reset! (:pending? binding) true)
    (swap! (:pending ledger) conj binding)
    (try (native! operation arguments)
         (catch Throwable failure (reset! (:launch-error binding) failure) (throw failure)))
    (when-not async? (synchronize-under-lock! ledger stream native!)))
  nil)

(defn retain-debt! [ledger owner]
  (cleanup/retain-owned-entry! (:debts ledger) {::cleanup/owner owner}) nil)

(defn synchronize! [ledger stream native!]
  (locking (:lock ledger)
    (synchronize-under-lock! ledger stream native!)
    ;; The root monitor precedes the debt registry monitor, including rollback adoption.
    (cleanup/release-entries! (:debts ledger) (vec @(:debts ledger))
                              #(cleanup/release! (::cleanup/owner %)))))
