(ns raster.compiler.backend.gpu.compile-fixture-report
  "Fail-closed aggregate reporting for independent hardware-free compile attempts.
   Reporting never manufactures an artifact or turns a refused workload into a skip.")

(def ^:dynamic ^:private *report* nil)

(defn- abort-cause? [error]
  (let [seen (java.util.IdentityHashMap.)]
    (loop [cause error]
      (cond
        (nil? cause) false
        (.containsKey seen cause) false
        (or (instance? InterruptedException cause)
            (instance? java.util.concurrent.CancellationException cause)
            (instance? Error cause)) true
        :else (do (.put seen cause true) (recur (.getCause ^Throwable cause)))))))

(defn attempt!
  "Run one independent fixture. In a collection, retain a bounded diagnostic and
   return nil on an ordinary compile failure so remaining fixtures can be checked.
   Outside a collection, preserve the original exception. Cancellation and VM
   failures always stop immediately. This is not a production compiler fallback."
  [context compile!]
  (if-not *report*
    (compile!)
    (try
      (let [result (compile!)]
        (swap! *report* update :passed inc)
        result)
      (catch Exception error
        (when (or (abort-cause? error)
                  (.isInterrupted (Thread/currentThread)))
          (throw error))
        (let [message (.getMessage error)
              diagnostic (merge (select-keys context [:function :target :dtype :schedule])
                                {:message (when message (subs message 0 (min 1024 (count message))))
                                 :exception (.getName (class error))}
                                (when (and message (> (count message) 1024))
                                  {:message-truncated? true})
                                (select-keys (ex-data error) [:reason :pass]))]
          (swap! *report* (fn [report]
                           (-> report
                               (update :failures conj diagnostic)
                               (update :first-error #(or % error)))))
          nil)))))

(defn collect!
  "Realize all independent fixture attempts inside one reporting scope.
   Any refused attempt makes the whole gate fail; successful artifact lists are
   returned only when every attempt succeeds. Lazy work remains in the scope."
  [emit!]
  (if *report*
    (vec (emit!))
    (binding [*report* (atom {:passed 0 :failures [] :first-error nil})]
    (let [artifacts (vec (emit!))
          {:keys [passed failures first-error]} @*report*]
      (when (seq failures)
        (throw (ex-info (str (count failures) " public GPU compile fixtures failed")
                        {:reason :gpu-public-fixture-failures
                         :attempted (+ passed (count failures)) :passed passed
                         :failures failures}
                        first-error)))
      artifacts))))
