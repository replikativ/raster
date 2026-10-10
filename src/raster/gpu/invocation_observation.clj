(ns ^:no-doc raster.gpu.invocation-observation
  "Internal host timing for ordinary synchronous invocation. Not device-event evidence.
   The public wrapper creates its own collector; no caller callback runs under lifetime locks.")

(def ^:dynamic *collector* nil)

(defn record-phase! [phase elapsed]
  (vswap! *collector* update-in [:phases-ns phase] (fnil + 0) elapsed))

(defmacro phase [name & body]
  `(if *collector*
     (let [started# (System/nanoTime)]
       (try ~@body
            (finally (record-phase! ~name (- (System/nanoTime) started#)))))
     (do ~@body)))

(defmacro with-outer-lock [lock & body]
  `(if (and *collector*
            (not (contains? (:phases-ns @*collector*) :outer-lock-acquisition)))
     (let [monitor# ~lock
           started# (System/nanoTime)]
       (locking monitor#
         (record-phase! :outer-lock-acquisition (- (System/nanoTime) started#))
         ~@body))
     (locking ~lock ~@body)))

(defmacro refresh-route [route bytes]
  `(when *collector*
     (vswap! *collector* update-in [:input-refresh ~route]
             (fn [previous#]
               {:operations (inc (get previous# :operations 0))
                :logical-bytes (+ (get previous# :logical-bytes 0) ~bytes)}))))

(defn observe [f]
  (let [collector (volatile! {:phases-ns {} :input-refresh {}})
        started (System/nanoTime)
        outputs (binding [*collector* collector] (f))
        elapsed (- (System/nanoTime) started)
        observations @collector]
    {:outputs outputs
     :report (assoc observations
                    :schema-version 1 :timing-source :host-monotonic
                    :scope :synchronous-compiled-invocation
                    :total-ns elapsed
                    :unpartitioned-ns (- elapsed (reduce + 0 (vals (:phases-ns observations)))))}))
