(ns raster.gpu.storage-representation
  "Actual session storage observations, without compiler/owner evidence or caches."
  (:require [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.gpu.core :as gpu]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang.foreign MemorySegment]))

(defn- run-probe! [session dt artifact]
  (let [key (keyword (str "storage-probe-" (random-uuid)))
        allocated? (volatile! false)
        handle (volatile! nil)
        primary (volatile! nil)
        bytes (byte-array (* 2 (dtype/bytes-of dt)))]
    (try
      (gpu/alloc! session {key [dt 2 nil]})
      (vreset! allocated? true)
      (vreset! handle (gpu/bind-kernel-call! session key artifact [key]))
      (gpu/run-kernel-graph! session @handle)
      (gpu/download-range! session key (MemorySegment/ofArray bytes) {:elements 2})
      (let [observation (mapv #(bit-and 255 %) bytes)]
        {:byte-order (probe/classify-bytes dt observation)
         :observed-bytes observation
         :observed-content (content/content-address-of (MemorySegment/ofArray bytes))})
      (catch Throwable error (vreset! primary error) (throw error))
      (finally
        (let [failure (volatile! @primary)]
          (doseq [release! (cond-> []
                            @handle (conj #(gpu/release-kernel-graph! session @handle))
                            @allocated? (conj #(gpu/free-buffer! session key)))]
            (try (release!)
                 (catch Throwable cleanup
                   (if-let [error @failure]
                     (when-not (identical? error cleanup) (.addSuppressed error cleanup))
                     (vreset! failure cleanup)))))
          (when (and @failure (nil? @primary)) (throw @failure))
          (when (and (nil? @failure)
                     (or (contains? (:buffers @session) key)
                         (contains? (:kernel-graphs @session) key)))
            (throw (ex-info "storage probe cleanup left owned resources live"
                            {:reason :storage-representation-cleanup}))))))))

(defn- require-quiescent! [session]
  (when (seq (:events @session))
    (throw (ex-info "storage observation requires no asynchronous events"
                    {:reason :storage-representation-unready}))))

(defn observe!
  "Measure one dtype using a generated two-element sentinel kernel in the actual session.
   Returns unsealed historical observation data, never a completion or ownership capability.
   The caller owns exclusive session use and lifetime throughout this synchronous operation.

   Optional `with-observation!` is an owner policy wrapper. Capability/artifact admission occurs
   before it is called; it must synchronously invoke its zero-argument observation exactly once
   under its own mutation/lifetime guard. Its return value is returned unchanged. This lets an
   owner invalidate or poison its own output state without moving those rules into the probe.
   The thunk is same-thread, one-shot and invalid once the wrapper returns or throws."
  ([session element-dtype] (observe! session element-dtype (fn [observe] (observe))))
  ([session element-dtype with-observation!]
   (when-not (ifn? with-observation!)
     (throw (ex-info "storage observation requires a callable owner wrapper"
                     {:reason :storage-representation-wrapper})))
   (let [dt (dtype/canon element-dtype)
         device (gpu/execution-device-info session)
         session-id (:session-id @session)]
     (require-quiescent! session)
     (when-not (contains? (:storage-types device) dt)
       (throw (ex-info "selected device does not advertise this storage dtype"
                       {:reason :storage-representation-unsupported :dtype dt})))
     (let [artifact (probe/emit-artifact dt (gpu/kernel-body-c-dialect session))
           thread (Thread/currentThread)
           active? (volatile! true)
           invoked? (atom false)
           observe
         (fn []
           (when-not (and @active? (identical? thread (Thread/currentThread)))
             (throw (ex-info "storage observation thunk is outside its synchronous owner scope"
                             {:reason :storage-representation-scope})))
           (when-not (compare-and-set! invoked? false true)
             (throw (ex-info "storage observation thunk is one-shot"
                             {:reason :storage-representation-replayed})))
           (require-quiescent! session)
           (when-not (and (= device (gpu/execution-device-info session))
                          (= session-id (:session-id @session)))
             (throw (ex-info "session/device changed before storage observation"
                             {:reason :storage-representation-device-changed})))
           (let [observation (run-probe! session dt artifact)
                 after (gpu/execution-device-info session)]
             (require-quiescent! session)
             (when-not (and (= device after) (= session-id (:session-id @session)))
               (throw (ex-info "session/device identity changed during storage measurement"
                               {:reason :storage-representation-device-changed})))
             (assoc observation :dtype dt :device device :session-id session-id
                    :device-fingerprint (fingerprint/fingerprint device)
                    :probe-fingerprint (fingerprint/fingerprint artifact))))]
       (try
         (let [result (with-observation! observe)]
           (when-not @invoked?
             (throw (ex-info "owner wrapper did not invoke its storage observation"
                             {:reason :storage-representation-not-observed})))
           result)
         (finally (vreset! active? false)))))))
