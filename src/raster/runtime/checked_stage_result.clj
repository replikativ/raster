(ns raster.runtime.checked-stage-result
  "Interpret a completed checked stage's private status protocol.

   This function neither submits work nor proves completion, storage ownership or
   initialization. The enclosing execution owner must hold the exact bound graph
   and its temporary lifetimes through synchronous completion and observation.
   Ordinary binding/publication gates remain closed until that owner is wired."
  (:require [raster.compiler.ir.emitted-parallel-equation :as emitted]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.passes.parallel.checked-map-stage :as checked]))

(def ^:private status-chunk-size 16384)

(defn- fail! [message data]
  (throw (ex-info message (assoc data :reason :checked-stage-observation-protocol))))

(defn observe!
  "Check a stage after its owner's synchronous completion boundary.

   Independently reconstruct the emitted equation; private buffer identities
   come only from that report. read-range! receives {:buffer :dtype :offset
   :elements} and must synchronously copy that exact owned temporary range to a
   fresh primitive array. Status reads are bounded; only the first failing lane's
   diagnostic is read. Empty extents perform no reads. Success returns nil, not
   an initialization certificate or an execution permission. Failure preserves
   the checked primitive conversion's actual exception.

   This is a protocol interpreter, not an API for accessing graph temporaries."
  ([equation scalar-values read-range!]
   (observe! equation scalar-values read-range! {}))
  ([equation scalar-values read-range! caller-options]
   (when-not (ifn? read-range!)
     (fail! "checked observation requires an owner-scoped range reader" {}))
   (let [report (emitted/validate-with-result-contracts equation caller-options)
         obligations (:stage-results report)]
     (when-not (and (= 1 (count obligations))
                    (= checked/kind (:kind (first obligations))))
       (fail! "checked observation requires one reconstructed checked map" {}))
     (let [{:keys [extent status diagnostics]} (first obligations)
           length (graph-call/resolve-integer scalar-values extent)]
       (when-not (<= 0 length Long/MAX_VALUE)
         (fail! "checked stage extent must fit nonnegative signed storage"
                {:extent length}))
       (loop [offset (long 0)]
         (when (< offset length)
           (let [elements (int (min status-chunk-size (- length offset)))
                 snapshot (read-range! {:buffer status :dtype :int
                                        :offset offset :elements elements})]
             (when-not (and (instance? int/1 snapshot)
                            (= elements (alength ^ints snapshot)))
               (fail! "status snapshot differs from the exact requested primitive range"
                      {:offset offset :elements elements}))
             (dotimes [index elements]
               (case (aget ^ints snapshot index)
                 0 nil
                 1 (let [lane (+ offset index)
                         diagnostic (read-range! {:buffer diagnostics :dtype :double
                                                  :offset lane :elements 1})]
                     (when-not (and (instance? double/1 diagnostic)
                                    (= 1 (alength ^doubles diagnostic)))
                       (fail! "failed lane lost its original Double diagnostic" {:lane lane}))
                     ;; Re-run only the immutable checked conversion, not the
                     ;; operand computation or any source effects. A valid
                     ;; failure diagnostic throws directly with source semantics.
                     (clojure.lang.RT/floatCast (aget ^doubles diagnostic 0))
                     (fail! "failed lane diagnostic does not fail the source conversion"
                            {:lane lane}))
                 (fail! "checked stage wrote an unknown status"
                        {:lane (+ offset index) :status (aget ^ints snapshot index)})))
             (recur (+ offset elements)))))))))
