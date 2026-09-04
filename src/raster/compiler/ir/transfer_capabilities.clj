(ns raster.compiler.ir.transfer-capabilities
  "The physical transfer contract a backend states per device: queryable facts a serving
   scheduler can plan against without inferring overlap from the shape of an API.

   Every fact is declared, none is defaulted: a backend that cannot state one of them does not
   satisfy the contract. `physically-serialized?` is stated separately from the queue fact so a
   backend with a logical transfer queue that maps onto its compute queue says so."
  (:require [raster.compiler.ir.validate :refer [fail!]]))

(def submissions
  "How a transfer is submitted: as device work completing through an event, or as a host copy
   that has completed when the call returns."
  #{:device-event :inline-host-copy})

(def host-stagings
  "Who owns the host-side staging copy of a transfer while it is in flight."
  #{:runtime-owned-native :caller-owned :none})

(def queue-orderings #{:in-order :out-of-order :inline})

(def host-memory-requirements
  "What host memory a transfer source or destination must be."
  #{:any :pinned :shared})

(def event-semantics
  "What awaiting a transfer's completion token establishes."
  #{:device-completion :already-complete})

(def peer-mechanisms #{:direct-copy :staged-through-host})

(def facts
  [:submission :host-staging :independent-physical-queue? :queue-ordering
   :async-h2d? :async-d2h? :peer-transfer? :peer-mechanisms
   :event-semantics :host-lease-until-await? :host-memory :physically-serialized?])

(defn validate!
  "Validate a backend's transfer capability map; returns it unchanged."
  [capabilities]
  (when-not (map? capabilities)
    (fail! "transfer capabilities must be a map" :transfer-capabilities-type
           {:actual (type capabilities)}))
  (let [missing (remove #(contains? capabilities %) facts)]
    (when (seq missing)
      (fail! "transfer capabilities must state every fact"
             :transfer-capabilities-missing {:missing (vec missing)})))
  (let [{:keys [submission host-staging independent-physical-queue? queue-ordering async-h2d?
                async-d2h? peer-transfer? host-lease-until-await? host-memory
                physically-serialized?]
         mechanisms :peer-mechanisms
         semantics :event-semantics} capabilities
        check (fn [fact ok? value]
                (when-not ok?
                  (fail! (str "transfer capability " fact " has an unsupported value")
                         :transfer-capabilities-value {:fact fact :value value})))]
    (check :submission (contains? submissions submission) submission)
    (check :host-staging (contains? host-stagings host-staging) host-staging)
    (check :independent-physical-queue? (boolean? independent-physical-queue?)
           independent-physical-queue?)
    (check :queue-ordering (contains? queue-orderings queue-ordering) queue-ordering)
    (check :async-h2d? (boolean? async-h2d?) async-h2d?)
    (check :async-d2h? (boolean? async-d2h?) async-d2h?)
    (check :peer-transfer? (boolean? peer-transfer?) peer-transfer?)
    (check :peer-mechanisms (and (vector? mechanisms) (every? peer-mechanisms mechanisms))
           mechanisms)
    (check :event-semantics (contains? event-semantics semantics) semantics)
    (check :host-lease-until-await? (boolean? host-lease-until-await?) host-lease-until-await?)
    (check :host-memory (contains? host-memory-requirements host-memory) host-memory)
    (check :physically-serialized? (boolean? physically-serialized?) physically-serialized?)
    ;; consistency: an inline copy, or an in-order queue shared with compute, serializes
    ;; transfers with compute (an out-of-order shared queue may still overlap them); a peer
    ;; route needs at least one mechanism; an inline host copy has no asynchrony, no device
    ;; event and no queue order to report, and a device-event submission has all three
    (when (and (or (= :inline queue-ordering)
                   (and (not independent-physical-queue?) (= :in-order queue-ordering)))
               (not physically-serialized?))
      (fail! "an inline copy or an in-order queue shared with compute is physically serialized"
             :transfer-capabilities-consistency
             {:independent-physical-queue? independent-physical-queue?
              :queue-ordering queue-ordering
              :physically-serialized? physically-serialized?}))
    (when (and (= :inline-host-copy submission)
               (not (and (= :inline queue-ordering) (= :already-complete semantics))))
      (fail! "an inline host copy has inline ordering and is complete when it returns"
             :transfer-capabilities-consistency
             {:submission submission :queue-ordering queue-ordering
              :event-semantics semantics}))
    (when (and (= :device-event submission)
               (not (and (not= :inline queue-ordering) (= :device-completion semantics))))
      (fail! "a device-event submission is queued device work completing through its event"
             :transfer-capabilities-consistency
             {:submission submission :queue-ordering queue-ordering
              :event-semantics semantics}))
    (when (not= peer-transfer? (boolean (seq mechanisms)))
      (fail! "peer transfer support and its mechanisms must agree"
             :transfer-capabilities-consistency
             {:peer-transfer? peer-transfer? :peer-mechanisms mechanisms}))
    (when (and (= :inline-host-copy submission) (or async-h2d? async-d2h?))
      (fail! "an inline host copy is complete on return; it is not asynchronous"
             :transfer-capabilities-consistency
             {:submission submission :async-h2d? async-h2d? :async-d2h? async-d2h?})))
  capabilities)
