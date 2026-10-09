(ns raster.runtime.artifact-provenance
  "Issuer-private in-process identity witnesses, without a registry or durable authentication.
   These never replace independent compiler validation or runtime ownership/lifetime checks.")

(defn issuer
  "Create a distinct issuer's seal/authentic? operations. Keep the returned issuer private.
   A seal binds :provenance-seal to the exact returned immutable object; copying or associating
   a new object cannot reuse it. Another issuer cannot authenticate this issuer's artifacts.
   Consumers must still check record type and their own semantic/lifetime/owner contracts.
   This is an accidental-substitution witness, not a security boundary against host reflection."
  []
  (let [token (Object.)]
    {:seal
     (fn [artifact]
       (let [owner (volatile! nil)
             sealed (assoc artifact :provenance-seal
                           (fn [candidate] (when (identical? candidate @owner) token)))]
         (vreset! owner sealed)
         sealed))
     :authentic?
     (fn [artifact]
       (let [seal (:provenance-seal artifact)]
         (and (fn? seal) (identical? token (seal artifact)))))}))
