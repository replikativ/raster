(ns raster.compiler.equation-artifact
  "Versioned persistent envelope for a lowerable equation-first compilation.

   This namespace owns representation and integrity, not storage policy. A store may persist an
   envelope only when template identity has established persistence eligibility. Loading verifies
   the outer identity and payload bytes before Boring may reconstruct typed compiler records."
  (:require [boring.core :as boring]
            [boring.records :as records]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint])
  (:import [clojure.lang IObj]
           [java.security MessageDigest]))

(def schema-version 1)

(defrecord SerializedSequence [kind values metadata])
(defrecord SerializedScalar [kind value])

;; The macro expands to an immutable literal registry at compile time. Wire input cannot select a
;; Var or constructor outside the Raster compiler record classes present in this build.
(def ^:private compiler-record-registry
  (records/auto-registry "raster.compiler"))

(def ^:private compiler-cbor-options
  {:profile :archival
   :registry compiler-record-registry
   :on-unknown-record :error})

(declare prepare-sequences restore-sequences fail!)

(defn- preserve-metadata [original transformed transform]
  (if (and (instance? IObj transformed) (seq (meta original)))
    (with-meta transformed (transform (meta original)))
    transformed))

(defn- prepare-sequences
  "Make the list/sequential distinction explicit; CBOR arrays intentionally do not carry it."
  [value]
  (let [prepared
        (cond
          (instance? Byte value) (->SerializedScalar :byte (long value))
          (instance? Short value) (->SerializedScalar :short (long value))
          (instance? Integer value) (->SerializedScalar :int (long value))

          (record? value)
          (reduce-kv (fn [record key field] (assoc record key (prepare-sequences field)))
                     value value)

          (map? value)
          (into (empty value) (map (fn [[key field]]
                                     [(prepare-sequences key) (prepare-sequences field)])) value)

          (vector? value) (mapv prepare-sequences value)
          (set? value) (into (empty value) (map prepare-sequences) value)
          (list? value) (->SerializedSequence :list (mapv prepare-sequences value)
                                              (some-> (meta value) prepare-sequences))
          (sequential? value) (->SerializedSequence :sequential (mapv prepare-sequences value)
                                                    (some-> (meta value) prepare-sequences))
          :else value)]
    (if (or (list? value) (sequential? value))
      prepared
      (preserve-metadata value prepared prepare-sequences))))

(defn- serialized-sequence? [value]
  (and value (= "raster.compiler.equation_artifact.SerializedSequence"
                (.getName (class value)))))

(defn- serialized-scalar? [value]
  (and value (= "raster.compiler.equation_artifact.SerializedScalar"
                (.getName (class value)))))

(defn- restore-sequences [value]
  (cond
    (serialized-scalar? value)
    (case (:kind value)
      :byte (byte (:value value))
      :short (short (:value value))
      :int (int (:value value))
      (fail! "artifact contains an unknown serialized scalar kind"
             :equation-artifact-scalar {:kind (:kind value)}))

    (serialized-sequence? value)
    (let [values (map restore-sequences (:values value))
          restored (case (:kind value)
                     :list (apply list values)
                     :sequential (seq (vec values))
                     (fail! "artifact contains an unknown serialized sequence kind"
                            :equation-artifact-sequence {:kind (:kind value)}))]
      (if (:metadata value) (with-meta restored (restore-sequences (:metadata value))) restored))

    :else
    (let [restored
          (cond
            (record? value)
            (reduce-kv (fn [record key field] (assoc record key (restore-sequences field)))
                       value value)

            (map? value)
            (into (empty value) (map (fn [[key field]]
                                       [(restore-sequences key) (restore-sequences field)])) value)

            (vector? value) (mapv restore-sequences value)
            (set? value) (into (empty value) (map restore-sequences) value)
            :else value)]
      (preserve-metadata value restored restore-sequences))))

(def ^:private identity-fields
  #{:semantic-request-fingerprint
    :compiler-build-fingerprint
    :source-dependency-fingerprint
    :target-descriptor-fingerprint})

(defn- fail! [message reason data]
  (throw (ex-info message (assoc data :reason reason :artifact :equation-first))))

(defn- validate-identity! [identity]
  (when-not (and (map? identity)
                 (= identity-fields (set (keys identity)))
                 (every? #(and (string? %) (not-empty %)) (vals identity)))
    (fail! "artifact identity requires four exact non-empty fingerprints"
           :equation-artifact-identity {:identity identity :required identity-fields}))
  identity)

(defn- validate-compilation! [compilation]
  (when-not (equation-first/equation-first-compilation? compilation)
    (fail! "artifact payload must be an EquationFirstCompilation"
           :equation-artifact-payload-type {:actual (type compilation)}))
  (when-not (and (symbol? (:function compilation))
                 (keyword? (:target compilation))
                 (keyword? (:dtype compilation))
                 (symbol? (:source-ns compilation)))
    (fail! "equation-first artifact has an invalid public compilation boundary"
           :equation-artifact-compilation-boundary
           {:function (:function compilation) :target (:target compilation)
            :dtype (:dtype compilation) :source-ns (:source-ns compilation)}))
  (invocation/validate! (get-in compilation [:semantic :attributes :invocation-plan]))
  (emitted-program/validate! (:emitted compilation))
  (doseq [kernel (:kernels compilation)] (kernel-artifact/validate! kernel))
  compilation)

(defn- byte-array? [value]
  (= "[B" (some-> value class .getName)))

(defn- byte-fingerprint [^bytes payload]
  (let [digest (.digest (MessageDigest/getInstance "SHA-256") payload)]
    (str "sha256:" (apply str (map #(format "%02x" (bit-and 0xff %)) digest)))))

(defn seal
  "Seal a compiled equation-first value under an already certified persistent identity."
  [identity compilation]
  (let [identity (validate-identity! identity)
        compilation (validate-compilation! compilation)
        compilation-fingerprint (semantic-fingerprint/fingerprint compilation)
        payload (boring/encode (prepare-sequences compilation) compiler-cbor-options)]
    {:schema-version schema-version
     :transport :boring-cbor-archival
     :kind :equation-first-compilation
     :identity identity
     :compilation-fingerprint compilation-fingerprint
     :payload-fingerprint (byte-fingerprint payload)
     :payload payload}))

(defn validate-envelope!
  "Validate the outer envelope and payload digest without constructing compiler records."
  [envelope]
  (when-not (map? envelope)
    (fail! "equation artifact envelope must be a map"
           :equation-artifact-envelope-type {:actual (type envelope)}))
  (when-not (= schema-version (:schema-version envelope))
    (fail! "equation artifact envelope schema is unsupported"
           :equation-artifact-schema
           {:expected schema-version :actual (:schema-version envelope)}))
  (when-not (= :boring-cbor-archival (:transport envelope))
    (fail! "equation artifact transport is unsupported"
           :equation-artifact-transport {:transport (:transport envelope)}))
  (when-not (= :equation-first-compilation (:kind envelope))
    (fail! "equation artifact kind is unsupported"
           :equation-artifact-kind {:kind (:kind envelope)}))
  (validate-identity! (:identity envelope))
  (when-not (and (string? (:compilation-fingerprint envelope))
                 (not-empty (:compilation-fingerprint envelope)))
    (fail! "equation artifact requires a semantic compilation fingerprint"
           :equation-artifact-compilation-fingerprint
           {:compilation-fingerprint (:compilation-fingerprint envelope)}))
  (when-not (byte-array? (:payload envelope))
    (fail! "equation artifact payload must be a byte array"
           :equation-artifact-payload-bytes {:actual (type (:payload envelope))}))
  (let [actual (byte-fingerprint (:payload envelope))]
    (when-not (= (:payload-fingerprint envelope) actual)
      (fail! "equation artifact payload digest does not match"
             :equation-artifact-integrity
             {:expected (:payload-fingerprint envelope) :actual actual})))
  envelope)

(defn open
  "Authenticate and reconstruct an equation-first compilation.

   `expected-identity` comes from current request/build/source/target certification. It is checked
   before the payload decoder can invoke any constructor in the fixed compiler-record registry."
  [expected-identity envelope]
  (let [expected-identity (validate-identity! expected-identity)
        envelope (validate-envelope! envelope)]
    (when-not (= expected-identity (:identity envelope))
      (fail! "equation artifact identity does not match the current compilation request"
             :equation-artifact-identity-mismatch
             {:expected expected-identity :actual (:identity envelope)}))
    (let [compilation
          (try
            (restore-sequences
             (boring/decode (:payload envelope) compiler-cbor-options))
            (catch Exception error
              (throw (ex-info "equation artifact payload is not valid compiler CBOR"
                              {:reason :equation-artifact-payload-decode
                               :artifact :equation-first}
                              error))))
          compilation (validate-compilation! compilation)
          actual (semantic-fingerprint/fingerprint compilation)]
      (when-not (= (:compilation-fingerprint envelope) actual)
        (fail! "decoded equation compilation differs from its semantic fingerprint"
               :equation-artifact-semantic-integrity
               {:expected (:compilation-fingerprint envelope) :actual actual}))
      compilation)))

(defn encode
  "Encode a validated envelope as archival CBOR. Identity never depends on these transport bytes."
  [envelope]
  (boring/encode (validate-envelope! envelope) {:profile :archival}))

(defn decode
  "Read and validate an outer CBOR envelope without constructing compiler records."
  [bytes]
  (try
    (validate-envelope! (boring/decode bytes {:profile :archival}))
    (catch clojure.lang.ExceptionInfo error (throw error))
    (catch Exception error
      (throw (ex-info "equation artifact transport is not valid CBOR"
                      {:reason :equation-artifact-transport :artifact :equation-first}
                      error)))))
