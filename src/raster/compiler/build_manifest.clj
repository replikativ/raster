(ns raster.compiler.build-manifest
  "Validated identity of the compiler implementation contained in a packaged Raster artifact.

   The build writes `raster/compiler-build.edn`. Source checkouts intentionally have no manifest
   and therefore cannot claim cross-process cache eligibility. This identity covers Raster's Git
   revision, dependency coordinates and build JVM; it does not replace a program's resolved source
   dependency manifest."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]))

(def schema-version 1)
(def resource-name "raster/compiler-build.edn")

(defn- fail! [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

(defn- non-blank-string? [value]
  (and (string? value) (not (str/blank? value))))

(defn- dependency-coordinate? [coordinate]
  (and (map? coordinate)
       (= 1 (count coordinate))
       (or (non-blank-string? (:mvn/version coordinate))
           (non-blank-string? (:git/sha coordinate))
           (true? (:unversioned coordinate)))))

(defn validate!
  "Validate and return a compiler-build manifest.

   `:unversioned` dependency entries are representable so development builds remain inspectable,
   but make the manifest incomplete and ineligible for persistent compiler artifacts."
  [manifest]
  (when-not (map? manifest)
    (fail! "compiler build manifest must be a map"
           :compiler-build-manifest-type {:actual (type manifest)}))
  (when-not (= schema-version (:schema-version manifest))
    (fail! "compiler build manifest schema is unsupported"
           :compiler-build-manifest-schema
           {:expected schema-version :actual (:schema-version manifest)}))
  (doseq [field [:library :version :revision]
          :let [value (get manifest field)]]
    (when-not (if (= field :library) (symbol? value) (non-blank-string? value))
      (fail! "compiler build manifest has an invalid identity field"
             :compiler-build-manifest-identity {:field field :value value})))
  (when-not (and (map? (:runtime manifest))
                 (non-blank-string? (get-in manifest [:runtime :java-version]))
                 (non-blank-string? (get-in manifest [:runtime :clojure-version])))
    (fail! "compiler build manifest requires explicit Java and Clojure versions"
           :compiler-build-manifest-runtime {:runtime (:runtime manifest)}))
  (when-not (and (map? (:dependencies manifest))
                 (every? symbol? (keys (:dependencies manifest)))
                 (every? dependency-coordinate? (vals (:dependencies manifest))))
    (fail! "compiler build manifest dependencies require one stable coordinate or :unversioned"
           :compiler-build-manifest-dependencies
           {:dependencies (:dependencies manifest)}))
  manifest)

(defn manifest-identity
  "Return the validated manifest, its canonical fingerprint, and conservative completeness facts."
  [manifest]
  (let [manifest (validate! manifest)
        unversioned (into #{} (keep (fn [[library coordinate]]
                                      (when (:unversioned coordinate) library)))
                          (:dependencies manifest))]
    {:manifest manifest
     :fingerprint (semantic-fingerprint/fingerprint manifest)
     :complete? (empty? unversioned)
     :blockers (cond-> #{} (seq unversioned) (conj :unversioned-build-dependencies))
     :unversioned-dependencies unversioned}))

(defn load-resource
  "Load and validate the build manifest visible to the current classloader, or return nil when
   running directly from a source checkout. A present malformed resource fails loud."
  []
  (when-let [resource (io/resource resource-name)]
    (with-open [reader (java.io.PushbackReader. (io/reader resource))]
      (manifest-identity (edn/read {:eof nil} reader)))))

(defonce ^:private current-build (delay (load-resource)))

(defn current-identity
  "Return this packaged compiler's build identity, or nil in an unmanifested source checkout."
  []
  @current-build)
