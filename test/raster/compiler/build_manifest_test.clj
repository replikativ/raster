(ns raster.compiler.build-manifest-test
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as manifest]))

(defn- valid-manifest []
  {:schema-version 2
   :library 'org.replikativ/raster
   :version "0.2.685"
   :revision "0123456789abcdef"
   :runtime {:java-version "25.0.1" :clojure-version "1.12.0"}
   :source-namespaces '[raster.compiler.pipeline raster.core]
   :dependencies {'org.clojure/clojure {:mvn/version "1.12.0"}
                  'org.replikativ/pattern {:git/sha "abcdef"}}})

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest complete-build-manifests-have-stable-identities
  (let [first (manifest/manifest-identity (valid-manifest))
        reordered (manifest/manifest-identity
                   (-> (valid-manifest)
                       (assoc :dependencies
                              (array-map 'org.replikativ/pattern {:git/sha "abcdef"}
                                         'org.clojure/clojure {:mvn/version "1.12.0"}))))]
    (is (:complete? first))
    (is (empty? (:blockers first)))
    (is (= (:fingerprint first) (:fingerprint reordered)))
    (is (string? (:fingerprint first)))))

(deftest unversioned-dependencies-fail-closed-for-persistence
  (let [identity (manifest/manifest-identity
                  (assoc-in (valid-manifest) [:dependencies 'local/library]
                            {:unversioned true}))]
    (is (false? (:complete? identity)))
    (is (= #{'local/library} (:unversioned-dependencies identity)))
    (is (= #{:unversioned-build-dependencies} (:blockers identity)))))

(deftest malformed-manifests-fail-loud
  (is (= :compiler-build-manifest-schema
         (reason-of #(manifest/manifest-identity
                      (assoc (valid-manifest) :schema-version 3)))))
  (is (= :compiler-build-manifest-dependencies
         (reason-of #(manifest/manifest-identity
                      (assoc-in (valid-manifest) [:dependencies 'bad/library] {}))))))

(deftest source-checkouts-do-not-invent-a-build-identity
  (with-redefs [io/resource (constantly nil)]
    (is (nil? (manifest/load-resource)))))
