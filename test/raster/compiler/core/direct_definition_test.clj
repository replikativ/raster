(ns raster.compiler.core.direct-definition-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.normal-return :as normal]))

(defn- with-definition [check]
  (let [scope (create-ns (gensym "raster.direct-definition."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm fresh [input :- (Array float)] :- (Array float)
                 (clojure.core/float-array (clojure.core/alength input))))
        (let [operation (symbol (str (ns-name scope)) "fresh")
              generic (ns-resolve scope 'fresh)
              signature (inference/registered-call-signature operation '[floats])
              implementation (:mangled-sym signature)
              query #(entry/direct-selected-definition operation implementation 1)]
          (check {:scope scope :operation operation :generic generic
                  :implementation implementation :query query})))
      (finally (remove-ns (ns-name scope))))))

(deftest plain-array-methods-have-correspondence-without-a-typed-receiver
  (with-definition
    (fn [{:keys [implementation query]}]
      (let [definition (query)]
        (is (map? definition))
        (is (nil? (get-in definition [:entry :typed-impl])))
        (is (= '[floats] (:tags definition)))
        (is (= {:presence :present :array-tag 'floats}
               (binding [entry/*emission-inputs* (atom {:static [] :values []})]
                 (normal/result-fact (list implementation 'input)))))))))

(deftest changed-generic-or-mangled-target-cannot-reuse-registration
  (with-definition
    (fn [{:keys [generic query]}]
      (let [definition (query) source (:source-var definition)]
        (with-redefs-fn {generic (fn [& _] nil)} #(is (nil? (query))))
        (with-redefs-fn {source (fn [& _] nil)} #(is (nil? (query))))
        (is (map? (query)))
        (.setDynamic ^clojure.lang.Var generic true)
        (try
          (with-bindings {generic (fn [& _] nil)} (is (nil? (query))))
          (finally (.setDynamic ^clojure.lang.Var generic false)))
        (let [metadata (meta generic)]
          (try
            (alter-meta! generic dissoc :raster.core/dispatch-owner)
            (is (nil? (query)) "legacy metadata does not authorize a new root snapshot")
            (finally (reset-meta! generic metadata))))))))

(deftest every-dispatch-rebuild-publishes-its-new-owner
  (with-definition
    (fn [{:keys [generic query]}]
      (let [original @generic]
        (dispatch/set-reducible! generic)
        (is (not (identical? original @generic)))
        (is (map? (query))))
      (core/prefer-method! generic '[floats] '[Object])
      (is (map? (query)))
      (core/remove-method! generic '[floats])
      (is (nil? (query))))))

(deftest copied-namespace-metadata-cannot-publish-a-new-var
  (with-definition
    (fn [{:keys [scope generic query]}]
      (let [metadata (meta generic) root @generic]
        (ns-unmap scope 'fresh)
        (let [replacement (intern scope 'fresh root)]
          (reset-meta! replacement metadata)
          (is (not (identical? replacement generic)))
          (is (nil? (query)) "the original publisher Var is part of correspondence"))))))
