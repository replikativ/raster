(ns raster.compiler.required-jit-walk-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.math]))

(defn- with-source-var [f]
  (let [ns' (create-ns (gensym "raster.test.required_walk_"))
        v (intern ns' 'kernel :old-implementation)]
    (binding [*ns* ns'] (refer 'clojure.core))
    (alter-meta! v assoc
                 :raster.core/deftm true
                 :raster.core/deftm-params '[x]
                 :raster.core/deftm-tags '[double]
                 :raster.core/deftm-annotations '[Double]
                 :raster.core/deftm-source-body '[(+ x 1.0)]
                 :raster.core/deftm-source-ns (ns-name ns'))
    (try (f v ns') (finally (remove-ns (ns-name ns'))))))

(defn- thrown [f]
  (try (f) nil (catch Throwable e e)))

(deftest required-walk-failures-never-compile-stale-body
  (doseq [silent? [false true]
          failure [(ex-info "fatal" {:reason :raster/fatal})
                   (ex-info "bug" {:reason :raster/bug})
                   (NullPointerException. "walker defect")
                   (AssertionError. "walker invariant")]]
    (with-source-var
      (fn [v ns']
        (let [walks (atom 0) compiles (atom 0)]
          (with-redefs-fn
            {#'core/jit-walk-with-tc (fn [& _] (swap! walks inc) (throw failure))
             (ns-resolve 'raster.core 'compile-typed-impl-fn)
             (delay (fn [& _] (swap! compiles inc) :new-implementation))}
            (fn []
              (binding [core/*jit-silent-fallback?* silent?]
                (is (identical? failure
                                (thrown #(core/do-bytecode-upgrade!
                                          v 'kernel '[x] '[double] 'double '[x] ns' nil
                                          :annotations '[Double] :source-body '[(+ x 1.0)]))))
                (is (identical? failure (thrown #(core/ensure-walked-body! v))))
                (is (= 2 @walks) "one attempt per required boundary, no structural retry")
                (is (zero? @compiles))
                (is (= :old-implementation @v))
                (is (nil? (:raster.core/deftm-walked-body (meta v))))))))))))

(deftest optional-bytecode-fallback-is-not-required-walk-fallback
  (with-source-var
    (fn [v ns']
      (with-redefs-fn
        {#'core/jit-walk-with-tc (fn [& _] '[x])
         (ns-resolve 'raster.core 'compile-typed-impl-fn)
         (delay (fn [& _] (throw (ex-info "unsupported bytecode" {:reason :unsupported-test-body}))))}
        (fn []
          (binding [core/*jit-silent-fallback?* true]
            (is (nil? (core/do-bytecode-upgrade! v 'kernel '[x] '[double] 'double '[x] ns' nil
                                               :annotations '[Double] :source-body '[x])))
            (is (= :old-implementation @v))))))))

(deftest bytecode-invariants-are-not-optional-fallback
  (doseq [failure [(ex-info "fatal" {:reason :raster/fatal})
                   (ex-info "bug" {:reason :raster/bug})
                   (AssertionError. "bytecode invariant")]]
    (with-source-var
      (fn [v ns']
        (with-redefs-fn
          {#'core/jit-walk-with-tc (fn [& _] '[x])
           (ns-resolve 'raster.core 'compile-typed-impl-fn) (delay (fn [& _] (throw failure)))}
          (fn []
            (binding [core/*jit-silent-fallback?* true]
              (is (identical? failure
                              (thrown #(core/do-bytecode-upgrade!
                                        v 'kernel '[x] '[double] 'double '[x] ns' nil))))
              (is (= :old-implementation @v)))))))))

(deftest defining-namespace-is-required-and-success-is-published
  (with-source-var
    (fn [v ns']
      (let [seen (atom [])]
        (with-redefs [core/jit-walk-with-tc (fn [_ _ _ _ defining-ns]
                                             (swap! seen conj defining-ns) '[x])]
          (is (= '[x] (core/ensure-walked-body! v)))
          (is (= [ns'] @seen))
          (is (= '[x] (:raster.core/deftm-walked-body (meta v))))
          (is (= '[x] (core/ensure-walked-body! v)))
          (is (= [ns'] @seen) "successful retained body avoids a second walk"))
        (alter-meta! v dissoc :raster.core/deftm-walked-body)
        (alter-meta! v assoc :raster.core/deftm-source-ns 'raster.missing.source.namespace)
        (is (= :missing-deftm-source-namespace
               (:reason (ex-data (thrown #(core/ensure-walked-body! v))))))
        (is (nil? (:raster.core/deftm-walked-body (meta v))))))))

(deftest fresh-walk-retains-compilation-loader
  (doseq [caller-loader [nil (clojure.lang.DynamicClassLoader.
                              (.getContextClassLoader (Thread/currentThread)))]]
  (with-source-var
    (fn [v ns']
      (let [seen (atom [])]
        (with-redefs-fn
          {#'core/jit-walk-with-tc (fn [& _]
                                   (swap! seen conj method-entry/*compilation-classloader*) '[x])
           (ns-resolve 'raster.core 'compile-typed-impl-fn)
           (delay (fn [& _]
                    (swap! seen conj method-entry/*compilation-classloader*)
                    (throw (ex-info "optional bytecode refusal" {}))))}
          (fn []
            (binding [core/*jit-silent-fallback?* true
                      method-entry/*compilation-classloader* caller-loader]
              (core/do-bytecode-upgrade! v 'kernel '[x] '[double] 'double '[x] ns' nil
                                        :annotations '[Double] :source-body '[x]))
            (is (= 2 (count @seen)))
            (is (some? (first @seen)))
            (is (identical? (first @seen) (second @seen)))
            (when caller-loader
              (is (identical? caller-loader (first @seen)))))))))))

(deftest actual-walk-uses-defining-alias-not-ambient-alias
  (with-source-var
    (fn [v ns']
      (let [ambient (create-ns (gensym "raster.test.ambient_walk_"))]
        (try
          (binding [*ns* ns'] (alias 'ops 'raster.math))
          (binding [*ns* ambient] (alias 'ops 'clojure.core))
          (alter-meta! v assoc :raster.core/deftm-source-body '[(ops/sin x)])
          (let [walked (binding [*ns* ambient] (core/ensure-walked-body! v))
                f (binding [*ns* ns'] (eval (list 'clojure.core/fn '[x] (first walked))))]
            (is (= (Math/sin 0.7) (f 0.7)))
            (is (= (vec walked) (:raster.core/deftm-walked-body (meta v)))))
          (finally (remove-ns (ns-name ambient))))))))
