(ns raster.ad.normalized-lazy-callee-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.ad.normalized-scalar :as normalized]))

(defn- spec [source-ns]
  {:mangled-sym 'lazy :params '[x] :tags '[double] :return-tag 'double
   :walked-body '[x] :source-ns-name (ns-name source-ns) :typed-iface-name nil
   :annotations nil :source-body nil})

(deftest lazy-realization-is-single-compile-without-body-execution
  (let [source-ns (create-ns (gensym "normalized_lazy_"))
        calls (atom 0) body-calls (atom 0)
        old (fn [& _] (swap! body-calls inc))
        compiled (fn [& _] (swap! body-calls inc))
        v (intern source-ns 'lazy old)
        pending (spec source-ns)]
    (try
      (alter-meta! v assoc :raster.core/jit-spec pending)
      (with-redefs [core/do-bytecode-upgrade!
                    (fn [& args]
                      (is (= [v 'lazy '[x] '[double] 'double '[x] source-ns nil
                              :annotations nil :source-body nil] (vec args)))
                      (swap! calls inc)
                      (alter-var-root v (constantly compiled)))]
        (core/ensure-bytecode-implementation! v)
        (core/ensure-bytecode-implementation! v))
      (is (= 1 @calls))
      (is (zero? @body-calls))
      (is (identical? compiled @v))
      (is (not (contains? (meta v) :raster.core/jit-spec)))
      (finally (remove-ns (ns-name source-ns))))))

(deftest failed-realization-keeps-original-failure-and-pending-spec
  (let [source-ns (create-ns (gensym "normalized_lazy_failure_"))
        v (intern source-ns 'lazy identity) pending (spec source-ns)
        failure (ex-info "required compilation failed" {})]
    (try
      (alter-meta! v assoc :raster.core/jit-spec pending)
      (with-redefs [core/do-bytecode-upgrade! (fn [& _] (throw failure))]
        (is (identical? failure (try (core/ensure-bytecode-implementation! v)
                                    nil (catch Throwable e e)))))
      (is (identical? identity @v))
      (is (= pending (:raster.core/jit-spec (meta v))))
      (finally (remove-ns (ns-name source-ns))))))

(deftest execution-capture-realizes-before-identity-but-never-recaptures
  (let [source-ns (create-ns (gensym "normalized_lazy_guard_"))
        old (fn [x] x) compiled (fn [x] x) replacement (fn [x] x)
        v (intern source-ns 'lazy old) calls (atom 0)]
    (try
      (alter-meta! v assoc :raster.core/jit-spec (spec source-ns))
      (let [guard! (with-redefs [core/do-bytecode-upgrade!
                                (fn [& _] (swap! calls inc)
                                  (alter-var-root v (constantly compiled)))]
                     (normalized/execution-guard source-ns (normalized/revisions)
                       [{:parameters '[x] :body '(lazy x)}]))]
        (is (= 1 @calls))
        (is (true? (guard!)))
        (alter-var-root v (constantly replacement))
        (is (= :ad-normalized-execution-callee
               (try (guard!) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (= 1 @calls)))
      (finally (remove-ns (ns-name source-ns))))))
