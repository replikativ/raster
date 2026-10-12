(ns raster.ad.normalized-prepared-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-element-test :as element]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.core :as core]
            [raster.numeric :as n]))

(def ^:dynamic receiver nil)
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- build
  ([] (build '(raster.numeric/* element element)))
  ([body]
   (normalized/linearize-map-program (#'element/map-program :double {:body body})
                                    (the-ns 'raster.ad.normalized-prepared-owner-test))))
(defn- arguments [prepared]
  (mapv {'n (long 2) 'x (double-array [2.0 -3.0])} (:inputs prepared)))
(defn- selected []
  ;; Select a stable implementation, not a first-call stub that this fixture's
  ;; own preparation subsequently replaces. No numerical invocation is needed.
  (core/ensure-bytecode-implementation! #'n/_star__m_double_double-impl)
  (:typed-impl (dispatch/selected-call-signature
               'raster.numeric/* '[double double]
               (inference/registered-call-selection 'raster.numeric/* '[double double]))))
(defn- alias-bundle []
  (build (with-meta '(.invk raster.ad.normalized-prepared-owner-test/receiver element element)
           {:raster.op/original 'raster.numeric/* :raster.type/tag 'double})))

(deftest proofs-and-materialization-are-prepare-only
  (let [bundle (build) validations (atom 0) projections (atom 0)
        validate normalized/validate! materialize route/materialize-host-map]
    (with-redefs [normalized/validate! (fn [b] (swap! validations inc) (validate b))
                  route/materialize-host-map (fn [p] (swap! projections inc) (materialize p))]
      (let [prepared (runtime/prepare bundle)
            proof-count @validations projection-count @projections
            args (arguments prepared)
            [y pb] ((:vjp prepared) args)]
        (is (= 3 proof-count) "adapter validates twice; captured callees are proved once more")
        (is (= 3 projection-count) "only three generated projections; source uses checked scalar bindings")
        (is (= [4.0 9.0] (vec y)))
        (dotimes [_ 3]
          (is (= [4.0 -6.0] (vec (nth (pb (double-array [1.0 1.0]))
                                     (.indexOf ^java.util.List (:inputs prepared) 'x)))))
          (is (= [2.0 -3.0] (vec (second ((:jvp prepared) args (double-array [0.5 0.5])))))))
        (is (= proof-count @validations))
        (is (= projection-count @projections))))))

(deftest effective-receiver-alias-and-direct-operation-identities-are-fenced
  (let [implementation (selected)]
    (binding [receiver implementation]
      (let [prepared (runtime/prepare (alias-bundle)) args (arguments prepared)
            [_ pb] ((:vjp prepared) args)]
        (is (= [4.0 9.0] (vec (first ((:vjp prepared) args)))))
        (binding [receiver (Object.)]
          (is (= :ad-normalized-execution-callee (reason #((:vjp prepared) args))))
          (is (= :ad-normalized-execution-callee (reason #(pb (double-array 2)))))
          (is (= :ad-normalized-execution-callee
                 (reason #((:jvp prepared) args (double-array 2))))))
        (is (= [4.0 9.0] (vec (first ((:vjp prepared) args)))))))
    (with-redefs [receiver implementation]
      (let [prepared (runtime/prepare (alias-bundle)) args (arguments prepared)]
        (with-redefs [receiver (Object.)]
          (is (= :ad-normalized-execution-callee (reason #((:vjp prepared) args)))))
        (with-redefs [n/* (fn [& _] 999.0)]
          (is (= :ad-normalized-execution-callee (reason #((:vjp prepared) args))))))))
  (let [prepared (runtime/prepare (build)) args (arguments prepared)]
    (with-redefs [n/* (fn [& _] 999.0)]
      (is (= :ad-normalized-execution-callee (reason #((:vjp prepared) args)))))))

(deftest epoch-fences-cover-compilation-and-execution-publication
  (let [bundle (build) revision (normalized/revisions) current (atom revision)
        compile-var #'runtime/compile-projection compile @compile-var]
    (with-redefs-fn {#'normalized/revisions #(deref current)
                    compile-var (fn [& args]
                                  (let [result (apply compile args)]
                                    (swap! current update :compiler-revision inc)
                                    result))}
      #(is (= :ad-normalized-stale-rules (reason (fn [] (runtime/prepare bundle)))))))
  (let [prepared (runtime/prepare (build)) args (arguments prepared)
        [_ pb] ((:vjp prepared) args)
        revision (normalized/revisions) current (atom revision)
        invoke-var #'runtime/invoke invoke @invoke-var]
    (doseq [operation [#((:vjp prepared) args)
                       #((:jvp prepared) args (double-array 2))
                       #(pb (double-array 2))]]
      (reset! current revision)
      (with-redefs-fn {#'normalized/revisions #(deref current)
                      invoke-var (fn [& args]
                                   (let [result (apply invoke args)]
                                     (swap! current update :template-revision inc)
                                     result))}
        #(is (= :ad-normalized-stale-rules (reason operation)))))))

(deftest execution-failure-keeps-original-identity
  (let [prepared (runtime/prepare (build)) args (arguments prepared)
        failure (ex-info "original execution failure" {:reason :fixture})]
    (with-redefs-fn {#'runtime/invoke (fn [& _] (throw failure))}
      #(is (identical? failure (try ((:vjp prepared) args) nil (catch Throwable e e)))))))

(deftest guard-reuses-scope-authority-and-does-not-inspect-quoted-data
  (let [guard (normalized/execution-guard
               *ns* (normalized/revisions)
               [{:parameters '[receiver]
                 :body '(let* [local (quote (raster.numeric/* a b))]
                          (receiver local))}])]
    (with-redefs [n/* (fn [& _] 9.0) receiver (Object.)]
      (is (true? (guard)) "local receiver and quoted operation have no global execution dependency"))))

(deftest scope-macros-and-calls-in-result-collections-are-dependencies
  (doseq [body ['(dotimes [i 0] i) '[(raster.numeric/* 2.0 3.0)]]]
    (let [guard (normalized/execution-guard *ns* (normalized/revisions)
                                          [{:parameters [] :body body}])
          macro-var #'clojure.core/dotimes
          macro-metadata (meta macro-var)]
      ;; Var.bindRoot may clear :macro even when with-redefs restores the old
      ;; callable. Restore the exact original metadata, not an invented flag.
      (try
        (with-redefs [clojure.core/dotimes (fn [& _] nil) n/* (fn [& _] 99.0)]
          (is (= :ad-normalized-execution-callee (reason guard))))
        (finally (reset-meta! macro-var macro-metadata))))))
