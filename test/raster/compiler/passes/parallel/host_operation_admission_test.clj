(ns raster.compiler.passes.parallel.host-operation-admission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-element-test :as maps]
            [raster.ad.normalized-map-sum-test :as sums]
            [raster.ad.normalized-indexed-map-test :as indexed]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- changed [program f]
  (soac/make (f (soac/facts program)) (soac/equations program) (soac/outputs program)))

(deftest admission-preserves-existing-logical-and-physical-projections
  (doseq [dt [:float :double]]
    (let [map-source (#'maps/map-program dt)
          sum-source (#'sums/source dt)
          indexed-bundles (for [it [:int :long]]
                            (normalized/linearize-indexed-map-program
                             (#'indexed/indexed-map dt it 'n 'm) *ns*))
          programs (concat [[map-source route/materialize-host-map]
                            [sum-source route/materialize-host-map-reduce]]
                           (for [bundle indexed-bundles
                                 phase [:primal :linear :transpose]]
                             [(get-in bundle [:gather-programs phase]) route/materialize-host-indexed-map]))]
      ;; Source generation stays in its existing owner. Compare the exact checked roles
      ;; supplied to it, avoiding string/gensym equality or a second execution engine.
      (with-redefs-fn {#'route/host-projection
                      (fn [program physical outputs]
                        {:program program :physical-inputs physical :output-values outputs})}
        (fn []
          (doseq [[program specialized] programs]
            (is (= (specialized program) (route/materialize-host-program program)))
            (is (identical? program (:program (route/materialize-host-program program))))))))))

(deftest specialization-boundaries-and-topology-remain-closed
  (let [pure (#'maps/map-program :double)
        gather (#'indexed/indexed-map :double :long 'n 'm)
        reduction (second (soac/equation-subprograms (#'sums/source :double)))]
    (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map gather))))
    (is (= :typed-soac-host-program-topology (reason #(route/materialize-host-program reduction))))
    (doseq [program [(changed pure #(assoc-in % [:attributes :unknown-host-contract] true))
                     (changed pure #(assoc-in % [:values 'x :memory-space] :device))]]
      (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-program program))))
      (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map program)))))
    (let [bad-gather (changed gather #(assoc-in % [:values 'out :ownership] :borrowed))]
      (is (= :typed-soac-host-indexed-boundary (reason #(route/materialize-host-program bad-gather))))
      (is (= :typed-soac-host-indexed-boundary (reason #(route/materialize-host-indexed-map bad-gather)))))))

(deftest generic-and-specialized-admission-validate-the-program-once
  (let [program (#'maps/map-program :double)
        validations (atom 0) validate soac/validate!]
    (with-redefs-fn {#'soac/validate! (fn [p] (swap! validations inc) (validate p))
                    #'route/host-projection (fn [_ physical outputs] [physical outputs])}
      (fn []
        (route/materialize-host-program program)
        (is (= 1 @validations))
        (route/materialize-host-map program)
        (is (= 2 @validations))))))
