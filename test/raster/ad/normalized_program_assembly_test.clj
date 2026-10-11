(ns raster.ad.normalized-program-assembly-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest generated-functional-map-is-the-same-typed-program
  (doseq [dt [:float :double] extent ['n 0]]
    (let [attributes {:index 'i :extent extent}
          region (soac/lambda-form '[element scale] '[(raster.numeric/* element scale)])
          values {'x (av/tensor {:dtype dt :shape [extent]})
                  'scale (av/tensor {:dtype dt :shape []})
                  'y (av/tensor {:dtype dt :shape [extent] :ownership :owned})
                  'n (av/tensor {:dtype :long :shape []})}
          inputs (if (= extent 'n) '[x scale n] '[x scale])
          equation (list '= 'generated '[y] (list 'map attributes '[x] '[scale] region))
          expected (soac/make
                    {:values (select-keys values (concat inputs '[y])) :inputs inputs
                     :equations {'generated (soac/default-equation-facts)}
                     :effects #{} :diagnostics [] :provenance {}
                     :attributes {:source-arithmetic numerics/retained-source-arithmetic}}
                    [equation] '[y])
          actual (#'normalized/generated-map-program values 'generated attributes '[x] '[scale] region '[y] [])]
      (is (fingerprint/equivalent? expected actual))
      (is (= inputs (:inputs (soac/facts actual))))
      (is (= inputs (vec (concat (soac/operation-inputs equation)
                                 (filter soac/value-id? (soac/operation-extents equation))))))
      (is (identical? region (:lambda (soac/operation-parts (first (soac/equations actual)))))))))

(deftest indexed-capacity-is-retained-without-becoming-an-executable-input
  (doseq [dt [:float :double] it [:int :long]]
    (let [attributes {:index 'i :extent 'n :attributes {:stable-array-captures '[data]}}
          region (soac/lambda-form '[index data-array] '[(clojure.core/aget data-array index)])
          values {'indices (av/tensor {:dtype it :shape '[n]})
                  'data (av/tensor {:dtype dt :shape '[m]})
                  'out (av/tensor {:dtype dt :shape '[n] :ownership :owned})
                  'n (av/tensor {:dtype :long :shape []})
                  'm (av/tensor {:dtype :long :shape []})}
          equation (list '= 'gather '[out] (list 'map attributes '[indices] '[data] region))
          expected (soac/make
                    (soac/default-program-facts
                     {:values values :inputs '[indices data n]
                      :equations {'gather (soac/default-equation-facts)}
                      :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
                    [equation] '[out])
          actual (#'normalized/generated-map-program values 'gather attributes '[indices] '[data] region '[out] '[n m])]
      (is (fingerprint/equivalent? expected actual))
      (is (= '[indices data n] (:inputs (soac/facts actual))))
      (is (= (get values 'm) (get-in (soac/facts actual) [:values 'm])))
      (is (not (some #{'m} (soac/operation-inputs (first (soac/equations actual)))))))))

(deftest generated-map-never-invents-missing-facts-or-dtypes
  (let [attributes {:index 'i :extent 'n}
        region (soac/lambda-form '[element] '[element])
        values {'x (av/tensor {:dtype :double :shape '[n]})
                'y (av/tensor {:dtype :double :shape '[n] :ownership :owned})
                'n (av/tensor {:dtype :long :shape []})}
        build (fn [v extra]
                (#'normalized/generated-map-program v 'map attributes '[x] [] region '[y] extra))]
    (is (= :ad-normalized-generated-map-facts (reason #(build values '[missing-capacity]))))
    (is (= :typed-soac-value (reason #(build (assoc-in values ['x :dtype] nil) []))))
    ;; Shared dialect owns the missing executable operand/type admission; no defaults are added.
    (is (= :typed-soac-unknown-value (reason #(build (dissoc values 'x) []))))))
