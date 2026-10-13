(ns raster.compiler.passes.parallel.typed-soac-realized-source-continuation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- source-form [result]
  (list 'let*
        '[prefix (clojure.core/long limit)
          clock-value (clojure.core/long (clojure.core/aget time-buffer 0))
          step (clojure.core/long
                (clojure.core/unchecked-add (clojure.core/long clock-value)
                                            (clojure.core/long 3)))
          effect-a (raster.par/map-void! i n
                     (clojure.core/aset out-a i step))
          effect-b (raster.par/map-void! i n
                     (clojure.core/aset out-b i step))]
        result))

(def options
  {:scalar-types {'n :long 'limit :long}
   :values {'time-buffer (av/tensor {:dtype :long
                                     :shape '[(extent time-buffer)]})}})

(defn- attempt [result]
  (route/attempt (source-form result) :long
                 {'time-buffer :long 'out-a :long 'out-b :long}
                 (assoc options :resident-uniform-input-loads? true)))

(deftest returned-effects-cannot-resurrect-removed-typed-constituents
  (let [result (attempt '[effect-a effect-b])]
    (is (= :typed-soac-unbound-scalar (get-in result [:declined :reason])))
    (is (= :realized-source-continuation
           (get-in result [:declined :details :invariant])))
    (is (= #{'effect-b}
           (set (get-in result [:declined :details :symbols])))
        "the direct returned effect is refused before its nested removed-scalar read can escape")))

(deftest ordinary-host-results-remain-valid-after-constituent-exclusion
  (doseq [result ['out-a
                  '(clojure.core/long limit)
                  '(loop* [j 0]
                     (if (clojure.core/< (clojure.core/long j)
                                         (clojure.core/long limit))
                       (recur (clojure.core/inc (clojure.core/long j)))
                       out-a))]]
    (let [attempted (attempt result)]
      (is (some? (:program attempted)) (pr-str result))
      (is (nil? (:declined attempted)) (pr-str result)))))

(deftest malformed-constituent-source-identity-cannot-own-a-host-binding
  (let [source (source-form 'out-a)
        typed (frontend/form->program
               source (merge options
                             {:dtype :long
                              :array-types {'time-buffer :long 'out-a :long 'out-b :long}}))
        fused (first (fusion/fusion-fixpoint typed))
        realized (first (resident/inline-uniform-input-loads fused))
        facts (dialect/facts realized)
        filtered-facts
        (update-in facts [:attributes :source-bindings]
                   #(vec (remove #{'prefix} %)))
        filtered (dialect/make filtered-facts (dialect/equations realized)
                               (dialect/outputs realized))
        [equation-id constituents]
        (some (fn [[equation-id equation-facts]]
                (when-let [constituents (seq (get-in equation-facts
                                                     [:attributes :fusion/constituents]))]
                  [equation-id (into {} constituents)]))
              (:equations facts))
        [constituent-id constituent-facts] (first constituents)
        repeated-equation-id (some #(when (not= equation-id %) %) (keys (:equations facts)))
        repeated-facts
        (update-in facts [:equations repeated-equation-id :attributes :fusion/constituents]
                   #(assoc (or % {}) constituent-id constituent-facts))
        repeated (dialect/make repeated-facts (dialect/equations realized)
                               (dialect/outputs realized))
        conflicting-facts
        (update-in facts [:equations repeated-equation-id :attributes :fusion/constituents]
                   #(assoc (or % {}) constituent-id
                           (assoc-in constituent-facts [:provenance :conflicting-copy] true)))
        conflicting (dialect/make conflicting-facts (dialect/equations realized)
                                  (dialect/outputs realized))
        forged-id (inc (long constituent-id))
        forged-facts
        (assoc-in facts [:equations equation-id :attributes :fusion/constituents]
                  (-> constituents (dissoc constituent-id)
                      (assoc forged-id constituent-facts)))
        forged (dialect/make forged-facts (dialect/equations realized)
                             (dialect/outputs realized))
        conflicting-failure
        (try (#'route/realize-source source conflicting) nil
             (catch clojure.lang.ExceptionInfo error error))
        failure (try (#'route/realize-source source forged) nil
                     (catch clojure.lang.ExceptionInfo error error))]
    (is (map? (#'route/realize-source source filtered))
        "component filtering preserves binder membership, not global positional indexing")
    (is (map? (#'route/realize-source source repeated))
        "recomputed fusion may repeat one identical source constituent")
    (is (= [:typed-soac-unbound-scalar :realized-constituent-provenance]
           ((juxt :reason :invariant) (ex-data conflicting-failure)))
        "one source identity cannot carry two conflicting constituent certificates")
    (is (some? failure))
    (is (= :typed-soac-unbound-scalar (:reason (ex-data failure))))
    (is (= :realized-constituent-provenance (:invariant (ex-data failure))))))
