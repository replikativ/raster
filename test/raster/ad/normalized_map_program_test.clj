(ns raster.ad.normalized-map-program-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- source [dt options]
  (#'element/map-program dt options))

(defn- build [source]
  (normalized/linearize-map-program source (the-ns 'raster.ad.normalized-map-program-test)))

(defn- evaluate-lane [bundle phase operands]
  (let [program (get-in bundle [:map-programs phase])
        equation (first (soac/equations program))
        {:keys [lambda]} (soac/operation-parts equation)
        dtypes (mapv #(get-in (soac/facts program) [:values % :dtype]) (nth equation 2))
        dt (first dtypes)]
    ;; Existing target-neutral JVM scalar backend is the oracle, not a new AD interpreter.
    (mapv :value (scalar/evaluate-region (:source-ns bundle) lambda
                                         (mapv #(hash-map :type dt :value %) operands) dtypes))))

(defn- reason [bundle]
  (try
    ;; Even a freshly minted producer certificate cannot justify inconsistent relations.
    (normalized/validate! (assoc bundle :certificate (#'normalized/certificate bundle)))
    nil
    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- alter-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

(defn- alter-lambda [program f]
  (let [equation (first (soac/equations program))
        operation (nth equation 3)
        operation (apply list (assoc (vec operation) 4 (f (nth operation 4))))
        equation (list '= (second equation) (nth equation 2) operation)]
    (soac/make (soac/facts program) [equation] (soac/outputs program))))

(deftest lifted-square-maps-have-fresh-owned-residuals-and-explicit-seeds
  (doseq [dt [:double :float]]
    (let [bundle (build (source dt {}))
          {:keys [source primal linear transpose interface]} (:map-programs bundle)
          cast (if (= dt :float) float double)
          residual (get-in interface [:residual-arrays 'element])]
      (is (identical? bundle (normalized/validate! bundle)))
      (is (not (contains? (:values (soac/facts source)) residual)))
      (is (= :owned (get-in (soac/facts primal) [:values residual :ownership])))
      (is (= '[n] (get-in (soac/facts primal) [:values residual :shape])))
      (is (= ['y residual] (soac/outputs primal)))
      (is (= [(first (:tangent-inputs interface)) residual]
             (:arrays (soac/operation-parts (first (soac/equations linear))))))
      (is (= [(get interface :cotangent-input) residual]
             (:arrays (soac/operation-parts (first (soac/equations transpose))))))
      (doseq [[x dx dy] [[3.0 -0.5 0.25] [-2.0 1.5 -3.0] [0.0 4.0 2.0]]]
        (let [[primal-value snapshot] (evaluate-lane bundle :primal [(cast x)])
              [pushed] (evaluate-lane bundle :linear [(cast dx) snapshot])
              [pulled] (evaluate-lane bundle :transpose [(cast dy) snapshot])]
          (is (= (cast (* x x)) primal-value))
          (is (= (cast x) snapshot))
          (is (= (cast (* 2.0 x dx)) pushed))
          (is (= (cast (* 2.0 x dy)) pulled))
          (is (= (* pushed dy) (* dx pulled))))))))

(deftest direct-scalar-captures-do-not-become-residual-arrays
  (let [bundle (build (source :double {:captures '[scale]
                                       :body '(raster.numeric/* element scale)}))
        interface (get-in bundle [:map-programs :interface])]
    (is (= {} (:residual-arrays interface)))
    (is (= {'scale 'scale} (:scalar-captures interface)))
    (is (= [6.0] (evaluate-lane bundle :primal [2.0 3.0])))
    (is (= [1.5] (evaluate-lane bundle :linear [0.5 3.0])))
    (is (= [-0.75] (evaluate-lane bundle :transpose [-0.25 3.0])))))

(deftest local-coefficients-are-snapshotted-not-recomputed-by-derivative-maps
  (let [bundle (build (source :double
                              {:captures '[scale]
                               :locals [(soac/local-value 'coefficient :double
                                                          '(raster.numeric/* scale scale))]
                               :body '(raster.numeric/* element coefficient)}))
        interface (get-in bundle [:map-programs :interface])
        [primal snapshot] (evaluate-lane bundle :primal [3.0 2.0])]
    (is (= #{'coefficient} (set (keys (:residual-arrays interface)))))
    (is (= {} (:scalar-captures interface)))
    (is (= [12.0 4.0] [primal snapshot]))
    (is (= [2.0] (evaluate-lane bundle :linear [0.5 snapshot])))
    (is (= [1.0] (evaluate-lane bundle :transpose [0.25 snapshot])))))

(deftest lifting-preserves-zero-extent-without-inventing-a-loss-reduction
  (let [bundle (build (source :double {:extent 0}))]
    (doseq [phase [:primal :linear :transpose]
            :let [program (get-in bundle [:map-programs phase])]]
      (is (= 'map (soac/operation-kind (first (soac/equations program)))))
      (is (every? #(= [0] (:shape (get-in (soac/facts program) [:values %])))
                  (soac/outputs program))))))

(deftest lifted-map-validation-checks-relations-not-just-a-producer-hash
  (let [bundle (build (source :double {}))
        residual (first (vals (get-in bundle [:map-programs :interface :residual-arrays])))
        primal (get-in bundle [:map-programs :primal])
        [eq] (soac/equations primal)
        reordered (soac/make (soac/facts primal)
                             [(list '= (second eq) (vec (reverse (nth eq 2))) (nth eq 3))]
                             (vec (reverse (soac/outputs primal))))]
    (is (= :ad-normalized-map-node-proof
           (reason (assoc bundle :primal primal))))
    (is (= :ad-normalized-map-interface
           (reason (assoc-in bundle [:map-programs :interface :residual-arrays] {}))))
    (is (= :ad-normalized-map-interface
           (reason (assoc-in bundle [:map-programs :interface :residual-arrays 'element] 'x))))
    (is (= :ad-normalized-map-interface
           (reason (assoc-in bundle [:map-programs :interface :residual-arrays]
                             {(with-meta 'element {:raster.type/tag 'float}) residual}))))
    (is (= :ad-normalized-map-interface
           (reason (assoc-in bundle [:map-programs :interface :scalar-captures] {'element 'x}))))
    (is (= :ad-normalized-map-programs
           (reason (assoc-in bundle [:map-programs :primal] reordered))))
    (is (= :ad-normalized-map-programs
           (reason (update-in bundle [:map-programs :linear]
                              alter-lambda
                              (fn [lambda]
                                (let [{:keys [parameters locals body-results]} (soac/lambda-parts lambda)
                                      locals (update-in locals [0 :init]
                                                        #(vary-meta % assoc :raster.type/tag 'float))]
                                  (soac/lambda-form parameters (soac/emit-locals locals) body-results)))))))
    (is (= :ad-normalized-map-programs
           (reason (update-in bundle [:map-programs :primal] alter-facts
                              #(assoc-in % [:values residual :ownership] :borrowed)))))
    (is (some? (reason (update-in bundle [:map-programs :primal] alter-facts
                                  #(assoc-in % [:equations 'ad-map-primal :aliases] {residual 'x})))))
    (is (some? (reason (update-in bundle [:map-programs :linear] alter-facts
                                  #(assoc-in % [:values residual :shape] '[other])))))
    (is (some? (reason (update-in bundle [:map-programs :linear] alter-facts
                                  #(assoc-in % [:values residual :dtype] :float)))))))

(deftest retained-map-stage-reconstruction-preserves-interface-identities
  (let [bundle (build (source :double {}))
        {:keys [source interface] :as retained} (:map-programs bundle)
        node (get-in bundle [:map-programs :proof :nodes 0])
        scalar (:scalar-derivation node)
        original-gensym clojure.core/gensym
        stage-prefixes #{"ad_residual_" "ad_tangent_" "ad_cotangent_" "ad_push_" "ad_pull_"}
        fresh-stage-identities (atom [])
        rebuilt (with-redefs [clojure.core/gensym
                              (fn
                                ([] (original-gensym))
                                ([prefix]
                                 (when (contains? stage-prefixes prefix)
                                   (swap! fresh-stage-identities conj prefix))
                                 (original-gensym prefix)))]
                  (#'normalized/construct-map-stage scalar source interface
                                                     (:equation-ids node)))]
    (is (empty? @fresh-stage-identities))
    (is (fingerprint/equivalent? interface (:interface rebuilt)))
    (is (fingerprint/equivalent? (dissoc retained :proof) rebuilt))))

(deftest exact-extent-value-identities-survive-lifting
  (doseq [extent ['count 'seq :extent-id [:binding 'n]]]
    (let [source (soac/remap-values (source :double {}) {'n extent})
          bundle (build source)]
      (doseq [phase [:primal :linear :transpose]
              :let [program (get-in bundle [:map-programs phase])]]
        (is (some #{extent} (:inputs (soac/facts program))))
        (is (= extent (soac/operation-extent (first (soac/equations program)))))
        (is (= (soac/extent-shape extent)
               (:shape (get-in (soac/facts program) [:values (first (soac/outputs program))]))))))))

(deftest scalar-capture-correspondence-is-metadata-sensitive
  (let [bundle (build (source :double {:captures '[scale]
                                       :body '(raster.numeric/* element scale)}))]
    (is (= :ad-normalized-map-interface
           (reason (assoc-in bundle [:map-programs :interface :scalar-captures 'scale]
                             (with-meta 'scale {:raster.type/tag 'float})))))))

(deftest lifted-map-requires-retained-integral-extent-evidence
  (let [primal (source :double {})]
    (doseq [facts [(assoc-in (soac/facts primal) [:values 'n :dtype] :double)
                   (assoc-in (soac/facts primal) [:values 'n :shape] '[n])]]
      (is (= :ad-normalized-map-boundary
             (try (build (alter-facts primal (constantly facts))) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest map-lifting-does-not-erase-placement-or-external-output-ownership
  (let [primal (source :double {})]
    (doseq [[facet value] [[:ownership :borrowed] [:ownership :external]
                           [:memory-space :device] [:placement {:device :other}]
                           [:sharding {:axis 0 :parts 2}]]]
      (is (= :ad-normalized-map-storage
             (try (build (alter-facts primal #(assoc-in % [:values 'y facet] value))) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (let [located (alter-facts primal
                               #(-> % (assoc-in [:values 'x :placement] {:device :same})
                                      (assoc-in [:values 'y :placement] {:device :same})))
          bundle (build located)]
      (doseq [phase [:primal :linear :transpose]
              id (soac/outputs (get-in bundle [:map-programs phase]))]
        (is (= {:device :same}
               (get-in (soac/facts (get-in bundle [:map-programs phase]))
                       [:values id :placement])))))))
