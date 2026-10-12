(ns raster.compiler.ir.soac-spine-append-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-sum-test :as sums]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- parts []
  (soac/equation-subprograms (#'sums/source :double)))

(defn- facts-change [p f]
  (soac/make (f (soac/facts p)) (soac/equations p) (soac/outputs p)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest explicit-spine-preserves-facts-and-output-order
  (let [[p c] (parts)
        p (facts-change p #(-> % (assoc :provenance {:producer :kept})
                               (assoc-in [:equations 'map-0 :provenance] {:map :kept})))
        c (facts-change c #(-> % (assoc :provenance {:consumer :kept})
                               (assoc-in [:equations 'sum-1 :provenance] {:sum :kept})))
        combined (soac/append-pure-spines p c {'mapped 'mapped} '[loss mapped])]
    (is (= '[xs weights size] (:inputs (soac/facts combined))))
    (is (= '[loss mapped] (soac/outputs combined)))
    (is (= (concat (soac/equations p) (soac/equations c)) (soac/equations combined)))
    (is (= {:producer :kept :consumer :kept} (:provenance (soac/facts combined))))
    (doseq [part [p c] [id facts] (:equations (soac/facts part))]
      (is (fingerprint/equivalent? facts (get-in (soac/facts combined) [:equations id]))))
    (is (identical? combined (soac/validate! combined)))))

(deftest conflicting-shared-value-facts-never-prefer-either-side
  (let [[p c] (parts)]
    (doseq [[left right] [[(av/tensor {:dtype :double :shape [1]})
                          (av/tensor {:dtype :float :shape [1]})]
                         [(av/tensor {:dtype :double :shape [1]})
                          (av/tensor {:dtype :double :shape [2]})]
                         [(av/tensor {:dtype :double :shape [1] :ownership :owned})
                          (av/tensor {:dtype :double :shape [1]})]]]
      (let [p (facts-change p #(assoc-in % [:values 'retained-fact] left))
            c (facts-change c #(assoc-in % [:values 'retained-fact] right))]
        (doseq [[a b boundary] [[p c {'mapped 'mapped}] [c p {}]]]
          (is (= :typed-soac-composition-facts
                 (reason #(soac/append-pure-spines a b boundary '[loss mapped])))))))))

(deftest non-identity-boundary-remaps-consumer-towards-producer
  (let [[p original] (parts)
        p (facts-change p #(assoc % :extension-contract {:opaque :same}))
        original (facts-change original #(assoc % :extension-contract {:opaque :same}))
        consumer (soac/remap-values original {'mapped 'consumer-input})
        combined (soac/append-pure-spines p consumer {'consumer-input 'mapped} '[loss mapped])]
    (is (not (contains? (:values (soac/facts combined)) 'consumer-input)))
    (is (= '[xs weights size] (:inputs (soac/facts combined))))
    (is (= '[loss mapped] (soac/outputs combined)))
    (is (= {:opaque :same} (:extension-contract (soac/facts combined))))
    (is (= (concat (soac/equations p) (soac/equations original)) (soac/equations combined)))
    (is (fingerprint/equivalent? (:equations (soac/facts original))
                                (select-keys (:equations (soac/facts combined)) ['sum-1])))))

(deftest boundaries-identities-and-numerical-contracts-refuse
  (let [[p c] (parts)]
    (is (= :typed-soac-composition-boundary
           (reason #(soac/append-pure-spines p c {} '[loss]))))
    (is (= :typed-soac-composition-boundary
           (reason #(soac/append-pure-spines
                     (soac/make (soac/facts p) (soac/equations p) []) c {} '[loss])))
        "private producer results cannot become an implicit boundary")
    (is (= :typed-soac-composition-boundary
           (reason #(soac/append-pure-spines p c {'size 'mapped} '[loss]))))
    (is (= :typed-soac-composition-numerics
           (reason #(soac/append-pure-spines p
                     (facts-change c (fn [f] (assoc-in f [:attributes :source-arithmetic] :different)))
                     {'mapped 'mapped} '[loss]))))
    (is (= :typed-soac-composition-numerics
           (reason #(soac/append-pure-spines p
                     (facts-change c (fn [f] (assoc f :extension-contract :consumer-only)))
                     {'mapped 'mapped} '[loss])))
        "an unknown consumer program contract is not silently discarded")
    (let [e (first (soac/equations c))
          duplicate (soac/make (-> (soac/facts c)
                                  (assoc :equations {'map-0 (get-in (soac/facts c) [:equations 'sum-1])}))
                              [(apply list (assoc (vec e) 1 'map-0))] (soac/outputs c))]
      (is (= :typed-soac-equation-ids
             (reason #(soac/append-pure-spines p duplicate {'mapped 'mapped} '[loss])))))
    (let [duplicate (soac/remap-values c {'mapped 'temporary-input 'loss 'mapped})]
      (is (= :typed-soac-remap-collision
             (reason #(soac/append-pure-spines p duplicate {'temporary-input 'mapped} '[mapped])))))
    (is (= :typed-soac-output-boundary
           (reason #(soac/append-pure-spines p c {'mapped 'mapped} '[xs]))))))

(deftest effects-aliases-and-provenance-conflicts-are-not-erased
  (let [[p c] (parts)]
    (doseq [changed [(facts-change c #(-> % (assoc :effects #{:memory/read})
                                         (assoc-in [:equations 'sum-1 :effects] #{:memory/read})))
                     (facts-change c #(assoc-in % [:equations 'sum-1 :aliases] {'loss 'mapped}))]]
      (is (= :typed-soac-composition-boundary
             (reason #(soac/append-pure-spines p changed {'mapped 'mapped} '[loss])))))
    (is (= :typed-soac-composition-facts
           (reason #(soac/append-pure-spines
                     (facts-change p (fn [f] (assoc f :provenance {:shared :producer})))
                     (facts-change c (fn [f] (assoc f :provenance {:shared :consumer})))
                     {'mapped 'mapped} '[loss]))))))

(deftest ad-owned-boundary-is-an-explicit-derivation-not-a-merge-preference
  (doseq [dt [:float :double] ownership [nil :owned]]
    (let [source (facts-change (#'sums/source dt)
                               #(assoc-in % [:values 'mapped :ownership] ownership))
          bundle (#'sums/build source 'xs)
          sum-programs (:sum-programs bundle)
          tangent (get-in bundle [:map-programs :interface :tangent-output])]
      (is (identical? source (:source sum-programs)))
      (is (= ownership (get-in (soac/facts source) [:values 'mapped :ownership])))
      (is (= :owned (get-in (soac/facts (:primal sum-programs)) [:values 'mapped :ownership])))
      (is (= :owned (get-in (soac/facts (:linear sum-programs)) [:values tangent :ownership])))))
  (let [source (#'sums/source :double) bundle (#'sums/build source 'xs)
        producer (get-in bundle [:map-programs :primal])
        reduction (second (soac/equations source))]
    (doseq [ownership [:borrowed :external]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (#'sums/build (facts-change source #(assoc-in % [:values 'mapped :ownership] ownership)) 'xs))))
    (doseq [changed [(facts-change producer #(assoc-in % [:values 'mapped :ownership] nil))
                     (facts-change producer #(assoc-in % [:values 'mapped :attributes] {:different true}))]]
      (is (= :ad-normalized-sum-boundary
             (reason #(#'normalized/append-ordered-sum changed reduction source 'loss)))))))
