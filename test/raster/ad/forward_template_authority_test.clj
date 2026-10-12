(ns raster.ad.forward-template-authority-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.templates :as templates]))

(defn- isolated [registry f]
  ;; Never clear or replace the process's actual rule atom contents.
  (with-redefs-fn {#'templates/template-registry (atom registry)} f))

(defn- registry-atom [] (var-get #'templates/template-registry))

(deftest copied-alias-resolves-to-the-existing-forward-owner
  (let [base {:params '[x] :structure {:class :elementwise :in #{0}}}
        registry {'Math/exp base 'raster.math/exp {:params '[x]}}]
    (isolated registry
      (fn []
        (doseq [op '[Math/exp raster.math/exp]]
          (let [[template canonical] (templates/resolve-forward-template op)]
            (is (= 'Math/exp canonical))
            (is (identical? base template))))
        (is (= registry @(registry-atom)))
        (is (nil? (templates/resolve-forward-template 'user.math/exp)))
        (is (nil? (templates/resolve-forward-template 'missing)))))))

(deftest explicit-alias-facets-are-not-replaced-with-base-rules
  (let [base-rule (fn [& _] :base)
        alias-rule (fn [& _] :alias)
        registry {'Math/exp {:jvp-fn base-rule :structure {:class :elementwise :in #{0}}}
                  'raster.math/exp {:jvp-fn alias-rule}
                  'clojure.core/+ {:structure {:class :linear :in #{0 1}}}
                  '+ {:jvp-fn base-rule}}]
    (isolated registry
      (fn []
        (is (= 'raster.math/exp (second (templates/resolve-forward-template 'raster.math/exp))))
        (is (identical? alias-rule (templates/op-jvp-fn 'raster.math/exp)))
        (is (= 'clojure.core/+ (second (templates/resolve-forward-template 'clojure.core/+))))
        (is (identical? base-rule (templates/op-jvp-fn 'Math/exp)))))))

(deftest fallback-does-not-discard-a-qualified-user-identity
  (isolated {'+ {:structure {:class :linear :in #{0 1}}}
             'clojure.core/+ {:params '[a b]}
             'user.math/exp {:params '[x]}}
    (fn []
      (is (= '+ (second (templates/resolve-forward-template 'clojure.core/+))))
      (is (nil? (templates/resolve-forward-template 'user.math/exp)))
      (is (nil? (templates/op-jvp-fn 'user.math/exp))))))

(deftest one-snapshot-resolution-and-cycle-refusal
  (let [old-rule (fn [& _] :old) new-rule (fn [& _] :new)
        original #'templates/resolve-template-in
        resolve-in @original]
    (isolated {'Math/exp {:jvp-fn old-rule} 'raster.math/exp {}}
      (fn []
        (with-redefs-fn {original
                        (fn [registry op]
                          (when (= op 'raster.math/exp)
                            (swap! (registry-atom) assoc 'Math/exp {:jvp-fn new-rule}))
                          (resolve-in registry op))}
          (fn []
            (is (identical? old-rule (:jvp-fn (first (templates/resolve-forward-template 'raster.math/exp)))))
            (is (identical? new-rule (get-in @(registry-atom) ['Math/exp :jvp-fn]))))))))
  (isolated {'first {} 'second {}}
    (fn []
      (with-redefs-fn {#'templates/qualified->base-op {'first 'second 'second 'first}}
        (fn []
          (is (nil? (templates/resolve-forward-template 'first)))
          (is (nil? (templates/op-jvp-fn 'first))))))))

(deftest derived-forward-rule-is-cached-at-the-resolved-owner
  (let [base (dissoc (templates/get-template '+) :jvp-fn)
        registry {'+ base 'raster.numeric/+ (dissoc base :structure)}]
    (isolated registry
      (fn []
        (let [revision (templates/registry-revision)
              rule (templates/op-jvp-fn 'raster.numeric/+)]
          (is (fn? rule))
          (is (identical? rule (get-in @(registry-atom) ['+ :jvp-fn])))
          (is (nil? (get-in @(registry-atom) ['raster.numeric/+ :jvp-fn])))
          (is (identical? rule (templates/op-jvp-fn 'raster.numeric/+)))
          (is (= revision (templates/registry-revision))))))))

(deftest mutation-during-alias-resolution-retries-rule-publication
  (let [old-rule (fn [& _] :old) new-rule (fn [& _] :new)
        resolve-in @#'templates/resolve-template-in
        changed? (atom false) derivations (atom 0)]
    (isolated {'Math/exp {:structure :old} 'raster.math/exp {}}
      (fn []
        (with-redefs-fn
          {#'templates/resolve-template-in
           (fn [registry op]
             (when (and (= 'raster.math/exp op) (compare-and-set! changed? false true))
               (swap! (registry-atom) assoc 'Math/exp {:jvp-fn new-rule}))
             (resolve-in registry op))
           #'templates/derive-jvp-fn (fn [& _] (swap! derivations inc) old-rule)}
          (fn []
            (is (identical? new-rule (templates/op-jvp-fn 'raster.math/exp)))
            (is (= 1 @derivations))
            (is (identical? new-rule (get-in @(registry-atom) ['Math/exp :jvp-fn])))
            (is (nil? (get-in @(registry-atom) ['raster.math/exp :jvp-fn])))))))))
