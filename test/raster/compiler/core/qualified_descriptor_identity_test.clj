(ns raster.compiler.core.qualified-descriptor-identity-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.nn]
            [raster.quant.kernels]))

(defn- with-descriptors [entries f]
  ;; No registration, compiler epoch changes, helper execution or persistent
  ;; registry mutation. Exact lookup and compatibility lookup see one snapshot.
  (with-redefs-fn
    {(ns-resolve 'raster.compiler.core.op-descriptor 'descriptor-registry)
     (atom entries)} f))

(deftest qualified-missing-specialization-cannot-borrow-another-owner
  (let [a {:shape {:dim-rule (fn [& _] 10)}
           :c-helper {:includes "A" :gen (fn [_] "A")}}
        b {:shape {:dim-rule (fn [& _] 20)}
           :c-helper {:includes "B" :gen (fn [_] "B")}}]
    (with-descriptors
      (array-map 'campaign.owner-a/probe a
                 'campaign.owner-b/probe_m_float b)
      (fn []
        (is (= [a 'campaign.owner-a/probe]
               (descriptor/resolve-op-descriptor 'campaign.owner-a/probe_m_double-impl)))
        (is (= [b 'campaign.owner-b/probe_m_float]
               (descriptor/resolve-op-descriptor 'campaign.owner-b/probe_m_float)))
        (is (nil? (descriptor/resolve-op-descriptor 'campaign.owner-b/probe_m_double-impl)))
        (is (nil? (:c-helper
                    (first (descriptor/resolve-op-descriptor
                             'campaign.owner-b/probe_m_double-impl)))))
        (is (nil? (descriptor/resolve-op-descriptor 'campaign.absent/probe_m_double)))
        (let [retained (with-meta '(.invk campaign.owner-a/probe_m_double-impl x)
                         {:raster.op/original 'campaign.owner-b/probe})]
          (is (nil? (descriptor/resolve-op-descriptor
                      (:operation (descriptor/call-description retained))))))
        ;; Bare compatibility is deliberately separate from qualified evidence.
        (is (= [a 'campaign.owner-a/probe]
               (descriptor/resolve-op-descriptor 'probe_m_double-impl)))))))

(deftest exact-specialization-and-nil-facet-withdrawal-win-over-base
  (let [base {:shape {:dim-rule :base} :c-helper {:includes "base"}}
        specialized {:shape nil :c-helper nil}]
    (with-descriptors
      {'campaign.owner/probe base
       'campaign.owner/probe_m_double-impl specialized}
      (fn []
        (is (= [specialized 'campaign.owner/probe_m_double-impl]
               (descriptor/resolve-op-descriptor 'campaign.owner/probe_m_double-impl)))
        (is (nil? (:c-helper (first (descriptor/resolve-op-descriptor
                                     'campaign.owner/probe_m_double-impl)))))))))

(deftest escaped-implementation-name-retains-its-qualified-owner
  (let [own {:effects {:mutating? true}}
        other {:effects {:mutating? false}}]
    (with-descriptors
      (array-map 'campaign.other/write! other 'campaign.owner/write! own)
      (fn []
        (is (= [own 'campaign.owner/write!]
               (descriptor/resolve-op-descriptor 'campaign.owner/write_bang_m_double-impl)))
        (is (nil? (descriptor/resolve-op-descriptor
                    'campaign.absent/write_bang_m_double-impl)))))))

(deftest existing-production-qualified-facets-survive-mangling
  ;; Consume actual declarations, but never invoke a deftm or C helper generator.
  (doseq [[base implementation facet]
          [['raster.nn/dense 'raster.nn/dense_m_doubles_doubles_doubles-impl :shape]
           ['raster.nn/dense-into! 'raster.nn/dense-into_bang_m_doubles_doubles_doubles_doubles-impl :shape]
           ['raster.quant.kernels/wi8-dot-q4 'raster.quant.kernels/wi8-dot-q4_m_bytes_long_bytes_long-impl :c-helper]
           ['raster.quant.kernels/wi8-dot-q4-x8 'raster.quant.kernels/wi8-dot-q4-x8_m_bytes_long_bytes_long_ints-impl :c-helper]]]
    (let [declared (descriptor/get-op-descriptor base)
          [resolved owner] (descriptor/resolve-op-descriptor implementation)]
      (is (some? (get declared facet)))
      (is (= base owner))
      (is (identical? (get declared facet) (get resolved facet))))))
