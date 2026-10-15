(ns raster.ad.normalized-proof-lifetime-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-prepared-owner-test :as owner]
            [raster.numeric :as n]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest node-scalar-and-outer-map-proof-share-one-derivative-dependency-fence
  (let [bundle (normalized/linearize-map-program
                (#'element/map-program :double {:body '(raster.math/exp element)})
                (the-ns 'raster.ad.normalized-proof-lifetime-test))
        revision (normalized/revisions)
        v #'n/* original @v original-meta (meta v)
        certificate #'normalized/certificate certify @certificate]
    (try
      (with-redefs-fn
        {certificate (fn [b]
                       (let [result (certify b)]
                         ;; Scalar-proof certification has already replayed the derivative-only
                         ;; multiplication. Mutate only after the enclosing map certificate.
                         (when (contains? b :map-programs)
                           (alter-var-root v (constantly (fn [& _] 99.0))))
                         result))}
        #(is (= :ad-normalized-execution-callee
                (reason (fn [] (normalized/validate! bundle))))))
      (is (= revision (normalized/revisions)))
      (finally (alter-var-root v (constantly original)) (reset-meta! v original-meta)))))

(deftest successful-proof-rechecks-actual-operation-observations
  (doseq [change [:value :metadata]]
    (let [bundle (#'owner/build) revision (normalized/revisions)
          v #'n/* original @v original-meta (meta v)
          certificate #'normalized/certificate certify @certificate]
      (try
        (with-redefs-fn
          {certificate (fn [b]
                         (let [result (certify b)]
                           (case change
                             :value (alter-var-root v (constantly (fn [& _] 99.0)))
                             :metadata (alter-meta! v assoc ::changed true))
                           result))}
          #(is (= :ad-normalized-execution-callee
                  (reason (fn [] (normalized/validate! bundle))))))
        (is (= revision (normalized/revisions)) "fault does not rely on a registry epoch")
        (finally (alter-var-root v (constantly original)) (reset-meta! v original-meta))))))

(deftest successful-proof-rechecks-effective-receiver-metadata
  (binding [owner/receiver (#'owner/selected)]
    (let [bundle (#'owner/alias-bundle) v #'owner/receiver original-meta (meta v)
          certificate #'normalized/certificate certify @certificate]
      (try
        (with-redefs-fn
          {certificate (fn [b]
                         (let [result (certify b)]
                           (alter-meta! v assoc ::changed true)
                           result))}
          #(is (= :ad-normalized-execution-callee
                  (reason (fn [] (normalized/validate! bundle))))))
        (finally (reset-meta! v original-meta))))))

(deftest later-observations-cannot-replace-the-first
  (let [bundle (#'owner/build) v #'n/* original @v original-meta (meta v)
        selected #'normalized/selected-call-contract! select @selected calls (atom 0)]
    (try
      (with-redefs-fn
        {selected (fn [& args]
                    (let [result (apply select args) call (swap! calls inc)]
                      (case call
                        1 (alter-var-root v (constantly (fn [& _] 99.0)))
                        2 (alter-var-root v (constantly original))
                        nil)
                      result))}
        #(is (= :ad-normalized-execution-callee
                (reason (fn [] (normalized/validate! bundle))))))
      (is (< 2 @calls))
      (is (identical? original @v) "final state alone cannot erase an observed disagreement")
      (finally (alter-var-root v (constantly original)) (reset-meta! v original-meta)))))

(deftest equal-receiver-value-does-not-bless-namespace-remapping
  (binding [owner/receiver (#'owner/selected)]
    (let [bundle (#'owner/alias-bundle) source-ns (the-ns 'raster.ad.normalized-prepared-owner-test)
          v #'owner/receiver value (var-get v) original-meta (meta v)
          certificate #'normalized/certificate certify @certificate]
      (try
        (with-redefs-fn
          {certificate (fn [b]
                         (let [result (certify b)]
                           (ns-unmap source-ns 'receiver)
                           (reset-meta! (intern source-ns 'receiver value) original-meta)
                           result))}
          #(is (= :ad-normalized-execution-callee
                  (reason (fn [] (normalized/validate! bundle))))))
        (finally
          (ns-unmap source-ns 'receiver)
          (.refer ^clojure.lang.Namespace source-ns 'receiver v))))))

(deftest receiver-contract-failure-precedes-new-lifetime-refusal
  (let [v #'owner/receiver original-meta (meta v)
        selected #'normalized/selected-call-contract! select @selected]
    (try
      (with-redefs [owner/receiver (#'owner/selected)]
        (let [bundle (#'owner/alias-bundle)]
          (with-redefs-fn
            {selected (fn [call & args]
                        (when (= 'raster.ad.normalized-prepared-owner-test/receiver
                                 (:implementation-op call))
                          (alter-var-root v (constantly (Object.))))
                        (apply select call args))}
            #(is (= :ad-normalized-call-contract
                    (reason (fn [] (normalized/validate! bundle))))))))
      (finally (reset-meta! v original-meta)))))

(deftest unchanged-alias-and-original-failures-retain-their-contract
  (binding [owner/receiver (#'owner/selected)]
    (let [bundle (#'owner/alias-bundle)]
      (is (identical? bundle (normalized/validate! bundle)))))
  (let [bundle (#'owner/build) v #'n/* original @v original-meta (meta v)
        failure (ex-info "original certificate failure" {:reason ::fixture})]
    (try
      (with-redefs-fn
        {#'normalized/certificate (fn [_]
                                   (alter-var-root v (constantly (fn [& _] 99.0)))
                                   (throw failure))}
        #(is (identical? failure
                        (try (normalized/validate! bundle) nil (catch Throwable e e)))))
      (finally (alter-var-root v (constantly original)) (reset-meta! v original-meta)))
    (is (= :ad-normalized-certificate
            (reason #(normalized/validate! (assoc bundle :certificate "invalid")))))))
