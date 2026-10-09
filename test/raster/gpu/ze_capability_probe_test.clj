(ns raster.gpu.ze-capability-probe-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ze-runtime :as ze]))

(deftest enumeration-enriches-each-device-with-its-own-capabilities
  (let [calls (atom [])]
    (with-redefs-fn {#'ze/query-module-capabilities
                    (fn [device]
                      (swap! calls conj device)
                      {:fp16? true :fp64? (= :first device)})}
      (fn []
        (is (= {:module-capabilities {:fp16? true :fp64? true}}
               (#'ze/enumerated-module-capabilities :first)))
        (is (= {:module-capabilities {:fp16? true :fp64? false}}
               (#'ze/enumerated-module-capabilities :second)))
        (is (= [:first :second] @calls))))))

(deftest optional-enumeration-failure-is-unknown-but-execution-query-fails
  (with-redefs-fn {#'ze/query-module-capabilities
                  (fn [_] (throw (ex-info "probe failed" {:reason :native-query})))
                  #'ze/ensure-init! (fn [] nil)}
    (fn []
      (let [result (#'ze/enumerated-module-capabilities :failed)]
        (is (not (contains? result :module-capabilities)))
        (is (= :unavailable (get-in result [:module-capabilities-query :status])))
        (is (= "probe failed" (get-in result [:module-capabilities-query :message]))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probe failed"
                           (ze/module-capabilities))))))
