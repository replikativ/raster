(ns raster.compiler.ir.static-validation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.static-validation :as validation]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-launch :as launch]))

(deftest success-is-local-and-identity-based
  (let [calls (atom 0) value {:x 1}
        check (fn [v] (swap! calls inc) v)]
    (validation/with-validation-scope
      #(do (is (identical? value (validation/validate check value)))
           (validation/validate check value)
           (validation/validate check (into {} value))
           (is (= 2 @calls))))
    (validation/validate check value)
    (validation/with-validation-scope #(validation/validate check value))
    (is (= 4 @calls))))

(deftest failures-and-validator-replacements-never-inherit-success
  (let [calls (atom 0) value {:x 1}
        good (fn [v] (swap! calls inc) v)
        bad (fn [_] (swap! calls inc) (throw (ex-info "invalid" {})))]
    (validation/with-validation-scope
      #(do (validation/validate good value)
           (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid"
                                 (validation/validate bad value)))
           (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid"
                                 (validation/validate bad value)))
           (is (= 3 @calls))))))

(deftest real-artifact-proof-cannot-authorize-malformed-copies
  (let [value (artifact/make {:kernel-name "static_scope"
                              :source "__kernel void static_scope(__global float* out) {}"
                              :abi [(abi/slot 'out :output :float :role :result)] :arguments '[out]
                              :launch (launch/spec {:workgroup-size [1] :group-count [1]})})
        bad (assoc value :source "__kernel void different_entry(__global float* out) {}")
        body (ns-resolve 'raster.compiler.ir.kernel-artifact 'validate-uncached!)
        original @body calls (atom 0)]
    (with-redefs-fn {body (fn [v] (swap! calls inc) (original v))}
      #(validation/with-validation-scope
        (fn []
          (is (identical? value (artifact/validate! value)))
          (artifact/validate! value)
          (is (thrown? clojure.lang.ExceptionInfo (artifact/validate! bad)))
          (is (thrown? clojure.lang.ExceptionInfo (artifact/validate! bad)))
          (is (= 3 @calls)))))))

(deftest nested-and-propagated-contexts-are-isolated
  (let [calls (atom 0) value {:x 1}
        check (fn [v] (swap! calls inc) v)]
    (validation/with-validation-scope
      (fn [] (validation/validate check value)
           (validation/with-validation-scope #(validation/validate check value))
           @(future (validation/validate check value))
           (validation/validate check value)))
    (is (= 3 @calls))))

(deftest exception-cleans-up-and-capacity-is-bounded
  (let [calls (atom 0) value {:x 1}
        check (fn [v] (swap! calls inc) v)]
    (is (thrown? clojure.lang.ExceptionInfo
                 (validation/with-validation-scope
                   #(do (validation/validate check value)
                        (throw (ex-info "request failed" {}))))))
    (validation/validate check value)
    (is (= 2 @calls))
    (reset! calls 0)
    (let [values (mapv #(hash-map :x %) (range 1100))]
      (validation/with-validation-scope
        #(do (doseq [v values] (validation/validate check v))
             (doseq [v values] (validation/validate check v))))
      (is (= (+ 1100 (- 1100 1024)) @calls)))))

(deftest nested-same-validator-retains-both-successes
  (let [calls (atom 0) child {:x 1} parent {:child child}]
    (letfn [(check [value]
              (swap! calls inc)
              (when-let [nested (:child value)] (validation/validate check nested))
              value)]
      (validation/with-validation-scope
        #(do (validation/validate check parent)
             (validation/validate check child)
             (validation/validate check parent)
             (is (= 2 @calls)))))))
