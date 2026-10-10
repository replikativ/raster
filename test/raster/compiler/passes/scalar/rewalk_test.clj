(ns raster.compiler.passes.scalar.rewalk-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.passes.scalar.rewalk :as rewalk]))

(deftest rewalk-failures-are-not-certified-passthroughs-test
  (doseq [failure [(ex-info "invariant" {:reason :raster/fatal})
                   (ex-info "compiler bug" {:reason :raster/bug})
                   (ex-info "missing type" {:form '(raster.par/reduce a init i n a)})
                   (NullPointerException. "implementation failure")]]
    (let [walks (atom 0)
          opts {:simplify? true :dtype :double :source-ns 'user
                :param-env {'x 'double}}
          observed (with-redefs [walker/walk-body
                                (fn [& _] (swap! walks inc) (throw failure))]
                     (try (rewalk/pe-rewalk 'x opts)
                          nil
                          (catch Exception exception exception)))]
      (is (identical? failure observed)
          "A failed walk cannot be replaced with the unverified pre-walk form")
      (is (= 1 @walks))
      (with-redefs [walker/walk-body (fn [& _] (throw failure))]
        (is (= 'x (rewalk/pe-rewalk 'x (assoc opts :simplify? false)))
            "Explicitly disabling the pass does not attempt a walk")))))

(deftest missing-accumulator-type-is-a-compilation-error-test
  (let [source '(raster.par/reduce acc (identity x) i 4 acc)
        observed (try (rewalk/pe-rewalk source
                                       {:simplify? true :dtype :double :source-ns 'user
                                        :param-env {'x 'double}})
                      nil
                      (catch clojure.lang.ExceptionInfo exception exception))]
    (is (instance? clojure.lang.ExceptionInfo observed))
    (is (re-find #"cannot infer accumulator type" (or (some-> observed ex-message) "")))
    (is (= '(identity x) (:init (ex-data observed))))
    (is (= source (:form (ex-data observed))))))

(deftest passthrough-when-disabled-test
  (testing "pe-rewalk is a no-op when simplify is disabled"
    (let [form '(let* [a (raster.numeric/+ x y)] a)]
      (is (= form (rewalk/pe-rewalk form {:simplify? false}))))))

(deftest rewalk-devirtualizes-and-preserves-effect-bindings-test
  (testing "scalar bindings are re-walked while effect bindings are preserved"
    (let [form '(let* [a (raster.numeric/+ x y)
                       _effect (println a)]
                      a)
          {:keys [form stats]} (rewalk/pe-rewalk form {:simplify? true
                                                       :active-params '[x y]
                                                       :dtype :double})]
      (is (= 2 (:pe-rewalk-bindings-before stats)))
      (is (= 2 (:pe-rewalk-bindings-after stats)))
      (is (= 'let* (first form)))
      (is (some #{'_effect} (take-nth 2 (second form))))
      (is (some seq? (tree-seq coll? seq form)))
      (is (some #(and (seq? %) (= '.invk (first %)))
                (tree-seq coll? seq form))))))
