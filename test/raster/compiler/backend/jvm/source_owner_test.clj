(ns raster.compiler.backend.jvm.source-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.core.value-completion :as completion]))

(defn- function-form [params body] (list 'fn params body))
(defn- context [captures]
  {:source-policy {:unchecked-math false} :source-ns *ns*
   :captures captures :reference-bindings {} :methods []})

(deftest actual-private-values-are-captured-without-installing-vars
  (let [value (Object.)
        private-function (fn [x] [value x])
        calls (atom 0)
        compiled (owner/compile-function
                  '[x] '(private-call x) function-form
                  (assoc (context [{:port 'private-call :value private-function}])
                         :guard #(swap! calls inc)))
        result (compiled 7.0)]
    (is (identical? value (first result)))
    (is (= 7.0 (second result)))
    (is (nil? (ns-resolve *ns* 'private-call)))
    (is (= 2 @calls) "the caller still owns runtime/residual fences")))

(deftest private-capture-factory-has-no-twenty-argument-limit
  (let [values (vec (repeatedly 25 #(Object.)))
        ports (mapv #(symbol (str "port" %)) (range 25))
        compiled (owner/compile-function
                  [] ports function-form
                  (context (mapv (fn [port value] {:port port :value value}) ports values)))]
    (is (every? true? (map identical? values (compiled))))))

(deftest reference-completion-consumes-the-actual-class-port
  (let [source (completion/complete-value (completion/reference-check-contract String) 'x)
        compiled (owner/compile-function
                  '[x] source function-form
                  (assoc (context [{:port 'carrier :value String}])
                         :reference-bindings {String 'carrier}))
        value (String. "not interned")]
    (is (identical? value (compiled value)))
    (is (nil? (compiled nil)))
    (is (thrown? ClassCastException (compiled (Object.)))))
  (let [error (try
                (owner/compile-function
                 '[x] 'x function-form
                 (assoc (context [{:port 'carrier :value Object}])
                        :reference-bindings {String 'carrier}))
                (catch clojure.lang.ExceptionInfo e e))]
    (is (= :source-owner-reference-port (:reason (ex-data error))))))

(deftest canonical-primitive-completion-uses-the-actual-compilation-policy
  (doseq [unchecked? [false true]]
    (let [policy {:unchecked-math unchecked?}
          source (completion/complete-value
                  (completion/primitive-input-contract Integer/TYPE policy) 'x)
          compiled (binding [*unchecked-math* (not unchecked?)]
                     (owner/compile-function '[x] source function-form
                                             (assoc (context []) :source-policy policy)))]
      (is (= 3 (compiled 3)))
      (if unchecked?
        (is (= 1 (compiled 4294967297)))
        (is (thrown? ArithmeticException (compiled 4294967297)))))))

(deftest captures-cannot-shadow-even-unused-nested-binders
  (doseq [[params source]
          [['[private] 1.0]
           ['[] '(let* [private 1.0] 2.0)]
           ['[] '(fn* [private] 2.0)]
           ['[] '(loop* [private 0] (if (< private 1) (recur (inc private)) 2.0))]]]
    (let [error (try (owner/compile-function params source function-form
                                            (context [{:port 'private :value (Object.)}]))
                     (catch clojure.lang.ExceptionInfo e e))]
      (is (= :source-owner-capture-scope (:reason (ex-data error)))))))

(deftest guard-refusal-prevents-compilation-publication
  (let [calls (atom 0)
        error (try (owner/compile-function
                    [] 1.0 function-form
                    (assoc (context []) :guard #(do (swap! calls inc)
                                                    (throw (ex-info "stale owner" {:reason :stale})))))
                   (catch clojure.lang.ExceptionInfo e e))]
    (is (= :stale (:reason (ex-data error))))
    (is (= 1 @calls))))
