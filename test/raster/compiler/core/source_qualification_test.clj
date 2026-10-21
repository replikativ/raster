(ns raster.compiler.core.source-qualification-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]))

(defn- temporary-namespaces [run]
  (let [names (mapv #(symbol (str "raster.qualification." % "." (gensym "owner")))
                    ["source" "named" "target" "consumer"])
        spaces (mapv create-ns names)]
    (try (apply run spaces)
         (finally (doseq [name names] (when (find-ns name) (remove-ns name)))))))

(defn- alias! [source spelling target]
  (binding [*ns* source]
    (ns-unalias source spelling)
    (alias spelling (ns-name target))))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- observe-qualification [body source]
  (binding [entry/*emission-inputs* (atom {:static [] :values []})]
    (let [qualified (inference/qualify-body-symbols body source #{}
                                                   entry/observe-source-resolution!)]
      {:body qualified :inputs @entry/*emission-inputs*})))

(deftest source-alias-wins-over-a-loaded-namespace-with-the-same-spelling
  (temporary-namespaces
   (fn [source named target consumer]
     (intern named 'value (fn [] :wrong-global))
     (intern target 'value (fn [] :original-alias))
     (alias! source (ns-name named) target)
     (let [body (list (symbol (str (ns-name named)) "value"))
           qualified (inference/qualify-body-symbols body source #{})]
       (is (= :original-alias (binding [*ns* source] (eval body))))
       (is (= (symbol (str (ns-name target)) "value") (first qualified)))
       (is (= :original-alias (binding [*ns* consumer] (eval qualified))))))))

(deftest var-qualification-retains-the-original-alias-lookup
  (temporary-namespaces
   (fn [source named target _]
     (intern named 'value 1)
     (let [selected (intern target 'value 2)]
       (alias! source 'chosen target)
       (let [{:keys [body inputs]} (observe-qualification 'chosen/value source)]
         (is (= (symbol (str (ns-name target)) "value") body))
         (is (contains? (get-in inputs [:definition-proofs selected :lookups])
                        [source 'chosen/value]))
         (is (entry/assert-definition-inputs-current! inputs))
         (alias! source 'chosen named)
         (is (= :stale-definition-proof-input
                (reason #(entry/assert-definition-inputs-current! inputs)))))))))

(deftest a-missing-alias-member-cannot-fall-through-to-a-global-namespace
  (temporary-namespaces
   (fn [source named target consumer]
     (intern named 'value (fn [] :wrong-global))
     (alias! source (ns-name named) target)
     (let [expression (list (symbol (str (ns-name named)) "value"))
           {:keys [body inputs]} (observe-qualification expression source)]
       (is (= (symbol (str (ns-name target)) "value") (first body)))
       (is (thrown? Exception (binding [*ns* source] (eval expression))))
       (is (thrown? Exception (binding [*ns* consumer] (eval body))))
       (is (entry/assert-definition-inputs-current! inputs))
       ;; A different still-missing target leaves ns-resolve's result nil.
       ;; The separately retained alias decision must nevertheless go stale.
       (alias! source (ns-name named) consumer)
       (is (= :stale-source-resolution (reason #(entry/assert-definition-inputs-current! inputs))))))))

(deftest class-qualification-and-misses-have-retained-identity-guards
  (temporary-namespaces
   (fn [source _ _ _]
     (.importClass ^clojure.lang.Namespace source java.util.Date)
     (let [{:keys [body inputs]} (observe-qualification '[Date Date] source)]
       (is (= '[java.util.Date java.util.Date] body))
       (is (= 1 (count (:producer-inputs inputs))) "repeated decisions share one guard")
       (is (entry/assert-definition-inputs-current! inputs))
       (ns-unmap source 'Date)
       (.importClass ^clojure.lang.Namespace source java.sql.Date)
       (is (= :stale-source-resolution (reason #(entry/assert-definition-inputs-current! inputs)))))
     (let [{:keys [body inputs]} (observe-qualification 'future-value source)]
       (is (= 'future-value body))
       (is (entry/assert-definition-inputs-current! inputs))
       (intern source 'future-value 3)
       (is (= :stale-source-resolution (reason #(entry/assert-definition-inputs-current! inputs))))))))

(deftest namespace-recreation-invalidates-a-qualified-missing-name
  (temporary-namespaces
   (fn [source named _ _]
     (let [body (symbol (str (ns-name named)) "missing")
           {:keys [inputs]} (observe-qualification body source)]
       (is (entry/assert-definition-inputs-current! inputs))
       (remove-ns (ns-name named))
       (create-ns (ns-name named))
       (is (= :stale-source-resolution (reason #(entry/assert-definition-inputs-current! inputs))))))))

(deftest a-detached-alias-target-cannot-authorize-its-replacement-namespace
  (temporary-namespaces
   (fn [source _ target _]
     (alias! source 'chosen target)
     (let [{:keys [body inputs]} (observe-qualification 'chosen/missing source)
           name (ns-name target)]
       (is (= (symbol (str name) "missing") body))
       (remove-ns name)
       (intern (create-ns name) 'missing 1)
       (is (identical? target (get (ns-aliases source) 'chosen)))
       (is (= :stale-source-resolution (reason #(entry/assert-definition-inputs-current! inputs))))))))

(deftest qualification-observes-no-quoted-or-lexical-lookups-and-propagates-refusals
  (temporary-namespaces
   (fn [source _ _ _]
     (let [observed (atom []) quoted '(quote (not-a-lookup))]
       (is (identical? quoted (inference/qualify-body-symbols quoted source #{} #(swap! observed conj %))))
       (is (= '(let* [local 2] local)
              (inference/qualify-body-symbols '(let* [local 2] local) source #{}
                                               #(swap! observed conj %))))
       (is (= 'parameter (inference/qualify-body-symbols 'parameter source #{'parameter}
                                                        #(swap! observed conj %))))
       (is (empty? @observed)))
     (is (= :observer-refusal
            (reason #(inference/qualify-body-symbols
                      'missing source #{}
                      (fn [_] (throw (ex-info "refused" {:reason :observer-refusal}))))))))))

(deftest refused-optional-qualification-does-not-leak-lookup-dependencies
  (temporary-namespaces
   (fn [source _ _ _]
     (let [owner (atom {:static [] :values []})]
       (binding [entry/*emission-inputs* owner]
         (is (nil? (entry/call-with-optional-definition-inputs
                    #(do (inference/qualify-body-symbols 'missing source #{}
                                                        entry/observe-source-resolution!)
                         nil))))
         (is (empty? (:producer-inputs @owner))))))))

(deftest imported-static-method-owner-retains-its-class-lookup
  (temporary-namespaces
   (fn [source _ _ consumer]
     (.importClass ^clojure.lang.Namespace source java.sql.Date)
     (let [{:keys [body inputs]} (observe-qualification '(Date/valueOf "2000-01-01") source)]
       (is (= 'java.sql.Date/valueOf (first body)))
       (is (= "2000-01-01" (str (binding [*ns* consumer] (eval body)))))
       (is (entry/assert-definition-inputs-current! inputs))
       (ns-unmap source 'Date)
       (.importClass ^clojure.lang.Namespace source java.util.Date)
       (is (= :stale-source-resolution (reason #(entry/assert-definition-inputs-current! inputs))))))))

(deftest structural-new-and-catch-classes-retain-their-lookup-history
  (temporary-namespaces
   (fn [source _ _ consumer]
     (.importClass ^clojure.lang.Namespace source java.io.IOException)
     (let [{:keys [body inputs]}
           (observe-qualification
            '(try (throw (new IOException)) (catch IOException error 7)) source)]
       (is (= 7 (binding [*ns* consumer] (eval body))))
       (is (entry/assert-definition-inputs-current! inputs))
       (ns-unmap source 'IOException)
       (is (= :stale-source-resolution (reason #(entry/assert-definition-inputs-current! inputs))))))))
