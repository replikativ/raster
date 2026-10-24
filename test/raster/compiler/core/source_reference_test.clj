(ns raster.compiler.core.source-reference-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]))

(defn- with-scopes [consume!]
  (let [names (repeatedly 3 #(gensym "raster.source-reference."))
        scopes (mapv create-ns names)]
    (try (apply consume! scopes)
         (finally
           (doseq [name names]
             (when (find-ns name) (remove-ns name)))))))

(defn- decision [source symbol]
  {:kind :symbol-resolution :source-ns source :symbol symbol
   :result (ns-resolve source symbol)})

(defn- collect-reference [observation]
  (let [owner (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* owner]
      (entry/observe-source-reference! observation))
    @owner))

(defn- reason [operation]
  (try (operation) nil
       (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))

(deftest reference-identity-survives-real-source-preparation-and-jit
  (with-scopes
    (fn [source caller _]
      (let [calls (atom 0)]
        (intern source 'calls calls)
        (binding [*ns* source]
          (refer 'clojure.core)
          (eval '(raster.core/deftm producer [x :- Double] :- Double
                   (do (swap! calls inc) x))))
        (binding [*ns* caller] (alias 'chosen (ns-name source)))
        (let [selected (core/resolve-deftm-var (ns-resolve source 'producer))
              receiver (ns-resolve source (symbol (str (:name (meta selected)) "-impl")))
              lookup (symbol "chosen" (str (.sym ^clojure.lang.Var receiver)))
              observation (decision caller lookup)
              reference (atom {:static [] :values []})
              full-before (atom {:static [] :values []})
              implementation (var-get receiver)]
          (binding [entry/*emission-inputs* reference]
            (is (identical? receiver (entry/observe-source-reference! observation)))
            (entry/observe-source-reference! observation))
          (is (= 1 (count (:producer-inputs @reference))))
          (is (empty? (:definition-proofs @reference)))
          (binding [entry/*emission-inputs* full-before]
            (entry/observe-source-resolution! observation))
          (core/prepare-deftm-frontend! selected)
          (is (entry/assert-definition-inputs-current! @reference))
          (is (= :stale-definition-proof-input
                 (reason #(entry/assert-definition-inputs-current! @full-before))))
          (let [full-after-source (atom {:static [] :values []})]
            (binding [entry/*emission-inputs* full-after-source]
              (entry/observe-source-resolution! observation))
            (core/ensure-bytecode-implementation! receiver)
            (is (not (identical? implementation (var-get receiver))))
            (is (entry/assert-definition-inputs-current! @reference))
            (is (= :stale-definition-proof-input
                   (reason #(entry/assert-definition-inputs-current! @full-after-source)))))
          (is (zero? @calls) "neither preparation runs the numerical body")
          (let [both (atom {:static [] :values []}) original (meta receiver)]
            (binding [entry/*emission-inputs* both]
              (entry/observe-source-resolution! observation)
              (let [full (get-in @both [:definition-proofs receiver])]
                (entry/observe-source-reference! observation)
                (entry/observe-source-reference! observation)
                (is (identical? full (get-in @both [:definition-proofs receiver])))))
            (try
              (alter-meta! receiver assoc ::changed (Object.))
              (is (entry/assert-definition-inputs-current! @reference))
              (is (= :stale-definition-proof-input
                     (reason #(entry/assert-definition-inputs-current! @both))))
              (finally (reset-meta! receiver original)))))))))

(deftest reference-guards-reject-lookup-and-canonical-var-identity-drift
  (doseq [mutation [:alias :reintern :target-namespace :source-namespace]]
    (with-scopes
      (fn [target caller replacement]
        (let [v (intern target 'value (Object.))
              target-name (ns-name target) caller-name (ns-name caller)]
          (intern replacement 'value (Object.))
          (binding [*ns* caller]
            (alias 'chosen target-name)
            (refer target-name :only '[value] :rename '{value imported}))
          (let [lookup (if (= mutation :reintern) 'imported 'chosen/value)
                observed (collect-reference (decision caller lookup))]
            (is (entry/assert-definition-inputs-current! observed))
            (case mutation
              :alias (binding [*ns* caller]
                       (ns-unalias caller 'chosen)
                       (alias 'chosen (ns-name replacement)))
              :reintern (do (ns-unmap target 'value)
                            (intern target 'value (Object.))
                            (is (identical? v (ns-resolve caller 'imported))))
              :target-namespace (do (remove-ns target-name)
                                    (intern (create-ns target-name) 'value (Object.))
                                    (is (identical? target (get (ns-aliases caller) 'chosen)))
                                    (is (identical? v (ns-resolve caller 'chosen/value))))
              :source-namespace (do (remove-ns caller-name)
                                    (create-ns caller-name)
                                    (is (identical? v (ns-resolve caller lookup)))))
            (is (= :stale-source-resolution
                   (reason #(entry/assert-definition-inputs-current! observed))))))))))

(deftest class-miss-and-namespace-decisions-retain-exact-identity
  (doseq [kind [:class :miss :alias :namespace]]
    (with-scopes
      (fn [target caller replacement]
        (.importClass caller 'ChosenClass String)
        (binding [*ns* caller] (alias 'chosen (ns-name target)))
        (let [observation (case kind
                            :class (decision caller 'ChosenClass)
                            :miss (decision caller 'missing)
                            :alias {:kind :alias-resolution :source-ns caller :symbol 'chosen :result target}
                            :namespace {:kind :namespace-resolution :source-ns caller
                                        :symbol (ns-name target) :result target})
              inputs (collect-reference observation)]
          (is (entry/assert-definition-inputs-current! inputs))
          (case kind
            :class (do (ns-unmap caller 'ChosenClass) (.importClass caller 'ChosenClass Long))
            :miss (intern caller 'missing 1)
            :alias (binding [*ns* caller]
                     (ns-unalias caller 'chosen)
                     (alias 'chosen (ns-name replacement)))
            :namespace (do (remove-ns (ns-name target)) (create-ns (ns-name target))))
          (is (= :stale-source-resolution
                 (reason #(entry/assert-definition-inputs-current! inputs)))))))))

(deftest accepted-global-type-facts-retain-full-snapshots
  (doseq [[attributes initial expected] [[{:tag 'double} 3.0 'double]
                                        [{:const true} (long 3) 'long]]
          mutation [:metadata :root :lookup]]
    (with-scopes
      (fn [target caller replacement]
        (let [v (intern target 'number initial)
              other (intern replacement 'number initial)
              owner (atom {:static [] :values []})]
          (doseq [variable [v other]] (alter-meta! variable merge attributes))
          (binding [*ns* caller]
            (alias 'chosen (ns-name target))
            (binding [entry/*emission-inputs* owner]
              (is (= expected (inference/infer-arg-tag 'chosen/number {})))))
          (is (contains? (:definition-proofs @owner) v))
          (is (identical? initial (get-in @owner [:definition-proofs v :value])))
          (is (entry/assert-definition-inputs-current! @owner))
          (case mutation
            :metadata (alter-meta! v assoc :tag 'float)
            :root (alter-var-root v (constantly (if (= expected 'double) 4.0 (long 4))))
            :lookup (binding [*ns* caller]
                      (ns-unalias caller 'chosen)
                      (alias 'chosen (ns-name replacement))))
          (is (= :stale-definition-proof-input
                 (reason #(entry/assert-definition-inputs-current! @owner)))))))))

(deftest unsuccessful-type-lookups-leave-no-owner-history
  (with-scopes
    (fn [target caller _]
      (intern target 'function (fn [value] value))
      (.importClass caller 'Callback java.lang.Runnable)
      (binding [*ns* caller]
        (alias 'chosen (ns-name target))
        (doseq [symbol '[chosen/function Callback absent]]
          (let [owner (atom {:static [] :values []}) before @owner]
            (binding [entry/*emission-inputs* owner]
              (is (nil? (inference/infer-arg-tag symbol {}))))
            (is (identical? before @owner))))))))
