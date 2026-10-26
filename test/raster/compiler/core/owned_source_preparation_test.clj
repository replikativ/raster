(ns raster.compiler.core.owned-source-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.method-entry :as entry]
            [raster.core :as core]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest original-source-demand-closes-a-cold-diamond-without-execution
  (let [scope (create-ns (gensym "raster.original-source-readiness."))
        events (atom [])
        chain '[level0 level1 level2 level3]]
    (try
      (intern scope 'events events)
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm leaf [x :- Double] :- Double
                 (do (swap! events conj :leaf) x)))
        (doseq [[name child] (reverse (map vector chain (conj (subvec chain 1) 'leaf)))]
          (eval (list 'raster.core/deftm name '[x :- Double] ':- 'Double (list child 'x))))
        (doseq [name '[left right]]
          (eval (list 'raster.core/deftm name '[x :- Double] ':- 'Double '(level0 x))))
        (eval '(raster.core/deftm unused [x :- Double] :- Double x))
        (eval '(raster.core/deftm caller [x :- Double] :- Double
                 (clojure.core/+ (left x) (right x)))))
      (let [source #(core/resolve-deftm-var (ns-resolve scope %))
            receiver #(ns-resolve scope (symbol (str (:name (meta (source %))) "-impl")))
            caller (source 'caller)
            root (receiver 'caller)
            descendants (mapv receiver (concat '[leaf] chain '[left right]))
            unused (receiver 'unused)
            before @root
            ;; This is exactly the source-only frontier used before target
            ;; observations. Publishing the frozen walk does not emit a receiver.
            _ (core/prepare-deftm-frontend! caller)
            _ (core/prepare-emission-source-walk! caller)
            _ (is (identical? before @root))
            _ (is (every? #(map? (:raster.core/jit-spec (meta %))) descendants))
            attempts (atom 0)
            {:keys [frontend inputs]}
            (entry/call-with-definition-inputs
             #(do (swap! attempts inc)
                  (let [frontend (core/prepare-owned-source-frontend! caller)]
                    {:frontend frontend :inputs @entry/*emission-inputs*})))
            implementation @root
            emission (:raster.core/bytecode-emission (meta root))]
        (is (<= @attempts 3) "cold depth does not expand the whole-owner retry policy")
        (is (empty? @events))
        (is (identical? implementation (:implementation frontend)))
        (is (identical? emission (:emission frontend)))
        (is (every? #(nil? (:raster.core/jit-spec (meta %))) descendants))
        (is (map? (:raster.core/jit-spec (meta unused))))
        (is (true? (entry/assert-definition-inputs-current! inputs)))
        (entry/call-with-definition-inputs #(core/prepare-owned-source-frontend! caller))
        (is (identical? implementation @root))
        (is (identical? emission (:raster.core/bytecode-emission (meta root))))
        (doseq [[label changed change]
                [[:original-source caller
                  #(update % :raster.core/deftm-source-body
                           (fn [body] (with-meta (into [] body) (meta body))))]
                 ;; Readiness retains semantic receiver selection, not arbitrary
                 ;; receiver metadata. Actual physical consumers separately pin
                 ;; its complete observation before deriving code facts.
                 [:receiver-policy root #(update % :no-inline not)]
                 [:actual-leaf-history (receiver 'leaf) #(assoc % ::changed (Object.))]]]
          (let [metadata (meta changed)]
            (try
              (alter-meta! changed change)
              (is (thrown? clojure.lang.ExceptionInfo
                           (entry/assert-definition-inputs-current! inputs))
                  (str "retained readiness rejects " label " change to " changed))
              (finally (reset-meta! changed metadata)))))
        (try
          (alter-var-root root (constantly (Object.)))
          (is (thrown? clojure.lang.ExceptionInfo
                       (entry/assert-definition-inputs-current! inputs))
              "a replaced root implementation cannot retain its original semantic selection")
          (finally (alter-var-root root (constantly implementation))))
        (is (true? (entry/assert-definition-inputs-current! inputs)))
        (is (empty? @events)))
      (finally (remove-ns (ns-name scope))))))

(deftest preparation-does-not-adopt-readiness-around-an-old-physical-observation
  (let [scope (create-ns (gensym "raster.original-source-stale."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm caller [x :- Double] :- Double (clojure.core/+ x 1.0))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'caller))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            collector (atom {:static [] :values []})]
        (core/prepare-deftm-frontend! source)
        (core/prepare-emission-source-walk! source)
        (is (= :missing-definition-preparation-owner
               (reason #(core/prepare-owned-source-frontend! source))))
        (binding [entry/*emission-inputs* collector]
          (let [frontend (entry/original-source-frontend source)
                _ (entry/frontend-body-entry-carriers frontend source)
                historical @collector
                original @receiver]
            (is (= :stale-definition-proof-input
                   (reason #(core/prepare-owned-source-frontend! source))))
            (is (not (identical? original @receiver)) "preparation published a genuinely new module")
            (is (= historical @collector)
                "a failed historical fence adopts no descendant readiness receipt")
            (is (= :stale-definition-proof-input
                   (reason #(entry/assert-definition-inputs-current! historical)))))))
      (finally (remove-ns (ns-name scope))))))
