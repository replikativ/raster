(ns raster.compiler.backend.jvm.emission-dependency-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core]
            [raster.numeric]
            [raster.ad.reverse :as reverse]
            [raster.ad.templates :as templates]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.jvm.hoisted :as hoisted]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftest actual-owned-emission-retries-changed-inputs-before-returning-code
  (let [scope (create-ns (gensym "raster.emission-retry."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double
                 (raster.numeric/* x x))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'square))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            _ (core/ensure-bytecode-implementation! receiver)
            {:keys [params tags return-tag body typed-interface]} (:raster.core/bytecode-emission (meta receiver))
            inputs (bytecode/emission-input-snapshots body scope)
            dependency (:var (first inputs))
            original (meta dependency)
            emitted (atom [])
            compile! (fn [change?]
                       (#'core/emit-owned-bytecode!
                        (fn [& args]
                          (let [impl (apply bytecode/compile-typed-impl! args)]
                            (swap! emitted conj impl)
                            (when (change? (count @emitted))
                              ;; The generated code is real; a consumed definition
                              ;; changes after emission but before publication.
                              (alter-meta! dependency update :raster.core/deftm-walked-body
                                           #(vector (list 'do (first %) 0.0))))
                            impl))
                        "raster.compiled.OwnedRetry" params tags return-tag body scope
                        typed-interface (:name (meta source))))]
        (is (var? dependency))
        (try
          (let [{:keys [implementation emission]} (compile! #(= 1 %))]
            (is (= 2 (count @emitted)))
            (is (not (identical? (first @emitted) implementation)))
            (is (identical? (last @emitted) implementation))
            (is (= 9.0 (implementation 3.0)))
            (is (identical? emission (method-entry/assert-emission-inputs-current! emission))))
          (finally (reset-meta! dependency original)))
        (reset! emitted [])
        (try
          (is (= :stale-emission-inputs
                 (try (compile! (constantly true)) :incorrectly-published
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= 3 (count @emitted)) "persistent change is bounded, not recertified")
          (finally (reset-meta! dependency original))))
      (finally (remove-ns (ns-name scope))))))

(deftest pipeline-diagnostics-do-not-invalidate-retained-gradients
  (let [scope (create-ns (gensym "raster.gradient-aot-diagnostics."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double
                 (raster.numeric/* x x))))
      (let [generic (ns-resolve scope 'square)
            selected (core/resolve-deftm-var generic)]
        (doseq [caller [generic selected]]
          (let [gradient (reverse/value+grad caller)
                root (var-get caller)
                observed (method-entry/emission-definition-snapshot caller)]
            (is (= [9.0 6.0] (vec (gradient 3.0))))
            (let [forward (pipeline/compile-aot caller :simd? false)]
              (is (= 9.0 (forward 3.0)))
              (is (contains? (meta caller) :raster.compiler.pipeline/compiled-diagnostics))
              (is (identical? root (var-get caller)))
              (is (method-entry/var-snapshot-current? observed))
              (is (= [9.0 6.0] (vec (gradient 3.0)))))
            (testing "other metadata remains a historical definition input"
              (let [original (meta caller)]
                (try
                  (alter-meta! caller assoc ::changed-definition true)
                  (is (not (method-entry/var-snapshot-current? observed)))
                  (is (thrown? clojure.lang.ExceptionInfo (gradient 3.0)))
                  (finally (reset-meta! caller original)))))
            (testing "changing actual source is not diagnostic publication"
              (let [original (meta selected)]
                (try
                  (alter-meta! selected assoc :raster.core/deftm-source-body '[42.0])
                  (is (thrown? clojure.lang.ExceptionInfo (gradient 3.0)))
                  (finally (reset-meta! selected original))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest typed-siblings-retain-distinct-namespace-identities
  (let [left (create-ns (gensym "raster.sibling-left."))
        right (create-ns (gensym "raster.sibling-right."))]
    (try
      (doseq [[scope constant] [[left 1.0] [right 10.0]]]
        (binding [*ns* scope]
          (refer 'clojure.core)
          (eval (list 'raster.core/deftm 'same-name ['x ':- 'Double] ':- 'Double
                      (list 'raster.numeric/+ 'x constant)))))
      (let [sources (mapv #(core/resolve-deftm-var (ns-resolve % 'same-name)) [left right])
            qualified (mapv #(symbol (str (ns-name (:ns (meta %))))
                                    (str (:name (meta %)))) sources)
            body [(list 'clojure.core/+
                        (list (first qualified) 'x)
                        (list 'clojure.core/* 10.0 (list (second qualified) 'x)))]
            {:keys [implementation emission]}
            (core/emit-owned-bytecode!
             (fn [class-name params tags return-tag body source-ns & _]
               (let [artifact (bytecode/compile-deftm-method!
                               class-name "run" params tags return-tag body source-ns)
                     method (:method artifact)]
                 (fn [value] (.invoke ^java.lang.reflect.Method method nil (object-array [value])))))
             "raster.compiled.DistinctNamespaceSiblings" '[x] '[double] 'double
             body left nil 'run)]
        (is (not (identical? (first sources) (second sources))))
        (is (= (name (first qualified)) (name (second qualified))))
        (is (= 123.0 (implementation 2.0)))
        (is (= 79.0 (implementation -2.0)))
        (doseq [source sources]
          (is (some #(identical? source (:var %)) (:dependency-snapshots emission)))
          (let [original (meta source)]
            (try
              (alter-meta! source assoc ::changed-sibling true)
              (is (= :stale-emission-inputs
                     (try (method-entry/assert-emission-inputs-current! emission) nil
                          (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
              (finally (reset-meta! source original))))))
      (finally
        (remove-ns (ns-name left))
        (remove-ns (ns-name right))))))

(deftest actual-root-and-static-emissions-retain-producer-inputs
  (let [scope (create-ns (gensym "raster.producer-emission."))
        current (atom true)
        producer (reify method-entry/RetainedProducerInput
                   (assert-producer-current! [this]
                     (when-not @current
                       (throw (ex-info "test producer changed" {:reason :test-stale-producer})))
                     this))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double
                 (raster.numeric/* x x))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'square))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            _ (core/ensure-bytecode-implementation! receiver)
            {:keys [params tags return-tag body typed-interface]}
            (:raster.core/bytecode-emission (meta receiver))
            retain! #(method-entry/adopt-definition-inputs! {:producer-inputs [producer]})
            {:keys [implementation emission]}
            (#'core/emit-owned-bytecode!
             (fn [& arguments] (retain!) (apply bytecode/compile-typed-impl! arguments))
             "raster.compiled.ProducerInput" params tags return-tag body scope
             typed-interface (:name (meta source)))
            compile-static bytecode/compile-deftm-method!
            artifact (with-redefs [bytecode/compile-deftm-method!
                                  (fn [& arguments] (retain!) (apply compile-static arguments))]
                       (#'bytecode/compile-and-cache-deftm! source))]
        (is (= 9.0 (implementation 3.0)))
        (is (= [producer] (:producer-inputs emission)))
        (is (= [producer] (get-in artifact [:raster.compiler/emission-inputs :producer-inputs])))
        (let [root-guard (method-entry/prepare-emission-input-guard! emission)
              static-guard (method-entry/prepare-emission-input-guard!
                            (:raster.compiler/emission-inputs artifact))]
          (reset! current false)
          (doseq [guard [root-guard static-guard]]
            (is (= :test-stale-producer
                   (try (guard) nil
                        (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest root-materialization-is-independent-of-an-unrelated-stale-analysis
  (let [scope (create-ns (gensym "raster.independent-materialization."))
        dependency (intern scope 'observed (Object.))
        parent (atom {:static [] :values []})]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double (raster.numeric/* x x))))
      (binding [method-entry/*emission-inputs* parent]
        (method-entry/observe-definition-input! dependency scope 'observed)
        (let [before @parent]
          (with-redefs-fn {dependency (Object.)}
            #(do
               (is (= 9.0 ((ns-resolve scope 'square) 3.0)))
               (is (identical? before @parent))
               (is (= :stale-definition-proof-input
                      (try (method-entry/assert-definition-inputs-current! @parent) nil
                           (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest emission-inputs-retain-transitive-static-source-observations
  (let [scope (create-ns (gensym "raster.emission-dependencies."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm grandchild [x :- Double] :- Double
                 (raster.numeric/* x x)))
        (eval '(raster.core/deftm child [x :- Double] :- Double (grandchild x)))
        (eval '(raster.core/deftm parent [x :- Double] :- Double (child x))))
      (let [grandchild (core/resolve-deftm-var (ns-resolve scope 'grandchild))
            child (core/resolve-deftm-var (ns-resolve scope 'child))
            parent (core/resolve-deftm-var (ns-resolve scope 'parent))
            _ (core/ensure-walked-body! grandchild)
            _ (core/ensure-walked-body! child)
            body (core/ensure-walked-body! parent)
            snapshots (bytecode/emission-input-snapshots body scope)
            observed (first (filter #(identical? child (:var %)) snapshots))]
        (is (some? observed))
        (is (some #(identical? grandchild (:var %)) snapshots))
        (is (every? #(method-entry/same-var-snapshot?
                     % (method-entry/var-snapshot (:var %))) snapshots))
        (with-redefs-fn {child (fn [& _] (throw (AssertionError. "must not invoke child")))}
          #(is (not (method-entry/same-var-snapshot? observed (method-entry/var-snapshot child)))))
        (is (method-entry/same-var-snapshot? observed (method-entry/var-snapshot child)))
        (let [original (meta child)]
          (try
            (alter-meta! child assoc :raster.core/deftm-walked-body '[999.0])
            (is (not (method-entry/same-var-snapshot? observed (method-entry/var-snapshot child))))
            (finally (reset-meta! child original)))))
      (finally (remove-ns (ns-name scope))))))

(deftest detached-vars-cannot-certify-replaced-emission-inputs
  (let [scope (create-ns (gensym "raster.detached-emission."))
        v (intern scope 'child (fn [] :old))
        observation (assoc (method-entry/var-snapshot v) :lookups #{[scope 'child]})
        emission {:dependency-snapshots [observation]
                  :static-inputs []
                  :definition-revision 10 :completed-definition-revision 10}
        reason #(try (method-entry/assert-emission-inputs-current! emission)
                     nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))]
    (try
      (is (method-entry/var-snapshot-current? observation))
      (ns-unmap scope 'child)
      (is (= :stale-emission-inputs (reason)))
      (intern scope 'child @v)
      ;; The detached original still compares equal to its saved observation.
      (is (method-entry/same-var-snapshot? observation (method-entry/var-snapshot v)))
      (is (= :stale-emission-inputs (reason)))
      (remove-ns (ns-name scope))
      (let [replacement (create-ns (ns-name scope))]
        (intern replacement 'child @v)
        (is (= :stale-emission-inputs (reason))))
      (finally (remove-ns (ns-name scope))))))

(deftest historical-emission-input-fence-does-not-bless-current-definitions
  (let [scope (create-ns (gensym "raster.historical-emission."))
        v (intern scope 'child (fn [] (throw (AssertionError. "must not execute child"))))
        emission {:dependency-snapshots [(assoc (method-entry/var-snapshot v) :lookups #{[scope 'child]})]
                  :static-inputs []
                  :definition-revision 10 :completed-definition-revision 10}
        reason #(try (method-entry/assert-emission-inputs-current! %)
                     nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))]
    (try
      (is (identical? emission (method-entry/assert-emission-inputs-current! emission)))
      (testing "root changes without an epoch bump are still stale"
        (with-redefs-fn {v (fn [] :new-definition)}
          #(is (= :stale-emission-inputs (reason emission))))
        (is (nil? (reason emission))))
      (testing "the emitter must supply observations, not a fresh empty default"
        (is (= :missing-emission-input-observations (reason (dissoc emission :dependency-snapshots))))
        (is (= :missing-emission-input-observations (reason (assoc emission :dependency-snapshots [{}]))))
        (is (= :missing-emission-input-observations
               (reason (assoc-in emission [:dependency-snapshots 0 :lookups] #{[:not-a-namespace 'child]})))))
      (testing "a change during emission is not repaired by matching current roots"
        (is (= :unstable-emission-inputs (reason (assoc emission :completed-definition-revision 11)))))
      (testing "metadata changes also invalidate historical source observations"
        (let [original (meta v)]
          (try
            (alter-meta! v assoc :raster.core/deftm-walked-body '[123.0])
            (is (= :stale-emission-inputs (reason emission)))
            (finally (reset-meta! v original)))))
      (testing "current cache identity is not historical cached-code provenance"
        (let [original (meta v)]
          (try
            (alter-meta! v assoc :raster.core/deftm-bytecode {:old-static-code true})
            (let [fresh (assoc emission :dependency-snapshots
                               [(assoc (method-entry/var-snapshot v) :lookups #{[scope 'child]})])]
              (is (nil? (reason fresh)) "an existing cache is not necessarily consumed")
              (is (= :unowned-static-emission-inputs
                     (reason (assoc fresh :static-inputs
                                    [{:source-var v :artifact (:raster.core/deftm-bytecode (meta v))}])))))
            (finally (reset-meta! v original)))))
      (finally (remove-ns (ns-name scope))))))

(deftest historical-inputs-retain-original-alias-and-refer-lookups
  (let [left (create-ns (gensym "raster.lookup-left."))
        right (create-ns (gensym "raster.lookup-right."))
        caller (create-ns (gensym "raster.lookup-caller."))]
    (try
      (doseq [scope [left right]]
        (binding [*ns* scope]
          (refer 'clojure.core)
          (eval '(raster.core/deftm child [x :- Double] :- Double (raster.numeric/* x x)))))
      (let [old (core/resolve-deftm-var (ns-resolve left 'child))
            replacement (core/resolve-deftm-var (ns-resolve right 'child))
            name (:name (meta old))
            assert-current (fn [inputs]
                             (method-entry/assert-emission-inputs-current!
                              {:dependency-snapshots inputs :static-inputs [] :definition-revision 0
                               :completed-definition-revision 0}))
            reason (fn [inputs] (try (assert-current inputs) nil
                                     (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
        (core/ensure-walked-body! old)
        (core/ensure-walked-body! replacement)
        (binding [*ns* caller]
          (alias 'h (ns-name left))
          (refer (ns-name left) :only [name]))
        (doseq [[lookup replace!]
                [[(symbol "h" (str name))
                  #(binding [*ns* caller] (ns-unalias caller 'h) (alias 'h (ns-name right)))]
                 [name #(do (ns-unmap caller name)
                            (.refer ^clojure.lang.Namespace caller name replacement))]]]
          (let [inputs (bytecode/emission-input-snapshots [(list lookup 'x)] caller)
                original (first (filter #(identical? old (:var %)) inputs))]
            (is (some? original))
            (is (nil? (reason inputs)))
            (replace!)
            (is (method-entry/same-var-snapshot? original (method-entry/var-snapshot old)))
            (is (= :stale-emission-inputs (reason inputs))))))
      (finally
        (doseq [scope [caller right left]] (remove-ns (ns-name scope)))))))

(deftest source-snapshots-distinguish-unbound-and-bound-nil
  (let [scope (create-ns (gensym "raster.unbound-emission."))
        v (intern scope 'unbound)]
    (try
      (let [before (method-entry/var-snapshot v)]
        (is (false? (:bound? before)))
        (is (method-entry/same-var-snapshot? before (method-entry/var-snapshot v)))
        (.bindRoot v nil)
        (is (not (method-entry/same-var-snapshot? before (method-entry/var-snapshot v)))))
      (finally (remove-ns (ns-name scope))))))

(deftest direct-live-observations-match-fresh-snapshot-semantics
  (let [scope (create-ns (gensym "raster.live-observation."))
        v (doto (intern scope 'value (Object.)) (.setDynamic))
        original-meta (meta v)
        original-root @v
        observations [(method-entry/var-snapshot v)
                      (method-entry/emission-definition-snapshot v)]
        assert-current! (fn [expected]
                          (doseq [[snapshot expected?] (map vector observations expected)]
                            (let [fresh (if (= :emission-definition (:snapshot-kind snapshot))
                                          (method-entry/emission-definition-snapshot v)
                                          (method-entry/var-snapshot v))]
                              (is (= expected?
                                     (method-entry/same-var-snapshot? snapshot fresh)
                                     (method-entry/var-value-current? snapshot))))))]
    (try
      (assert-current! [true true])
      (with-bindings {v (Object.)} (assert-current! [false false]))
      (alter-meta! v assoc :raster.core/deftm-bytecode {:unused :cache})
      (assert-current! [false true])
      (alter-meta! v assoc :changed true)
      (assert-current! [false false])
      (reset-meta! v original-meta)
      (.bindRoot v nil)
      (assert-current! [false false])
      (let [bound-nil (method-entry/var-snapshot v)]
        (is (method-entry/var-value-current? bound-nil))
        (.unbindRoot v)
        (is (not (method-entry/var-value-current? bound-nil)))
        (let [unbound (method-entry/var-snapshot v)]
          (is (method-entry/var-value-current? unbound))
          (.bindRoot v nil)
          (is (not (method-entry/var-value-current? unbound)))))
      (finally
        (.bindRoot v original-root)
        (reset-meta! v original-meta)
        (remove-ns (ns-name scope))))))

(deftest emission-guards-retain-process-local-producer-dependencies
  (let [current (atom true)
        checks (atom 0)
        producer (reify method-entry/RetainedProducerInput
                   (assert-producer-current! [this]
                     (swap! checks inc)
                     (when-not @current
                       (throw (ex-info "test producer changed" {:reason :test-stale-producer})))
                     this))
        emission {:dependency-snapshots [] :static-inputs [] :producer-inputs [producer]
                  :definition-revision 1 :completed-definition-revision 1}
        guard (method-entry/prepare-emission-input-guard! emission)]
    (is (identical? emission (guard)))
    (reset! checks 0)
    (guard)
    (is (= 1 @checks) "the prepared fast guard must not skip a producer dependency")
    (reset! current false)
    (doseq [invoke [guard #(method-entry/assert-emission-inputs-current! emission)]]
      (is (= :test-stale-producer
             (try (invoke) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))

(deftest prepared-emission-guards-preserve-historical-live-checks
  (let [scope (create-ns (gensym "raster.prepared-emission."))
        caller (create-ns (gensym "raster.prepared-emission-caller."))
        leaf (intern scope 'leaf (Object.))
        source (intern scope 'source (Object.))
        original-leaf-meta (meta leaf)
        observe #(assoc (method-entry/var-snapshot %) :lookups #{[scope (:name (meta %))]})
        base {:dependency-snapshots [] :static-inputs []
              :definition-revision 1 :completed-definition-revision 1}
        artifact {:raster.compiler/emission-inputs
                  (assoc base :dependency-snapshots [(observe leaf)]
                         :source-snapshot (method-entry/emission-definition-snapshot source))}
        emission (assoc base :static-inputs [{:source-var source :artifact artifact}
                                            {:source-var source :artifact artifact}])
        guard (method-entry/prepare-emission-input-guard! emission)
        reason (fn [invoke]
                 (try (invoke) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
        parity! (fn [expected]
                  (is (= expected (reason guard)))
                  (is (= expected (reason #(method-entry/assert-emission-inputs-current! emission)))))]
    (try
      (is (identical? emission (guard)))
      (with-redefs-fn {leaf (Object.)} #(parity! :stale-emission-inputs))
      (parity! nil)
      (try
        (alter-meta! leaf assoc :changed true)
        (parity! :stale-emission-inputs)
        (finally (reset-meta! leaf original-leaf-meta)))
      (let [original (meta source)]
        (try
          (alter-meta! source assoc :raster.core/deftm-bytecode {:unused :new-cache})
          (parity! nil) ;; The existing definition snapshot excludes cache output.
          (alter-meta! source assoc :changed true)
          (parity! :stale-static-emission-source)
          (finally (reset-meta! source original))))
      (testing "preparation does not accept stale or owner-mismatched lineage"
        (doseq [malformed [(dissoc emission :dependency-snapshots)
                           (assoc emission :dependency-snapshots [{}])
                           (assoc emission :completed-definition-revision 2)]]
          (is (= (reason #(method-entry/assert-emission-inputs-current! malformed))
                 (reason #(method-entry/prepare-emission-input-guard! malformed)))))
        (with-redefs-fn {leaf (Object.)}
          #(is (= :stale-emission-inputs
                  (reason (fn [] (method-entry/prepare-emission-input-guard! emission))))))
        (is (= :unowned-static-emission-inputs
               (reason #(method-entry/prepare-emission-input-guard!
                         (assoc-in emission [:static-inputs 1 :source-var] leaf))))))
      (testing "live effective thread bindings are not replaced by root-only checks"
        (let [dynamic-var (doto (intern scope 'dynamic-value (Object.)) (.setDynamic))
              inputs (assoc base :dependency-snapshots [(observe dynamic-var)])
              dynamic-guard (method-entry/prepare-emission-input-guard! inputs)]
          (with-bindings {dynamic-var (Object.)}
            (is (= :stale-emission-inputs (reason dynamic-guard))))
          (is (identical? inputs (dynamic-guard)))))
      (testing "alias lookup changes invalidate an otherwise unchanged Var"
        (binding [*ns* caller] (alias 'h (ns-name scope)))
        (let [inputs (assoc base :dependency-snapshots
                            [(assoc (method-entry/var-snapshot leaf) :lookups #{[caller 'h/leaf]})])
              lookup-guard (method-entry/prepare-emission-input-guard! inputs)]
          (is (identical? inputs (lookup-guard)))
          (ns-unalias caller 'h)
          (is (= :stale-emission-inputs (reason lookup-guard)))))
      (testing "a detached historical Var cannot bless a replacement"
        (ns-unmap scope 'leaf)
        (intern scope 'leaf @leaf)
        (parity! :stale-emission-inputs))
      (testing "recreating a namespace does not revive its old observations"
        (let [replacement-scope (create-ns (gensym "raster.prepared-replaced-ns."))
              original (intern replacement-scope 'value (Object.))
              inputs (assoc base :dependency-snapshots
                            [(assoc (method-entry/var-snapshot original)
                                    :lookups #{[replacement-scope 'value]})])
              namespace-guard (method-entry/prepare-emission-input-guard! inputs)]
          (try
            (remove-ns (ns-name replacement-scope))
            (intern (create-ns (ns-name replacement-scope)) 'value @original)
            (is (= :stale-emission-inputs (reason namespace-guard)))
            (finally (remove-ns (ns-name replacement-scope))))))
      (finally
        (remove-ns (ns-name caller))
        (remove-ns (ns-name scope))))))

(deftest static-cache-retains-actual-inputs-and-renews-a-stale-definition
  (let [scope (create-ns (gensym "raster.static-cache-lineage."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double (raster.numeric/* x x))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'square))
            cached (#'bytecode/compile-and-cache-deftm! source)
            invoke (fn [artifact] (.invoke ^java.lang.reflect.Method (:method artifact)
                                          nil (object-array [3.0])))
            reason (fn [f] (try (f) nil
                                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
        (is (= 9.0 (invoke cached)))
        (is (= (:body (:raster.compiler/emission-inputs cached)) (core/ensure-walked-body! source)))
        (is (identical? cached (method-entry/assert-static-emission-current! source cached)))
        (is (identical? cached (#'bytecode/current-cached-deftm! source cached)))
        (let [other (intern scope 'other @source)]
          (is (= :unowned-static-emission-inputs
                 (reason #(method-entry/assert-emission-inputs-current!
                           {:dependency-snapshots [] :definition-revision 0 :completed-definition-revision 0
                            :static-inputs [{:source-var source :artifact cached}
                                            {:source-var other :artifact cached}]})))
              "a memo hit must not bypass the artifact's source owner"))
        (let [legacy (dissoc cached :raster.compiler/emission-inputs)]
          (alter-meta! source assoc :raster.core/deftm-bytecode legacy)
          (is (= :unowned-static-emission-inputs
                 (reason #(method-entry/assert-static-emission-current! source legacy))))
          (let [renewed (#'bytecode/current-cached-deftm! source legacy)]
            (is (= 9.0 (invoke renewed)))
            (is (map? (:raster.compiler/emission-inputs renewed)))))
        (alter-meta! source assoc :raster.core/deftm-walked-body '[42.0]
                     :raster.core/deftm-walked-body-typed '[42.0])
        (is (= :stale-static-emission-source
               (reason #(method-entry/assert-static-emission-current! source cached))))
        (let [renewed (#'bytecode/current-cached-deftm! source cached)]
          (is (not (identical? cached renewed)))
          (is (= 42.0 (invoke renewed)))
          (is (= 9.0 (invoke cached)) "held original code is not mutated")
          (is (identical? renewed (:raster.core/deftm-bytecode (meta source))))
          (is (identical? renewed (method-entry/assert-static-emission-current! source renewed)))))
      (finally (remove-ns (ns-name scope))))))

(deftest static-ad-renewal-consumes-frozen-source-not-walk-or-runtime-caches
  (let [scope (create-ns (gensym "raster.static-ad-renewal."))
        rule-op 'raster.numeric/*
        original-rule (templates/get-template rule-op)
        macro-calls (atom 0)]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(defmacro frozen-square [x] (list 'raster.numeric/* x x)))
        (eval '(raster.core/deftm loss [x :- Double] :- Double (frozen-square x)))
        (eval '(raster.core/deftm derivative [x :- Double] :- Double
                 (let [vg ((raster.ad.reverse/value+grad (var loss) :wrt [0]) x)]
                   (clojure.core/nth vg 1))))
        (eval '(raster.core/deftm consumer [x :- Double] :- Double (derivative x)))
        (eval '(raster.core/deftm jit-consumer [x :- Double] :- Double (consumer x))))
      (let [loss (ns-resolve scope 'loss)
            source (core/resolve-deftm-var (ns-resolve scope 'derivative))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            _ (doseq [v [(core/resolve-deftm-var loss) source]] (core/prepare-deftm-frontend! v))
            frontend (method-entry/call-with-definition-inputs #(method-entry/original-source-frontend source))
            prepared (reverse/prepare-value+grad loss [0])
            invoke (fn [artifact] (.invoke ^java.lang.reflect.Method (:method artifact)
                                          nil (object-array [3.0])))
            reason (fn [f] (try (f) nil
                                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
            prepare! core/prepare-original-static-source!]
        ;; Poison ONLY a replaceable cache before the static source snapshot.
        ;; The exact old/new guards must therefore reach the real rule boundary.
        (alter-meta! source assoc :raster.core/deftm-walked-body '[42.0]
                     :raster.core/deftm-walked-body-typed '[42.0])
        (with-redefs-fn {(ns-resolve scope 'frozen-square)
                        (fn [& _] (swap! macro-calls inc)
                          (throw (ex-info "frozen macro was reexpanded" {})))}
          (fn []
            (let [old (#'bytecode/compile-and-cache-deftm! source {:prepare-source prepare!})
                  runtime (var-get receiver)]
              (is (= 6.0 (invoke old)))
              (templates/register-template!
               rule-op {:params '[a b] :result 'result :adjoint 'dy
                        :grads '[(raster.numeric/* dy (raster.numeric/* 2.0 b))
                                 (raster.numeric/* dy (raster.numeric/* 2.0 a))]})
              (is (= :ad-prepared-program-stale-rules
                     (reason #(method-entry/assert-static-emission-current! source old))))
              (is (= :ad-prepared-program-stale-rules
                     (reason #(#'bytecode/current-cached-deftm! source old))))
              (let [spec {:name 'probe :params [(with-meta 'x {:tag 'double})]
                          :return-tag 'double :source-ns scope
                          :walked-body [(list (symbol (str (ns-name scope))
                                                      (str (:name (meta source)))) 'x)]}
                    compile! #(bytecode/compile-specialized-class!
                               "raster.compiled.StaticADSibling" [spec] %)]
                (is (= :ad-prepared-program-stale-rules
                       (reason #(method-entry/call-with-definition-inputs (fn [] (compile! nil)))))
                    "direct sibling discovery cannot fall through to poisoned walked source")
                (let [sibling (method-entry/call-with-definition-inputs #(compile! prepare!))]
                  (is (= 12.0 (.invoke ^java.lang.reflect.Method (get-in sibling [:methods 'probe])
                                      nil (object-array [3.0]))))
                  (is (identical? old (:raster.core/deftm-bytecode (meta source)))
                      "a new direct sibling does not refresh an unused static artifact")))
              (doseq [[prepare expected]
                      [[(constantly nil) :unowned-static-renewal-source]
                       [(fn [_] {:frontend (assoc frontend :body '[42.0]) :body '[42.0]})
                        :unowned-checked-result-frontend]
                       [(fn [_] (throw (ex-info "preparation refused" {:reason :test-static-preparation})))
                        :test-static-preparation]
                       [(fn [v]
                          (let [prepared-source (prepare! v)]
                            (templates/register-template! rule-op (templates/get-template rule-op))
                            prepared-source))
                        :ad-prepared-program-stale-rules]]]
                (is (= expected
                       (reason #(method-entry/call-with-definition-inputs
                                 (fn [] (#'bytecode/current-cached-deftm! source old prepare))))))
                (is (identical? old (:raster.core/deftm-bytecode (meta source)))))
              (let [replacement (atom nil)
                    interpose (fn [v]
                                (when-not @replacement
                                  (reset! replacement
                                          (#'bytecode/compile-and-cache-deftm!
                                           v {:prepare-source prepare!})))
                                (prepare! v))]
                (is (= :stale-definition-proof-input
                       (reason #(binding [method-entry/*emission-inputs* (atom {:static [] :values []})]
                                  (#'bytecode/current-cached-deftm! source old interpose)))))
                (is (identical? @replacement (:raster.core/deftm-bytecode (meta source)))
                    "a reconstruction cannot overwrite a genuinely published intervening artifact")
                (is (= 12.0 (invoke @replacement))))
              ;; Equal public registration still advances the intentionally
              ;; conservative rule epoch; the next static acquisition renews.
              (templates/register-template! rule-op (templates/get-template rule-op))
              (let [attempts (atom 0)
                    fresh (method-entry/call-with-definition-inputs
                           (fn []
                             (swap! attempts inc)
                             (#'bytecode/current-cached-deftm!
                              source (:raster.core/deftm-bytecode (meta source)) prepare!)))]
                (is (= 2 @attempts))
                (is (not (identical? old fresh)))
                (is (= 12.0 (invoke fresh)) "the changed USED rule owns the new derivative")
                (is (identical? fresh (#'bytecode/current-cached-deftm! source fresh)))
                (is (identical? runtime (var-get receiver)) "static rebuild does not publish a runtime receiver")
                (is (= :ad-prepared-program-stale-rules
                       (reason #(method-entry/assert-static-emission-current! source old))))
                (is (= :ad-prepared-program-stale-rules
                       (reason #(reverse/assert-prepared-program-source! prepared))))
                (is (not= '[42.0] (get-in fresh [:raster.compiler/emission-inputs :body])))
                (let [compile-hoisted! hoisted/compile-hoisted-class!
                      callback-path (atom [])
                      compiled (with-redefs [hoisted/compile-hoisted-class!
                                             (fn [name specs options]
                                               (swap! callback-path conj
                                                      (identical? prepare! (:prepare-static-source options)))
                                               (compile-hoisted! name specs options))]
                                 (pipeline/compile-aot (ns-resolve scope 'consumer)
                                                       :inline? false :simd? false))]
                  (is (= 12.0 (compiled 3.0)))
                  (is (and (seq @callback-path) (every? true? @callback-path)))
                  (is (not-any? #(identical? prepare! %)
                                (tree-seq coll? seq (meta compiled)))
                      "the lexical preparer is not published as callable metadata"))
                (let [emit! bytecode/compile-typed-impl!
                      callbacks (atom [])
                      value (with-redefs [bytecode/compile-typed-impl!
                                          (fn [& arguments]
                                            (swap! callbacks conj (last arguments))
                                            (apply emit! arguments))]
                              ((ns-resolve scope 'jit-consumer) 3.0))]
                  (is (= 12.0 value))
                  (is (some #(identical? prepare! %) @callbacks)
                      "the named runtime compiler passes the same lexical static preparer"))
                (is (zero? @macro-calls)))))))
      (finally
        (templates/register-template! rule-op original-rule)
        (remove-ns (ns-name scope))))))

(deftest original-static-source-keeps-its-policy-under-opposite-ambient-math
  (let [scope (create-ns (gensym "raster.static-source-policy."))
        outcome (fn [^java.lang.reflect.Method method]
                  (try {:value (.invoke method nil (object-array [Long/MAX_VALUE]))}
                       (catch java.lang.reflect.InvocationTargetException failure
                         {:error (class (.getCause failure))})))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (doseq [[name policy] [['checked false] ['unchecked true]]]
          (binding [*unchecked-math* policy]
            (eval (list 'raster.core/deftm name '[x :- Long] ':- 'Long
                        (if policy '(clojure.core/unchecked-add x 1)
                            '(clojure.core/+ x 1)))))))
      (doseq [[name policy expected] [['checked false {:error ArithmeticException}]
                                     ['unchecked true {:value Long/MIN_VALUE}]]]
        (let [source (core/resolve-deftm-var (ns-resolve scope name))
              _ (core/prepare-deftm-frontend! source)
              prepare! core/prepare-original-static-source!
              observed-policy (atom [])
              emit-body-var (ns-resolve 'raster.compiler.backend.jvm.bytecode 'emit-body)
              emit-body @emit-body-var
              spec {:name 'probe :params [(with-meta 'x {:tag 'long})]
                    :return-tag 'long :source-ns scope
                    :walked-body [(list (symbol (str (ns-name scope))
                                                (str (:name (meta source)))) 'x)]}
              [artifact sibling]
              (with-redefs-fn
                {emit-body-var
                 (fn [code body locals context]
                   (when (and (identical? scope (:source-ns context))
                              (some '#{clojure.core/+ clojure.core/unchecked-add}
                                    (tree-seq coll? seq body)))
                     (swap! observed-policy conj *unchecked-math*))
                   (emit-body code body locals context))}
                #(binding [*unchecked-math* (not policy)]
                   [(#'bytecode/compile-and-cache-deftm! source {:prepare-source prepare!})
                    (method-entry/call-with-definition-inputs
                     (fn [] (bytecode/compile-specialized-class!
                             "raster.compiled.StaticPolicySibling" [spec] prepare!)))]))]
          ;; Checked and explicit unchecked operators retain distinct overflow
          ;; behavior. Independently observe the declaration's policy INSIDE
          ;; both real emitters, not merely at the callback/AD-expansion seam.
          (is (= expected (outcome (:method artifact))))
          (is (= expected (outcome (get-in sibling [:methods 'probe]))))
          (is (and (<= 2 (count @observed-policy))
                   (every? #(= policy %) @observed-policy)))
          (is (= policy (get-in (method-entry/call-with-definition-inputs
                                #(method-entry/original-source-frontend source))
                               [:source-entry :source-policy :unchecked-math])))))
      (finally (remove-ns (ns-name scope))))))

(deftest static-parent-tracks-consumed-grandchild-not-unused-sibling-cache
  (let [scope (create-ns (gensym "raster.static-grandchild."))
        loader (clojure.lang.DynamicClassLoader. (.getContextClassLoader (Thread/currentThread)))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm grandchild [x :- Double] :- Double (raster.numeric/* x x)))
        (eval '(raster.core/deftm child [x :- Double] :- Double (grandchild x)))
        (eval '(raster.core/deftm parent [x :- Double] :- Double (child x))))
      (let [grandchild (core/resolve-deftm-var (ns-resolve scope 'grandchild))
            child (core/resolve-deftm-var (ns-resolve scope 'child))
            parent (core/resolve-deftm-var (ns-resolve scope 'parent))
            qualified #(symbol (str (ns-name scope)) (str (:name (meta %))))
            invoke (fn [artifact] (.invoke ^java.lang.reflect.Method (:method artifact)
                                          nil (object-array [3.0])))]
        ;; Retained direct typed calls are supported bytecode input. The parent
        ;; rebuilds child as a same-class sibling; that sibling's grandchild
        ;; reaches the external static-cache consumer (not another sibling).
        (doseq [[v callee] [[child grandchild] [parent child]]]
          (let [body [(list (qualified callee) 'x)]]
            (alter-meta! v assoc :raster.core/deftm-walked-body body
                         :raster.core/deftm-walked-body-typed body)))
        (binding [method-entry/*compilation-classloader* loader]
          (let [old (#'bytecode/compile-and-cache-deftm! grandchild)]
            (is (= 9.0 (invoke old)))
            (alter-meta! grandchild assoc :raster.core/deftm-walked-body '[42.0]
                         :raster.core/deftm-walked-body-typed '[42.0])
            ;; Existing legacy child cache is NOT used: child is rebuilt as a
            ;; sibling. It must not cause three futile provenance retries.
            (alter-meta! child assoc :raster.core/deftm-bytecode {:unused-legacy-cache true})
            (let [artifact (#'bytecode/compile-and-cache-deftm! parent)
                  inputs (:raster.compiler/emission-inputs artifact)]
              (is (= 42.0 (invoke artifact)))
              (is (= 9.0 (invoke old)))
              (is (some #(identical? grandchild (:source-var %)) (:static-inputs inputs)))
              (is (not-any? #(identical? child (:source-var %)) (:static-inputs inputs)))
              (is (identical? artifact (method-entry/assert-static-emission-current! parent artifact)))
              (alter-meta! grandchild assoc :raster.core/deftm-walked-body '[43.0]
                           :raster.core/deftm-walked-body-typed '[43.0])
              (is (= :stale-emission-inputs
                     (try (method-entry/assert-static-emission-current! parent artifact) nil
                          (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest static-cache-retains-the-numeric-roots-actually-folded-into-code
  (let [scope (create-ns (gensym "raster.static-numeric-input."))
        coefficient (intern scope 'coefficient 3.0)]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm scaled [x :- Double] :- Double
                 (raster.numeric/* coefficient x))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'scaled))
            cached (#'bytecode/compile-and-cache-deftm! source)
            invoke (fn [artifact] (.invoke ^java.lang.reflect.Method (:method artifact)
                                          nil (object-array [2.0])))
            inputs (:raster.compiler/emission-inputs cached)]
        (is (= 6.0 (invoke cached)))
        (is (some #(and (identical? coefficient (:var %)) (= 3.0 (:value %)))
                  (:dependency-snapshots inputs)))
        (alter-var-root coefficient (constantly 4.0))
        (is (= :stale-emission-inputs
               (try (method-entry/assert-static-emission-current! source cached) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (let [renewed (#'bytecode/current-cached-deftm! source cached)]
          (is (= 8.0 (invoke renewed)))
          (is (= 6.0 (invoke cached)))
          (is (identical? renewed (method-entry/assert-static-emission-current! source renewed))))
        (.setDynamic ^clojure.lang.Var coefficient)
        (is (= :thread-bound-emission-constant
               (with-bindings {coefficient 5.0}
                 (try (#'bytecode/compile-and-cache-deftm! source) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
      (finally (remove-ns (ns-name scope))))))
