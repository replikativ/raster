(ns raster.compiler.backend.jvm.emission-dependency-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core]
            [raster.numeric]
            [raster.compiler.core.method-entry :as method-entry]
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
