(ns raster.compiler.backend.jvm.frontend-reduction-correspondence-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.numeric :as numeric]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm source-sum [xs :- (Array double) count :- Long] :- Double
  (par/reduce acc 0.0 i count (+ acc (aget xs i))))

(deftm source-float-sum [xs :- (Array float) count :- Long] :- Float
  (par/reduce acc (float 0.0) i count (+ acc (aget xs i))))

(deftm source-binding [n :- Long] :- Long
  (let [size (numeric/* n 2)] (long size)))

(defonce ^:private cold-source-evaluations (atom 0))

(deftm cold-sum [xs :- (Array double) count :- Long] :- Double
  (swap! cold-source-evaluations inc)
  (par/reduce acc 0.0 i count (+ acc (aget xs i))))

(use-fixtures :each
  (fn [run]
    (binding [entry/*emission-inputs* (atom {:static [] :values []})]
      (run))))

(defn- receiver [source]
  (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl"))))

(defn- captured [function]
  (let [source (core/resolve-deftm-var function)]
    (core/ensure-current-owned-bytecode! (receiver source))
    (entry/original-source-frontend source)))

(defn- reduction-forms [body]
  (filter par-ir/par-reduce-form? (tree-seq coll? seq body)))

(deftest frozen-source-occurrence-reaches-only-its-own-emitted-reduction
  (doseq [[function payload] [[#'source-sum Double] [#'source-float-sum Float]]]
    (let [frontend (captured function)
          original (first (reduction-forms (:body frontend)))
          observed (entry/frontend-reduction-emission frontend original)
          emitted-source (first (vals (:reduction-catalog (:emission frontend))))]
      (is (some? observed))
      (is (identical? original (:source-input observed)))
      (is (not (identical? original (:source observed))))
      (is (= {:carrier Object :presence :present :builtin-class payload}
             (:result-fact observed)))
      (is (nil? (entry/frontend-reduction-emission
                 frontend (with-meta (apply list original) (meta original)))))
      (is (nil? (entry/frontend-reduction-emission frontend (assoc (vec original) 2 7.0))))
      (is (nil? (entry/frontend-reduction-emission frontend emitted-source))))))

(deftest input-body-and-output-body-identities-are-independent-fences
  (let [frontend (captured #'source-sum)
        original (first (reduction-forms (:body frontend)))
        emission (:emission frontend)]
    (is (nil? (bytecode/retained-input-reduction-emission
               emission (with-meta (vec (map identity (:body frontend))) (meta (:body frontend))) original)))
    (is (nil? (bytecode/retained-input-reduction-emission
               (assoc emission :body (with-meta (vec (map identity (:body emission))) (meta (:body emission))))
               (:body frontend) original)))
    (is (nil? (bytecode/retained-input-reduction-emission
               (dissoc emission :input-origins) (:body frontend) original)))
    (is (nil? (bytecode/retained-input-reduction-emission
               (assoc emission :input-origins {}) (:body frontend) original)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (entry/frontend-reduction-emission
                  (assoc frontend :body (vec (map identity (:body frontend)))) original)))
    (is (nil? (entry/frontend-reduction-emission
               (assoc frontend :emission (assoc emission :implementation (Object.))) original)))
    (is (nil? (entry/frontend-reduction-emission
               (assoc frontend :emission (assoc emission :frontend-source-owner (Object.))) original)))
    (let [changed (into {} emission)]
      (is (= emission changed))
      (is (not (identical? emission changed)))
      (is (nil? (entry/frontend-reduction-emission (assoc frontend :emission changed) original))))
    (is (nil? (binding [entry/*emission-inputs* nil]
                (entry/frontend-reduction-emission frontend original))))))

(deftest equal-distinct-occurrences-and-repeated-object-occurrences-do-not-collapse
  ;; Isolate the identity relation from physical emission. Actual emission and
  ;; source-owner guards are exercised by the preceding and following tests.
  (let [left '(raster.par/reduce acc 0.0 i count (+ acc i))
        right (apply list left)
        body [left right]
        owned (bytecode/retain-invocation-occurrences body)
        emission {:body (:body owned) :input-origins (:input-origins owned)}]
    (is (= left right))
    (is (not (identical? left right)))
    (with-redefs [bytecode/retained-reduction-emission (fn [_ source] {:source source})]
      (let [left-result (bytecode/retained-input-reduction-emission emission body left)
            right-result (bytecode/retained-input-reduction-emission emission body right)]
        (is (identical? left (:source-input left-result)))
        (is (identical? right (:source-input right-result)))
        (is (not (identical? (:source left-result) (:source right-result)))))
      (let [ambiguous-body [left left]
            ambiguous (bytecode/retain-invocation-occurrences ambiguous-body)]
        (is (nil? (bytecode/retained-input-reduction-emission
                   {:body (:body ambiguous) :input-origins (:input-origins ambiguous)}
                   ambiguous-body left)))))))

(deftest bodies-without-executable-occurrences-retain-no-input-origin-owner
  (is (nil? (:input-origins
             (bytecode/retain-invocation-occurrences '[42 (quote (+ x 1.0))])))))

(deftest original-call-and-binding-share-origin-ownership-not-physical-inference
  (let [frontend (captured #'source-binding)
        scope (first (:body frontend))
        original (nth (second scope) 1)
        emission (:emission frontend)
        binding-observation (entry/frontend-binding-context frontend scope 0)
        invocation (entry/frontend-invocation-context frontend original)]
    (is (some? binding-observation))
    (is (identical? scope (:source-input binding-observation)))
    ;; The selected interface really returns Object. Its Long source hint is
    ;; not permission to report a primitive result or insert an unboxing.
    (is (= :ref (get-in binding-observation [:after :type])))
    (is (= Object (:return-carrier invocation)))
    (is (identical? original (:source-input invocation)))
    (is (nil? (entry/frontend-binding-context frontend scope 1)))
    (is (nil? (entry/frontend-binding-context frontend original 0)))
    (is (nil? (entry/frontend-invocation-context frontend scope)))
    (is (nil? (entry/frontend-reduction-emission frontend original)))
    (doseq [[query node] [[#(entry/frontend-invocation-context %1 %2) original]
                         [#(entry/frontend-binding-context %1 %2 0) scope]]]
      (is (nil? (query frontend (with-meta (apply list node) (meta node)))))
      (is (nil? (query (assoc frontend :emission (into {} emission)) node)))
      (is (nil? (query (assoc frontend :emission (assoc emission :input-origins {})) node)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (query (assoc-in frontend [:definition :receiver-var] (receiver
                            (core/resolve-deftm-var #'source-sum))) node)))
      (is (nil? (binding [entry/*emission-inputs* nil] (query frontend node)))))))

(deftest call-and-binding-origin-identity-fences
  (doseq [[left query physical]
          [['(+ n 2) bytecode/retained-input-invocation-context
            #'bytecode/retained-invocation-context]
           ['(let* [size (+ n 2)] size)
            #(bytecode/retained-input-binding-context %1 %2 %3 0)
            #'bytecode/retained-binding-context]]]
    (let [right (apply list left)
          body [left right]
          owned (bytecode/retain-invocation-occurrences body)
          emission {:body (:body owned) :input-origins (:input-origins owned)}]
      (with-redefs-fn {physical (fn [_ source & _] {:source source})}
        (fn []
          (let [a (query emission body left) b (query emission body right)]
            (is (some? a))
            (is (identical? left (:source-input a)))
            (is (identical? right (:source-input b)))
            (is (not (identical? (:source a) (:source b)))))
          (is (nil? (query emission (mapv identity body) left)))
          (is (nil? (query (assoc emission :body (mapv identity (:body owned))) body left)))
          (is (nil? (query (dissoc emission :input-origins) body left)))
          (is (nil? (query (assoc emission :input-origins {}) body left)))
          (let [body [left left]
                owned (bytecode/retain-invocation-occurrences body)]
            (is (nil? (query {:body (:body owned) :input-origins (:input-origins owned)} body left))))
          (let [body [(list 'quote left)]
                owned (bytecode/retain-invocation-occurrences body)]
            (is (nil? (query {:body (:body owned) :input-origins (:input-origins owned)} body left)))))))))

(deftest edited-observations-cannot-acquire-the-original-frontend-authority
  (let [frontend (captured #'source-sum)
        original (first (reduction-forms (:body frontend)))
        emission (:emission frontend)
        observed (entry/frontend-reduction-emission frontend original)
        witness (get-in observed [:expansion :witness])
        carry (second (get-in observed [:loop :variables]))
        changed (-> emission
                    (assoc-in [:reduction-contexts [witness :loop] :locals carry :type] :float)
                    (assoc-in [:reduction-contexts [witness :recur] :targets 1 :type] :float)
                    (assoc-in [:reduction-contexts [witness :exit] :local :type] :float)
                    (assoc-in [:reduction-contexts [witness :exit] :input-type] :float))]
    ;; The low-level observer assumes emitter-owned records. The public owner
    ;; boundary must authenticate the ENTIRE map, not just its opaque bodies.
    (is (= Float (get-in (bytecode/retained-input-reduction-emission
                          changed (:body frontend) original) [:result-fact :builtin-class])))
    (is (nil? (entry/frontend-reduction-emission (assoc frontend :emission changed) original)))
    (is (= Double (get-in (entry/frontend-reduction-emission frontend original)
                         [:result-fact :builtin-class])))))

(deftest cold-query-does-not-materialize-a-jvm-emission
  (let [source (core/resolve-deftm-var #'cold-sum)
        implementation (receiver source)
        _ (core/prepare-deftm-frontend! source)
        before (var-get implementation)
        evaluations @cold-source-evaluations
        frontend (with-redefs [core/do-bytecode-upgrade!
                              (fn [& _] (throw (ex-info "cold source query emitted code" {})))]
                   (entry/original-source-frontend source))
        original (first (reduction-forms (:body frontend)))]
    (is (some? frontend))
    (is (= evaluations @cold-source-evaluations))
    (is (some? (:raster.core/jit-spec (meta implementation))))
    (is (nil? (:emission frontend)))
    (is (nil? (entry/frontend-reduction-emission frontend original)))
    (doseq [expression (tree-seq coll? seq (:body frontend)) :when (seq? expression)]
      (is (nil? (entry/frontend-invocation-context frontend expression)))
      (is (nil? (entry/frontend-binding-context frontend expression 0))))
    (is (identical? before (var-get implementation)))
    (is (nil? (:raster.core/bytecode-emission (meta implementation))))))

(deftest source-owner-staleness-is-not-repaired-by-the-origin-relation
  (let [frontend (captured #'source-sum)
        original (first (reduction-forms (:body frontend)))
        source (get-in frontend [:source-entry :source-var])
        metadata (meta source)]
    (try
      (alter-meta! source assoc :raster.core/source-policy {:unchecked-math true})
      (is (thrown? clojure.lang.ExceptionInfo
                   (entry/frontend-reduction-emission frontend original)))
      (finally (reset-meta! source metadata)))
    (is (some? (entry/frontend-reduction-emission frontend original)))))

(deftest original-call-and-binding-history-cannot-be-refreshed
  (let [frontend (captured #'source-binding)
        scope (first (:body frontend))
        original (nth (second scope) 1)
        source (get-in frontend [:source-entry :source-var])
        metadata (meta source)]
    (try
      (alter-meta! source assoc :raster.core/source-policy {:unchecked-math true})
      (is (thrown? clojure.lang.ExceptionInfo
                   (entry/frontend-invocation-context frontend original)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (entry/frontend-binding-context frontend scope 0)))
      (finally (reset-meta! source metadata)))
    (is (some? (entry/frontend-invocation-context frontend original)))
    (is (some? (entry/frontend-binding-context frontend scope 0)))))

(deftest successful-physical-observations-retain-transitive-module-history
  (let [frontend (captured #'source-binding)
        scope (first (:body frontend))
        original (nth (second scope) 1)
        dependency (some #(when (= #'numeric/_star__m_long_long (:var %)) (:var %))
                         (get-in frontend [:emission :dependency-snapshots]))
        parent (atom {:static [] :values []})]
    (is (var? dependency) "use a real transitive input of the emitted caller")
    (binding [entry/*emission-inputs* parent]
      (is (some? (entry/frontend-invocation-context frontend original)))
      (let [producer (first (:producer-inputs @parent))]
        (is (satisfies? entry/RetainedProducerInput producer))
        (is (some? (entry/frontend-binding-context frontend scope 0)))
        (is (= 1 (count (:producer-inputs @parent))))
        (is (identical? producer (first (:producer-inputs @parent))))))
    ;; The query has returned and the request-local collector is no longer bound.
    ;; A later consumer must still reject changes below the caller's selection.
    (let [historical @parent metadata (meta dependency)]
      (is (not (contains? (:definition-proofs historical) dependency)))
      (is (entry/assert-definition-inputs-current! historical))
      (try
        (alter-meta! dependency assoc ::changed-after-query true)
        (is (= :stale-emission-inputs
               (try (entry/assert-definition-inputs-current! historical) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
        (finally (reset-meta! dependency metadata)))
      (is (entry/assert-definition-inputs-current! historical)))))

(deftest refused-physical-observations-retain-no-speculative-inputs
  (let [frontend (captured #'source-binding)
        scope (first (:body frontend)) original (nth (second scope) 1)
        parent (atom {:static [] :values []}) before @parent]
    (binding [entry/*emission-inputs* parent]
      (is (nil? (entry/frontend-invocation-context frontend (apply list original))))
      (is (identical? before @parent))
      (is (nil? (entry/frontend-binding-context frontend scope 1)))
      (is (identical? before @parent))
      (is (nil? (entry/frontend-invocation-context
                 (assoc frontend :emission (into {} (:emission frontend))) original)))
      (is (identical? before @parent)))))

(deftest reduction-observation-uses-the-same-historical-producer-contract
  (let [frontend (captured #'source-sum)
        original (first (reduction-forms (:body frontend)))
        parent (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* parent]
      (is (some? (entry/frontend-reduction-emission frontend original)))
      (is (= 1 (count (:producer-inputs @parent))))
      (is (entry/assert-definition-inputs-current! @parent)))))
