(ns raster.compiler.backend.jvm.completion-context-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.normal-return :as normal-return]))

(defn- emit [body]
  (let [interface (types/ensure-fn-interface! ['Object])]
    (core/emit-owned-bytecode!
     bytecode/compile-typed-impl! "raster.compiled.CompletionContext"
     '[x] ['Object] 'Object body *ns* (:iface-name interface) 'completion-context)))

(defn- observe [emission body expression]
  (bytecode/retained-input-completion-context emission body expression))

(deftest actual-completion-carriers-are-distinct-from-object-parent-boxing
  (let [context (completion/bytecode-operand-context {})]
    (doseq [[contract operand raw completed input expected]
            [[completion/object-input-contract 'x :ref :ref "value" "value"]
             [(completion/reference-check-contract String) 'x :ref :ref "value" "value"]
             [(completion/bytecode-reference-input-contract String context) 'x :ref :ref "value" "value"]
             [(completion/bytecode-integral-input-contract Long/TYPE context) 'x :ref :long 3.75 3]
             [(completion/bytecode-floating-input-contract Float/TYPE context) 'x :ref :float 3.75 (float 3.75)]
             [(completion/bytecode-floating-input-contract Double/TYPE context) 'x :ref :double 3.75 3.75]
             [(completion/bytecode-result-context-contract context :double) 1.25 :double :double nil 1.25]
             [(completion/bytecode-result-context-contract context :int) (int 7) :int :int nil (int 7)]
             [(completion/bytecode-result-context-contract context :bool) true :bool :bool nil true]
             [(completion/bytecode-result-context-contract context :ref) 'x :ref :ref "value" "value"]]
            object-parent? [false true]]
      (let [expression (completion/complete-value contract operand)
            root (if object-parent?
                   (completion/complete-value completion/object-input-contract expression)
                   expression)
            ;; A binding initializer uses emit-form. A top-level Object return
            ;; may itself select Object consumption before reaching the node.
            body [(list 'let* ['result root] 'result)]
            {:keys [implementation emission]} (emit body)
            observed (observe emission body expression)]
        (is (= expected (implementation input)))
        (is (= raw (:input-carrier observed)))
        (is (= completed (:completion-carrier observed)))
        (is (= (if object-parent? :ref completed) (:returned-carrier observed)))
        (is (= object-parent? (:object-parent? observed)))
        (if (= :ref completed)
          (is (nil? (:result-fact observed)) "a reference does not prove a nonnull scalar payload")
          (let [tag (if (= :bool completed) 'boolean (symbol (name completed)))
                carrier (types/tag->java-class tag)]
            (is (= {:carrier (if object-parent? Object carrier)
                    :presence :present :builtin-class (get types/primitive-info tag)}
                   (:result-fact observed)))))
        (is (identical? expression (:source-input observed)))
        (is (some #(identical? % (:source observed)) (vals (:completion-catalog emission))))
        (is (nil? (:reference-class observed)) "a ref stack carrier is not an exact runtime Class")
        (is (nil? (bytecode/retained-input-invocation-context emission body expression))
            "a canonical completion is not a JVM invocation")
        (when object-parent?
          (is (some? (observe emission body root)) "nested completions retain separate owners"))))))

(deftest occurrence-ownership-rejects-copies-and-shared-source-occurrences
  (let [expression (completion/complete-value completion/object-input-contract 'x)
        body [expression]
        {:keys [emission]} (emit body)
        copied (with-meta (apply list expression) (meta expression))]
    (is (some? (observe emission body expression)))
    (is (nil? (observe emission (vec (map identity body)) expression)))
    (is (nil? (observe emission body copied)))
    (let [duplicate-body [(list 'do expression expression)]
          duplicate (:emission (emit duplicate-body))]
      (is (nil? (observe duplicate duplicate-body expression))))
    (let [quoted-body [(list 'quote expression)]
          quoted (:emission (emit quoted-body))]
      (is (empty? (:completion-catalog quoted)))
      (is (nil? (observe quoted quoted-body expression))))))

(deftest stale-completion-evidence-is-not-refreshed-by-querying
  (let [expression (completion/complete-value completion/object-input-contract 'x)
        body [expression]
        {:keys [emission]} (emit body)]
    (with-redefs [entry/assert-emission-inputs-current!
                  (fn [_] (throw (ex-info "stale test owner" {:reason :stale-definition-proof-input})))]
      (is (= :stale-definition-proof-input
             (try (observe emission body expression) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest exact-nonbootstrap-array-class-is-retained-without-inference
  (let [donor (:implementation (emit ['x]))
        array (java.lang.reflect.Array/newInstance (class donor) 0)
        carrier (class array)
        expression (completion/complete-value (completion/reference-check-contract carrier) 'x)
        body [expression]
        {:keys [implementation emission]} (emit body)
        observed (observe emission body expression)]
    (is (some? (.getClassLoader carrier)))
    (is (identical? carrier (get-in observed [:contract :target-carrier])))
    (is (= :ref (:input-carrier observed) (:completion-carrier observed)))
    (is (identical? array (implementation array)))
    (is (nil? (implementation nil)))
    (is (thrown? ClassCastException (implementation (object-array 0))))))

(deftest failed-completion-emission-publishes-no-successful-observation
  (let [compile bytecode/compile-typed-impl!
        inputs (atom nil)
        expression (completion/complete-value
                    (completion/bytecode-result-context-contract
                     (completion/bytecode-operand-context {}) :double) 'x)]
    (with-redefs [bytecode/compile-typed-impl!
                  (fn [& arguments]
                    (reset! inputs entry/*emission-inputs*)
                    (apply compile arguments))]
      (is (= :bytecode-result-context-carrier-mismatch
             (try (emit [expression]) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (some? @inputs))
    (is (empty? (:completion-contexts @@inputs))))
  (let [malformed (list completion/completion-symbol {:source-context :unknown} 'x)]
    (is (= :invalid-value-completion
           (try (emit [malformed]) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest divergence-does-not-claim-an-unemitted-conversion
  (let [context (completion/bytecode-operand-context {})]
    (doseq [contract [completion/object-input-contract
                     (completion/reference-check-contract String)
                     (completion/bytecode-reference-input-contract String context)
                     (completion/bytecode-integral-input-contract Long/TYPE context)
                     (completion/bytecode-floating-input-contract Double/TYPE context)
                     (completion/bytecode-result-context-contract context :double)]
            object-parent? [false true]]
      (let [expression (completion/complete-value contract '(throw x))
            root (if object-parent?
                   (completion/complete-value completion/object-input-contract expression)
                   expression)
            ;; Neither the later initializer nor the body is reachable.
            body [(list 'let* ['result root 'later 'result] 'later)]
            {:keys [implementation emission]} (emit body)
            observed (observe emission body expression)]
        (is (= [:diverge :diverge :diverge :diverge]
               (mapv observed [:input-carrier :completion-carrier :returned-carrier :adaptation])))
        (is (nil? (:result-fact observed)))
        (is (thrown? IllegalStateException (implementation (IllegalStateException. "operand"))))))))

(deftest emitted-completion-facts-compose-through-the-shared-normal-return-analysis
  (let [expression (completion/complete-value
                    (completion/bytecode-floating-input-contract
                     Double/TYPE (completion/bytecode-operand-context {})) 'x)
        boxed (completion/complete-value completion/object-input-contract expression)
        body [(list 'let* ['converted boxed 'alias 'converted] 'alias)]
        {:keys [implementation emission]} (emit body)
        resolve-result (fn [node] (:result-fact (observe emission body node)))
        expected {:carrier Object :presence :present :builtin-class Double}]
    (is (nil? (normal-return/completed-scalar-fact boxed)))
    (let [direct-body [(list 'let* ['converted expression] 'converted)]
          direct-emission (:emission (emit direct-body))]
      (is (= {:carrier Double/TYPE :presence :present :builtin-class Double}
             (normal-return/completed-scalar-fact
              expression {} *ns*
              (fn [node] (:result-fact (observe direct-emission direct-body node)))))))
    (is (= expected (normal-return/completed-scalar-fact boxed {} *ns* resolve-result)))
    (is (= expected (normal-return/completed-scalar-fact (first body) {} *ns* resolve-result)))
    (is (= expected
           (get (normal-return/completed-scalar-bindings
                 (mapv vec (partition 2 (second (first body)))) {} *ns* resolve-result) 'alias)))
    (is (nil? (normal-return/completed-scalar-fact
               (apply list expression) {} *ns* resolve-result))
        "a copied completion cannot borrow the original emitted result")
    (is (= 2.5 (implementation 2.5)))
    (is (thrown? NullPointerException (implementation nil))
        "nonnull-on-normal-return is not proof that the conversion succeeds")
    (with-redefs [entry/assert-emission-inputs-current!
                  (fn [_] (throw (ex-info "stale test owner" {:reason :stale-definition-proof-input})))]
      (is (= :stale-definition-proof-input
             (try (normal-return/completed-scalar-fact boxed {} *ns* resolve-result) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest discarded-body-divergence-does-not-emit-later-forms
  (doseq [source ['(do (throw x) missing-later)
                  '(let* [] (throw x) missing-later)]
          completed? [false true]]
    (let [expression (if completed?
                       (completion/complete-value
                        (completion/bytecode-result-context-contract
                         (completion/bytecode-operand-context {}) :ref)
                        source)
                       source)
          body [expression]
          {:keys [implementation emission]} (emit body)]
      (is (thrown? IllegalStateException (implementation (IllegalStateException. "stop"))))
      (when completed?
        (is (= :diverge (:completion-carrier (observe emission body expression))))))))

(deftest discarded-false-nil-and-category-two-values-do-not-stop-the-body
  (let [{:keys [implementation]} (emit ['(do nil false 1.25 (long 7) (int 3) true x)])
        value (Object.)]
    (is (identical? value (implementation value)))))

(deftest malformed-owned-completion-facts-fail-instead-of-becoming-types
  (let [expression (completion/complete-value completion/object-input-contract 'x)]
    (doseq [fact [false "not a fact"
                  {:carrier String :presence :present :builtin-class String}
                  {:carrier Object :presence :present}
                  {:carrier Double/TYPE :presence :present :builtin-class Float}
                  {:carrier Double/TYPE :presence :unknown :builtin-class Double}]]
      (is (= :invalid-completion-result-fact
             (try (normal-return/completed-scalar-fact expression {} *ns* (constantly fact)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
