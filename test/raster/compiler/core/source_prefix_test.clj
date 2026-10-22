(ns raster.compiler.core.source-prefix-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.par :as par]
            [raster.compiler.core.source-prefix :as prefix]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.backend.jvm.terminal-source :as projection]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.gpu.core :as gpu]))

(deftm selected-prefix [n :- Long] :- (Array double)
  (let [size (numeric/* n 2)
        out (double-array size)]
    (par/map! out i size double 1.0)))

(deftm literal-count [unused :- Long] :- Long 7)

(deftm direct-scalar-root [n :- Long] :- Long (numeric/* n 2))
(deftm first-argument-call-root [n :- Long] :- Long (numeric/* (numeric/+ n 1) 2))
(deftm later-argument-call-root [n :- Long] :- Long (numeric/* 2 (numeric/+ n 1)))

(deftm unused-checked-helper-input [rows :- Long chunks :- Long] :- (Array double)
  (let [size (literal-count (numeric/* rows chunks))
        out (double-array size)]
    (par/map! out i size double 1.0)))

(deftm shadowed-prefix [n :- Long] :- (Array double)
  (let [n (numeric/* n 2)
        out (double-array n)]
    (par/map! out i n double 1.0)))

(deftm nested-input-prefix [n :- Long] :- (Array double)
  (let [size (numeric/* (long n) 2)
        out (double-array size)]
    (par/map! out i size double 1.0)))

(deftm allocation-before-selected [n :- Long] :- (Array double)
  (let [early (double-array n)
        size (numeric/* n 2)
        out (double-array size)]
    (par/map! out i size double (aget early 0))))

(defn- source-with-bindings [source bindings]
  (with-meta (list* (first source) bindings (nnext source)) (meta source)))

(defn- reclose [& arguments]
  (entry/call-with-definition-inputs #(apply prefix/close arguments)))

(defn- capture [function]
  (let [observed (atom {})
        begin prefix/begin close prefix/close
        target :ocl:source-prefix-proof
        descriptor {:device-id target :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}
        compilation
        (with-redefs [prefix/begin (fn [frontend parameters]
                                   (swap! observed assoc :frontend frontend)
                                   (begin frontend parameters))
                      prefix/close (fn [& arguments]
                                     (let [result (apply close arguments)]
                                       (swap! observed assoc :inputs (vec (take 5 arguments))
                                              :construction (nth arguments 5 nil)
                                              :correspondence result)
                                       result))]
          (equation/compile function {:target target :dtype :double} descriptor))]
    (assoc @observed :compilation compilation)))

(deftest direct-root-preparation-is-bounded-and-never-executes-source
  (doseq [[function candidate?] [[#'direct-scalar-root false]
                                [#'first-argument-call-root true]
                                [#'later-argument-call-root false]]]
    (let [resolved (core/resolve-deftm-var function)]
      (core/prepare-deftm-frontend! resolved)
      (entry/call-with-definition-inputs
       (fn []
         (let [frontend (entry/original-source-frontend resolved)
               input (prefix/begin frontend (get-in frontend [:source-entry :params]))
               calls (atom [])
               prepare! #(swap! calls conj %)]
           (is (some? input) "retention alone is not preparation or correspondence")
           (is (= (when candidate? true) (prefix/prepare-prefix-entry! input prepare!)))
           (is (= (if candidate? [(get-in frontend [:definition :receiver-var])] []) @calls))
           (reset! calls [])
           (doseq [foreign [nil {}]]
             (is (nil? (prefix/prepare-prefix-entry! foreign prepare!))))
           (when candidate?
             (with-redefs [entry/assert-frontend-selection!
                           (fn [& _] (throw (ex-info "wrong original receiver" {:reason :test-selection})))]
               (is (= :test-selection
                      (try (prefix/prepare-prefix-entry! input prepare!) nil
                           (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
           (is (empty? @calls) "failed selection/foreign input never reaches preparation")))))))

(deftest source-phase-history-is-local-and-checked-before-prefix-verification
  (let [{:keys [inputs construction correspondence]} (capture #'selected-prefix)
        [input normalization normalized represented program] inputs]
    (is (some? construction))
    (is (some? correspondence))
    (is (some? (reclose input normalization normalized represented program construction)))
    (is (nil? (reclose input normalization normalized represented program {})))
    (is (nil? (reclose input normalization normalized
                      (with-meta (apply list represented) (meta represented)) program construction)))
    (is (nil? (reclose input {} normalized represented program construction)))))

(deftest unused-helper-input-is-not-erased-before-source-ownership
  ;; Complete the original numerical JIT before retaining compilation history.
  (is (= (vec (repeat 7 1.0)) (vec (unused-checked-helper-input 2 3))))
  (is (thrown? ArithmeticException (unused-checked-helper-input Long/MAX_VALUE 2)))
  (let [{:keys [compilation]} (capture #'unused-checked-helper-input)
        program (:semantic compilation)
        source (:source program)
        first-expression (second (second source))
        order (invocation/source-order (get-in program [:attributes :invocation-plan]) program)
        allocations (atom 0)]
    (is (some #(= 'raster.numeric/* (descriptor/semantic-op %))
              (filter seq? (tree-seq coll? seq first-expression))))
    (is (contains? (:flags (effects/descriptor first-expression)) :unclosed-source-arithmetic))
    (is (some? (:equation (first order))) "argument evaluation precedes allocation")
    (is (some? (:invocation-step (first order)))
        "the unused checked argument has one ordered scalar executor")
    (is (some? (:storage (second order))))
    (with-redefs [gpu/alloc! (fn [& _] (swap! allocations inc)
                             (throw (AssertionError. "unclosed argument reached device allocation")))]
      (is (= :unclosed-ordered-source-executor
             (try (equation/lower compilation [Long/MAX_VALUE 2]) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "an executor plan alone does not prove the original helper argument interval"))
    (is (zero? @allocations))))

(deftest original-occurrence-is-separate-from-logical-result-conversion
  (let [{:keys [frontend inputs correspondence compilation]} (capture #'selected-prefix)
        [input normalization normalized represented program] inputs
        step (first (filter invocation/scalar-compute?
                            (get-in program [:attributes :invocation-plan :steps])))
        original (prefix/original-step correspondence frontend program step)]
    (is (some? input))
    (is (some? correspondence))
    (is (= 0 (:ordinal original)))
    (is (= 'size (:binding original)))
    (is (identical? (first (:body frontend)) (:scope original)))
    (is (identical? (second (second (:scope original))) (:expression original)))
    (is (= '.invk (first (:expression original))))
    (is (= 'clojure.core/long (first (:result-expression original))))
    (is (= (:expression original) (second (:result-expression original))))
    (is (identical? step (:step original)))
    (is (some? (reclose input normalization normalized represented program)))
    (is (nil? (reclose input nil normalized represented program)))
    (is (nil? (reclose input {} normalized represented program)))
    (is (nil? (reclose input normalization (apply list normalized) represented program)))
    (is (nil? (prefix/original-step correspondence frontend program nil)))
    (is (nil? (prefix/original-step correspondence frontend program (into {} step))))
    (is (nil? (prefix/original-step correspondence frontend (assoc program :copied true) step)))
    (is (nil? (prefix/original-step correspondence (into {} frontend) program step)))
    (is (= [[2]] (mapv #(get-in % [:view :shape]) (vals (:nodes (equation/lower compilation [1]))))))))

(deftest invocation-only-scalar-correspondence-retains-exact-conversions
  (let [{:keys [inputs]} (capture #'selected-prefix)
        [input normalization normalized represented] inputs
        program (last inputs)
        step (first (filter invocation/scalar-compute?
                            (get-in program [:attributes :invocation-plan :steps])))
        expression (second (second (nth inputs 3)))
        final-expression (second (second (:source program)))
        corresponds? @(ns-resolve 'raster.compiler.core.source-prefix 'scalar-source-corresponds?)
        {:keys [parameters]} (dialect/lambda-parts (:region step))
        with-body #(assoc step :region (dialect/lambda-form parameters [%]))]
    (is (nil? (:equation (first (invocation/source-order
                                (get-in program [:attributes :invocation-plan]) program)))))
    (is (corresponds? expression final-expression step nil {'n 'long}))
    ;; An unchanged invocation-only source is also legal; adding a result cast
    ;; is not required merely to construct correspondence.
    (is (corresponds? expression expression (with-body expression) nil {'n 'long}))
    (doseq [[changed-final changed-step]
            [[expression step]
             [final-expression (with-body expression)]
             [final-expression (with-body '(clojure.core/long 7))]
             [final-expression (assoc-in step [:value :dtype] :int)]
             [final-expression (assoc-in step [:operands 0 :symbol] 'foreign)]
             [(list 'clojure.core/int expression) step]]]
      (is (not (corresponds? expression changed-final changed-step nil {'n 'long}))))
    ;; A coherently edited plan can still be internally valid. That must not
    ;; authorize replacing the authenticated source occurrence by new work.
    (let [materialized-expression (second final-expression)
          replacement (with-meta (apply list (concat (butlast materialized-expression) [3]))
                        (meta materialized-expression))
          result (with-meta (list (first final-expression) replacement) (meta final-expression))
          changed-step (with-body result)
          changed (-> program
                      (assoc :source (source-with-bindings (:source program)
                                                          (assoc (second (:source program)) 1 result)))
                      (update-in [:attributes :invocation-plan :steps]
                                 #(mapv (fn [candidate] (if (identical? candidate step)
                                                         changed-step candidate)) %)))]
      (is (some? (invocation/validate-against! (get-in changed [:attributes :invocation-plan]) changed)))
      (is (nil? (reclose input normalization normalized represented changed))))))

(deftest history-and-original-frontend-identity-remain-required
  (let [{:keys [frontend inputs correspondence]} (capture #'selected-prefix)
        [input normalization normalized represented program] inputs
        step (first (filter invocation/scalar-compute?
                            (get-in program [:attributes :invocation-plan :steps])))
        source (get-in frontend [:source-entry :source-var])
        metadata (meta source)]
    (is (nil? (reclose nil normalization normalized represented program)))
    (is (nil? (reclose {} normalization normalized represented program)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (prefix/begin (assoc frontend :body (mapv identity (:body frontend))) '[n])))
    (try
      (alter-meta! source assoc :raster.core/source-policy {:unchecked-math true})
      (is (thrown? clojure.lang.ExceptionInfo
                   (reclose input normalization normalized represented program)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (prefix/original-step correspondence frontend program step)))
      (finally (reset-meta! source metadata)))
    (is (some? (prefix/original-step correspondence frontend program step)))))

(deftest original-step-adopts-source-identity-into-a-fresh-publication-owner
  (let [{:keys [frontend inputs correspondence]} (capture #'selected-prefix)
        program (last inputs)
        step (first (filter invocation/scalar-compute?
                            (get-in program [:attributes :invocation-plan :steps])))
        owner (atom {:static [] :values []})
        source (get-in frontend [:source-entry :source-var])
        metadata (meta source) raw (:raster.core/deftm-source-body metadata)]
    (binding [entry/*emission-inputs* owner]
      (is (some? (prefix/original-step correspondence frontend program step))))
    (try
      (alter-meta! source assoc :raster.core/deftm-source-body (with-meta (mapv identity raw) (meta raw)))
      (is (= metadata (meta source)))
      (is (every? entry/var-snapshot-current? (vals (:definition-proofs @owner))))
      (is (= :stale-definition-proof-input
             (try (entry/assert-definition-inputs-current! @owner) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
      (finally (reset-meta! source metadata)))))

(deftest whole-prefix-must-survive-not-just-the-selected-call
  (let [{:keys [inputs]} (capture #'selected-prefix)
        [input normalization normalized represented program] inputs
        pairs (second normalized)
        rewrites [(source-with-bindings normalized (vec (drop 2 pairs)))
                  (source-with-bindings normalized (vec (concat (take 2 pairs) pairs)))
                  (source-with-bindings normalized (vec (concat (drop 2 pairs) (take 2 pairs))))
                  ;; Same numerical product, different original operand order.
                  (source-with-bindings normalized (assoc pairs 1 (list '.invk (second (second pairs)) 2 'n)))
                  (source-with-bindings normalized (update pairs 1 vary-meta assoc :tag 'double))
                  (source-with-bindings normalized (update pairs 1 vary-meta assoc :raster.effect/preserve-evaluation true))
                  (source-with-bindings normalized (assoc pairs 1 '(foreign.namespace/producer n 2)))]]
    (doseq [changed rewrites]
      (is (nil? (reclose input normalization changed represented program))))
    (doseq [changed rewrites]
      (is (nil? (reclose input normalization normalized changed program))))))

(deftest result-materialization-remains-an-independently-validated-contract
  (let [{:keys [inputs]} (capture #'selected-prefix)
        [input normalization normalized represented program] inputs
        expression (second (second (:source program)))]
    (doseq [replacement [(second expression) (list 'clojure.core/double (second expression))]
            :let [changed (update program :source
                                  #(source-with-bindings % (assoc (second %) 1 replacement)))]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (reclose input normalization normalized represented changed))))))

(deftest shadowed-original-parameters-do-not-acquire-prefix-authority
  (let [{:keys [correspondence]} (capture #'shadowed-prefix)]
    (is (nil? correspondence))))

(deftest producer-prepares-callee-before-caller-without-query-side-effects
  (let [prepared (atom []) prepare! core/ensure-current-owned-bytecode!
        {:keys [frontend inputs correspondence compilation]}
        (with-redefs [core/ensure-current-owned-bytecode!
                      (fn [receiver] (swap! prepared conj receiver) (prepare! receiver))]
          (capture #'selected-prefix))
        [_ normalization normalized represented program] inputs
        step (first (filter invocation/scalar-compute?
                            (get-in program [:attributes :invocation-plan :steps])))
        original (prefix/original-step correspondence frontend program step)
        receiver (get-in frontend [:definition :receiver-var])]
    ;; A cold batch invalidates the whole compilation and runs again warm.
    ;; A warm compile validates the same ordered pair without replacing code.
    (is (contains? #{2 4} (count @prepared)))
    (is (every? #(= [#'numeric/_star__m_long_long-impl receiver] (vec %))
                (partition 2 @prepared)))
    (with-redefs [core/ensure-current-owned-bytecode!
                  (fn [_] (throw (AssertionError. "an observation query prepared code")))]
      (binding [entry/*emission-inputs* (atom {:static [] :values []})]
        (is (some? (prefix/begin frontend '[n])))
        (is (some? (reclose (first inputs) normalization normalized represented program)))
        (is (identical? (:expression original)
                        (:expression (prefix/original-step correspondence frontend program step))))
        (is (= Object (:return-carrier (entry/frontend-invocation-context frontend (:expression original)))))
        (is (= :ref (get-in (entry/frontend-binding-context frontend (:scope original) (:ordinal original))
                           [:after :type])))))
    (is (map? (equation/lower compilation [1])))))

(deftest unsupported-and-foreign-prefixes-perform-no-preparation
  (let [prepared (atom [])
        {:keys [correspondence frontend inputs]}
        (with-redefs [core/ensure-current-owned-bytecode! #(swap! prepared conj %)]
          (capture #'nested-input-prefix))
        program (last inputs)]
    (is (some? correspondence) "the unsupported operand is not a missing prefix proof")
    (is (empty? @prepared))
    (binding [entry/*emission-inputs* (atom {:static [] :values []})]
      (doseq [[owner source semantic] [[nil frontend program]
                                       [correspondence (into {} frontend) program]
                                       [correspondence frontend (assoc program :foreign true)]
                                       [correspondence frontend program]]]
        (is (nil? (prefix/prepare-selected-calls! owner source semantic #(swap! prepared conj %)))))
      (is (empty? (:definition-proofs @entry/*emission-inputs*)))
      (is (empty? @prepared)))
    (let [compilation (:compilation (capture #'nested-input-prefix))
          failure (try (equation/lower compilation [1]) nil
                       (catch clojure.lang.ExceptionInfo error (ex-data error)))]
      (is (= :unclosed-ordered-source-executor (:reason failure)))
      (is (= '.invk (first (get-in failure [:original-source :expression])))))))

(defn- outcome [run]
  (try (let [value (run)] [:value (class value) value])
       (catch Exception failure [:error (class failure)])))

(deftest original-selected-execution-preserves-overflow-and-changed-inputs
  (let [{:keys [compilation]} (capture #'selected-prefix)
        step (first (filter invocation/scalar-compute?
                            (get-in compilation [:semantic :attributes :invocation-plan :steps])))
        execute #(get (owner/evaluate-ordered-invocation-step compilation step
                                                             {'n {:type :long :value %}}) :value)]
    (doseq [n [0 -3 1 63 64 1000 Long/MAX_VALUE Long/MIN_VALUE]]
      (is (= (outcome #(Math/multiplyExact (long n) (long 2)))
             (outcome #(numeric/* n 2)) (outcome #(execute n)))))
    (doseq [n [0 1 3]]
      (is (= [[(* 2 n)]]
             (mapv #(get-in % [:view :shape]) (vals (:nodes (equation/lower compilation [n])))))))
    (doseq [value [nil 1.0 (int 1)]]
      (is (= :ordered-scalar-source-arguments
             (try (execute value) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
    (let [source #'numeric/_star__m_long_long metadata (meta source)]
      (try
        (alter-meta! source assoc ::changed-helper true)
        (is (= :stale-definition-proof-input
               (try (execute 1) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
        (finally (reset-meta! source metadata))))))

(deftest eager-source-counts-run-in-source-order-without-native-evidence
  (let [compilation (:compilation (capture #'selected-prefix))
        before-allocation (:compilation (capture #'allocation-before-selected))
        execute owner/evaluate-ordered-invocation-step
        calls (atom 0)
        forbidden (fn [& _] (throw (AssertionError. "pure specialization acquired native evidence")))]
    (with-redefs [gpu/make-session forbidden gpu/alloc! forbidden gpu/zero-buffer! forbidden
                  invocation-link/prepare-source-allocation forbidden
                  owner/evaluate-ordered-invocation-step
                  (fn [& args] (swap! calls inc) (apply execute args))]
      (doseq [n [-1 (long Integer/MAX_VALUE) Long/MAX_VALUE]]
        (is (= (outcome #(alength (selected-prefix n)))
               (outcome #(equation/lower compilation [n])))))
      (doseq [n [2147483648 2147483649]]
        (is (= (- (* 2 n) 4294967296) (alength (double-array (* 2 n)))))
        (is (= :invocation-ordered-allocation-shape
               (try (equation/lower compilation [n]) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
      (reset! calls 0)
      ;; The first constructor fails before the otherwise-overflowing multiply.
      (is (= [:error NegativeArraySizeException]
             (outcome #(equation/lower before-allocation [Long/MAX_VALUE]))))
      (is (zero? @calls))
      (is (= (outcome #(alength (allocation-before-selected Long/MAX_VALUE)))
             (outcome #(equation/lower before-allocation [Long/MAX_VALUE]))))
      (is (zero? @calls))
      (is (map? (equation/lower before-allocation [1])))
      (is (= 1 @calls) "successful prefix computes the selected source call once"))))

(deftest eager-source-prefix-rejects-used-state-and-unconsumed-computations
  (let [compilation (:compilation (capture #'selected-prefix))
        initial (invocation-link/begin-ordered-realization compilation [1] nil)
        calls (atom 0)
        forbidden (fn [& _] (swap! calls inc)
                    (throw (AssertionError. "invalid prefix executed source work")))
        failure (fn [state]
                  (try (invocation-link/materialize-owned-source-prefix state forbidden forbidden nil)
                       nil
                       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
    (doseq [changed [(assoc initial :host-scalars {'foreign {:type :long :value 1}})
                     (assoc initial :host-steps {:foreign {}})
                     (assoc-in initial [:realized :buffers] {'foreign :storage})
                     (assoc-in initial [:realized :storage] {:foreign {}})
                     (assoc-in initial [:realized :loop-scratch] {:foreign {}})
                     (assoc initial :pending {:kind :source-allocation})
                     (assoc initial :acquired-storage #{:foreign})
                     (assoc initial :source-cursor 1)
                     (assoc-in initial [:materialization :next-step] 1)]]
      (is (= :invocation-eager-source-start (failure changed))))
    ;; Removing the source entries must not turn the terminal materializer into
    ;; a second route that executes skipped source computations out of order.
    (is (= :invocation-eager-source-unconsumed (failure (assoc initial :source-order []))))
    (is (zero? @calls))))

(deftest resolved-projector-preserves-the-returned-reference
  ;; Isolate backend mechanics with an interface-compatible instrumented target.
  ;; This deliberately substituted bridge is NOT a frontend authority: the
  ;; compiler's original-owner checks reject foreign implementations below.
  (let [{:keys [frontend]} (capture #'selected-prefix)
        source (nth (second (first (:body frontend))) 1)
        source-ns (get-in frontend [:source-entry :source-ns])
        returned (Long. (long 1000)) calls (atom 0)]
    (entry/call-with-definition-inputs
     (fn []
       (let [callee (entry/original-source-frontend #'numeric/_star__m_long_long)
             bridge (entry/frontend-return-bridge callee)
             method (:invocation-method bridge)
             interface (.getDeclaringClass ^java.lang.reflect.Method method)
             target (java.lang.reflect.Proxy/newProxyInstance
                     (.getClassLoader interface) (into-array Class [interface])
                     (reify java.lang.reflect.InvocationHandler
                       (invoke [_ _ actual arguments]
                         (is (= method actual))
                         (is (= [5 2] (vec arguments)))
                         (swap! calls inc)
                         returned)))
             projected (projection/project-original-invocation
                        source [{:symbol 'n :carrier Long/TYPE}] source-ns
                        {:unchecked-math false} (assoc bridge :implementation target))
             execute (owner/compile-function
                      (:params projected) (:source projected) (fn [params body] (list 'fn* params body))
                      (merge projected {:source-policy {:unchecked-math false} :source-ns source-ns}))]
         (is (= {:carrier Object :boxed-carrier Long :non-null? true} (:result-facts projected)))
         (is (identical? returned (execute [5])))
         (is (= 1 @calls))
         (is (not-any? #(and (symbol? %) (= "compute_static" (name %)))
                       (tree-seq coll? seq (:source projected)))))))))

(deftest original-selected-executor-refuses-wrong-physical-completions
  (let [call entry/frontend-invocation-context binding-context entry/frontend-binding-context
        bridge entry/frontend-return-bridge
        changes
        [{#'entry/frontend-invocation-context #(some-> (call %1 %2) (assoc :return-carrier Long/TYPE))}
         {#'entry/frontend-invocation-context #(some-> (call %1 %2) (assoc :invocation-mode :primitive-indy))}
         {#'entry/frontend-invocation-context #(some-> (call %1 %2) (assoc-in [:inputs 0 :ports 'n :type] :ref))}
         {#'entry/frontend-invocation-context #(some-> (call %1 %2) (assoc-in [:inputs 0 :context :object-result?] true))}
         {#'entry/frontend-invocation-context #(some-> (call %1 %2) (assoc-in [:receiver-input :context :object-result?] true))}
         {#'entry/frontend-invocation-context #(some-> (call %1 %2)
                                                     (assoc :method (.getMethod clojure.lang.IFn "invoke"
                                                                              (into-array Class [Object Object]))))}
         {#'entry/frontend-binding-context #(some-> (binding-context %1 %2 %3) (assoc-in [:after :type] :long))}
         {#'entry/frontend-binding-context #(some-> (binding-context %1 %2 %3) (assoc :initializer '(foreign n 2)))}
         {#'entry/frontend-return-bridge #(some-> (bridge %) (assoc :implementation (Object.)))}
         {#'entry/frontend-return-bridge #(some-> (bridge %) (assoc :payload-carrier Double/TYPE))}
         {#'entry/frontend-return-bridge #(some-> (bridge %) (assoc :invocation-method nil))}]]
    ;; Deliberately corrupt internal observations to test the consumer's
    ;; completion checks. Public queries separately reject forged module maps.
    (doseq [change changes]
      (with-redefs-fn change
        (fn []
          (let [compilation (:compilation (capture #'selected-prefix))]
            (is (= :unclosed-ordered-source-executor
                   (try (equation/lower compilation [1]) nil
                        (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))))))

(deftest alpha-transport-checks-order-capture-literal-bits-and-declarations
  ;; Isolate the bounded transformation verifier; public tests above exercise
  ;; actual producers and the exact program/step/source-owner fences.
  (let [relation @(ns-resolve 'raster.compiler.core.source-prefix 'prefix-renaming)
        original '[[a (free-helper x)] [b (free-helper a)]]]
    (is (= '{x x a first b second}
           (relation original '[[first (free-helper x)] [second (free-helper first)]] '[x] {'x 'long})))
    (is (nil? (relation original '[[x (free-helper x)] [b (free-helper x)]] '[x] {'x 'long})))
    (is (nil? (relation original
                        '[[free-helper (free-helper x)] [b (free-helper free-helper)]] '[x] {'x 'long})))
    ;; A future binder is not in scope in an earlier initializer. Removing
    ;; every binder from the free set would incorrectly approve this capture.
    (is (nil? (relation '[[a 1] [b (helper c)] [c 2]]
                        '[[c 1] [b (helper c)] [d 2]] [] {})))
    (is (nil? (relation '[[a 1] [c (helper c)]]
                        '[[c 1] [d (helper c)]] [] {})))
    (is (some? (relation '[[a (helper a)] [b a]]
                         '[[a (helper a)] [result a]] [] {})))
    (is (nil? (relation '[[a -0.0]] '[[a 0.0]] [] {})))
    (is (nil? (relation [['a (long 1)]] [['a (int 1)]] [] {})))
    (let [typed (with-meta 'x {:raster.type/tag 'long :tag 'long})]
      (is (some? (relation '[[a x]] [['a typed]] '[x] {'x 'long})))
      (is (nil? (relation '[[a x]] [['a typed]] '[x] {})))
      (is (nil? (relation '[[a x]] [['a typed]] '[x] {'x 'double}))))
    (let [compare-expression @(ns-resolve 'raster.compiler.core.source-prefix 'same-expression?)
          quoted '(quote (helper x))]
      (is (true? (compare-expression quoted quoted {'x 'renamed} {})))
      (is (false? (compare-expression quoted (apply list quoted) {} {})))
      (is (not (compare-expression '(unknown-factory x)
                                   (with-meta '(unknown-factory x) {:tag 'doubles}) {} {}))))))
