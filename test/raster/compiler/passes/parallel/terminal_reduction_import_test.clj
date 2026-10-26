(ns raster.compiler.passes.parallel.terminal-reduction-import-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.structured-control-route :as structured]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.source-import :as source-import]))

(core/deftm nonterminal-sum [xs :- (Array double)] :- Double
  (do (par/sum xs) 0.0))

(defn- selected-frontend [generic]
  (let [source (core/resolve-deftm-var generic)]
    (core/prepare-emission-source-walk! source)
    (entry/call-with-definition-inputs
     #(let [original (entry/original-source-frontend source)]
        (entry/prepare-frontend-source! original source core/ensure-current-owned-bytecode!)))))

(defn- options [original]
  (let [receiver (get-in original [:definition :implementation-op])]
    {:frontend original :input-body (:body original) :root (first (:body original))
     :caller-expression (list '.invk receiver 'input) :arguments '[input]
     :result-symbol (with-meta 'answer {:tag 'Object :raster.type/tag 'double})
     :source-position 3
     :values {'input (av/tensor {:dtype :double :shape [4]})}
     :array-types {'input :double} :scalar-types {}}))

(defn- construct [options]
  (entry/call-with-definition-inputs #(source-import/analyze-terminal-reduction-import options)))

(defn- return-contract [options]
  (entry/call-with-definition-inputs #(source-import/terminal-reduction-return-contract options)))

(defn- reason [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest prepared-terminal-return-contract-does-not-analyze-or-prepare-source
  (let [original (selected-frontend #'par/sum)
        request (select-keys (options original) [:frontend :input-body :root])
        forbidden (fn [& _] (throw (AssertionError. "return query must not construct numerical source")))
        returned (with-redefs [frontend/normalize-source forbidden
                              frontend/form->program-components forbidden
                              inline/expand-original-helper-source forbidden
                              entry/prepare-frontend-source! forbidden
                              core/ensure-current-owned-bytecode! forbidden]
                   (return-contract request))
        outer (completion/canonical-parts (:recipe returned))
        consumer (completion/canonical-parts (:operand outer))
        context (completion/canonical-parts (:operand consumer))]
    (is (identical? returned (source-import/validate-terminal-reduction-return! returned)))
    (is (identical? original (:frontend returned)))
    (is (identical? (:input-body request) (:input-body returned)))
    (is (identical? (:root request) (:root returned)))
    (is (= (get-in original [:source-entry :params]) (:parameters returned)))
    (is (= [(.arrayType Double/TYPE)] (:parameter-carriers returned)))
    (is (identical? Double/TYPE (:result-carrier returned)))
    (is (= 'double (:result-tag returned)))
    (is (= (get-in original [:source-entry :source-policy]) (:source-policy returned)))
    (is (= completion/object-input-contract (:contract outer)))
    (is (= :ref (get-in consumer [:contract :expected-input-carrier])))
    (is (= :jvm-bytecode-result-context (get-in context [:contract :source-context])))
    (is (identical? (:value-port returned) (:operand context)))
    (is (instance? java.lang.reflect.Method (:compute-method returned)))
    (is (instance? java.lang.reflect.Method (:invocation-method returned)))
    (is (not (contains? returned :component)))
    (is (= :terminal-reduction-import-correspondence
           (reason #(source-import/validate-terminal-reduction-import! returned))))
    (doseq [changed [(dissoc returned :receipt)
                     (assoc returned :recipe (with-meta (apply list (:recipe returned))
                                               (meta (:recipe returned))))
                     (assoc returned :parameter-carriers (vec (seq (:parameter-carriers returned))))]]
      (is (= :terminal-reduction-return-correspondence
             (reason #(source-import/validate-terminal-reduction-return! changed)))))))

(deftest terminal-return-query-keeps-original-root-return-and-history-refusals
  (let [original (selected-frontend #'par/sum)
        request (options original)
        copied-root (with-meta (apply list (:root request)) (meta (:root request)))]
    (is (= :terminal-reduction-import-owner
           (reason #(source-import/terminal-reduction-return-contract request))))
    (doseq [changed [(assoc request :input-body (vec (seq (:input-body request))))
                     (assoc request :root copied-root)]]
      (is (= :terminal-reduction-import-root (reason #(return-contract changed)))))
    (is (= :terminal-reduction-import-return
           (reason #(return-contract (update-in request [:frontend :emission] (fn [x] (into {} x)))))))
    (let [observe bytecode/retained-input-reduction-emission]
      (doseq [change [#(assoc-in % [:result-fact :builtin-class] Float)
                      #(assoc-in % [:result-fact :presence] :unknown)
                      #(assoc-in % [:root-result :result-type] :double)
                      #(assoc-in % [:root-result :context :void-context] true)
                      #(dissoc % :root-result)]]
        (with-redefs [bytecode/retained-input-reduction-emission
                      (fn [& arguments] (some-> (apply observe arguments) change))]
          (is (= :terminal-reduction-import-return (reason #(return-contract request)))))))
    (let [bridge entry/frontend-boxed-entry-bridge]
      (with-redefs [entry/frontend-boxed-entry-bridge
                    (fn [frontend]
                      (some-> (bridge frontend)
                              (assoc :compute-parameter-carriers [(.arrayType Float/TYPE)])))]
        (is (= :terminal-reduction-import-return (reason #(return-contract request))))))
    (is (= :terminal-reduction-import-root
           (reason #(return-contract (options (selected-frontend #'nonterminal-sum))))))
    (let [returned (return-contract request)
          source (get-in original [:source-entry :source-var])
          receiver (get-in original [:definition :receiver-var])]
      (with-redefs-fn {receiver (Object.)}
        #(is (some? (reason (fn [] (source-import/validate-terminal-reduction-return! returned))))))
      (with-redefs-fn {source (fn [& _] (throw (AssertionError. "must never execute")))}
        #(is (some? (reason (fn [] (source-import/validate-terminal-reduction-return! returned)))))))))

(deftest actual-terminal-sum-import-is-analyzed-before-publishing-unresolved-obligations
  (let [original (selected-frontend #'par/sum)
        request (options original)
        imported (construct request)
        component (:component imported)
        program (:typed-program component)
        recipe (get-in imported [:obligations :return :recipe])
        outer (completion/canonical-parts recipe)
        consumer (completion/canonical-parts (:operand outer))
        root-context (completion/canonical-parts (:operand consumer))]
    (is (identical? imported (source-import/validate-terminal-reduction-import! imported)))
    (is (= program (dialect/validate! program)))
    (is (= 1 (count (dialect/equations program))))
    (is (= :double (get-in component [:values (first (:outputs component)) :dtype])))
    (is (= (:source-position request) (:source-position imported)))
    (is (= (get-in original [:source-entry :source-policy]) (:source-policy imported)))
    (is (= completion/object-input-contract (:contract outer)))
    (is (= :jvm-bytecode-floating-input (get-in consumer [:contract :source-context])))
    (is (= :ref (get-in consumer [:contract :expected-input-carrier])))
    (is (identical? Double/TYPE (get-in consumer [:contract :target-carrier])))
    (is (= :jvm-bytecode-result-context (get-in root-context [:contract :source-context])))
    (is (= :ref (get-in root-context [:contract :result-carrier])))
    (is (instance? java.lang.reflect.Method (get-in imported [:obligations :return :compute-method])))
    (is (= :unresolved (get-in imported [:obligations :return :status])))
    (is (= :unresolved (get-in imported [:obligations :import :status])))
    (is (= :unresolved (get-in imported [:obligations :caller-binding :status])))
    (is (= (:result-symbol request) (get-in imported [:obligations :caller-binding :binder])))
    (is (= [] (:argument-bindings imported))
        "ordinary lexical array substitution needs no invented argument evaluation")
    (is (= [{:argument 'input :parameter-carrier (.arrayType Double/TYPE) :status :unresolved}]
           (get-in imported [:obligations :inputs])))
    (is (not (completion/contains-completion? (:normalized-source imported)))
        "bytecode return recipes stay outside the numerical component")
    (is (not-any? #(or (class? %) (instance? java.lang.reflect.Method %)
                      (satisfies? entry/RetainedProducerInput %))
                  (tree-seq coll? seq program)))
    (is (= :terminal-reduction-import-correspondence
           (reason #(source-import/validate-terminal-reduction-import!
                     (update imported :component (fn [component] (into {} component)))))))
    (is (= :terminal-reduction-import-correspondence
           (reason #(source-import/validate-terminal-reduction-import! (dissoc imported :receipt)))))
    (let [before (inline/expand-for-backends (:caller-expression request))]
      (is (not (completion/contains-completion? before))
          "ordinary expansion receives no constructor hook or return-recipe insertion"))))

(deftest terminal-import-refuses-copied-foreign-and-unsupported-source-obligations
  (let [original (selected-frontend #'par/sum)
        request (options original)
        copied-root (with-meta (apply list (:root request)) (meta (:root request)))]
    (is (= :terminal-reduction-import-owner
           (reason #(source-import/analyze-terminal-reduction-import request))))
    (doseq [changed [(assoc request :input-body (vec (seq (:input-body request))))
                     (assoc request :root copied-root)]]
      (is (= :terminal-reduction-import-root (reason #(construct changed)))))
    (is (= :unowned-checked-result-frontend
           (reason #(construct (assoc-in request [:frontend :body] [copied-root])))))
    (is (= :terminal-reduction-import-return
           (reason #(construct (update-in request [:frontend :emission] (fn [emission] (into {} emission))))))
        "an equal copied module is not the historical frontend emission")
    (doseq [changed [(assoc request :arguments ['missing])
                     (assoc request :caller-expression '(foreign/sum input))
                     (assoc request :arguments ['(identity input)])
                     (assoc request :source-position -1)
                     (assoc-in request [:values 'input] (av/tensor {:dtype :float :shape [4]}))
                     (assoc-in request [:values 'input] (av/tensor {:dtype :double :shape []}))]]
      (is (= :terminal-reduction-import-arguments (reason #(construct changed)))))
    (let [observe bytecode/retained-input-reduction-emission]
      (doseq [change [#(assoc-in % [:result-fact :builtin-class] Float)
                      #(assoc-in % [:result-fact :presence] :unknown)
                      #(assoc-in % [:root-result :result-type] :double)
                      #(assoc-in % [:root-result :context :void-context] true)
                      #(dissoc % :root-result)]]
        (with-redefs [bytecode/retained-input-reduction-emission
                      (fn [& arguments] (some-> (apply observe arguments) change))]
          (is (= :terminal-reduction-import-return (reason #(construct request)))))))
    (let [other (selected-frontend #'nonterminal-sum)]
      (is (= :terminal-reduction-import-root (reason #(construct (options other))))))
    (let [imported (construct request)
          source (get-in original [:source-entry :source-var])
          receiver (get-in original [:definition :receiver-var])]
      (with-redefs-fn {receiver (Object.)}
        #(is (some? (reason (fn [] (source-import/validate-terminal-reduction-import! imported))))))
      (with-redefs-fn {source (fn [& _] (throw (AssertionError. "must never execute")))}
        #(is (some? (reason (fn [] (construct request)))))))))

(deftest actual-dot-product-retains-policy-and-fresh-component-identities
  (let [original (selected-frontend #'par/dot-product)
        arguments '[left right]
        request (assoc (options original)
                       :arguments arguments
                       :caller-expression (list* '.invk (get-in original [:definition :implementation-op]) arguments)
                       :values (zipmap arguments (repeat (av/tensor {:dtype :double :shape [7]})))
                       :array-types (zipmap arguments (repeat :double)))
        first-import (binding [*unchecked-math* true] (construct request))
        second-import (binding [*unchecked-math* :warn-on-boxed] (construct request))]
    (is (= (get-in original [:source-entry :source-policy]) (:source-policy first-import)))
    (is (= (:source-policy first-import) (:source-policy second-import)))
    (is (= #{'left 'right} (set (:inputs (dialect/facts (get-in first-import [:component :typed-program]))))))
    (is (not= (get-in first-import [:component :outputs]) (get-in second-import [:component :outputs]))
        "independent imports receive ordinary hygienically fresh results")
    (is (= [Double/TYPE Double/TYPE]
           (mapv #(.getComponentType ^Class (:parameter-carrier %))
                 (get-in first-import [:obligations :inputs]))))
    (is (= {:source-position 3 :status :unresolved} (get-in first-import [:obligations :placement])))
    (is (= :terminal-reduction-import-correspondence
           (reason #(source-import/validate-terminal-reduction-import!
                     (assoc first-import :normalized-source (:normalized-source second-import))))))))

(deftest preceding-functional-map-supplies-the-actual-dot-import-arguments
  (let [original (selected-frontend #'par/dot-product)
        receiver (get-in original [:definition :implementation-op])
        arguments '[scaled input]
        request (assoc (options original) :arguments arguments :source-position 1
                       :caller-expression (list* '.invk receiver arguments)
                       :array-types {'scaled :double 'input :double})
        supplied-scaled (av/tensor {:dtype :double :shape [4]})
        at-call (atom nil)
        imported-at-call (atom nil)
        advance @#'frontend/advance-source-description
        source (list 'let*
                     ['scaled '(raster.par/pmap i 4 double
                                 (* (double coefficient) (clojure.core/aget input i)))
                      (:result-symbol request) (:caller-expression request)
                      'scaled '(clojure.core/println scaled)
                      'later '(raster.par/pmap j 17 double (double j))]
                     [(:result-symbol request) 'later])
        analyzed (with-redefs-fn
                   {#'frontend/advance-source-description
                    (fn [dtype segmented? state [ordinal _ :as binding]]
                      (when (and (= 1 ordinal)
                                 (identical? (:caller-expression request) (second (second binding))))
                        ;; Consume the preceding map's facts BEFORE the helper
                        ;; or any suffix description has been analyzed.
                        (let [before (get-in state [:source-value-analysis :values])]
                          (reset! at-call before)
                          (reset! imported-at-call (construct (assoc request :values before)))))
                      (advance dtype segmented? state binding))}
                   #(frontend/form->program-components
                    source {:dtype :double :array-types {'input :double}
                          :scalar-types {'coefficient :float}
                          :values {'input (av/tensor {:dtype :double :shape [4]})
                                   'coefficient (av/tensor {:dtype :float :shape []})
                                   'scaled supplied-scaled
                                   'later (av/tensor {:dtype :double :shape [17]})}
                          :lexical-inputs '[input coefficient] :segmented-plans? true}))
        before (get-in analyzed [:preceding-values 1])
        imported @imported-at-call]
    (is (= before @at-call) "the published prefix is the actual input to next-binding analysis")
    (is (= #{'input 'coefficient} (set (keys (get-in analyzed [:preceding-values 0])))))
    (is (= :double (get-in before ['scaled :dtype])))
    (is (= [4] (get-in before ['scaled :shape])))
    (is (not (identical? supplied-scaled (get before 'scaled)))
        "a later/refined global entry is not the producer of an earlier logical value")
    (is (not (contains? before 'later)))
    (is (not (contains? (get-in analyzed [:preceding-values 2]) 'later)))
    (is (contains? (get-in analyzed [:preceding-values 2]) 'scaled))
    (is (not (contains? (get-in analyzed [:preceding-values 3]) 'scaled))
        "a host-side shadow must invalidate the previous numerical binding's facts")
    (is (= ['scaled 'input] (mapv :argument (get-in imported [:obligations :inputs]))))
    (is (every? #(= :unresolved (:status %)) (get-in imported [:obligations :inputs])))
    (is (= :unresolved (get-in imported [:obligations :placement :status])))))

(deftest import-receipt-retains-only-its-local-construction-history
  (doseq [[construct! validate!]
          [[source-import/analyze-terminal-reduction-import source-import/validate-terminal-reduction-import!]
           [source-import/terminal-reduction-return-contract source-import/validate-terminal-reduction-return!]]]
   (let [request (options (selected-frontend #'par/sum))
        parent-checks (atom 0)
        unrelated (reify entry/RetainedProducerInput
                    (assert-producer-current! [this] (swap! parent-checks inc) this))
        parent (atom {:static [] :values [] :producer-inputs [unrelated]})
        imported (binding [entry/*emission-inputs* parent]
                   (construct! request))]
    (is (pos? @parent-checks) "construction still fences the parent's original history")
    (is (some #(identical? unrelated %) (:producer-inputs @parent))
        "independent construction adopts child inputs without discarding parent dependencies")
    (reset! parent-checks 0)
    (validate! imported)
    (is (zero? @parent-checks)
        "receipt validation must not recursively revisit unrelated parent producers")
    (entry/assert-definition-inputs-current! @parent)
    (is (= 1 @parent-checks) "the enclosing owner still validates its own producer exactly once")
    (let [stale (reify entry/RetainedProducerInput
                  (assert-producer-current! [_]
                    (throw (ex-info "stale parent" {:reason ::stale-parent}))))]
      (is (= ::stale-parent
             (reason #(binding [entry/*emission-inputs* (atom {:static [] :values []
                                                              :producer-inputs [stale]})]
                        (construct! request))))
          "isolating child history must not bypass the parent's freshness fence")))))

(defn- planned-completion-source
  ([] (planned-completion-source false))
  ([configuration]
  (let [{:keys [alias? prefix?]} (if (map? configuration) configuration {:alias? configuration})
        rounded (with-meta 'rounded {:tag 'float :raster.type/tag 'float})
        input (with-meta 'input {:tag 'Object})
        operand (if alias? (with-meta 'aliased {:tag 'Object}) input)
        expression (completion/complete-value
                    (completion/source-floating-cast-contract :float {:unchecked-math false}) operand)
        prefix (if prefix?
                 ['early-written '(raster.par/map-void! i 1
                                    (clojure.core/aset early i (float 17.0)))] [])
        source (list 'let* (into (into prefix (if alias? [operand input] []))
                                [rounded expression 'out '(clojure.core/float-array 1)
                                 'written (list 'raster.par/map-void! 'i 1
                                                (list 'clojure.core/aset 'out 'i rounded))]) 'out)
        options {:dtype :float
                 :values (cond-> {input (av/make {:kind :opaque :memory-space :host :ownership :borrowed})}
                           prefix? (assoc 'early (av/tensor {:dtype :float :shape [1]})))
                 :public-parameters (cond-> [input] prefix? (conj 'early))
                 :active-params (cond-> [input] prefix? (conj 'early))
                 :array-types (if prefix? {'early :float} {}) :scalar-types {} :source-ns *ns*
                 :segmented-plans? true :resident-reductions? true}
        program (:program (typed-route/attempt source :float {} options))
        semantic (structured/promote-program program options)
        plan (get-in semantic [:attributes :invocation-plan])
        step (first (filter invocation/source-completion? (:steps plan)))]
    {:source-program semantic :plan plan :step step})))

(defn- completion-adapter [request]
  (entry/call-with-definition-inputs #(source-import/compile-source-completion-step request)))

(deftest planned-completion-adapter-is-owned-by-the-real-source-plan-and-step
  (let [request (assoc (planned-completion-source) :frontend (selected-frontend #'par/sum))
        {:keys [source-program plan step]} request
        visits (atom 0)
        parent-guard (reify entry/RetainedProducerInput
                       (assert-producer-current! [this] (swap! visits inc) this))
        parent (atom {:static [] :values [] :producer-inputs [parent-guard]})
        forbidden (fn [& _] (throw (AssertionError. "adapter must not analyze, prepare or JIT source")))
        adapter (binding [entry/*emission-inputs* parent]
                  (with-redefs [frontend/normalize-source forbidden
                                frontend/form->program-components forbidden
                                entry/prepare-frontend-source! forbidden
                                core/ensure-current-owned-bytecode! forbidden]
                    (source-import/compile-source-completion-step request)))
        method ^java.lang.reflect.Method (get-in adapter [:region :method])
        invoke (get-in adapter [:region :invoke])]
    (is (invocation/source-completion? step))
    (is (seq (:operations (first (:equations source-program))))
        "the fixture owns an actual analyzed array writer, not a placeholder operation")
    (is (pos? @visits))
    (reset! visits 0)
    (is (identical? adapter (source-import/validate-source-completion-step! adapter source-program plan step)))
    (is (= [Object] (vec (.getParameterTypes method))))
    (is (identical? Float/TYPE (.getReturnType method)))
    (is (identical? (:expression step) (:expression adapter)))
    (is (= 0 (:source-position adapter)))
    (is (instance? Float (invoke (double -0.0))))
    (is (= (Float/floatToRawIntBits (float -0.0))
           (Float/floatToRawIntBits (invoke (double -0.0)))))
    (is (thrown? IllegalArgumentException (invoke Double/MAX_VALUE)))
    (with-redefs-fn {(ns-resolve 'raster.compiler.passes.scalar.source-import
                                'source-completion-binding!) forbidden}
      #(do
         (is (identical? adapter
                         (source-import/validate-source-completion-step! adapter source-program plan step)))
         (is (== 0.25 (invoke (double 0.25))))))
    (is (zero? @visits) "adapter and region guards do not capture unrelated parent history")
    (doseq [[p pl s] [[(with-meta source-program {:copy true}) plan step]
                       [source-program (with-meta plan {:copy true}) step]
                       [source-program plan (with-meta step {:copy true})]]]
      (is (= :source-completion-step-correspondence
             (reason #(source-import/validate-source-completion-step! adapter p pl s)))))
    (doseq [changed [(dissoc adapter :receipt)
                     (assoc adapter :region (assoc (:region adapter) :invoke (constantly (float 1.0))))]]
      (is (= :source-completion-step-correspondence
             (reason #(source-import/validate-source-completion-step! changed source-program plan step)))))
    (let [calls (atom 0)
          number (proxy [Number] []
                   (intValue [] (throw (AssertionError. "wrong overload")))
                   (longValue [] (throw (AssertionError. "wrong overload")))
                   (floatValue [] (swap! calls inc) (float 0.25))
                   (doubleValue [] (swap! calls inc) (double 0.25)))
          source (get-in request [:frontend :source-entry :source-var])]
      (is (== 0.25 (invoke number)))
      (is (pos? @calls))
      (reset! calls 0)
      (with-redefs-fn {source (fn [& _] (throw (AssertionError. "must not invoke owner")))}
        #(do
           (is (some? (reason (fn []
                                (source-import/validate-source-completion-step! adapter source-program plan step)))))
           (is (thrown? clojure.lang.ExceptionInfo (invoke number)))))
      (is (zero? @calls) "stale owner refusal precedes Number conversion effects"))))

(deftest planned-completion-adapter-refuses-detached-or-altered-source-bindings
  (let [request (assoc (planned-completion-source) :frontend (selected-frontend #'par/sum))
        {:keys [source-program plan step]} request
        bindings (second (:source source-program))
        replace-bindings (fn [bindings]
                           (assoc request :source-program
                                  (assoc source-program :source
                                         (with-meta (list 'let* bindings 'out)
                                           (meta (:source source-program))))))]
    (is (= :source-completion-step-owner
           (reason #(source-import/compile-source-completion-step request))))
    (is (= :source-completion-step-owner
           (reason #(completion-adapter (assoc request :step (with-meta step {:copy true}))))))
    (is (= :source-completion-step-owner
           (reason #(completion-adapter (assoc request :plan (with-meta plan {:copy true}))))))
    (doseq [[changed expected]
            [[(assoc bindings 0 (with-meta (first bindings) {:tag 'double}))
              :invocation-source-step-mismatch]
             [(assoc bindings 1 (with-meta (apply list (second bindings))
                                   (meta (second bindings))))
              :source-completion-step-source]]]
      (is (= expected
             (reason #(completion-adapter (replace-bindings changed))))))
    (let [reordered (vec (concat (subvec bindings 2 4) (subvec bindings 0 2) (subvec bindings 4)))]
      (is (some? (reason #(completion-adapter (replace-bindings reordered))))))
    (is (= :source-completion-step-frontend
           (reason #(completion-adapter (assoc-in request [:frontend :emission] nil)))))
    (is (= :source-completion-step-frontend
           (reason #(completion-adapter
                     (update-in request [:frontend :emission] (fn [emission] (into {} emission)))))))))
