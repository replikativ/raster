(ns raster.compiler.passes.parallel.terminal-reduction-import-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.binding-completion :as binding-completion]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.passes.parallel.structured-control-route :as structured]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.scalar.source-import :as source-import]
            [raster.compiler.pipeline :as pipeline]))

(core/deftm nonterminal-sum [xs :- (Array double)] :- Double
  (do (par/sum xs) 0.0))

(core/deftm public-float-return [xs :- (Array double)] :- Float
  (float (aget xs 0)))

(core/deftm public-float-entry [value :- Float] :- Float value)

(core/deftm public-object-float-entry [value :- Object] :- Float (float value))

(defn- selected-frontend [generic]
  (let [source (core/resolve-deftm-var generic)]
    (core/prepare-emission-source-walk! source)
    (entry/call-with-definition-inputs
     #(let [original (entry/original-source-frontend source)]
        (entry/prepare-frontend-source! original source core/ensure-current-owned-bytecode!)))))

(defn- options [original]
  {:frontend original :input-body (:body original) :root (first (:body original))})

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
(deftest preceding-functional-map-supplies-the-actual-dot-call-prefix
  (let [original (selected-frontend #'par/dot-product)
        receiver (get-in original [:definition :implementation-op])
        arguments '[scaled input]
        caller-expression (list* '.invk receiver arguments)
        result-symbol (with-meta 'answer {:tag 'Object :raster.type/tag 'double})
        supplied-scaled (av/tensor {:dtype :double :shape [4]})
        at-call (atom nil)
        advance @#'frontend/advance-source-description
        source (list 'let*
                     ['scaled '(raster.par/pmap i 4 double
                                 (* (double coefficient) (clojure.core/aget input i)))
                      result-symbol caller-expression
                      'scaled '(clojure.core/println scaled)
                      'later '(raster.par/pmap j 17 double (double j))]
                     [result-symbol 'later])
        analyzed (with-redefs-fn
                   {#'frontend/advance-source-description
                    (fn [dtype segmented? state [ordinal _ :as binding]]
                      (when (and (= 1 ordinal)
                                 (identical? caller-expression (second (second binding))))
                        ;; Consume the preceding map's facts BEFORE the helper
                        ;; or any suffix description has been analyzed.
                        (reset! at-call (get-in state [:source-value-analysis :values])))
                      (advance dtype segmented? state binding))}
                   #(frontend/form->program-components
                    source {:dtype :double :array-types {'input :double}
                          :scalar-types {'coefficient :float}
                          :values {'input (av/tensor {:dtype :double :shape [4]})
                                   'coefficient (av/tensor {:dtype :float :shape []})
                                   'scaled supplied-scaled
                                   'later (av/tensor {:dtype :double :shape [17]})}
                          :lexical-inputs '[input coefficient] :segmented-plans? true}))
        before (get-in analyzed [:preceding-values 1])]
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
    (is (= [{:dtype :double :shape [4]} {:dtype :double :shape [4]}]
           (mapv #(select-keys (get before %) [:dtype :shape]) arguments))
        "the actual lexical helper arguments see only their preceding producers")))

(deftest terminal-return-receipt-retains-only-its-local-construction-history
  (let [construct! source-import/terminal-reduction-return-contract
        validate! source-import/validate-terminal-reduction-return!
        request (options (selected-frontend #'par/sum))
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
          "isolating child history must not bypass the parent's freshness fence"))))

(defn- contextual-witness-request [hint]
  (let [original (selected-frontend #'par/sum)
        returned (return-contract (select-keys (options original) [:frontend :input-body :root]))
        object (completion/canonical-parts (:recipe returned))
        converted (completion/canonical-parts (:operand object))
        contextual (completion/canonical-parts (:operand converted))
        binder (with-meta 'current-reduction {:tag hint :raster.type/tag 'double})
        expression (completion/complete-value (:contract contextual) (:root returned))
        scope (list 'let* [binder expression] binder)
        lineage (:source-entry original)]
    {:source scope :binder binder :expression expression
     :options {:source-var (:source-var lineage) :source-ns (:source-ns lineage)
               :lexical-inputs (:params lineage) :source-policy (:source-policy lineage)}}))

(deftest current-source-witness-uses-actual-entry-and-binding-carriers
  (let [{:keys [source binder expression options]} (contextual-witness-request 'Object)
        witness (entry/call-with-definition-inputs
                 #(source-import/prepare-contextual-source-witness source options))
        proof (entry/call-with-definition-inputs
               #(source-import/observe-contextual-reduction! witness 0 binder expression))]
    (is (identical? source (:scope witness)))
    (is (identical? (:operand (completion/canonical-parts expression)) (:root proof)))
    (is (= 'double (:result-tag proof)))
    (is (identical? (.arrayType Double/TYPE)
                    (get-in proof [:before (first (:lexical-inputs options)) :reference-class])))
    (is (= :contextual-source-witness-owner
           (reason #(source-import/observe-contextual-reduction! witness 0 binder expression))))
    (doseq [[position changed-binder changed-expression]
            [[1 binder expression]
             [0 (with-meta (symbol (name binder)) (meta binder)) expression]
             [0 binder (with-meta (apply list expression) (meta expression))]]]
      (is (= :contextual-source-reduction-form
             (reason #(entry/call-with-definition-inputs
                       (fn [] (source-import/observe-contextual-reduction!
                               witness position changed-binder changed-expression)))))))
    (doseq [changed [(assoc options :source-ns *ns*)
                     (assoc options :lexical-inputs '[foreign])
                     (update-in options [:source-policy :unchecked-math] not)]]
      (with-redefs [core/emit-owned-bytecode! (fn [& _] (throw (AssertionError. "invalid witness emitted code")))]
        (is (= :contextual-source-witness-entry
               (reason #(entry/call-with-definition-inputs
                         (fn [] (source-import/prepare-contextual-source-witness source changed))))))))
    (doseq [hint ['String 'doubles]]
      (let [{:keys [source binder expression options]} (contextual-witness-request hint)]
        (is (= :contextual-source-reduction-observation
               (reason #(entry/call-with-definition-inputs
                         (fn []
                           (let [witness (source-import/prepare-contextual-source-witness source options)]
                             (source-import/observe-contextual-reduction! witness 0 binder expression))))))
            "a normal Double root does not discharge an incompatible binding CHECKCAST")))
    (let [{:keys [source binder expression options]} (contextual-witness-request 'double)]
      (is (= 'double
             (:result-tag
              (entry/call-with-definition-inputs
               #(source-import/observe-contextual-reduction!
                 (source-import/prepare-contextual-source-witness source options) 0 binder expression))))
          "a logical Double hint does not replace the actual observed Object carrier"))))

(deftest cached-current-source-witness-reuse-retains-the-original-guard
  (let [{:keys [source binder expression options]} (contextual-witness-request 'Object)
        parent (atom {:static [] :values []})
        initial @parent
        cached (atom nil)]
    (binding [entry/*emission-inputs* parent]
      (is (nil? (entry/call-with-optional-definition-inputs
                 #(do (reset! cached (source-import/prepare-contextual-source-witness source options)) nil))))
      (is (identical? initial @parent) "declining the first attempt rolls back its adoption")
      (is (= 'double (:result-tag (source-import/observe-contextual-reduction! @cached 0 binder expression))))
      (is (some #(identical? (:receipt @cached) %) (:producer-inputs @parent))
          "cache reuse must adopt the original receipt in the successful owner"))
    (with-redefs-fn
      {(:source-var options) (fn [& _] (throw (AssertionError. "source must not execute")))}
      #(is (= :stale-definition-proof-input
              (reason (fn [] (entry/assert-definition-inputs-current! @parent))))
           "publication history still detects a replaced source after decline and reuse"))))

(deftest current-reference-provenance-uses-before-class-and-adopts-only-accepted-witnesses
  (let [{:keys [binder expression options]} (contextual-witness-request 'Object)
        formal (first (:lexical-inputs options))
        boxed (with-meta 'boxed-array {:tag 'Object :raster.type/tag 'doubles})
        checked (with-meta 'checked-array {:tag 'doubles})
        compatible (with-meta 'compatible-array {:tag 'doubles})
        incompatible (with-meta 'incompatible-array {:tag 'floats})
        alias (with-meta 'array-alias {:tag 'doubles})
        numeric (with-meta 'numeric-input {:tag 'double})
        bindings [boxed (completion/complete-value completion/object-input-contract formal)
                  checked (completion/complete-value
                           (completion/bytecode-boxed-parameter-contract (.arrayType Double/TYPE)) boxed)
                  compatible (completion/complete-value
                              (completion/bytecode-boxed-parameter-contract (.arrayType Double/TYPE)) formal)
                  incompatible (completion/complete-value
                                (completion/bytecode-boxed-parameter-contract (.arrayType Float/TYPE)) formal)
                  alias formal
                  numeric (completion/complete-value (completion/bytecode-boxed-parameter-contract Double/TYPE) formal)
                  binder expression]
        source (list 'let* bindings binder)
        parent (atom {:static [] :values []})
        initial @parent
        cached (atom nil)
        accepts (atom [])
        accept! (fn [proposal] (swap! accepts conj proposal) true)
        observe (fn [ordinal predicate]
                  (source-import/observe-reference-provenance!
                   @cached ordinal (nth bindings (* 2 ordinal))
                   (nth bindings (inc (* 2 ordinal))) predicate))]
    (binding [entry/*emission-inputs* parent]
      (is (nil? (entry/call-with-optional-definition-inputs
                 #(do (reset! cached (source-import/prepare-contextual-source-witness source options)) nil))))
      (is (identical? initial @parent))
      (let [context (bytecode/retained-input-binding-context
                     (:emission @cached) (:input-body @cached) source 1)]
        (is (identical? Object (get-in context [:before boxed :reference-class])))
        (is (identical? (.arrayType Double/TYPE) (get-in context [:after :reference-class]))))
      (doseq [ordinal [1 3 5]]
        (is (nil? (observe ordinal accept!))))
      (is (empty? @accepts) "unknown-before, incompatible target and primitive conversion never reach acceptance")
      (is (identical? initial @parent) "an after-array Class cannot adopt a provenance witness")
      (let [marker (reify entry/RetainedProducerInput (assert-producer-current! [this] this))]
        (is (nil? (observe 2 (fn [_]
                              (entry/adopt-definition-inputs! {:producer-inputs [marker]}) false))))
        (is (identical? initial @parent) "a rejected declaration predicate rolls back its own observations"))
      (doseq [ordinal [0 2 4]]
        (let [proposal (observe ordinal accept!)]
          (is (= {:formal formal :carrier (.arrayType Double/TYPE) :current-source? true} proposal))))
      (is (= 3 (count @accepts)))
      (is (some #(identical? (:receipt @cached) %) (:producer-inputs @parent))
          "all successful source projections adopt the SAME cached witness")
      (is (= :contextual-source-reference-form
             (reason #(source-import/observe-reference-provenance!
                       @cached 0 (with-meta (symbol (name boxed)) (meta boxed))
                       (second bindings) accept!))))
      (is (= :contextual-source-reference-form
             (reason #(source-import/observe-reference-provenance!
                       @cached 0 boxed (with-meta (apply list (second bindings)) (meta (second bindings))) accept!)))))
    (with-redefs-fn
      {(:source-var options) (fn [& _] (throw (AssertionError. "must not execute source")))}
      #(is (= :stale-definition-proof-input
              (reason (fn [] (entry/assert-definition-inputs-current! @parent))))))))

(deftest functional-map-reference-projection-is-batched-before-current-reduction-analysis
  (entry/call-with-definition-inputs
   (fn []
     (let [{:keys [binder expression options]} (contextual-witness-request 'Object)
           formal (first (:lexical-inputs options))
           array-id #(with-meta % {:tag 'doubles :raster.type/tag 'doubles})
           mapped (array-id 'mapped)
           first-alias (array-id 'first-alias)
           second-alias (array-id 'second-alias)
           checked #(completion/complete-value
                     (completion/bytecode-reference-input-contract
                      (.arrayType Double/TYPE) (completion/bytecode-operand-context {:object-result? true})) %)
           root (completion/canonical-parts expression)
           expression (completion/complete-value
                       (:contract root) (util/subst-syms {formal second-alias} (:operand root)))
           source (list 'let* ['early (list 'clojure.core/aset formal 0 17.0)
                              mapped (list 'raster.par/pmap 'row (list 'clojure.core/alength formal)
                                           'double (list 'clojure.core/aget formal 'row))
                              first-alias (checked mapped) second-alias (checked first-alias)
                              binder expression] binder)
           options (merge options {:dtype :double :public-parameters (:lexical-inputs options)
                                   :array-types {formal :double} :scalar-types {}
                                   :source-binding-context (binding-completion/capture-resolution-context
                                                            #'bytecode/binding-completion-plan)})
           construct (#'pipeline/contextual-reduction-constructor options)
           emit source-import/prepare-contextual-source-witness
           owners (atom [])
           result (with-redefs [source-import/prepare-contextual-source-witness
                               (fn [current options]
                                 (let [owner (emit current options)] (swap! owners conj owner) owner))]
                    (frontend/call-with-contextual-reduction-constructor
                     construct
                     #(let [normalized (frontend/normalize-source source options)]
                        {:source normalized :analysis (frontend/source-fragment-analysis normalized options)})))]
       (is (= 2 (count @owners)) "one pre-projection witness, one for the changed source—not one per cast")
       (is (not-any? completion/reference-input-parts
                     (take-nth 2 (rest (second (:source result))))))
       (is (= 1 (count (filter #(= :reduce (:kind %)) (get-in result [:analysis :descriptions])))))
       (is (not (identical? (:scope (first @owners)) (:scope (second @owners)))))
       (doseq [values [[1.0 2.0 3.0] nil]]
         (let [invoke (fn [owner]
                        (let [input (when values (double-array values))]
                          (try
                            [:value (clojure.lang.Reflector/invokeInstanceMethod
                                     (:implementation owner) "invk" (object-array [input]))
                             (vec input)]
                            (catch Throwable error [:error (class error)]))))]
           (is (= (invoke (first @owners)) (invoke (second @owners)))
               "prefix stores and null failure survive reference-check projection")))))))

(defn- public-return-witness [original]
  (let [lineage (:source-entry original)
        expression (completion/complete-value
                    (completion/source-floating-cast-contract :float (:source-policy lineage))
                    (list 'clojure.core/aget (first (:params lineage)) 0))
        source (list 'let* ['published expression] 'published)]
    (entry/call-with-definition-inputs
     #(source-import/prepare-contextual-source-witness
       source {:source-var (:source-var lineage) :source-ns (:source-ns lineage)
               :lexical-inputs (:params lineage) :source-policy (:source-policy lineage)}))))

(deftest original-public-float-return-uses-the-prepared-original-method-not-the-object-witness
  (let [original (selected-frontend #'public-float-return)
        witness (public-return-witness original)
        forbidden (fn [& _] (throw (AssertionError. "public return query must not prepare or emit code")))
        returned (with-redefs [entry/prepare-frontend-source! forbidden
                              core/ensure-current-owned-bytecode! forbidden
                              core/emit-owned-bytecode! forbidden]
                   (entry/call-with-definition-inputs
                    #(source-import/original-public-float-return-contract original witness)))
        boxed (completion/canonical-parts (:recipe returned))
        converted (completion/canonical-parts (:operand boxed))]
    (is (identical? returned (source-import/validate-original-public-float-return! returned witness)))
    (is (identical? original (:frontend returned)))
    (is (identical? witness (:witness returned)))
    (is (identical? Float/TYPE (:result-carrier returned)))
    (is (identical? Float/TYPE (.getReturnType ^java.lang.reflect.Method (:compute-method returned))))
    (is (identical? Object (.getReturnType ^java.lang.reflect.Method (:method witness))))
    (is (not (identical? (:compute-method returned) (:method witness))))
    (is (= (:parameters witness) (:parameters returned)))
    (is (= (:parameter-carriers witness) (:parameter-carriers returned)))
    (is (= (:source-policy witness) (:source-policy returned)))
    (is (= completion/object-input-contract (:contract boxed)))
    (is (identical? Float/TYPE (get-in converted [:contract :target-carrier])))
    (doseq [changed [(dissoc returned :receipt)
                     (assoc returned :recipe (apply list (:recipe returned)))
                     (assoc returned :result-carrier Double/TYPE)
                     (assoc returned :parameter-carriers [(.arrayType Float/TYPE)])]]
      (is (= :original-public-float-return-correspondence
             (reason #(source-import/validate-original-public-float-return! changed)))))
    (is (= :original-public-float-return-correspondence
           (reason #(source-import/validate-original-public-float-return! returned (into {} witness)))))
    (let [bridge entry/frontend-boxed-entry-bridge]
      (doseq [change [#(assoc % :payload-carrier Double/TYPE)
                      #(assoc % :compute-method (:method witness))
                      #(assoc % :compute-parameter-carriers [(.arrayType Float/TYPE)])
                      #(assoc % :body-context (assoc (:body-context %) :return-stack-type :double))]]
        (is (= :original-public-float-return
               (with-redefs [entry/frontend-boxed-entry-bridge (fn [frontend] (some-> (bridge frontend) change))]
                 (reason #(entry/call-with-definition-inputs
                           (fn [] (source-import/original-public-float-return-contract original witness)))))))))
    (with-redefs-fn
      {(get-in original [:source-entry :source-var]) (fn [& _] (throw (AssertionError. "must not execute")))}
      #(is (= :stale-definition-proof-input
              (reason (fn [] (source-import/validate-original-public-float-return! returned))))))))

(deftest original-public-float-return-refuses-a-real-double-compute-bridge
  (let [original (selected-frontend #'par/sum)
        witness (public-return-witness original)]
    (is (= :original-public-float-return
           (reason #(entry/call-with-definition-inputs
                     (fn [] (source-import/original-public-float-return-contract original witness))))))))

(defn- terminal-float-request [original variant]
  (let [lineage (:source-entry original)
        parameter (first (:params lineage))
        boxed (completion/complete-value completion/object-input-contract
                                         (list 'clojure.core/aget parameter 0))
        rounded (if (= variant :misleading-hint) (with-meta 'rounded {:tag 'double}) 'rounded)
        cast (completion/complete-value
              (completion/source-floating-cast-contract
               (if (= variant :unchecked) :unchecked-float :float) (:source-policy lineage)) 'boxed)
        bindings (into ['boxed boxed rounded cast 'alias rounded
                        'boxed-tail (completion/complete-value completion/object-input-contract 'alias)
                        'tail 'boxed-tail]
                       (case variant
                         :extra-use ['observer '(do rounded 1)]
                         :capture ['observer '(fn* [] rounded)]
                         []))
        terminal (if (= variant :branch) '(if true tail tail) 'tail)
        source (list 'let* bindings terminal)]
    (entry/call-with-definition-inputs
     #(let [witness (source-import/prepare-contextual-source-witness
                     source {:source-var (:source-var lineage) :source-ns (:source-ns lineage)
                             :lexical-inputs (:params lineage) :source-policy (:source-policy lineage)})]
        (source-import/original-public-float-return-contract original witness)))))

(deftest terminal-float-publication-observes-only-the-sole-consumer-checked-tail
  (let [original (selected-frontend #'public-float-return)
        returned (terminal-float-request original nil)
        forbidden (fn [& _] (throw (AssertionError. "terminal observation must not compile source")))
        publication (with-redefs [core/emit-owned-bytecode! forbidden
                                 core/ensure-current-owned-bytecode! forbidden
                                 entry/prepare-frontend-source! forbidden]
                      (entry/call-with-definition-inputs
                       #(source-import/observe-terminal-float-publication returned)))]
    (is (identical? publication (source-import/validate-terminal-float-publication! publication)))
    (is (identical? returned (:public-return publication)))
    (is (identical? (:witness returned) (:witness publication)))
    (is (identical? (get-in returned [:witness :scope]) (:source publication)))
    (is (= 'rounded (:result publication)))
    (is (= 1 (:ordinal publication)))
    (is (= 'tail (:terminal publication)))
    (is (= [:identity :object-input :identity :checked-float] (mapv :kind (:chain publication))))
    (doseq [changed [(dissoc publication :receipt)
                     (assoc publication :chain (into [] (:chain publication)))
                     (assoc publication :source (apply list (:source publication)))]]
      (is (= :terminal-float-publication-correspondence
             (reason #(source-import/validate-terminal-float-publication! changed)))))
    (let [candidate (terminal-float-request original :misleading-hint)
          witness (:witness candidate)
          binding (bytecode/retained-input-binding-context
                   (:emission witness) (:input-body witness) (:scope witness) 1)]
      (is (= 'double (:tag (meta (nth (second (:scope witness)) 2)))))
      (is (= :float (get-in binding [:after :type]))
          "the misleading primitive hint does not change this actual emitted slot")
      (is (= 'rounded (:result (entry/call-with-definition-inputs
                                #(source-import/observe-terminal-float-publication candidate))))))
    (let [observe-binding bytecode/retained-input-binding-context]
      (is (= :terminal-float-publication
             (with-redefs [bytecode/retained-input-binding-context
                           (fn [emission input-body scope ordinal]
                             (let [observed (observe-binding emission input-body scope ordinal)]
                               (if (= 1 ordinal)
                                 (assoc-in observed [:after :type] :double)
                                 observed)))]
               (reason #(entry/call-with-definition-inputs
                         (fn [] (source-import/observe-terminal-float-publication returned))))))
          "an incompatible actual after-carrier refuses even with a genuine checked Float completion"))
    (doseq [variant [:unchecked :extra-use :capture :branch]
            :let [candidate (terminal-float-request original variant)]]
      (is (= :terminal-float-publication
             (reason #(entry/call-with-definition-inputs
                       (fn [] (source-import/observe-terminal-float-publication candidate)))))
          (name variant)))))

(deftest terminal-float-footer-preserves-the-checked-source-prefix
  (doseq [variant [nil :misleading-hint]]
   (let [original (selected-frontend #'public-float-return)
        returned (terminal-float-request original variant)
        publication (entry/call-with-definition-inputs
                     #(source-import/observe-terminal-float-publication returned))
        {:keys [form buffer writer]}
        (entry/call-with-definition-inputs
         #(source-import/construct-terminal-float-publication publication))
        source (:source publication)
        original-bindings (second source)
        bindings (second form)
        witness (:witness publication)
        compiled (core/emit-owned-bytecode!
                  bytecode/compile-whole-source-impl! "raster.compiled.FloatPublicationFooter"
                  (:parameters witness) (:tags witness) 'float [form] (:source-ns witness)
                  (:interface-class witness) 'float-publication-footer)
        observe (fn [function x]
                  (try
                    (let [value (function (double-array [x]))]
                      [:float (Float/floatToRawIntBits value)])
                    (catch Throwable error [:error (class error)])))
        reference (get-in original [:implementation])
        actual (:implementation compiled)]
    (is (= (+ 4 (count original-bindings)) (count bindings)))
    (is (every? true? (map identical? original-bindings bindings))
        "all original bindings, including checked conversion and aliases, stay in source order")
    (is (= [buffer '(clojure.core/float-array 1)]
           (subvec bindings (count original-bindings) (+ 2 (count original-bindings)))))
    (is (= writer (nth bindings (+ 2 (count original-bindings)))))
    (is (= (list 'clojure.core/float (list 'clojure.core/aget buffer 0)) (nth form 2)))
    (doseq [x [1.25 -0.0 Double/NaN (Double/longBitsToDouble 0x7ff8002460000000)
              Double/POSITIVE_INFINITY Double/MAX_VALUE]]
      (is (= (observe reference x) (observe actual x))
          (str "the new footer preserves Float raw bits or original failure for " x))))))

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

(defn- witnessed-result-import-source [target]
  (entry/call-with-definition-inputs
   (fn []
     (let [original (selected-frontend #'public-float-return)
           context-contract (:contract (completion/canonical-parts
                                        (:expression (contextual-witness-request 'Object))))
           context (:operand-context context-contract)
           total (with-meta 'total {:tag 'Object :raster.type/tag 'double})
           root (completion/complete-value
                 context-contract '(raster.par/reduce acc 0.0 i 3 (+ acc (clojure.core/aget xs i))))
           snapshot (completion/complete-value
                     completion/object-input-contract
                     (if target
                       (completion/complete-value
                        (completion/planned-bytecode-input-contract target context :ref) total)
                       total))
           checked (completion/complete-value
                    (completion/source-floating-cast-contract :float {:unchecked-math false}) 'saved)
           out (with-meta 'out {:tag 'floats :raster.type/tag 'floats})
           source (list 'let* [total root 'saved snapshot 'rounded checked
                              out '(clojure.core/float-array 1)
                              'written '(raster.par/map-void! i 1 (clojure.core/aset out i rounded))] out)
           options {:dtype :double :target-dialect :opencl-portable :resident-reductions? false
                    :values {'xs (av/tensor {:dtype :double :shape [3]})}
                    :public-parameters '[xs] :active-params '[xs] :lexical-inputs '[xs]
                    :array-types {'xs :double} :scalar-types {}
                    :source-var (get-in original [:source-entry :source-var])
                    :source-ns (get-in original [:source-entry :source-ns])
                    :source-policy (get-in original [:source-entry :source-policy])
                    :source-binding-context (binding-completion/capture-resolution-context
                                             #'bytecode/binding-completion-plan)}]
       (frontend/call-with-contextual-reduction-constructor
        (#'pipeline/contextual-reduction-constructor options)
        #(let [numerical (:program (typed-route/attempt source :double {'xs :double} options))
               semantic (structured/promote-program numerical options)
               emitted (:program (c-family/emit-program (structured/schedule-program semantic options) options))
               plan (get-in semantic [:attributes :invocation-plan])
               step (first (filter invocation/program-result-import? (:steps plan)))
               witness (source-import/prepare-contextual-source-witness (:source semantic) options)]
           {:frontend original :source-program semantic :emitted emitted :plan plan :step step :witness witness}))))))

(defn- result-import-adapter [request]
  (entry/call-with-definition-inputs #(source-import/compile-program-result-import-step request)))

(deftest float-publication-routing-does-not-prepare-noncandidates
  (let [cast (completion/complete-value
              (completion/source-floating-cast-contract :float {:unchecked-math false}) 'input)
        boxed (completion/complete-value completion/object-input-contract 'rounded)
        candidate (list 'let* ['rounded cast 'answer boxed] 'answer)
        forbidden (fn [& _] (throw (AssertionError. "noncandidate prepared original source")))
        construct (#'pipeline/contextual-reduction-constructor {})]
    (is (source-import/terminal-float-publication-candidate? candidate))
    (with-redefs [core/prepare-emission-source-walk! forbidden
                  source-import/prepare-contextual-source-witness forbidden]
      (doseq [source ['(let* [x 1.0] x)
                      (list 'let* ['rounded cast 'answer boxed] '(double answer))
                      (list 'let* ['answer boxed 'rounded cast] 'answer)
                      (list 'let* ['rounded cast 'answer boxed 'rounded 1.0] 'answer)
                      (list 'let* [['rounded] cast 'answer boxed] 'answer)]]
        (is (false? (source-import/terminal-float-publication-candidate? source)))
        (is (nil? (construct {:kind :terminal-float-publication :source source})))))))

(deftest final-compilation-attaches-source-actions-to-its-exact-program
  (require 'raster.compiler.equation-first)
  (let [{:keys [frontend source-program emitted plan]}
        (witnessed-result-import-source nil)
        source-var (get-in frontend [:source-entry :source-var])
        options {:source-var source-var
                 :source-ns (get-in frontend [:source-entry :source-ns])
                 :public-parameters '[xs]}
        compilation {:semantic source-program :emitted emitted}
        attach! (ns-resolve 'raster.compiler.equation-first 'attach-source-action-adapters)
        attached (entry/call-with-definition-inputs
                  #(do (entry/demand-original-source-frontend! source-var)
                       (attach! compilation options)))
        completions (filter invocation/source-completion? (:steps plan))
        imports (filter invocation/program-result-import? (:steps plan))]
    (is (seq completions))
    (is (seq imports))
    (is (= (set (map :id completions)) (set (keys (:source-completion-adapters attached)))))
    (is (= (set (map :id imports)) (set (keys (:program-result-import-adapters attached)))))
    (doseq [step completions]
      (is (some? (source-import/validate-source-completion-step!
                  (get-in attached [:source-completion-adapters (:id step)]) source-program plan step))))
    (doseq [step imports]
      (is (some? (source-import/validate-program-result-import-step!
                  (get-in attached [:program-result-import-adapters (:id step)])
                  source-program emitted plan step))))
    (is (= :equation-first-source-action-owner
           (entry/call-with-definition-inputs #(reason (fn [] (attach! compilation {}))))))
    (let [ordinary {:semantic {:attributes {:invocation-plan {:steps []}}}}
          forbidden (fn [& _] (throw (AssertionError. "source-free compilation requested JVM readiness")))]
      (with-redefs [entry/demand-original-source-frontend! forbidden
                    entry/prepared-original-source-frontend! forbidden]
        (is (identical? ordinary (attach! ordinary {}))
            "ordinary source-free compilation does not prepare a JVM source witness")))))

(deftest current-result-import-adapter-owns-the-numerical-port-and-boxed-return-recipe
  (doseq [target [nil Double/TYPE Float/TYPE]
          :let [{:keys [source-program emitted plan step witness] :as request} (witnessed-result-import-source target)
                forbidden (fn [& _] (throw (AssertionError. "result import query must not prepare original code")))
                adapter (with-redefs [core/ensure-current-owned-bytecode! forbidden
                                      entry/prepare-frontend-source! forbidden]
                          (result-import-adapter request))
                box (get-in adapter [:region :box-region])
                return (get-in adapter [:region :return-region])
                box-method ^java.lang.reflect.Method (:method box)
                return-method ^java.lang.reflect.Method (:method return)]]
    (is (invocation/program-result-import? step))
    (is (seq (:operations (:producer adapter))) "the producer is an actual typed numerical reduction")
    (is (identical? Double (:payload-class adapter)))
    (is (identical? Double/TYPE (:payload-carrier adapter)))
    (is (= :double (:dtype adapter)))
    (is (identical? witness (:witness adapter)))
    (is (identical? adapter (source-import/validate-program-result-import-step! adapter source-program emitted plan step)))
    (is (= [Double/TYPE] (vec (.getParameterTypes box-method))))
    (is (identical? Object (.getReturnType box-method)))
    (is (= [Object] (vec (.getParameterTypes return-method))))
    (is (identical? Object (.getReturnType return-method)))
    (doseq [value [0.0 -0.0 1.25 (Double/longBitsToDouble 0x7ff8000000001234) Double/POSITIVE_INFINITY]]
      (let [root ((:invoke box) value)
            actual ((:invoke return) root)]
        (is (instance? Double root))
        (if (identical? target Float/TYPE)
          (do (is (instance? Float actual))
              (is (= (Float/floatToRawIntBits (.floatValue ^Number root)) (Float/floatToRawIntBits actual))))
          (do (is (instance? Double actual))
              (is (= (Double/doubleToRawLongBits value) (Double/doubleToRawLongBits actual)))))
        (when-not target (is (identical? root actual)))))
    (doseq [changed [(assoc adapter :payload-class Float) (assoc adapter :dtype :float)
                     (assoc adapter :producer (into {} (:producer adapter)))
                     (assoc adapter :region (assoc (:region adapter) :invoke identity))]]
      (is (= :program-result-import-correspondence
             (reason #(source-import/validate-program-result-import-step! changed source-program emitted plan step)))))
    (doseq [[program physical invocation import]
            [[(with-meta source-program {:copied true}) emitted plan step]
             [source-program (with-meta emitted {:copied true}) plan step]
             [source-program emitted (with-meta plan {:copied true}) step]
             [source-program emitted plan (with-meta step {:copied true})]]]
      (is (= :program-result-import-correspondence
             (reason #(source-import/validate-program-result-import-step! adapter program physical invocation import)))))))

(deftest current-result-import-adapter-refuses-altered-current-or-emitted-ownership
  (let [{:keys [frontend source-program emitted plan step witness] :as request} (witnessed-result-import-source nil)
        adapter (result-import-adapter request)
        observe source-import/observe-contextual-reduction!]
    (doseq [change [#(assoc-in % [:result-fact :builtin-class] Float)
                    #(assoc % :result-tag 'float)]]
      (is (= :program-result-import-payload
             (with-redefs [source-import/observe-contextual-reduction! (fn [& args] (change (apply observe args)))]
               (reason #(result-import-adapter request))))))
    (doseq [changed [(assoc request :step (with-meta step {:copied true}))
                     (assoc request :witness (assoc witness :scope (with-meta (:scope witness) {:copied true})))
                     (update-in request [:source-program :source]
                                #(list 'let* (assoc (second %) 0 'foreign-producer) (nth % 2)))
                     (update-in request [:emitted :equations 0 :algorithm]
                                #(apply list (assoc (vec %) 3 ['foreign-output])))]]
      (is (some? (reason #(result-import-adapter changed)))))
    (with-redefs-fn
      {(get-in frontend [:source-entry :source-var]) (fn [& _] (throw (AssertionError. "must not execute")))}
      #(do
         (is (some? (reason (fn [] (source-import/validate-program-result-import-step!
                                   adapter source-program emitted plan step)))))
         (is (thrown? clojure.lang.ExceptionInfo ((get-in adapter [:region :box-region :invoke]) 1.0)))
         (is (thrown? clojure.lang.ExceptionInfo ((get-in adapter [:region :return-region :invoke]) 1.0)))))))

(defn- witnessed-completion-source [original opaque-input?]
  (let [boxed (completion/complete-value completion/object-input-contract 'value)
        unboxed (completion/complete-value (completion/bytecode-boxed-parameter-contract Float/TYPE) 'boxed)
        source (list 'let* ['boxed boxed 'unboxed unboxed
                           'out '(clojure.core/float-array 1)
                           'written '(raster.par/map-void! i 1 (clojure.core/aset out i unboxed))] 'out)
        input-value (if opaque-input?
                      (av/make {:kind :opaque :memory-space :host :ownership :borrowed})
                      (av/tensor {:dtype :float :shape []}))
        options {:dtype :float :values {'value input-value}
                 :public-parameters '[value] :lexical-inputs '[value] :active-params '[value]
                 :array-types {} :scalar-types (if opaque-input? {} {'value :float})
                 :source-var (get-in original [:source-entry :source-var])
                 :source-ns (get-in original [:source-entry :source-ns])
                 :source-policy (get-in original [:source-entry :source-policy])}
        numerical (:program (typed-route/attempt source :float {} options))
        semantic (structured/promote-program numerical options)
        plan (get-in semantic [:attributes :invocation-plan])
        witness (entry/call-with-definition-inputs
                 #(source-import/prepare-contextual-source-witness (:source semantic) options))]
    {:frontend original :source-program semantic :plan plan :witness witness
     :steps (filterv invocation/source-completion? (:steps plan))}))

(defn- witnessed-literal-completion-source [value]
  (let [original (selected-frontend #'public-float-entry)
        boxed (vary-meta (completion/complete-value completion/object-input-contract value)
                         assoc :raster.effect/preserve-evaluation true)
        unboxed (completion/complete-value (completion/bytecode-boxed-parameter-contract Float/TYPE) 'boxed)
        source (list 'let* ['boxed boxed 'unboxed unboxed
                           'out '(clojure.core/float-array 1)
                           'written '(raster.par/map-void! i 1 (clojure.core/aset out i unboxed))] 'out)
        options {:dtype :float :values {'value (av/tensor {:dtype :float :shape []})}
                 :public-parameters '[value] :lexical-inputs '[value] :active-params '[value]
                 :array-types {} :scalar-types {'value :float}
                 :source-var (get-in original [:source-entry :source-var])
                 :source-ns (get-in original [:source-entry :source-ns])
                 :source-policy (get-in original [:source-entry :source-policy])}
        numerical (:program (typed-route/attempt source :float {} options))
        semantic (structured/promote-program numerical options)
        plan (get-in semantic [:attributes :invocation-plan])
        step (first (filter invocation/source-completion? (:steps plan)))
        witness (entry/call-with-definition-inputs
                 #(source-import/prepare-contextual-source-witness (:source semantic) options))]
    {:frontend original :source-program semantic :plan plan :step step :witness witness}))

(deftest witnessed-numeric-literal-completion-keeps-zero-ports-and-actual-object-carrier
  (doseq [value [(float -0.0) (double -0.0) (int 17) (long 9007199254740993)
                (byte -7) (short 257)]]
    (let [{:keys [source-program plan step witness] :as request} (witnessed-literal-completion-source value)
          adapter (completion-adapter request)
          method ^java.lang.reflect.Method (get-in adapter [:region :method])
          invoke (get-in adapter [:region :invoke])]
      (is (= :literal (:input-kind adapter)))
      (is (empty? (:operands step)))
      (is (nil? (:input-port adapter)))
      (is (empty? (.getParameterTypes method)))
      (is (identical? Object (.getReturnType method)))
      (is (= (class value) (class (invoke))))
      (is (identical? value (invoke)) "Object literal identity belongs to its source owner")
      (is (identical? adapter (source-import/validate-source-completion-step! adapter source-program plan step)))
      (doseq [fake [(dissoc adapter :receipt)
                    (assoc-in adapter [:region :invoke] (constantly value))
                    (assoc adapter :input-port {:type :ref :reference-class Object})]]
        (is (= :source-completion-step-correspondence
               (reason #(source-import/validate-source-completion-step! fake source-program plan step)))))
      (is (some? (reason #(completion-adapter (dissoc request :witness)))))
      (is (some? (reason #(completion-adapter
                          (assoc request :witness (assoc witness :scope (with-meta (:scope witness) {:copied true})))))))
      (let [observe bytecode/retained-input-binding-context]
        (is (= :source-completion-step-carrier
               (with-redefs [bytecode/retained-input-binding-context
                             (fn [& args] (assoc (apply observe args) :after
                                                {:type :ref :reference-class Double :verified true}))]
                 (reason #(completion-adapter request)))))))))

(deftest witnessed-completion-adapters-use-current-float-object-ports-and-results
  (let [original (selected-frontend #'public-float-entry)
        {:keys [source-program plan witness steps] :as request} (witnessed-completion-source original false)
        [boxed unboxed] (mapv #(completion-adapter (assoc request :step %)) steps)]
    (is (= 2 (count steps)))
    (is (= [:scalar :opaque] [(:input-kind boxed) (:result-kind boxed)]))
    (is (= [:opaque :scalar] [(:input-kind unboxed) (:result-kind unboxed)]))
    (doseq [[adapter input-class output-class] [[boxed Float/TYPE Object] [unboxed Object Float/TYPE]]
            :let [method ^java.lang.reflect.Method (get-in adapter [:region :method])]]
      (is (= [input-class] (vec (.getParameterTypes method))))
      (is (identical? output-class (.getReturnType method)))
      (is (identical? witness (:witness adapter)))
      (is (identical? adapter (source-import/validate-source-completion-step!
                              adapter source-program plan (:step adapter)))))
    (doseq [value [(float 0.25) (float -0.0) (Float/intBitsToFloat 0x7fc01234) Float/POSITIVE_INFINITY]]
      (let [object ((get-in boxed [:region :invoke]) value)
            actual ((get-in unboxed [:region :invoke]) object)]
        (is (instance? Float object))
        (is (= (Float/floatToRawIntBits value) (Float/floatToRawIntBits actual)))))
    (doseq [adapter [boxed unboxed]
            changed [(assoc adapter :input-port (into {} (:input-port adapter)))
                     (assoc adapter :observed (into {} (:observed adapter)))
                     (assoc adapter :witness (into {} witness))]]
      (is (= :source-completion-step-correspondence
             (reason #(source-import/validate-source-completion-step!
                       changed source-program plan (:step adapter))))))
    (let [observe bytecode/retained-input-binding-context]
      (doseq [change [#(assoc-in % [:before 'value :type] :double)
                      #(assoc % :after {:type :ref :reference-class Float :verified true})]]
        (is (= :source-completion-step-carrier
               (with-redefs [bytecode/retained-input-binding-context
                             (fn [& args] (change (apply observe args)))]
                 (reason #(completion-adapter (assoc request :step (first steps)))))))))
    (is (some? (reason #(completion-adapter
                        (assoc request :step (first steps) :witness
                               (assoc witness :scope (with-meta (:scope witness) {:copied true})))))))))

(deftest witnessed-opaque-float-entry-keeps-number-effects-and-historical-owner
  (let [original (selected-frontend #'public-object-float-entry)
        {:keys [source-program plan steps] :as request} (witnessed-completion-source original true)
        adapter (completion-adapter (assoc request :step (second steps)))
        invoke (get-in adapter [:region :invoke])
        calls (atom 0)
        number (proxy [Number] []
                 (intValue [] (throw (AssertionError. "wrong overload")))
                 (longValue [] (throw (AssertionError. "wrong overload")))
                 (doubleValue [] (throw (AssertionError. "wrong overload")))
                 (floatValue [] (swap! calls inc) (float -0.0)))]
    (is (= (Float/floatToRawIntBits (float -0.0)) (Float/floatToRawIntBits (invoke number))))
    (is (= 1 @calls))
    (is (thrown? NullPointerException (invoke nil)))
    (is (thrown? ClassCastException (invoke true)))
    (with-redefs-fn
      {(get-in original [:source-entry :source-var]) (fn [& _] (throw (AssertionError. "must not execute")))}
      #(do
         (is (some? (reason (fn [] (source-import/validate-source-completion-step!
                                   adapter source-program plan (:step adapter))))))
         (is (thrown? clojure.lang.ExceptionInfo (invoke number)))))
    (is (= 1 @calls) "stale owner refuses before another Number conversion")))

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
