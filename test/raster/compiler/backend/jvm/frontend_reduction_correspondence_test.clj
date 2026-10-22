(ns raster.compiler.backend.jvm.frontend-reduction-correspondence-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.numeric :as numeric]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-prefix :as prefix]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.backend.jvm.split :as split]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm source-sum [xs :- (Array double) count :- Long] :- Double
  (par/reduce acc 0.0 i count (+ acc (aget xs i))))

(deftm source-float-sum [xs :- (Array float) count :- Long] :- Float
  (par/reduce acc (float 0.0) i count (+ acc (aget xs i))))

(deftm source-binding [n :- Long] :- Long
  (let [size (numeric/* n 2)] (long size)))

(deftm partitioned-reduction-scalar [xs :- (Array float)] :- Float
  (let [sum (par/reduce acc (float 0.0) i 4 (numeric/+ acc (aget xs i)))
        reciprocal (numeric// (float 1.0) sum)]
    reciprocal))

(defonce ^:private cold-source-evaluations (atom 0))

(deftm cold-sum [xs :- (Array double) count :- Long] :- Double
  (swap! cold-source-evaluations inc)
  (par/reduce acc 0.0 i count (+ acc (aget xs i))))

(defonce renewal-source-evaluations (atom 0))

(defn- fresh-renewal-fixture []
  ;; Fresh declarations make the cold→warm lifecycle repeatable in one REPL.
  ;; The caller removes this scope even when compilation or an assertion fails.
  (let [scope (create-ns (gensym "raster.test.renewal."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm leaf [xs :- (Array double) count :- Long] :- Double
                 (swap! raster.compiler.backend.jvm.frontend-reduction-correspondence-test/renewal-source-evaluations inc)
                 (raster.par/reduce acc 0.0 i count (+ acc (aget xs i)))))
        (eval '(raster.core/deftm caller [xs :- (Array double) count :- Long] :- Double
                 (leaf xs count))))
      {:scope scope :caller (core/resolve-deftm-var (ns-resolve scope 'caller))
       :callee (core/resolve-deftm-var (ns-resolve scope 'leaf))}
      (catch Throwable failure
        (remove-ns (ns-name scope))
        (throw failure)))))

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

(defn- with-fresh-reduction-binding
  ([float? split? run] (with-fresh-reduction-binding float? split? 1 run))
  ([float? split? count-expression run]
  ;; Fresh source owners keep forced-split/unsplit coverage independent of test
  ;; order and of the bytecode already cached by another REPL qualification.
  (let [scope (create-ns (gensym "raster.test.extracted."))]
    (try
      (binding [*ns* scope
                entry/*emission-inputs* (atom {:static [] :values []})]
        (refer 'clojure.core)
        (let [declaration (if float?
                '(raster.core/deftm objective [xs :- (Array float)] :- Float
                   (let [sum (raster.par/reduce acc (float 0.0) i 1
                               (raster.numeric/+ acc (aget xs i)))
                         reciprocal (raster.numeric// (float 1.0) sum)]
                     reciprocal))
                '(raster.core/deftm objective [xs :- (Array double)] :- Double
                   (let [sum (raster.par/reduce acc 0.0 i 1
                               (raster.numeric/+ acc (aget xs i)))
                         reciprocal (raster.numeric// 1.0 sum)]
                     reciprocal)))
              body (last declaration)
              bindings (second body)
              reduction (second bindings)
              ;; Construct each original test declaration BEFORE its compiler
              ;; producer runs, never edit retained source/evidence afterward.
              reduction (apply list (assoc (vec reduction) 4 count-expression))
              body (list* (first body) (assoc bindings 1 reduction) (nnext body))]
          (eval (apply list (concat (butlast declaration) [body]))))
        (with-redefs [split/estimate-form-size (constantly (if split? 1000000 0))]
          (run (captured (ns-resolve scope 'objective)))))
      (finally (remove-ns (ns-name scope)))))))

(deftest extracted-reduction-keeps-actual-helper-return-and-binding-boundaries
  (doseq [float? [false true]]
    (with-fresh-reduction-binding
      float? true
      (fn [frontend]
        (let [scope (first (:body frontend))
              original (nth (second scope) 1)
              observed (entry/frontend-extracted-reduction-binding frontend scope 0)
              method (get-in observed [:helper :method])
              actual (if float? Float/TYPE Double/TYPE)]
          (is (some? observed))
          (is (identical? original (:source-input observed)))
          (is (identical? scope (:scope-input observed)))
          (is (= :ref (get-in observed [:helper :emitted-result-type])))
          (is (= actual (.getReturnType ^java.lang.reflect.Method method)))
          (is (= (if float? :float :double)
                 (get-in observed [:helper :return-stack-type])
                 (get-in observed [:binding :after :type])))
          (is (= :module-static (get-in observed [:invocation :invocation-mode])))
          (is (= method (get-in observed [:invocation :method])))
          (is (identical? (get-in observed [:binding :initializer])
                          (get-in observed [:invocation :expression])))
          (is (= Object (get-in observed [:reduction :result-fact :carrier]))
              "the reduction Object payload is not relabelled as its helper return")
          (is (nil? (entry/frontend-extracted-reduction-binding
                     frontend (with-meta (apply list scope) (meta scope)) 0)))
          (doseq [index [-1 1 2 Long/MAX_VALUE nil 0.5]]
            (is (nil? (entry/frontend-extracted-reduction-binding frontend scope index)))))))))

(deftest unsplit-reduction-does-not-invent-an-extracted-helper
  (with-fresh-reduction-binding
    false false
    (fn [frontend]
      (let [scope (first (:body frontend))]
        (is (some? (entry/frontend-reduction-emission frontend (nth (second scope) 1))))
        (is (nil? (entry/frontend-extracted-reduction-binding frontend scope 0)))))))

(deftest direct-reduction-consumer-preserves-actual-carriers-and-completions
  (doseq [float? [false true] split? [false true]]
    (with-fresh-reduction-binding
      float? split?
      (fn [frontend]
        (let [parameters (get-in frontend [:source-entry :params])
              input (prefix/begin-reduction-interval frontend parameters)
              observed (prefix/observed-reduction-interval input)
              producer-type (if float? :float :double)
              producer-class (if float? Float/TYPE Double/TYPE)
              producer (get-in observed [:producer :binding])
              consumer (get-in observed [:consumer :binding])
              call (get-in observed [:consumer :invocation])]
          (is (some? input))
          (is (some? observed))
          (is (identical? (first (:body frontend)) (:scope observed)))
          (is (= [0 1] (:ordinals observed)))
          (is (= (if split? :extracted-reduction :direct-reduction)
                 (get-in observed [:producer :kind])))
          (is (= (if split? producer-class Object)
                 (get-in observed [:producer :completed-value :carrier])))
          (is (= (if split? producer-type :ref) (get-in producer [:after :type])
                 (get-in call [:input-completions 1 :emitted-type])))
          (is (= (if split? :identity :coerce) (get-in call [:input-completions 1 :adaptation])))
          (is (= (:after producer) (get-in consumer [:before 'sum])))
          (is (= Object (:return-carrier call)
                 (get-in observed [:consumer :computation :result-carrier])))
          (is (= :ref (get-in consumer [:after :type])))
          (is (nil? (prefix/begin frontend parameters))
              "the old prefix family stops before the first parallel operation")
          (is (nil? (prefix/close input nil nil nil {:source (:scope observed)})))
          (is (nil? (prefix/observed-reduction-interval nil)))
          (is (nil? (prefix/begin-reduction-interval frontend (conj parameters 'extra)))))))))

(deftest direct-reduction-consumer-refusals-do-not-retain-speculative-inputs
  (with-fresh-reduction-binding
    false false
    (fn [frontend]
      (let [input (prefix/begin-reduction-interval frontend (get-in frontend [:source-entry :params]))
            observe entry/frontend-completed-invocation-context
            reduction entry/frontend-reduction-emission
            selected entry/selected-parameter-carriers
            foreign (.getMethod Double "valueOf" (into-array Class [Double/TYPE]))]
        (doseq [[path change]
                [[[:method] (constantly foreign)]
                 [[:return-carrier] (constantly Double/TYPE)]
                 [[:receiver] (constantly 'foreign/receiver)]
                 [[:receiver-input :ports] (constantly {'receiver {:type :ref}})]
                 [[:receiver-input :context] (constantly nil)]
                 [[:receiver-input :context :preserve-source-call-results?] (constantly :unknown)]
                 [[:inputs 1 :ports 'sum :slot] inc]
                 [[:input-completions 1 :emitted-type] (constantly :double)]
                 [[:input-completions 1 :parameter-carrier] (constantly Float/TYPE)]
                 [[:inputs] #(conj % (last %))]
                 [[:input-completions] #(conj % (last %))]
                 [[:input-completions 0] (constantly nil)]
                 [[:input-completions] pop]]]
          (let [before @entry/*emission-inputs*]
            (with-redefs [entry/frontend-completed-invocation-context
                          (fn [& arguments] (update-in (apply observe arguments) path change))]
              (is (nil? (prefix/observed-reduction-interval input)) (str "changed call boundary: " path)))
            (is (identical? before @entry/*emission-inputs*)
                "a refused whole interval cannot retain individually successful subqueries")))
        (doseq [[key value] [[:presence :unknown] [:carrier Double/TYPE]]]
          (let [before @entry/*emission-inputs*]
            (with-redefs [entry/frontend-reduction-emission
                          (fn [& arguments] (assoc-in (apply reduction arguments) [:result-fact key] value))]
              (is (nil? (prefix/observed-reduction-interval input))))
            (is (identical? before @entry/*emission-inputs*))))
        (with-redefs [entry/selected-parameter-carriers
                      (fn [& arguments]
                        (assoc-in (apply selected arguments) [:definition :entry :typed-impl] (Object.)))]
          (is (= :retained-frontend-owner-mismatch
                 (try (prefix/observed-reduction-interval input) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
        (is (= :unowned-checked-result-frontend
               (try
                 (prefix/begin-reduction-interval
                  (assoc frontend :body (with-meta (mapv identity (:body frontend))
                                         (meta (:body frontend))))
                  (get-in frontend [:source-entry :params]))
                 nil
                 (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))))

(deftest original-closed-count-is-not-a-constructed-domain-proof
  (doseq [split? [false true]
          count-expression [0 1 2 -1 Long/MAX_VALUE]]
    (with-fresh-reduction-binding
      false split? count-expression
      (fn [frontend]
        (let [input (prefix/begin-reduction-interval frontend (get-in frontend [:source-entry :params]))
              observed (prefix/observed-reduction-interval input)
              closed (:closed-count observed)]
          (is (some? observed) "the conditional scalar observation does not assume count success")
          (if (contains? #{0 1} count-expression)
            (do (is (= count-expression (:value closed)))
                (is (instance? Integer (:value closed)))
                (is (identical? (get-in observed [:producer :reduction :bound :initializer])
                                (:source closed)))
                (is (identical? (get-in observed [:producer :reduction :bound]) (:bound closed)))
                (is (= (list 'clojure.core/int count-expression) (:proof-expression closed)))
                (is (= :before-initializer (:evaluation closed)))
                (is (not-any? #(contains? closed %)
                              [:domain-proved? :read-proved? :prepared :replayable?])))
            (is (nil? closed) "other counts keep their source completion/domain obligation")))))))

(deftest closed-count-does-not-erase-unchecked-cast-effect-obligations
  (doseq [[expression expected]
          [['(clojure.core/unchecked-int 4294967296) 0]
           ['(clojure.core/unchecked-int 4294967297) 1]]]
    (with-fresh-reduction-binding
      false false expression
      (fn [frontend]
        (let [input (prefix/begin-reduction-interval frontend (get-in frontend [:source-entry :params]))
              observed (prefix/observed-reduction-interval input)
              proof-expression (list 'clojure.core/int expression)]
          (is (some? observed))
          ;; Value evaluation knows unchecked narrowing, but that does not
          ;; discharge the original expression's independent effect obligation.
          (is (= expected (:value (raster.compiler.core.numeric-constant/source-value
                                  proof-expression))))
          (is (nil? (:closed-count observed))))))))

(deftest closed-count-rejects-post-emission-bootstrap-replacement
  (with-fresh-reduction-binding
    false false
    (fn [frontend]
      (let [input (prefix/begin-reduction-interval frontend (get-in frontend [:source-entry :params]))
            scope (get-in frontend [:source-entry :source-ns])]
        (is (= 1 (get-in (prefix/observed-reduction-interval input) [:closed-count :value])))
        ;; Change only this fresh namespace after bytecode emission. A newly
        ;; resolved cast must not authenticate the previously emitted program.
        (ns-unmap scope 'int)
        (intern scope 'int identity)
        (is (= :stale-definition-proof-input
               (try (prefix/observed-reduction-interval input)
                    nil
                    (catch clojure.lang.ExceptionInfo failure
                      (:reason (ex-data failure))))))))))

(deftest closed-count-does-not-borrow-cast-or-sibling-authority
  (with-fresh-reduction-binding
    false false
    (fn [frontend]
      (let [input (prefix/begin-reduction-interval frontend (get-in frontend [:source-entry :params]))
            producer (:producer (prefix/observed-reduction-interval input))
            check (ns-resolve 'raster.compiler.core.source-prefix 'closed-reduction-count)
            before @entry/*emission-inputs*]
        (doseq [[path value]
                [[[:reduction :bound :unretained-sibling-calls?] true]
                 [[:reduction :bound :unretained-sibling-calls?] nil]
                 [[:reduction :bound :before 'int] {:type :ref :slot 0}]
                 [[:reduction :bound :initializer] '(foreign/int 1)]
                 [[:reduction :bound :initializer] '(clojure.core/int unknown-count)]
                 [[:reduction :bound :initializer] '(do (clojure.core/int 1))]
                 [[:reduction :bound :initializer]
                  (with-meta '(clojure.core/int 1) {:raster.effect/preserve-evaluation true})]]]
          (is (nil? (check frontend (assoc-in producer path value))) (str "unclosed count: " path))
          (is (identical? before @entry/*emission-inputs*)))))))

(deftest extracted-binding-requires-the-complete-physical-chain
  (with-fresh-reduction-binding
    true true
    (fn [frontend]
      (let [scope (first (:body frontend))
            emission (:emission frontend)
            observed (entry/frontend-extracted-reduction-binding frontend scope 0)
            helper (:helper observed)
            body-token (get-in observed [:invocation :target-body])
            call-token (:raster.compiler.backend.jvm.bytecode/call-occurrence
                        (meta (get-in helper [:extraction :replacement])))
            body-path [:body-contexts body-token]
            call-path [:invocation-contexts call-token]
            root-path [:reduction-contexts [(get-in observed [:reduction :expansion :witness])
                                           :bound-scope] :root-result]
            query #(bytecode/retained-input-extracted-reduction-binding % (:body frontend) scope 0)
            foreign (.getMethod Double "valueOf" (into-array Class [Double/TYPE]))]
        (is (some? (query emission)))
        (doseq [[path value]
                [[body-path nil]
                 [(conj body-path :method) foreign]
                 [(conj body-path :method-owner) Double]
                 [(conj body-path :return-stack-type) :double]
                 [(conj body-path :return-descriptor) "D"]
                 [(conj body-path :emitted-result-type) :long]
                 [(conj body-path :emitted-result-type) :float]
                 [(conj body-path :emitted-result-type) :double]
                 [(conj body-path :body) 42]
                 [root-path nil]
                 [(conj root-path :context) {}]
                 [(into body-path [:extraction :source]) (apply list (get-in helper [:extraction :source]))]
                 [(conj body-path :body) [(apply list (first (:body helper)))]]
                 [(conj call-path :status) :ambiguous]
                 [(conj call-path :invocation-mode) :global-typed-indy]
                 [(conj call-path :target-body) (Object.)]
                 [(conj call-path :target-owner) nil]
                 [(conj call-path :target-descriptor) "([F)D"]
                 [(conj call-path :input-completions) []]
                 [(conj call-path :inputs) 42]
                 [(conj call-path :input-completions) 42]
                 [(into call-path [:input-completions 0]) 42]
                 [(into call-path [:inputs 0 :ports]) 42]
                 [(into call-path [:input-completions 0 :parameter-descriptor]) "[D"]
                 [(into call-path [:input-completions 0 :target-stack-type]) :double]
                 [(into call-path [:input-completions 0 :emitted-type]) :double]
                 [(conj call-path :expression) (apply list (get-in observed [:binding :initializer]))]
                 [[:invocation-catalog call-token]
                  (with-meta (apply list (get-in helper [:extraction :replacement]))
                    (meta (get-in helper [:extraction :replacement])))]]]
          (is (nil? (query (assoc-in emission path value))) (str "incomplete extraction edge: " path)))
        (is (nil? (query (assoc-in emission [:body-contexts (Object.)] helper)))
            "multiple helper owners of the same occurrence are ambiguous")
        (is (nil? (entry/frontend-extracted-reduction-binding
                   (assoc frontend :emission (assoc-in emission (conj body-path :method) foreign)) scope 0))
            "a copied/edited emission cannot acquire frontend authority")))))

(deftest method-extraction-retains-bindings-but-not-original-initializers
  ;; Size is a compiler decision; force it without depending on source size or
  ;; redefining the compile-time constant TARGET-SIZE. No original body is run
  ;; to construct these observations.
  (with-redefs [split/estimate-form-size (constantly 1000000)]
    (let [frontend (captured #'partitioned-reduction-scalar)
          scope (first (:body frontend))
          first-binding (entry/frontend-binding-context frontend scope 0)
          second-binding (entry/frontend-binding-context frontend scope 1)
          original-reduction (nth (second scope) 1)
          original-scalar (nth (second scope) 3)
          scalar-context (entry/frontend-invocation-context frontend original-scalar)]
      (is (some? first-binding))
      (is (some? second-binding))
      (is (= ['sum 'reciprocal] (mapv :binder [first-binding second-binding])))
      (is (not= original-reduction (:initializer first-binding))
          "the actual initializer is an extracted static helper call")
      (is (identical? original-scalar (:source-input scalar-context)))
      (is (identical? (:initializer second-binding) (:expression scalar-context))
          "the untouched scalar still identifies its actual initializer")
      (is (nil? (entry/frontend-invocation-context frontend (:initializer first-binding)))
          "a generated helper call is not an original source call")
      (is (= 'sum (:binder first-binding)))
      (is (= :float (get-in first-binding [:after :type]))
          "report the actual split static return, not an unsplit Object carrier")
      (is (= (:after first-binding) (get-in second-binding [:before 'sum])))
      (is (nil? (entry/frontend-binding-context
                 frontend (with-meta (apply list scope) (meta scope)) 0)))
      (is (nil? (bytecode/retained-body-context (:emission frontend)))
          "root binding identity does not authorize an original unsplit body")
      (is (= :ad-unclosed-bytecode-body-region
             (try
               ((requiring-resolve 'raster.ad.reverse/owned-bytecode-body-program)
                (:emission frontend) (get-in frontend [:emission :params]) [])
               nil
               (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "the production AD entry guard still refuses the unowned whole body")
      (doseq [[values expected] [[[1.0 2.0 3.0 4.0] (float 0.1)]
                                 [[2.0 3.0 4.0 5.0] (float (/ 1.0 14.0))]]]
        (is (= (Float/floatToRawIntBits expected)
               (Float/floatToRawIntBits (partitioned-reduction-scalar (float-array values)))))))))

(deftest method-extraction-refuses-changed-binding-layout
  (let [binder (with-meta 'value {:raster.type/tag 'float})
        scope (list 'let* [binder '(float 1.0)] binder)
        rebuild #'bytecode/partitioned-binding-scope]
    (is (= scope (rebuild scope (second scope) (nnext scope))))
    (is (identical? binder (first (second (rebuild scope (second scope) (nnext scope))))))
    (doseq [bindings [[] [binder '(float 1.0) 'extra 2]
                      [(with-meta 'value (meta binder)) '(float 1.0)]
                      [binder]]]
      (is (= :partitioned-binding-layout
             (try (rebuild scope bindings (nnext scope)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

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
    (is (nil? (entry/frontend-return-bridge frontend)))
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
        (let [bridge (entry/frontend-return-bridge frontend)]
          (is (= Object (:result-carrier bridge)))
          (is (= Long/TYPE (:payload-carrier bridge)))
          (is (identical? (:implementation frontend) (:implementation bridge))))
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

(deftest return-bridge-needs-the-same-authenticated-whole-module
  (let [frontend (captured #'source-binding)
        parent (atom {:static [] :values []}) before @parent]
    (binding [entry/*emission-inputs* parent]
      (doseq [changed [(assoc frontend :emission (into {} (:emission frontend)))
                       (assoc-in frontend [:emission :implementation] (Object.))
                       (assoc-in frontend [:emission :typed-return-bridges] [])]]
        (is (nil? (entry/frontend-return-bridge changed)))
        (is (identical? before @parent)))
      (is (some? (entry/frontend-return-bridge frontend)))
      (is (= 1 (count (:producer-inputs @parent))))
      (is (entry/assert-definition-inputs-current! @parent)))
    (is (nil? (binding [entry/*emission-inputs* nil] (entry/frontend-return-bridge frontend))))))

(deftest reference-return-observations-retain-only-the-authenticated-whole-module
  (let [scope (create-ns (gensym "raster.reference-return-owner."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        ;; A primitive dimension selects the typed-wrapper path under test;
        ;; a reference-only identity may use the ordinary IFn implementation.
        (eval '(raster.core/deftm reference [xs :- (Array float) count :- Long] :- (Array float) xs)))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'reference) {:dtype :float})
            _ (core/prepare-deftm-frontend! source)
            implementation (receiver source)
            before-implementation (var-get implementation)
            cold (entry/original-source-frontend source)]
        (is (some? cold))
        (entry/assert-frontend-source! cold source)
        (is (nil? (:emission cold)))
        (with-redefs [core/ensure-bytecode-implementation!
                      (fn [& _] (throw (AssertionError. "reference return query forced JIT")))]
          (is (nil? (entry/frontend-reference-return-bridge cold))))
        (is (identical? before-implementation (var-get implementation)))
        (is (nil? (:raster.core/bytecode-emission (meta implementation))))
        ;; Promotion replaces the implementation. A producer that observed its
        ;; cold root must retry under a fresh owner, not refresh that history.
        (let [cold-history @entry/*emission-inputs*
              frontend (binding [entry/*emission-inputs* (atom {:static [] :values []})]
                         (captured source))
              parent (atom {:static [] :values []}) before @parent
              metadata (meta source)]
          (is (thrown? clojure.lang.ExceptionInfo
                       (entry/assert-definition-inputs-current! cold-history)))
          (binding [entry/*emission-inputs* parent]
            (doseq [changed [(assoc frontend :emission (into {} (:emission frontend)))
                             (assoc-in frontend [:emission :implementation] (Object.))
                             (assoc-in frontend [:emission :typed-return-bridges] [])]]
              (is (nil? (entry/frontend-reference-return-bridge changed)))
              (is (identical? before @parent)))
            (let [bridge (entry/frontend-reference-return-bridge frontend)]
              (is (= :reference-static-return (:kind bridge)))
              (is (= "[F" (.getName ^Class (:body-return-carrier bridge))))
              (is (identical? (:implementation frontend) (:implementation bridge)))
              (is (not-any? #(contains? bridge %) [:non-null? :payload-carrier :executor])))
            (is (= 1 (count (:producer-inputs @parent))))
            (is (entry/assert-definition-inputs-current! @parent))
            (try
              (alter-meta! source assoc :raster.core/source-policy {:unchecked-math true})
              (is (thrown? clojure.lang.ExceptionInfo
                           (entry/frontend-reference-return-bridge frontend)))
              (is (thrown? clojure.lang.ExceptionInfo
                           (entry/assert-definition-inputs-current! @parent)))
              (finally (reset-meta! source metadata)))
            (is (entry/assert-definition-inputs-current! @parent)))
          (is (nil? (binding [entry/*emission-inputs* nil]
                      (entry/frontend-reference-return-bridge frontend))))))
      (finally (remove-ns (ns-name scope))))))

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

(deftest owned-preparation-renews-real-bytecode-in-callee-before-caller-order
  (let [{scope :scope caller-source :caller leaf-source :callee} (fresh-renewal-fixture)
        caller (receiver caller-source)
        callee (receiver leaf-source)
        metadata (meta leaf-source)
        before @renewal-source-evaluations]
    (try
      (doseq [invalidate-caller? [false true]]
        (core/ensure-current-owned-bytecode! caller)
        (let [old-emission (:raster.core/bytecode-emission (meta caller))
              old-guard (entry/prepare-emission-input-guard! old-emission)
              attempts (atom 0) prepared (atom [])]
          (is (nil? (:raster.core/jit-spec (meta caller))))
          (is (= (not invalidate-caller?) (map? (:raster.core/jit-spec (meta callee)))))
          ;; A callee promotion need not invalidate an inlined caller. Test that
          ;; current code is retained, then invalidate a REAL imported source
          ;; snapshot to exercise renewal rather than assuming it is necessary.
          (when invalidate-caller?
            (alter-meta! leaf-source assoc ::changed-before-preparation true))
          (let [inputs
                (binding [entry/*emission-inputs* nil]
                  (entry/call-with-definition-inputs
                   (fn []
                     (swap! attempts inc)
                     (doseq [v [callee caller]]
                       (entry/observe-definition-input! v (:ns (meta v)) (:name (meta v))))
                     (entry/prepare-owned-definition-inputs!
                      (fn [v]
                        (is (nil? entry/*emission-inputs*))
                        (swap! prepared conj v)
                        (core/ensure-current-owned-bytecode! v))
                      [callee caller])
                     @entry/*emission-inputs*)))
                current (:raster.core/bytecode-emission (meta caller))]
            (is (= 2 @attempts))
            (is (= [callee caller callee caller] @prepared))
            (is (= (not invalidate-caller?) (identical? old-emission current)))
            (is (= (not invalidate-caller?)
                   (identical? (:implementation old-emission) (:implementation current))))
            (is (identical? current
                            (get-in inputs [:definition-proofs caller :metadata :raster.core/bytecode-emission])))
            (is (entry/assert-definition-inputs-current! inputs))
            (is (entry/assert-emission-inputs-current! current))
            (is (= (when invalidate-caller? :stale-emission-inputs)
                   (try (old-guard) nil
                        (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
            (is (= before @renewal-source-evaluations)))))
      (finally
        (reset-meta! leaf-source metadata)
        (remove-ns (ns-name scope))))))
