(ns raster.compiler.passes.parallel.auxiliary-semantic-witness-test
  (:require [raster.compiler.ir.invocation-source :as invocation-source]
            [clojure.test :refer [deftest is]]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-route :as typed]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.scalar.host-abstract-value :as host-av]
            [raster.compiler.passes.parallel.structured-control-route :as structured]
            [raster.compiler.passes.parallel.segop-lower-pass :as lowering]))

(defn- contextual-map-reduce-fixture []
  (let [source '(let* [mapped (raster.par/pmap i 8 double
                               (* (clojure.core/aget x i) (clojure.core/aget x i)))
                      total (raster.par/reduce acc 0.0 j 8
                              (+ acc (clojure.core/aget mapped j)))] total)
        envelope (:program (typed/attempt source :double {'x :double}
                                          {:resident-reductions? false}))
        program (:algorithm (first (:equations envelope)))
        equation (first (dialect/equations program))
        operand (nth (second source) 3)
        context (completion/bytecode-result-context-contract
                 (completion/bytecode-operand-context {:object-result? true}) :ref)
        expression (with-meta (completion/complete-value context operand)
                     {:file "contextual-return.clj" :line 17 :column 5})
        protected (list 'let* (assoc (second source) 3 expression) 'total)
        contexts (#'typed/contextual-source-bindings (vec (partition 2 (second protected))))]
    {:source source :protected protected :old-operand operand :expression expression
     :context context :envelope envelope :program program :equation equation :contexts contexts}))

(defn- contextual-wrappers [source]
  (filter #(= :jvm-bytecode-result-context
              (get-in (completion/canonical-parts %) [:contract :source-context]))
          (tree-seq coll? seq source)))

(deftest contextual-reduction-reconstruction-wraps-the-new-fused-source-once
  (let [{:keys [source protected old-operand expression context envelope program equation contexts]}
        (contextual-map-reduce-fixture)
        realize #(let [fresh (#'typed/realize-equation program equation)]
                    [fresh (#'typed/preserve-contextual-reduction-source program equation contexts fresh)])]
    (is (some? envelope))
    (is (= 1 (count (:equations envelope))) "the actual public typed route fused map into reduce")
    (is (= 'reduce (dialect/operation-kind equation)))
    (is (= '[x] (:inputs (dialect/facts program))))
    (is (empty? (contextual-wrappers (:source (#'typed/realize-source source program))))
        "ordinary source reconstruction does not invent a result context")
    (doseq [_ (range 2)
            :let [[fresh rebuilt] (realize)
                  wrapped (:source rebuilt)
                  parts (completion/canonical-parts wrapped)]]
      (is (par/par-reduce-form? (:source fresh)))
      (is (not= old-operand (:source fresh)) "the old mapped-buffer read is not the fused reduction")
      (is (not (identical? old-operand (:operand parts))))
      (is (identical? (:source fresh) (:operand parts)))
      (is (= context (:contract parts)))
      (is (= (meta expression) (meta wrapped)))
      (is (identical? wrapped (second (first (:pairs rebuilt)))))
      (is (= 1 (count (contextual-wrappers wrapped))))
      (is (identical? fresh (#'typed/preserve-contextual-reduction-source
                             program equation {:bindings {} :positions {}} fresh))))
    (doseq [_ (range 2)
            :let [rebuilt (:source (#'typed/realize-source protected program))
                  wrappers (vec (contextual-wrappers rebuilt))]]
      (is (= 1 (count wrappers)) "whole-source reconstruction neither drops nor duplicates the context")
      (is (= context (:contract (completion/canonical-parts (first wrappers)))))
      (is (not (identical? old-operand (:operand (completion/canonical-parts (first wrappers)))))))))

(deftest contextual-reduction-reconstruction-refuses-mismatched-boundaries
  (let [{:keys [protected program equation contexts]} (contextual-map-reduce-fixture)
        fresh (#'typed/realize-equation program equation)
        result (first (nth equation 2))
        with-facts (fn [facts] (with-meta (apply list (assoc (vec program) 1 facts)) (meta program)))
        changed-operation (fn [f]
                            (apply list (assoc (vec equation) 3
                                               (apply list (f (vec (nth equation 3)))))))
        reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
    (doseq [[candidate operation realized]
            [[program equation (update fresh :placement inc)]
             [program equation (assoc fresh :site [:binding 'wrong])]
             [program equation (assoc fresh :pairs [['wrong (:source fresh)]])]
             [program equation (update fresh :pairs conj (first (:pairs fresh)))]
             [program equation (update fresh :pairs conj ['auxiliary 0.0])]
             [program (apply list (assoc (vec equation) 2 [result 'extra])) fresh]
             [program (changed-operation #(update % 1 assoc :result-transform {:kind :unsupported})) fresh]
             [(with-facts (assoc-in (dialect/facts program) [:values result :representation]
                                   {:kind :resident-scalar-buffer :elements 1})) equation fresh]
             [(with-facts (assoc-in (dialect/facts program)
                                   [:equations (second equation) :attributes :result-storage]
                                   [{:destination 'out :access :write}])) equation fresh]
             [program equation (assoc fresh :source (apply list (:source fresh)))]]]
      (is (= :typed-soac-contextual-source
             (reason #(#'typed/preserve-contextual-reduction-source candidate operation contexts realized)))))
    (is (= :typed-soac-contextual-source
           (reason #(#'typed/realize-source protected (apply list (assoc (vec program) 2 [])))))
        "whole-source realization refuses a lost contextual binder")))

(defn- functional-program []
  (:program (typed/attempt
             '(let* [result (raster.par/pmap i 8 double (clojure.core/aget x i))] result)
             :double {'x :double})))

(defn- aliased-program []
  (:program (typed/attempt
             '(let* [left (raster.par/map! a i 8 double (clojure.core/aget x i))
                     right (raster.par/map! b j 8 double (clojure.core/aget x j))]
                [left right])
             :double {'x :double 'a :double 'b :double})))

(defn- replace-binding [program binding expression]
  (update program :source
          (fn [source]
            (with-meta (list* (first source)
                              (vec (mapcat (fn [[id value]]
                                             [id (if (= id binding) expression value)])
                                           (partition 2 (second source))))
                              (nnext source))
              (meta source)))))

(defn- semantic-variants [expression]
  (cond-> [(with-meta expression (assoc (meta expression) :tag 'float))
           (with-meta expression (assoc (meta expression) :raster.type/tag 'float))
           (with-meta expression (assoc (meta expression) :raster.op/original 'raster.numeric/-))]
    (seq? expression)
    (conj (with-meta (list* (with-meta (first expression)
                             (assoc (meta (first expression)) :raster.op/original 'raster.numeric/-))
                           (rest expression))
            (meta expression)))))

(defn- relocated [expression]
  (with-meta expression (assoc (meta expression) :file "relocated.clj" :line 90
                              :column 7 :end-line 91 :end-column 9)))

(deftest auxiliary-allocation-and-alias-witnesses-use-semantic-source-equality
  (let [programs [(functional-program) (aliased-program)]
        all-witnesses (mapcat #(mapcat (fn [equation] (get-in equation [:attributes :realization-bindings]))
                                      (:equations %)) programs)]
    (is (every? some? programs))
    (is (= #{:result-allocation :storage-alias} (set (map :kind all-witnesses))))
    (doseq [program programs
            :let [witnesses (vec (mapcat #(get-in % [:attributes :realization-bindings]) (:equations program)))]]
    (doseq [{:keys [binding expression]} witnesses]
      (is (invocation-source/realization-binding? program binding expression))
      (let [moved (relocated expression)]
        (is (invocation-source/realization-binding? program binding moved))
        (is (invocation-source/realization-binding? (replace-binding program binding moved) binding expression)))
      (doseq [changed (semantic-variants expression)]
        (is (= expression changed) "ordinary equality hides the semantic metadata mutation")
        (is (false? (invocation-source/realization-binding? program binding changed)))
        (is (false? (invocation-source/realization-binding? (replace-binding program binding changed)
                                               binding expression)))))
    ;; Updating both the source and advisory allocation witness cannot override its independently
    ;; reconstructed typed allocation. Alias destination membership remains its separate owner check.
    (doseq [{:keys [binding expression]} (filter #(= :result-allocation (:kind %)) witnesses)
            changed (semantic-variants expression)]
      (let [forged (-> (replace-binding program binding changed)
                       (update :equations
                               #(mapv (fn [equation]
                                        (update-in equation [:attributes :realization-bindings]
                                                   (fn [entries]
                                                     (mapv (fn [entry]
                                                             (cond-> entry (= binding (:binding entry))
                                                               (assoc :expression changed))) entries)))) %)))]
        (is (false? (invocation-source/realization-binding? forged binding changed))))))))

(deftest host-prefix-and-allocation-executors-retain-semantic-expression-witnesses
  (let [check (ns-resolve 'raster.compiler.passes.parallel.structured-control-route 'validate-host-coverage!)
        expression '(clojure.core/double-array n)
        program {:source (list 'let* ['buffer expression] 'buffer) :equations []}]
    (doseq [allocation? [false true]
            :let [owner (cond-> program allocation?
                          (assoc :attributes {:allocations [{:destination 'buffer
                                                            :source-expression expression}]}))
                  prefix (if allocation? [] [['buffer expression]])]]
      (is (= owner (check owner prefix #{})))
      (let [moved (replace-binding owner 'buffer (relocated expression))]
        (is (= moved (check moved prefix #{}))))
      (doseq [changed (semantic-variants expression)]
        (is (= :structured-control-host-continuation
               (try (check (replace-binding owner 'buffer changed) prefix #{}) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest promotion-cannot-replace-stale-auxiliary-ownership-with-a-prefix-allocation
  (let [program (functional-program)
        owner (first (:equations program))
        {:keys [binding expression]} (first (get-in owner [:attributes :realization-bindings]))
        options {:dtype :double :public-parameters '[x] :array-types {'x :double}}
        promote #(structured/promote-program % options)
        replace-witness (fn [program changed]
                          (update-in program [:equations 0 :attributes :realization-bindings]
                                     (fn [entries]
                                       (mapv #(if (= binding (:binding %))
                                                (assoc % :expression changed) %) entries))))]
    (is (some? binding))
    (is (some? (promote program)))
    (doseq [changed (semantic-variants expression)
            candidate [(replace-binding program binding changed)
                       (replace-witness program changed)
                       (replace-witness (replace-binding program binding changed) changed)]]
      (is (= :structured-control-host-continuation
             (try (promote candidate) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest singleton-packet-source-admission-preserves-semantic-metadata
  (let [packet (functional-program)
        equation (first (:equations packet))
        binding (second (:site equation))
        expression (:source equation)
        admit (ns-resolve 'raster.compiler.passes.parallel.segop-lower-pass 'singleton-packet)
        run (fn [candidate]
              (with-redefs [typed/attempt (fn [& _] {:program candidate})
                            ;; Keep this test at packet admission, before independent scheduling.
                            lowering/segop-lower-pass (fn [p _] {:form p})]
                (admit (first (:results equation)) expression {:dtype :double} (atom #{}) 'map)))]
    (is (some? (:primary (run packet))))
    (is (some? (:primary (run (replace-binding packet binding (relocated expression))))))
    (doseq [changed (semantic-variants expression)]
      (is (= :typed-map-source
             (try (run (replace-binding packet binding changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:stage (ex-data error)))))))))

(deftest analyzed-allocation-source-retains-its-declared-host-scalar-semantics
  (let [source '(let* [size (clojure.core/* nrows width)
                       out (clojure.core/float-array size)
                       effect (raster.par/map-void! i nrows
                                (clojure.core/aset out i (float (clojure.core/aget x i))))]
                  effect)
        options {:dtype :float :public-parameters '[x nrows width]
                 :array-types {'x :float}
                 :scalar-types {'nrows :long 'width :long 'size :long}}
        program (:program (typed/attempt source :float {'x :float}
                                        {:scalar-types (:scalar-types options)}))
        allocation (first (get-in program [:attributes :allocations]))
        expression (:source-expression allocation)
        retained (get (into {} (map vec (partition 2 (second (:source program))))) 'out)
        promote #(structured/promote-program % options)]
    (is (= 'out (:destination allocation)))
    (is (= 'long (:tag (meta (second expression)))))
    (is (= 'long (:raster.type/tag (meta (second retained)))))
    (is (some? (promote program)))
    (doseq [changed (semantic-variants expression)]
      (is (= :structured-control-host-continuation
             (try (promote (replace-binding program 'out changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= :structured-control-host-continuation
             (try (promote (assoc-in program [:attributes :allocations 0 :source-expression] changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (let [changed (with-meta (list (first expression)
                                  (with-meta (second expression) {:tag 'int :raster.type/tag 'int}))
                    (meta expression))]
      (is (= :structured-control-host-continuation
             (try (promote (replace-binding program 'out changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest unused-source-allocation-still-executes-its-constructor-check
  (let [source '(let* [unused (clojure.core/double-array count)
                       out (clojure.core/double-array 1)
                       effect (raster.par/map! out i 1 double 7.0)]
                  out)
        ;; Qualify realization of already analyzed source ownership. Earlier
        ;; optional scalar DCE has a separate allocation-removability policy.
        analyzed (frontend/form->program source {:dtype :double :scalar-types {'count :long}})
        program (typed/analyzed-program-envelope source analyzed)
        retained (into {} (map vec (partition 2 (second (:source program)))))
        run (fn [form count] (eval (list 'let* ['count count] form)))]
    (is (some? program))
    (is (= '(clojure.core/double-array count) (get retained 'unused)))
    (is (= 2 (count (get-in program [:attributes :allocations]))))
    (doseq [form [source (:source program)]]
      (is (thrown? NegativeArraySizeException (run form -1)))
      (is (= [7.0] (vec (run form 2)))))
    (let [promote #(structured/promote-program
                    % {:public-parameters '[count] :scalar-types {'count :long}
                       :array-types {} :dtype :double})
          promoted (promote program)
          allocation-index (first (keep-indexed (fn [i allocation]
                                                  (when (= 'unused (:destination allocation)) i))
                                                (get-in program [:attributes :allocations])))]
      (is (nil? (get-in program [:values 'unused])))
      (is (= :double (get-in promoted [:values 'unused :dtype])))
      (is (= '[count] (get-in promoted [:values 'unused :shape])))
      ;; The numerical result names the traversal, whereas the complete source
      ;; returns its destination. Promotion retains that physical boundary
      ;; reference without turning it into a numerical read or changing the
      ;; equation. Equality of the two interfaces would erase this distinction.
      (is (= [] (:inputs program)))
      (is (= '[effect] (:outputs program)))
      (is (= '[out] (:inputs promoted)))
      (is (= '[out] (:outputs promoted)))
      (is (= (mapv :algorithm (:equations program)) (mapv :algorithm (:equations promoted))))
      (is (= (get-in program [:attributes :allocations allocation-index :source-expression])
             (get-in promoted [:attributes :allocations allocation-index :source-expression])))
      (doseq [[field changed] [[:dtype :float] [:extent 2]]]
        (is (= :structured-control-source-allocation-value
               (try (promote (assoc-in program [:attributes :allocations allocation-index field] changed))
                    nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (doseq [[field changed] [[:dtype :float] [:extent 2] [:initialization :copy]]]
        (is (= :structured-control-source-allocation-value
               (try (promote (assoc-in promoted [:attributes :allocations allocation-index field] changed))
                    nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (let [analyze host-av/analyze-capacities]
        (is (= :structured-control-source-allocation-value
               (with-redefs [host-av/analyze-capacities
                             (fn [& args] (update (apply analyze args) :definition-values dissoc 'unused))]
                 (try (promote program) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
      (is (= :structured-control-host-continuation
             (try (promote (update-in program [:attributes :allocations]
                                      #(conj % (nth % allocation-index))))
                  nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest analyzed-binding-transport-requires-exact-unique-original-ownership
  (let [transport (ns-resolve 'raster.compiler.passes.parallel.typed-soac-route
                              'retain-analyzed-binding-expressions)
        binding (with-meta 'out {:line 1})
        form (with-meta (list 'let* (with-meta [binding '(clojure.core/float-array size)] {:line 2})
                              'out) {:line 3})
        expression '(clojure.core/float-array ^long size)
        descriptor {:id 0 :sym 'out :expr expression}
        projected (transport form [descriptor] :allocation-source-binding-mismatch)]
    (is (identical? binding (first (second projected))))
    (is (= (meta form) (meta projected)))
    (is (= (meta (second form)) (meta (second projected))))
    (is (= (nnext form) (nnext projected)))
    (let [source-expression (second (second form))
          contextual (assoc descriptor :source-expression source-expression)
          retained (transport form [contextual] :protected-source-binding-mismatch)]
      (is (identical? source-expression (second (second retained)))
          "a numerical projection does not replace the source initializer")
      (is (identical? form (transport form [(dissoc contextual :expr)]
                                     :protected-source-binding-mismatch))
          "an ordinary numerical descriptor need not duplicate its structured algorithm as :expr")
      (is (= :protected-source-binding-mismatch
             (try (transport form [(assoc contextual :source-expression
                                          (apply list source-expression))]
                             :protected-source-binding-mismatch)
                  nil (catch clojure.lang.ExceptionInfo error (:invariant (ex-data error)))))
          "a copied initializer does not borrow an analyzed occurrence"))
    (doseq [invariant [:allocation-source-binding-mismatch :protected-source-binding-mismatch]
            descriptions [[(dissoc descriptor :id)] [(assoc descriptor :id -1)]
                          [(assoc descriptor :id 1)] [(assoc descriptor :sym 'foreign)]
                          [(dissoc descriptor :expr)] [descriptor descriptor]]]
      (is (= invariant
             (try (transport form descriptions invariant) nil
                  (catch clojure.lang.ExceptionInfo error (:invariant (ex-data error)))))))))
