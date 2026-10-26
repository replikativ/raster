(ns raster.compiler.passes.scalar.host-abstract-value-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.types :as types]
            [raster.compiler.passes.scalar.host-abstract-value :as host-av]))

(def ^:private initial
  (av/tensor {:dtype :double :shape ['?] :representation {:kind :plain}}))

(defn- object-input [operand]
  (completion/complete-value completion/object-input-contract operand))

(defn- reference-entry [carrier operand]
  (completion/complete-value (completion/bytecode-boxed-parameter-contract carrier) operand))

(deftest opaque-aliases-do-not-manufacture-integral-extent-equalities
  (let [opaque (av/make {:kind :opaque :memory-space :host :ownership :borrowed})
        analysis (host-av/analyze-capacities
                  '(let* [alias input again alias] again)
                  {:values {'input opaque} :lexical-inputs '[input]})]
    (is (= opaque (get-in analysis [:definition-values 'alias])))
    (is (= opaque (get-in analysis [:definition-values 'again])))
    (is (empty? (:equalities analysis)))
    (is (empty? (:aliases analysis)) "opaque references are not buffer-capacity aliases")))

(deftest reference-entry-transports-only-existing-compatible-source-capacity
  (doseq [[dtype tag] [[:float 'floats] [:double 'doubles]]]
    (let [input (with-meta 'input {:raster.type/tag tag})
          boxed (with-meta 'boxed {:raster.type/tag tag :tag 'Object})
          checked (with-meta 'checked {:raster.type/tag tag :tag tag})
          value (av/tensor {:dtype dtype :shape ['m] :ownership :external
                            :attributes {:nullable true :source :public}})
          entry (reference-entry (get types/primitive-array-info tag) boxed)
          allocation (with-meta
                       (list '.invk (symbol "raster.arrays" (str "zeros-like_m_" tag "_long-impl")) checked 'n)
                       {:raster.op/original 'raster.arrays/zeros-like})
          source (list 'let* [boxed (object-input input) checked entry
                             'out allocation 'returned '(unknown.library/write! out)] 'returned)
          analysis (host-av/analyze-capacities
                    source {:values {input value} :scalar-types {'n :long}
                            :lexical-inputs [input 'n] :reference-aliases {'returned 'out}})]
      (is (= value (get-in analysis [:definition-values boxed])))
      (is (= value (get-in analysis [:definition-values checked])))
      (is (= {boxed input checked boxed 'returned 'out} (:aliases analysis)))
      (doseq [id '[out returned]]
        (is (= {:dtype dtype :shape ['n]}
               (select-keys (get-in analysis [:definition-values id]) [:dtype :shape]))))
      (is (identical? entry (nth (second source) 3)) "analysis retains the actual CHECKCAST expression")
      (is (= :jvm-bytecode-boxed-parameter-input
             (get-in (completion/canonical-parts entry) [:contract :source-context])))
      (is (= {:nullable true :source :public} (get-in analysis [:definition-values checked :attributes])))
      (is (nil? (get-in analysis [:definition-values checked :builtin-class]))))))

(deftest reference-entry-refuses-guessed-carriers-and-incompatible-declarations
  (let [double-class (get types/primitive-array-info 'doubles)
        buffer (av/tensor {:dtype :double :shape ['n]})
        values {'input buffer}
        entry (reference-entry double-class 'input)]
    (is (= 'input (host-av/bound-buffer-alias-source entry values #{'input} nil)))
    (doseq [expression [(reference-entry (get types/primitive-array-info 'floats) 'input)
                        (reference-entry Object 'input)
                        (reference-entry String 'input)
                        (reference-entry Double/TYPE 'input)
                        (completion/complete-value (completion/reference-check-contract double-class) 'input)
                        (object-input '(identity input))
                        (reference-entry double-class '(identity input))
                        '(clojure.core/identity input)
                        (reference-entry double-class (with-meta 'input {:raster.type/tag 'floats}))
                        (vary-meta entry assoc :raster.type/tag 'floats)
                        (list completion/completion-symbol
                              (assoc (completion/bytecode-boxed-parameter-contract double-class) :extra true) 'input)]]
      (is (nil? (host-av/bound-buffer-alias-source expression values #{'input} nil))))
    (doseq [[available known] [[#{} values] [#{'input} {}]
                              [#{'input} {'input (av/tensor {:dtype :double :shape []})}]]]
      (is (nil? (host-av/bound-buffer-alias-source entry known available nil))))
    (let [binder (with-meta 'checked {:raster.type/tag 'floats})
          analysis (host-av/analyze-capacities (list 'let* [binder entry] binder)
                                               {:values values :lexical-inputs '[input]})]
      (is (nil? (get-in analysis [:definition-values binder])))
      (is (not (contains? (:available analysis) binder))))
    (doseq [options [{:values values :lexical-inputs []}
                     {:values values :lexical-inputs '[input] :non-buffer-bindings '[input]}]
            :let [analysis (host-av/analyze-capacities (list 'let* ['checked entry] 'checked) options)]]
      (is (nil? (get-in analysis [:definition-values 'checked])))
      (is (nil? (get-in analysis [:aliases 'checked]))))
    (let [source (list 'let* ['checked entry 'input '(clojure.core/double-array n)] 'checked)
          analysis (host-av/analyze-capacities source {:values values :scalar-types {'n :long}
                                                       :lexical-inputs '[n]})]
      (is (nil? (get-in analysis [:definition-values 'checked]))))
    (is (= :host-abstract-value-reference-alias
           (try
             (host-av/analyze-capacities
              (list 'let* ['checked (reference-entry (get types/primitive-array-info 'floats) 'input)] 'checked)
              {:values values :lexical-inputs '[input] :reference-aliases {'checked 'input}})
             nil
             (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "a certified alias map cannot bypass the retained entry carrier check")))

(deftest lexical-buffer-aliases-preserve-the-complete-source-value
  (let [value (av/tensor {:dtype :double :shape ['n] :ownership :external
                          :effects #{:memory/read} :attributes {:source :caller}
                          :memory-space :device :placement {:device :arc}
                          :sharding {:mesh :local}})
        source (list 'let* ['y (object-input 'x) 'z 'y 'w (object-input 'z)] 'w)
        analysis (host-av/analyze source {:values {'x value} :lexical-inputs '[x]})]
    (is (= {'y 'x 'z 'y 'w 'z} (:aliases analysis)))
    (doseq [id '[x y z w]] (is (= value (get-in analysis [:values id]))))
    (is (= #{'x 'y 'z 'w} (:available analysis)))
    (is (nil? (:aliases (host-av/analyze source {:values {'x value}}))))
    (is (nil? (get-in (host-av/analyze source {:values {'x value}}) [:values 'y])))))

(deftest alias-availability-is-not-inferred-from-future-seed-facts
  (let [value (av/tensor {:dtype :double :shape ['n]})]
    (doseq [expression ['x (object-input 'x)]
            :let [source (list 'let* ['y expression 'z 'y 'x '(clojure.core/double-array n)] 'z)
                  analysis (host-av/analyze source {:values {'x value 'y value 'z value}
                                                   :lexical-inputs '[n]})]]
      (is (= {} (:aliases analysis)))
      (is (not (contains? (:available analysis) 'y)))
      (is (not (contains? (:available analysis) 'z))))
    (is (nil? (host-av/bound-buffer-alias-source 'x {'x value} #{} nil)))
    (is (= 'x (host-av/bound-buffer-alias-source 'x {'x value} #{'x} nil)))))

(deftest buffer-alias-queries-do-not-erase-checks-or-infer-physical-carriers
  (let [buffer (av/tensor {:dtype :double :shape ['n]})
        scalar (av/tensor {:dtype :double :shape []})
        available #{'x 'scalar}
        values {'x buffer 'scalar scalar}]
    (doseq [expression [(object-input 'scalar)
                        '(clojure.core/float x)
                        (completion/complete-value (completion/reference-check-contract Object) 'x)
                        (object-input '(identity x))
                        (object-input (with-meta 'x {:raster.type/tag 'floats}))
                        (with-meta (object-input 'x) {:raster.type/tag 'double})]]
      (is (nil? (host-av/bound-buffer-alias-source expression values available nil))))
    (doseq [expected [(assoc buffer :dtype :float)
                       (assoc buffer :representation {:kind :quantized :scheme :q4-k})
                       scalar]]
      (is (nil? (host-av/bound-buffer-alias-source (object-input 'x) values available expected))))
    (is (= :conflicting-value-completion-declaration
           (try (host-av/bound-buffer-alias-source
                 (with-meta (object-input (with-meta 'x {:raster.type/tag 'doubles}))
                   {:raster.type/tag 'floats}) values available nil) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
    (is (= {} (:aliases (host-av/analyze
                        (list 'let* ['boxed (object-input 'scalar) 'copied 'scalar] 'copied)
                        {:values {'scalar scalar} :lexical-inputs '[scalar]}))))
    (is (= scalar (get-in (host-av/analyze '(let* [copied scalar] copied)
                                          {:values {'scalar scalar} :lexical-inputs '[scalar]})
                          [:values 'copied])))))

(deftest buffer-alias-analysis-refuses-unstable-flat-identities
  (let [value (av/tensor {:dtype :double :shape ['n]})]
    (doseq [[source inputs]
            [[(list 'let* ['y 'x 'x (object-input 'input)] 'y) '[x input]]
             [(list 'let* ['x (object-input 'input) 'y 'x 'x (object-input 'input)] 'y) '[input]]]]
      (is (= {} (:aliases (host-av/analyze source
                                          {:values {'x value 'input value}
                                           :lexical-inputs inputs})))))
    (is (= {} (:aliases (host-av/analyze (list 'let* [['y] 'x] 'y)
                                        {:values {'x value} :lexical-inputs '[x]})))
        "destructuring is not a flat SSA alias binder")
    (let [id (with-meta 'y {:raster.type/tag 'floats})]
      (is (= {} (:aliases (host-av/analyze (list 'let* [id (object-input 'x)] id)
                                          {:values {'x value} :lexical-inputs '[x]})))))))

(deftest certified-non-buffer-returns-cannot-become-aliases-through-logical-facts
  (let [value (av/tensor {:dtype :double :shape ['n]})
        source (list 'let*
                     ['effect '(raster.par/stencil! destination [input] 1 :dirichlet double i n
                                                  (clojure.core/aget input i))
                      'wrapped (object-input 'effect) 'bare 'wrapped 'again (object-input 'bare)
                      'other-effect (object-input 'input)]
                     'again)]
    (doseq [excluded ['[effect other-effect] #{'effect 'other-effect}]
            :let [analysis (host-av/analyze
                            source {:values (zipmap '[input destination effect wrapped bare again] (repeat value))
                                    :lexical-inputs '[input destination n]
                                    :non-buffer-bindings excluded})]]
      (is (= {} (:aliases analysis)))
      (is (= :tensor (get-in analysis [:values 'effect :kind])))
      (is (= :double (get-in analysis [:values 'effect :dtype])))
      (is (= ['n] (get-in analysis [:values 'effect :shape]))
          "destination-write inference remains logical, not evidence of a host buffer"))
    (doseq [invalid [nil '(effect) [:effect] #{42}]]
      (is (= :host-abstract-value-non-buffer-bindings
             (try (host-av/analyze source {:non-buffer-bindings invalid}) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))

(defn- source
  [copy-length]
  (list 'let*
        ['n '(int (clojure.core/alength u0))
         'u '(clojure.core/aclone u0)
         'scratch '(raster.numeric/similar_m_doubles u0)
         'next '(raster.par/pmap i (clojure.core/alength u) double
                                 (+ (clojure.core/aget u i) 1.0))
         'copy (list 'java.lang.System/arraycopy 'next '(int 0) 'u '(int 0) copy-length)]
        'copy))

(defn- binding-map
  [source]
  (into {} (map vec) (partition 2 (second source))))

(deftest lexical-child-analysis-retains-only-preceding-relations
  (let [source '(let* [n (int (clojure.core/alength input))
                      storage (clojure.core/double-array n)] storage)
        parent (reduce (fn [context [id expression]]
                         (host-av/advance-binding context id expression))
                       (host-av/begin-analysis source
                                               {:values {'input initial}
                                                :lexical-inputs '[input]})
                       (partition 2 (second source)))
        child (host-av/enter-scalar-parameter parent 'iteration :long)
        child (host-av/advance-binding child 'length '(clojure.core/alength storage))
        parent-analysis (host-av/finish-analysis parent)
        child-analysis (host-av/finish-analysis child)]
    (is (= :long (get-in child-analysis [:values 'iteration :dtype])))
    (is (contains? (:available child-analysis) 'iteration))
    (is (not (contains? (:values parent-analysis) 'iteration)))
    (is (not (contains? (:values parent-analysis) 'length))
        "a child scope cannot introduce future facts into its parent")
    (is (every? (fn [[expression value]]
                  (= value (get-in child-analysis [:equalities expression])))
                (:equalities parent-analysis)))
    (is (not-any? #{'iteration} (tree-seq coll? seq (:equalities child-analysis)))
        "a loop index is unknown, not an assertion that it equals zero")
    (doseq [id '[input storage n]]
      (is (= :host-abstract-value-parameter
             (try (host-av/enter-scalar-parameter parent id :long) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (= :host-abstract-value-parameter
           (try (host-av/enter-scalar-parameter
                 (host-av/begin-analysis '(let* [] nil)
                                        {:values {'input (av/tensor {:dtype :double
                                                                    :shape ['shape-only]})}})
                 'shape-only :long) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "freshness includes symbols used only in retained shapes")))

(deftest host-shape-refinement-proves-a-complete-state-transition
  (let [analysis (host-av/analyze (source 'n) {:values {'u0 initial} :dtype :double})
        copy-expression (get (binding-map (source 'n)) 'copy)]
    (is (= ['n] (get-in analysis [:values 'u0 :shape])))
    (is (= ['n] (get-in analysis [:values 'u :shape])))
    (is (= ['n] (get-in analysis [:values 'scratch :shape])))
    (is (= ['n] (get-in analysis [:values 'next :shape])))
    (is (= 'n (get-in analysis [:equalities '(clojure.core/alength u)]))
        "the syntax extent used by TypedSOAC retains its relational shape proof")
    (is (= {:source 'next :destination 'u :extent 'n :dtype :double}
           (host-av/full-array-copy analysis copy-expression)))))

(deftest source-length-carriers-retain-analyzed-types-and-explicit-conversions
  (let [length (with-meta 'length {:raster.type/tag 'long})
        expression '(.invk raster.arrays/alength_m_doubles-impl input)
        source (list 'let* [length expression 'n (list 'int length)] 'n)
        analysis (host-av/analyze source {:values {'input initial}})]
    (is (= :long (get-in analysis [:values 'length :dtype])))
    (is (= :int (get-in analysis [:values 'n :dtype])))
    (is (= :int (get-in (host-av/analyze '(let* [n (clojure.core/alength input)] n)
                                          {:values {'input initial}})
                                  [:values 'n :dtype])))))

(deftest physical-constructor-capacity-retains-its-already-bound-count
  (let [source '(let* [n (clojure.core/alength input)
                      later (long (clojure.core/alength input))
                      storage (clojure.core/double-array n)] storage)
        analysis (host-av/analyze-capacities
                  source {:values {'input initial} :lexical-inputs '[input]})]
    (is (= '[n] (get-in analysis [:definition-values 'storage :shape])))
    (is (= '[later] (get-in analysis [:values 'storage :shape]))
        "logical relational refinement remains independent of physical allocation")
    (is (= :int (get-in analysis [:definition-values 'n :dtype])))
    (is (= :long (get-in analysis [:definition-values 'later :dtype])))))

(deftest partial-or-offset-copies-are-not-functionalized
  (let [analysis (host-av/analyze (source 'n) {:values {'u0 initial} :dtype :double})
        complete (get (binding-map (source 'n)) 'copy)]
    (testing "a shorter length lacks a full-state proof"
      (is (nil? (host-av/full-array-copy
                 analysis (list 'java.lang.System/arraycopy
                                'next 0 'u 0 '(dec n))))))
    (testing "an offset copy lacks a full-state proof"
      (is (nil? (host-av/full-array-copy
                 analysis (apply list (assoc (vec complete) 2 1))))))
    (testing "extra arguments do not masquerade as the arraycopy primitive"
      (is (nil? (host-av/full-array-copy analysis (concat complete [:extra])))))
    (testing "equal shape and dtype do not bridge incompatible representations"
      (is (nil? (host-av/full-array-copy
                 (assoc-in analysis [:values 'u :representation]
                           {:kind :quantized :scheme :q4-k})
                 complete))))))

(deftest declared-array-types-create-valid-conservative-seeds
  (let [analysis (host-av/analyze '(let* [] nil) {:array-types {'x :float}})]
    (is (= :float (get-in analysis [:values 'x :dtype])))
    (is (= '[(extent x)] (get-in analysis [:values 'x :shape])))))

(deftest retained-cast-and-descriptor-facts-drive-refinement
  (let [float-input (av/tensor {:dtype :float32 :shape ['m]
                                :representation {:kind :plain}})
        analysis
        (host-av/analyze
         '(let* [n (clojure.core/alength x)
                 y (raster.par/pmap i n float (float (clojure.core/aget x i)))
                 unknown (foo/similarity x)]
                y)
         {:values {'x float-input} :dtype :double})]
    (is (= :int (get-in analysis [:values 'n :dtype])))
    (is (= ['n] (get-in analysis [:values 'x :shape])))
    (is (= :float (get-in analysis [:values 'y :dtype])))
    (is (= ['n] (get-in analysis [:values 'y :shape])))
    (is (nil? (get-in analysis [:values 'unknown])))))

(deftest retained-scalar-metadata-is-propagated-without-operator-inference
  (let [binding (with-meta 'derived {:raster.type/tag 'double})
        expression (with-meta '(unknown.library/scalar-op input) {:tag 'double})
        analysis (host-av/analyze
                  (list 'let* [binding expression
                               'untyped '(unknown.library/scalar-op input)]
                        binding)
                  {})]
    (is (= {:kind :tensor :dtype :double :shape []}
           (select-keys (get-in analysis [:values binding]) [:kind :dtype :shape])))
    (is (nil? (get-in analysis [:values 'untyped])))))

(deftest complete-parallel-writes-retain-their-destination-contract
  (let [source
        '(let* [n (int (clojure.core/alength u))
                written (raster.par/stencil! scratch [u] 1 :dirichlet double i n
                                             (clojure.core/aget u i))]
               written)
        input (av/tensor {:dtype :double :shape ['extent] :representation {:kind :plain}})
        analysis (host-av/analyze source {:values {'u input 'scratch input}})
        expression (nth (second source) 3)]
    (is (= ['n] (get-in analysis [:values 'written :shape])))
    (is (= {:destination 'scratch :extent 'n :dtype :double :kind :stencil}
           (host-av/full-array-write analysis 'written expression)))
    (testing "a partial destination iteration space has no complete-write proof"
      (is (nil? (host-av/full-array-write
                 analysis 'written (apply list (assoc (vec expression) 7 '(dec n)))))))))

(deftest source-shaped-allocation-has-fresh-ownership
  (let [external (av/tensor {:dtype :double :shape ['n] :ownership :external
                             :effects #{:memory/read}
                             :attributes {:source :caller}})
        analysis (host-av/analyze
                  '(let* [copy (clojure.core/aclone input)] copy)
                  {:values {'input external}})
        copy (get-in analysis [:values 'copy])]
    (is (nil? (:ownership copy))
        "the relational pass clears caller ownership; allocation assigns physical ownership later")
    (is (= #{} (:effects copy)))
    (is (= {} (:attributes copy)))
    (is (= (select-keys external [:dtype :shape :logical-layout :representation])
           (select-keys copy [:dtype :shape :logical-layout :representation])))))

(deftest parametric-allocation-uses-its-exemplar-dtype-and-explicit-extent
  (let [input (av/tensor {:dtype :float :shape ['m] :representation {:kind :plain}})
        allocation (with-meta
                     '(.invk raster.arrays/zeros-like_m_floats_long-impl input n)
                     {:raster.op/original 'raster.arrays/zeros-like})
        analysis
        (host-av/analyze
         (list 'let* ['canonical-n 'n 'out allocation] 'out)
         {:values {'input input}
          :scalar-types {'n :long}})]
    (is (= {:dtype :float :shape ['canonical-n]}
           (select-keys (get-in analysis [:values 'out]) [:dtype :shape])))))

(deftest malformed-input-contracts-fail-loud
  (testing "authoritative values must be AbstractValues"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"expected an AbstractValue"
                          (host-av/analyze '(let* [] nil)
                                           {:values {'x {:dtype :double :shape ['n]}}}))))
  (testing "a dangling binding is never silently discarded"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"even binding vector"
                          (host-av/analyze '(let* [x 1 y] x) {})))))

(deftest incremental-host-analysis-reuses-the-same-flat-transfer
  (doseq [[form options]
          [[(source 'n) {:values {'u0 initial}}]
           [(list 'let* ['y (object-input 'x) 'z 'y] 'z)
            {:values {'x initial} :lexical-inputs '[x]}]]]
    (is (= (host-av/analyze form options)
           (host-av/finish-analysis
            (reduce (fn [context [id expression]] (host-av/advance-binding context id expression))
                    (host-av/begin-analysis form options) (partition 2 (second form))))))))

(deftest source-capacities-do-not-equate-an-independent-target-with-an-allocation
  (let [analysis (host-av/analyze-capacities
                  '(let* [out (clojure.core/double-array n)
                          target-length (clojure.core/alength target)] out)
                  {:array-types {'target :double} :scalar-types {'n :long}
                   :lexical-inputs '[target n]})]
    (is (= ['n] (get-in analysis [:definition-values 'out :shape])))
    (is (= ['n] (get-in analysis [:values 'out :shape])))
    (is (= '[(extent target)] (get-in analysis [:definition-values 'target :shape])))
    (is (= ['target-length] (get-in analysis [:values 'target :shape])))
    (is (not (contains? (:equalities analysis) 'n)))))

(deftest source-capacities-freeze-definitions-before-later-relational-refinement
  (let [source '(let* [out (clojure.core/double-array n)
                      size (clojure.core/alength out)
                      copy (clojure.core/aclone out)] copy)
        options {:scalar-types {'n :long} :lexical-inputs '[n]}
        analysis (host-av/analyze-capacities source options)]
    (is (= ['n] (get-in analysis [:definition-values 'out :shape])))
    (is (= ['size] (get-in analysis [:values 'out :shape])))
    (is (= ['size] (get-in analysis [:definition-values 'copy :shape])))
    (is (= (host-av/analyze source options) (dissoc analysis :definition-values)))
    (is (not (contains? (host-av/analyze source options) :definition-values)))))

(deftest source-capacities-use-certified-preceding-reference-returns
  (let [input (av/tensor {:dtype :double :shape ['m] :ownership :external
                          :attributes {:nullable true :source :public}})
        source '(let* [out (clojure.core/double-array n)
                       returned (unknown.library/write! out input)
                       copied returned] copied)
        analysis (host-av/analyze-capacities
                  source {:values {'input input} :scalar-types {'n :long}
                          :lexical-inputs '[input n]
                          :reference-aliases {'returned 'out}})]
    (is (= ['n] (get-in analysis [:definition-values 'returned :shape])))
    (is (= ['n] (get-in analysis [:definition-values 'copied :shape])))
    (is (= {'returned 'out 'copied 'returned} (:aliases analysis)))
    (is (= input (get-in analysis [:definition-values 'input])))
    (is (= input (get-in analysis [:values 'input])))
    (is (nil? (:definition-values (host-av/analyze source {:values {'input input}}))))))

(deftest source-capacities-cannot-borrow-future-seeds
  (let [future (av/tensor {:dtype :double :shape ['wrong]})
        source '(let* [early later
                       unknown (unknown.library/result)
                       later (clojure.core/double-array n)] later)
        options {:values {'later future 'early future 'unknown future}
                 :array-types {'unknown :double} :scalar-types {'n :long}
                 :lexical-inputs '[n]}
        analysis (host-av/analyze-capacities source options)]
    (is (nil? (get-in analysis [:definition-values 'early])))
    (is (nil? (get-in analysis [:definition-values 'unknown])))
    (is (= ['n] (get-in analysis [:definition-values 'later :shape])))
    (is (= :host-abstract-value-reference-alias
           (try (host-av/analyze-capacities
                 source (assoc options :reference-aliases {'early 'later})) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(deftest source-capacities-exclude-logical-non-buffer-results-and-unstable-identities
  (let [source '(let* [effect (raster.par/stencil! destination [input] 1 :dirichlet double i n
                                                 (clojure.core/aget input i))]
                      effect)
        analysis (host-av/analyze-capacities
                  source {:array-types {'input :double 'destination :double}
                          :scalar-types {'n :long}
                          :lexical-inputs '[input destination n]
                          :non-buffer-bindings '[effect]})]
    (is (= ['n] (get-in analysis [:values 'effect :shape]))
        "the existing transfer still records the logical tensor result")
    (is (nil? (get-in analysis [:definition-values 'effect]))
        "a certified effect-only host result is not a source buffer capacity"))
  (let [analysis (host-av/analyze-capacities
                  '(let* [input (clojure.core/double-array n)] input)
                  {:array-types {'input :double} :scalar-types {'n :long}
                   :lexical-inputs '[input n]})]
    (is (nil? (get-in analysis [:definition-values 'input]))
        "an unstable identity exposes neither its input seed nor its later definition")))

(deftest source-capacities-reject-invalid-reference-alias-assertions
  (let [value (av/tensor {:dtype :double :shape ['n]})
        options {:values {'input value 'other value} :lexical-inputs '[input other]}
        reason (fn [source options]
                 (try (host-av/analyze-capacities source options) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))]
    (doseq [aliases [nil [] {'out 42} {42 'input}]]
      (is (= :host-abstract-value-reference-aliases
             (reason '(let* [out input] out) (assoc options :reference-aliases aliases)))))
    (doseq [[source extra]
            [['(let* [out other] out) {:reference-aliases {'out 'input}}]
             ['(let* [out (unknown.library/result)] out)
              {:reference-aliases {'out 'missing}}]
             ['(let* [out input] out)
              {:reference-aliases {'missing 'input}}]
             ['(let* [out input] out)
              {:reference-aliases {'out 'input} :non-buffer-bindings '[out]}]
             ['(let* [effect input out (unknown.library/result)] out)
              {:reference-aliases {'out 'effect} :non-buffer-bindings '[effect]}]
             ['(let* [out input out input] out) {:reference-aliases {'out 'input}}]]]
      (is (= :host-abstract-value-reference-alias (reason source (merge options extra)))))
    (let [out (with-meta 'out {:raster.type/tag 'floats})]
      (is (= :host-abstract-value-reference-alias
             (reason (list 'let* [out '(unknown.library/result)] out)
                     (assoc options :reference-aliases {out 'input})))))
    (is (= :host-abstract-value-reference-alias
           (reason (list 'let* ['out (with-meta '(unknown.library/result)
                                     {:raster.type/tag 'floats})] 'out)
                   (assoc options :reference-aliases {'out 'input}))))))

(deftest logical-effect-results-cannot-launder-source-capacities
  (let [clone-id (with-meta 'cloned {:raster.type/tag 'doubles})
        zeros-id (with-meta 'zeros {:raster.type/tag 'doubles})
        zeros-expression (with-meta
                           '(.invk raster.arrays/zeros-like_m_doubles_long-impl effect n)
                           {:raster.op/original 'raster.arrays/zeros-like})
        source (list 'let*
                     ['effect '(raster.par/stencil! destination [input] 1 :dirichlet double i n
                                                    (clojure.core/aget input i))
                      clone-id '(clojure.core/aclone effect)
                      zeros-id zeros-expression
                      'clone-alias 'cloned 'zeros-alias 'zeros
                      'size '(clojure.core/alength effect)
                      'independent '(clojure.core/double-array n)]
                     'independent)
        options {:array-types {'input :double 'destination :double}
                 :scalar-types {'n :long} :lexical-inputs '[input destination n]
                 :non-buffer-bindings '[effect]}
        analysis (host-av/analyze-capacities source options)]
    (doseq [id '[effect cloned zeros clone-alias zeros-alias]]
      (is (nil? (get-in analysis [:definition-values id])))
      (is (not (contains? (:aliases analysis) id))))
    (is (= ['n] (get-in analysis [:definition-values 'independent :shape]))
        "alength of a logical-only result cannot rename a real source capacity")
    (is (= [] (get-in analysis [:definition-values 'size :shape]))
        "the existing scalar result contract may still be retained")
    (is (= :host-abstract-value-reference-alias
           (try (host-av/analyze-capacities
                 source (assoc options :reference-aliases {'clone-alias 'cloned})) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
    (is (= (:values (host-av/analyze source options)) (:values analysis)))
    (is (= (:equalities (host-av/analyze source options)) (:equalities analysis)))
    (is (= ['size] (get-in analysis [:values 'cloned :shape])))))

(deftest hinted-exemplar-results-require-preceding-stable-source-buffers
  (doseq [[bindings inputs]
          [[['copied (with-meta
                       '(.invk raster.arrays/zeros-like_m_floats_long-impl future n)
                       {:raster.op/original 'raster.arrays/zeros-like})
             'future '(clojure.core/float-array n)] '[n]]
           [['input '(clojure.core/float-array n)
             'copied (with-meta
                       '(.invk raster.arrays/zeros-like_m_floats_long-impl input n)
                       {:raster.op/original 'raster.arrays/zeros-like})] '[input n]]]]
    (let [hinted (mapv #(if (= 'copied %) (with-meta 'copied {:raster.type/tag 'floats}) %)
                       bindings)
          source (list 'let* (into hinted ['alias 'copied]) 'alias)
          analysis (host-av/analyze-capacities
                    source {:array-types {'input :float 'future :float}
                            :scalar-types {'n :long} :lexical-inputs inputs})]
      (is (nil? (get-in analysis [:definition-values 'copied])))
      (is (nil? (get-in analysis [:definition-values 'alias])))
      (is (not (contains? (:aliases analysis) 'alias))))))

(deftest unstable-scalar-binders-cannot-publish-capacity-equalities
  (doseq [[bindings inputs]
          [[['out '(clojure.core/double-array n)
             'size '(clojure.core/alength out) 'size 0
             'later '(clojure.core/double-array n)] '[n]]
           [['out '(clojure.core/double-array n)
             'size '(clojure.core/alength out)
             'later '(clojure.core/double-array n)] '[n size]]]]
    (let [source (list 'let* bindings 'later)
          options {:scalar-types {'n :long 'size :long} :lexical-inputs inputs}
          analysis (host-av/analyze-capacities source options)]
      (is (= ['n] (get-in analysis [:definition-values 'out :shape])))
      (is (= ['n] (get-in analysis [:definition-values 'later :shape])))
      (is (nil? (get-in analysis [:definition-values 'size])))
      (is (= (:values (host-av/analyze source (update options :scalar-types select-keys inputs)))
             (:values analysis))
          "both logical views start from the same public-only seeds")
      (is (= ['size] (get-in analysis [:values 'later :shape]))))))
