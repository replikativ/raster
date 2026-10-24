(ns raster.compiler.passes.scalar.host-abstract-value-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.host-abstract-value :as host-av]))

(def ^:private initial
  (av/tensor {:dtype :double :shape ['?] :representation {:kind :plain}}))

(defn- object-input [operand]
  (completion/complete-value completion/object-input-contract operand))

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
