(ns raster.compiler.passes.scalar.host-buffer-alias-extent-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.passes.scalar.host-abstract-value :as host-av]))

(defn- object-input [operand]
  (completion/complete-value completion/object-input-contract operand))

(defn- reference-entry [tag operand]
  (completion/complete-value
   (completion/bytecode-boxed-parameter-contract (get types/primitive-array-info tag)) operand))

(defn- analyze [value expression]
  (host-av/analyze-capacities
   (list 'let* ['alias expression] 'alias)
   {:values {'input value} :lexical-inputs '[input]}))

(deftest exact-buffer-reference-chains-retain-local-backing-identities
  (let [value (av/tensor {:dtype :float :shape '[(extent input)]
                         :ownership :external :attributes {:nullable true}})
        object (object-input 'input)
        checked (reference-entry 'floats 'boxed)
        source (list 'let* ['boxed object 'checked checked 'alias 'checked] 'alias)
        result (host-av/analyze-capacities source {:values {'input value} :lexical-inputs '[input]})]
    (is (= {'boxed 'input 'checked 'boxed 'alias 'checked} (:aliases result)))
    (doseq [id '[input boxed checked alias]]
      (is (= [(list 'extent id)] (get-in result [:definition-values id :shape])))
      (is (= (dissoc value :shape) (dissoc (get-in result [:definition-values id]) :shape))))
    (is (identical? object (nth (second source) 1)))
    (is (identical? checked (nth (second source) 3))
        "a conditional alias fact does not remove or execute the retained CHECKCAST")))

(deftest explicit-capacities-and-nonflat-storage-are-never-rebased
  (let [base (av/tensor {:dtype :float :shape '[(extent input)]})]
    (doseq [[reason value]
            [[:scalar-count (assoc base :shape '[capacity])]
             [:static-count (assoc base :shape [17])]
             [:foreign-backing (assoc base :shape '[(extent unrelated)])]
             [:multidimensional (assoc base :shape '[(extent input) 2])]
             [:logical-view (assoc base :logical-layout {:kind :strided})]
             [:packed-storage (assoc base :representation {:kind :packed})]]]
      (testing (name reason)
        (let [result (analyze value 'input)]
          (is (= value (get-in result [:definition-values 'alias])))
          (is (= {'alias 'input} (:aliases result))))))))

(deftest array-class-or-shape-similarity-cannot-authorize-rebasing
  (let [value (av/tensor {:dtype :float :shape '[(extent input)]})]
    (doseq [expression [(reference-entry 'doubles 'input)
                        '(clojure.core/identity input)
                        (object-input '(clojure.core/identity input))
                        (reference-entry 'floats '(clojure.core/identity input))]]
      (let [result (analyze value expression)]
        (is (nil? (get-in result [:aliases 'alias])))
        (is (not= '[(extent alias)] (get-in result [:definition-values 'alias :shape])))))
    (let [result (host-av/analyze-capacities
                  '(let* [alias input] alias)
                  {:values {'input value} :lexical-inputs []})]
      (is (nil? (get-in result [:aliases 'alias])))
      (is (nil? (get-in result [:definition-values 'alias]))))))

(deftest aliases-of-source-allocations-retain-the-constructor-count
  (let [result (host-av/analyze-capacities
                (list 'let* ['input '(clojure.core/float-array count)
                             'boxed (object-input 'input)
                             'checked (reference-entry 'floats 'boxed)] 'checked)
                {:scalar-types {'count :long} :lexical-inputs '[count]})]
    (doseq [id '[input boxed checked]]
      (is (= '[count] (get-in result [:definition-values id :shape]))))
    (is (= {'boxed 'input 'checked 'boxed} (:aliases result)))))

(deftest rebound-capacity-still-resolves-from-the-original-runtime-buffer
  (let [input-value (av/tensor {:dtype :float :shape '[(extent input)]})
        alias-value (get-in (analyze input-value 'input) [:definition-values 'alias])
        plan (invocation/from-prefix
              {:id :self-backed-buffer-alias :parameters '[input]
               :parameter-values {'input input-value}
               :bindings '[[alias input]] :binding-values {'alias alias-value}
               :program-values {'alias alias-value}
               :program-inputs '[alias] :program-outputs '[alias]})]
    (doseq [length [0 3 17]]
      (let [source (float-array length)
            initial (materialization/begin-materialization plan [source])
            advanced (materialization/materialize-next initial nil)
            result (materialization/finish-materialization advanced)
            buffer (get-in result [:program-buffers 'alias])]
        (is (= [length] (:shape buffer)))
        (is (identical? source (:source buffer)))
        (let [id (:id (first (:steps plan)))
              forged (assoc-in advanced [:values id :source] (float-array length))]
          (is (= :invocation-materialization-state-producer
                 (try (materialization/finish-materialization forged) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))))
