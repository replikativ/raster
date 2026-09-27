(ns raster.compiler.passes.parallel.effect-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.effect-source :as source]))

(deftest result-binding-encloses-only-the-ordered-continuation
  (let [effects [{:source '(swap! log conj result)}
                 {:loop {:carries [{:result 'result :dtype :long}]
                         :effects [{:source '(swap! log conj :inside)}]}}
                 {:source '(swap! log conj result)}]
        form (source/ordered-effects
              effects nil
              {:emit-store :source
               :emit-loop (fn [_ emit-body] (list 'do (emit-body nil) [11]))})
        execute (eval (list 'fn '[log result] form))
        log (atom [])]
    (is (contains? (util/free-syms form) 'result)
        "the preceding effect reads the outer value, not the exported result")
    (is (nil? (execute log 7)))
    (is (= [7 :inside 11] @log))))

(deftest ordinary-loops-and-empty-regions-preserve-effect-order
  (let [form (source/ordered-effects
              [{:source '(swap! log conj :before)}
               {:loop {:effects [{:source '(swap! log conj :inside)}]}}
               {:source '(swap! log conj :after)}]
              nil {:emit-store :source :emit-loop (fn [_ emit-body] (emit-body nil))})
        execute (eval (list 'fn '[log] form))
        log (atom [])]
    (is (= :tail (source/ordered-effects [] :tail {})))
    (is (nil? (execute log)))
    (is (= [:before :inside :after] @log))))

(deftest guarded-region-dominates-its-local-initializers
  (let [form (source/ordered-effects
              [{:region {:predicate 'active?
                         :locals [{:id 'checked :dtype :long :init '(inc n)}]
                         :effects [{:source '(swap! log conj checked)}]}}
               {:source '(swap! log conj :after)}]
              nil {:emit-store :source
               :emit-region (fn [locals body]
                              (list 'let* (vec (mapcat (juxt :id :init) locals)) body))})
        execute (eval (list 'fn '[log active? n] form))]
    (let [log (atom [])]
      (is (nil? (execute log false Long/MAX_VALUE)))
      (is (= [:after] @log)))
    (let [log (atom [])]
      (is (nil? (execute log true 4)))
      (is (= [5 :after] @log)))))

(deftest counted-recurrence-is-inside-an-exported-effect-result
  (let [loop {:index 'k :lower 0 :extent 2 :locals []
              :effects [{:destination 'counter
                         :conflict (dialect/reducing-scatter-conflict '+ :int)
                         :destination-index 0 :predicate true :value '(inc acc)
                         :result 'previous :result-dtype :int}]
              :carries [{:parameter 'acc :result 'sum :dtype :int
                         :init 0 :update 'previous}]}
        form (source/ordered-effects
              [{:loop loop}] 'sum
              {:emit-store :value
               :emit-loop (partial source/counted-loop
                                   (partial source/storage-cast true))})
        execute (eval (list 'fn [] form))]
    (is (not (contains? (util/free-syms form) 'previous))
        "the recurrence consumes the atomic result inside its binding")
    (is (= 2 (execute)))))

(deftest product-fold-tuple-is-bound-once-after-its-captures
  (let [product
        '(product-fold {:accumulators [sum squares] :identities [0.0 0.0]
                        :dtypes [:float :float] :index i :lower 0 :extent n
                        :association :ordered}
           (lambda [sum squares i]
             (region [] [(+ sum mean) (+ squares (* mean mean))])))
        locals [{:id 'mean :dtype :float :init 2.0}
                {:id 'sum-result :dtype :float
                 :init (list 'product-component product 0)}
                {:id 'squares-result :dtype :float
                 :init (list 'product-component product 1)}]
        form (source/typed-locals (partial source/storage-cast true)
                                  locals '[sum-result squares-result])
        execute (eval (list 'fn '[n] form))]
    (is (= [6.0 12.0] (mapv double (execute 3))))
    (is (= 1 (count (filter #(and (seq? %) (= 'loop* (first %)))
                            (tree-seq coll? seq form))))
        "both projections share one tuple loop after the mean binding")))
