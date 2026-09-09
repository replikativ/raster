(ns raster.compiler.passes.parallel.effect-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.effect-source :as source]))

(deftest result-binding-encloses-only-the-ordered-continuation
  (let [effects [{:source '(swap! log conj result)}
                 {:loop {:carry {:result 'result}
                         :effects [{:source '(swap! log conj :inside)}]}}
                 {:source '(swap! log conj result)}]
        form (source/ordered-effects
              effects {:emit-store :source
                       :emit-loop (fn [_ body] (list 'do body 11))})
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
              {:emit-store :source :emit-loop (fn [_ body] body)})
        execute (eval (list 'fn '[log] form))
        log (atom [])]
    (is (nil? (source/ordered-effects [] {})))
    (is (nil? (execute log)))
    (is (= [:before :inside :after] @log))))
