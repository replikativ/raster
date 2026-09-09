(ns raster.compiler.ir.effect-region-scope-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn- region []
  (list 'effect-region
        [(list 'let-value 'local :long 'seed)]
        [(list 'effect 'out :unique 0 true 'local)
         (list 'effect-loop {:index 'k :lower 0 :carry {:parameter 'acc :result 'sum :dtype :long}}
               'n 'local
               (list 'lambda '[k acc]
                     (list 'effect-region
                           [(list 'let-value 'term :long '(aget source k))]
                           [(list 'effect 'out :unique 'k true 'term)]
                           (with-meta '(+ acc term) {:raster.type/tag 'long}))))
         (list 'effect 'out :unique 1 true 'sum)]))

(deftest scope-authority-models-interleaved-results-as-sequential-bindings
  (let [r (region) scope (form/scope-info r)]
    (is (:sequential? scope))
    (is (= ['local nil 'sum nil] (get-in scope [:scopes 0 :binders])))
    (is (= #{'seed 'out 'source 'n} (util/free-syms r)))
    (is (= r ((:rebuild scope) (:scopes scope) (:outer scope))))
    (let [before (list 'effect-region (second r)
                       (assoc (nth r 2) 0 (list 'effect 'out :unique 0 true 'sum)))]
      (is (contains? (util/free-syms before) 'sum) "result is not in scope before its loop"))))

(deftest malformed-scope-shapes-are-not-destructured-as-binding-regions
  (doseq [r ['(effect-region 1 2) '(effect-region [17] []) '(effect-loop {} 3 1)]]
    (is (nil? (form/scope-info r)))))

(deftest substitution-avoids-capturing-external-values-in-the-continuation
  (let [r (util/subst-syms {'seed 'sum} (region))
        effects (nth r 2)
        carry (:carry (dialect/effect-parts (second effects)))
        result (:result carry)]
    (is (not= 'sum result))
    (is (= 'sum (nth (first (second r)) 3)))
    (is (= result (last (last effects))))
    (is (= #{'sum 'out 'source 'n} (util/free-syms r)))
    (is (= 'long (:raster.type/tag (meta (:update carry)))))
    (is (= ['effect 'effect-loop 'effect] (mapv first effects)))))

(deftest alpha-conversion-preserves-lexical-scope-and-recurrence-links
  (let [r (util/alpha-convert (region))
        effects (nth r 2)
        loop (dialect/effect-parts (second effects))
        {:keys [locals]} (dialect/lambda-parts (:lambda loop))
        carry (:carry loop)]
    (is (= (util/free-syms (region)) (util/free-syms r)))
    (is (not= 'k (:index loop)))
    (is (not= 'acc (:parameter carry)))
    (is (not= 'sum (:result carry)))
    (is (= (:result carry) (last (last effects))))
    (is (= (list '+ (:parameter carry) (:id (first locals))) (:update carry)))
    (is (= (:index loop) (last (:init (first locals)))))))
