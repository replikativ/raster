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

(defn- atomic-region []
  (list 'effect-region []
        [(list 'effect 'counts (dialect/reducing-scatter-conflict '+ :int) 0 true 'seed
               {:result 'ticket :dtype :int})
         (list 'effect 'out :unique 'ticket true 'seed)]))

(deftest atomic-results-use-the-same-sequential-scope-as-loop-results
  (let [r (atomic-region) scope (form/scope-info r)]
    (is (= ['ticket nil] (get-in scope [:scopes 0 :binders])))
    (is (= #{'counts 'out 'seed} (util/free-syms r)))
    (is (= r ((:rebuild scope) (:scopes scope) (:outer scope))))
    (let [atomic (first (nth r 2))]
      (is (= #{'counts 'seed} (util/free-syms atomic)))
      (is (contains? (util/free-syms (list 'effect-region []
                                         [(nth (nth r 2) 1) atomic])) 'ticket)
          "the result is unavailable before its effect"))))

(deftest atomic-result-substitution-and-alpha-renaming-preserve-declarations
  (let [r (atomic-region)
        substituted (util/subst-syms {'seed 'ticket} r)
        renamed (util/alpha-convert r)]
    (doseq [form [substituted renamed]]
      (let [[atomic store] (nth form 2)
            result (:result (nth atomic 6))]
        (is (not= 'ticket result))
        (is (= result (nth store 3)))
        (is (= :int (:dtype (nth atomic 6))))
        (is (= (nth (first (nth r 2)) 2) (nth atomic 2)))))
    (is (= 'ticket (nth (first (nth substituted 2)) 5)))
    (is (= #{'counts 'out 'ticket} (util/free-syms substituted)))
    (is (= (util/free-syms r) (util/free-syms renamed)))
    (is (= (util/alpha-normalize r) (util/alpha-normalize renamed)))))

(deftest atomic-rebinding-preserves-metadata-and-static-contracts
  (let [r (atomic-region)
        atomic (with-meta (first (nth r 2)) {:line 17})
        declaration (with-meta (nth atomic 6) {:line 18})
        atomic (with-meta (apply list (assoc (vec atomic) 6 declaration)) (meta atomic))
        r (list 'effect-region [] [atomic (second (nth r 2))])
        rebound (util/subst-syms {'seed 'ticket '+ 'changed-operator} r)
        actual (first (nth rebound 2))]
    (is (= (meta atomic) (meta actual)))
    (is (= (meta declaration) (meta (nth actual 6))))
    (is (= (nth atomic 2) (nth actual 2)) "the conflict algebra is not a lexical use")
    (is (not= 'ticket (:result (nth actual 6))))))
