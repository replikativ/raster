(ns raster.compiler.passes.parallel.structured-prefix-composition-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.structured-control-frontend :as frontend]
            [raster.compiler.passes.parallel.structured-control-route :as route]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed]
            [raster.gpu.core :as gpu]))

(def ^:private options
  {:dtype :double :source-ns *ns* :public-parameters '[input steps]
   :values {'input (av/tensor {:dtype :double :shape ['extent] :representation {:kind :plain}})
            'steps (av/tensor {:dtype :long :shape []})}
   :array-types {'input :double} :scalar-types {'steps :long}})

(def ^:private source
  '(let* [n (int (clojure.core/alength input))
          state (clojure.core/aclone input)
          scratch (clojure.core/double-array n)
          count (int steps)
          time-loop (dotimes [tick count]
                      (let* [next (raster.par/pmap j n double
                                                  (+ (clojure.core/aget state j) 1.0))]
                        (java.lang.System/arraycopy next 0 state 0 n)))
          result (raster.par/pmap k n double (* 2.0 (clojure.core/aget state k)))]
     result))

(defn- reduction-return-source [result]
  (list 'let*
        (into (cond-> (vec (drop-last 8 (second source)))
                (= result '[snapshot total])
                (into ['snapshot '(raster.par/pmap j n double
                                                   (* 3.0 (clojure.core/aget input j)))]))
              (concat (take-last 6 (drop-last 2 (second source)))
                      ['total '(raster.par/reduce acc 0.0 k n
                                  (+ acc (clojure.core/aget state k)))]
                      (when (= result '[total mapped])
                        ['mapped '(raster.par/pmap j n double
                                                  (+ total (clojure.core/aget state j)))])))
        result))

(deftest whole-source-return-owns-suffix-reduction-residency
  (let [forbidden (fn [& _] (throw (AssertionError. "semantic producer touched a device")))]
    (with-redefs [gpu/make-session forbidden gpu/alloc! forbidden]
      (doseq [result ['total '[state total] '[snapshot total] '[total mapped]]]
        (let [source (reduction-return-source result)
              decomposition (frontend/form->structured-loop source options)
              program (route/program-envelope decomposition (assoc options :resident-reductions? true))
              resident (route/validated-public-resident-output program 'total)]
          (is (some? resident) (str "whole return " result))
          (is (= {:kind :resident-scalar-buffer :elements 1} (:representation resident)))
          (is (= :device (:memory-space resident)))
          (is (some #{'total} (:outputs program)))
          (is (some #(= [] (:shape (get (:values program) %))) (:outputs program))))))))

(deftest source-free-and-disabled-residency-do-not-acquire-a-return-demand
  (let [source (reduction-return-source 'total)
        decomposition (frontend/form->structured-loop source options)]
    (doseq [[decomposition options]
            [[decomposition (assoc options :resident-reductions? false)]
             [(select-keys decomposition [:loop :loop-binding :suffix-bindings :outer-body :outer-values])
              {:dtype :double :source-ns *ns* :resident-reductions? true}]]]
      (let [program (route/program-envelope decomposition options)]
        (is (nil? (route/validated-public-resident-output program 'total)))
        (is (= {:kind :plain} (get-in program [:values 'total :representation])))))))

(deftest prefix-to-suffix-values-preserve-producers-and-reject-contract-drift
  (let [source (typed/normalize-source (reduction-return-source '[snapshot total]) options)
        decomposition (frontend/form->structured-loop source options)
        prefix (#'route/prefix-program decomposition options)
        trip-id (second (control/loop-index (:loop decomposition)))
        project #(#'route/suffix-prefix-values %1 %2 (:loop decomposition))
        entry (project decomposition prefix)
        refused? (fn [decomposition prefix]
                   (= :structured-control-prefix-suffix-value
                      (try (project decomposition prefix) nil
                           (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))]
    (is (identical? (get-in prefix [:envelope :values 'snapshot]) (get entry 'snapshot)))
    (is (contains? entry 'n) "the producer's shape dependency remains declared")
    (doseq [changed [(assoc-in decomposition [:outer-values 'snapshot :dtype] :float)
                     (assoc-in decomposition [:outer-values 'snapshot :shape] [999])]]
      (is (refused? changed prefix)))
    (is (refused? (update decomposition :outer-values dissoc 'n)
                  (update-in prefix [:envelope :values] dissoc 'n)))
    (is (symbol? trip-id))
    (is (refused? (assoc-in decomposition [:outer-values 'snapshot :shape] [trip-id])
                  (-> prefix
                      (assoc-in [:envelope :values 'snapshot :shape] [trip-id])
                      (assoc-in [:envelope :values trip-id]
                                (av/tensor {:dtype :double :shape []})))))
    (is (not (contains? (project decomposition
                                 (update-in prefix [:envelope :outputs]
                                            #(vec (remove #{'snapshot} %))))
                        'snapshot)))
    (is (not (contains? entry 'state)) "loop storage contracts take precedence over prefix state")))

(deftest prefix-value-identity-does-not-resolve-a-local-as-a-core-function
  (let [decomposition (frontend/form->structured-loop source options)
        value (av/tensor {:dtype :double :shape []})
        entry (#'route/suffix-prefix-values
               {:outer-values {'first value} :outer-equalities {}
                :suffix-bindings [] :outer-body ['first]}
               {:envelope {:values {'first value} :outputs ['first]}}
               (:loop decomposition))]
    (is (identical? value (get entry 'first)))))

(deftest checked-host-tail-keeps-its-recipes-without-numerical-admission
  (let [source (typed/normalize-source source options)
        decomposition (frontend/form->structured-loop source options)
        prefix (list 'let* (vec (mapcat identity (:prefix-bindings decomposition))) nil)
        facts (typed/source-fragment-analysis prefix options)
        allocation (some #(when (= 'scratch (:destination %)) %)
                         (get-in facts [:attributes :allocations]))]
    (is (some? decomposition))
    (is (seq (:numerical-binding-ids facts)))
    (is (nil? (typed/form->program prefix options))
        "the checked int after the copy is still not an ordinary numerical island")
    (is (= '(clojure.core/double-array n) (:source-expression allocation)))
    (is (= (:source-binding-id allocation)
           (.indexOf (mapv first (:prefix-bindings decomposition)) 'scratch)))
    (is (some #{'count} (get-in facts [:attributes :source-bindings])))))

(deftest prefix-copy-feeds-the-loop-and-retained-host-tail-in-order
  (let [forbidden (fn [& _] (throw (AssertionError. "producer touched a device")))]
    (with-redefs [gpu/make-session forbidden gpu/alloc! forbidden]
      (let [decomposition (frontend/form->structured-loop source options)
            semantic (route/program-envelope decomposition options)
            loop-equation (some #(when (control/loop-program? (:algorithm %)) %) (:equations semantic))
            loop (:algorithm loop-equation)
            prefix (filter #(= :prefix (first (:id %))) (:equations semantic))
            suffix (filter #(= :suffix (first (:id %))) (:equations semantic))
            initial (:initial (first (control/carried loop)))
            plan (get-in semantic [:attributes :invocation-plan])
            order (invocation/source-order plan semantic)
            binders (vec (take-nth 2 (second (:source semantic))))
            position #(.indexOf binders %)
            copy (some #(when (some #{initial} (:results %)) %) prefix)]
        (is (some? copy))
        (is (not= 'input initial) "zero trips start from the independent copy, not borrowed input")
        (is (not-any? invocation/buffer-clone? (:steps plan)))
        (is (< (position (second (:site copy))) (position 'scratch)
               (position 'count) (position 'time-loop)))
        (is (= '(int steps) (get (into {} (map vec) (partition 2 (second (:source semantic)))) 'count)))
        (is (= (mapv :id (:equations semantic)) (vec (keep :equation order))))
        (is (some (set (control/outer-results loop)) (mapcat :operands suffix)))
        (is (= 1 (count (filter #(= 'scratch (:storage %)) order))))
        (is (not-any? fn? (tree-seq coll? seq semantic)))))))

(deftest independently-generated-fragment-results-have-disjoint-scopes
  (let [typed (typed/form->program
               '(let* [effect (raster.par/map2! a b i n double
                                               (clojure.core/aget x i)
                                               (clojure.core/aget y i))] effect)
               {:dtype :double :array-types {'a :double 'b :double 'x :double 'y :double}})
        results #(set (mapcat (fn [equation] (nth equation 2)) (soac/equations %)))
        before (#'route/scope-generated-results typed :prefix)
        after (#'route/scope-generated-results typed :suffix)]
    (is (every? vector? (results typed)))
    (is (seq (results typed)))
    (is (empty? (set/intersection (results before) (results after))))
    (is (= before (soac/validate! before)))
    (is (= after (soac/validate! after)))
    (is (= (mapv #(soac/result-storage (soac/facts typed) (second %)) (soac/equations typed))
           (mapv #(soac/result-storage (soac/facts before) (second %)) (soac/equations before))))))

(deftest an-interleaved-checked-host-operation-still-refuses-numerical-admission
  (let [bindings (second source)
        bindings (vec (concat (subvec bindings 0 8)
                              ['extra '(raster.par/pmap i n double (clojure.core/aget state i))]
                              (subvec bindings 8)))
        source (list 'let* bindings (last source))
        decomposition (frontend/form->structured-loop source options)]
    (is (some? decomposition))
    (is (= :structured-control-unsupported-prefix
           (try (route/program-envelope decomposition options) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
