(ns raster.compiler.ir.soac-dialect-test
  (:require [clojure.test :refer [deftest is testing]]
            [pattern.nanopass.dialect :as pattern-dialect]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]))

(def ^:private tensor
  (av/tensor {:dtype :float :shape '[n]}))

(def ^:private extent
  (av/tensor {:dtype :long :shape []}))

(defn- map-program
  ([] (map-program {}))
  ([{:keys [values inputs equation-facts effects body]
     :or {values {'x tensor 'y tensor 'n extent}
          inputs '[n x]
          equation-facts (dialect/default-equation-facts)
          effects #{}
          body '(* x-element x-element)}}]
   (dialect/make
    (dialect/default-program-facts
     {:values values
      :inputs inputs
      :equations {'map-0 equation-facts}
      :effects effects})
    [(list '= 'map-0 '[y]
           (list 'map {:index 'i :extent 'n} '[x] []
                 (list 'lambda '[x-element] [body])))]
    '[y])))

(deftest typed-soac-is-a-pattern-declared-recursive-dialect
  (let [program (map-program)]
    (is (pattern-dialect/valid? dialect/TypedSOAC program))
    (is (= program (dialect/validate! program)))
    (is (= :map (keyword (name (dialect/operation-kind
                                (first (dialect/equations program)))))))
    (is (= '[x] (dialect/operation-inputs (first (dialect/equations program)))))))

(deftest scalar-regions-are-recursively-validated
  (testing "a non-language object cannot hide inside a scalar call"
    (let [bad-body (list '* 'x-element (Object.))]
      (try
        (map-program {:body bad-body})
        (is false "invalid scalar child must be rejected")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :typed-soac-syntax (:reason (ex-data exception)))))))))

(deftest value-and-effect-side-tables-are-authoritative
  (testing "every expression-spine value requires an AbstractValue"
    (try
      (map-program {:values {'y tensor 'n extent}})
      (is false "unknown array operand must be rejected")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-unknown-value (:reason (ex-data exception)))))))

  (testing "program effects equal the union of equation effects"
    (try
      (map-program {:equation-facts
                    (assoc (dialect/default-equation-facts) :effects #{:memory/read})})
      (is false "an unstated equation effect must be rejected")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-effects (:reason (ex-data exception)))))))

  (testing "stated effects validate without changing functional syntax"
    (let [program (map-program
                   {:equation-facts
                    (assoc (dialect/default-equation-facts) :effects #{:memory/read})
                    :effects #{:memory/read}})]
      (is (= #{:memory/read} (:effects (dialect/facts program)))))))

(deftest equations-are-ordered-ssa
  (let [values {'x tensor 'a tensor 'b tensor 'n extent}
        facts (dialect/default-program-facts
               {:values values
                :inputs '[n x]
                :equations {'use-a (dialect/default-equation-facts)
                            'define-a (dialect/default-equation-facts)}})
        equations
        [(list '= 'use-a '[b]
               (list 'map {:index 'i :extent 'n} '[a] []
                     (list 'lambda '[a-element] '[(* a-element 2.0)])))
         (list '= 'define-a '[a]
               (list 'map {:index 'i :extent 'n} '[x] []
                     (list 'lambda '[x-element] '[(* x-element x-element)])))]]
    (try
      (dialect/make facts equations '[b])
      (is false "an equation cannot consume a result defined later")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-use-before-definition (:reason (ex-data exception))))))))

(deftest captures-have-explicit-lexical-parameters
  (let [scale-id [:model :scale]
        scalar (av/tensor {:dtype :float :shape []})
        program
        (dialect/make
         (dialect/default-program-facts
          {:values {'x tensor 'y tensor 'n extent scale-id scalar}
           :inputs ['n 'x scale-id]
           :equations {'map-0 (dialect/default-equation-facts)}})
         [(list '= 'map-0 '[y]
                (list 'map {:index 'i :extent 'n} '[x] [scale-id]
                      (list 'lambda '[x-element scale]
                            '[(* x-element scale)])))]
         '[y])]
    (is (= {:accumulators []
            :elements '[x-element]
            :capture-parameters '[scale]}
           (dialect/parameter-layout (first (dialect/equations program))))))

  (testing "the expression spine cannot hide an undeclared dependency"
    (try
      (map-program {:body '(* x-element scale)})
      (is false "free scalar symbols require capture operands and parameters")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-unbound-scalar (:reason (ex-data exception))))))))

(deftest compound-stable-extent-ids-have-an-unambiguous-shape-reference
  (let [extent-id [:batch :rows]
        shaped (av/tensor {:dtype :float :shape (dialect/extent-shape extent-id)})
        program
        (dialect/make
         (dialect/default-program-facts
          {:values {extent-id extent 'x shaped 'y shaped}
           :inputs [extent-id 'x]
           :equations {'map-0 (dialect/default-equation-facts)}})
         [(list '= 'map-0 '[y]
                (list 'map {:index 'i :extent extent-id} '[x] []
                      (list 'lambda '[x-element] '[(* x-element 2.0)])))]
         '[y])]
    (is (= [(list 'value extent-id)] (:shape (get-in (dialect/facts program)
                                                     [:values 'x]))))))
