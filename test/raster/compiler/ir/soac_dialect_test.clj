(ns raster.compiler.ir.soac-dialect-test
  (:require [clojure.test :refer [deftest is testing]]
            [pattern.nanopass.dialect :as pattern-dialect]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.scan :as scan]
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
                 (dialect/lambda-form '[x-element] [body])))]
    '[y])))

(deftest typed-soac-is-a-pattern-declared-recursive-dialect
  (let [program (map-program)]
    (is (pattern-dialect/valid? dialect/TypedSOAC program))
    (is (= program (dialect/validate! program)))
    (is (= :map (keyword (name (dialect/operation-kind
                                (first (dialect/equations program)))))))
    (is (= '[x] (dialect/operation-inputs (first (dialect/equations program)))))))

(deftest tuple-map-storage-is-ordered-typed-and-effectful
  (let [left [:effect-map 0 0]
        right [:effect-map 0 1]
        storage [{:destination 'a :access :write :host-return :effect}
                 {:destination 'b :access :read-write :host-return :effect}]
        equation (list '= 'map-0 [left right]
                       (list 'map {:index 'i :extent 'n} '[x b] []
                             (dialect/lambda-form '[x-element b-element]
                                                  '[x-element (+ b-element 1.0)])))
        equation-facts (assoc (dialect/default-equation-facts)
                              :effects #{:memory/write}
                              :aliases {left 'a right 'b}
                              :attributes {:result-storage storage})
        facts (dialect/default-program-facts
               {:values {'n extent 'x tensor 'a tensor 'b tensor
                         left tensor right tensor}
                :inputs '[b n x]
                :equations {'map-0 equation-facts}
                :effects #{:memory/write}})
        program (dialect/make facts [equation] [])
        remapped (dialect/remap-values
                  program
                  {left [:logical :left]
                   right [:logical :right]
                   'a [:storage :a]
                   'b [:storage :b]
                   'x [:argument :x]
                   'n [:shape :n]})
        remapped-equation (first (dialect/equations remapped))]
    (is (= ['a 'b] (dialect/physical-results program equation)))
    (is (= storage (dialect/result-storage program 'map-0)))
    (is (= [[:storage :a] [:storage :b]]
           (dialect/physical-results remapped remapped-equation))
        "value remapping retains the ordered physical storage identity")
    (is (= {[:logical :left] [:storage :a]
            [:logical :right] [:storage :b]}
           (get-in (dialect/facts remapped) [:equations 'map-0 :aliases])))
    (testing "storage cannot drift from result order"
      (try
        (dialect/make
         (assoc-in facts [:equations 'map-0 :aliases right] 'a)
         [equation] [])
        (is false "misaligned storage aliases must fail")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :typed-soac-result-storage-alias (:reason (ex-data exception)))))))
    (testing "storage destinations require their own AbstractValue"
      (try
        (dialect/make (update facts :values dissoc 'b) [equation] [])
        (is false "unknown physical storage must fail")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :typed-soac-result-storage-value (:reason (ex-data exception)))))))))

(deftest scalar-region-locals-are-explicit-typed-and-ordered-ssa
  (let [equation
        (list '= 'map-0 '[y]
              (list 'map {:index 'i :extent 'n} '[x] []
                    (dialect/lambda-form
                     '[element]
                     [(dialect/local-value 'shifted :float '(+ element 1.0))
                      (dialect/local-value 'squared :float '(* shifted shifted))]
                     '[squared])))
        facts (dialect/default-program-facts
               {:values {'n extent 'x tensor 'y tensor}
                :inputs '[n x]
                :equations {'map-0 (dialect/default-equation-facts)}})
        program (dialect/make facts [equation] '[y])
        parts (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= [{:id 'shifted :dtype :float :init '(+ element 1.0)}
            {:id 'squared :dtype :float :init '(* shifted shifted)}]
           (:locals parts)))
    (is (= program (dialect/validate! program)))
    (testing "a local initializer cannot read a later SSA definition"
      (let [bad-equation
            (list '= 'map-0 '[y]
                  (list 'map {:index 'i :extent 'n} '[x] []
                        (dialect/lambda-form
                         '[element]
                         [(dialect/local-value 'shifted :float '(+ squared 1.0))
                          (dialect/local-value 'squared :float '(* element element))]
                         '[shifted])))]
        (try
          (dialect/make facts [bad-equation] '[y])
          (is false "use-before-definition inside a region must fail")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-soac-unbound-local (:reason (ex-data exception))))))))
    (testing "a local cannot shadow a lambda parameter"
      (let [bad-equation
            (list '= 'map-0 '[y]
                  (list 'map {:index 'i :extent 'n} '[x] []
                        (dialect/lambda-form
                         '[element]
                         [(dialect/local-value 'element :float '(+ element 1.0))]
                         '[element])))]
        (try
          (dialect/make facts [bad-equation] '[y])
          (is false "lexical SSA binders must be unique")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-soac-region-binders (:reason (ex-data exception))))))))
    (testing "a local dtype is checked against the authoritative dtype facets"
      (doseq [unsupported [:made-up :half :f32]]
        (let [bad-equation
              (list '= 'map-0 '[y]
                    (list 'map {:index 'i :extent 'n} '[x] []
                          (dialect/lambda-form
                           '[element]
                           [(dialect/local-value 'local unsupported '(+ element 1.0))]
                           '[local])))]
          (try
            (dialect/make facts [bad-equation] '[y])
            (is false "unsupported scalar-region dtype must fail at the IR boundary")
            (catch clojure.lang.ExceptionInfo exception
              (is (= :typed-soac-local-dtype (:reason (ex-data exception)))))))))))

(deftest scan-mode-has-a-distinct-typed-and-effectful-contract
  (let [out (av/tensor {:dtype :float :shape '[(unknown-dimension out)]})
        algebra (scan/->AssociativeScan 'acc 0.0 '+ 'element '(float 0.0) :float)
        facts (dialect/default-program-facts
               {:values {'n extent 'x tensor 'out out 'result tensor}
                :inputs '[n out x]
                :equations
                {'scan-0 (assoc (dialect/default-equation-facts)
                                :effects #{:memory/write}
                                :aliases '{result out}
                                :attributes {:destination 'out})}
                :effects #{:memory/write}})
        program (dialect/make
                 facts
                 [(list '= 'scan-0 '[result]
                        (list 'scan
                              {:mode :inclusive :index 'i :extent 'n
                               :accumulators '[acc] :identities [0.0]
                               :dtypes [:float] :algebra [algebra]
                               :attributes {:stable-array-captures '[out]}}
                              '[x] '[out]
                              (dialect/lambda-form '[acc element destination]
                                                   '[(+ acc element)])))]
                 '[result])
        equation (first (dialect/equations program))]
    (is (= program (dialect/validate! program)))
    (is (= 'scan (dialect/operation-kind equation)))
    (is (= {:accumulators '[acc]
            :elements '[element]
            :capture-parameters '[destination]}
           (dialect/parameter-layout equation)))
    (is (= '[x out] (dialect/operation-inputs equation)))
    (testing "exclusive mode requires its n+1 result contract"
      (let [[_ attributes arrays captures lambda] (nth equation 3)
            exclusive-result (av/tensor {:dtype :float :shape '[(clojure.core/inc n)]})
            exclusive-facts (assoc-in facts [:values 'result] exclusive-result)
            exclusive-equation (list '= 'scan-0 '[result]
                                     (list 'scan (assoc attributes :mode :exclusive)
                                           arrays captures lambda))]
        (is (= :exclusive
               (get-in (dialect/operation-parts
                        (first (dialect/equations
                                (dialect/make exclusive-facts [exclusive-equation] '[result]))))
                       [:attributes :mode])))))
    (testing "unknown scan modes cannot enter the typed dialect"
      (let [[_ attributes arrays captures lambda] (nth equation 3)
            bad-equation (list '= 'scan-0 '[result]
                               (list 'scan (assoc attributes :mode :unknown)
                                     arrays captures lambda))]
        (try
          (dialect/make facts [bad-equation] '[result])
          (is false "unknown mode must fail TypedSOAC syntax validation")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-soac-syntax (:reason (ex-data exception))))))))))

(defn- stencil-program
  []
  (let [double-tensor (av/tensor {:dtype :double :shape '[n]})
        scalar-double (av/tensor {:dtype :double :shape []})
        storage [{:destination 'du :access :write :host-return :buffer}]
        equation
        (list '= 'stencil-0 '[result]
              (list 'stencil
                    {:index 'i :extent 'n :radius 1 :boundary :dirichlet
                     :dtypes [:double]
                     :attributes {:stable-array-captures '[u]}}
                    [] '[alpha u]
                    (dialect/lambda-form
                     '[a input]
                     '[(double (* a (+ (clojure.core/aget input (- i 1))
                                       (* -2.0 (clojure.core/aget input i))
                                       (clojure.core/aget input (+ i 1)))))])))
        equation-facts (assoc (dialect/default-equation-facts)
                              :effects #{:memory/write}
                              :aliases '{result du}
                              :attributes {:result-storage storage
                                           :host-binding 'result})
        facts (dialect/default-program-facts
               {:values {'n extent 'u double-tensor 'du double-tensor
                         'alpha scalar-double 'result double-tensor}
                :inputs '[alpha n u]
                :equations {'stencil-0 equation-facts}
                :effects #{:memory/write}})]
    (dialect/make facts [equation] '[result])))

(deftest stencil-domain-and-storage-contracts-are-validated
  (let [program (stencil-program)
        equation (first (dialect/equations program))
        [_ attributes arrays captures lambda] (nth equation 3)
        facts (dialect/facts program)]
    (is (= program (dialect/validate! program)))
    (is (= '[du] (dialect/physical-results program equation)))
    (is (= {:accumulators [] :elements [] :capture-parameters '[a input]}
           (dialect/parameter-layout equation)))
    (testing "radius and boundary policy are part of typed syntax"
      (doseq [bad-attributes [(assoc attributes :radius 0)
                              (assoc attributes :boundary :periodic)]]
        (try
          (dialect/make facts
                        [(list '= 'stencil-0 '[result]
                               (list 'stencil bad-attributes arrays captures lambda))]
                        '[result])
          (is false "unsupported stencil domains must fail syntax validation")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-soac-syntax (:reason (ex-data exception))))))))
    (testing "the materialized result dtype agrees with the stencil equation"
      (try
        (dialect/make
         (assoc-in facts [:values 'result] (av/tensor {:dtype :float :shape '[n]}))
         [equation] '[result])
        (is false "a result dtype mismatch must fail at the TypedSOAC boundary")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :typed-soac-stencil-result-type (:reason (ex-data exception)))))))
    (testing "shifted loads remain closed over explicit tensor captures"
      (let [bad-lambda (dialect/lambda-form
                        '[a input]
                        '[(+ a (clojure.core/aget missing (- i 1)))])]
        (try
          (dialect/make facts
                        [(list '= 'stencil-0 '[result]
                               (list 'stencil attributes arrays captures bad-lambda))]
                        '[result])
          (is false "an unbound neighborhood array must not enter a scalar region")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-soac-unbound-scalar (:reason (ex-data exception))))))))
    (testing "every tensor load is affine and inside the declared radius"
      (doseq [bad-index '[(+ i 2) (* i 1)]]
        (let [bad-lambda (dialect/lambda-form
                          '[a input]
                          [(list '+ 'a (list 'clojure.core/aget 'input bad-index))])]
          (try
            (dialect/make facts
                          [(list '= 'stencil-0 '[result]
                                 (list 'stencil attributes arrays captures bad-lambda))]
                          '[result])
            (is false "an unproved neighborhood index must not reach scheduling")
            (catch clojure.lang.ExceptionInfo exception
              (is (= :typed-soac-stencil-index (:reason (ex-data exception)))))))))
    (testing "value remapping preserves the tensor/storage boundary and rejects collisions"
      (let [remapped (dialect/remap-values
                      program {'u [:argument :u] 'du [:storage :du]
                               'result [:result :stencil]})
            remapped-equation (first (dialect/equations remapped))]
        (is (= [[:storage :du]]
               (dialect/physical-results remapped remapped-equation)))
        (is (= [[:argument :u]]
               (get-in (dialect/operation-parts remapped-equation)
                       [:attributes :attributes :stable-array-captures]))))
      (try
        (dialect/remap-values program {'u :same 'du :same})
        (is false "two physical identities cannot collapse during remapping")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :typed-soac-remap-collision (:reason (ex-data exception)))))))))

(deftest pure-scalar-shape-equations-are-ordered-typed-ssa
  (let [program
        (dialect/make
         (dialect/default-program-facts
          {:values {'x tensor 'n extent 'y tensor}
           :inputs '[x]
           :equations {'shape (dialect/default-equation-facts)
                       'map-0 (dialect/default-equation-facts)}})
         [(list '= 'shape '[n]
                (list 'scalar {:dtypes [:long]} '[x]
                      (dialect/lambda-form '[array] '[(clojure.core/alength array)])))
          (list '= 'map-0 '[y]
                (list 'map {:index 'i :extent 'n} '[x] []
                      (dialect/lambda-form '[element] '[(* element 2.0)])))]
         '[y])
        scalar-equation (first (dialect/equations program))]
    (is (= program (dialect/validate! program)))
    (is (= 'scalar (dialect/operation-kind scalar-equation)))
    (is (= '[x] (dialect/operation-inputs scalar-equation)))
    (is (nil? (dialect/operation-extent scalar-equation)))
    (is (= {:accumulators [] :elements [] :capture-parameters '[array]}
           (dialect/parameter-layout scalar-equation)))))

(deftest stable-array-capture-role-is-typed-and-validated
  (let [weights (av/tensor {:dtype :float :shape '[(unknown-dimension weights)]})
        program
        (dialect/make
         (dialect/default-program-facts
          {:values {'x tensor 'weights weights 'n extent 'y tensor}
           :inputs '[n weights x]
           :equations {'map-0 (dialect/default-equation-facts)}})
         [(list '= 'map-0 '[y]
                (list 'map {:index 'i :extent 'n
                            :attributes {:stable-array-captures '[weights]}}
                      '[x] '[weights]
                      (dialect/lambda-form
                       '[element weights-value]
                       '[(+ element (clojure.core/aget weights-value i))])))]
         '[y])
        remapped (dialect/remap-values program
                                       {'weights [:argument :weights]
                                        'x [:argument :x]
                                        'n [:shape :n]
                                        'y [:result :y]})
        remapped-equation (first (dialect/equations remapped))]
    (is (= program (dialect/validate! program)))
    (is (= [[:argument :weights]]
           (get-in (dialect/operation-parts remapped-equation)
                   [:attributes :attributes :stable-array-captures])))
    (is (= [(list 'unknown-dimension [:argument :weights])]
           (:shape (get-in (dialect/facts remapped)
                           [:values [:argument :weights]]))))
    (try
      (dialect/make
       (assoc-in (dialect/facts program) [:values 'weights]
                 (av/tensor {:dtype :float :shape []}))
       (dialect/equations program) (dialect/outputs program))
      (is false "a scalar value cannot satisfy a stable tensor capture role")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-stable-array-type (:reason (ex-data exception))))))))

(deftest scalar-regions-are-recursively-validated
  (testing "walker-devirtualized scalar calls remain explicit functional expressions"
    (let [program (map-program
                   {:body '(.invk raster.numeric/_star__m_float_float-impl
                                  x-element x-element)})]
      (is (= program (dialect/validate! program)))))

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
                     (dialect/lambda-form '[a-element] '[(* a-element 2.0)])))
         (list '= 'define-a '[a]
               (list 'map {:index 'i :extent 'n} '[x] []
                     (dialect/lambda-form '[x-element] '[(* x-element x-element)])))]]
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
                      (dialect/lambda-form '[x-element scale]
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
                      (dialect/lambda-form '[x-element] '[(* x-element 2.0)])))]
         '[y])]
    (is (= [(list 'value extent-id)] (:shape (get-in (dialect/facts program)
                                                     [:values 'x]))))))

(deftest alpha-remapping-preserves-static-extents-and-renames-value-facts
  (let [static-tensor (av/tensor {:dtype :float :shape [8]})
        program
        (dialect/make
         (dialect/default-program-facts
          {:values {'x static-tensor 'y static-tensor}
           :inputs '[x]
           :equations {'map-0 (dialect/default-equation-facts)}})
         [(list '= 'map-0 '[y]
                (list 'map {:index 'i :extent 8} '[x] []
                      (dialect/lambda-form '[x-element] '[(* x-element 2.0)])))]
         '[y])
        remapped (dialect/remap-values program {'x [:argument 0] 'y [:result 0]})
        equation (first (dialect/equations remapped))]
    (is (= [[:argument 0]] (:inputs (dialect/facts remapped))))
    (is (= [[:result 0]] (dialect/outputs remapped)))
    (is (= 8 (dialect/operation-extent equation)))
    (is (= [[:argument 0]] (dialect/operation-inputs equation)))
    (is (= [8] (:shape (get-in (dialect/facts remapped)
                               [:values [:result 0]]))))))
