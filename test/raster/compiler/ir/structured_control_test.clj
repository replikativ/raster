(ns raster.compiler.ir.structured-control-test
  (:require [clojure.test :refer [deftest is testing]]
            [pattern.nanopass.dialect :as pattern-dialect]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]))

(def ^:private tensor (av/tensor {:dtype :float :shape '[n]}))
(def ^:private inner-tensor (av/tensor {:dtype :float :shape '[n-in]}))
(def ^:private extent (av/tensor {:dtype :long :shape []}))
(def ^:private scalar (av/tensor {:dtype :float :shape []}))

(defn- fixture
  []
  (let [body-equation
        (list '= 'advance '[u-next]
              (list 'map {:index 'i :extent 'n-in}
                    '[u-in] '[alpha-in iteration]
                    (soac/lambda-form
                     '[u-value alpha-value iteration-value]
                     '[(if (>= iteration-value 0)
                         (+ u-value alpha-value)
                         u-value)])))
        body-facts
        (soac/default-program-facts
         {:values {'iteration extent 'n-in extent 'alpha-in scalar
                   'u-in inner-tensor 'u-next inner-tensor}
          :inputs '[iteration n-in alpha-in u-in]
          :equations {'advance (soac/default-equation-facts)}})
        body-program (soac/make body-facts [body-equation] '[u-next])
        outer-values {'steps extent 'n extent 'alpha scalar 'u0 tensor 'u-final tensor}
        loop-facts {:id 'time-loop :effects #{}
                    :provenance {:source :rk4-like}
                    :attributes {:association :sequential}}
        index '[iteration steps]
        invariants [{:outer 'n :parameter 'n-in}
                    {:outer 'alpha :parameter 'alpha-in}]
        carried [{:initial 'u0 :parameter 'u-in
                  :result 'u-next :output 'u-final}]
        program (control/make loop-facts index invariants carried body-program outer-values)]
    {:program program :body body-program :outer-values outer-values}))

(deftest structured-loop-is-a-pattern-declared-functional-fixpoint
  (let [{:keys [program outer-values]} (fixture)]
    (is (pattern-dialect/valid? control/TypedStructuredControl program))
    (is (= program (control/validate! program outer-values)))
    (is (= '[steps n alpha u0] (control/outer-operands program)))
    (is (= '[u-final] (control/outer-results program)))
    (is (= '[iteration n-in alpha-in u-in] (control/body-inputs program)))
    (is (= '[iteration n-in alpha-in u-in] (control/used-body-inputs program)))
    (is (= '[u-next] (control/body-results program)))
    (testing "zero trips retain the same typed initial/output contract"
      (let [zero-trip (assoc (vec program) 2 '[iteration 0])
            zero-trip (apply list zero-trip)]
        (is (= zero-trip (control/validate! zero-trip outer-values)))))))

(deftest structured-loop-boundaries-are-certified
  (let [{:keys [program outer-values]} (fixture)]
    (testing "body inputs are ordered, not inferred from a set"
      (let [bad-body (soac/make
                      (assoc (soac/facts (control/body program))
                             :inputs '[iteration alpha-in n-in u-in])
                      (soac/equations (control/body program))
                      (soac/outputs (control/body program)))
            bad (apply list (assoc (vec program) 5 bad-body))]
        (try
          (control/validate! bad outer-values)
          (is false "reordered body inputs must fail")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-loop-body-inputs (:reason (ex-data exception))))))))

    (testing "declared binders may be unused by a minimal TypedSOAC boundary"
      (let [body (control/body program)
            equation (first (soac/equations body))
            operation (nth equation 3)
            operation (assoc (vec operation) 3 '[alpha-in])
            operation (apply list operation)
            lambda (soac/lambda-form '[u-value alpha-value] '[(+ u-value alpha-value)])
            operation (apply list (assoc (vec operation) 4 lambda))
            equation (apply list (assoc (vec equation) 3 operation))
            facts (assoc (soac/facts body) :inputs '[n-in alpha-in u-in])
            unused-iteration-body (soac/make facts [equation] (soac/outputs body))
            unused-iteration (apply list (assoc (vec program) 5 unused-iteration-body))]
        (is (= '[n-in alpha-in u-in] (control/used-body-inputs unused-iteration)))
        (is (= unused-iteration (control/validate! unused-iteration outer-values)))))

    (testing "loop-carried values retain one AbstractValue contract"
      (let [bad-values (assoc outer-values 'u-final
                              (av/tensor {:dtype :double :shape '[n]}))]
        (try
          (control/validate! program bad-values)
          (is false "a changed carry dtype must fail")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-loop-value-mismatch (:reason (ex-data exception))))))))

    (testing "a symbolic trip count is an integer scalar"
      (let [bad-values (assoc outer-values 'steps scalar)]
        (try
          (control/validate! program bad-values)
          (is false "floating trip counts must fail")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-loop-trip-count-type (:reason (ex-data exception))))))))

    (testing "effects cannot be hidden by the control wrapper"
      (let [bad (apply list (assoc (vec program) 1
                                   (assoc (control/facts program) :effects #{:io})))]
        (try
          (control/validate! bad outer-values)
          (is false "loop effects must equal body effects")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-loop-effects (:reason (ex-data exception))))))))

    (testing "association is explicitly sequential"
      (let [bad (apply list (assoc (vec program) 1
                                   (assoc-in (control/facts program)
                                             [:attributes :association]
                                             :parallel)))]
        (try
          (control/validate! bad outer-values)
          (is false "structured time loops cannot silently reassociate")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-loop-syntax (:reason (ex-data exception))))))))))
