(ns raster.ad.normalized-map-element-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.core :as core]
            [raster.numeric :as n]))

(defn- map-program
  ([dt] (map-program dt {}))
  ([dt {:keys [extent locals body captures values equation-facts attributes]
        :or {extent 'n
             locals [] body '(raster.numeric/* element element)
             captures [] equation-facts (soac/default-equation-facts) attributes {}}}]
   (let [tensor (av/tensor {:dtype dt :shape [extent]})
         scalar (av/tensor {:dtype dt :shape []})
         values (merge {'x tensor 'y tensor 'n (av/tensor {:dtype :long :shape []})}
                       (zipmap captures (repeat scalar)) values)
         eq (list '= 'map-0 '[y]
                  (list 'map (merge {:index 'i :extent extent} attributes) '[x] captures
                        (soac/lambda-form (into '[element] captures) locals [body])))]
     (soac/make (soac/default-program-facts
                 {:values values :inputs (into (if (symbol? extent) '[n x] '[x]) captures)
                  :effects (:effects equation-facts)
                  :equations {'map-0 equation-facts}}) [eq] '[y]))))

(defn- emit [description]
  (binding [*ns* (:source-ns description)]
    (eval (list 'fn (:params description) (first (:walked-body description))))))

(defn- bundle [program]
  (normalized/linearize-map-element program (the-ns 'raster.ad.normalized-map-element-test)))

(defn- reason [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(def receiver nil)

(deftest map-element-reuses-scalar-program-entry-without-source-reanalysis
  (let [primal (map-program :double)
        result (with-redefs [core/ensure-walked-body!
                             (fn [& _] (throw (ex-info "source walk is forbidden" {})))
                             inference/qualify-body-symbols
                             (fn [& _] (throw (ex-info "source qualification is forbidden" {})))]
                 (bundle primal))
        vg (emit (normalized/value+grad-body result))
        jvp (emit (normalized/jvp-body result))]
    (is (identical? result (normalized/validate! result)))
    (is (= '[element] (:params result) (:active-inputs result)))
    (is (every? #(= [] (:shape %)) (vals (:values (soac/facts (:primal result))))))
    (doseq [x [-2.0 0.0 3.0]]
      (is (= [(* x x) (* 2.0 x)] (vg x)))
      (is (= [(* x x) (* 2.0 x -0.25)] (jvp x -0.25)))
      (let [h 1.0e-5
            fd (/ (- (* (+ x h) (+ x h)) (* (- x h) (- x h))) (* 2.0 h))]
        (is (< (Math/abs (- fd (second (vg x)))) 1.0e-9))))))

(deftest map-element-preserves-ordered-typed-locals-and-inactive-captures
  (let [locals [(soac/local-value 'square :double '(raster.numeric/* element element))
                (soac/local-value 'scaled :double '(raster.numeric/* square scale))
                (soac/local-value 'shifted :double '(raster.numeric/+ scaled bias))]
        result (bundle (map-program :double {:locals locals :captures '[scale bias]
                                             :body 'shifted}))
        vg (emit (normalized/value+grad-body result))
        jvp (emit (normalized/jvp-body result))]
    (is (= '[element scale bias] (:params result)))
    (is (= '[element] (:active-inputs result)))
    (is (= [17.0 12.0 nil nil] (vg 2.0 3.0 5.0)))
    (is (= [17.0 -3.0] (jvp 2.0 3.0 5.0 -0.25)))
    (is (seq (:residuals result)))
    (let [empty-bundle (bundle (map-program :double {:extent 0 :locals locals
                                                     :captures '[scale bias] :body 'shifted}))]
      (is (identical? empty-bundle (normalized/validate! empty-bundle))
          "zero extent does not change the scalar region's admission"))))

(deftest homogeneous-float-map-region-retains-rounding-space
  (let [result (bundle (map-program :float))
        vg (emit (normalized/value+grad-body result))
        jvp (emit (normalized/jvp-body result))
        primal-and-gradient (vg (float 3.0))
        primal-and-tangent (jvp (float 3.0) (float 0.25))]
    (is (= [(float 9.0) (float 6.0)] primal-and-gradient))
    (is (= [(float 9.0) (float 1.5)] primal-and-tangent))
    (is (every? #(instance? Float %) (concat primal-and-gradient primal-and-tangent)))))

(deftest map-element-transpose-accepts-an-explicit-cotangent-seed
  (let [result (bundle (map-program :double))
        jvp (emit (normalized/jvp-body result))
        transpose (emit {:source-ns (:source-ns result)
                         :params (conj (:residuals result) (:cotangent-seed result))
                         :walked-body [(list 'let* (#'normalized/bindings (:transpose result))
                                             (soac/outputs (:transpose result)))]})]
    (is (= '[element] (:residuals result)))
    (doseq [[x dx dy] [[3.0 -0.5 0.25] [-2.0 1.5 -3.0] [0.0 4.0 2.0]]]
      (let [[_ pushed] (jvp x dx)
            [pulled] (transpose x dy)]
        (is (= (* 2.0 x dy) pulled))
        (is (= (* pushed dy) (* dx pulled)))))))

(deftest frontend-map-facets-and-defining-namespace-are-preserved
  (let [selected (dispatch/selected-call-signature
                  'raster.numeric/* '[double double]
                  (inference/registered-call-selection 'raster.numeric/* '[double double]))
        expression (with-meta '(.invk raster.ad.normalized-map-element-test/receiver element element)
                     {:raster.op/original 'raster.numeric/* :raster.type/tag 'double})
        program (map-program :double
                             {:attributes {:attributes {:stable-array-captures []
                                                        :pointwise-storage-inputs []}}
                              :body expression})]
    (with-redefs [receiver (:typed-impl selected)]
      (binding [*ns* (the-ns 'user)]
        (let [result (bundle program)
              vg (emit (normalized/value+grad-body result))]
          (is (= [9.0 6.0] (vg 3.0)))
          (is (= (the-ns 'raster.ad.normalized-map-element-test) (:source-ns result)))))))
  (doseq [[facets expected] [[{:pointwise-storage-inputs ['x]}
                              :typed-soac-pointwise-storage-inputs]
                             [{:unknown-semantic-facet []} :ad-normalized-map-boundary]]]
    (is (= expected
           (reason #(bundle (map-program :double {:attributes {:attributes facets}})))))))

(deftest map-element-refuses-storage-and-type-contract-drift
  (doseq [values [{'x (av/tensor {:dtype :double :shape '[n]
                                  :representation {:kind :quantized :scheme :q4-k}})}
                  {'x (av/tensor {:dtype :double :shape '[n]
                                  :logical-layout {:kind :strided}})}
                  {'y (av/tensor {:dtype :float :shape '[n]})}
                  {'scale (av/tensor {:dtype :float :shape []})}
                  {'scale (av/tensor {:dtype :double :shape '[n]})}]]
    (is (= :ad-normalized-map-space
           (reason #(bundle (map-program :double {:values values :captures '[scale]}))))))
  (is (= :ad-normalized-map-space
         (reason #(bundle (map-program :double
                                       {:locals [(soac/local-value 'v :float 1.0)]
                                        :body 'v})))))
  (is (= :ad-normalized-map-space
         (reason #(bundle (map-program :double
                                       {:locals [(soac/local-value
                                                  (with-meta 'v {:raster.type/tag 'float})
                                                  :double 1.0)] :body 'v}))))))

(deftest map-element-does-not-grant-array-effects-or-index-differentiation
  (let [pure (map-program :double)
        effectful (map-program :double
                               {:equation-facts (assoc (soac/default-equation-facts)
                                                       :effects #{:memory/write})})]
    (is (= :ad-normalized-map-boundary (reason #(bundle effectful))))
    (is (= :ad-normalized-map-boundary
           (reason #(normalized/linearize-map-element pure nil)))))
  (is (some? (reason #(bundle (map-program :double {:body '(raster.numeric/+ element i)})))))
  (is (some? (reason #(bundle (map-program :double {:body '(clojure.core/aget x i)})))))
  (is (= :ad-normalized-operation
         (reason #(bundle (map-program :double {:body '(Math/pow element element)})))))
  (is (some? (reason #(bundle (map-program :double {:body '(float element)}))))))

(deftest map-element-does-not-drop-numerical-policy-or-storage-evidence
  (let [pure (map-program :double)
        refact (fn [facts] (soac/make facts (soac/equations pure) (soac/outputs pure)))]
    (doseq [facts [(assoc-in (soac/facts pure) [:attributes :scalar-math]
                             {:mode :bounded-error})
                   (assoc-in (soac/facts pure) [:equations 'map-0 :attributes :scalar-math]
                             {:mode :bounded-error})]]
      (is (= :ad-normalized-map-boundary (reason #(bundle (refact facts))))))
    (is (some? (reason #(bundle (map-program :double
                                             {:equation-facts
                                              (assoc (soac/default-equation-facts)
                                                     :aliases {'y 'x})})))))
    (is (some? (reason #(bundle (map-program :double
                                             {:attributes {:attributes
                                                           {:stable-array-captures ['scale]}}
                                              :captures '[scale]})))))
    (is (some? (reason #(bundle (map-program :double
                                             {:locals [(soac/local-value 'v :double (float 1.0))]
                                              :body 'v}))))
        "local projection may not synthesize exact widening to satisfy its declaration")
    (is (some? (reason #(bundle (map-program :double
                                             {:locals [(soac/local-value 'v :unknown 1.0)]
                                              :body 'v})))))
    (is (= :ad-normalized-map-boundary
           (reason #(bundle (soac/make (soac/default-program-facts {}) [] [])))))))

(deftest scalar-and-map-entry-never-dispatch-a-lexical-value-as-a-global-intrinsic
  (is (= :ad-normalized-callee
         (reason #(bundle (map-program :double {:captures '[double]
                                                :body '(double element)})))))
  (is (= :ad-normalized-callee
         (reason #(bundle (map-program :double
                                        {:locals [(soac/local-value '* :double 2.0)]
                                         :body '(* element element)})))))
  (let [scalar (av/tensor {:dtype :double :shape []})
        equation (list '= 'result '[output]
                       (list 'scalar {:dtypes [:double]} '[element double]
                             (soac/lambda-form '[e d] '[(d e)])))
        program (soac/make (soac/default-program-facts
                           {:values {'element scalar 'double scalar 'output scalar}
                            :inputs '[element double]
                            :equations {'result (soac/default-equation-facts)}})
                           [equation] '[output])]
    (is (= :ad-normalized-callee
           (reason #(normalized/linearize-program
                     program {:params '[element double] :tags '[double double]
                              :active-inputs '[element]
                              :source-ns (the-ns 'raster.ad.normalized-map-element-test)})))))
  (let [result (bundle (map-program :double {:captures '[double]
                                            :body '(clojure.core/double element)}))
        vg (emit (normalized/value+grad-body result))]
    (is (= [3.0 1.0 nil] (vg 3.0 7.0))
        "a qualified core cast is not shadowed by a scalar data capture")))
