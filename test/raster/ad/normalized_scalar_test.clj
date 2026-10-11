(ns raster.ad.normalized-scalar-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-scalar-public-test :as public]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]
            [raster.ad.templates :as templates]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defn- reason [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))

(defn- linear-program [expression inputs]
  (let [result (with-meta 'result {:raster.type/tag 'double})
        value (av/tensor {:dtype :double :shape []})
        equation (frontend/scalar-equation {:id :equation :sym result :expr expression}
                                           (zipmap inputs (repeat :double)) {})]
    (soac/make
     (soac/default-program-facts
      {:values (assoc (zipmap inputs (repeat value)) result value)
       :inputs inputs :equations {:equation (soac/default-equation-facts)}})
     [equation] [result])))

(deftest checked-program-entry-does-not-revisit-source
  (let [primal (linear-program '(raster.numeric/* coefficient t) '[coefficient t])
        interface {:params '[coefficient t] :tags '[double double] :active-inputs '[t]
                   :source-ns (the-ns 'raster.ad.normalized-scalar-test)}
        bundle (with-redefs [core/ensure-walked-body!
                            (fn [& _] (throw (ex-info "unexpected source walk" {})))
                            inference/qualify-body-symbols
                            (fn [& _] (throw (ex-info "unexpected source qualification" {})))]
                 (normalized/linearize-program primal interface))
        emit (fn [body]
               (binding [*ns* (:source-ns body)]
                 (eval (list 'fn (:params body) (first (:walked-body body))))))
        vg (emit (normalized/value+grad-body bundle))
        directional (emit (normalized/jvp-body bundle))]
    (is (identical? primal (:primal bundle)))
    (is (identical? bundle (normalized/validate! bundle)))
    (is (= [6.0 nil 3.0] (vg 3.0 2.0)))
    (is (= [6.0 1.5] (directional 3.0 2.0 0.5)))
    (doseq [invalid [(assoc interface :tags '[float float])
                     (assoc interface :params '[coefficient coefficient])
                     (assoc interface :params '[coefficient t result] :tags '[double double double])
                     (assoc interface :active-inputs '[t t])
                     (assoc interface :active-inputs '[missing])
                     (assoc interface :source-ns nil)]]
      (is (= :ad-normalized-program-interface
             (reason #(normalized/linearize-program primal invalid)))))))

(deftest program-entry-keeps-unused-inputs-but-never-reinterprets-other-operations
  (let [primal (linear-program '(raster.numeric/* coefficient t) '[coefficient t])
        facts (assoc-in (soac/facts primal) [:values 'unused] (av/tensor {:dtype :double :shape []}))
        program (soac/make facts (soac/equations primal) (soac/outputs primal))
        interface {:params '[coefficient t unused] :tags '[double double double]
                   :active-inputs '[unused] :source-ns *ns*}]
    (is (= '[unused] (:active-inputs (normalized/linearize-program program interface))))
    (let [equation (first (soac/equations primal))
          operation (nth equation 3)]
      (doseq [kind '[reduce map scan]]
        (let [changed (apply list (assoc (vec operation) 0 kind))
              changed-equation (apply list (assoc (vec equation) 3 changed))]
          (is (= :ad-normalized-region
                 (reason #(#'normalized/equation-expression changed-equation)))))))))

(deftest certificate-projects-only-the-declared-runtime-field
  (let [bundle (normalized/linearize-program
                (linear-program '(raster.numeric/* coefficient t) '[coefficient t])
                {:params '[coefficient t] :tags '[double double] :active-inputs '[t]
                 :source-ns (the-ns 'raster.ad.normalized-scalar-test)})
        extra (assoc bundle :source-namespace 'caller.evidence)
        certified (assoc extra :certificate (#'normalized/certificate extra))]
    (is (identical? certified (normalized/validate! certified)))
    (is (= :ad-normalized-certificate
           (reason #(normalized/validate! (assoc certified :source-namespace 'changed.evidence)))))
    (is (= :ad-normalized-certificate
           (reason #(normalized/validate! (dissoc certified :source-namespace)))))
    (is (= :semantic-fingerprint-unsupported
           (reason #(#'normalized/certificate (assoc bundle :source-namespace (Object.))))))
    (is (= :ad-normalized-bundle-interface
           (reason #(normalized/validate! (assoc bundle :source-ns 'not.a.namespace)))))))

(deftest linearity-requires-zero-preservation-not-just-dependence
  (doseq [[expression inputs tangents coefficients expected]
          [['(raster.numeric/* t t) '[t] '[t] [] :ad-normalized-nonlinear]
           ['(raster.numeric// coefficient t) '[coefficient t] '[t] '[coefficient]
            :ad-normalized-nonlinear]
           ['(raster.numeric// t t) '[t] '[t] [] :ad-normalized-nonlinear]
           ['(raster.numeric/+ t coefficient) '[t coefficient] '[t] '[coefficient]
            :ad-normalized-affine-offset]
           [2.0 [] [] [] :ad-normalized-nonlinear-output]]]
    (is (= expected (reason #(normalized/validate-linear!
                             (linear-program expression inputs) tangents coefficients)))))
  (let [program (linear-program '(raster.numeric/* coefficient t) '[coefficient t])]
    (is (= '#{t result} (normalized/validate-linear! program '[t] '[coefficient]))))
  (let [program (linear-program '(raster.numeric// t coefficient) '[t coefficient])]
    (is (= '#{t result} (normalized/validate-linear! program '[t] '[coefficient]))))
  (is (= :ad-normalized-unknown-type
         (reason #(normalized/validate-linear!
                   (linear-program (with-meta '(raster.numeric/* coefficient t)
                                     {:raster.type/tag 'float}) '[coefficient t])
                   '[t] '[coefficient])))))

(def receiver-alias @#'raster.numeric/_star__m_double_double-impl)

(deftest retained-calls-require-selected-formals-result-and-implementation
  (let [expression (with-meta
                     '(.invk raster.numeric/_star__m_double_double-impl coefficient t)
                     {:raster.op/original 'raster.numeric/* :raster.type/tag 'double})
        program (linear-program expression '[coefficient t])
        selected (dispatch/selected-call-signature
                  'raster.numeric/* '[double double]
                  (inference/registered-call-selection 'raster.numeric/* '[double double]))]
    (is (= '#{t result} (normalized/validate-linear! program '[t] '[coefficient])))
    (is (= '#{t result}
           (normalized/validate-linear!
            (linear-program '(raster.numeric/* coefficient t) '[coefficient t])
            '[t] '[coefficient])))
    (let [aliased-expression (with-meta
                               '(.invk raster.ad.normalized-scalar-test/receiver-alias coefficient t)
                               (meta expression))
          aliased-program (linear-program aliased-expression '[coefficient t])]
      ;; Other namespaces may refresh registered methods after this namespace loads.
      ;; The alias positive refers to the current selection, not its load-time snapshot.
      (with-redefs [receiver-alias (:typed-impl selected)]
        (is (= '#{t result}
               (normalized/validate-linear! aliased-program '[t] '[coefficient])))
        (with-redefs [receiver-alias (Object.)]
          (is (= :ad-normalized-call-contract
                 (reason #(normalized/validate-linear! aliased-program '[t] '[coefficient])))))))
    (doseq [contradictory [nil
                          (assoc selected :tags '[float float])
                          (assoc selected :return-tag 'float)
                          (assoc selected :promotion-casts '[float nil])
                          (assoc selected :selection :parametric)
                          (assoc selected :typed-impl nil)]]
      (with-redefs [dispatch/selected-call-signature (constantly contradictory)]
        (is (= :ad-normalized-call-contract
               (reason #(normalized/validate-linear! program '[t] '[coefficient]))))))
    ;; Semantic metadata cannot bless a different executable receiver.
    (let [wrong-receiver (with-meta
                           '(.invk raster.numeric/_star__m_float_float-impl coefficient t)
                           (meta expression))]
      (is (= :ad-normalized-call-contract
             (reason #(normalized/validate-linear!
                       (linear-program wrong-receiver '[coefficient t])
                       '[t] '[coefficient])))))))

(deftest bundle-rejects-stale-residuals-and-semantic-metadata
  (let [bundle (:raster.ad/normalized-bundle
                (meta (reverse/value+grad #'public/square :representation :normalized-scalar)))
        linear (:linear bundle)
        facts (soac/facts linear)
        [eq & more] (soac/equations linear)
        {:keys [lambda]} (soac/operation-parts eq)
        [_ _ [_ _ [expression]]] lambda
        changed (vary-meta expression assoc :raster.op/original 'user/unknown-op)
        changed-eq (assoc (vec eq) 3
                         (assoc (vec (nth eq 3)) 3
                                (soac/lambda-form (:parameters (soac/lambda-parts lambda)) [changed])))
        changed-linear (soac/make facts (into [(apply list changed-eq)] more) (soac/outputs linear))]
    (is (identical? bundle (normalized/validate! bundle)))
    (is (= :ad-normalized-linear-interface
           (reason #(normalized/jvp-body (assoc bundle :residuals [])))))
    (is (= :ad-normalized-certificate
           (reason #(normalized/jvp-body (assoc bundle :linear changed-linear)))))
    (is (= :ad-normalized-stale-rules
           (reason #(normalized/value+grad-body (update bundle :template-revision dec)))))
    (is (= :ad-normalized-certificate
           (reason #(normalized/jvp-body (assoc bundle :certificate "changed")))))
    (let [revision (templates/registry-revision)]
      (with-redefs [templates/registry-revision (constantly (inc revision))]
        (is (= :ad-normalized-stale-rules (reason #(normalized/jvp-body bundle))))))))

(deftm constant-result [x :- Double] :- Double 2.0)
(deftm identity-result [x :- Double] :- Double x)

(deftest typed-zero-excluded-slots-and-exceptional-arithmetic
  (is (= [2.0 0.0] ((reverse/value+grad #'constant-result :representation :normalized-scalar) 3.0)))
  (is (= [2.0 0.0] ((jvp/jvp #'constant-result :representation :normalized-scalar) 3.0 1.0)))
  (is (= [9.0 nil] ((reverse/value+grad #'public/square :representation :normalized-scalar :wrt []) 3.0)))
  (is (= [9.0 0.0] ((jvp/jvp #'public/square :representation :normalized-scalar :wrt []) 3.0)))
  (let [[value tangent] ((jvp/jvp #'public/square :representation :normalized-scalar)
                        Double/POSITIVE_INFINITY 0.0)]
    (is (= Double/POSITIVE_INFINITY value))
    (is (Double/isNaN tangent) "an explicit zero seed does not activate StrongZero"))
  (let [[value gradient] ((reverse/value+grad #'identity-result :representation :normalized-scalar) -0.0)]
    (is (= (Double/doubleToRawLongBits -0.0) (Double/doubleToRawLongBits value)))
    (is (= 1.0 gradient))))
