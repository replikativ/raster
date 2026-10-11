(ns raster.ad.normalized-scalar-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-scalar-public-test :as public]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]
            [raster.ad.templates :as templates]
            [raster.core :refer [deftm]]
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

(deftest linearity-requires-zero-preservation-not-just-dependence
  (doseq [[expression inputs tangents coefficients expected]
          [['(raster.numeric/* t t) '[t] '[t] [] :ad-normalized-nonlinear]
           ['(raster.numeric/+ t coefficient) '[t coefficient] '[t] '[coefficient]
            :ad-normalized-affine-offset]
           [2.0 [] [] [] :ad-normalized-nonlinear-output]]]
    (is (= expected (reason #(normalized/validate-linear!
                             (linear-program expression inputs) tangents coefficients)))))
  (let [program (linear-program '(raster.numeric/* coefficient t) '[coefficient t])]
    (is (= '#{t result} (normalized/validate-linear! program '[t] '[coefficient]))))
  (is (= :ad-normalized-unknown-type
         (reason #(normalized/validate-linear!
                   (linear-program (with-meta '(raster.numeric/* coefficient t)
                                     {:raster.type/tag 'float}) '[coefficient t])
                   '[t] '[coefficient])))))

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
