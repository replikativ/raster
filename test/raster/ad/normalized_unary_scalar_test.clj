(ns raster.ad.normalized-unary-scalar-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-map-element-test :as maps]
            [raster.ad.templates :as templates]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]
            [raster.core :refer [deftm]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.scalar.effects :as effects]))

(deftm direct-exp [x :- Double] :- Double (Math/exp x))
(deftm direct-log [x :- Double] :- Double (Math/log x))

(deftest direct-java-double-calls-reach-public-normalized-ad
  (doseq [[v x f derivative] [[#'direct-exp 0.75 #(Math/exp %) #(Math/exp %)]
                             [#'direct-log 2.0 #(Math/log %) #(/ 1.0 %)]]]
    (let [vg (reverse/value+grad v :representation :normalized-scalar)
          directional (jvp/jvp v :representation :normalized-scalar)
          [value gradient] (vg x)
          [jvalue tangent] (directional x 0.125)]
      (is (= (f x) value jvalue))
      (is (= (derivative x) gradient))
      (is (= (* 0.125 gradient) tangent)))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- scalar-program [dt inputs bindings]
  (let [ids (concat inputs (take-nth 2 bindings))
        values (zipmap ids (repeat (av/tensor {:dtype dt :shape []})))
        equations (mapv (fn [index [result expression]]
                          (frontend/scalar-equation
                           {:id index :sym (with-meta result {:raster.type/tag (if (= dt :float) 'float 'double)})
                            :expr expression} (zipmap ids (repeat dt)) {}))
                        (range) (partition 2 bindings))]
    (soac/make (soac/default-program-facts
                {:values values :inputs inputs
                 :equations (zipmap (range (count equations)) (repeat (soac/default-equation-facts)))})
               equations [(last (take-nth 2 bindings))])))

(deftest unary-maps-preserve-typed-carriers-snapshots-and-gradient-pairing
  (doseq [[dt array cast tolerance h] [[:double double-array double 1.0e-8 1.0e-5]
                                      [:float float-array float 8.0e-4 0.005]]
          [op xs derivative] [['raster.math/exp [-0.5 0.75] #(Math/exp %)]
                              ['raster.math/log [0.5 2.0] #(/ 1.0 %)]]]
    (let [source (#'maps/map-program dt {:body (list op 'element)})
          bundle (normalized/linearize-map-program source *ns*)
          prepared (runtime/prepare bundle)
          x (array xs) args (mapv {'n (long 2) 'x x} (:inputs prepared))
          seed (array [0.7 -1.2]) direction (array [0.125 -0.5])
          [y pb] ((:vjp prepared) args)
          [jy tangent] ((:jvp prepared) args direction)
          slot (.indexOf ^java.util.List (:inputs prepared) 'x)
          gradient (nth (pb seed) slot)
          f (if (= op 'raster.math/exp) #(Math/exp %) #(Math/log %))]
      (is (= (class x) (class y) (class tangent) (class gradient)))
      (is (= (vec y) (vec jy) (mapv #(cast (f (double (cast %)))) xs)))
      (doseq [i (range 2)]
        (let [xi (double (aget x i)) si (double (aget seed i))
              expected (* si (derivative xi))
              loss-at (fn [offset]
                        (let [changed (array x)]
                          (aset changed i (cast (+ xi offset)))
                          (let [[output _] ((:vjp prepared) (assoc args slot changed))]
                            (reduce + 0.0 (map * (seq output) (seq seed))))))
              finite-difference (/ (- (loss-at h) (loss-at (- h))) (* 2.0 h))]
          (is (< (Math/abs (- expected (double (aget gradient i)))) tolerance))
          (is (< (Math/abs (- finite-difference (double (aget gradient i)))) tolerance))))
      (is (< (Math/abs (- (reduce + 0.0 (map * (seq tangent) (seq seed)))
                         (reduce + 0.0 (map * (seq direction) (seq gradient))))) tolerance))
      (dotimes [i 2] (aset x i (cast 99.0)) (aset y i (cast -99.0)))
      (is (= (vec gradient) (vec (nth (pb seed) slot)))))))

(deftest nonlinear-tangents-refuse-and-frozen-coefficients-remain-legal
  (doseq [op '[raster.math/exp raster.math/log Math/exp Math/log]]
    (is (= :ad-normalized-nonlinear
           (reason #(normalized/validate-linear!
                     (scalar-program :double '[t] ['result (list op 't)]) '[t] []))))
    (let [p (scalar-program :double '[coefficient t]
                           ['k (list op 'coefficient) 'result '(raster.numeric/* k t)])]
      (is (= '#{t result} (normalized/validate-linear! p '[t] '[coefficient])))))
  (is (= :ad-normalized-nonlinear-output
         (reason #(normalized/validate-linear!
                   (scalar-program :double [] '[result (Math/exp 0.0)]) [] [])))))

(deftest unary-admission-never-invents-float-native-semantics-or-purity
  (doseq [op '[Math/exp Math/log]]
    (let [p (scalar-program :float '[x] ['result (list op 'x)])]
      (is (= :ad-normalized-call-contract
             (reason #(normalized/linearize-program
                       p {:params '[x] :tags '[float] :active-inputs '[x] :source-ns *ns*}))))))
  (with-redefs [effects/analyze-effect (constantly :io)]
    (is (= :ad-normalized-operation
           (reason #(normalized/linearize-map-program
                     (#'maps/map-program :double {:body '(raster.math/exp element)}) *ns*))))))

(deftest explicit-forward-alias-without-structure-does-not-borrow-base-permission
  (let [registry @(var-get #'templates/template-registry)
        isolated (atom (assoc registry 'raster.math/exp
                              {:params '[x] :jvp-fn (fn [& _] nil)}))]
    (with-redefs-fn {#'templates/template-registry isolated}
      #(is (= :ad-normalized-operation
              (reason (fn []
                        (normalized/linearize-map-program
                         (#'maps/map-program :double {:body '(raster.math/exp element)}) *ns*))))))))

(deftest unary-rule-replacement-invalidates-old-bundles-and-uses-new-rule
  (let [registry @(var-get #'templates/template-registry)
        isolated (atom registry :meta (meta (var-get #'templates/template-registry)))]
    (with-redefs-fn {#'templates/template-registry isolated}
      (fn []
        (let [source (#'maps/map-program :double {:body '(raster.math/exp element)})
              old (normalized/linearize-map-program source *ns*)
              base (templates/get-template 'Math/exp)
              original (:grads-fn base)
              calls (atom 0)]
          ;; Keep the mathematics unchanged, but replace the declared rule identity.
          ;; No mutation reaches the process's real template atom.
          (templates/register-template!
           'Math/exp (assoc (dissoc base :jvp-fn) :grads-fn
                            (fn [& args] (swap! calls inc) (apply original args))))
          (is (= :ad-normalized-stale-rules (reason #(normalized/validate! old))))
          (let [fresh (normalized/linearize-map-program source *ns*)
                prepared (runtime/prepare fresh)
                args (mapv {'n (long 2) 'x (double-array [-0.5 0.75])} (:inputs prepared))
                [_ pb] ((:vjp prepared) args)
                slot (.indexOf ^java.util.List (:inputs prepared) 'x)]
            (is (pos? @calls))
            (is (not= (:template-revision old) (:template-revision fresh)))
            (is (= (mapv #(Math/exp %) [-0.5 0.75])
                   (vec (nth (pb (double-array [1.0 1.0])) slot))))))))))
