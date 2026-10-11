(ns raster.ad.normalized-map-lane-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-surface-map-test :as surface]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]))

(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- typed-output [source equation output]
  (with-meta output (assoc (meta output) :raster.type/tag
                          (dtype/scalar-tag-for-dtype
                           (get-in (soac/facts source) [:values (first (nth equation 2)) :dtype])))))
(defn- same-projection? [left right]
  (and (identical? (get-in left [:interface :source-ns]) (get-in right [:interface :source-ns]))
       (fingerprint/equivalent? (update left :interface dissoc :source-ns)
                                (update right :interface dissoc :source-ns))))
(defn- lane [source equation output]
  (#'normalized/project-map-equation-lane source equation
                                            (the-ns 'raster.ad.normalized-map-lane-test)
                                            (typed-output source equation output) {}))
(defn- wrapper [source output]
  (#'normalized/project-map-element source (the-ns 'raster.ad.normalized-map-lane-test)
                                   (typed-output source (first (soac/equations source)) output) {}))

(defn- two-stage [dt]
  (let [values {'x (av/tensor {:dtype dt :shape '[n]})
                'u (av/tensor {:dtype dt :shape '[n] :ownership :owned})
                'y (av/tensor {:dtype dt :shape '[n] :ownership :owned})
                'scale (av/tensor {:dtype dt :shape []})
                'n (av/tensor {:dtype :long :shape []})}
        first-equation (list '= 'first '[u]
                             (list 'map {:index 'i :extent 'n} '[x] '[scale]
                                   (soac/lambda-form '[element scale] '[(raster.numeric/* element scale)])))
        second-equation (list '= 'second '[y]
                              (list 'map {:index 'j :extent 'n} '[u] '[scale]
                                    (soac/lambda-form '[element scale]
                                                      [(list 'raster.numeric/+ 'element 'scale)])))
        equation-facts (fn [index]
                         (assoc (soac/default-equation-facts
                                 {:front-end :analyzed-source :source-binding-id index})
                                :attributes {:source-prefix? (zero? index)}))]
    (soac/make
     (soac/default-program-facts
      {:values values :inputs '[x scale n]
       :equations {'first (equation-facts 0) 'second (equation-facts 1)}
       :attributes {:source-arithmetic numerics/retained-source-arithmetic
                    :source-dialect :closed-clojure :source-bindings '[u y]
                    :host-binding-ids [] :host-read-values [] :host-read-sites [] :allocations []}})
     [first-equation second-equation] '[y])))

(deftest exact-member-lane-preserves-one-map-projection-and-certificate
  (doseq [dt [:float :double]]
    (let [source (#'element/map-program dt {:captures '[scale]
                                           :locals [(soac/local-value 'v dt '(raster.numeric/* element scale))]
                                           :body '(raster.numeric/* v v)})
          equation (first (soac/equations source))
          output (with-meta 'lane-output {:raster.type/tag (if (= dt :float) 'float 'double)})
          projected (lane source equation output) old (wrapper source output)
          bundle (normalized/linearize-program (:primal old) (:interface old))]
      (is (identical? source (:source-program projected)))
      (is (identical? equation (:source-equation projected)))
      (is (same-projection? old (select-keys projected [:primal :interface])))
      (is (= [output] (soac/outputs (:primal projected))))
      (is (= dt (get-in (soac/facts (:primal projected)) [:values output :dtype])))
      (is (= (:certificate bundle) (#'normalized/certificate (assoc bundle :primal (:primal projected)))))
      (is (identical? bundle (normalized/validate! bundle))))))

(deftest original-equation-stages-and-frontend-bookkeeping-are-not-fused-or-erased
  (doseq [dt [:float :double]]
    (let [source (two-stage dt)
          equations (soac/equations source)
          projections (soac/equation-subprograms source)]
      (is (= :ad-normalized-map-boundary (reason #(wrapper source 'output))))
      (doseq [[equation projected] (map vector equations projections)]
        (let [result (lane source equation 'output)
              shared (#'normalized/project-checked-map-lane
                       source projected equation (the-ns 'raster.ad.normalized-map-lane-test)
                       (typed-output source equation 'output) {})]
          (is (identical? equation (:source-equation result)))
          (is (identical? source (:source-program result)))
          (is (same-projection? result shared))
          (is (= '[element scale] (mapv #(with-meta % nil) (get-in result [:interface :params]))))
          (is (= [dt] (mapv #(get-in (soac/facts (:primal result)) [:values % :dtype])
                            (soac/outputs (:primal result)))))))
      (let [facts (soac/facts source)
            malformed (apply list (assoc (vec source) 1 (assoc-in facts [:equations 'second :attributes :source-prefix?] true)))
            effectful (apply list (assoc (vec source) 1
                                        (-> facts (assoc :effects #{:memory/write})
                                            (assoc-in [:equations 'second :effects] #{:memory/write}))))]
        (is (= :ad-normalized-map-boundary (reason #(lane malformed (first equations) 'output))))
        (is (= :ad-normalized-map-boundary (reason #(lane effectful (first equations) 'output))))))))

(deftest lane-preserves-selected-call-metadata-and-refuses-narrowing-or-lexical-dispatch
  (doseq [[v dt] [[#'surface/square-float :float] [#'surface/square-double :double]]]
    (let [source (#'surface/source-program v dt)
          equation (first (soac/equations source))
          projected (lane source equation 'output)]
      (is (same-projection? (wrapper source 'output) (select-keys projected [:primal :interface])))
      (is (identical? equation (:source-equation projected)))
      (let [receivers (fn [form] (mapv second (filter #(and (seq? %) (= '.invk (first %)))
                                                    (tree-seq coll? seq form))))]
        (is (seq (receivers equation)))
        (is (fingerprint/equivalent? (receivers equation) (receivers (:primal projected)))))
      (is (= dt (get-in (soac/facts (:primal projected)) [:values 'output :dtype])))))
  (let [source (#'element/map-program :float
                                    {:body (list 'raster.numeric/+ 'element (float 1.0))})
        projected (lane source (first (soac/equations source)) 'output)
        primal (:primal projected)
        region (#'normalized/scalar-program-lambda primal (get-in projected [:interface :params])
                                                    (soac/outputs primal))
        [result] (scalar/evaluate-region (the-ns 'raster.ad.normalized-map-lane-test)
                                        region [{:type :float :value (float 16777216.0)}] [:float])]
    ;; Independent Float checkpoint: widening the lane addition would produce 16777217.
    (is (instance? Float (:value result)))
    (is (= (Float/floatToRawIntBits (float 16777216.0))
           (Float/floatToRawIntBits (:value result)))))
  (doseq [[options expected]
          [[{:locals [(soac/local-value 'v :double (float 1.0))] :body 'v} :ad-normalized-unknown-type]
           [{:captures '[double] :body '(double element)} :ad-normalized-callee]
           [{:locals [(soac/local-value '* :double 2.0)] :body '(* element element)} :ad-normalized-callee]]]
    (let [source (#'element/map-program :double options) equation (first (soac/equations source))]
      (is (= expected (reason #(wrapper source 'output))))
      (is (= expected (reason #(lane source equation 'output)))))))

(deftest single-map-wrapper-keeps-source-validation-count-and-exact-member-check
  (let [source (#'element/map-program :double) equation (first (soac/equations source))
        calls (atom 0) original soac/validate!]
    (with-redefs [soac/validate! (fn [p] (when (identical? p source) (swap! calls inc)) (original p))]
      (wrapper source 'output))
    (is (= 1 @calls))
    (is (= :typed-soac-equation-projection
           (reason #(lane source (with-meta (assoc (vec equation) 1 'not-a-member) nil) 'output))))))
