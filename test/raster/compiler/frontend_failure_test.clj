(ns raster.compiler.frontend-failure-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.pipeline :as pipeline]))

(defn- caught [f]
  (try (f) nil (catch Throwable error error)))

(defn- source [raw?]
  (with-meta (fn [])
    (cond-> {:raster.core/deftm-params '[x] :raster.core/deftm-tags '[double]
             :raster.core/deftm-walked-body-typed [:stale]}
      raw? (assoc :raster.core/deftm-source-body '[x]))))

(deftest available-source-never-falls-back-after-a-walker-failure
  (doseq [marker [(ex-info "fatal" {:reason :raster/fatal})
                  (ex-info "bug" {:reason :raster/bug})
                  (NullPointerException. "walker implementation")
                  (ex-info "missing type" {:reason :missing-type})
                  (AssertionError. "walker invariant")]]
    (with-redefs [pipeline/resolve-deftm-var (fn [f & _] f)
                  inference/safe-tc-binding-tags (constantly nil)
                  walker/walk-body (fn [& _] (throw marker))]
      (is (identical? marker (caught #(pipeline/get-walked-body (source true))))))))

(deftest source-less-legacy-body-remains-a-separate-admission
  (with-redefs [pipeline/resolve-deftm-var (fn [f & _] f)
                walker/walk-body (fn [& _] (throw (AssertionError. "source-less body must not walk")))]
    (is (= [:stale] (pipeline/get-walked-body (source false))))))

(deftest compilation-uses-the-shared-optional-binding-tag-boundary
  (let [calls (atom [])]
    (with-redefs [inference/safe-tc-binding-tags
                  (fn [& args] (swap! calls conj args) {'y 'double})
                  walker/walk-body (fn [form opts]
                                     (is (= {'y 'double} (:tc-binding-tags opts)))
                                     (is (= :float (:element-dtype opts)))
                                     form)]
      (is (= '[x] (pipeline/walk-body-with-tc '[x] '[x] '[double] '[Double] *ns* :float)))
      (is (= 1 (count @calls))))))

(defn- analysis [entry]
  (case entry
    :analyze (inference/tc-analyze-deftm-body 'probe '[x] '[Double] '[x] *ns*)
    :safe (inference/safe-tc-binding-tags 'probe '[x] '[Double] '[x] *ns*)
    :infer (inference/tc-infer-binding-tags {'x 'double} 'x)))

(deftest compiler-invariants-escape-every-optional-tc-attempt
  (doseq [entry [:analyze :safe :infer]
          stage [:initialization :checked :retry]
          marker [(ex-info "fatal" {:reason :raster/fatal})
                  (ex-info "bug" {:reason :raster/bug})
                  (AssertionError. "checker invariant")]]
    (let [calls (atom 0)]
      (with-redefs-fn
        {#'inference/tc-check-form-info
         (fn [& _]
           (when (= :initialization stage) (throw marker))
           (fn [& _]
             (swap! calls inc)
             (if (and (= :retry stage) (= 1 @calls))
               (throw (Exception. "optional checked-AST incompatibility"))
               (throw marker))))}
        #(is (identical? marker (caught (fn [] (analysis entry))))))
      (is (= (case stage :initialization 0 :checked 1 :retry 2) @calls)))))

(deftest outer-binding-tag-wrapper-does-not-suppress-invariants
  (doseq [marker [(ex-info "fatal" {:reason :raster/fatal})
                  (ex-info "bug" {:reason :raster/bug})
                  (AssertionError. "analysis invariant")]]
    (with-redefs [inference/tc-analyze-deftm-body (fn [& _] (throw marker))]
      (is (identical? marker (caught #(analysis :safe)))))))

(deftest optional-tc-declines-and-retry-retain-their-contract
  (testing "fully untyped methods do not initialize the checker"
    (with-redefs-fn {#'inference/tc-check-form-info
                    (fn [& _] (throw (AssertionError. "untyped TC initialization")))}
      #(is (nil? (inference/tc-analyze-deftm-body 'untyped '[x] '[nil] '[x] *ns*)))))
  (testing "reported type errors never contribute partial binding tags"
    (with-redefs-fn
      {#'inference/tc-check-form-info (fn [& _] (fn [& _] {:type-errors [(ex-info "unsupported" {})]
                                                         :checked-ast {}}))
       #'inference/collect-binding-types-from-checked-ast
       (fn [& _] (throw (AssertionError. "unclean AST must not contribute tags")))}
      #(is (nil? (analysis :analyze)))))
  (testing "the existing checked-AST compatibility retry can still provide tags"
    (let [options (atom [])]
      (with-redefs-fn
        {#'inference/tc-check-form-info
         (fn [& _] (fn [_ opts]
                     (swap! options conj opts)
                     (if (:checked-ast opts)
                       (throw (Exception. "primitive tag compilation"))
                       {:ret {:t nil} :checked-ast {}})))
         #'inference/collect-binding-types-from-checked-ast (constantly {'y 'double})}
        #(is (= {'y 'double} (analysis :safe))))
      (is (= [{:checked-ast true :check-config {:check-form-eval :never}} {}] @options))))
  (testing "ordinary checker exceptions remain optional rather than misclassified as invariants"
    (with-redefs-fn
      {#'inference/tc-check-form-info (fn [& _] (throw (Exception. "optional checker unavailable")))}
      #(doseq [entry [:analyze :safe :infer]] (is (nil? (analysis entry))))))
  (testing "missing usable checked AST contributes no binding tags"
    (with-redefs-fn
      {#'inference/tc-check-form-info (fn [& _] (fn [& _] {:ret {:t nil}}))}
      #(is (= {} (analysis :safe)))))
  (testing "exhausted compatibility attempts still decline optional enrichment"
    (doseq [entry [:analyze :safe :infer]]
      (let [calls (atom 0)]
        (with-redefs-fn
          {#'inference/tc-check-form-info
           (fn [& _] (fn [& _] (swap! calls inc) (throw (Exception. "unsupported checker form"))))}
          #(binding [*out* (java.io.StringWriter.)]
             (is (nil? (analysis entry)))))
        (is (= 2 @calls))))))
