(ns raster.ad.normalized-scalar-runtime-guard-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-scalar-public-test :as public]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]
            [raster.ad.templates :as templates]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.numeric :as n]))

(deftest revision-snapshots-follow-semantic-dependency-initialization
  (let [calls (atom [])
        initialized? (atom false)]
    (with-redefs [effects/init-raster-context!
                  (fn [] (swap! calls conj :initialize) (reset! initialized? true))
                  templates/registry-revision
                  (fn [] (is @initialized?) (swap! calls conj :templates) 42)
                  dispatch/compiler-definition-revision
                  (fn [] (is @initialized?) (swap! calls conj :compiler) 73)]
      (is (= {:template-revision 42 :compiler-revision 73} (normalized/revisions)))
      (is (= [:initialize :templates :compiler] @calls)))))

(def ^:dynamic receiver nil)
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest template-arguments-retain-the-owning-programs-declarations
  (let [value (av/tensor {:dtype :double :shape []})
        reference (with-meta 'x {:tag 'Object})
        call (#'normalized/call! {'x value} (list 'raster.numeric/* reference reference))]
    (is (= '[double double] (mapv #(-> % meta :raster.type/tag) (:arguments call))))
    (is (= '[Object Object] (mapv #(-> % meta :tag) (:arguments call))))
    (is (= {:tag 'Object} (meta reference)))
    (is (= :ad-normalized-value-declaration
           (reason #(#'normalized/call!
                     {'x value}
                     (list 'raster.numeric/*
                           (with-meta 'x {:raster.type/tag 'float}) reference)))))))

(defn- constructors []
  [[#(reverse/value+grad #'public/square :representation :normalized-scalar)
    [3.0] [9.0 6.0] #'reverse/make-runtime-value+grad-fn]
   [#(jvp/jvp #'public/square :representation :normalized-scalar)
    [3.0 0.5] [9.0 3.0] #'jvp/make-runtime-jvp-fn]])
(defn- alias-bundle []
  (let [value (av/tensor {:dtype :double :shape []})
        body (with-meta '(.invk raster.ad.normalized-scalar-runtime-guard-test/receiver v v)
               {:raster.op/original 'raster.numeric/* :raster.type/tag 'double})
        eq (list '= 'result '[output]
                 (list 'scalar {:dtypes [:double]} '[x] (soac/lambda-form '[v] [body])))
        program (soac/make (soac/default-program-facts
                           {:inputs '[x] :values {'x value 'output value}
                            :equations {'result (soac/default-equation-facts)}}) [eq] '[output])]
    (normalized/linearize-program program
                                  {:params '[x] :tags '[double] :active-inputs '[x]
                                   :source-ns (the-ns 'raster.ad.normalized-scalar-runtime-guard-test)})))

(deftest guarded-public-closures-retain-results-and-metadata-without-hot-proof-replay
  (doseq [[construct arguments expected _] (constructors)]
    (let [validations (atom 0) validate normalized/validate!]
      (with-redefs [normalized/validate! (fn [bundle] (swap! validations inc) (validate bundle))]
        (let [f (construct) prepared-count @validations]
          (is (pos? prepared-count))
          (dotimes [_ 3] (is (= expected (apply f arguments))))
          (is (= prepared-count @validations))
          (is (thrown? clojure.lang.ArityException (f))
              "the wrapper preserves the existing runtime function's wrong-arity failure")
          (is (:raster.core/deftm (meta f)))
          (is (some? (:raster.ad/normalized-bundle (meta f))))
          (is (= (count (:raster.core/deftm-params (meta f)))
                 (count (:raster.core/deftm-tags (meta f)))))
          (is (= 1 (count (:raster.core/deftm-walked-body (meta f))))))))))

(deftest both-public-directions-use-one-runtime-lifecycle-with-existing-proof-counts
  (doseq [[construct arguments expected _] (constructors)]
    (let [calls (atom 0) validations (atom 0)
          prepare normalized/guarded-runtime validate normalized/validate!]
      (with-redefs [normalized/guarded-runtime
                    (fn [bundle parameters body compile-runtime]
                      (swap! calls inc)
                      (prepare bundle parameters body
                               (fn []
                                 (is (identical? (:source-ns bundle) *ns*)
                                     "runtime compilation retains source namespace")
                                 (compile-runtime))))
                    normalized/validate! (fn [bundle] (swap! validations inc) (validate bundle))]
        (let [f (construct)]
          (is (= 1 @calls))
          (is (= 3 @validations) "construction, body projection, and post-capture proof remain")
          (is (= expected (apply f arguments)))
          (is (= 1 @calls))
          (is (= 3 @validations)))))))

(deftest epochs-and-effective-callees-refuse-before-execution
  (doseq [[construct arguments _ _] (constructors)]
    (let [f (construct) revision (normalized/revisions)]
      (with-redefs [normalized/revisions (constantly (update revision :compiler-revision inc))]
        (is (= :ad-normalized-stale-rules (reason #(apply f arguments)))))
      (with-redefs [n/* (fn [& _] 777.0)]
        (is (= :ad-normalized-execution-callee (reason #(apply f arguments)))))))
  (let [selected (:typed-impl (dispatch/selected-call-signature
                             'raster.numeric/* '[double double]
                             (inference/registered-call-selection 'raster.numeric/* '[double double])))]
    (binding [receiver selected]
      (with-redefs [normalized/prepare (fn [& _] (alias-bundle))]
        (doseq [[construct arguments expected _] (constructors)]
          (let [f (construct)]
            (is (= expected (apply f arguments)))
            (binding [receiver (Object.)]
              (is (= :ad-normalized-execution-callee (reason #(apply f arguments)))))
            (is (= expected (apply f arguments)))))))))

(deftest compilation-and-invocation-publication-have-epoch-fences
  (doseq [[construct arguments expected compiler-var] (constructors)]
    (let [revision (normalized/revisions) current (atom revision) compile @compiler-var]
      (with-redefs-fn {#'normalized/revisions #(deref current)
                      compiler-var (fn [& args]
                                     (let [f (apply compile args)]
                                       (swap! current update :template-revision inc)
                                       f))}
        #(is (= :ad-normalized-stale-rules (reason construct)))))
    (let [revision (normalized/revisions) current (atom revision) calls (atom 0)
          f (with-redefs-fn
              {compiler-var (fn [& _]
                              (fn [& _]
                                (swap! calls inc)
                                (swap! current update :compiler-revision inc)
                                expected))}
              construct)]
      (with-redefs [normalized/revisions #(deref current)]
        (is (= :ad-normalized-stale-rules (reason #(apply f arguments))))
        (is (= 1 @calls))
        (is (= :ad-normalized-stale-rules (reason #(apply f arguments))))
        (is (= 1 @calls))))))

(deftest original-execution-failure-is-not-masked-by-post-fence
  (doseq [[construct arguments _ compiler-var] (constructors)]
    (let [failure (ex-info "original scalar execution failure" {:reason :fixture})
          current (atom (normalized/revisions))
          f (with-redefs-fn {compiler-var (fn [& _]
                                           (fn [& _]
                                             (swap! current update :compiler-revision inc)
                                             (throw failure)))} construct)]
      (with-redefs [normalized/revisions #(deref current)]
        (is (identical? failure (try (apply f arguments) nil (catch Throwable e e)))))))
  (doseq [[construct _ _ compiler-var] (constructors)]
    (let [failure (ex-info "original scalar compilation failure" {:reason :compile-fixture})
          current (atom (normalized/revisions))]
      (with-redefs-fn {#'normalized/revisions #(deref current)
                      compiler-var (fn [& _]
                                     (swap! current update :compiler-revision inc)
                                     (throw failure))}
        #(is (identical? failure (try (construct) nil (catch Throwable e e))))))))
