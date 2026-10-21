(ns raster.compiler.core.source-construction-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-prefix-test :as fixture]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.core.util :as util]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.pipeline :as pipeline]))

(defn- capture []
  ;; Finish the ordinary numerical JIT before retaining historical inputs.
  (fixture/unused-checked-helper-input 2 3)
  (let [captured (atom nil) construct transport/with-source-construction-history
        target :ocl:source-construction-test
        compilation
        (with-redefs [transport/with-source-construction-history
                      (fn [normalization run]
                        (let [result (construct normalization run)]
                          (reset! captured (assoc result :normalization normalization))
                          result))]
          (equation/compile #'fixture/unused-checked-helper-input {:target target :dtype :double}
                            {:device-id target :device-type :gpu :backend :ocl
                             :subgroup-dialect :opencl-portable :max-workgroup-size 256}))]
    (assoc @captured :compilation compilation)))

(defn- phase-view [{:keys [source-construction normalization form]}]
  (entry/call-with-definition-inputs
   #(transport/source-construction-view source-construction normalization form)))

(defn- builder-view [input expansion]
  (entry/call-with-definition-inputs #(transport/builder-expansion-view input expansion)))

(defn- origin-view [input expression]
  (entry/call-with-definition-inputs #(transport/construction-occurrence input expression)))

(defn- history [normalization run]
  (entry/call-with-definition-inputs
   #(transport/with-source-construction-history normalization run)))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftm recursive-argument-prefix [n :- Long] :- (Array double)
  (let [size (fixture/literal-count (fixture/literal-count n))
        out (double-array size)]
    (par/map! out i size double 1.0)))

(deftest actual-source-phase-connects-elaboration-to-the-shared-builder
  (let [{:keys [source-construction compilation] :as captured} (capture)
        view (phase-view captured)
        event (first (:builder-expansions view))
        owned (builder-view source-construction (:expansion event))
        origin (:caller-origin owned)
        original (second (second (first (:body (:frontend origin)))))]
    (is (seq (:type-elaborations view)))
    (is (= 1 (count (:builder-expansions view))))
    (is (identical? original (:original origin)))
    (is (identical? (:caller event) (:constructed origin)))
    (is (= 1 (count (:construction-edges origin))))
    (is (identical? (:caller event) (:constructed (last (:construction-edges origin)))))
    (is (every? true? (map identical? (:arguments event)
                           (descriptor/call-args (:caller event)))))
    (is (some? (:callee-import owned)))
    (is (identical? (:walked-body event) (get-in owned [:callee-import :walked-body])))
    (is (= 'size (:result-symbol owned)))
    (is (identical? (:expansion event) (:expansion owned)))
    (is (nil? (:source-construction compilation)))
    (is (nil? (get-in compilation [:options :source-construction])))
    (is (nil? (get-in compilation [:semantic :attributes :source-construction])))
    (is (= :unclosed-ordered-source-executor
           (reason #(equation/lower compilation [Long/MAX_VALUE 2])))
        "builder provenance is not formal completion or final interval execution permission")))

(deftest construction-queries-require-exact-phase-and-output-identities
  (let [{:keys [source-construction normalization form] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        caller (:caller event) expansion (:expansion event)
        copied-caller (with-meta (apply list caller) (meta caller))
        copied-form (with-meta (apply list form) (meta form))]
    (is (nil? (transport/source-construction-view source-construction normalization form)))
    (is (nil? (transport/construction-occurrence source-construction caller)))
    (is (nil? (transport/builder-expansion-view source-construction expansion)))
    (entry/call-with-definition-inputs
     (fn []
       (is (nil? (transport/source-construction-view {} normalization form)))
       (is (nil? (transport/source-construction-view source-construction {} form)))
       (is (nil? (transport/source-construction-view source-construction normalization copied-form)))
       (is (nil? (transport/construction-occurrence source-construction copied-caller)))
       (is (nil? (transport/builder-expansion-view source-construction (into {} expansion))))))
    (let [forbidden (fn [& _] (throw (AssertionError. "query reconstructed source")))]
      (with-redefs-fn {#'pipeline/tag-expr-types forbidden
                      #'util/uniquify-rebindings forbidden
                      #'inline/build-inline-expansion forbidden}
        #(do (is (some? (phase-view captured)))
             (is (some? (origin-view source-construction caller)))
             (is (some? (builder-view source-construction expansion))))))
    (is (= {:form :plain} (transport/with-source-construction-history normalization (constantly :plain))))))

(deftest construction-chains-do-not-grant-type-or-completion-permission
  (let [{:keys [source-construction normalization] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        original (:normalized (:caller-origin (builder-view source-construction (:expansion event))))
        changed (with-meta original (assoc (meta original) :tag 'double))
        result (history normalization
                        #(do (transport/observe-type-elaboration! original changed {'rows 'long})
                             changed))
        origin (origin-view (:source-construction result) changed)]
    (is (some? origin) "the observed origin survives even an incompatible tag change")
    (is (= 'double (:tag (meta (:constructed origin)))))
    (is (not-any? #(contains? origin %) [:completion :result-carrier :identity-cast :executor]))
    (is (= 1 (count (:construction-edges origin))))
    (let [result (history normalization
                          #(do (transport/observe-type-elaboration! original original {}) original))]
      (is (some? (origin-view (:source-construction result) original)))
      (is (empty? (:type-elaborations
                   (entry/call-with-definition-inputs
                    #(transport/source-construction-view (:source-construction result)
                                                         normalization original))))))
    (doseq [mode [:duplicate :fan-out :fan-in :cycle]]
      (let [other (with-meta (apply list original) {:tag 'long})
            result (history normalization
                            #(do
                               (transport/observe-type-elaboration! original changed {})
                               (case mode
                                 :duplicate (transport/observe-type-elaboration! original changed {})
                                 :fan-out (transport/observe-type-elaboration! original other {})
                                 :fan-in (transport/observe-type-elaboration! other changed {})
                                 :cycle (transport/observe-type-elaboration! changed original {}))
                               changed))]
        (is (nil? (origin-view (:source-construction result) changed)) (str mode))))))

(deftest builder-arguments-cannot-be-replaced-by-equal-or-rewritten-inputs
  (let [{:keys [source-construction normalization] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        imported (:callee-import (builder-view source-construction (:expansion event)))
        callee {:params (:parameters imported) :tags (:tags imported)
                :walked-body (:walked-body imported) :source-import (:source-import event)}
        caller (:normalized (:caller-origin (builder-view source-construction (:expansion event))))
        arguments (vec (descriptor/call-args caller))
        build @#'inline/build-inline-expansion]
    (doseq [mode [:original :copied-argument :missing-argument :repeated-caller]]
      (let [actual (case mode
                     :copied-argument [(with-meta (apply list (first arguments)) (meta (first arguments)))]
                     :missing-argument []
                     arguments)
            result (history normalization
                            #(if (= mode :missing-argument)
                               ;; Model an independently transformed builder input; observation is
                               ;; provenance only and the query must reject the lost argument.
                               (let [expansion {:argument-bindings [] :bindings []}]
                                 (transport/observe-builder-expansion! caller callee actual {} 'result expansion)
                                 expansion)
                               (let [expansion (build callee actual {} inline/subst-syms 'result caller)]
                                 (when (= mode :repeated-caller)
                                   (build callee actual {} inline/subst-syms 'other caller))
                                 expansion)))
            view (builder-view (:source-construction result) (:form result))]
        (if (= mode :original)
          (is (some? view))
          (is (nil? view) (str mode)))))))

(deftest construction-history-keeps-original-frontend-and-callee-guards
  (let [{:keys [source-construction] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        owned (builder-view source-construction (:expansion event))
        caller (get-in owned [:caller-origin :frontend :source-entry :source-var])
        callee (get-in owned [:callee-import :frontend :source-entry :source-var])]
    (doseq [source [caller callee]]
      (let [metadata (meta source)
            body (:raster.core/deftm-source-body metadata)]
        (try
          (alter-meta! source assoc :raster.core/deftm-source-body
                       (with-meta (mapv identity body) (meta body)))
          (is (= :stale-definition-proof-input (reason #(phase-view captured))))
          (is (= :stale-definition-proof-input
                 (reason #(builder-view source-construction (:expansion event)))))
          (finally (reset-meta! source metadata)))))))

(deftest recursive-builder-observes-caller-before-rewriting-arguments
  (recursive-argument-prefix 2)
  (let [normalized
        (entry/call-with-definition-inputs
         (fn []
           (let [imported (#'inline/try-resolve-walked-body #'recursive-argument-prefix_m_long)
                 frontend (:frontend (transport/body-import-view (:source-import imported)
                                                                 (:walked-body imported)))]
             (transport/normalize-frontend-bindings frontend '[n]))))
        caller (second (second (first (:walked-body normalized))))
        nested (first (descriptor/call-args caller))]
    (doseq [rewrite-argument? [false true]]
      (let [actual-caller (if rewrite-argument? caller nested)
            result (history (:source-normalization normalized)
                            #(#'inline/inline-invk actual-caller))
            input (:source-construction result)
            phase (entry/call-with-definition-inputs
                   #(transport/source-construction-view input (:source-normalization normalized)
                                                        (:form result)))
            events (:builder-expansions phase)
            outer (first (filter #(identical? actual-caller (:caller %)) events))
            view (builder-view input (:expansion outer))]
        (is (= 7 (:form result)))
        (is (some? outer))
        (if rewrite-argument?
          (do (is (= 2 (count events)))
              (is (nil? view) "rewritten arguments cannot borrow the original call's occurrences")
              (is (some? (builder-view input (:expansion (first events))))))
          (do (is (= 1 (count events)))
              (is (some? view))))))))
