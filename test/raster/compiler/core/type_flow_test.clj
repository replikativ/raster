(ns raster.compiler.core.type-flow-test
  "Tests for type information flow through the compiler pipeline.
  Verifies that namespaced type metadata (:raster.type/*, :raster.buffer/*,
  :raster.effect/*, :raster.op/*) propagates correctly through passes."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.core.types :as types]
            [raster.numeric]))

;; ================================================================
;; Walker type propagation
;; ================================================================

(deftest walker-env-tracks-param-types
  (testing "Walker ctx tracks parameter types in unified type-env"
    (let [ctx (walker/make-ctx {:type-env {'x {:tag 'double} 'y {:tag 'doubles}}
                                :source-ns *ns*})]
      (is (= 'double (walker/ctx-get-tag ctx 'x)))
      (is (= 'doubles (walker/ctx-get-tag ctx 'y)))
      (is (nil? (walker/ctx-get-tag ctx 'z))))))

(deftest walker-outputs-raster-op-original
  (testing "Walker sets :raster.op/original on devirtualized .invk forms"
    (let [walked (walker/walk-body '(let [z (raster.numeric/+ x y)] z)
                                   {:type-env {'x {:tag 'double} 'y {:tag 'double}}})]
      (let [bindings (partition 2 (second walked))
            [_ init] (first bindings)]
        ;; init should be the devirtualized .invk form with :raster.op/original
        (is (= 'raster.numeric/+ (:raster.op/original (meta init)))
            ".invk form should have :raster.op/original metadata")))))

(deftest walker-propagates-fn-info-through-let
  (testing "IFn__ tags in let bindings auto-derive fn-info"
    ;; When a let binding has an IFn__ tag (e.g., from field access),
    ;; the walker should populate fn-info so fn-args-safe? passes.
    (let [ctx (walker/make-ctx
               {:type-env {'f {:tag 'raster.fn.IFn__doubles_doubles_double__void
                               :fn-info {:iface-name 'raster.fn.IFn__doubles_doubles_double__void}}}
                :source-ns *ns*})]
      (is (walker/ctx-get-fn-info ctx 'f)
          "Symbol with IFn__ tag should have fn-info in context"))))

(deftest walker-ctx-assoc-type-auto-fn-info
  (testing "ctx-assoc-type auto-derives fn-info from IFn__ tags"
    (let [ctx (walker/make-ctx {:source-ns *ns*})
          ctx' (walker/ctx-assoc-type ctx 'f 'raster.fn.IFn__doubles_doubles_double__void)]
      (is (= 'raster.fn.IFn__doubles_doubles_double__void
             (walker/ctx-get-tag ctx' 'f)))
      (is (= {:iface-name 'raster.fn.IFn__doubles_doubles_double__void}
             (walker/ctx-get-fn-info ctx' 'f))
          "fn-info should be auto-derived from IFn__ tag"))))

;; ================================================================
;; CSE type preservation
;; ================================================================

(deftest cse-preserves-type-metadata
  (testing "CSE normalization preserves :raster.type/* keys on symbols"
    (let [sym (with-meta 'x {:raster.type/tag 'doubles
                             :raster.type/fn-info {:iface-name 'IFn__}
                             :some-other-key 42})
          ;; Simulate CSE normalization
          normalized (require 'raster.compiler.passes.scalar.cse)
          norm-fn @(resolve 'raster.compiler.passes.scalar.cse/normalize-expr)
          result (norm-fn sym)]
      (is (= 'doubles (:raster.type/tag (meta result)))
          ":raster.type/tag should survive CSE")
      (is (:raster.type/fn-info (meta result))
          ":raster.type/fn-info should survive CSE")
      (is (nil? (:some-other-key (meta result)))
          "Non-type keys should be stripped by CSE"))))

;; ================================================================
;; Helper functions
;; ================================================================

(deftest sym-type-tag-prefers-namespaced
  (testing "sym-type-tag prefers :raster.type/tag over :tag"
    (is (= 'doubles (types/sym-type-tag
                     (with-meta 'x {:raster.type/tag 'doubles :tag 'Object}))))
    (is (= 'double (types/sym-type-tag
                    (with-meta 'x {:tag 'double})))
        "Falls back to :tag when :raster.type/tag absent")))

(deftest fn-type-tag-detection
  (testing "fn-type-tag? identifies IFn__ symbols"
    (is (types/fn-type-tag? 'raster.fn.IFn__double__double))
    (is (types/fn-type-tag? 'raster.fn.IFn__doubles_doubles_double__void))
    (is (not (types/fn-type-tag? 'double)))
    (is (not (types/fn-type-tag? 'doubles)))
    (is (not (types/fn-type-tag? nil)))))
