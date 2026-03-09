(ns raster.compiler.pipeline-test
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest testing is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.simplify :as simplify]
            [raster.compiler.ad.flatten :as flat]
            [raster.compiler.ir.dialects :as dialects]
            [raster.ad.reverse :as rad]))

(defn- form-contains?
  "True if sym appears anywhere in a nested form (symbols, lists, vectors)."
  [form sym]
  (cond
    (= form sym) true
    (and (symbol? form) (symbol? sym)
         (= (name form) (name sym))) true
    (seq? form) (some #(form-contains? % sym) form)
    (vector? form) (some #(form-contains? % sym) form)
    :else false))

;; ================================================================
;; AD template path verification
;; ================================================================

(deftest ad-transform-uses-templates-test
  (testing "AD transform produces flat code with templates (not closure pullback lookup)"
    (let [form '(let* [r (* x y)] r)
          transformed (rad/transform-body form '[x y])]
      ;; Template path: should NOT contain get-pullback-factory runtime lookups
      (is (not (form-contains? transformed 'get-pullback-factory))
          "Template path should be used, not closure pullback lookup"))))

;; ================================================================
;; Flatten + simplify integration
;; ================================================================

(deftest flatten-simplify-integration-test
  (testing "flatten + simplify cleans up AD noise"
    (let [;; f(x,y) = x*y -> df/dx = y, df/dy = x
          form '(let* [r (raster.numeric/* x y)
                       result__1 r]
                      [result__1
                       (fn* [dy__rad]
                            (let* [d_r__2 dy__rad
                                   dg_x__3 (raster.numeric/* d_r__2 y)
                                   dg_y__4 (raster.numeric/* d_r__2 x)]
                                  [dg_x__3 dg_y__4]))])
          ;; Flatten AD output (no optimizer — that's now user code)
          flat-result (flat/flatten-ad-form form)
          flat-form (:form flat-result)
          simplified (simplify/simplify-derivative flat-form)]
      ;; After simplification, (* 1.0 y) should become y
      ;; The form should not have closure anymore
      (is (not (form-contains? simplified 'fn*))
          "No closures should remain after flattening")
      ;; Should have the primal result as body
      (is (= 'let* (first simplified))
          "Should be a flat let* form"))))

;; ================================================================
;; Pipeline function existence
;; ================================================================

(deftest pipeline-api-test
  (testing "pipeline functions exist"
    (is (fn? pipeline/compile-aot))
    (is (fn? pipeline/show-pipeline))))

;; ================================================================
;; Pass registry
;; ================================================================

(deftest pass-rename-test
  (testing "core pipeline pass specs exist"
    (let [specs @#'pipeline/pass-specs]
      (is (contains? specs :lower))
      (is (contains? specs :fixpoint))
      (is (contains? specs :dce))
      (is (contains? specs :expand))))
  (testing "removed pass keys are gone"
    (let [specs @#'pipeline/pass-specs]
      (is (not (contains? specs :inline)))
      (is (not (contains? specs :post-ad-inline)))
      (is (not (contains? specs :normalize)))
      (is (not (contains? specs :cse)))
      (is (not (contains? specs :simplify)))
      (is (not (contains? specs :peephole)))
      (is (not (contains? specs :par-fuse))))))

;; ================================================================
;; Dialect validation
;; ================================================================

(deftest dialect-checkers-test
  (testing "dialect-checkers registry has all pipeline stages"
    (let [checkers (set (keys dialects/dialect-checkers))
          pass-targets (into #{} (map :to) (vals @#'pipeline/pass-specs))
          missing (set/difference pass-targets checkers)]
      (is (empty? missing)
          (str "Missing dialect checkers for pipeline stages: " (sort missing))))))

(deftest diagnostic-runner-validates-pass-output-test
  (testing "diagnostic runner enforces declared output dialects"
    (let [invalid-pass :test-invalid-flattened
          invalid-form '(+ x y)
          original-specs @#'pipeline/pass-specs]
      (alter-var-root #'pipeline/pass-specs
                      assoc
                      invalid-pass
                      {:from :walked
                       :to :flattened
                       :fn (fn [_form _opts] invalid-form)})
      (try
        (try
          (#'pipeline/run-passes-diagnostic '(let* [x 1.0] x) [invalid-pass] {})
          (is false "Expected diagnostic runner to reject invalid pass output")
          (catch clojure.lang.ExceptionInfo ex
            (let [data (ex-data ex)]
              (is (= invalid-pass (:pass data)))
              (is (= :flattened (:dialect data)))
              (is (:validation-error data)))))
        (finally
          (alter-var-root #'pipeline/pass-specs (constantly original-specs)))))))

(deftest valid-walked-test
  (testing "walked forms are recognized"
    (is (dialects/valid-walked?
         '(let* [a (* x y)
                 b (+ a 1.0)]
                b)))
    (is (dialects/valid-walked? '(let* [] x)))
    (is (dialects/valid-walked?
         '(let* [i (double x)] i))))
  (testing "terminals are valid walked expressions"
    ;; Numbers, symbols, strings are valid terminals in the Walked dialect
    (is (dialects/valid-walked? 42))
    (is (dialects/valid-walked? 'x))))

(deftest valid-flattened-test
  (testing "valid flattened forms"
    (is (dialects/valid-flattened?
         '(let* [a (* x y)] a)))
    (is (dialects/valid-flattened?
         '(let* [a (* x y)
                 b (+ a 1.0)]
                [b a]))))
  (testing "non-flattened rejected"
    (is (not (dialects/valid-flattened? '(+ x y))))
    (is (not (dialects/valid-flattened? '[x y])))))

(deftest valid-rewalked-test
  (testing "rewalked forms reject raw scalar calls in walkable bindings"
    (is (not (dialects/valid-rewalked?
              '(let* [a (raster.numeric/+ x y)] a))))
    (let [result (dialects/validate-rewalked
                  '(let* [a (raster.numeric/+ x y)] a))]
      (is (= :pe-rewalked (:fail result)))
      (is (= '(raster.numeric/+ x y) (get-in result [:details :offending])))))
  (testing "rewalked forms allow effect bindings to stay direct"
    (is (dialects/valid-rewalked?
         '(let* [a (.invk _plus__m_double_double-impl x y)
                 _effect (println a)]
                a)))))

(deftest valid-ad-transformed-test
  (testing "let*-wrapped AD output"
    (is (dialects/valid-ad-transformed?
         '(let* [r (* x y)]
                [r (fn* [dy] [dy dy])]))))
  (testing "raw 2-vector AD output is no longer canonical at this stage"
    (is (not (dialects/valid-ad-transformed?
              '[x (fn* [dy] [dy])])))
    (is (= :ad-transformed
           (:fail (dialects/validate-ad-transformed '[x (fn* [dy] [dy])])))))
  (testing "non-AD forms rejected"
    (is (not (dialects/valid-ad-transformed? '(+ x y))))
    (is (not (dialects/valid-ad-transformed? '[x y z])))))

(deftest validate-walked-test
  (testing "valid-walked? accepts walked forms"
    (is (dialects/valid-walked?
         '(let* [a (* x y)] a))))
  (testing "valid-walked? accepts terminals"
    (is (dialects/valid-walked? 42)))
  (testing "validate-walked returns map for structurally invalid"
    (let [result (dialects/validate-walked {:not "an expr"})]
      (is (map? result))
      (is (= :walked (:fail result)))))
  (testing "valid-let*? rejects non-let forms"
    (is (not (dialects/valid-let*? 42)))
    (is (not (dialects/valid-let*? '(+ x y))))
    (is (dialects/valid-let*? '(let* [a 1] a)))))
