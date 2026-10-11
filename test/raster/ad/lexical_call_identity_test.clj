(ns raster.ad.lexical-call-identity-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.backend.jvm.lexical-core-call-test :as source-controls]))

(deftm lexical-square [x :- Double] :- Double
  (let [double (fn [v] (* v v))] (double x)))

(deftm nested-lexical-square [x :- Double] :- Double
  (let [inc (fn [v] (* v v))]
    (let [double (fn [v] (inc v))] (double x))))

(deftm explicit-do-lexical-square [x :- Double] :- Double
  (let [double (fn [v] (do (let [z v] (* z z))))] (double x)))

(deftm multiple-body-lexical-square [x :- Double] :- Double
  (let [double (fn [v] (+ v 1.0) (* v v))] (double x)))

(deftm captured-before-shadow [x :- Double] :- Double
  (let [double (fn [v] (+ x (* v v)))]
    (let [x 100.0] (+ (double 3.0) (* x 0.0)))))

(deftm qualified-core-not-lexical [x :- Double] :- Double
  (let [double (fn [v] (* v v))]
    (+ (clojure.core/double x) (double 2.0))))

(deftm parameter-callee [double :- clojure.lang.IFn, x :- Double] :- Double
  (double x))

(deftm parameter-shadows-deftm [lexical-square :- clojure.lang.IFn, x :- Double] :- Double
  (lexical-square x))

(deftm parameter-shadows-shape-read [alength :- clojure.lang.IFn, x :- Double] :- Double
  (alength x))

(deftm inactive-parameter-core-name [+ :- clojure.lang.IFn, x :- Double] :- Double
  (clojure.core/+ x (+ 3.0)))

(deftm opaque-captured-zero-argument [x :- Double] :- Double
  (let [double (fn [] x)] (+ (double) (double))))

(deftm conditional-captured-zero-argument [x :- Double, choose :- Boolean] :- Double
  (let [double (if choose (fn [] x) (fn [] (* x x)))] (double)))

(deftm opaque-captures-core-name [inc :- Double] :- Double
  (let [double (fn [] inc)] (+ (double) (double))))

(deftm inactive-captured-call [x :- Double] :- Double
  (let [k 3.0 double (fn [] k)] (+ x (+ (double) (double)))))

(deftm inactive-formal-shadow [x :- Double] :- Double
  (let [double (fn [x] x)] (+ x (+ (double 2.0) (double 2.0)))))

(deftm inactive-effect-region [state :- clojure.lang.Atom, x :- Double] :- Double
  (+ x (do (swap! state conj :first)
           (let [k 3.0] (do (swap! state conj :second) k)))))

(defn- refusal [construct target wrt]
  (try (if (= construct rev/value+grad)
         (construct target :mode :reverse :wrt wrt)
         (construct target))
       nil
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest transparent-local-functions-preserve-independent-derivatives
  (doseq [[target x primal derivative]
          [[#'lexical-square 3.0 9.0 6.0]
           [#'nested-lexical-square 3.0 9.0 6.0]
           [#'captured-before-shadow 3.0 12.0 1.0]
           [#'qualified-core-not-lexical 3.0 7.0 1.0]
           [#'inactive-captured-call 3.0 9.0 1.0]
           [#'inactive-formal-shadow 3.0 7.0 1.0]]]
    (is (= primal (target x)))
    (is (= derivative (/ (- (target (+ x 0.125)) (target (- x 0.125))) 0.25)))
    (is (= [primal derivative] ((rev/value+grad target :mode :reverse) x)))
    (is (= [primal derivative] ((jvp/jvp target) x 1.0))))
  ;; Existing independent Float feedback oracle remains a distinct program,
  ;; not the Double checkpoint counterexample that should be refused.
  (is (= 0.0 (source-controls/float-feedback-oracle 16777216.5)))
  (is (= [0.0 0.0] ((rev/value+grad #'source-controls/lexical-float-feedback) 16777216.5)))
  (is (= [0.0 0.0] ((jvp/jvp #'source-controls/lexical-float-feedback) 16777216.5 1.0))))

(deftest inactive-parameter-core-name-keeps-its-executable-callee
  (let [callee #(* % %)]
    (is (= 12.0 (inactive-parameter-core-name callee 3.0)))
    (is (= [12.0 nil 1.0] ((rev/value+grad #'inactive-parameter-core-name :mode :reverse :wrt [1]) callee 3.0)))
    (is (= [12.0 1.0] ((jvp/jvp #'inactive-parameter-core-name) callee 3.0 1.0)))))

(deftest opaque-active-callees-fail-with-specific-contracts
  (is (= 9.0 (parameter-callee #(* % %) 3.0)))
  (is (= 103.0 (parameter-shadows-deftm #(+ % 100.0) 3.0)))
  (is (= 9.0 (parameter-shadows-shape-read #(* % %) 3.0)))
  (is (= 6.0 (opaque-captured-zero-argument 3.0)))
  (is (= 3.0 (conditional-captured-zero-argument 3.0 true)))
  (is (= 9.0 (conditional-captured-zero-argument 3.0 false)))
  (doseq [construct [rev/value+grad jvp/jvp]]
    (is (= :ad-active-lexical-callee (refusal construct #'parameter-callee [1])))
    (is (= :ad-active-lexical-callee (refusal construct #'parameter-shadows-deftm [1])))
    (is (= :ad-active-lexical-callee (refusal construct #'parameter-shadows-shape-read [1])))
    (is (= :ad-active-closure (refusal construct #'opaque-captured-zero-argument [0])))
    (is (= :ad-active-closure (refusal construct #'conditional-captured-zero-argument [0])))
    (is (= :ad-active-closure (refusal construct #'opaque-captures-core-name [0])))))

(deftest actual-active-regions-are-correct-or-specifically-declined-before-execution
  (doseq [target [#'explicit-do-lexical-square #'multiple-body-lexical-square]]
    (is (= 9.0 (target 3.0)))
    (is (= 6.0 (/ (- (target 3.125) (target 2.875)) 0.25)))
    (doseq [[construct invoke] [[#(rev/value+grad % :mode :reverse) #(% 3.0)]
                                [jvp/jvp #(% 3.0 1.0)]]]
      (let [prepared (try {:function (construct target)}
                          (catch clojure.lang.ExceptionInfo e {:refusal (:reason (ex-data e))}))]
        (if-let [function (:function prepared)]
          (is (= [9.0 6.0] (invoke function)))
          (is (= :ad-active-unlinearized-region (:refusal prepared))))))))

(deftest inactive-regions-preserve-effect-order-and-execute-only-at-invocation
  (let [state (atom [])]
    (is (= 5.0 (inactive-effect-region state 2.0)))
    (is (= [:first :second] @state)))
  (doseq [[construct invoke expected]
          [[#(rev/value+grad % :mode :reverse :wrt [1]) #(%1 %2 2.0) [5.0 nil 1.0]]
           [jvp/jvp #(%1 %2 2.0 1.0) [5.0 1.0]]]]
    (let [state (atom []) differentiated (construct #'inactive-effect-region)]
      (is (= [] @state))
      (is (= expected (invoke differentiated state)))
      (is (= [:first :second] @state)))))

(defn- execute-source [body params args]
  (apply (eval (list 'fn params body)) args))

(def ^:dynamic *argument-value* 3)

(deftest nonlambda-let-initializers-are-not-parsed-as-arities
  (let [body '(let* [number 3.0 flag true absent nil text "plain"
                    values [1 2] entries {:value 4} members #{5} alias number]
                [alias flag absent text values entries members])
        prepared (inline/inline-transparent-lexical-calls body {})]
    (is (= [3.0 true nil "plain" [1 2] {:value 4} #{5}]
           (execute-source body [] [])))
    (is (= (execute-source body [] []) (execute-source prepared [] [])))))

(deftest beta-keeps-call-site-order-unused-arguments-and-capture-snapshots
  (let [body '(let* [double (fn* [v unused]
                             (do (swap! state conj :body) (+ v v)))]
                (if take?
                  (double (do (swap! state conj :first) 3)
                          (do (swap! state conj :unused) 4))
                  -1))
        prepared (inline/inline-transparent-lexical-calls body {'state nil 'take? nil})]
    (doseq [[take? expected events] [[true 6 [:first :unused :body]] [false -1 []]]]
      (doseq [source [body prepared]]
        (let [state (atom [])]
          (is (= expected (execute-source source '[state take?] [state take?])))
          (is (= events @state))))))
  (let [body '(let* [x 3 double (fn* [v] (+ x v))]
                (let* [x 100] (double 2)))
        prepared (inline/inline-transparent-lexical-calls body {})]
    (is (= 5 (execute-source body [] [])))
    (is (= 5 (execute-source prepared [] []))))
  (let [body '(let* [double (fn* [v unused]
                             (binding [raster.ad.lexical-call-identity-test/*argument-value* 100] v))]
                (double raster.ad.lexical-call-identity-test/*argument-value* 4))
        prepared (inline/inline-transparent-lexical-calls body {})]
    ;; Symbolic actuals are read at application, not after the callee changes
    ;; its dynamic environment. Neither form mutates a global root.
    (is (= 3 (execute-source body [] [])))
    (is (= 3 (execute-source prepared [] [])))))

(deftest mutable-annotated-and-escaping-lambdas-are-not-beta-proofs
  (let [body '(loop* [double (fn* [v] (* v v)) k 0]
                (if (< k 1)
                  (recur (fn* [v] (+ v 1)) (inc k))
                  (double x)))
        prepared (inline/inline-transparent-lexical-calls body {'x 'double})]
    (is (= 4.0 (execute-source body '[x] [3.0])))
    (is (= 4.0 (execute-source prepared '[x] [3.0]))))
  (let [body '(let* [double (fn* [v] (if (> v 0) (recur (dec v)) v))]
                (double x))
        prepared (inline/inline-transparent-lexical-calls body {'x 'double})]
    (is (= 0.0 (execute-source body '[x] [3.0])))
    (is (= 0.0 (execute-source prepared '[x] [3.0]))))
  (doseq [body ['(let* [double (fn* [^long v] (* v v))] (double x))
                '(let* [double (fn* ^long [v] v)] (double x))
                '(let* [double (fn* self [v] (self v))] (double x))
                '(let* [double (fn* [v & vs] v)] (double x))
                '(let* [double (fn* [v] (* v v))] [double (double x)])]]
    (let [prepared (inline/inline-transparent-lexical-calls body {'x 'double})
          calls (filter #(and (seq? %) (:raster.op/lexical-callee (meta %)))
                        (tree-seq coll? seq prepared))]
      (is (seq calls))
      (is (every? #(nil? (:operation (op/call-description %))) calls)))))

(deftest lexical-invk-cannot-revive-recorded-global-identity
  (let [call (with-meta '(.invk double x) {:raster.op/original 'clojure.core/double})
        prepared (inline/inline-transparent-lexical-calls call {'double 'clojure.lang.IFn 'x 'double})
        descriptor (op/call-description prepared)]
    (is (= 'double (:lexical-callee descriptor)))
    (is (= 'double (:implementation-op descriptor)))
    (is (= '[x] (:arguments descriptor)))
    (is (nil? (:operation descriptor)))
    (is (nil? (op/semantic-op prepared)))
    (let [renamed (util/subst-syms {'double 'renamed-receiver} prepared)
          projected (op/call-description renamed)]
      (is (= 'renamed-receiver (:lexical-callee projected)))
      (is (= 'renamed-receiver (:implementation-op projected)))
      (is (nil? (:operation projected))))))

(deftest lexical-invk-name-is-not-discrete-shape-evidence
  (let [active? (ns-resolve 'raster.ad.reverse 'init-active?)
        call (with-meta '(.invk alength_m_local x)
               {:raster.op/lexical-callee 'alength_m_local
                :raster.op/original 'clojure.core/alength})]
    (is (true? (active? call {'x true})))
    (is (nil? (:operation (op/call-description call))))))

(deftest semantic-invk-arithmetic-is-not-parsed-as-direct-call-operands
  (let [normalize (ns-resolve 'raster.ad.reverse 'binaryize-core-arithmetic)
        retained (with-meta '(.invk add-impl a b c)
                   {:raster.op/original 'clojure.core/+})]
    (is (= retained (normalize retained)))
    (is (= (meta retained) (meta (normalize retained))))))

(deftest lexical-key-presence-masks-only-the-exact-global-deftm-call
  (let [context {:source-ns (the-ns 'raster.ad.lexical-call-identity-test)
                 :type-env {'lexical-square {:tag nil}}}]
    (is (= :call (walker/classify-form '(lexical-square 3.0) context)))
    (is (= :deftm-call (walker/classify-form '(lexical-square 3.0) (assoc context :type-env {}))))
    (is (= :deftm-call
           (walker/classify-form '(raster.ad.lexical-call-identity-test/lexical-square 3.0) context)))))

(deftest scope-transport-preserves-vector-return-hints-without-restamping-formals
  (let [formal (with-meta 'v {:raster.type/tag 'double})
        params (with-meta [formal] {:tag 'long :fixture :return-vector})
        expression (list 'fn* params (list '+ formal 'outside))]
    (doseq [transported [(util/alpha-convert expression)
                         (util/subst-syms {'outside 'replacement} expression)]]
      (let [vector (second transported) binder (first vector)]
        (is (= (meta params) (meta vector)))
        (is (= 'double (:raster.type/tag (meta binder))))
        (is (nil? (:tag (meta binder))) "vector return hint does not become a formal coercion")))))
