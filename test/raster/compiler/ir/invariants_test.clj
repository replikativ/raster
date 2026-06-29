(ns raster.compiler.ir.invariants-test
  "Tests for compiler boundary invariants — esp. I4 (closed core)."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.ir.invariants :as inv]))

;; ================================================================
;; I4 — closed core: no macro binding/control form survives the walker
;; ================================================================

(deftest non-closed-core-catches-macro-forms
  (testing "bare macro binding/control heads are flagged"
    (is (= '(let [b 1] b)   (inv/non-closed-core '(let* [a (let [b 1] b)] a))))
    (is (= '(when y z)      (inv/non-closed-core '(if x (when y z) w))))
    (is (some? (inv/non-closed-core '(loop [i 0] (recur i)))))
    (is (some? (inv/non-closed-core '(fn [x] x))))
    (is (some? (inv/non-closed-core '(case x 0 a 1 b))))
    (is (some? (inv/non-closed-core '(clojure.core/let [a 1] a)))
        "qualified macro forms are flagged too"))
  (testing "whitelist catches ANY macro, not just a hardcoded blacklist"
    ;; regression for F1: these slipped through the old blacklist
    (is (some? (inv/non-closed-core '(do (when-let [a b] a)))))
    (is (some? (inv/non-closed-core '(do (if-let [a b] a c)))))
    (is (some? (inv/non-closed-core '(do (for [a b] a)))))
    (is (some? (inv/non-closed-core '(do (letfn [(f [x] x)] (f 1)))))
        "macro letfn flagged; closed-core letfn* is not")))

(deftest non-closed-core-memo-is-ns-correct
  (testing "a bare macro head gets the correct PER-NAMESPACE verdict (memo keyed on [ns head])"
    ;; regression: the memo must not cache one ns's resolution and reuse it under
    ;; another *ns* — `when` is clojure.core/when in a real ns but unresolvable in
    ;; a refer-less ns. Scanning the refer-less ns FIRST must not poison the real ns.
    (create-ns 'inv-test-bare-ns)
    (is (false? (boolean (binding [*ns* (the-ns 'inv-test-bare-ns)]
                           (inv/non-closed-core '(do (when a b))))))
        "bare when is not a resolvable macro in a refer-less ns")
    (is (true? (boolean (binding [*ns* (the-ns 'clojure.core)]
                          (inv/non-closed-core '(do (when a b))))))
        "bare when IS the clojure.core macro in a real ns — flagged, not poisoned")))

(deftest non-closed-core-accepts-closed-core
  (testing "the closed core + preserved SOAC primitives pass"
    (is (nil? (inv/non-closed-core
               '(let* [a 1]
                      (loop* [i 0]
                             (if (clojure.core/< i n)
                               (recur (clojure.core/inc i))
                               (case* a 0 1 2)))))))
    (is (nil? (inv/non-closed-core '(fn* [x] (clojure.core/+ x 1)))))
    (is (nil? (inv/non-closed-core '(dotimes [i n] (raster.par/map! out j m nil (clojure.core/+ x j)))))
        "dotimes + raster.par/* are PRESERVED primitives, not macro forms")
    (is (nil? (inv/non-closed-core '(letfn* [f (fn* [x] x)] (f 1))))
        "letfn* is closed-core (a special form)")))

(deftest non-closed-core-opaque-to-quote-and-source-body
  (testing "quoted code-as-data is not scanned"
    (is (nil? (inv/non-closed-core '(let* [a 1] (quote (let [b 1] (when c d))))))))
  (testing "raw ftm source-body payload is pre-macroexpand and skipped"
    (is (nil? (inv/non-closed-core
               '(ftm [x] :- Double
                     :raster.walker/source-body [(let [a 1] (when a x))]
                     (.invk impl x)))))
    (is (some? (inv/non-closed-core
                '(ftm [x] :raster.walker/source-body [(let [a 1] a)] (when x y))))
        "a real macro in the ftm WALKED body is still caught (only source-body skipped)"))
  (testing "source-body strip is gated on ftm (F2): a stray marker elsewhere does not mask a sibling"
    (is (some? (inv/non-closed-core
                '(some-call x :raster.walker/source-body (let [a 1] a) y)))
        "the (let ...) sibling after a non-ftm stray marker is still scanned")))
