(ns raster.compiler.walked-body-canonical-test
  "R1 characterization — pins the resolver + walked-body invariants that the
  reconciliation must preserve: one canonical dispatch->impl resolver (S1) and
  ensure-walked-body! that resolves-first so a dispatch var yields its body
  rather than nil (S2, the debug-tooling footgun that misdiagnosed #78).

  The fail-loud read-API distinction (genuinely-non-deftm vs is-deftm-but-walk-
  failed) arrives with S3."
  (:refer-clojure :exclude [+ - * / aget aset alength])
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as rcore :refer [deftm]]
            [raster.numeric :refer [+ - * /]]
            [raster.arrays :refer [aget aset alength alloc-like]]
            [raster.compiler.pipeline :as pl]
            [raster.gpu.core :as gc]
            [raster.ad.reverse :as rev]
            [raster.compiler.core.types :as types]))

;; A controlled, non-parametric deftm so meta keys are concrete and stable.
(deftm wbc-sq [x :- (Array Double) n :- Long] :- (Array Double)
  (let [out (alloc-like x n)]
    (dotimes [i n]
      (aset out i (* (aget x i) (aget x i))))
    out))

;; ---------------------------------------------------------------------------
;; S1 — canonical resolver: the three callers agree on the backing impl var
;; ---------------------------------------------------------------------------

(deftest resolver-callers-agree
  (testing "dispatch var resolves to the SAME backing ::deftm impl var across callers"
    (let [dv #'wbc-sq
          canon (rcore/resolve-deftm-var dv nil)
          via-pipeline (#'pl/resolve-deftm-var dv)
          via-gpu (gc/resolve-deftm-var dv)
          via-reverse (rev/resolve-deftm-var dv)]
      (is (:raster.core/deftm (meta canon)) "canonical resolves to a ::deftm impl var")
      (is (= canon via-pipeline) "pipeline wrapper == canonical")
      (is (= canon via-gpu) "gpu wrapper == canonical")
      (is (= canon via-reverse) "reverse wrapper == canonical (single overload)")))
  (testing "an already-resolved impl var is returned unchanged (idempotent)"
    (let [impl (rcore/resolve-deftm-var #'wbc-sq nil)]
      (is (= impl (rcore/resolve-deftm-var impl nil)))))
  (testing "mangling matches DEFINITION-time exactly (types/mangle, not manual _m_ join)"
    ;; re-deriving the mangled name from the impl var's own tags via types/mangle
    ;; must resolve back to the SAME var — i.e. the resolver used the identical
    ;; mangling scheme definition-time used (the manual _m_ join would diverge for
    ;; operator-named / compound-tag deftms).
    (let [impl (rcore/resolve-deftm-var #'wbc-sq nil)
          tags (:raster.core/deftm-tags (meta impl))]
      (is (var? impl))
      (is (= impl (ns-resolve (:ns (meta impl)) (types/mangle 'wbc-sq tags)))
          "types/mangle roundtrips to the same backing var"))))

;; ---------------------------------------------------------------------------
;; ensure-walked-body! current behavior (pins the footgun S3 will fix)
;; ---------------------------------------------------------------------------

(deftest ensure-walked-body-resolve-first
  (testing "on a RESOLVED impl var: non-empty walked body (hot path, skips resolve)"
    (let [impl (rcore/resolve-deftm-var #'wbc-sq nil)
          wb (rcore/ensure-walked-body! impl)]
      (is (seq wb) "resolved impl var yields a non-empty walked body")))
  (testing "on a DISPATCH var: resolve-first yields the body (S2 footgun fix)"
    ;; Was the REPL debug-tooling footgun that misdiagnosed #78 (task #53): a
    ;; dispatch var silently returned nil because the body lives on the backing
    ;; impl var. S2 resolves-first, so the dispatch var now yields the SAME body.
    (let [impl (rcore/resolve-deftm-var #'wbc-sq nil)]
      (is (seq (rcore/ensure-walked-body! #'wbc-sq))
          "dispatch var yields a non-empty walked body")
      (is (= (rcore/ensure-walked-body! #'wbc-sq)
             (rcore/ensure-walked-body! impl))
          "dispatch var body == impl var body")))
  (testing "contract preserved: non-deftm / ambiguous dispatch var fall through to nil"
    ;; the inline/op_descriptor/pe callers treat nil as 'not inlinable' — must hold.
    (is (nil? (rcore/ensure-walked-body! #'clojure.core/+))
        "genuinely non-deftm var -> nil (not a crash)")))
