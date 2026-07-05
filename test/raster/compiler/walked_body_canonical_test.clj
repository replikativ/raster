(ns raster.compiler.walked-body-canonical-test
  "R1 characterization — pins the walked-body / resolver invariants that the
  content-addressed rewrite (S2+) must preserve, and validates the hasch
  content-key mechanism on REAL raster deftm bodies.

  These are deliberately GREEN on the pre-S2 code: they capture what already
  works (so S2 cannot silently regress it) plus the properties the S2 memo key
  relies on. The fail-loud read-API tests (dispatch-var must throw, not return
  nil) arrive WITH S3, when that behavior change lands — today the silent-nil
  footgun is pinned as `current behavior` so the S3 flip is visible in the diff."
  (:refer-clojure :exclude [+ - * / aget aset alength])
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as rcore :refer [deftm]]
            [raster.numeric :refer [+ - * /]]
            [raster.arrays :refer [aget aset alength alloc-like]]
            [raster.compiler.pipeline :as pl]
            [raster.gpu.core :as gc]
            [raster.ad.reverse :as rev]
            [raster.compiler.core.types :as types]
            [hasch.core :as hasch]))

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

(deftest ensure-walked-body-current-behavior
  (testing "on a RESOLVED impl var: non-empty walked body"
    (let [impl (rcore/resolve-deftm-var #'wbc-sq nil)
          wb (rcore/ensure-walked-body! impl)]
      (is (seq wb) "resolved impl var yields a non-empty walked body")))
  (testing "on a DISPATCH var: currently returns nil (silent-nil footgun)"
    ;; This is the REPL footgun R1 targets. S3 makes this fail-loud (throw with a
    ;; 'resolve first' message). Pinned here so that behavior change is explicit.
    (is (nil? (rcore/ensure-walked-body! #'wbc-sq))
        "PRE-S3: dispatch var silently yields nil (to become fail-loud in S3)")))

;; ---------------------------------------------------------------------------
;; hasch content-key mechanism — the S2 memo key, validated on real bodies
;; ---------------------------------------------------------------------------

(defn- content-key
  "The S2 memo key: pure fn of (source-body, params, tags, annotations,
  source-ns, dtype). Uses hasch's DEFAULT hash — the de-risking finding is that
  raster source bodies carry no load-bearing symbol metadata, so the default
  (which drops symbol metadata) is exactly right."
  [impl-var dtype]
  (let [m (meta impl-var)]
    (hasch/uuid [(:raster.core/deftm-source-body m)
                 (:raster.core/deftm-params m)
                 (:raster.core/deftm-tags m)
                 (:raster.core/deftm-annotations m)
                 (:raster.core/deftm-source-ns m)
                 dtype])))

(deftest hasch-content-key
  (let [impl (rcore/resolve-deftm-var #'wbc-sq nil)]
    (testing "stable across calls (deterministic)"
      (is (= (content-key impl :double) (content-key impl :double))))
    (testing "dtype is part of the key"
      (is (not= (content-key impl :double) (content-key impl :float))))
    (testing "source-body drives the key (different body => different key)"
      (let [k1 (content-key impl :double)
            k2 (hasch/uuid ['(different body form) (:raster.core/deftm-params (meta impl))
                            (:raster.core/deftm-tags (meta impl))
                            (:raster.core/deftm-annotations (meta impl))
                            (:raster.core/deftm-source-ns (meta impl)) :double])]
        (is (not= k1 k2))))
    (testing "de-risking finding: incidental symbol metadata (:line/:column) does NOT change the key"
      ;; two structurally-identical bodies differing only in symbol position meta
      ;; must hash the same — walking does not depend on source position.
      (let [body-a '(let [y (* x x)] y)
            body-b (clojure.walk/postwalk
                    (fn [f] (if (symbol? f) (with-meta f {:line 99 :column 7}) f))
                    '(let [y (* x x)] y))]
        (is (= (hasch/uuid body-a) (hasch/uuid body-b))
            "hasch default drops symbol metadata => position-independent memo key")))))
