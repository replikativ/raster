(ns raster.compiler.no-inline-test
  "Guards the ^:no-inline opacity contract — load-bearing for the quant/BLAS/tile
   kernels (a ^:no-inline op must stay an opaque call so the backend emits its
   registered kernel, not the inlined scalar body). Asserts the SINGLE shared
   predicate (dispatch/no-inline?) that the inliner + bytecode backend both consult,
   on the generic var, the `_m_` devirtualized child, and a plain deftm."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.dispatch :as dispatch]))

(deftm ^:no-inline opaque-op
  [a :- Double, b :- Double] :- Double
  (raster.numeric/+ a b))

(deftm plain-op
  [a :- Double, b :- Double] :- Double
  (raster.numeric/+ a b))

(deftest no-inline-predicate
  (testing "^:no-inline lands on the generic var (var + qualified symbol)"
    (is (dispatch/no-inline? #'opaque-op))
    (is (dispatch/no-inline? 'raster.compiler.no-inline-test/opaque-op)))
  (testing "a devirtualized `_m_` child resolves opacity via its dispatch parent"
    ;; the impl/child symbol need not resolve to a var — the parent before `_m_` does
    (is (dispatch/no-inline? 'raster.compiler.no-inline-test/opaque-op_m_double_double))
    (is (dispatch/no-inline? 'raster.compiler.no-inline-test/opaque-op_m_anything)))
  (testing "a plain deftm is inlinable (no false positives)"
    (is (not (dispatch/no-inline? #'plain-op)))
    (is (not (dispatch/no-inline? 'raster.compiler.no-inline-test/plain-op)))
    (is (not (dispatch/no-inline? 'raster.compiler.no-inline-test/plain-op_m_double_double))))
  (testing "non-deftm / unresolvable symbols are not opaque"
    (is (not (dispatch/no-inline? 'some.unknown.ns/nope)))
    (is (not (dispatch/no-inline? 'not-qualified)))))
