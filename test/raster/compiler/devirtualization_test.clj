(ns raster.compiler.devirtualization-test
  "Gate against type-transport regressions on the LAZY-JIT path.

  A deftm's lazy-JIT code is exactly the walker's output. If type inference fails
  to type a binding, the ops consuming it stay runtime-dispatched (raster.numeric/*
  instead of devirtualized .invk) — ~270ns vs ~4ns each, silently ~5-70x slower in
  hot loops. Correctness tests can't see this (dispatched ops compute the right
  answer), so we assert it directly: hot kernels must fully devirtualize.

  This would have caught the TC-source-ns hole (optimize-layout! went 0→19
  dispatched when TC ran in the wrong namespace)."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.tooling.inspect :as inspect]
            [raster.compiler.pipeline :as pipeline]
            [raster.umap :as umap]
            [raster.umap.graph :as graph]
            [raster.spatial.nndescent :as nnd]))

(def ^:private get-walked-body @#'pipeline/get-walked-body)

(defn- dispatch-counts
  "Walk all walked-body forms of a deftm var and total {:devirtualized :dispatched}."
  [v]
  (let [wb (get-walked-body v nil)
        forms (if (vector? wb) wb [wb])]
    (reduce (fn [acc f]
              (merge-with + acc (select-keys (inspect/analyze-devirtualization f)
                                             [:devirtualized :dispatched])))
            {:devirtualized 0 :dispatched 0}
            forms)))

(deftest hot-kernels-fully-devirtualized
  (testing "hot deftm kernels devirtualize 100% on the lazy-JIT walk path"
    ;; These must be exactly 0 — any runtime-dispatched numeric op in these tight
    ;; loops is a type-transport regression (e.g. TC not run in the source ns).
    (doseq [v [#'umap/optimize-layout!
               #'graph/smooth-knn-dist!     ; inline (if ...) result now typed
               #'graph/membership-strengths!
               #'nnd/cos-dist
               #'nnd/local-join!]]
      (let [{:keys [devirtualized dispatched]} (dispatch-counts v)]
        (is (zero? dispatched)
            (str v " has " dispatched " undevirtualized dispatch op(s) ("
                 devirtualized " devirtualized) — type-transport regression on the "
                 "lazy-JIT path; check that TC binding types reach the walker"))))))

;; NOTE: analyze-devirtualization currently only detects un-devirtualized
;; raster.numeric/raster.math ops, NOT un-devirtualized user-deftm calls (e.g.
;; byte-dist inside local-join-bytes!, which still runs via runtime dispatch).
;; Extending it to flag user-deftm dispatch is tracked follow-up work.
