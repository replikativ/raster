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
            [raster.spatial.nndescent :as nnd]))

;; NOTE: the UMAP-layer devirt cases (optimize-layout-chunk!, graph kernels) moved
;; to the umap-rstr library's test suite when UMAP was factored out. This gate now
;; covers the raster-resident hot kernels (NN-descent).

(def ^:private get-walked-body @#'pipeline/get-walked-body)

(defn- dispatch-counts
  "Walk all walked-body forms of a deftm var and total {:devirtualized :dispatched}.
  Parametric kernels (All [T]) are instantiated per element type; pass an explicit
  dtype to pick one overload (devirtualization is dtype-independent)."
  [v dtype]
  (let [wb (get-walked-body v dtype)
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
    ;; [var dtype] — dtype disambiguates parametric (All [T]) overloads.
    (doseq [[v dtype] [[#'nnd/cos-dist :double]           ; parametric (f32+f64)
                       [#'nnd/local-join! nil]]]
      (let [{:keys [devirtualized dispatched]} (dispatch-counts v dtype)]
        (is (zero? dispatched)
            (str v " has " dispatched " undevirtualized dispatch op(s) ("
                 devirtualized " devirtualized) — type-transport regression on the "
                 "lazy-JIT path; check that TC binding types reach the walker"))))))

;; NOTE: analyze-devirtualization currently only detects un-devirtualized
;; raster.numeric/raster.math ops, NOT un-devirtualized user-deftm calls (e.g.
;; byte-dist inside local-join-bytes!, which still runs via runtime dispatch).
;; Extending it to flag user-deftm dispatch is tracked follow-up work.
