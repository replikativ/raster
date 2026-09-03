(ns raster.compiler.explain-pipeline-truth-test
  "EXPLAIN-PIPELINE MUST DESCRIBE WHAT RAN. CLAUDE.md makes observability a design principle; this
   pins the four ways the tool lied before Phase 0:

     * it reported the HOST CPU as 'Hardware:' for a :ze:0 compile;
     * the former metadata-only segop pass could leave the form `=` to the previous stage, and the
       `[unchanged]` short-circuit dropped its stats — hiding the diagnostic exactly when it existed;
     * three live passes had no label and fell out of a hand-maintained table that carried three
       dead ones — a fourth disagreeing description of the pipeline, in the tool meant to be the
       truth;
     * the kernel record dropped :strategy/:declines/:tile, so nothing could say which leaf a
       kernel came from or why a faster one was refused.

   GPU-gated: it compiles a real deftm for :ze:0. Device-free for the label-derivation check."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.compiler.pipeline :as pl]
            [raster.linalg.contract]))

(defn- explain [v & opts]
  (let [out (java.io.StringWriter.)]
    (binding [*out* out] (apply pl/explain-pipeline v opts))
    (str out)))

(deftest labels-are-derived-from-the-pass-specs-not-a-parallel-table
  (testing "every pass in forward-passes prints a stage header — a pass cannot silently vanish
            from the explanation because someone forgot to add it to a label table"
    (let [s (explain #'raster.linalg.contract/contract-mm)
          headers (set (map second (re-seq #"--- Stage \d+: ([^\n]+?) ---" s)))
          specs (var-get (resolve 'raster.compiler.pipeline/pass-specs))]
      (doseq [p pl/forward-passes]
        (let [label (or (:label (get specs p)) (name p))]
          (is (contains? headers label) (str "pass " p " has no stage header")))))))

(deftest the-hardware-line-names-the-target-not-the-host
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "explain-pipeline: target device")
    (let [s (explain #'raster.linalg.contract/contract-mm :target-device :ze:0)
          hw (re-find #"Hardware: [^\n]*" s)]
      (is (re-find #"\[:ze:0\]" (str hw)) (str "expected the target in the line, got: " hw))
      (is (not (re-find #"(?i)core\(tm\)|ultra" (str hw))) "must not report the host CPU")
      (testing "and says which numbers the hardware reported vs which came from a table"
        (is (re-find #"provenance: \d+ detected, \d+ catalogued" s))))))

(deftest a-declined-conversion-is-visible-even-though-the-form-did-not-change
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "explain-pipeline: declines")
    (let [s (explain #'raster.linalg.contract/contract-mm :target-device :ze:0)
          stage-section (fn [label]
                          (second (re-find (re-pattern (str "--- Stage \\d+: " label
                                                            " ---([\\s\\S]*?)(?=--- Stage|--- Kernels|\\z)"))
                                           s)))
          segop-lower (stage-section "segop-lower")
          backend (stage-section "backend")]
      (testing "segop-lower records the contraction as a first-class SegContract equation; this
                stage used to print a decline, and its stats must remain visible"
        (is (re-find #":segops-lowered 1" (str segop-lower)))
        (is (not (re-find #"DECLINED a conversion" (str segop-lower)))
            "segop-lower no longer declines"))
      (testing "the backend stage that declines the typed contraction dispatch (symbolic dims) says
                so in its own stats, so the decline is visible where it happens"
        (is (re-find #":typed-contraction-dispatch-declines" (str backend)))
        (is (re-find #"DECLINED a conversion" (str backend))))
      (testing "the kernel section names the leaf, the headline reason, and every leaf that refused"
        (is (re-find #"--- Kernels ---" s))
        (is (re-find #"strategy=:portable-segred" s))
        (is (re-find #"fallback-reason=:symbolic-dims" s))
        (is (re-find #"declined :dpas: :dtype-not-dpas" s))
        (is (re-find #"declined :regtiled: :symbolic-dims" s))))))
