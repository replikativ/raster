(ns raster.compiler.segop-boundary-test
  "SEGOP LOWERING MAY NO LONGER WARN AND RETURN NIL. Device-free.

   `design/compiler-north-star.md` §3.5 names this code directly. Both sites printed
   `WARNING: SegOp lowering failed for <sym>: <msg>` to stderr and returned nil:

     passes/parallel/segop_lower_pass.clj  — par form → SOAC → SegOp
     backend/gpu/opencl_pass.clj           — the on-the-fly SegMap/SegRed door

   The defect is not the warning, it is the CONFLATION. `nil` meant two different things at once:
   \"this binding is an ordinary value, nothing to do\" and \"this is a parallel form the middle end
   cannot represent\". A genuine coverage gap therefore looked exactly like an ordinary binding, and
   the only trace was a line on stderr that no pass, stat or diagnostic could see. In `opencl_pass`
   it is worse: the SegOp path and the legacy generator increment the SAME counter, so nothing
   downstream could say which of two code generators produced a kernel, or why the modern one
   declined.

   Map and reduction are full conversions: an unrepresentable SegOp is an illegal remaining
   operation with no alternate emitter."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.segop-lower-pass :as slp]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.ir.soac :as soac]))

(defn- stats-of [form] (:stats (slp/segop-lower-pass form {})))

(deftest an-ordinary-binding-is-silent
  (testing "most bindings are not par forms. Reporting them as 'declined' would drown the signal —
            absence of a decline must keep meaning 'nothing to lower here'"
    (let [st (stats-of '(let* [a (clojure.core/+ 1 2)] a))]
      (is (zero? (:segops-lowered st)))
      (is (nil? (:segops-declined st)) "no key at all, not an empty vector of nothing"))))

(deftest a-lowerable-par-form-still-lowers
  (testing "the lowering ATTEMPT is unchanged — nothing that lowered before may stop lowering.
            (This caught a real regression: a SOAC node is a RECORD, records satisfy `map?` and are
            always truthy, so a compact or/if-let version misread every SUCCESS as a decline.)"
    (let [st (stats-of '(let* [o (raster.par/map! O i 256 nil (clojure.core/* i 2.0))] o))]
      (is (= 1 (:segops-lowered st)))
      (is (empty? (:segops-declined st))))))

(deftest an-unrepresentable-par-form-becomes-a-diagnostic
  (testing "a par form with no lowering rule is the case that used to vanish onto stderr"
    (let [st (stats-of '(let* [o (raster.par/scatter! O IDX V 256)] o))
          d (first (:segops-declined st))]
      (is (zero? (:segops-lowered st)))
      (is (= 1 (count (:segops-declined st))))
      (testing "…and it names what §3.5 asks for"
        (is (= 'raster.par/scatter! (:op d)) "the operation")
        (is (contains? #{:soac :segop} (:stage d)) "which conversion")
        (is (= :segop (:target-dialect d)) "the target dialect")
        (is (some? (:reason d)) "the missing rule")
        (is (some? (:fallback d)) "and what happens instead — a stated outcome, not an absence")))))

(deftest a-fatal-reason-still-escapes
  (testing "a violated invariant is not a missing lowering rule. Recording one as a conversion
            decline would let the pipeline continue on the legacy path with the bug intact — the
            loud-to-silent trade this change exists to prevent"
    (let [fatal (var-get (requiring-resolve
                          'raster.compiler.passes.parallel.segop-lower-pass/fatal-reasons))]
      (is (contains? fatal :raster/fatal))
      (is (contains? fatal :raster/bug)))))

(deftest implementation-exceptions-are-not-conversion-declines
  (testing "an implementation bug in par→SOAC escapes instead of silently selecting a fallback"
    (with-redefs [soac/par-form->soac
                  (fn [& _] (throw (NullPointerException. "simulated compiler bug")))]
      (is (thrown-with-msg? NullPointerException #"simulated compiler bug"
                            (stats-of '(let* [o (raster.par/map! O i 256 nil i)] o))))))
  (testing "nil and an unsupported SOAC type have distinct diagnostics"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"produced no SOAC node"
                          (soac-lower/lower-soac nil :cpu:0)))
    (try
      (soac-lower/lower-soac nil :cpu:0)
      (is false "nil SOAC must throw")
      (catch clojure.lang.ExceptionInfo e
        (is (= :no-soac-node (:reason (ex-data e))))))))

(deftest the-gpu-door-records-conversion-declines
  (testing "A form the SegOp path handles reports no decline at all."
    (let [op (requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
          r (op '(raster.par/map! O i 8192 nil (clojure.core/* i 2.0)) :dtype :double)]
      (is (= 1 (:ze-maps (:stats r))) "still lowered through the SegOp path")
      (is (empty? (:segop-declined (:stats r))) "so nothing to explain"))
    (testing "and the recorder itself distinguishes decline from invariant violation"
      (let [attempt (var-get (requiring-resolve
                              'raster.compiler.backend.gpu.opencl-pass/segop-attempt))
            stats (atom {})]
        (testing "an ordinary lowering failure is recorded before its caller rejects the op"
          (is (nil? (attempt stats :segmap '(raster.par/map! O i 4 nil x) :double
                             :none
                             (fn [] (throw (ex-info "no rule" {:reason :no-lowering-rule}))))))
          (let [d (first (:segop-declined @stats))]
            (is (= :segmap (:kind d)))
            (is (= :no-lowering-rule (:reason d)))
            (is (= :none (:fallback d)))))
        (testing "a lowering that simply produces nothing is ALSO recorded, not silently dropped"
          (is (nil? (attempt stats :segred '(raster.par/reduce a 0.0 i 4 x) :double
                             :none (fn [] nil))))
          (is (= :lowering-produced-nothing
                 (:reason (last (:segop-declined @stats))))))
        (testing "…while an invariant violation escapes to the caller"
          (is (thrown? clojure.lang.ExceptionInfo
                       (attempt stats :segmap '(raster.par/map! O i 4 nil x) :double :none
                                (fn [] (throw (ex-info "bug" {:reason :raster/bug})))))))
        (testing "…and an unstructured implementation exception is never a decline"
          (is (thrown-with-msg? NullPointerException #"implementation bug"
                                (attempt stats :segmap '(raster.par/map! O i 4 nil x) :double :none
                                         (fn [] (throw (NullPointerException.
                                                        "implementation bug")))))))))))

(deftest map-is-a-full-conversion-with-no-legacy-emitter
  (testing "a missing SegMap rule leaves an illegal op and cannot silently change code generators"
    (with-redefs [soac-lower/lower-soac (fn [& _] nil)]
      (try
        ((requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
         '(raster.par/map! out i n float (clojure.core/aget a i))
         :dtype :float :min-elements 0)
        (is false "an unlowered map must fail full conversion")
        (catch clojure.lang.ExceptionInfo e
          (is (= :illegal-op-remains (:reason (ex-data e))))
          (is (= :segop (:target-dialect (ex-data e))))
          (is (= :none (:fallback (ex-data e))))
          (is (= :none (get-in (ex-data e) [:decline :fallback]))))))))

(deftest reduction-is-a-full-conversion-with-no-legacy-emitter
  (testing "a missing SegRed rule leaves an illegal op and cannot silently change code generators"
    (with-redefs [soac-lower/lower-soac (fn [& _] nil)]
      (try
        ((requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
         '(raster.par/reduce acc 0.0 i n (+ acc (clojure.core/aget a i)))
         :dtype :float :min-elements 0)
        (is false "an unlowered reduction must fail full conversion")
        (catch clojure.lang.ExceptionInfo e
          (is (= :illegal-op-remains (:reason (ex-data e))))
          (is (= :segop (:target-dialect (ex-data e))))
          (is (= :none (:fallback (ex-data e))))
          (is (= :none (get-in (ex-data e) [:decline :fallback]))))))))
