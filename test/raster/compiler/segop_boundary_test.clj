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
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.ir.parallel-program :as program]
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

(deftest scan-crosses-the-boundary-as-one-scheduled-kernel-graph
  (let [r (slp/segop-lower-pass
           '(let* [result (raster.par/scan out acc 0.0 i n double
                                           (+ acc (aget values i)))]
                  result)
           {})
        p (:form r)
        scheduled (program/kernel-graph-for-binding
                   p 'result '(raster.par/scan out acc 0.0 i n double
                                               (+ acc (aget values i))))]
    (is (= 1 (get-in r [:stats :segops-lowered])))
    (is (= 1 (get-in r [:stats :kernel-graphs-lowered])))
    (is (kernel-graph/kernel-graph? scheduled))
    (is (= 3 (count (:nodes scheduled))))
    (is (= [:intra-block :block-scan nil]
           (mapv #(get-in % [:operation :phase]) (:nodes scheduled))))))

(deftest a-general-scan-recurrence-declines-parallel-scheduling
  (let [r (slp/segop-lower-pass
           '(let* [result (raster.par/scan out h 0.0 i n double
                                           (Math/tanh (+ (* w h) (aget values i))))]
                  result)
           {})
        d (first (get-in r [:stats :segops-declined]))]
    (is (zero? (get-in r [:stats :segops-lowered])))
    (is (zero? (get-in r [:stats :kernel-graphs-lowered])))
    (is (= :segop (:stage d)))
    (is (= :scan-not-associative (:reason d)))
    (is (= :backend-relowers-or-uses-specialized-codegen (:fallback d)))))

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

(deftest direct-offset-map-enters-the-typed-boundary-before-compatibility
  (let [source '(raster.par/map! out i n :offset base float (aget x i))
        opts {:target-device :ze:0
              :dtype :float
              :array-types {'out :float 'x :float}
              :scalar-types {'n :long 'base :long}}
        scheduled (with-redefs [soac/par-form->soac
                                (fn [& _]
                                  (throw (AssertionError. "legacy SOAC was constructed")))]
                    (slp/schedule-single-operation 'out source opts))
        operation (first (:operations scheduled))
        emitted ((requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
                 source :dtype :float :device-id :ze:0 :min-elements 0
                 :scalar-types (:scalar-types opts) :array-types (:array-types opts))]
    (is (some? (:algorithm scheduled)))
    (is (= :typed-soac (get-in operation [:algorithm-dialect])))
    (is (= :unique (:write-conflict operation)))
    (let [kernel (first (:kernels emitted))
          kernel-source (:source kernel)]
      (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
      (is (re-find #"out_\[.*base.*rstr_i" kernel-source))
      (is (not (re-find #"inout_result\[idx\] =" kernel-source))
          "a unique scatter must not acquire a second implicit dense store"))))

(deftest direct-single-operation-refuses-to-drop-hoisted-scalar-equations
  (let [source '(raster.par/map! out i (* n stride) float (aget x i))
        opts {:dtype :float
              :array-types {'out :float 'x :float}
              :scalar-types {'n :long 'stride :long}}]
    (try
      (slp/schedule-single-operation 'out source opts)
      (is false "a singleton projection must not discard its typed host prefix")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :direct-operation-requires-program (:reason (ex-data exception))))))))

(deftest direct-mini-program-preserves-hoisted-scalar-equations
  (let [source '(raster.par/gather out x indices n stride)
        {:keys [program operations]}
        (slp/schedule-single-program
         'result source
         {:target-device :ze:0
          :dtype :float
          :array-types {'out :float 'x :float 'indices :int}
          :scalar-types {'n :long 'stride :long}})
        [extent-equation parallel-equation] (:equations program)
        operation (first operations)]
    (is (= 2 (count (:equations program))))
    (is (true? (get-in extent-equation [:attributes :host-only])))
    (is (= (:results extent-equation)
           (filterv (set (:results extent-equation)) (:operands parallel-equation))))
    (is (= (first (:results extent-equation))
           (-> operation :space :dims first :bound)))
    (is (some? (:source program))
        "the direct backend receives executable host control, not a source-free algorithm")))

(deftest direct-backends-consume-the-complete-typed-mini-program
  (let [source '(raster.par/map! out i (* n stride) float (aget x i))
        opts [:dtype :float :min-elements 0
              :array-types {'out :float 'x :float}
              :scalar-types {'n :long 'stride :long}]
        [gpu jvm]
        (with-redefs [soac/par-form->soac
                      (fn [& _]
                        (throw (AssertionError. "legacy SOAC was constructed")))]
          [(apply (requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
                  source :device-id :ze:0 opts)
           (apply (requiring-resolve 'raster.compiler.backend.jvm.par-simd/simd-pass)
                  source opts)])]
    (doseq [{:keys [form stats]} [gpu jvm]]
      (is (= :typed-soac (get-in stats [:direct-scheduling :route])))
      (is (= 1 (get-in stats [:direct-scheduling :typed-scalar-equations])))
      (is (some #(and (symbol? %) (.startsWith (name %) "rstr_extent_"))
                (take-nth 2 (second form)))
          "the hoisted extent remains an executable binding in the backend form"))
    (is (zero? (get-in gpu [:stats :fallback])))
    (is (zero? (get-in jvm [:stats :fallback])))))

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

(deftest direct-map-is-a-full-typed-conversion-with-no-legacy-emitter
  (testing "the direct adapter does not need compatibility SOAC lowering for an admitted map"
    (with-redefs [soac-lower/lower-soac (fn [& _] nil)]
      (let [compiled ((requiring-resolve
                       'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
                      '(raster.par/map! out i n float (clojure.core/aget a i))
                      :dtype :float :min-elements 0)]
        (is (= 1 (get-in compiled [:stats :ze-maps])))
        (is (= 1 (get-in compiled [:stats :segop-relowered])))))))

(deftest direct-reduction-is-a-full-typed-conversion-with-no-legacy-emitter
  (testing "the direct adapter does not need compatibility SOAC lowering for an admitted reduction"
    (with-redefs [soac-lower/lower-soac (fn [& _] nil)]
      (let [compiled ((requiring-resolve
                       'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
                      '(raster.par/reduce acc 0.0 i n
                                          (+ acc (clojure.core/aget a i)))
                      :dtype :float :min-elements 0)]
        (is (= 1 (get-in compiled [:stats :ze-reduces])))
        (is (= 1 (get-in compiled [:stats :segop-relowered])))))))
