(ns raster.compiler.passes.scalar.mem-merge-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.mem-merge :as mem-merge]))

;; ================================================================
;; Allocation detection
;; ================================================================

(deftest detect-allocations
  (testing "Identifies double-array allocations"
    (let [info (#'mem-merge/alloc-info '(double-array 100))]
      (is (some? info))
      (is (= :cpu (:device info)))
      (is (= 100 (:size info)))))

  (testing "Identifies float-array allocations"
    (let [info (#'mem-merge/alloc-info '(float-array n))]
      (is (some? info))
      (is (= 'n (:size info)))))

  (testing "Non-allocation returns nil"
    (is (nil? (#'mem-merge/alloc-info '(+ a b))))
    (is (nil? (#'mem-merge/alloc-info 42)))))

;; ================================================================
;; Liveness analysis
;; ================================================================

(deftest liveness-basic
  (testing "Basic liveness computation"
    (let [pairs [['a '(double-array 10)]
                 ['b '(double-array 10)]
                 ['c '(+ a b)]]
          liveness (#'mem-merge/compute-liveness pairs ['c])]
      ;; 'a' defined at 0, last used at 2 (in (+ a b))
      (is (= 0 (:def (get liveness 'a))))
      (is (= 2 (:last-use (get liveness 'a))))
      ;; 'b' defined at 1, last used at 2
      (is (= 1 (:def (get liveness 'b))))
      (is (= 2 (:last-use (get liveness 'b))))
      ;; 'c' defined at 2, used in body (index 3)
      (is (= 2 (:def (get liveness 'c))))
      (is (= 3 (:last-use (get liveness 'c)))))))

(deftest liveness-non-overlapping
  (testing "Non-overlapping lifetimes detected"
    (let [pairs [['a '(double-array 10)]
                 ['b '(foo a)]               ;; a last used here
                 ['c '(double-array 10)]      ;; c starts after a is dead
                 ['d '(bar c)]]
          liveness (#'mem-merge/compute-liveness pairs ['d])]
      ;; a: [0,1], c: [2,3] — non-overlapping
      (is (not (#'mem-merge/ranges-overlap?
                (get liveness 'a) (get liveness 'c)))))))

;; ================================================================
;; Interference graph
;; ================================================================

(deftest interference-overlapping
  (testing "Overlapping lifetimes create interference"
    (let [pairs [['a '(double-array 10)]
                 ['b '(double-array 10)]
                 ['c '(+ a b)]]
          liveness (#'mem-merge/compute-liveness pairs ['c])
          infos {'a {:type :double-array :size 10 :device :cpu}
                 'b {:type :double-array :size 10 :device :cpu}}
          graph (#'mem-merge/build-interference-graph #{'a 'b} liveness infos)]
      ;; a and b overlap (both alive at index 2)
      (is (contains? (get graph 'a) 'b))
      (is (contains? (get graph 'b) 'a)))))

(deftest interference-non-overlapping
  (testing "Non-overlapping lifetimes have no interference"
    ;; a used only in binding 1, c starts at binding 2
    (let [pairs [[(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 10)]
                 ['b '(foo a)]
                 [(with-meta 'c {:raster.buffer/hoistable true}) '(double-array 10)]
                 ['d '(bar c)]]
          liveness (#'mem-merge/compute-liveness pairs ['d])
          infos {'a {:type :double-array :size 10 :device :cpu}
                 'c {:type :double-array :size 10 :device :cpu}}
          graph (#'mem-merge/build-interference-graph #{'a 'c} liveness infos)]
      (is (not (contains? (get graph 'a #{}) 'c))))))

;; ================================================================
;; Graph coloring
;; ================================================================

(deftest coloring-no-interference
  (testing "Non-interfering buffers get same color"
    (let [graph {'a #{} 'b #{}}
          coloring (#'mem-merge/greedy-color graph)]
      (is (= (get coloring 'a) (get coloring 'b))))))

(deftest coloring-with-interference
  (testing "Interfering buffers get different colors"
    (let [graph {'a #{'b} 'b #{'a}}
          coloring (#'mem-merge/greedy-color graph)]
      (is (not= (get coloring 'a) (get coloring 'b))))))

(deftest coloring-three-nodes
  (testing "3-clique needs 3 colors"
    (let [graph {'a #{'b 'c} 'b #{'a 'c} 'c #{'a 'b}}
          coloring (#'mem-merge/greedy-color graph)]
      (is (= 3 (count (set (vals coloring))))))))

;; ================================================================
;; Full pass: merge-memory-blocks
;; ================================================================

(deftest merge-non-overlapping-buffers
  (testing "Non-overlapping hoistable buffers are merged"
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 10)
                      'b '(foo a)
                      (with-meta 'c {:raster.buffer/hoistable true}) '(double-array 10)
                      'd '(bar c)]
                     'd)
          {:keys [form stats]} (mem-merge/merge-memory-blocks form)]
      ;; Should have merged 2 blocks into 1
      (is (pos? (:bytes-saved stats)))
      (is (< (:colors stats) (:blocks stats))))))

(deftest no-merge-overlapping
  (testing "Overlapping hoistable buffers are not merged"
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 10)
                      (with-meta 'b {:raster.buffer/hoistable true}) '(double-array 10)
                      'c '(+ a b)]  ;; both a and b alive at index 2
                     'c)
          {:keys [stats]} (mem-merge/merge-memory-blocks form)]
      (is (= 0 (:bytes-saved stats))))))

(deftest no-merge-different-types
  (testing "Different array types are not merged"
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 10)
                      'b '(foo a)
                      (with-meta 'c {:raster.buffer/hoistable true}) '(float-array 10)
                      'd '(bar c)]
                     'd)
          {:keys [stats]} (mem-merge/merge-memory-blocks form)]
      ;; Different types can't merge
      (is (= 0 (:bytes-saved stats))))))

(deftest no-merge-different-devices
  (testing "Different device buffers are not merged"
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 10)
                      'b '(foo a)
                      (with-meta 'c {:raster.buffer/hoistable true}) '(double-array 10)
                      'd '(bar c)]
                     'd)
          ;; a on CPU, c on CUDA
          {:keys [stats]} (mem-merge/merge-memory-blocks form
                                                         :device-env {'a :cpu 'c :cuda})]
      (is (= 0 (:bytes-saved stats))))))

(deftest single-buffer-unchanged
  (testing "Single buffer returns unchanged"
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 10)
                      'b '(foo a)]
                     'b)
          {:keys [stats]} (mem-merge/merge-memory-blocks form)]
      (is (= 0 (:bytes-saved stats))))))

(deftest non-hoistable-ignored
  (testing "Non-hoistable buffers are ignored"
    (let [form '(let* [a (double-array 10)
                       b (foo a)
                       c (double-array 10)
                       d (bar c)]
                      d)
          {:keys [stats]} (mem-merge/merge-memory-blocks form)]
      (is (= 0 (:bytes-saved stats))))))

;; ================================================================
;; Semantics preservation: eval before/after merge
;; ================================================================

(deftest merge-preserves-semantics
  (testing "mem-merge preserves eval semantics for non-overlapping buffers"
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 5)
                      '_ '(aset a 0 42.0)
                      'r1 '(aget a 0)
                      (with-meta 'b {:raster.buffer/hoistable true}) '(double-array 5)
                      '_ '(aset b 0 99.0)
                      'r2 '(aget b 0)]
                     '(+ r1 r2))
          ;; Eval original
          result-before (eval form)
          ;; Apply mem-merge
          {:keys [form]} (mem-merge/merge-memory-blocks form)
          result-after (eval form)]
      (is (== 141.0 result-before) "Original should compute 42+99=141")
      (is (== 141.0 result-after) "Merged should also compute 141")))

  (testing "mem-merge preserves semantics when buffers overlap"
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 5)
                      (with-meta 'b {:raster.buffer/hoistable true}) '(double-array 5)
                      '_ '(aset a 0 10.0)
                      '_ '(aset b 0 20.0)
                      'r '(+ (aget a 0) (aget b 0))]
                     'r)
          result-before (eval form)
          {:keys [form]} (mem-merge/merge-memory-blocks form)
          result-after (eval form)]
      (is (== 30.0 result-before))
      (is (== 30.0 result-after) "Overlapping buffers must not be merged")))

  (testing "mem-merge zero-fills reused buffers (accumulation pattern)"
    ;; Buffer 'a' gets written with values, result consumed immediately.
    ;; Then buffer 'b' (non-overlapping) is used with accumulation
    ;; (read-before-write via aget+aset).
    ;; Without zero-fill, 'b' would inherit stale data from 'a'.
    ;; NOTE: use unique binding names (not _) so liveness analysis works correctly
    (let [form (list 'let*
                     [(with-meta 'a {:raster.buffer/hoistable true}) '(double-array 4)
                      (with-meta 'e1 {:raster.effect/effectful true}) '(aset a 0 1.0)
                      (with-meta 'e2 {:raster.effect/effectful true}) '(aset a 1 2.0)
                      (with-meta 'e3 {:raster.effect/effectful true}) '(aset a 2 3.0)
                      (with-meta 'e4 {:raster.effect/effectful true}) '(aset a 3 4.0)
                      ;; consume a fully — after this, a is dead
                      'sum-a '(+ (aget a 0) (aget a 1) (aget a 2) (aget a 3))
                      ;; b starts after a is dead — non-overlapping, will be merged
                      (with-meta 'b {:raster.buffer/hoistable true}) '(double-array 4)
                      ;; Accumulation: read b[i] (should be 0) + value
                      (with-meta 'e5 {:raster.effect/effectful true}) '(aset b 0 (+ (aget b 0) 10.0))
                      (with-meta 'e6 {:raster.effect/effectful true}) '(aset b 1 (+ (aget b 1) 20.0))
                      'sum-b '(+ (aget b 0) (aget b 1))]
                     ;; Only sum-a and sum-b in body (not a or b directly)
                     (list '+ 'sum-a 'sum-b))
          result-before (eval form)
          {:keys [form stats]} (mem-merge/merge-memory-blocks form)
          result-after (eval form)]
      (is (pos? (:bytes-saved stats)) "Should merge non-overlapping buffers")
      (is (== 40.0 result-before) "Original: 1+2+3+4 + 10+20 = 40")
      (is (== 40.0 result-after) "Merged must also be 40 (zero-fill prevents stale data)"))))

;; ================================================================
;; Type-safe zero-fill (regression for the LeNet-f32 [J->[S cast)
;; ================================================================

(deftest zero-fill-type-safety
  ;; A mem-merged long[]/int[] buffer must be zero-filled with (a) a :tag-typed
  ;; array symbol and (b) an UNEVALUATED cast value form. Otherwise the JVM
  ;; bytecode backend resolves Arrays/fill to fill(short[],short) — an untyped
  ;; array + literal-int 0 both stack-type ambiguously toward short[] — emitting
  ;; a `[J -> [S` runtime cast (the LeNet-f32 maxpool-argmax crash). closure.clj
  ;; guards the identical hazard for hoisted buffers.
  (testing "long-array: tagged sym + UNEVALUATED (long 0), never a literal int"
    (let [[head sym val] (#'mem-merge/zero-fill-expr 'buf :long-array)]
      (is (= 'java.util.Arrays/fill head))
      (is (= 'longs (:tag (meta sym))) "array sym carries :tag longs")
      (is (seq? val) "fill value is an unevaluated form, not a collapsed literal")
      (is (= '(long 0) val))))
  (testing "int-array: tagged sym + UNEVALUATED (int 0)"
    (let [[_ sym val] (#'mem-merge/zero-fill-expr 'buf :int-array)]
      (is (= 'ints (:tag (meta sym))))
      (is (seq? val))
      (is (= '(int 0) val))))
  (testing "double/float remain unambiguous (value type alone pins the overload)"
    (let [[_ _ dval] (#'mem-merge/zero-fill-expr 'buf :double-array)
          [_ _ fval] (#'mem-merge/zero-fill-expr 'buf :float-array)]
      (is (= 0.0 dval))
      (is (= '(float 0.0) fval)))))
