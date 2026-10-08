(ns raster.compiler.backend.jvm.par-simd-test
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.string]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as numeric]
            [raster.compiler.pipeline :as pipeline]
            [raster.par :as par]
            [raster.compiler.ir.par :as ir.par]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.backend.jvm.segop-simd :as segop-simd]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.materialize :as materialize]))

;; ================================================================
;; Body analysis tests (segop-simd)
;; ================================================================

(deftm mixed-precision-square-map
  [src :- (Array float) gain :- Double] :- (Array float)
  (let [out (arrays/alloc-like src (arrays/alength src))]
    (par/map! out i (arrays/alength src) float
      (numeric/+ (arrays/aget src i) (float (numeric/* gain gain))))
    out))

(deftest simd-keeps-double-computation-before-float-narrowing
  (let [scalar (pipeline/compile-aot #'mixed-precision-square-map :simd? false)
        vectorized (pipeline/compile-aot #'mixed-precision-square-map :simd? true)]
    ;; Non-multiple lengths exercise vector lanes and the original scalar tail.
    ;; Changed inputs rule out a fixture-specific constant or cached result.
    (doseq [length [65 129] gain [1.00000006 1.00000018]]
      (let [input (float-array (repeat length 0.0))
            expected (float (* (double gain) (double gain)))
            reference (scalar input gain)
            actual (vectorized input gain)]
        (is (every? #(= (Float/floatToRawIntBits expected)
                       (Float/floatToRawIntBits (float %))) reference))
        (is (= (vec reference) (vec actual)))))))

(deftest simd-admission-respects-retained-operation-precision
  (let [typed (fn [tag form] (with-meta form {:raster.type/tag tag}))
        wide (typed 'double '(* gain gain))
        narrow (typed 'float '(+ (aget src i) gain))]
    (is (segop-simd/simd-able? narrow 'i :float)
        "homogeneous typed arithmetic keeps the vector fast path")
    (is (segop-simd/simd-able? wide 'i :double))
    (is (not (segop-simd/simd-able? (list 'float wide) 'i :float))
        "a narrowing cast does not license early narrowing of its computation")
    (is (not (segop-simd/simd-able? (list 'double narrow) 'i :double))
        "a widening cast does not license erasing inner Float rounding")))

(deftest stencil-admission-keeps-the-active-species
  (let [stencil {:space {:dims [{:name 'i :bound 'n}]} :dtype :float
                 :out-sym 'out :inputs '#{a} :cast-fn 'float
                 :radius 1 :boundary :dirichlet}
        body (fn [tag] (with-meta '(+ (aget a i) gain) {:raster.type/tag tag}))]
    (is (some? (par-simd/compile-segstencil (assoc stencil :lambda (body 'float)))))
    (is (nil? (par-simd/compile-segstencil (assoc stencil :lambda (body 'double)))))))

(deftest reduction-admission-checks-the-root-operation
  (let [operation (fn [tag]
                    {:space {:dims [{:name 'i :bound 'n}]} :dtype :float :inputs '#{x}
                     :reduction
                     (reduction/scalar
                      {:accumulator 'acc :neutral (float 0) :dtype :float
                       :result 'out :index 'i
                       :step-result (with-meta '(+ acc (aget x i))
                                      {:raster.type/tag tag})})})]
    (is (some? (segop-simd/compile-segred (operation 'float))))
    (is (nil? (segop-simd/compile-segred (operation 'double))))))

(deftest simd-able-simple-ops
  (testing "Simple arithmetic ops are SIMD-able"
    (is (segop-simd/simd-able? '(+ (aget a i) (aget b i)) 'i))
    (is (segop-simd/simd-able? '(* 2.0 (aget a i)) 'i))
    (is (segop-simd/simd-able? '(Math/sqrt (aget a i)) 'i))
    (is (segop-simd/simd-able? '(Math/fma (aget a i) (aget b i) (aget c i)) 'i))))

(deftest simd-able-nested-ops
  (testing "Nested arithmetic is SIMD-able"
    (is (segop-simd/simd-able?
         '(+ (aget u i) (* dt (aget k i))) 'i))
    (is (segop-simd/simd-able?
         '(Math/exp (+ (aget a i) (* 2.0 (aget b i)))) 'i))))

(deftest simd-able-scalars
  (testing "Scalars are SIMD-able"
    (is (segop-simd/simd-able? 42.0 'i))
    (is (segop-simd/simd-able? 'dt 'i))
    (is (segop-simd/simd-able? '(double dt) 'i))))

(deftest typed-conversion-simd-admission-preserves-precision-boundaries
  (let [conversion (fn [source target rounding overflow operand]
                     (dialect/scalar-convert
                      {:source-dtype source :target-dtype target
                       :rounding rounding :overflow overflow
                       :source-op 'clojure.core/float}
                      operand))]
    (is (segop-simd/simd-able?
         (conversion :float :float :exact :exact '(+ (aget a i) (float 1.0))) 'i :float)
        "an all-f32 materialization boundary remains vectorizable")
    (is (not (segop-simd/simd-able?
              (conversion :float :float :exact :exact '(+ (aget a i) (float 1.0))) 'i :double))
        "a float materialization boundary cannot be erased inside a double Vector species")
    (is (not (segop-simd/simd-able?
              (conversion :double :float :nearest-even :ieee '(+ (aget a i) 1.0)) 'i :float))
        "mixed-width vector conversion declines until lane-shape conversion is explicit")
    (is (segop-simd/simd-able? '(float (+ (aget a i) (float 1.0))) 'i :float))
    (is (not (segop-simd/simd-able?
              '(float (+ (aget a i) (float 1.0))) 'i :double))
        "a source float cast is not erased under a wider active species")))

(deftest not-simd-able
  (testing "Non-SIMD operations return false"
    (is (not (segop-simd/simd-able? '(some-fn (aget a i)) 'i)))))

(deftest simd-able-compare-blend
  (testing "if with comparison is SIMD-able via compare+blend"
    (is (segop-simd/simd-able? '(if (> (aget a i) 0.0) 1.0 0.0) 'i))
    (is (segop-simd/simd-able? '(if (< (aget a i) (aget b i)) (aget a i) (aget b i)) 'i))))

(deftest simd-able-qualified-ops
  (testing "Qualified core ops are SIMD-able"
    (is (segop-simd/simd-able? '(clojure.core/+ (aget a i) (aget b i)) 'i))
    (is (segop-simd/simd-able? '(clojure.core/* 2.0 (aget a i)) 'i))))

;; ================================================================
;; SegOp SIMD compilation tests
;; ================================================================

(defn- all-nodes [form]
  (tree-seq #(or (seq? %) (vector? %)) seq form))

(defn- par->segmap [form]
  (let [info (ir.par/extract-par-map-info form)
        s (soac/par-form->soac (:out info) form 0)]
    (first (soac-lower/lower-soac s :cpu:0))))

(defn- par->segred [form]
  (let [s (soac/par-form->soac 'result form 0)]
    (first (soac-lower/lower-soac s :cpu:0))))

(deftest compile-segmap-simple
  (testing "compile-segmap produces SIMD form for simple body"
    (let [form '(raster.par/map! out i (alength out) double (+ (aget a i) 1.0))
          segmap (par->segmap form)
          result (segop-simd/compile-segmap segmap 'out 'double)]
      (is result "Should produce SIMD form")
      (is (seq? result))
      (is (some #(= 'jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED %)
                (all-nodes result))
          "Should reference SPECIES_PREFERRED")
      (is (some #(and (seq? %) (= 'loop* (first %)))
                (all-nodes result))
          "Should have loop*"))))

(deftest compile-segmap-scalar-ops
  (testing "compile-segmap handles scalar multiplication"
    (let [form '(raster.par/map! out i 100 double (* 2.0 (aget a i)))
          segmap (par->segmap form)]
      (is (segop-simd/compile-segmap segmap 'out 'double)
          "Should produce SIMD form for scalar*array"))))

(deftest compile-segmap-binary
  (testing "compile-segmap handles binary array ops"
    (let [form '(raster.par/map! out i 100 double (+ (aget a i) (aget b i)))
          segmap (par->segmap form)]
      (is (segop-simd/compile-segmap segmap 'out 'double)
          "Should produce SIMD form for array+array"))))

;; ----------------------------------------------------------------------------
;; Regression: same array read at TWO distinct affine bases must NOT collapse to
;; one vector load. Before the load-site fix, vector loads were keyed by array
;; symbol, so (- (aget data (+ ab i)) (aget data (+ bb i))) loaded `data` once and
;; emitted (.sub v v) = 0 — silently wrong (0.0 at dim=16/64, partial elsewhere).
;; ----------------------------------------------------------------------------

(deftest collect-load-sites-distinguishes-bases
  (testing "one array at two affine bases yields two distinct load sites"
    (is (= '[[data ab] [data bb]]
           (segop-simd/collect-load-sites
            '(let [df (- (aget data (+ ab d)) (aget data (+ bb d)))] (* df df))
            'd))))
  (testing "same base appears once (CSE), simple index has nil base"
    (is (= '[[data ab]]
           (segop-simd/collect-load-sites '(* (aget data (+ ab d)) (aget data (+ ab d))) 'd)))
    (is (= '[[a nil] [b nil]]
           (segop-simd/collect-load-sites '(+ (aget a i) (aget b i)) 'i)))))

(defn- from-array-loads [form]
  (filter #(and (seq? %)
                (= 'jdk.incubator.vector.DoubleVector/fromArray (first %)))
          (all-nodes form)))

(deftest compile-segmap-two-bases-distinct-loads
  (testing "par/map! reading one array at two bases emits two distinct loads"
    (let [form '(raster.par/map! out i 100 double
                                 (- (aget data (+ ab i)) (aget data (+ bb i))))
          segmap (par->segmap form)
          result (segop-simd/compile-segmap segmap 'out 'double)
          loads (from-array-loads result)
          offsets (set (map #(nth % 3) loads))]
      (is result "Should produce SIMD form")
      (is (>= (count loads) 2) "Should emit a separate load per base, not collapse to one")
      (is (= 2 (count offsets))
          (str "The two loads must use distinct offset expressions, got: " offsets)))))

(deftest compile-segred-two-bases-distinct-loads
  (testing "par/reduce of (* df df) with df across two bases emits two distinct loads"
    (let [form '(raster.par/reduce acc 0.0 d 100
                                   (let [df (- (aget data (+ ab d)) (aget data (+ bb d)))]
                                     (+ acc (* df df))))
          segred (par->segred form)
          result (segop-simd/compile-segred segred)
          loads (from-array-loads result)
          ;; per accumulator there are 2 distinct-base loads; distinct *offset shapes*
          ;; (modulo the k*lanes term) must reference both ab and bb
          offset-syms (set (mapcat #(filter symbol? (all-nodes (nth % 3))) loads))]
      (is result "Should produce SIMD form")
      (is (contains? offset-syms 'ab) "loads must reference base ab")
      (is (contains? offset-syms 'bb) "loads must reference base bb — not collapse to one"))))

(deftest compile-segmap-fma
  (testing "compile-segmap handles fma"
    (let [form '(raster.par/map! out i 100 double
                                 (Math/fma dt (aget k i) (aget u i)))
          segmap (par->segmap form)]
      (is (segop-simd/compile-segmap segmap 'out 'double)
          "Should produce SIMD form for fma"))))

(deftest compile-segmap-compare-blend
  (testing "compile-segmap handles if+comparison via compare+blend"
    (let [form '(raster.par/map! out i 100 double
                                 (if (> (aget a i) 0.0) 1.0 0.0))
          segmap (par->segmap form)
          result (segop-simd/compile-segmap segmap 'out 'double)]
      (is result "Should produce SIMD form for conditional body")
      (is (some #(= 'jdk.incubator.vector.VectorOperators/GT %)
                (all-nodes result))
          "Should use VectorOperators/GT")
      (is (some #(and (seq? %) (= '.blend (first %)))
                (all-nodes result))
          "Should use .blend"))))

;; ================================================================
;; SegOp SIMD reduce compilation
;; ================================================================

(deftest compile-segred-sum
  (testing "compile-segred handles sum reduction"
    (let [form '(raster.par/reduce acc 0.0 i 100 (+ acc (aget a i)))
          segred (par->segred form)
          result (segop-simd/compile-segred segred)]
      (is result "Should produce SIMD form for sum reduction")
      (is (some #(and (seq? %) (= '.reduceLanes (first %)))
                (all-nodes result))
          "Should use reduceLanes"))))

(deftest compile-segred-dot-product
  (testing "compile-segred handles dot product"
    (let [form '(raster.par/reduce acc 0.0 i 100
                                   (+ acc (* (aget a i) (aget b i))))
          segred (par->segred form)]
      (is (segop-simd/compile-segred segred)
          "Should produce SIMD form for dot product"))))

(deftest compile-segred-qualified-ops
  (testing "compile-segred handles qualified core ops"
    (let [form '(raster.par/reduce acc 0.0 i 100
                                   (clojure.core/+ (double acc)
                                                   (clojure.core/* (clojure.core/aget a i)
                                                                   (clojure.core/aget b i))))
          segred (par->segred form)]
      (is (segop-simd/compile-segred segred)
          "Should produce SIMD form for qualified dot product"))))

;; ================================================================
;; SIMD pass (pipeline integration)
;; ================================================================

(deftest simd-pass-replaces-par-map
  (testing "simd-pass replaces par/map! with SIMD code"
    (let [form '(raster.par/map! out i 100 double (+ (aget a i) 1.0))
          {:keys [form stats]} (par-simd/simd-pass form)]
      (is (not (ir.par/par-map-form? form)) "Par form should be replaced")
      (is (= 1 (:simd-maps stats))))))

(deftest simd-pass-skips-small
  (testing "simd-pass skips small arrays"
    (let [form '(raster.par/map! out i 3 double (+ (aget a i) 1.0))
          {:keys [form stats]} (par-simd/simd-pass form)]
      (is (not (ir.par/par-map-form? form)) "Par form should be expanded")
      (is (= 1 (:skipped-small stats))))))

(deftest simd-pass-compare-blend
  (testing "simd-pass vectorizes conditional bodies via compare+blend"
    (let [form '(raster.par/map! out i 100 double
                                 (if (> (aget a i) 0.0) 1.0 0.0))
          {:keys [form stats]} (par-simd/simd-pass form)]
      (is (not (ir.par/par-map-form? form)) "Par form should be replaced")
      (is (= 1 (:simd-maps stats))))))

(deftest simd-pass-inside-let
  (testing "simd-pass handles par forms inside let"
    (let [form '(let [n (alength a)]
                  (raster.par/map! out i n double (+ (aget a i) 1.0)))
          {:keys [form stats]} (par-simd/simd-pass form)]
      (is (not (some #(and (seq? %) (= 'raster.par/map! (first %)))
                     (tree-seq seq? seq form)))
          "No par forms should remain"))))

(deftest direct-offset-map-uses-the-shared-typed-schedule
  (testing "a direct backend call retains out[base+i] semantics through shared scheduling"
    (let [source '(let* [out (double-array [99.0 99.0 99.0 99.0 99.0 99.0])
                         x (double-array [10.0 11.0 12.0])
                         base 2
                         n 3]
                        (raster.par/map! out i n :offset base double (aget x i)))
          {:keys [form stats]} (par-simd/simd-pass source :min-elements 0 :dtype :double)
          result (eval form)]
      (is (= 1 (:fallback stats))
          "the typed unique-scatter has no JVM vector-store schedule yet")
      (is (= 1 (:segop-relowered stats))
          "the production typed route explicitly declines the missing JVM indexed-store schedule")
      (is (= :offset-map-requires-indexed-store
             (get-in stats [:direct-scheduling :segops-declined 0 :reason])))
      (is (= [99.0 99.0 10.0 11.0 12.0 99.0] (vec result))))))

;; ================================================================
;; From walked deftm body
;; ================================================================

(deftm simd-test-fn [a :- (Array double),
                     b :- (Array double)] :- (Array double)
  (broadcast [a b] (+ a b)))

(deftest simd-from-walked-body
  (testing "SIMD pass works on walked deftm body (after materialize)"
    (let [v (resolve 'raster.compiler.backend.jvm.par-simd-test/simd-test-fn)
          dt (:raster.core/dispatch-table (meta v))
          method (first (first (vals @dt)))
          mangled (symbol (str "simd-test-fn_m_"
                               (clojure.string/join "_" (:tags method))))
          backing-var (ns-resolve 'raster.compiler.backend.jvm.par-simd-test mangled)
          walked (first (:raster.core/deftm-walked-body (meta backing-var)))
          ;; Materialize pure par/pmap → alloc + par/map! before SIMD pass
          {:keys [form]} (materialize/materialize-pass walked {})
          {:keys [stats]} (par-simd/simd-pass form)]
      (is (or (pos? (:simd-maps stats))
              (pos? (:fallback stats)))
          "Should have processed the par form"))))
