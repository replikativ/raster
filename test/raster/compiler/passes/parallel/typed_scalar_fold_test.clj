(ns raster.compiler.passes.parallel.typed-scalar-fold-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.nn :as nn]))

(def ^:private dot-map
  '(let* [y (raster.par/pmap
             row rows float
             (raster.par/reduce
              acc 0.0 col width
              (+ acc
                 (* (clojure.core/aget x (+ (* row width) col))
                    (clojure.core/aget w col)))))]
     y))

(def ^:private ordered-map
  '(let* [y (raster.par/pmap
             row rows float
             (raster.par/reduce
              acc 0.0 col width
              (Math/tanh (+ acc (clojure.core/aget w col)))))]
     y))

(def ^:private options
  {:dtype :float
   :array-types {'x :float 'w :float}
   :scalar-types {'rows :long 'width :long}})

(defn- scalar-fold
  [program]
  (first
   (filter dialect/scalar-fold-form?
           (tree-seq coll? seq (dialect/equations program)))))

(defn- nested-loop-source []
  (let [loop-form
        '(loop* [^{:raster.type/tag long} j 0
                 ^{:raster.type/tag float} acc 0.0]
           (if (< (long j) width)
             (let* [^{:raster.type/tag long} off (* j depth)
                    ^{:raster.type/tag float}
                    dot (loop* [^{:raster.type/tag long} d 0
                                ^{:raster.type/tag float} inner 0.0]
                          (if (< (long d) depth)
                            (recur (inc (long d))
                                   (+ inner (clojure.core/aget x (+ off d))))
                            inner))]
               (recur (inc (long j)) (+ acc dot)))
             acc))]
    (list 'let* (vector 'y (list 'raster.par/pmap 'row 'rows 'float loop-form)) 'y)))

(deftest fold-is-a-scoped-functional-term
  (let [fold '(fold {:accumulator acc :index i :identity 0.0 :dtype :float
                     :extent n :association :ordered}
                    (lambda [acc i]
                      (region [] [(+ acc (clojure.core/aget x (+ base i)))])))]
    (is (= '#{n x base} (util/free-syms fold)))
    (is (= '(raster.par/reduce acc 0.0 i n
                               (+ acc (clojure.core/aget replacement (+ base i))))
           (projection/scalar-folds->source
            (util/subst-syms {'x 'replacement 'i 'must-not-capture} fold))))))

(deftest associative-nested-reduce-carries-a-checked-reassociation-certificate
  (let [program (frontend/form->program dot-map options)
        fold (scalar-fold program)
        {:keys [attributes lambda]} (dialect/scalar-fold-parts fold)
        {:keys [parameters locals body-results]} (dialect/lambda-parts lambda)]
    (is (dialect/program-form? (dialect/validate! program)))
    (is (= :float (:dtype attributes))
        "the enclosing typed map supplies the reduction dtype")
    (is (= :implementation-defined (:association attributes)))
    (is (scan/associative-scan? (:algebra attributes)))
    (is (= '[acc col] parameters))
    (is (empty? locals))
    (is (= 1 (count body-results)))))

(deftest general-nested-recurrence-remains-ordered
  (let [program (frontend/form->program ordered-map options)
        {:keys [attributes]} (dialect/scalar-fold-parts (scalar-fold program))]
    (is (= :ordered (:association attributes)))
    (is (nil? (:algebra attributes)))))

(deftest source-loop-recurrence-canonicalizes-only-for-the-exact-ordered-grammar
  (let [loop-form (with-meta
                    '(loop* [^{:raster.type/tag long} i 0
                             ^{:raster.type/tag float} acc 0.0]
                       (if (< (long i) width)
                         (recur (inc (long i))
                                ^{:raster.type/tag float}
                                (+ acc (clojure.core/aget w i)))
                         acc))
                    {:raster.type/tag 'float})
        fold (#'frontend/canonicalize-scalar-folds loop-form :float)]
    (is (dialect/scalar-fold-form? fold))
    (is (= :ordered (get-in (dialect/scalar-fold-parts fold)
                            [:attributes :association])))
    (is (not-any? #(and (seq? %) (contains? #{'loop 'loop*} (first %)))
                  (tree-seq coll? seq fold))))
  (testing "nonzero origins and transformed exits retain their source spelling"
    (doseq [form ['(loop* [^{:raster.type/tag long} i 1
                            ^{:raster.type/tag float} acc 0.0]
                     (if (< (long i) width)
                       (recur (inc (long i)) (+ acc (clojure.core/aget w i))) acc))
                  '(loop* [^{:raster.type/tag long} i 0
                            ^{:raster.type/tag float} acc 0.0]
                     (if (< (long i) width)
                       (recur (inc (long i)) (+ acc (clojure.core/aget w i)))
                       (float acc)))]]
      (let [result (#'frontend/canonicalize-scalar-folds form :float)]
        (is (not (dialect/scalar-fold-form? result)))
        (is (= form result))))))

(deftest nested-recurrences-use-lexical-fold-locals-through-kernel-body
  (let [source (nested-loop-source)
        options {:dtype :float :array-types {'x :float}
                 :scalar-types {'rows :long 'width :long 'depth :long}}
        program (frontend/form->program source options)
        nodes (tree-seq coll? seq (dialect/equations program))
        folds (filter dialect/scalar-fold-form? nodes)
        routed (route/attempt source :float {'x :float} options)
        scheduled (:form (segop-lower/segop-lower-pass
                          (:program routed) {:dtype :float :target-device :ocl:0}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                         :dtype :float :min-elements 1)
        kernel (first (:kernels emitted))]
    (is (= program (dialect/validate! program)))
    (is (= 2 (count folds)))
    (is (not-any? #(and (seq? %) (contains? #{'loop 'loop*} (first %))) nodes))
    (is (= '[off dot]
           (mapv :id (:locals (dialect/lambda-parts
                               (:lambda (dialect/scalar-fold-parts (first folds))))))))
    (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
    (is (= 2 (count (re-seq #"for \(long [^ ]*fold_index_" (:source kernel)))))))

(deftest jvm-consumes-the-typed-fold-without-compatibility-relowering
  (let [execute
        (fn [source]
          (let [scheduled (:form (pipeline/schedule-parallel-form source options))
                emitted (par-simd/simd-pass scheduled :min-elements 1)
                f (eval (list 'fn '[x w rows width] (:form emitted)))]
            {:result (f (float-array [1 2 3 4 5 6])
                        (float-array [2 3 4]) 2 3)
             :stats (:stats emitted)
             :form (:form emitted)}))
        dot (execute dot-map)
        ordered (execute ordered-map)
        ordered-reference
        (reduce (fn [acc value] (Math/tanh (+ acc value)))
                0.0 [2.0 3.0 4.0])]
    (is (= [20.0 47.0] (mapv double (:result dot))))
    (is (= {:simd-maps 1 :simd-reduces 0 :fallback 0 :fused 0
            :skipped-small 0 :segop-reused 1}
           (:stats dot)))
    (is (< (Math/abs (- ordered-reference
                       (double (first (:result ordered)))))
           1.0e-6))
    (is (not (some #{'jdk.incubator.vector.FloatVector/SPECIES_PREFERRED}
                   (tree-seq coll? seq (:form ordered))))
        "an ordered recurrence must not silently take the reassociating SIMD leaf")))

(deftest realistic-dense-program-has-no-nested-reduction-compatibility-debt
  (let [report (pipeline/compile-report #'nn/predict-fn)
        fast (pipeline/compile-aot #'nn/predict-fn)
        args [(double-array [1.0 0.0 0.0 1.0])
              (double-array [0.0 0.0])
              (double-array [0.5 0.5])
              (double-array [0.25])
              (double-array [1.0 -1.0])]]
    (is (= {:segops 2 :kernel-graphs 0 :structured-loops 0 :typed-reused 2
            :typed-scalar-equations 4 :backend-reused 2
            :backend-relowered 0 :fallback 0}
           (:lowering report)))
    (is (= (vec (apply nn/predict-fn args))
           (vec (apply fast args))))))
