(ns raster.compiler.passes.parallel.typed-scalar-fold-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.gpu.kernel-body-opencl :as body-emit]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.dl.nn :as dl-nn]
            [raster.nn :as nn]))

(deftest product-fold-components-share-one-multi-carry-loop
  (let [product
        '(product-fold {:accumulators [sum squares]
                        :identities [0.0 0.0]
                        :dtypes [:float :float]
                        :index i :lower 0 :extent width :association :ordered}
           (lambda [sum squares i]
             (region [(let-value value :float (clojure.core/aget x i))]
                     [(+ sum value) (+ squares (* value value))])))
        lowerer (scalar-body/make-lowerer
                 {:arrays #{'x} :array-types {'x :float}
                  :scalar-types {'width :long} :index-scope #{}
                  :lower-index (fn [expression _] expression)
                  :decline! (fn [rule message data]
                              (throw (ex-info message (assoc data :rule rule))))})
        lowered ((:lower-region lowerer)
                 {:bindings []
                  :results [(list 'product-component product 0)
                            (list 'product-component product 1)]}
                 [:float :float] {} {'width :long})
        loops (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                      (:operations lowered))
        loop (first loops)]
    (is (= [:float :float] (:types lowered)))
    (is (= 1 (count loops)) "component projections must not duplicate the fold")
    (is (= 2 (count (:iter-args loop))))
    (is (= 2 (count (:results loop))))
    (is (= 2 (count (:values (peek (:operations loop))))))))

(def ^:private product-fold-map
  '(let* [y
          (raster.par/pmap
           row rows float
           (let* [sums
                  (loop* [i 0
                          ^{:raster.type/tag float} sum (float 0.0)
                          ^{:raster.type/tag float} squares (float 0.0)]
                    (if (< (long i) width)
                      (let* [^{:raster.type/tag float} value
                             (clojure.core/aget x (+ (* row width) i))]
                        (recur (inc (long i))
                               ^{:raster.type/tag float} (+ sum value)
                               ^{:raster.type/tag float} (+ squares (* value value))))
                      [sum squares]))
                  ^{:raster.type/tag float} sum-result
                  (clojure.core/nth sums (long 0))
                  ^{:raster.type/tag float} square-result
                  (clojure.core/nth sums (long 1))]
             (+ sum-result square-result)))]
     y))

(def ^:private projected-product-fold-map
  '(let* [y
          (raster.par/pmap
           row rows float
           (let* [^{:raster.type/tag float} total
                  (loop* [i 0
                          ^{:raster.type/tag float} sum 0.0
                          ^{:raster.type/tag float} squares 0.0]
                    (if (< (long i) width)
                      (let* [^{:raster.type/tag float} value
                             (clojure.core/aget x (+ (* row width) i))]
                        (recur (inc (long i))
                               ^{:raster.type/tag float} (+ sum value)
                               ^{:raster.type/tag float} (+ squares (* value value))))
                      ^{:raster.type/tag float} (+ sum squares)))]
             total))]
     y))

(deftest mapped-product-recurrence-reaches-one-gpu-loop
  (let [options {:dtype :float :array-types {'x :float}
                 :scalar-types {'rows :long 'width :long}}
        routed (route/attempt product-fold-map :float (:array-types options) options)
        program (:program routed)
        algorithm (-> program :equations first :algorithm)
        scheduled (segop-lower/segop-lower-pass
                   program {:dtype :float :target-device :ocl:0})
        emitted (opencl-pass/opencl-pass (:form scheduled) :device-id :ocl:0
                                         :dtype :float :min-elements 1)
        artifact (first (:kernels emitted))
        source (:source artifact)
        kernel-body (get-in artifact [:attributes :kernel-body])
        jvm-emitted (par-simd/simd-pass
                     (:form (pipeline/schedule-parallel-form product-fold-map options))
                     :min-elements 1)
        jvm-form (:form jvm-emitted)
        execute (eval (list 'fn '[x rows width] jvm-form))]
    (is (= :typed-soac (get-in routed [:stats :route])))
    (is (= algorithm (dialect/validate! algorithm)))
    (is (= 2 (count (filter dialect/product-fold-form?
                            (tree-seq coll? seq algorithm)))))
    (is (= 1 (count (:kernels emitted))))
    (is (= 1 (count (re-seq #"for \(long [^ ]*product_fold_index_" source)))
        "two component uses must share one emitted multi-carry loop")
    (doseq [target [:cuda :hip]]
      (let [target-source (body-emit/emit-scalar-kernel
                           "product_fold" kernel-body {:target-dialect target})]
        (is (= 1 (count (re-seq #"for \(long(?: long)? [^ ]*product_fold_index_"
                                target-source))))))
    (is (= 2 (count (re-seq #"loop\*" (pr-str jvm-form))))
        "JVM has one outer map loop and one shared product loop")
    (is (= [20.0 92.0]
           (mapv double (execute (float-array [1 2 3 4 5 6]) 2 3))))))

(deftest projected-multi-carry-recurrence-stays-on-the-typed-kernel-body-route
  (let [options {:dtype :float :array-types {'x :float}
                 :scalar-types {'rows :long 'width :long}}
        routed (route/attempt projected-product-fold-map :float (:array-types options) options)
        program (:program routed)
        algorithm (-> program :equations first :algorithm)
        scheduled (segop-lower/segop-lower-pass
                   program {:dtype :float :target-device :ocl:0})
        emitted (opencl-pass/opencl-pass (:form scheduled) :device-id :ocl:0
                                         :dtype :float :min-elements 1)
        artifact (first (:kernels emitted))]
    (is (= :typed-soac (get-in routed [:stats :route])))
    (is (= 2 (count (filter dialect/product-fold-form?
                            (tree-seq coll? seq algorithm)))))
    (is (= 1 (count (:kernels emitted))))
    (is (= :portable-segmap
           (get-in artifact [:attributes :kernel-body :attributes :kind])))
    (is (= 1 (count (re-seq #"for \(long [^ ]*product_fold_index_"
                            (:source artifact)))))))

(deftest public-layer-norm-backward-reuses-the-product-region
  (let [pipeline (pipeline/show-pipeline #'dl-nn/layer-norm-backward-dx
                                         :dtype :float :target-device :ocl:0)
        kernels (:kernels pipeline)
        product-loop-lines
        (mapcat #(filter (fn [line]
                           (and (.contains line "for (")
                                (.contains line "product_fold_index")))
                         (str/split-lines (:source %)))
                kernels)]
    (is (= 2 (get-in pipeline [:segop-lowered-stats :segops-lowered])))
    (is (= 2 (count kernels)) "initialization and the fused backward body remain explicit")
    (is (every? #(= :kernel-body (get-in % [:attributes :emission-route])) kernels))
    (is (= 1 (count product-loop-lines))
        "sum and square-sum uses share one loop inside the generated backward kernel")))

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
        conversion (first
                    (filter dialect/scalar-convert-form?
                            (tree-seq coll? seq (dialect/equations program))))
        {:keys [parameters locals body-results]} (dialect/lambda-parts lambda)]
    (is (dialect/program-form? (dialect/validate! program)))
    (is (= [:double :float]
           ((juxt :source-dtype :target-dtype)
            (:attributes (dialect/scalar-convert-parts conversion))))
        "the map result conversion retains the source recurrence and float storage dtypes")
    (is (= :double (:dtype attributes))
        "the Fold executes at the conversion's source dtype, before float materialization")
    (is (= :implementation-defined (:association attributes)))
    (is (scan/associative-scan? (:algebra attributes)))
    (is (= '[acc col] parameters))
    (is (empty? locals))
    (is (= 1 (count body-results)))))

(deftest canonical-result-conversion-canonicalizes-its-fold-at-the-source-dtype
  (let [attributes {:source-dtype :double :target-dtype :float
                    :rounding :nearest-even :overflow :ieee
                    :source-op 'clojure.core/float}
        conversion (dialect/scalar-convert
                    attributes '(raster.par/reduce acc 0.0 i n
                                                    (+ acc (clojure.core/aget x i))))
        result (#'frontend/canonicalize-scalar-folds conversion :float)
        nested (#'frontend/canonicalize-scalar-folds (list '+ 1.0 conversion) :float)
        {:keys [attributes operand]} (dialect/scalar-convert-parts result)
        nested-conversion (first (filter dialect/scalar-convert-form?
                                         (tree-seq coll? seq nested)))
        nested-fold (:operand (dialect/scalar-convert-parts nested-conversion))]
    (is (= {:source-dtype :double :target-dtype :float
            :rounding :nearest-even :overflow :ieee
            :source-op 'clojure.core/float}
           attributes))
    (is (dialect/scalar-fold-form? operand))
    (is (= :double (get-in (dialect/scalar-fold-parts operand) [:attributes :dtype])))
    (is (dialect/scalar-fold-form? nested-fold)
        "a conversion nested under arithmetic protects its recurrence from the outer dtype")
    (is (= :double (get-in (dialect/scalar-fold-parts nested-fold) [:attributes :dtype])))
    (is (= :float
           (get-in (dialect/scalar-convert-parts result) [:attributes :target-dtype])))))

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
  (testing "an inclusive upper bound is retained without overflow-prone bound arithmetic"
    (let [form '(loop* [^{:raster.type/tag long} i 0
                        ^{:raster.type/tag float} acc 0.0]
                  (if (<= (long i) limit)
                    (let* [^{:raster.type/tag float} value (clojure.core/aget w i)]
                      (recur (inc (long i)) (+ acc value)))
                    acc))
          fold (#'frontend/canonicalize-scalar-folds form :float)
          attributes (:attributes (dialect/scalar-fold-parts fold))
          projected (projection/scalar-folds->source fold)]
      (is (= :inclusive (:upper-bound attributes)))
      (is (= 'loop* (first projected)))
      (is (= 'clojure.core/<= (first (second (nth projected 2)))))))
  (testing "a dynamic lower bound remains an explicit part of the Fold domain"
    (let [form '(loop* [^{:raster.type/tag long} i start
                        ^{:raster.type/tag float} acc 0.0]
                  (if (< (long i) width)
                    (recur (inc (long i)) (+ acc (clojure.core/aget w i)))
                    acc))
          fold (#'frontend/canonicalize-scalar-folds form :float)]
      (is (= 'start (get-in (dialect/scalar-fold-parts fold) [:attributes :lower])))
      (is (= 'start (second (second (projection/scalar-folds->source fold)))))))
  (testing "non-integral origins and transformed exits retain their source spelling"
    (doseq [form ['(loop* [^{:raster.type/tag long} i 0.5
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

(deftest ordered-fold-may-start-from-an-earlier-typed-local
  (let [source
        '(let* [y
                (raster.par/pmap
                 row rows float
                 (let* [^{:raster.type/tag float} negative-limit -1.0e38]
                   (loop* [^{:raster.type/tag long} i 0
                           ^{:raster.type/tag float} acc negative-limit]
                     (if (< (long i) width)
                       (recur (inc (long i))
                              (+ acc (clojure.core/aget w i)))
                       acc))))]
           y)
        routed (route/attempt source :float (:array-types options) options)
        program (-> routed :program :equations first :algorithm)
        fold (scalar-fold program)
        broken
        (util/postwalk-preserving-meta
         (fn [form]
           (if (dialect/scalar-fold-form? form)
             (let [{:keys [attributes lambda]} (dialect/scalar-fold-parts form)]
               (util/remake form 'fold (assoc attributes :identity 'missing-identity) lambda))
             form))
         program)
        scheduled (:form (segop-lower/segop-lower-pass
                          (:program routed) {:dtype :float :target-device :ocl:0}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                         :dtype :float :min-elements 1)]
    (is (= :typed-soac (get-in routed [:stats :route])))
    (is (= program (dialect/validate! program)))
    (is (dialect/scalar-fold-form? fold))
    (is (symbol? (get-in (dialect/scalar-fold-parts fold) [:attributes :identity])))
    (is (= -1.0e38
           (-> program dialect/equations first dialect/operation-parts :lambda
               dialect/lambda-parts :locals first :init)))
    (is (= [:kernel-body]
           (mapv #(get-in % [:attributes :emission-route]) (:kernels emitted))))
    (is (thrown? clojure.lang.ExceptionInfo (dialect/validate! broken)))))

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
