(ns raster.compiler.passes.parallel.typed-soac-frontend-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.par]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.par :as ir-par]
            [raster.compiler.ir.soac :as legacy-soac]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.patterns :as patterns]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(deftest generated-row-major-coordinates-carry-their-domain-type
  (let [matched (patterns/match-nested-dotimes-row-major-map
                 '(dotimes [i rows]
                    (dotimes [j cols]
                      (aset out (+ (* i cols) j) 1.0))))
        locals (take-nth 2 (second (:value-expr matched)))]
    (is (some? matched))
    (is (= ['long 'long] (mapv #(get (meta %) :raster.type/tag) locals)))
    (is (some? (#'frontend/typed-map-region (:value-expr matched))))))

(deftest primitive-array-clones-normalize-to-fresh-identity-maps
  (let [source '(let* [result (clojure.core/aclone input)] result)
        options {:dtype :float :array-types {'input :float}}
        normalized (frontend/normalize-source source options)]
    (is (some #{'raster.arrays/alloc-like} (flatten normalized)))
    (is (some #{'raster.par/map!} (flatten normalized)))
    (is (not-any? #{'clojure.core/aclone} (flatten normalized)))
    (is (= normalized (frontend/normalize-source normalized options))
        "copy normalization is stable across compiler fixpoints")
    (is (= source (frontend/normalize-source source {:dtype :float}))
        "an untyped/object clone does not acquire a guessed device representation")))

(deftest direct-allocation-lengths-are-partially-evaluated-before-shape-analysis
  (let [source '(let* [buffer (clojure.core/float-array
                               (clojure.core/alength (raster.arrays/zeros-like input n)))
                       effect (raster.par/map! buffer i n float
                                               (clojure.core/aget input i))]
                      effect)
        normalized (frontend/normalize-source
                    source {:dtype :float :array-types {'input :float}
                            :scalar-types {'n :long}})]
    (is (some #{'(clojure.core/float-array n)}
              (tree-seq coll? seq normalized)))
    (is (not-any? #(and (seq? %) (= 'clojure.core/alength (first %)))
                  (tree-seq coll? seq normalized)))))

(deftest map-let-spines-become-typed-locals
  (let [body '(let* [^float p1 (* (aget x i) (aget x i))
                    ^float p2 (* p1 p1)
                    ^float p3 (* p2 p2)] p3)
        source (list 'let* ['y (list 'raster.par/pmap 'i 'n 'float body)] 'y)
        program (frontend/form->program source
                                        {:dtype :float :array-types {'x :float}
                                         :scalar-types {'n :long}})
        parts (dialect/operation-parts (first (dialect/equations program)))
        region (dialect/lambda-parts (:lambda parts))]
    (is (some? program))
    (is (= 3 (count (:locals region))))
    (is (= '[p1 p2 p3] (mapv :id (:locals region))))))

(deftest map-let-spines-require-complete-scalar-type-evidence
  (doseq [body ['(let* [p (* x x)] p)
                '(let* [^float p (* x x) ^float p (* p p)] p)
                '(let* [^float p (* x x)] (observe! p) p)
                '(let* [^floats p x] p)]]
    (is (nil? (#'frontend/typed-map-region body)))))

(deftest counted-store-loops-use-the-existing-effect-dialect
  (let [source '(let* [result (dotimes [i n]
                               (aset out i 1.0)
                               (aset scratch i 2.0))] result)
        options {:dtype :double :array-types {'out :double 'scratch :double}
                 :scalar-types {'n :long}}
        normalized (frontend/normalize-source source options)
        bindings (second normalized)
        program (frontend/form->program normalized options)
        run (fn [form n]
              (let [out (double-array 3) scratch (double-array 3)
                    f (eval (list 'fn '[n out scratch] form))]
                (try
                  [(f n out scratch) (vec out) (vec scratch)]
                  (catch Exception e [(class e) (vec out) (vec scratch)]))))]
    (is (some? program))
    (is (= 'long (:tag (meta (first bindings)))))
    (is (= 'raster.par/map-void! (first (last bindings))))
    (is (= normalized (frontend/normalize-source normalized options)))
    (doseq [n [Long/MIN_VALUE -1 0 1 3 2147483648 Long/MAX_VALUE]]
      (is (= (run source n) (run normalized n)) (str "count " n)))))

(deftest rectangular-dotimes-nests-linearize-before-independent-ownership
  (let [source '(let* [effect
                       (dotimes [row rows]
                         (dotimes [column columns]
                           (if (clojure.core/< column visible)
                             (clojure.core/aset out
                                                (clojure.core/+ (clojure.core/* row columns)
                                                                column)
                                                (double (clojure.core/+ row column))))))]
                      effect)
        options {:dtype :double :array-types {'out :double}
                 :scalar-types {'rows :long 'columns :long 'visible :long}}
        normalized (frontend/normalize-source source options)
        operation (last (take-nth 2 (rest (second normalized))))
        program (frontend/form->program normalized options)
        equation (last (dialect/equations program))
        run (fn [form rows columns visible]
              (let [out (double-array (max 0 (* (max 0 rows) (max 0 columns))))
                    f (eval (list 'fn '[rows columns visible out] form))]
                [(f rows columns visible out) (vec out)]))]
    (is (= 'raster.par/map-void! (first operation)))
    (is (some? program))
    (is (= 'scatter (dialect/operation-kind equation))
        "the ordinary mixed-radix proof, not the source rewrite, certifies unique writes")
    (doseq [[rows columns visible] [[2 3 2] [1 4 8] [0 3 2] [-1 3 2] [2 -3 2]]]
      (is (= (run source rows columns visible)
             (run normalized rows columns visible))
          (str "shape " [rows columns visible])))))

(deftest counted-row-maps-flatten-into-one-typed-dense-domain
  (let [source '(let* [effect
                       (dotimes [row rows]
                         (raster.par/map! out column columns
                                          :offset (* row columns) double
                                          (+ (aget out (+ (* row columns) column))
                                             (+ row column))))]
                      out)
        options {:dtype :double :array-types {'out :double}
                 :scalar-types {'rows :long 'columns :long}}
        normalized (frontend/normalize-source source options)
        map-form (some #(when (and (seq? %) (= 'raster.par/map! (first %))) %)
                       (take-nth 2 (rest (second normalized))))
        result (route/attempt source :double {'out :double}
                              {:scalar-types (:scalar-types options)})
        original-fn (eval (list 'fn '[rows columns out] (ir-par/expand-par-forms source)))
        executable-normalized
        (walk/postwalk #(if (symbol? %) (with-meta % nil) %) normalized)
        normalized-fn (eval (list 'fn '[rows columns out]
                                  (ir-par/expand-par-forms executable-normalized)))]
    (is (some? map-form))
    (is (not-any? #{'dotimes} (flatten normalized)))
    (is (= :typed-soac (get-in result [:program :dialect])))
    (is (:typed-validated (:stats result)))
    (is (= 1 (count (filter #(and (seq? (:source %))
                                  (= 'raster.par/map! (first (:source %))))
                            (get-in result [:program :equations]))))
        "the host no longer launches one row kernel at a time")
    (is (not-any? #{'(+ (* row columns) column)} (tree-seq coll? seq map-form))
        "the proved row-major coordinate is the dense lane for inout ownership")
    (doseq [[rows columns] [[2 3] [1 4] [0 3] [-1 3] [2 -3]]]
      (let [size (* (max 0 rows) (max 0 columns))
            left (double-array (repeat size 1.0))
            right (double-array (repeat size 1.0))]
        (is (= (vec (original-fn rows columns left))
               (vec (normalized-fn rows columns right)))
            (str "shape " [rows columns])))))
  (let [options {:dtype :double :array-types {'out :double}
                 :scalar-types {'rows :long 'columns :long}}
        wrong-offset '(let* [effect
                             (dotimes [row rows]
                               (raster.par/map! out column columns
                                                :offset (+ (* row columns) 1) double 1.0))]
                            out)
        dependent '(let* [effect
                          (dotimes [row rows]
                            (raster.par/map! out column row
                                             :offset (* row row) double 1.0))]
                         out)]
    (doseq [source [wrong-offset dependent]]
      (is (some #{'dotimes} (flatten (frontend/normalize-source source options)))
          "non-rectangular or non-row-major maps remain explicit host control"))))

(deftest irregular-loop-nests-remain-source-control
  (let [options {:dtype :double :array-types {'out :double}
                 :scalar-types {'rows :long 'columns :long}}
        dependent '(let* [effect (dotimes [row rows]
                                   (dotimes [column (clojure.core/long row)]
                                     (clojure.core/aset out column 1.0)))]
                         effect)
        imperfect '(let* [effect (dotimes [row rows]
                                   (clojure.core/aset out row 1.0)
                                   (dotimes [column columns]
                                     (clojure.core/aset out column 2.0)))]
                         effect)]
    (doseq [source [dependent imperfect]]
      (let [loop-expression (nth (second source) 1)]
        (is (nil? (#'frontend/flatten-rectangular-dotimes loop-expression))
            "a nonrectangular source loop must not acquire a flattened-domain certificate")
        ;; Later structured-control/effect analysis remains free to support it sequentially.
        (frontend/normalize-source source options)))))

(deftest three-dimensional-row-major-source-enters-one-typed-equation
  (let [source '(let* [^long row-width (clojure.core/* channels kernel)
                       effect
                       (dotimes [position positions]
                         (dotimes [channel channels]
                           (dotimes [tap kernel]
                             (clojure.core/aset
                              patches
                              (clojure.core/+ (clojure.core/* position row-width)
                                              (clojure.core/* channel kernel) tap)
                              (float (clojure.core/aget
                                      input
                                      (clojure.core/+ (clojure.core/* position channels)
                                                      channel)))))))]
                      effect)
        options {:dtype :float :array-types {'input :float 'patches :float}
                 :scalar-types {'positions :long 'channels :long 'kernel :long}}
        normalized (frontend/normalize-source source options)
        program (frontend/form->program normalized options)
        equation (last (dialect/equations program))]
    (is (some? program))
    (is (= 'scatter (dialect/operation-kind equation)))
    (is (= :unique (get-in (dialect/operation-parts equation) [:attributes :conflict])))
    (is (= 1 (count (filter #(= 'scatter (dialect/operation-kind %))
                            (dialect/equations program)))))))

(deftest rectangular-axis-permutations-keep-derived-dimensions-opaque
  (let [source '(let* [^long l-out
                       (clojure.core/+ 1 (clojure.core/quot
                                          (clojure.core/- length kernel) stride))
                       ^long col-cols (clojure.core/* batch l-out)
                       effect
                       (dotimes [b batch]
                         (dotimes [c channels]
                           (dotimes [k kernel]
                             (dotimes [p l-out]
                               (clojure.core/aset
                                cols
                                (clojure.core/+
                                 (clojure.core/*
                                  (clojure.core/+ (clojure.core/* c kernel) k)
                                  col-cols)
                                 (clojure.core/+ (clojure.core/* b l-out) p))
                                (float p))))))]
                      effect)
        options {:dtype :float :array-types {'cols :float}
                 :scalar-types {'batch :long 'channels :long 'length :long
                                'kernel :long 'stride :long}}
        normalized (frontend/normalize-source source options)
        program (frontend/form->program normalized options)
        equation (last (dialect/equations program))]
    (is (some? program))
    (is (= 'scatter (dialect/operation-kind equation))
        "a bijective axis permutation is a unique scatter, not an ordered effect loop")
    (is (= :unique (get-in (dialect/operation-parts equation) [:attributes :conflict])))))

(deftest index-proof-canonicalizes-typed-products-under-source-casts
  (let [product (with-meta
                  '(.invk raster.numeric/_star__m_long_long-impl kh kw)
                  {:raster.op/original 'raster.numeric/* :raster.type/tag 'long})]
    (is (= '(int (clojure.core/* kh kw))
           (#'frontend/canonical-index-arithmetic (list 'int product))))))

(deftest checked-counts-and-prefix-scalars-retain-one-ordered-evaluation
  (let [options {:dtype :double :array-types {'out :double}
                 :scalar-types {'n :long}}
        normalized (frontend/normalize-source
                    '(let* [result (dotimes [i (int n)] (aset out i 1.0))] result)
                    options)
        program (frontend/form->program normalized options)
        equations (dialect/equations program)]
    (is (= 'raster.par/map-void! (first (last (second normalized))))
        "a checked count is retained while the store loop enters the effect dialect")
    (is (= '[scalar scalar map]
           (mapv dialect/operation-kind equations))
        "the converted count and nonnegative launch extent precede the lowered store map")
    (is (= [:long :long]
           (mapv #(-> % dialect/operation-parts :attributes :dtypes first)
                 (take 2 equations)))
        "both the widened count and clamped extent retain their long ABI")
    (is (= [true true true]
           (mapv #(get-in (dialect/facts program)
                          [:equations (second %) :attributes :source-prefix?])
                 equations))
        "equation facts retain the frontend's source-order prefix proof")
    (let [widened-result (-> equations first dialect/operation-parts :lambda
                             dialect/lambda-parts :body-results first)]
      (is (= 'clojure.core/long (first widened-result)))
      (is (= '(int %capture0) (second widened-result))
          "the outer exact widening does not erase the inner checked narrowing")
      (is (= 'long
             (or (types/sym-type-tag widened-result)
                 (some-> widened-result first
                         descriptor/cast-result-tag)))
          "the scalar result owns long through retained metadata or its authoritative cast")))
  (let [options {:dtype :double :array-types {'out :double}
                 :scalar-types {'n :long}}
        source '(let* [checked (clojure.core/int n)
                       result (raster.par/map! out i n double 1.0)]
                      result)
        equations (some-> source (frontend/form->program options) dialect/equations)]
    (is (= '[scalar map] (mapv dialect/operation-kind equations))
        "an otherwise-unused checked prefix remains an ordered scalar equation")
    (is (= :int (-> equations first dialect/operation-parts :attributes :dtypes first))
        "the explicit cast descriptor, not the program default, types its result")))

(deftest checked-scalars-after-numerical-work-decline-this-route
  (let [options {:dtype :double :array-types {'out :double}
                 :scalar-types {'n :long}}
        source '(let* [first-result (raster.par/map! out i n double 1.0)
                       checked (clojure.core/int n)]
                      first-result)]
    (is (nil? (frontend/form->program source options))
        "host-only staging cannot move a later throw before an independent device write")))

(deftest certified-checked-scalars-after-host-allocation-bound-a-following-typed-island
  (let [source '(let* [out (double-array (long n))
                       checked (clojure.core/int n)
                       result (raster.par/map! out i checked double 1.0)]
                      result)
        result (route/attempt source :double {} {:scalar-types {'n :long}})
        program (:program result)
        source-symbols (mapv first (partition 2 (second (:source program))))]
    (is (= :typed-soac (:dialect program)))
    (is (= [1] (get-in program [:attributes :host-binding-ids]))
        "the checked cast stays a host boundary rather than becoming a hoisted scalar equation")
    (is (= '[out checked result] source-symbols)
        "allocation, checked cast, and device operation retain their source order")
    (is (= :int (get-in program [:values 'checked :dtype]))
        "the following island consumes the retained source ABI, not an extent-derived guess")
    (is (= 1 (count (:equations program))))
    (is (:typed-validated (:stats result)))))

(deftest retained-local-types-prove-generated-identity-casts-after-a-map
  (let [source '(let* [first-result (raster.par/map! tmp i n long 1)
                       ^long rows (* n n)
                       ^long capacity (* (long rows) (long n))
                       result (raster.par/map! out j n long capacity)]
                      result)
        options {:dtype :long :array-types {'tmp :long 'out :long}
                 :scalar-types {'n :long}}
        program (frontend/form->program (frontend/normalize-source source options) options)
        capacity-equation (some #(when (= 'capacity (first (nth % 2))) %)
                                (dialect/equations program))]
    (is (some? program))
    (is (= 'scalar (dialect/operation-kind capacity-equation)))
    (is (false? (get-in (dialect/facts program)
                        [:equations (second capacity-equation) :attributes :source-prefix?]))
        "generated long identity casts do not become a false late checked-scalar obligation")))

(deftest counted-conflicting-stores-retain-order-and-explicit-buffer-return
  (let [options {:dtype :double :array-types {'out :double} :scalar-types {'n :long}}
        source '(let* [r (dotimes [i n] (aset out 0 (double i)))] out)
        program (frontend/form->program (frontend/normalize-source source options) options)
        equation (last (dialect/equations program))
        result (first (nth equation 2))
        renamed (dialect/remap-values program {'out [:argument 0] result [:result 0]})]
    (is (= 'effect-map (dialect/operation-kind equation)))
    (is (= :sequential (:iteration-order (:attributes (dialect/operation-parts equation)))))
    (is (= [result] (dialect/outputs program)))
    (is (= '[(extent out)] (get-in (dialect/facts program) [:values result :shape])))
    (is (= '[(extent [:argument 0])]
           (get-in (dialect/facts renamed) [:values [:result 0] :shape])))
    (is (= [[:argument 0]]
           (dialect/physical-results renamed (last (dialect/equations renamed)))))))

(deftest counted-stores-preserve-source-rounding-before-destination-conversion
  (let [options {:dtype :double :array-types {'out :double 'x :double}
                 :scalar-types {'n :long}}
        source '(let* [r (dotimes [i n] (aset out 0 (float (aget x i))))] out)
        program (frontend/form->program (frontend/normalize-source source options) options)
        equation (last (dialect/equations program))
        operation (dialect/operation-parts equation)]
    (is (some? program))
    (is (= [:double] (get-in operation [:attributes :dtypes]))
        "the typed destination boundary still owns the final double conversion")
    (is (some #(and (seq? %) (= 'float (first %)))
              (tree-seq coll? seq equation))
        "the source-rounded float remains before the typed destination conversion")))

(deftest unknown-or-mutable-counts-are-not-normalized
  (doseq [bound ['(next-count!) '(aget counts 0)]]
    (let [source (list 'let* ['r (list 'dotimes ['i bound] '(aset out i 1.0))] 'r)]
      (is (= source (frontend/normalize-source source
                                              {:array-types {'out :double 'counts :long}}))))))

(def ^:private source
  '(let* [^long n (clojure.core/alength x)
          y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
          total (raster.par/reduce acc 0.0 j n (+ acc (clojure.core/aget y j)))]
         total))

(deftest scalar-metadata-captures-retain-shape-witness-arrays
  (let [program
        (frontend/form->program
         '(let* [^long cols (clojure.core/alength x)
                 result (raster.par/pmap j cols double (clojure.core/aget y j))]
            result)
         {:dtype :double
          :array-types {'x :double 'y :double}
          :scalar-types {'cols :long}})
        values (:values (dialect/facts program))]
    (is (some? program))
    (is (= ['scalar 'map] (mapv dialect/operation-kind (dialect/equations program))))
    (is (= #{'x 'y} (set (:inputs (dialect/facts program)))))
    (is (= :double (get-in values ['x :dtype])))
    (is (= '[(unknown-dimension x)] (get-in values ['x :shape])))
    (is (= {:dtype :long :shape []}
           (select-keys (get values 'cols) [:dtype :shape])))))

(deftest explicit-shape-witness-values-refine-their-array-declarations
  (let [source '(let* [^long cols (clojure.core/alength x)
                       result (raster.par/pmap j cols float (clojure.core/aget y j))]
                  result)
        exact (av/tensor {:dtype :double :shape [3]
                          :representation {:kind :strided :stride 2}
                          :memory-space :global})
        options {:dtype :float
                 :array-types {'x :double 'y :float}
                 :scalar-types {'cols :long}
                 :values {'x exact}}
        program (frontend/form->program source options)
        conflict-reason
        (fn [value]
          (try
            (frontend/form->program source (assoc options :values {'x value}))
            nil
            (catch clojure.lang.ExceptionInfo exception
              (:reason (ex-data exception)))))]
    (is (= exact (get-in (dialect/facts program) [:values 'x]))
        "compatible exact shape and representation facts survive unchanged")
    (let [multidimensional (av/tensor {:dtype :double :shape [1 3]})
          result (frontend/form->program source
                                         (assoc options :values {'x multidimensional}))]
      (is (some? result) "a source array may back a multidimensional logical tensor")
      (is (= multidimensional (get-in (dialect/facts result) [:values 'x]))))
    (is (= :source-value-conflict
           (conflict-reason (av/tensor {:dtype :float :shape [3]})))
        "an explicit value may not override the declared element dtype")
    (is (= :source-value-conflict
           (conflict-reason (av/tensor {:dtype :double :shape []})))
        "an explicit scalar may not replace a declared array")))

(deftest physical-allocation-initialization-survives-host-scaffolding
  (let [contracts
        (fn [allocation]
          (let [source (list 'let* ['output allocation
                                   'written '(raster.par/map! output i n nil (aget input i))]
                             'written)
                program (frontend/form->program
                         source {:dtype :double :array-types {'input :float 'output :float}
                                 :scalar-types {'n :long}})]
            (is (some? program))
            (get-in (dialect/facts program) [:attributes :allocations])))]
    (doseq [[allocation initialization]
            [['(clojure.core/float-array n) :zero]
             ['(raster.math/zeros-like input n) :zero]
             ['(raster.math/alloc-like input n) :unspecified]
             ['(clojure.core/aclone input) :copy]]]
      (is (= [{:destination 'output :source-binding-id 0
               :extent 'n
               :initialization initialization :dtype :float}]
             (contracts allocation))))
    (is (empty? (contracts '(clojure.core/float-array n 7))))
    (is (empty? (contracts '(clojure.core/float-array [1.0 2.0]))))))

(deftest returned-buffer-identity-is-normalized-before-access-contracts
  (let [source '(let* [r (raster.par/contract C [[i 4]] [] (clojure.core/aget A i))
                       alias r
                       mapped (raster.par/map! C j 4 nil (clojure.core/aget alias j))
                       closure (fn [C] r)
                       data (quote r)]
                 mapped)
        normalized (frontend/normalize-source source {:array-types {'A :float 'C :float}})
        bindings (into {} (map vec (partition 2 (second normalized))))]
    (is (= 'raster.par/contract (first (get bindings 'r))) "producer effect stays")
    (is (= 'C (get bindings 'alias)))
    (is (= '(clojure.core/aget C j) (last (get bindings 'mapped))))
    (is (= #{'C} (util/free-syms (get bindings 'closure))) "replacement avoids capture")
    (is (= '(quote r) (get bindings 'data)))
    (is (= 'mapped (last normalized)))))

(deftest same-return-type-is-not-a-buffer-identity-proof
  (let [source '(let* [r (raster.par/reduce acc A i 1 B)] r)
        normalized (frontend/normalize-source source {:array-types {'A :float 'B :float}})]
    (is (= 'r (last normalized)))))

(deftest returned-buffer-alias-survives-an-incoming-name-being-shadowed
  (let [source '(let* [r (raster.par/contract C [[i 4]] [] (clojure.core/aget A i))
                       C B
                       mapped (raster.par/map! D j 4 nil (clojure.core/aget r j))]
                 r)
        normalized (frontend/normalize-source source {:array-types {'A :float 'B :float
                                                                    'C :float 'D :float}})
        pairs (mapv vec (partition 2 (second normalized)))
        renamed (first (second pairs))
        consumer (second (last pairs))]
    (is (not= 'C renamed))
    (is (= 'B (second (second pairs))))
    (is (= '(clojure.core/aget C j) (last consumer)))
    (is (= 'r (last normalized)))))

(deftest physical-reads-consume-the-preceding-logical-writer
  (let [options {:dtype :float :array-types {'A :float 'B :float 'C :float 'D :float}}
        outputs (fn [source]
                  (-> source (frontend/normalize-source options)
                      (frontend/form->program options) dialect/outputs set))]
    (is (= '#{consumed last-write}
           (outputs '(let* [first-write (raster.par/map! C i 4 nil (aget A i))
                            consumed (raster.par/map! D j 4 nil (aget first-write j))
                            last-write (raster.par/map! C k 4 nil (aget B k))]
                       last-write)))
        "a read before overwrite consumes the first writer, not the final writer")
    (is (= '#{last-write}
           (outputs '(let* [first-write (raster.par/map! C i 4 nil (aget A i))
                            last-write (raster.par/map! C j 4 nil (aget first-write j))]
                       last-write)))
        "inout reads are processed before registering their own write")
    (is (= '#{first-write last-write}
           (outputs '(let* [first-write (raster.par/map! C i 4 nil (aget A i))
                            observed (observe first-write)
                            last-write (raster.par/map! C j 4 nil (aget first-write j))]
                       last-write)))
        "an opaque host read keeps its preceding producer externally visible")))

(deftest direct-front-end-builds-the-typed-map-reduction-program
  (let [direct (frontend/form->program source {:dtype :float :array-types {'x :float}})
        equations (dialect/equations direct)]
    (is (= '[scalar map reduce] (mapv dialect/operation-kind equations)))
    (is (= '[x] (:inputs (dialect/facts direct))))
    (is (= '[total] (dialect/outputs direct)))
    (is (= :float (:dtype (get-in (dialect/facts direct) [:values 'y]))))
    (is (= :double (:dtype (get-in (dialect/facts direct) [:values 'total]))))
    (is (= :analyzed-source (get-in (dialect/facts direct) [:provenance :front-end])))
    (is (every? #(contains? (:provenance %) :source-binding-id)
                (vals (:equations (dialect/facts direct)))))))

(deftest production-route-does-not-load-the-record-adapter-as-its-front-door
  (let [aliases (ns-aliases 'raster.compiler.passes.parallel.typed-soac-route)]
    (is (= 'raster.compiler.passes.parallel.typed-soac-frontend
           (ns-name (get aliases 'frontend))))
    (is (not (contains? aliases 'legacy)))
    (is (not (contains? aliases 'adapter)))
    (is (= :analyzed-source
           (get-in (route/attempt source :float {'x :float}) [:stats :front-end])))))

(deftest scalar-result-types-survive-elementization-and-vertical-fusion
  (let [product (with-meta
                  '(* (clojure.core/aget x i) 2.0)
                  {:raster.type/tag 'float})
        source (list 'let*
                     (vector 'mapped (list 'raster.par/pmap 'i 'n 'float product)
                             'total '(raster.par/reduce acc 0.0 j n
                                                        (+ acc (clojure.core/aget mapped j))))
                     'total)
        routed (route/attempt source :float {'x :float})
        products (filter #(and (seq? %) (= '* (first %)))
                         (tree-seq coll? seq (get-in routed [:program :equations])))]
    (is (= 1 (get-in routed [:stats :vertical])))
    (is (seq products))
    (is (every? #(= 'float (:raster.type/tag (meta %))) products)
        "source type proofs remain attached after aget substitution and index rewriting")))

(deftest compound-parallel-extents-become-typed-scalar-ssa
  (let [source '(let* [step (raster.par/map! target i
                                             (clojure.core/* nrows width)
                                             float (clojure.core/aget x i))]
                      step)
        normalized (frontend/normalize-source source)
        program (frontend/form->program
                 normalized
                 {:dtype :float
                  :array-types {'x :float 'target :float}
                  :scalar-types {'nrows :long 'width :long}})
        equations (dialect/equations program)]
    (is (= '[rstr_extent_0 step] (mapv (comp first #(nth % 2)) equations)))
    (is (= ['scalar 'map] (mapv dialect/operation-kind equations)))
    (is (= 'rstr_extent_0 (dialect/operation-extent (second equations))))
    (is (= :long (:dtype (get-in (dialect/facts program) [:values 'rstr_extent_0]))))
    (is (= :analyzed-source
           (get-in (route/attempt source :float {'x :float 'target :float}
                                  {:scalar-types {'nrows :long 'width :long}})
                   [:stats :front-end])))))

(deftest named-checked-products-remain-scalar-extents-after-cast-normalization
  (let [source '(let* [^long size (clojure.core/* (clojure.core/long nrows)
                                                  (clojure.core/long width))
                       step (raster.par/map! target i size float
                                             (clojure.core/aget x i))]
                      step)
        options {:dtype :float
                 :array-types {'x :float 'target :float}
                 :scalar-types {'nrows :long 'width :long}}
        normalized (frontend/normalize-source source options)
        program (frontend/form->program normalized options)
        equations (dialect/equations program)
        size-equation (first equations)
        size-body (-> size-equation dialect/operation-parts :lambda
                      dialect/lambda-parts :body-results first)]
    (is (= '[size step] (mapv (comp first #(nth % 2)) equations)))
    (is (= ['scalar 'map] (mapv dialect/operation-kind equations)))
    (is (= 'size (dialect/operation-extent (second equations)))
        "normalizing identity long casts must not inline a compound expression into an extent")
    (is (= 'clojure.core/* (first size-body)))
    (is (= 2 (count (filter #{'clojure.core/long} (flatten size-body))))
        "the checked product remains one source-ordered scalar computation")))

(deftest retained-array-shapes-prove-compound-parallel-extents
  (let [program
        (frontend/form->program
         '(let* [total (raster.par/reduce acc 0.0 i
                                          (clojure.core/alength x)
                                          (+ acc (clojure.core/aget x i)))]
                total)
         {:dtype :double
          :values {'x (av/tensor {:dtype :double :shape ['n]})}
          :array-types {'x :double}
          :scalar-types {'n :long}})
        equation (first (dialect/equations program))]
    (is (= 'n (dialect/operation-extent equation)))
    (is (= ['n] (get-in (dialect/facts program) [:values 'x :shape])))))

(deftest devirtualized-output-allocation-is-generated-scaffolding
  (let [allocation (with-meta
                     '(.invk raster.arrays/alloc-like_m_array_long-impl x size)
                     {:raster.op/original 'raster.arrays/alloc-like})
        source (list 'let*
                     (vector 'size '(clojure.core/* m n)
                             'out allocation
                             'result '(raster.par/map! out i n float
                                                       (clojure.core/aget x i)))
                     'result)
        direct (frontend/form->program
                source {:dtype :float :array-types {'x :float 'out :float}
                        :scalar-types {'m :long 'n :long 'size :long}})
        routed (route/attempt source :float {'x :float 'out :float}
                              {:scalar-types {'m :long 'n :long 'size :long}})]
    (is (= ['scalar 'map] (mapv dialect/operation-kind (dialect/equations direct))))
    (is (= :analyzed-source
           (get-in routed [:stats :front-end])))
    (is (= ['size '(clojure.core/long (clojure.core/* m n))]
           (vec (take 2 (second (get-in routed [:program :source])))))
        "host allocation retains the transitive scalar binding that computes its extent")))

(deftest untyped-physical-allocation-capacity-fails-closed
  ;; Retaining allocation capacity must not reintroduce a local expression-type guess.  The
  ;; regular typed-analysis path stamps this binding; direct callers must supply the same fact.
  (let [allocation (with-meta
                     '(.invk raster.arrays/alloc-like_m_array_long-impl x size)
                     {:raster.op/original 'raster.arrays/alloc-like})
        source (list 'let*
                     (vector 'size '(clojure.core/* m n)
                             'out allocation
                             'result '(raster.par/map! out i n float
                                                       (clojure.core/aget x i)))
                     'result)
        reason (try
                 (frontend/form->program source {:dtype :float :array-types {'x :float 'out :float}
                                                 :scalar-types {'m :long 'n :long}})
                 nil
                 (catch clojure.lang.ExceptionInfo error
                   (:reason (ex-data error))))]
    (is (= :unsupported-scalar-binding reason))))

(deftest guarded-dense-write-is-an-explicit-inout-map
  (let [program
        (frontend/form->program
         '(let* [step (raster.par/map-void! i n
                                            (if (< i limit)
                                              (clojure.core/aset
                                               out i (clojure.core/aget src i))))]
                step)
         {:dtype :float
          :array-types {'src :float 'out :float}
          :scalar-types {'n :long 'limit :long}})
        equation (first (dialect/equations program))
        facts (dialect/facts program)]
    (is (= 'map (dialect/operation-kind equation)))
    (is (= '[out src limit] (dialect/operation-inputs equation)))
    (is (= [{:destination 'out :access :read-write :host-return :effect}]
           (get-in facts [:equations 0 :attributes :result-storage])))
    (is (= '(if (< i %capture0) %element1 %element0)
           (first (:body-results
                   (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))))))))

(deftest unique-indexed-write-is-a-typed-scatter
  (let [scatter '(raster.par/map-void!
                  i n
                  (clojure.core/aset out
                                     (raster.par/unique-index
                                      (clojure.core/aget indices i))
                                     (clojure.core/aget src i)))
        source (list 'let* ['step scatter] 'step)
        program (frontend/form->program
                 source {:dtype :float
                         :array-types {'indices :int 'src :float 'out :float}
                         :scalar-types {'n :long}})
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        write (dialect/write-parts
               (first (:body-results (dialect/lambda-parts (:lambda operation)))))]
    (is (= 'scatter (:kind operation)))
    (is (= :unique (get-in operation [:attributes :conflict])))
    (is (= '[indices out src] (dialect/operation-inputs equation)))
    (is (= {:destination-index '(clojure.core/aget %capture0 i) :predicate 1
            :value '(clojure.core/aget %capture2 i)} write))
    (is (= [{:destination 'out :access :read-write :host-return :effect}]
           (get-in (dialect/facts program) [:equations 0 :attributes :result-storage])))
    (is (= :analyzed-source
           (get-in (route/attempt source :float
                                  {'indices :int 'src :float 'out :float}
                                  {:scalar-types {'n :long}})
                   [:stats :front-end])))))

(deftest unique-scatter-values-canonicalize-captured-identity-folds
  (let [source
        '(let* [step
                (raster.par/map-void!
                 i rows
                 (clojure.core/aset
                  out (raster.par/unique-index (+ base i))
                  (float
                   (loop* [^{:raster.type/tag long} j 0
                           ^{:raster.type/tag float} acc seed]
                     (if (< (long j) width)
                       (recur (inc (long j))
                              (+ acc (clojure.core/aget x (+ (* i width) j))))
                       acc)))))]
           step)
        options {:dtype :float :array-types {'x :float 'out :float}
                 :scalar-types {'rows :long 'width :long 'base :long 'seed :float}}
        routed (route/attempt source :float (:array-types options) options)
        equation (-> routed :program :equations first :algorithm dialect/equations first)
        nodes (tree-seq coll? seq equation)
        fold (first (filter dialect/scalar-fold-form? nodes))]
    (is (= :typed-soac (get-in routed [:stats :route])))
    (is (= 'scatter (dialect/operation-kind equation)))
    (is (dialect/scalar-fold-form? fold))
    (is (symbol? (get-in (dialect/scalar-fold-parts fold) [:attributes :identity])))
    (is (not-any? #(and (seq? %) (contains? #{'loop 'loop*} (first %))) nodes))))

(deftest explicitly-typed-constant-identities-canonicalize-to-scalar-folds
  (let [source
        '(let* [step
                (raster.par/map-void!
                 i rows
                 (clojure.core/aset
                  out i
                  (float
                   (loop* [^{:raster.type/tag long} j 0
                           ^{:raster.type/tag float} acc (float 0.0)]
                     (if (< (long j) width)
                       (recur (inc (long j))
                              (+ acc (clojure.core/aget x (+ (* i width) j))))
                       acc)))))]
           step)
        options {:dtype :float :array-types {'x :float 'out :float}
                 :scalar-types {'rows :long 'width :long}}
        routed (route/attempt source :float (:array-types options) options)
        equation (-> routed :program :equations first :algorithm dialect/equations first)
        nodes (tree-seq coll? seq equation)
        fold (first (filter dialect/scalar-fold-form? nodes))
        attributes (:attributes (dialect/scalar-fold-parts fold))]
    (is (= :typed-soac (get-in routed [:stats :route])))
    (is (dialect/scalar-fold-form? fold))
    (is (= :ordered (:association attributes)))
    (is (= :float (:dtype attributes)))
    (is (= (Float/floatToRawIntBits 0.0)
           (Float/floatToRawIntBits (:identity attributes))))
    (is (not-any? #(and (seq? %) (contains? #{'loop 'loop*} (first %))) nodes))))

(deftest offset-map-is-an-injective-typed-scatter
  (let [source '(let* [result
                       (raster.par/map! out i n :offset base float
                                        (clojure.core/aget src i))]
                      result)
        program (frontend/form->program
                 source {:dtype :float
                         :array-types {'src :float 'out :float}
                         :scalar-types {'n :long 'base :long}})
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        write (dialect/write-parts
               (first (:body-results (dialect/lambda-parts (:lambda operation)))))]
    (is (= 'scatter (:kind operation)))
    (is (= :unique (get-in operation [:attributes :conflict])))
    (is (= '(clojure.core/+ %capture0 i) (:destination-index write)))
    (is (= '[base out src] (dialect/operation-inputs equation)))
    (is (= [{:destination 'out :access :read-write :host-return :buffer}]
           (dialect/result-storage program (second equation))))))

(deftest typed-map-reduce-and-scatter-lower-without-compatibility-records
  (let [map-program
        (frontend/form->program
         '(let* [result (raster.par/pmap i n float (clojure.core/aget x i))]
                result)
         {:dtype :float :array-types {'x :float} :scalar-types {'n :long}})
        reduce-program
        (frontend/form->program
         '(let* [result (raster.par/reduce acc 0.0 i n
                                           (+ acc (clojure.core/aget x i)))]
                result)
         {:dtype :float :array-types {'x :float} :scalar-types {'n :long}})
        scatter-program
        (frontend/form->program
         '(let* [effect
                 (raster.par/map-void!
                  i n
                  (clojure.core/aset out
                                     (raster.par/unique-index
                                      (clojure.core/aget indices i))
                                     (clojure.core/aget x i)))]
                effect)
         {:dtype :float
          :array-types {'indices :int 'x :float 'out :float}
          :scalar-types {'n :long}})
        reject (fn [& _]
                 (throw (ex-info "typed lowering constructed a compatibility SOAC record" {})))
        [mapped reduced scattered]
        (with-redefs [legacy-soac/->SoacMap reject
                      legacy-soac/->SoacReduce reject]
          [(first (soac-lower/lower-typed-map map-program :ocl:0 :dtype :float))
           (first (soac-lower/lower-typed-reduce reduce-program :ocl:0 :dtype :float))
           (first (soac-lower/lower-typed-scatter scatter-program :ocl:0 :dtype :float))])]
    (is (instance? raster.compiler.ir.segop.SegMap mapped))
    (is (instance? raster.compiler.ir.segop.SegRed reduced))
    (is (instance? raster.compiler.ir.segop.SegMap scattered))))

(deftest parallel-semantics-enter-only-their-exact-typed-operation
  (testing "a certified inclusive scan is represented directly, with destination facts"
    (let [program (frontend/form->program
                   '(let* [result (raster.par/scan target acc 0.0 i n float
                                                   (+ acc (clojure.core/aget x i)))]
                          result)
                   {:dtype :float :array-types {'x :float 'target :float}})
          equation (first (dialect/equations program))]
      (is (= 'scan (dialect/operation-kind equation)))
      (is (= :inclusive (get-in (dialect/operation-parts equation) [:attributes :mode])))
      (is (= '{result target} (get-in (dialect/facts program) [:equations 0 :aliases])))
      (is (= ['target] (dialect/physical-results program equation)))
      (is (= [{:destination 'target :access :write :host-return :buffer}]
             (dialect/result-storage program 0)))
      (is (empty? (get-in (dialect/operation-parts equation)
                          [:attributes :attributes :stable-array-captures])))
      (is (= #{:memory/write} (:effects (dialect/facts program))))))

  (testing "exclusive scan is the same certified operation with a distinct result mode"
    (let [program (frontend/form->program
                   '(let* [result (raster.par/scan-exclusive target acc 0.0 i n float
                                                             (+ acc (clojure.core/aget x i)))]
                          result)
                   {:dtype :float :array-types {'x :float 'target :float}})
          equation (first (dialect/equations program))]
      (is (= 'scan (dialect/operation-kind equation)))
      (is (= :exclusive (get-in (dialect/operation-parts equation) [:attributes :mode])))
      (is (= '[(clojure.core/inc n)]
             (:shape (get-in (dialect/facts program) [:values 'result]))))))
  (testing "a general recurrence cannot be mislabeled as an associative scan"
    (try
      (frontend/form->program
       '(let* [result (raster.par/scan target h 0.0 i n float
                                       (Math/tanh (+ h (clojure.core/aget x i))))]
              result)
       {:dtype :float :array-types {'x :float 'target :float}})
      (is false "an uncertified recurrence must decline")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :scan-not-associative (:reason (ex-data exception)))))))
  (testing "destination-writing map identity and effects are explicit compiler facts"
    (let [program (frontend/form->program
                   '(let* [step (raster.par/map! target i n float
                                                 (+ (clojure.core/aget x i) 1.0))]
                          step)
                   {:dtype :float :array-types {'x :float 'target :float}})
          facts (dialect/facts program)]
      (is (= 'map (dialect/operation-kind (first (dialect/equations program)))))
      (is (= '[x] (dialect/operation-inputs (first (dialect/equations program))))
          "a write-only destination is not duplicated as a semantic read operand")
      (is (= '{step target} (get-in facts [:equations 0 :aliases])))
      (is (= [{:destination 'target :access :write :host-return :buffer}]
             (get-in facts [:equations 0 :attributes :result-storage])))
      (is (some? (get-in facts [:values 'target]))
          "the physical output boundary retains its own value contract")
      (is (= #{:memory/write} (:effects facts)))))
  (testing "a source binder and destination still require distinct SSA identities"
    (is (nil? (frontend/form->program
               '(let* [target (raster.par/map! target i n float
                                               (+ (clojure.core/aget x i) 1.0))]
                      target)
               {:dtype :float :array-types {'x :float 'target :float}}))))
  (testing "a read/write destination is one explicit typed operand"
    (let [program (frontend/form->program
                   '(let* [step (raster.par/map! target i n float
                                                 (+ (clojure.core/aget target i) 1.0))]
                          step)
                   {:dtype :float :array-types {'target :float}})
          equation (first (dialect/equations program))
          facts (dialect/facts program)]
      (is (= '[target] (dialect/operation-inputs equation)))
      (is (= :read-write
             (get-in facts [:equations 0 :attributes :result-storage 0 :access])))
      (is (= '{step target} (get-in facts [:equations 0 :aliases]))))))

(deftest contraction-enters-as-a-general-typed-segmented-reduction
  (let [source
        '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                          (* (clojure.core/aget A (+ (* i k) l))
                                             (clojure.core/aget B (+ (* l n) j))))]
               step)
        options {:dtype :float
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m :long 'n :long 'k :long}}
        program (frontend/form->program source options)
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        routed (route/attempt source :float (:array-types options)
                              {:scalar-types (:scalar-types options)})
        scheduled
        (with-redefs [contraction-facts/contraction-facts
                      (fn [& _]
                        (throw (ex-info "typed scheduling reparsed source" {})))]
          ((requiring-resolve
            'raster.compiler.passes.parallel.segop-lower-pass/segop-lower-pass)
           (:program routed) {:target-device :ze:0 :dtype :float}))
        segred (-> scheduled :form :equations first :operations first)]
    (is (= 'segmented-reduce (:kind operation)))
    (is (= '[[i m] [j n]] (get-in operation [:attributes :segment-axes])))
    (is (= '[m n k] (dialect/operation-extents equation)))
    (is (= [{:destination 'C :access :write :host-return :buffer}]
           (get-in (dialect/facts program) [:equations 0 :attributes :result-storage])))
    (is (= '[m n] (:shape (get-in (dialect/facts program)
                                  [:values (first (nth equation 2))]))))
    (is (= :analyzed-source (get-in routed [:stats :front-end])))
    (is (instance? raster.compiler.ir.segop.SegRed segred))
    (is (= :contraction (:phase segred)))
    (is (= :hardware-contraction-candidates (get-in segred [:schedule :strategy])))
    (is (= '[C] (reduction/results (:reduction segred))))
    (is (nil? (some #(when (instance? raster.compiler.ir.soac.SoacContract %) %)
                    (tree-seq coll? seq (:form scheduled)))))
    (is (= :typed-soac (get-in (-> scheduled :form :equations first)
                               [:attributes :algorithm-dialect])))))

(deftest product-reduction-keeps-discarded-components-out-of-ssa-results
  (let [expression
        '(raster.par/product-reduce!
          [nil indices]
          [[best-value -1.0e38 :float] [best-index 2147483647 :int]]
          [[row nrows]] col width
          []
          [(clojure.core/aget values (clojure.core/+ (clojure.core/* row width) col))
           (int col)]
          [[left-value right-value] [left-index right-index]]
          []
          [(if (> right-value left-value) right-value left-value)
           (if (> right-value left-value) right-index left-index)]
          {:associative? true :commutative? true})
        source (list 'let* ['effect expression] 'effect)
        options {:dtype :float
                 :array-types {'values :float 'indices :int}
                 :scalar-types {'nrows :long 'width :long}}
        program (frontend/form->program source options)
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        routed (route/attempt source :float (:array-types options)
                              {:scalar-types (:scalar-types options)})
        scheduled
        (with-redefs [legacy-soac/->SoacMap
                      (fn [& _]
                        (throw (ex-info "typed product reduction constructed SoacMap" {})))
                      legacy-soac/->SoacReduce
                      (fn [& _]
                        (throw (ex-info "typed product reduction constructed SoacReduce" {})))]
          ((requiring-resolve
            'raster.compiler.passes.parallel.segop-lower-pass/segop-lower-pass)
           (:program routed) {:target-device :ocl:0 :dtype :float}))
        product (-> scheduled :form :equations first :operations first)
        remapped (dialect/remap-values program {'values [:binding 'values]})
        remapped-operation (dialect/operation-parts
                            (first (dialect/equations remapped)))]
    (is (= 'product-reduce (:kind operation)))
    (is (= [1] (get-in operation [:attributes :result-components])))
    (is (= [:float :int] (get-in operation [:attributes :dtypes])))
    (is (= 1 (count (nth equation 2)))
        "the winning value remains an algebra component but is not a fake SSA output")
    (is (= ['indices] (dialect/physical-results program equation)))
    (is (= program (dialect/validate! program)))
    (is (= :analyzed-source (get-in routed [:stats :front-end])))
    (is (instance? raster.compiler.ir.segop.SegRed product))
    (is (= :product (:phase product)))
    (is (= [:float :int] (reduction/dtypes (:reduction product))))
    (is (= #{'indices} (:outputs product)))
    (is (= :typed-soac (:algorithm-dialect product)))
    (is (some #{[:binding 'values]} (:captures remapped-operation)))
    (is (= (:element-lambda operation) (:element-lambda remapped-operation)))
    (is (= (:combine-lambda operation) (:combine-lambda remapped-operation)))
    (is (= remapped (dialect/validate! remapped)))))

(deftest product-reduction-combine-is-a-closed-scalar-region
  (let [expression
        '(raster.par/product-reduce!
          [nil indices]
          [[best-value -1.0e38 :float] [best-index 2147483647 :int]]
          [[row nrows]] col width
          []
          [(clojure.core/aget values (clojure.core/+ (clojure.core/* row width) col))
           (int col)]
          [[left-value right-value] [left-index right-index]]
          []
          [row (if (> right-value left-value) right-index left-index)]
          {:associative? true :commutative? true})]
    (try
      (frontend/form->program
       (list 'let* ['effect expression] 'effect)
       {:dtype :float
        :array-types {'values :float 'indices :int}
        :scalar-types {'nrows :long 'width :long}})
      (is false "a segment-axis capture must not enter the binary combine algebra")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-product-reduce-combine-closure
               (:reason (ex-data exception))))))))

(deftest boundary-stencil-is-a-direct-typed-scheduled-operation
  (let [source
        '(let* [result
                (raster.par/stencil!
                 du [u] 1 :dirichlet double i n
                 (* alpha inv-dx2
                    (+ (clojure.core/aget u (clojure.core/- i 1))
                       (* -2.0 (clojure.core/aget u i))
                       (clojure.core/aget u (clojure.core/+ i 1)))))]
               result)
        options {:dtype :double
                 :array-types {'du :double 'u :double}
                 :scalar-types {'n :long 'alpha :double 'inv-dx2 :double}}
        program (frontend/form->program source options)
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        routed (route/attempt source :double (:array-types options)
                              {:scalar-types (:scalar-types options)})
        scheduled
        ((requiring-resolve
          'raster.compiler.passes.parallel.segop-lower-pass/segop-lower-pass)
         (:program routed) {:target-device :cpu:0 :dtype :double})
        stencil (-> scheduled :form :equations first :operations first)]
    (is (= 'stencil (:kind operation)))
    (is (= {:radius 1 :boundary :dirichlet :dtype :double}
           (select-keys (assoc (:attributes operation)
                               :dtype (first (get-in operation [:attributes :dtypes])))
                        [:radius :boundary :dtype])))
    (is (= '[u] (get-in operation [:attributes :attributes :stable-array-captures])))
    (is (= ['du] (dialect/physical-results program equation)))
    (is (= :analyzed-source (get-in routed [:stats :front-end])))
    (is (instance? raster.compiler.ir.segop.SegStencil stencil))
    (is (= #{'u} (:inputs stencil)))
    (is (= #{'du} (:outputs stencil)))
    (is (= #{'alpha 'inv-dx2} (:scalars stencil)))
    (is (= :no-write-alias (:aliasing stencil)))
    (is (= :typed-soac (:algorithm-dialect stencil)))
    (testing "the verifier rejects in-place neighborhood updates before scheduling"
      (let [aliased (list 'let* ['result
                                 '(raster.par/stencil!
                                   u [u] 1 :dirichlet double i n
                                   (+ (clojure.core/aget u (clojure.core/- i 1))
                                      (clojure.core/aget u (clojure.core/+ i 1))))]
                          'result)]
        (try
          (frontend/form->program
           aliased {:dtype :double :array-types {'u :double}
                    :scalar-types {'n :long}})
          (is false "an aliased output invalidates stable neighborhood reads")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :typed-soac-stable-read-alias (:reason (ex-data exception))))))))))

(deftest explicit-contraction-result-transform-stays-in-typed-soac
  (let [transform {:acc 'acc
                   :expr '(raster.numeric/+ acc (clojure.core/aget bias j))
                   :operands [{:sym 'bias :dtype :float
                               :map {:groups [[['j 128]]]}}]
                   :scalars []
                   :dtype :float}
        source (list 'let* ['step
                            (apply list
                                   (concat
                                    '(raster.par/contract C [[i 128] [j 128]] [[l 128]]
                                                          (* (clojure.core/aget A (+ (* i 128) l))
                                                             (clojure.core/aget B (+ (* l 128) j))))
                                    [:epilogue transform]))]
                     'step)
        program (frontend/form->program
                 source {:dtype :half
                         :array-types {'A :half 'B :half 'C :half 'bias :float}})
        equation (first (dialect/equations program))
        attributes (:attributes (dialect/operation-parts equation))
        typed-transform (:result-transform attributes)]
    (is (= [{:value 'bias :parameter '%result-operand0 :dtype :float
             :map {:groups [[['j 128]]]}}]
           (:operands typed-transform)))
    (is (= '[acc %result-operand0]
           (:parameters (dialect/lambda-parts (:lambda typed-transform)))))
    (is (= '[(raster.numeric/+ acc (clojure.core/aget %result-operand0 j))]
           (:body-results (dialect/lambda-parts (:lambda typed-transform)))))
    (is (= '[A B bias] (get-in attributes [:attributes :stable-array-captures])))
    (is (= '[A B bias] (:inputs (dialect/facts program))))
    (is (= program (dialect/validate! program)))
    (let [remapped (dialect/remap-values program {'bias [:binding 'bias]})
          remapped-transform
          (get-in (dialect/operation-parts (first (dialect/equations remapped)))
                  [:attributes :result-transform])]
      (is (= [:binding 'bias] (get-in remapped-transform [:operands 0 :value])))
      (is (= (:lambda typed-transform) (:lambda remapped-transform))
          "SSA value remapping must not rewrite the lexical scalar region")
      (is (= remapped (dialect/validate! remapped))))))

(deftest contraction-result-transform-cannot-hide-an-untyped-capture
  (let [transform {:acc 'acc
                   :expr '(raster.numeric/+ acc undeclared)
                   :operands [] :scalars [] :dtype :float}
        contract (apply list
                        (concat
                         '(raster.par/contract C [[i 8] [j 8]] [[l 8]]
                                               (* (clojure.core/aget A (+ (* i 8) l))
                                                  (clojure.core/aget B (+ (* l 8) j))))
                         [:epilogue transform]))]
    (try
      (frontend/form->program
       (list 'let* ['step contract] 'step)
       {:dtype :half :array-types {'A :half 'B :half 'C :half}})
      (is false "an undeclared result-transform value must not enter TypedSOAC")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-result-transform-expression
               (:reason (ex-data exception))))))))

(deftest destination-shapes-refine-across-ordered-maps
  (let [program (frontend/form->program
                 '(let* [first-step (raster.par/map! first-out i n float
                                                     (+ (clojure.core/aget x i) 1.0))
                         second-step (raster.par/map! second-out j n float
                                                      (* (clojure.core/aget first-out j) 2.0))]
                        second-step)
                 {:dtype :float
                  :array-types {'x :float 'first-out :float 'second-out :float}})]
    (is (= '[n] (get-in (dialect/facts program) [:values 'first-out :shape])))
    (is (= 2 (count (dialect/equations program))))))

(deftest effect-only-pointwise-writes-have-logical-results-and-physical-storage
  (doseq [[label expression canonical-storage-conversion?]
          [["map2"
            '(raster.par/map2! a b i n float
                               (+ (clojure.core/aget x i) 1.0)
                               (* (clojure.core/aget y i) 2.0))
            true]
           ["independent multi-store map-void"
            '(raster.par/map-void!
              i n
              (do (clojure.core/aset a i (float (+ (clojure.core/aget x i) 1.0)))
                  (clojure.core/aset b i (float (* (clojure.core/aget y i) 2.0)))))
            false]]]
    (testing label
      (let [program (frontend/form->program
                     (list 'let* ['effect expression] 'effect)
                     {:dtype :float
                      :array-types {'x :float 'y :float 'a :float 'b :float}})
            equation (first (dialect/equations program))
            body-results (-> equation dialect/operation-parts :lambda
                             dialect/lambda-parts :body-results)
            facts (dialect/facts program)
            results (vec (nth equation 2))]
        (is (= [[:effect-map 0 0] [:effect-map 0 1]] results))
        (is (= ['a 'b] (dialect/physical-results program equation)))
        (is (= [:write :write]
               (mapv :access (dialect/result-storage program 0))))
        (is (= #{:memory/write} (:effects facts)))
        (if canonical-storage-conversion?
          (is (= [[:double :float] [:double :float]]
                 (mapv (fn [result]
                         (let [attributes (:attributes (dialect/scalar-convert-parts result))]
                           [(:source-dtype attributes) (:target-dtype attributes)]))
                       body-results))
              "every map2 result retains its complete source-to-storage conversion")
          (is (every? #(= 'float
                          (some-> % descriptor/semantic-op descriptor/cast-result-tag))
                      body-results)
              "explicit user store casts remain source operations rather than implicit storage terms"))
        (is (= [] (dialect/outputs program))
            "the host nil result is not mislabeled as a tensor result")))))

(deftest returned-write-destination-projects-to-its-logical-result
  (let [program
        (frontend/form->program
         '(let* [effect (raster.par/map! out i n float
                                         (+ (clojure.core/aget x i) 1.0))]
                out)
         {:dtype :float :array-types {'x :float 'out :float}})
        equation (first (dialect/equations program))
        body (-> equation dialect/operation-parts :lambda
                 dialect/lambda-parts :body-results first)]
    (is (= ['out] (dialect/physical-results program equation)))
    (is (dialect/scalar-convert-form? body)
        "the source map result conversion remains an explicit typed boundary")
    (is (= '(clojure.core/float (+ %element0 1.0))
           (projection/scalar-folds->source body)))
    (is (= (vec (nth equation 2)) (dialect/outputs program))
        "a returned destination denotes the fresh logical result, not an undeclared buffer")))

(deftest typed-shared-locals-become-one-region-ssa-spine
  (let [expression
        '(raster.par/map-void!
          i n
          (let* [^float shifted (+ (clojure.core/aget x i) 1.0)
                 ^float squared (* shifted shifted)]
                (clojure.core/aset a i (float shifted))
                (clojure.core/aset b i (float squared))))
        program (frontend/form->program
                 (list 'let* ['effect expression] 'effect)
                 {:dtype :float :array-types {'x :float 'a :float 'b :float}})
        equation (first (dialect/equations program))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= [{:id 'rstr_local_0 :dtype :float
             :init '(+ %element0 1.0)}
            {:id 'rstr_local_1 :dtype :float
             :init '(* rstr_local_0 rstr_local_0)}]
           locals))
    (is (= '[(float rstr_local_0) (float rstr_local_1)] body-results))
    (is (= '[n x] (:inputs (dialect/facts program))))
    (is (not-any? #{'rstr_local_0 'rstr_local_1}
                  (keys (:values (dialect/facts program))))
        "region-local SSA values are lexical, not fake program inputs")))

(deftest nested-typed-scopes-share-one-alpha-stable-region
  (let [expression
        '(raster.par/map-void!
          i n
          (let* [^float shifted (+ (clojure.core/aget x i) 1.0)]
                (let* [^float squared (* shifted shifted)]
                      (clojure.core/aset out i (float squared)))))
        program (frontend/form->program
                 (list 'let* ['effect expression] 'effect)
                 {:dtype :float :array-types {'x :float 'out :float}})
        equation (first (dialect/equations program))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= [{:id 'rstr_local_0 :dtype :float
             :init '(+ %element0 1.0)}
            {:id 'rstr_local_1 :dtype :float
             :init '(* rstr_local_0 rstr_local_0)}]
           locals))
    (is (= '[(float rstr_local_1)] body-results))))

(deftest pointwise-effect-map-retains-a-source-written-checked-cast
  (let [program (frontend/form->program
                 '(let* [effect
                          (raster.par/map-void!
                           i n (clojure.core/aset out i
                                                 (clojure.core/int
                                                  (clojure.core/aget input i))))]
                         effect)
                 {:dtype :int :array-types {'input :long 'out :int}
                  :scalar-types {'n :long}})
        equation (first (dialect/equations program))
        result (-> equation dialect/operation-parts :lambda
                   dialect/lambda-parts :body-results first)]
    (is (= 'map (dialect/operation-kind equation)))
    (is (= 'clojure.core/int (first result))
        "the source cast stays in the scalar region; result storage owns no source policy")))

(deftest walked-identity-cast-does-not-hide-an-additive-scatter
  (let [index '(clojure.core/aget slots i)
        accumulator (list 'clojure.core/aget 'out index)
        contribution '(clojure.core/aget input i)
        addition (with-meta
                   (list '.invk 'raster.numeric/_plus__m_float_float-impl
                         accumulator contribution)
                   {:tag 'float :raster.type/tag 'float
                    :raster.op/original 'clojure.core/+})
        source (list 'let* ['effect
                            (list 'raster.par/map-void! 'i 'n
                                  (list 'clojure.core/aset 'out index
                                        (list 'float addition)))]
                     'effect)
        program (frontend/form->program
                 source {:dtype :float
                         :array-types {'slots :int 'out :float 'input :float}
                         :scalar-types {'n :long}})
        equation (first (dialect/equations program))
        {:keys [attributes lambda]} (dialect/operation-parts equation)
        expression (first (:body-results (dialect/lambda-parts lambda)))
        effect (dialect/write-parts expression)]
    (is (= 'scatter (dialect/operation-kind equation)))
    (is (= :reduce (get-in attributes [:conflict :kind])))
    (is (= '(clojure.core/aget %capture0 i) (:value effect))
        "the typed contribution, not a destination read outside the atomic update, is retained")))

(deftest guarded-branch-locals-remain-inside-an-independent-reducing-effect
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (let* [^long position (clojure.core/- i padding)]
                   (if (clojure.core/>= position 0)
                     (let* [^long destination (clojure.core/rem position width)]
                       (clojure.core/aset
                        out destination
                        (clojure.core/+ (clojure.core/aget out destination)
                                        (clojure.core/aget input i)))))))]
               effect)
        program (frontend/form->program
                 source {:dtype :float :array-types {'out :float 'input :float}
                         :scalar-types {'n :long 'padding :long 'width :long}})
        equation (first (dialect/equations program))
        {:keys [attributes lambda]} (dialect/operation-parts equation)
        guarded (first (:body-results (dialect/lambda-parts lambda)))
        parts (dialect/effect-parts guarded)]
    (is (= 'effect-map (dialect/operation-kind equation)))
    (is (= :independent (:iteration-order attributes)))
    (is (= 'effect-when (first guarded)))
    (is (= '(clojure.core/>= rstr_local_0 0) (:predicate parts)))
    (is (= [:long] (mapv :dtype (:locals (:region parts)))))
    (is (= :reduce (-> parts :region :body-results first dialect/effect-parts :conflict :kind)))
    (is (= program (dialect/validate! program)))))

(deftest guarded-counted-store-loop-remains-inside-the-conditional-region
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i rows
                 (if (clojure.core/> enabled 0)
                   (dotimes [j width]
                     (clojure.core/aset
                      out (clojure.core/+ (clojure.core/* i width) j)
                      (clojure.core/aget input
                                        (clojure.core/+ (clojure.core/* i width) j))))))]
               effect)
        program (frontend/form->program
                 source {:dtype :float :array-types {'out :float 'input :float}
                         :scalar-types {'rows :long 'width :long 'enabled :int}})
        equation (first (dialect/equations program))
        guarded (-> equation dialect/operation-parts :lambda
                    dialect/lambda-parts :body-results first)
        region (:region (dialect/effect-parts guarded))
        loop-effect (first (:body-results region))]
    (is (= 'effect-map (dialect/operation-kind equation)))
    (is (= 'effect-when (first guarded)))
    (is (= '(clojure.core/> %capture0 0)
           (:predicate (dialect/effect-parts guarded))))
    (is (= 'effect-loop (first loop-effect)))
    (is (= :unique
           (-> guarded dialect/effect-parts vector dialect/effect-part-leaves
               first :conflict)))
    (is (= program (dialect/validate! program)))))

(deftest removable-top-level-conditional-loop-normalizes-to-a-guarded-effect-map
  (let [source
        '(let* [effect
                (if (clojure.core/> norm max-norm)
                  (let* [^double scale (clojure.core// max-norm norm)]
                    (dotimes [i n]
                      (clojure.core/aset
                       grads i
                       (clojure.core/* (clojure.core/aget grads i) scale))))
                  nil)]
               grads)
        program (:program
                 (route/attempt
                  source :double {'grads :double}
                  {:scalar-types {'n :long 'norm :double 'max-norm :double}}))
        algorithm (-> program :equations first :algorithm)
        equation (first (dialect/equations algorithm))
        guarded (-> equation dialect/operation-parts :lambda
                    dialect/lambda-parts :body-results first)
        parts (dialect/effect-parts guarded)]
    (is (= 'effect-map (dialect/operation-kind equation)))
    (is (= 'effect-when (first guarded)))
    (is (= '(clojure.core/> %capture1 %capture0) (:predicate parts)))
    (is (= [:double] (mapv :dtype (:locals (:region parts)))))
    (is (= :unique (-> parts :region :body-results first dialect/effect-parts :conflict)))
    (is (= algorithm (dialect/validate! algorithm)))))

(deftest a-nested-scalar-reduction-is-lifted-and-fused-before-a-guarded-effect
  (let [reduction (with-meta
                    '(raster.par/reduce acc (float 0.0) i n
                       (+ acc (* (clojure.core/aget grads i)
                                 (clojure.core/aget grads i))))
                    {:tag 'float :raster.type/tag 'float :raster.type/elem-type :float})
        norm-expression
        (with-meta (list '.invk 'raster.numeric/sqrt_m_float-impl reduction)
          {:tag 'float :raster.type/tag 'float :raster.op/original 'raster.numeric/sqrt})
        norm (with-meta 'norm {:tag 'float :raster.type/tag 'float})
        source
        (list 'let*
              [norm norm-expression
               'effect
               '(if (> norm max-norm)
                  (let* [^double scale (/ max-norm (double norm))]
                    (dotimes [i n]
                      (clojure.core/aset grads i
                                        (* (clojure.core/aget grads i) scale))))
                  nil)]
              'grads)
        program (:program
                 (route/attempt source :float {'grads :float}
                                {:scalar-types {'n :long 'max-norm :double}}))
        algorithms (mapv :algorithm (:equations program))
        equations (mapv (comp first dialect/equations) algorithms)
        reduction-operation (dialect/operation-parts (first equations))]
    (is (= '[reduce effect-map] (mapv (comp :kind dialect/operation-parts) equations)))
    (is (= '[norm] (nth (first equations) 2)))
    (is (dialect/result-transform? (get-in reduction-operation
                                           [:attributes :result-transform])))
    (is (every? true? (map = algorithms (map dialect/validate! algorithms))))))

(deftest functional-map-result-casts-become-typed-conversion-terms
  (doseq [[cast overflow] [['int :trap] ['unchecked-int :wrap]]]
    (let [source (list 'let*
                       ['result (list 'raster.par/map! 'out 'i 'n cast
                                     '(clojure.core/aget input i))]
                       'result)
          program (frontend/form->program
                   source {:dtype :int :array-types {'input :long 'out :int}
                           :scalar-types {'n :long}})
          expression (-> program dialect/equations first dialect/operation-parts :lambda
                         dialect/lambda-parts :body-results first)
          {:keys [attributes operand]} (dialect/scalar-convert-parts expression)]
      (is (= 'raster.compiler.ir.soac-dialect/scalar-convert (first expression)))
      (is (= {:source-dtype :long :target-dtype :int
              :rounding :exact :overflow overflow
              :source-op (symbol "clojure.core" (name cast))}
             attributes))
      (is (= '%element0 operand)))))

(deftest raw-array-inputs-receive-the-entry-dtype-before-conversion-elaboration
  (let [program
        (frontend/form->program
         '(let* [y (raster.par/pmap i n float
                                    (* (clojure.core/aget x i)
                                       (clojure.core/aget x i)))
                 z (raster.par/pmap j n float
                                    (+ (clojure.core/aget y j) 1.0))]
                z)
         {:dtype :float})
        conversions
        (mapv #(-> % dialect/operation-parts :lambda dialect/lambda-parts
                   :body-results first dialect/scalar-convert-parts)
              (dialect/equations program))]
    (is program)
    (is (= 2 (count (dialect/equations program))))
    (is (= [[:float :float] [:double :float]]
           (mapv (fn [conversion]
                   ((juxt :source-dtype :target-dtype) (:attributes conversion)))
                 conversions))
        "the first map materializes its float result; the double-literal consumer narrows once")
    (is (= :float (get-in (dialect/facts program) [:values 'x :dtype])))))

(deftest primitive-cast-operands-retain-their-dtype-during-source-inference
  (let [program
        (frontend/form->program
         '(let* [result (raster.par/pmap i n float
                                         (raster.numeric/+ (clojure.core/aget x i)
                                                           (clojure.core/float 1.0)))]
                result)
         {:dtype :float :array-types {'x :float}})
        conversion (some-> program dialect/equations first dialect/operation-parts :lambda
                           dialect/lambda-parts :body-results first
                           dialect/scalar-convert-parts)]
    (is program)
    (is (= [:float :float]
           ((juxt :source-dtype :target-dtype) (:attributes conversion))))))

(deftest ordered-loop-map-result-takes-its-source-dtype-from-the-carry
  (let [source
        '(let* [result
                (raster.par/map! out i n float
                                 (loop* [j 0 acc (float 0.0)]
                                        (if (< (long j) (long width))
                                          (let* [value
                                                 (clojure.core/aget
                                                  x (+ (* (long i) (long width)) (long j)))]
                                                (recur (inc (long j))
                                                       (+ (float acc) (float value))))
                                          acc)))]
               result)
        program (frontend/form->program
                 source {:dtype :float :array-types {'x :float 'out :float}
                         :scalar-types {'n :long 'width :long}})
        body (some-> program dialect/equations first dialect/operation-parts :lambda
                     dialect/lambda-parts :body-results first)
        conversion (dialect/scalar-convert-parts body)
        loop-expression (:operand conversion)
        matched (patterns/match-ordered-reduce-loop loop-expression)]
    (is program)
    (is (= 'loop* (first loop-expression))
        "this frontend checkpoint retains source control before scalar-region canonicalization")
    (is matched)
    (is (= '(float 0.0) (:acc-init matched)))
    (is (= (:acc-sym matched) (:else-expr matched))
        "the sole loop exit is the float accumulator whose type owns the map result")
    (is (= [:float :float]
           ((juxt :source-dtype :target-dtype) (:attributes conversion))))))

(deftest agreeing-conditional-branches-own-the-map-source-dtype
  (let [source '(let* [y (raster.par/pmap i n float
                                           (if (> (clojure.core/aget x i) threshold)
                                             (clojure.core/aget x i)
                                             (* alpha (clojure.core/aget x i))))]
                         y)
        program (frontend/form->program
                 source {:dtype :float :array-types {'x :float}
                         :scalar-types {'n :long 'threshold :float 'alpha :float}})
        conversion (-> program dialect/equations first dialect/operation-parts :lambda
                       dialect/lambda-parts :body-results first dialect/scalar-convert-parts)]
    (is program)
    (is (= [:float :float]
           ((juxt :source-dtype :target-dtype) (:attributes conversion))))
    (is (nil? (frontend/form->program
               '(let* [y (raster.par/pmap i n float
                                           (if flag (clojure.core/aget x i) 0.0))]
                  y)
               {:dtype :float :array-types {'x :float}
                :scalar-types {'n :long 'flag :int}}))
        "a consumer cast cannot invent one source dtype for mixed conditional exits")))

(deftest closed-core-integer-case-becomes-a-typed-conditional-map
  (let [expression
        '(raster.par/map-void!
          i n
          (let* [^int choice (clojure.core/aget choices i)]
                (case* choice 0 0 nil
                       {0 [0 (clojure.core/aset out i (float 10.0))]
                        1 [1 (clojure.core/aset out i (float 20.0))]}
                       :compact :int)))
        program (frontend/form->program
                 (list 'let* ['effect expression] 'effect)
                 {:dtype :float :array-types {'choices :int 'out :float}})
        equation (first (dialect/equations program))
        {:keys [body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))
        result (first body-results)]
    (is (= 'map (dialect/operation-kind equation)))
    (is (= 'out (first (dialect/physical-results program equation))))
    (is (not-any? #{'case* :compact :int} (flatten result)))
    (is (some #{'clojure.core/==} (flatten result)))
    (is (some #{10.0 20.0} (flatten result)))))

(deftest aligned-conditional-store-retains-its-agreeing-value-type
  (let [rounded (with-meta '(float 10.0)
                  {:tag 'double :raster.type/tag 'double})
        source (list 'let*
                     ['effect
                      (list 'raster.par/map-void! 'i 'n
                            (list 'if '(clojure.core/== i 0)
                                  (list 'clojure.core/aset 'out 'i rounded)
                                  '(clojure.core/aset out i (float 20.0))))]
                     'effect)
        program (frontend/form->program
                 source
                 {:dtype :float :array-types {'out :float} :scalar-types {'n :long}})
        equation (first (dialect/equations program))
        result (-> equation dialect/operation-parts :lambda
                   dialect/lambda-parts :body-results first)
        conditional (some #(when (and (seq? %) (= 'if (first %))
                                      (= 'float (types/sym-type-tag %)))
                             %)
                          (tree-seq coll? seq result))]
    (is (= 'if (first conditional)))
    (is (= 'float (types/sym-type-tag conditional))
        "the explicit cast descriptor, not stale metadata, types the merged value-if")))

(deftest noninteger-case-representation-declines-the-typed-route
  (let [expression
        '(raster.par/map-void!
          i n
          (case* choice 0 0 nil
                 {0 [:stay (clojure.core/aset out i (float 10.0))]}
                 :compact :hash-equiv))]
    (is (nil? (frontend/form->program
               (list 'let* ['effect expression] 'effect)
               {:dtype :float :array-types {'out :float}})))))

(deftest typed-local-snapshots-may-feed-several-updated-destinations
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (let* [^float m-new (+ (clojure.core/aget m i)
                                        (clojure.core/aget grad i))
                        ^float v-new (+ (clojure.core/aget v i)
                                        (* (clojure.core/aget grad i)
                                           (clojure.core/aget grad i)))]
                       (clojure.core/aset m i (float m-new))
                       (clojure.core/aset v i (float v-new))
                       (clojure.core/aset param i
                                          (float (- (clojure.core/aget param i)
                                                    (/ m-new (+ v-new 1.0)))))))]
               effect)
        program (frontend/form->program
                 source {:dtype :float
                         :array-types {'grad :float 'm :float 'v :float 'param :float}})
        equation (first (dialect/equations program))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= 2 (count locals)))
    (is (= 3 (count body-results)))
    (is (= ['m 'v 'param] (dialect/physical-results program equation)))
    (is (= [:read-write :read-write :read-write]
           (mapv :access (dialect/result-storage program 0))))))

(deftest untyped-ordered-void-bodies-decline-the-typed-effect-map
  (doseq [[label body]
          [["an untyped shared local cannot enter a typed region"
            '(let* [v (+ (clojure.core/aget x i) 1.0)]
                   (clojure.core/aset a i (float v))
                   (clojure.core/aset b i (float (* v v))))]
           ["transitively shared locals without retained types also decline"
            '(let* [u (+ (clojure.core/aget x i) 1.0)
                    v (* u 2.0)]
                   (clojure.core/aset a i (float v))
                   (clojure.core/aset b i (float u)))]]]
    (testing label
      (is (nil? (frontend/form->program
                 (list 'let* ['effect (list 'raster.par/map-void! 'i 'n body)] 'effect)
                 {:dtype :float
                  :array-types {'x :float 'a :float 'b :float 'indices :int}}))))))

(deftest mixed-unique-and-reduction-effects-enter-one-typed-effect-map
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (do
                   (if (> (clojure.core/aget x i) 0.0)
                     (clojure.core/aset
                      out
                      (raster.par/unique-index (clojure.core/aget slots i))
                      (float (clojure.core/aget x i))))
                   (raster.par/atomic-add! total 0 (float (clojure.core/aget x i)))
                   (raster.par/atomic-add! count 0 (int 1))))]
               effect)
        program (frontend/form->program
                 source {:dtype :float
                         :array-types {'x :float 'slots :int 'out :float
                                       'total :float 'count :int}})
        equation (first (dialect/equations program))
        {:keys [attributes destinations lambda]} (dialect/operation-parts equation)
        {:keys [body-results]} (dialect/lambda-parts lambda)
        effect-parts (mapv dialect/effect-parts body-results)]
    (is (= 'effect-map (dialect/operation-kind equation)))
    (is (= ['out 'total 'count] destinations))
    (is (= [:float :float :int] (:dtypes attributes)))
    (is (= [:unique :reduce :reduce]
           (mapv #(if (= :unique (:conflict %))
                    :unique (get-in % [:conflict :kind]))
                 effect-parts)))
    (is (= [:float :int]
           (mapv #(get-in % [:conflict :dtype]) (rest effect-parts))))
    (is (= [:write :read-write :read-write]
           (mapv :access (dialect/result-storage program 0))))
    (is (= [] (dialect/outputs program)))
    (is (= program (dialect/validate! program)))))

(deftest unproved-indirect-write-is-explicitly-sequential
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (do
                   (clojure.core/aset out (clojure.core/aget slots i)
                                      (float (clojure.core/aget x i)))
                   (raster.par/atomic-add! total 0 (float 1.0))))]
               effect)
        program (frontend/form->program
                 source {:dtype :float
                         :array-types {'x :float 'slots :int 'out :float 'total :float}})
        equation (first (dialect/equations program))
        {:keys [attributes lambda]} (dialect/operation-parts equation)
        effects (:body-results (dialect/lambda-parts lambda))]
    (is (= 'effect-map (dialect/operation-kind equation)))
    (is (= :sequential (:iteration-order attributes)))
    (is (= [:ordered :reduce]
           (mapv (fn [effect]
                   (let [conflict (:conflict (dialect/effect-parts effect))]
                     (if (keyword? conflict) conflict (:kind conflict))))
                 effects)))
    (is (= program (dialect/validate! program)))))

(deftest a-renamed-allocation-written-through-its-alias-is-scaffolding
  ;; `(let [y (float-array n) out y] (map-void! … (aset out i …)))`: the write goes through the
  ;; alias, so the allocation shares its physical identity and stays host scaffolding instead of
  ;; declining as a scalar binding without a dtype.
  (let [source '(let* [y (clojure.core/float-array n)
                       out y
                       step (raster.par/map-void! i n
                                                  (clojure.core/aset
                                                   out i (float (clojure.core/aget x i))))]
                      step)
        program (frontend/form->program source {:dtype :float :array-types {'x :float}
                                                :scalar-types {'n :long}})
        routed (route/attempt source :float {'x :float} {:scalar-types {'n :long}})]
    (is (= ['map] (mapv dialect/operation-kind (dialect/equations program))))
    (is (= :typed-soac (get-in routed [:stats :route])))
    (is (= '[y (clojure.core/float-array n) out y]
           (vec (take 4 (second (get-in routed [:program :source]))))))))

(deftest physical-allocation-capacity-retains-its-typed-scalar-definition
  ;; The allocation remains scaffolding, but its size is an AbstractValue shape consumed by
  ;; later graph allocation.  Dropping this pure scalar equation leaves a stable extent name
  ;; with no executable/provable definition at the C-family graph boundary.
  (let [source '(let* [rstr_extent_0 (* nrows width)
                       out (clojure.core/float-array rstr_extent_0)
                       step (raster.par/map-void! i nrows
                                                  (clojure.core/aset
                                                   out i (float (clojure.core/aget x i))))]
                      step)
        program (frontend/form->program source {:dtype :float :array-types {'x :float}
                                                :scalar-types {'nrows :long 'width :long
                                                               'rstr_extent_0 :long}})
        equations (dialect/equations program)]
    (is (= ['scalar 'map] (mapv dialect/operation-kind equations)))
    (is (= 'rstr_extent_0 (first (nth (first equations) 2))))
    (is (= :long (get-in (dialect/facts program) [:values 'rstr_extent_0 :dtype])))
    (is (= program (dialect/validate! program)))))

(deftest a-blas-gemm-call-is-the-contraction-it-computes
  ;; `(dgemm-nt! A B C m k n 1 0)` is `C[m,n] = A[m,k]·B[n,k]ᵀ`: the devirtualized BLAS effect
  ;; normalizes to the explicit `par/contract` whose typed equation is a segmented reduction
  ;; over `k` with free axes `[m n]`, so a block containing it no longer falls back to the host.
  (let [call (with-meta
               (list '.invk
                     'raster.linalg.blas/dgemm-nt!_m_floats_floats_floats_long_long_long_float_float-impl
                     'A 'B 'C 'm 'k 'n '(float 1.0) '(float 0.0))
               {:raster.op/original 'raster.linalg.blas/dgemm-nt!
                :raster.type/tag 'floats :tag 'floats})
        source (list 'let* ['C '(clojure.core/float-array (clojure.core/* m n)) 'r call] 'r)
        program (frontend/form->program
                 (frontend/normalize-source source)
                 {:dtype :float :array-types {'A :float 'B :float}
                  :scalar-types {'m :long 'k :long 'n :long}})
        equation (first (dialect/equations program))
        {:keys [attributes lambda]} (dialect/operation-parts equation)
        {:keys [body-results]} (dialect/lambda-parts lambda)]
    (is (= ['segmented-reduce] (mapv dialect/operation-kind (dialect/equations program))))
    (is (= '[[rstr_gemm_i_1 m] [rstr_gemm_j_1 n]] (:segment-axes attributes)))
    (is (= 'k (:extent attributes)))
    (is (= [:float] (:dtypes attributes)))
    (is (= '(clojure.core/+ (clojure.core/* rstr_gemm_j_1 %capture2) rstr_gemm_l_1)
           (-> body-results first (nth 2) (nth 2) (nth 2)))
        "the nt variant reads B as [n,k]: B[j*k + l]")
    (is (= [{:destination 'C :access :write :host-return :buffer}]
           (get-in (dialect/facts program) [:equations 1 :attributes :result-storage])))))

(deftest batched-blas-gemm-is-the-same-contraction-with-one-more-free-axis
  (doseq [[operation variant]
          [['raster.linalg.blas/batched-gemm-nn! :nn]
           ['raster.linalg.blas/batched-gemm-nt! :nt]]]
    (let [call (with-meta
                 (list '.invk (symbol "generated" (str (name operation) "-impl"))
                       'A 'B 'C 'batch 'm 'k 'n 'scale)
                 {:raster.op/original operation
                  :raster.type/tag 'floats :tag 'floats})
          source (list 'let* ['result call] 'result)
          options {:dtype :float :array-types {'A :float 'B :float 'C :float}
                   :scalar-types {'batch :long 'm :long 'k :long 'n :long
                                  'scale :float}}
          program (frontend/form->program (frontend/normalize-source source options) options)
          equation (first (dialect/equations program))
          {:keys [attributes lambda]} (dialect/operation-parts equation)
          body (first (:body-results (dialect/lambda-parts lambda)))
          b-index (some (fn [expression]
                          (when (and (seq? expression)
                                     (descriptor/aget-op?
                                      (descriptor/semantic-op expression))
                                     (= '%capture1
                                        (first (descriptor/call-args expression))))
                            (second (descriptor/call-args expression))))
                        (tree-seq coll? seq body))]
      (is (= 'segmented-reduce (dialect/operation-kind equation)))
      (is (= '[[rstr_gemm_batch_0 batch] [rstr_gemm_i_0 m] [rstr_gemm_j_0 n]]
             (:segment-axes attributes)))
      (is (= 'k (:extent attributes)))
      (is (some #{'scale} (dialect/operation-inputs equation)))
      (is (every? (set (flatten b-index))
                  '[rstr_gemm_batch_0 rstr_gemm_j_0 rstr_gemm_l_0])
          "B's ordinary row-major index retains the batch, column, and reduction axes")
      (is (= (case variant :nn 'rstr_gemm_j_0 :nt 'rstr_gemm_l_0)
             (last b-index))
          "the final coordinate distinguishes B[batch,k,n] from B[batch,n,k]")
      (is (= [{:destination 'C :access :write :host-return :buffer}]
             (get-in (dialect/facts program)
                     [:equations (second equation) :attributes :result-storage]))))))

(defn- accumulating-gemm-call
  [beta]
  (with-meta
    (list '.invk
          'raster.linalg.blas/dgemm!_m_floats_floats_floats_long_long_long_float_float-impl
          'A 'B 'C 'm 'k 'n '(float 1.0) beta)
    {:raster.op/original 'raster.linalg.blas/dgemm!
     :raster.type/tag 'floats :tag 'floats}))

(deftest a-blas-gemm-with-a-non-zero-beta-reads-its-destination-in-the-result-transform
  ;; `(dgemm! A B C m k n 1 beta)` is `C[i,j] := Σ_l A[i,l]·B[l,j] + beta·C[i,j]`: the same
  ;; contraction, with a result transform that reads the destination element it overwrites.
  ;; The destination is then read-write storage of one kernel rather than a host BLAS effect.
  (let [types {:dtype :float :array-types {'A :float 'B :float 'C :float}
               :scalar-types {'m :long 'k :long 'n :long 'beta :float}}
        program-for (fn [beta]
                      (let [source (list 'let* ['r (accumulating-gemm-call beta)] 'r)]
                        (frontend/form->program (frontend/normalize-source source types) types)))
        program (program-for '(float 1.0))
        equation (first (dialect/equations program))
        {:keys [attributes]} (dialect/operation-parts equation)
        transform (:result-transform attributes)]
    (is (= ['segmented-reduce] (mapv dialect/operation-kind (dialect/equations program))))
    (is (= '[[rstr_gemm_i_0 m] [rstr_gemm_j_0 n]] (:segment-axes attributes)))
    (is (= [{:value 'C :parameter '%result-operand0 :dtype :float
             :map (axis-map/of-axes '[[rstr_gemm_i_0 m] [rstr_gemm_j_0 n]])}]
           (:operands transform))
        "the destination is the one transform operand, read at the store coordinates")
    (is (= [] (:scalars transform)) "beta = 1 is folded")
    (is (= [{:destination 'C :access :read-write :host-return :buffer}]
           (get-in (dialect/facts program)
                   [:equations (second equation) :attributes :result-storage])))
    (testing "a scalar beta is a transform scalar"
      (let [program (program-for 'beta)
            {:keys [attributes]} (dialect/operation-parts (first (dialect/equations program)))]
        (is (= [{:value 'beta :parameter '%result-scalar0 :dtype :float}]
               (:scalars (:result-transform attributes))))))
    (testing "a beta that is neither a literal nor a scalar value stays a host call"
      (let [source (list 'let* ['r (accumulating-gemm-call '(clojure.core/aget S 0))] 'r)]
        (is (= source (frontend/normalize-source source types)))))))

(deftest scalar-cast-aliases-require-a-retained-widening-proof
  (doseq [[source-type cast eliminated?] [[:int 'long true] [:long 'long true]
                                         [:int 'int true] [:long 'int false]
                                         [:float 'long false] [nil 'int false]]]
    (let [source (list 'let* ['count (list cast 'n)
                             'result '(raster.par/map! out i count float (aget x i))]
                       'result)
          normalized (frontend/normalize-source
                      source {:scalar-types (if source-type {'n source-type} {})
                              :array-types {'x :float 'out :float}})
          map-source (nth (second normalized) 3)]
      (is (= (if eliminated? 'n 'count) (nth map-source 3))
          (str source-type " → " cast)))))

(deftest a-shadowed-cast-name-is-not-a-conversion-proof
  (let [source '(let* [int convert count (int n)
                       result (raster.par/map! out i count float (aget x i))]
                      result)]
    (is (= source (frontend/normalize-source
                   source {:scalar-types {'n :int} :array-types {'x :float 'out :float}})))))

(deftest narrowing-extents-remain-typed-scalar-equations
  (let [options {:dtype :float :array-types {'x :float 'out :float}
                 :scalar-types {'n :long}}
        normalized (frontend/normalize-source
                    '(let* [result (raster.par/map! out i (int n) float (aget x i))] result)
                    options)
        extent (first (second normalized))
        program (frontend/form->program normalized options)
        equations (dialect/equations program)]
    (is (= '(int n) (second (second normalized))))
    (is (= 'int (:tag (meta extent))))
    (is (= '[scalar map] (mapv dialect/operation-kind equations)))))

(deftest identical-checked-extents-reuse-only-dominating-immutable-values
  (let [source (fn [extent]
                 (list 'let*
                       ['a (list 'raster.par/map! 'out-a 'i extent 'float 1.0)
                        'b (list 'raster.par/map! 'out-b 'j extent 'float 2.0)]
                       'b))
        checked-casts (fn [form]
                        (filter #(and (seq? %) (= 'int (first %)))
                                (tree-seq coll? seq form)))
        immutable (frontend/normalize-source
                   (source '(int n))
                   {:scalar-types {'n :long}
                    :array-types {'out-a :float 'out-b :float}})
        array-backed (frontend/normalize-source
                      (source '(int (aget counts 0)))
                      {:array-types {'counts :long 'out-a :float 'out-b :float}})]
    (is (= 1 (count (checked-casts immutable)))
        "the first immutable checked extent dominates and proves the repeated conversion")
    (is (= 2 (count (checked-casts array-backed)))
        "array-backed checked extents remain distinct evaluations")))

(deftest fixed-rng-inputs-keep-their-ordered-checked-conversions
  (let [options {:dtype :long :array-types {'seeds :long}
                 :scalar-types {'n :long 'seed :long}}
        normalized (frontend/normalize-source
                    '(let* [result (raster.par/rng-fill! seeds n (int seed))] result) options)
        bindings (second normalized)
        program (frontend/form->program normalized options)]
    (is (= '(clojure.core/int n) (nth bindings 1)))
    (is (= '(clojure.core/long (int seed)) (nth bindings 3)))
    (is (= '[int long] (mapv #(-> % meta :raster.type/tag) [(nth bindings 0) (nth bindings 2)])))
    (is (= '[scalar scalar map] (mapv dialect/operation-kind (dialect/equations program))))
    (is (= normalized (frontend/normalize-source normalized options))
        "normalization re-entry does not introduce another set of guards")))

(deftest fixed-active-id-inputs-share-the-ordered-scalar-normalizer
  (let [options {:dtype :int :array-types {'ids :int}
                 :scalar-types {'n-active :long 'population :long 'seed :long}}
        normalized (frontend/normalize-source
                    '(let* [result (raster.par/active-ids!
                                    ids n-active (int population) (int seed))] result)
                    options)
        bindings (second normalized)
        program (frontend/form->program normalized options)]
    (is (= '(clojure.core/int n-active) (nth bindings 1)))
    (is (= '(clojure.core/long (int population)) (nth bindings 3)))
    (is (= '(clojure.core/long (int seed)) (nth bindings 5)))
    (is (= '[int long long]
           (mapv #(-> % meta :raster.type/tag)
                 [(nth bindings 0) (nth bindings 2) (nth bindings 4)])))
    (is (= '[scalar scalar scalar map]
           (mapv dialect/operation-kind (dialect/equations program))))
    (is (= normalized (frontend/normalize-source normalized options)))))

(deftest indexed-operation-counts-use-the-shared-scalar-normalizer
  (doseq [operation ['(raster.par/gather out x indices count)
                     '(raster.par/scatter! out x indices count)
                     '(raster.par/reduce-by-key out indices x count +)]
          cast ['long 'int]]
    (let [expression (apply list (assoc (vec operation) 4 (list cast 'n)))
          options {:dtype :float :array-types {'x :float 'out :float 'indices :int}
                   :scalar-types {'n :long}}
          normalized (frontend/normalize-source (list 'let* ['result expression] 'result)
                                                options)
          bindings (second normalized)
          program (frontend/form->program normalized options)
          equations (dialect/equations program)]
      (if (= cast 'int)
        (do (is (= '(int n) (second bindings)))
            (is (= 'int (:tag (meta (first bindings)))))
            (is (= 'scalar (dialect/operation-kind (first equations)))))
        (is (= 'n (nth (second bindings) 4))))
      (is (some? program) (str (first operation) " / " cast)))))

(deftest equal-sizes-spelled-through-host-bindings-are-one-extent
  ;; `n1 = seq`, `n2 = dff`, `(* n1 n2)` and `(* seq dff)` denote one size, and `(alength y)`
  ;; over `y = (float-array n)` is `n`. Without those identities the buffer `y` would receive
  ;; two shapes and the whole form would decline with :source-value-conflict.
  (let [source '(let* [^long n1 seq
                       ^long n2 dff
                       ^long total (clojure.core/* seq dff)
                       ^long again (clojure.core/* n1 n2)
                       y (clojure.core/float-array total)
                       fill (raster.par/map! y i again float (clojure.core/aget x i))
                       z (raster.par/pmap j (clojure.core/alength y) float
                                          (clojure.core/* 2.0 (clojure.core/aget y j)))]
                      z)
        normalized (frontend/normalize-source source {:array-types {'x :float}
                                                      :scalar-types {'seq :long 'dff :long}})
        program (frontend/form->program normalized
                                        {:dtype :float :array-types {'x :float}
                                         :scalar-types {'seq :long 'dff :long}})
        equations (dialect/equations program)
        maps (filter #(= 'map (dialect/operation-kind %)) equations)]
    (is (= '[scalar map map] (mapv dialect/operation-kind equations))
        "the canonical size keeps one scalar equation; its restatement is an alias")
    (is (= '[total total] (mapv dialect/operation-extent maps))
        "both maps iterate the one canonical extent")
    (is (= (get-in (dialect/facts program) [:values 'y :shape])
           (get-in (dialect/facts program) [:values 'z :shape]))
        "the allocation and its consumer agree on one shape")))

(deftest a-recomputed-array-read-is-not-an-alias-across-a-kernel
  ;; `n1 = (aget counts 0)` recomputed after a kernel that writes `counts` is a fresh value; only
  ;; array-free scalars and array lengths alias across kernels.
  (let [source '(let* [^long n0 (clojure.core/aget counts 0)
                       e0 (raster.par/map-void! i m (clojure.core/aset counts 0 (float 7.0)))
                       ^long n1 (clojure.core/aget counts 0)
                       e1 (raster.par/map-void! j n1 (clojure.core/aset out j (float 1.0)))]
                      e1)
        normalized (frontend/normalize-source source)
        pairs (partition 2 (second normalized))
        second-kernel (some (fn [[binder init]] (when (= 'e1 binder) init)) pairs)]
    (is (= 'n1 (nth second-kernel 2))
        "the second launch keeps its own post-kernel length")))

(deftest an-explicit-float-array-keeps-its-element-type-under-a-double-policy
  ;; `float-array` names its element type; only allocators without one (`alloc-like`,
  ;; `zeros-like`) follow the kernel dtype. A double kernel must not widen a float buffer.
  (let [source '(let* [^floats z (clojure.core/float-array n)
                       step (raster.par/map! z i n float (clojure.core/aget x i))]
                      step)
        program (frontend/form->program source {:dtype :double :array-types {'x :double}
                                                :scalar-types {'n :long}})]
    (is (= :float (get-in (dialect/facts program) [:values 'z :dtype])))))

(deftest sibling-loops-sharing-a-source-index-name-get-distinct-ssa-indices
  (let [source '(let* [effect
                       (raster.par/map-void!
                        r rows
                        (do
                          (loop* [i 0]
                            (if (clojure.core/< i feat)
                              (do (clojure.core/aset out (clojure.core/+ (clojure.core/* r feat) i)
                                                     (float 1.0))
                                  (recur (clojure.core/inc i)))))
                          (loop* [i 0]
                            (if (clojure.core/< i feat)
                              (do (clojure.core/aset out2 (clojure.core/+ (clojure.core/* r feat) i)
                                                     (float 2.0))
                                  (recur (clojure.core/inc i)))))))]
                      effect)
        program (frontend/form->program source {:dtype :float
                                                :array-types {'out :float 'out2 :float}
                                                :scalar-types {'rows :long 'feat :long}})
        equation (first (dialect/equations program))
        {:keys [lambda]} (dialect/operation-parts equation)
        loops (filter :loop (map dialect/effect-parts
                                 (:body-results (dialect/lambda-parts lambda))))]
    (is (= 2 (count loops)))
    (is (= 2 (count (distinct (map :index loops))))
        "each effect-loop binds its own index symbol")
    (is (= program (dialect/validate! program)))))

(deftest a-copy-constructor-over-an-array-is-not-a-length
  ;; `(float-array x)` with an array `x` copies it; only a positively scalar operand is a length.
  (let [source '(let* [y (clojure.core/float-array x)
                       z (raster.par/pmap i (clojure.core/alength y) float
                                          (clojure.core/aget y i))]
                      z)
        normalized (frontend/normalize-source source {:array-types {'x :float}})
        z-init (some (fn [[binder init]] (when (= 'z binder) init))
                     (partition 2 (second normalized)))]
    (is (not= 'x (nth z-init 2))
        "the extent stays the copy's length, not the copied array")))

(deftest a-region-local-reading-a-destination-serializes-the-loop-map
  ;; `previous = out[0]` observes earlier work items' writes, so the loop map cannot run its
  ;; items independently even though every row store is injective.
  (let [source '(let* [effect
                       (raster.par/map-void!
                        r rows
                        (let* [^float previous (clojure.core/aget out 0)]
                          (loop* [i 0]
                            (if (clojure.core/< i feat)
                              (do (clojure.core/aset out (clojure.core/+ (clojure.core/* r feat) i)
                                                     (float (clojure.core/+ previous 1.0)))
                                  (recur (clojure.core/inc i)))))))]
                      effect)
        program (frontend/form->program source {:dtype :float :array-types {'out :float}
                                                :scalar-types {'rows :long 'feat :long}})
        {:keys [attributes]} (dialect/operation-parts (first (dialect/equations program)))]
    (is (= :sequential (:iteration-order attributes)))))

(deftest effect-loop-locals-cannot-reference-later-locals
  (let [program (frontend/form->program
                 '(let* [effect
                         (raster.par/map-void!
                          r rows
                          (loop* [i 0]
                            (if (clojure.core/< i feat)
                              (let* [^float a (float (clojure.core/aget x (clojure.core/+ (clojure.core/* r feat) i)))
                                     ^float b (float (clojure.core/* a 2.0))]
                                (clojure.core/aset out (clojure.core/+ (clojure.core/* r feat) i) b)
                                (recur (clojure.core/inc i))))))]
                        effect)
                 {:dtype :float :array-types {'x :float 'out :float}
                  :scalar-types {'rows :long 'feat :long}})
        swapped (walk/postwalk
                 (fn [form]
                   (if (and (seq? form) (= 'effect-region (first form)) (= 2 (count (second form))))
                     (list 'effect-region (vec (reverse (second form))) (nth form 2))
                     form))
                 program)]
    (is (= program (dialect/validate! program)))
    (is (= :typed-soac-effect-loop
           (try (dialect/validate! swapped) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
        "a loop local may only reference the locals before it")))

(deftest a-unique-index-claim-is-checked-against-its-own-dependency-slice
  (testing "an unrelated array read does not authorize a claim on a decidable colliding index"
    (is (= :unique-index-not-provable
           (try (frontend/form->program
                 '(let* [effect (raster.par/map-void!
                                 i n
                                 (let* [^long k (long (clojure.core/aget slots i))]
                                   (clojure.core/aset out (raster.par/unique-index 0) (float k))))]
                        effect)
                 {:dtype :float :array-types {'slots :int 'out :float} :scalar-types {'n :long}})
                nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "a claim on an index that reads an array is honoured"
    (let [program (frontend/form->program
                   '(let* [effect (raster.par/map-void!
                                   i n
                                   (let* [^long k (long (clojure.core/aget slots i))]
                                     (clojure.core/aset out (raster.par/unique-index k) (float 1.0))))]
                          effect)
                   {:dtype :float :array-types {'slots :int 'out :float} :scalar-types {'n :long}})
          {:keys [attributes]} (dialect/operation-parts (first (dialect/equations program)))]
      (is (= :unique (:conflict attributes))))))

(deftest a-triangular-store-loop-is-not-certified-unique
  ;; `4r + i` for `i < r + 1` collides ((4,4) and (5,0) both write 20); the loop extent is not an
  ;; invariant radix, so the map stays :sequential.
  (let [program (frontend/form->program
                 '(let* [effect (raster.par/map-void!
                                 r rows
                                 (loop* [i 0]
                                   (if (clojure.core/< i (clojure.core/+ r 1))
                                     (do (clojure.core/aset out (clojure.core/+ (clojure.core/* r 4) i)
                                                            (float 1.0))
                                         (recur (clojure.core/inc i))))))]
                        effect)
                 {:dtype :float :array-types {'out :float} :scalar-types {'rows :long}})
        {:keys [attributes]} (dialect/operation-parts (first (dialect/equations program)))]
    (is (= :sequential (:iteration-order attributes)))))

(deftest an-unprovable-claim-declines-instead-of-failing-the-compile
  (let [routed (route/attempt '(let* [effect (raster.par/map-void!
                                              i n
                                              (clojure.core/aset out (raster.par/unique-index (clojure.core/* i stride))
                                                                 (float 1.0)))]
                                     effect)
                              :float {'out :float} {:scalar-types {'n :long 'stride :long}})]
    (is (= :unique-index-not-provable (get-in routed [:declined :reason])))))

(deftest a-destination-reading-transform-follows-the-contraction-dtype
  ;; A double-spelled GEMM (`doubles` call tag) compiled under the float policy writes a float
  ;; buffer; the destination it reads back is that same float buffer, so the transform's
  ;; operand and result dtypes are the contraction's, not the call tag's.
  (let [call (with-meta
               (list '.invk
                     'raster.linalg.blas/dgemm!_m_doubles_doubles_doubles_long_long_long_double_double-impl
                     'A 'B 'C 'm 'k 'n 1.0 1.0)
               {:raster.op/original 'raster.linalg.blas/dgemm!
                :raster.type/tag 'doubles :tag 'doubles})
        types {:dtype :float :array-types {'A :float 'B :float 'C :float}
               :scalar-types {'m :long 'k :long 'n :long}}
        program (frontend/form->program
                 (frontend/normalize-source (list 'let* ['r call] 'r) types) types)
        {:keys [attributes]} (dialect/operation-parts (first (dialect/equations program)))
        transform (:result-transform attributes)]
    (is (= [:float] (:dtypes attributes)))
    (is (= :float (:result-dtype transform)))
    (is (= [:float] (mapv :dtype (:operands transform))))))

(defn- effect-map-order
  "`:independent`/`:sequential` for an effect map; a proven unique offset write becomes a
   certified scatter, reported as `:unique`."
  [source types]
  (let [program (frontend/form->program (frontend/normalize-source source types) types)
        equation (last (dialect/equations program))
        {:keys [attributes]} (dialect/operation-parts equation)]
    (if (= 'scatter (dialect/operation-kind equation))
      (:conflict attributes)
      (:iteration-order attributes))))

(deftest a-uniform-array-read-is-an-invariant-offset
  ;; kv-append: cache[posbuf[0]·kvrow + i] over i < kvrow; posbuf[0] is read, never written,
  ;; at an index free of the map, so it is an invariant factor and the map is independent
  (let [types {:dtype :float :array-types {'src :float 'cache :float 'posbuf :long}
               :scalar-types {'kvrow :long}}
        map-over (fn [index]
                   (list 'let* ['effect (list 'raster.par/map-void! 'i 'kvrow
                                              (list 'clojure.core/aset 'cache index
                                                    '(clojure.core/aget src i)))]
                         'effect))]
    (is (= :unique
           (effect-map-order
            (map-over '(clojure.core/+ (clojure.core/* (clojure.core/aget posbuf 0) kvrow) i))
            types))
        "proven unique: a certified scatter, no ordering claim")
    (testing "a read that varies with the map index is data the algebra cannot see"
      (is (= :sequential
             (effect-map-order
              (map-over '(clojure.core/+ (clojure.core/* (clojure.core/aget posbuf i) kvrow) i))
              types))))
    (testing "a read of the destination itself is not invariant"
      (is (= :sequential
             (effect-map-order
              (map-over '(clojure.core/+ (clojure.core/* (clojure.core/aget cache 0) kvrow) i))
              types))))))

(deftest stores-at-one-address-in-exclusive-arms-are-one-write
  ;; attention prefill: both arms write sc[(i·n-q + hq)·nrows + j]; one work item, one element
  (let [source '(let* [effect (raster.par/map-void!
                               idx (clojure.core/* nrows (clojure.core/* n-q nrows))
                               (let* [^long per-i (clojure.core/* n-q nrows)
                                      ^long i (clojure.core/quot idx per-i)
                                      ^long rest0 (clojure.core/rem idx per-i)
                                      ^long hq (clojure.core/quot rest0 nrows)
                                      ^long j (clojure.core/rem rest0 nrows)
                                      ^long row (clojure.core/+ (clojure.core/* i n-q) hq)]
                                 (if (clojure.core/< i j)
                                   (clojure.core/aset sc (clojure.core/+ (clojure.core/* row nrows) j)
                                                      (float -1.0e30))
                                   (clojure.core/aset sc (clojure.core/+ (clojure.core/* row nrows) j)
                                                      (clojure.core/aget q j)))))]
                  effect)]
    (is (= :unique
           (effect-map-order source {:dtype :float :array-types {'sc :float 'q :float}
                                     :scalar-types {'nrows :long 'n-q :long}})))))

(deftest host-control-cannot-hide-an-unequated-parallel-leaf
  (let [source
        '(let* [seed (raster.par/map! initial i n float (aget x i))]
           (loop* [h 0 acc seed]
             (if (< h heads)
               (let* [next (raster.par/map! scratch i n float
                                             (+ (aget acc i) (aget x i)))]
                 (recur (inc h) next))
               acc)))]
    (is (nil? (frontend/form->program
               source {:dtype :float
                       :array-types {'initial :float 'scratch :float 'x :float}
                       :scalar-types {'heads :long 'n :long}}))
        "a validated TypedSOAC program must own every parallel leaf as an equation")))
