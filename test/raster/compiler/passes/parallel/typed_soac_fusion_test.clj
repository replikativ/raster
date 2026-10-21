(ns raster.compiler.passes.parallel.typed-soac-fusion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as typed-fusion]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]))

;; Positive fusion fixtures explicitly request IEEE narrowing. The paired
;; checked-float controls below retain the source exception/materialization fence.
(def ^:private map-map-source
  '(let* [y (raster.par/pmap i n float (clojure.core/unchecked-float (* (aget x i) (aget x i))))
          z (raster.par/pmap j n float (clojure.core/unchecked-float (+ (aget y j) 1.0)))]
         z))

(def ^:private horizontal-map-source
  '(let* [u (raster.par/pmap i n float (clojure.core/unchecked-float (* (aget a i) 2.0)))
          v (raster.par/pmap j n float (clojure.core/unchecked-float (+ (aget b j) 1.0)))]
         [u v]))

(defn- source-program
  ([source array-types]
   (source-program source array-types {'n :long}))
  ([source array-types scalar-types]
   (frontend/form->program source {:dtype :float
                                   :array-types array-types
                                   :scalar-types scalar-types})))

(deftest fusion-placement-belongs-only-to-the-surviving-boundary
  (doseq [left [{:source-binding-id 2}
                {:legacy-soac-id 2}
                {}]]
    (let [merged (#'typed-fusion/merge-provenance
                  (assoc left :fused-equations [2])
                  {:source-binding-id 5 :legacy-soac-id 5 :fused-equations [5]}
                  :horizontal-map)]
      (is (= left (select-keys merged [:source-binding-id :legacy-soac-id])))
      (is (= [2 5] (:fused-equations merged)))
      (is (= :horizontal-map (:fusion merged))))))

(deftest horizontal-motion-declines-missing-local-allocation-origin
  (let [source '(let* [out-a (clojure.core/float-array n)
                       u (raster.par/map! out-a i n float (aget a i))
                       out-b (clojure.core/float-array n)
                       v (raster.par/map! out-b j n float (aget b j))]
                      [u v])
        program (source-program source {'a :float 'b :float})]
    (doseq [alter [(fn [allocations] (filterv #(not= 'out-b (:destination %)) allocations))
                  (fn [allocations] (mapv #(if (= 'out-b (:destination %))
                                            (dissoc % :source-binding-id) %) allocations))]]
      (let [facts (update-in (dialect/facts program) [:attributes :allocations] alter)
            incomplete (dialect/make facts (dialect/equations program) (dialect/outputs program))
            [_ stats] (typed-fusion/fusion-fixpoint incomplete)
            infos (mapv #'typed-fusion/equation-info (dialect/equations incomplete))]
        (is (zero? (:horizontal stats)))
        ;; Exercise the availability guard directly with conflicting input evidence. Do not
        ;; weaken the dialect validator to admit an inconsistent whole-program input boundary.
        (with-redefs [dialect/facts (constantly (update facts :inputs conj 'out-b))]
          (is (nil? (#'typed-fusion/horizontal-placement-index
                     incomplete infos 0 1
                     (#'typed-fusion/horizontal-boundary incomplete (nth infos 0))
                     (#'typed-fusion/horizontal-boundary incomplete (nth infos 1))))))))))

(deftest map-map-fusion-has-the-certified-functional-result
  (let [typed-input (source-program map-map-source {'x :float})
        [typed-result typed-stats] (typed-fusion/fusion-fixpoint typed-input)
        operation (dialect/operation-parts (first (dialect/equations typed-result)))]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} typed-stats))
    (is (= 1 (count (dialect/equations typed-result))))
    (is (= 'map (:kind operation)))
    (is (= '[(clojure.core/float
              (clojure.core/unchecked-float
               (+ (clojure.core/float (clojure.core/unchecked-float (* %element0 %element0))) 1.0)))]
           (mapv projection/scalar-folds->source
                 (:body-results (dialect/lambda-parts (:lambda operation))))))
    (is (= '[n x] (:inputs (dialect/facts typed-result))))
    (is (not (contains? (:values (dialect/facts typed-result)) 'y)))
    (let [{:keys [parameters body-results]} (dialect/lambda-parts (:lambda operation))
          fused (eval (list 'fn parameters (projection/scalar-folds->source (first body-results))))
          source (eval (list 'fn '[x n] map-map-source))]
      (doseq [values [[1.25 -2.5 0.0] [Float/MAX_VALUE -0.0 (- Float/MAX_VALUE)]]]
        (let [input (float-array values)]
          (is (= (mapv #(Float/floatToRawIntBits %) (source input 3))
                 (mapv #(Float/floatToRawIntBits (fused %)) input))))))))

(deftest fusion-preserves-pointwise-storage-capacity-obligations
  (doseq [[source array-types expected]
          [[map-map-source {'x :float} #{'x}]
           [horizontal-map-source {'a :float 'b :float} #{'a 'b}]]]
    (let [program (frontend/form->program
                   source {:dtype :float :array-types array-types :scalar-types {'n :long}
                           :values (into {} (map (fn [[id dtype]]
                                                  [id (av/tensor {:dtype dtype
                                                                  :shape [(list 'extent id)]})]))
                                         array-types)})
          [fused _] (typed-fusion/fusion-fixpoint program)
          operation (dialect/operation-parts (first (dialect/equations fused)))]
      (is (= 1 (count (dialect/equations fused))))
      (is (= expected (set (get-in operation [:attributes :attributes :pointwise-storage-inputs]))))
      (is (= fused (dialect/validate! fused))))))

(deftest vertical-fusion-preserves-a-checked-producer-completion-boundary
  (let [source '(let* [y (raster.par/map! tmp i n int (clojure.core/aget input i))
                       z (raster.par/map! out j n int (clojure.core/aget y j))]
                      z)
        program (frontend/form->program
                 source {:dtype :int
                         :array-types {'input :long 'tmp :int 'out :int}
                         :scalar-types {'n :long}})
        [fused stats] (typed-fusion/fusion-fixpoint program)
        conversions (filter dialect/scalar-convert-form?
                            (tree-seq coll? seq (dialect/equations fused)))]
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations fused))))
    (is (= 1 (count (filter #(= :trap
                                (get-in (dialect/scalar-convert-parts %)
                                        [:attributes :overflow]))
                           conversions)))
        "the producer check remains in its materialized equation")))

(deftest checked-float-producer-preserves-the-materialization-boundary
  (doseq [[producer expected-fusions]
          [['(clojure.core/aget input i) 0]
           ['(clojure.core/unchecked-float (clojure.core/aget input i)) 1]]]
    (let [source (list 'let* ['y (list 'raster.par/pmap 'i 'n 'float producer)
                              'z '(raster.par/pmap j n float (clojure.core/aget y j))] 'z)
          program (source-program source {'input :double})
          [result stats] (typed-fusion/fusion-fixpoint program)]
      (is (= expected-fusions (:vertical stats)))
      (is (= (- 2 expected-fusions) (count (dialect/equations result)))))))

(deftest checked-float-producer-throws-before-later-output-writes
  (let [source '(let* [y (raster.par/pmap i n float (clojure.core/aget input i))
                       z (raster.par/map! out j n float (clojure.core/aget y j))] z)
        program (source-program source {'input :double 'out :float})
        [result stats] (typed-fusion/fusion-fixpoint program)
        {:keys [lambda]} (dialect/operation-parts (first (dialect/equations result)))
        {:keys [parameters body-results]} (dialect/lambda-parts lambda)
        parameter (with-meta (first parameters) {:tag 'double})
        produce (eval (list 'fn [parameter]
                            (projection/scalar-folds->source (first body-results))))
        execute (eval (list 'fn [(with-meta 'input {:tag 'doubles})
                                 (with-meta 'out {:tag 'floats}) 'n] source))]
    (is (zero? (:vertical stats)))
    (is (= :trap (get-in (dialect/scalar-convert-parts (first body-results))
                         [:attributes :overflow])))
    (is (thrown? IllegalArgumentException (produce Double/MAX_VALUE)))
    (let [output (float-array [9.0 9.0])]
      (is (thrown? IllegalArgumentException
                   (execute (double-array [1.0 Double/MAX_VALUE]) output 2)))
      (is (= [9.0 9.0] (vec output))
          "a later consumer cannot write before checked producer completion"))))

(deftest vertical-candidate-enumeration-checks-each-producer-once
  (let [program (source-program
                 '(let* [u (raster.par/pmap i n float (+ (aget x i) 1.0))
                         v (raster.par/pmap j n float (+ (aget u j) 2.0))
                         w (raster.par/pmap k n float (+ (aget v k) 3.0))
                         z (raster.par/pmap l n float (+ (aget w l) 4.0))]
                        z)
                 {'x :float})
        calls (atom [])
        proof-var #'typed-fusion/exceptional-conversion-region?
        original @proof-var
        enumerate #(mapv (juxt :producer-index :consumer-index)
                         (#'typed-fusion/vertical-candidates program nil))
        candidates (with-redefs-fn
                     {proof-var (fn [p producer]
                                  (swap! calls conj (:id producer))
                                  (original p producer))}
                     enumerate)]
    (is (= [[0 1] [1 2] [2 3]] candidates))
    (is (= 3 (count @calls)))
    (is (every? #(= 1 %) (vals (frequencies @calls))))
    ;; There is no retained authority across calls or changed programs.
    (reset! calls [])
    (is (empty? (with-redefs-fn
                  {proof-var (fn [_ producer]
                               (swap! calls conj (:id producer))
                               true)}
                  enumerate)))
    (is (= 3 (count @calls)))
    (is (every? #(= 1 %) (vals (frequencies @calls))))
    (is (empty?
         (with-redefs-fn
           {proof-var (fn [& _] (throw (ex-info "unused producer proof was forced" {})))}
           #(#'typed-fusion/vertical-candidates
              (source-program '(let* [u (raster.par/pmap i n float (aget x i))] u)
                              {'x :float})
              nil))))))

(deftest unrelated-vertical-pairs-never-force-effect-analysis
  (let [proof-var #'typed-fusion/exceptional-conversion-region?]
    (doseq [[source arrays scalars]
            [['(let* [u (raster.par/pmap i n float (+ (aget a i) 1.0))
                      v (raster.par/pmap j n float (+ (aget b j) 2.0))] [u v])
              {'a :float 'b :float} {'n :long}]
             ['(let* [u (raster.par/pmap i n float (+ (aget a i) 1.0))
                      v (raster.par/pmap j m float (+ (aget u j) 2.0))] v)
              {'a :float} {'n :long 'm :long}]]]
      (let [program (source-program source arrays scalars)]
        (is (empty?
             (with-redefs-fn
               {proof-var (fn [& _] (throw (ex-info "unrelated scalar proof was forced" {})))}
               #(#'typed-fusion/vertical-candidates program nil))))))))

(deftest structural-prefilter-preserves-ordered-matching-candidates
  (let [program (source-program
                 '(let* [u (raster.par/pmap i n float (+ (aget a i) 1.0))
                         unrelated (raster.par/pmap j n float (+ (aget b j) 2.0))
                         v (raster.par/pmap k n float (+ (aget u k) 3.0))]
                        [unrelated v])
                 {'a :float 'b :float})
        calls (atom [])
        proof-var #'typed-fusion/exceptional-conversion-region?
        original @proof-var
        enumerate #(#'typed-fusion/vertical-candidates program nil)
        candidates (with-redefs-fn
                     {proof-var (fn [p producer]
                                  (swap! calls conj (:id producer))
                                  (original p producer))}
                     enumerate)]
    (is (= [[0 2]] (mapv (juxt :producer-index :consumer-index) candidates)))
    (is (= 1 (count @calls)))
    (reset! calls [])
    (is (empty? (with-redefs-fn
                  {proof-var (fn [_ producer] (swap! calls conj (:id producer)) true)}
                  enumerate)))
    (is (= 1 (count @calls)))
    (is (= 3 (count (dialect/equations program))))))

(deftest horizontal-candidate-enumeration-checks-each-map-once
  (let [program (source-program
                 '(let* [u (raster.par/pmap i n float (+ (aget a i) 1.0))
                         v (raster.par/pmap j m float (+ (aget b j) 2.0))
                         w (raster.par/pmap k k-size float (+ (aget c k) 3.0))
                         z (raster.par/pmap l l-size float (+ (aget d l) 4.0))]
                        [u v w z])
                 {'a :float 'b :float 'c :float 'd :float}
                 {'n :long 'm :long 'k-size :long 'l-size :long})
        calls (atom [])
        proof-var #'typed-fusion/exceptional-conversion-region?
        original @proof-var]
    (is (nil? (with-redefs-fn
                {proof-var (fn [p map-info]
                             (swap! calls conj (:id map-info))
                             (original p map-info))}
                #(#'typed-fusion/horizontal-candidate program))))
    (is (= 4 (count @calls)))
    (is (every? #(= 1 %) (vals (frequencies @calls))))
    (reset! calls [])
    ;; A fresh enumeration does not reuse the prior nontrapping result. Short-circuiting
    ;; rejects each pair at its left operand, so the final map needs no check at all.
    (is (nil? (with-redefs-fn
                {proof-var (fn [_ map-info] (swap! calls conj (:id map-info)) true)}
                #(#'typed-fusion/horizontal-candidate program))))
    (is (= 3 (count @calls)))
    (is (every? #(= 1 %) (vals (frequencies @calls))))
    (is (nil? (with-redefs-fn
                {proof-var (fn [& _] (throw (ex-info "unpaired map proof was forced" {})))}
                #(#'typed-fusion/horizontal-candidate
                  (source-program '(let* [u (raster.par/pmap i n float (aget a i))] u)
                                  {'a :float}))))))
  (let [program (source-program horizontal-map-source {'a :float 'b :float})]
    (is (= {:left-index 0 :right-index 1}
           (select-keys (#'typed-fusion/horizontal-candidate program)
                        [:left-index :right-index])))))

(deftest horizontal-fusion-preserves-checked-map-completion-order
  (let [source '(let* [u (raster.par/map! left i n int
                                           (clojure.core/aget long-input i))
                       v (raster.par/map! right j n int 1)]
                      [u v])
        program (frontend/form->program
                 source {:dtype :int
                         :array-types {'long-input :long 'left :int 'right :int}
                         :scalar-types {'n :long}})
        [result stats] (typed-fusion/fusion-fixpoint program)
        execute (eval (list 'fn '[long-input left right n] source))
        long-input (long-array [(inc (long Integer/MAX_VALUE))])
        left (int-array [17])
        right (int-array [23])]
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations result))))
    (is (try (execute long-input left right 1) false
             (catch ArithmeticException _ true)))
    (is (= [17] (vec left)))
    (is (= [23] (vec right))
        "the later independent destination is unchanged when the first map conversion traps")))

(deftest fused-float-map-retains-its-materialization-rounding
  (let [source '(let* [y (raster.par/pmap i n float
                                           (* (clojure.core/aget x i)
                                              (clojure.core/aget x i)))
                       z (raster.par/pmap j n float
                                           (+ (clojure.core/aget y j) -16785408.0))]
                      z)
        program (frontend/form->program source {:dtype :float
                                                :array-types {'x :float}
                                                :scalar-types {'n :long}})
        [fused stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations fused))
        {:keys [lambda]} (dialect/operation-parts equation)
        {:keys [parameters body-results]} (dialect/lambda-parts lambda)
        projected (projection/scalar-folds->source (first body-results))
        execute (eval (list 'fn [(first parameters)] projected))]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (zero? (double (execute (float 4097.0))))
        "the producer float cast rounds 16785409 to 16785408 before consumer arithmetic")))

(deftest map-reduce-fusion-has-the-certified-functional-result
  (let [source '(let* [y (raster.par/pmap i n float (* (aget x i) (aget x i)))
                       sum (raster.par/reduce acc 0.0 j n (+ acc (aget y j)))]
                      sum)
        typed-input (source-program source {'x :float})
        [typed-result typed-stats] (typed-fusion/fusion-fixpoint typed-input)]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} typed-stats))
    (is (= 'reduce (first (nth (first (dialect/equations typed-result)) 3))))
    (is (not-any? #{'y} (flatten (dialect/equations typed-result))))))

(deftest map-scan-fusion-preserves-the-certified-resident-boundary
  (doseq [[source mode result-shape]
          [['(let* [mapped (raster.par/pmap i n float
                                            (* (clojure.core/aget x i) 2.0))
                    result (raster.par/scan out acc 0.0 j n float
                                            (+ acc (clojure.core/aget mapped j)))]
                   result)
            :inclusive '[n]]
           ['(let* [mapped (raster.par/pmap i n float
                                            (* (clojure.core/aget x i) 2.0))
                    result (raster.par/scan-exclusive out acc 0.0 j n float
                                                      (+ acc (clojure.core/aget mapped j)))]
                   result)
            :exclusive '[(clojure.core/inc n)]]]]
    (let [program (frontend/form->program source
                                          {:dtype :float
                                           :array-types {'x :float 'out :float}})
          mapped-id (first (nth (first (dialect/equations program)) 2))
          [result stats] (typed-fusion/fusion-fixpoint program)
          equation (first (dialect/equations result))
          operation (dialect/operation-parts equation)
          result-id (first (nth equation 2))
          equation-facts (get-in (dialect/facts result) [:equations (second equation)])]
      (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
      (is (= 1 (count (dialect/equations result))))
      (is (= 'scan (:kind operation)))
      (is (= mode (get-in operation [:attributes :mode])))
      (is (= '[x] (:arrays operation)))
      (is (= result-shape (get-in (dialect/facts result) [:values result-id :shape])))
      (is (= #{:memory/write} (:effects equation-facts)))
      (is (= {result-id 'out} (:aliases equation-facts)))
      (is (= [{:destination 'out :access :write :host-return :buffer}]
             (get-in equation-facts [:attributes :result-storage])))
      (is (not-any? #{mapped-id} (flatten (dialect/equations result))))
      (is (= result (dialect/validate! result))))))

(deftest map-scan-fusion-declines-an-aliased-destination-read
  (let [source '(let* [mapped (raster.par/pmap i n float
                                               (* (clojure.core/aget x i) 2.0))
                       result (raster.par/scan x acc 0.0 j n float
                                               (+ acc (clojure.core/aget mapped j)))]
                      result)
        program (frontend/form->program source {:dtype :float :array-types {'x :float}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations result)))
        "materializing the map preserves its complete read before the scan mutates x")))

(deftest horizontal-map-fusion-has-the-certified-functional-result
  (let [typed-input (source-program horizontal-map-source {'a :float 'b :float})
        [typed-result typed-stats] (typed-fusion/fusion-fixpoint typed-input)
        equation (first (dialect/equations typed-result))
        lambda (nth (nth equation 3) 4)
        region (dialect/lambda-parts lambda)]
    (is (= {:vertical 0 :horizontal 1 :iterations 2} typed-stats))
    (is (= '[u v] (nth equation 2)))
    (is (= '[%element0 %element1] (second lambda)))
    (is (= '[(clojure.core/float (clojure.core/unchecked-float (* %element0 2.0)))
             (clojure.core/float (clojure.core/unchecked-float (+ %element1 1.0)))]
           (mapv projection/scalar-folds->source (:body-results region))))
    (let [expressions (mapv projection/scalar-folds->source (:body-results region))
          fused (eval (list 'fn (:parameters region) expressions))
          source (eval (list 'fn '[a b n] horizontal-map-source))
          a (float-array [Float/MAX_VALUE -0.0 -2.5])
          b (float-array [-1.0 Float/MAX_VALUE 0.25])
          [left right] (source a b 3)
          bits #(mapv (fn [x] (Float/floatToRawIntBits x)) %)]
      (is (= (mapv (fn [a b] (bits [a b])) left right)
             (mapv (fn [a b] (bits (fused a b))) a b))))))

(deftest captured-map-fusion-has-the-certified-functional-result
  (let [source '(let* [y (raster.par/pmap i n float (* (aget x i) scale))
                       z (raster.par/pmap j n float (+ (aget y j) bias))]
                      z)
        typed-input (source-program source {'x :float}
                                    {'n :long 'scale :float 'bias :float})
        [typed-result typed-stats] (typed-fusion/fusion-fixpoint typed-input)
        equation (first (dialect/equations typed-result))
        operation (nth equation 3)
        lambda (nth operation 4)
        region (dialect/lambda-parts lambda)]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} typed-stats))
    (is (= '[bias scale] (nth operation 3)))
    (is (= '[%element0 %capture0 %capture1] (second lambda)))
    (is (= '[(clojure.core/float
              (+ (clojure.core/float (* %element0 %capture1)) %capture0))]
           (mapv projection/scalar-folds->source (:body-results region))))))

(deftest typed-multi-consumer-placement-is-hardware-costed
  (let [source '(let* [a (raster.par/pmap i n float
                                          (Math/exp (Math/exp (Math/exp (aget x i)))))
                       b (raster.par/pmap j n float (* (aget a j) 2.0))
                       c (raster.par/reduce acc 0.0 k n (+ acc (aget a k)))]
                      [b c])
        program (source-program source {'x :float})
        poor-am {:ridge {:float 2.0}}
        rich-am {:ridge {:float 100.0}}
        [materialized materialized-stats] (typed-fusion/fusion-fixpoint program poor-am)
        [recomputed recomputed-stats] (typed-fusion/fusion-fixpoint program rich-am)
        materialized-witness (-> materialized-stats :placements first)
        recomputed-witness (-> recomputed-stats :placements first)]
    (is (= 0 (:vertical materialized-stats)))
    (is (= 3 (count (dialect/equations materialized))))
    (is (= :materialize (:decision materialized-witness)))
    (is (= '[1 2] (:consumers materialized-witness)))
    (is (= (:placements materialized-stats)
           (get-in (dialect/facts materialized) [:attributes :fusion/placements])))

    (is (= 2 (:vertical recomputed-stats)))
    (is (= 2 (count (dialect/equations recomputed))))
    (is (= :recompute (:decision recomputed-witness)))
    (is (= 2 (:consumer-count recomputed-witness)))
    (is (not-any? #{'a} (mapcat #(nth % 2) (dialect/equations recomputed))))
    (is (= recomputed (dialect/validate! recomputed)))))

(deftest typed-fusion-rewrites-the-canonical-reduction-region
  (let [source '(let* [tmp (raster.par/pmap i n float (* (aget x i) 2.0))
                       sum (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]
                      sum)
        program (source-program source {'x :float})
        [fused _] (typed-fusion/fusion-fixpoint program)
        reconstructed (dialect/equations fused)
        conversions (filter dialect/scalar-convert-form?
                            (tree-seq coll? seq reconstructed))]
    (is (not-any? #{'tmp} (flatten reconstructed))
        "the eliminated map result cannot survive in ProductReduction")
    (is (some #(= {:source-dtype :double :target-dtype :float
                   :rounding :nearest-even :overflow :ieee
                   :source-op 'clojure.core/float}
                  (:attributes (dialect/scalar-convert-parts %)))
              conversions)
        "the materialized map conversion is part of the inlined value semantics")))

(deftest scalar-bound-dense-result-fuses-with-its-elementwise-consumer
  (let [source '(let* [rows (clojure.core/alength b)
                       h (raster.par/pmap i rows double (+ (aget b i) 1.0))
                       a (raster.par/pmap k (clojure.core/alength h) double
                                          (max 0.0 (aget h k)))]
                      a)
        program (frontend/form->program source {:dtype :double
                                                :array-types {'b :double}
                                                :scalar-types {'rows :long}})
        [fused stats] (typed-fusion/fusion-fixpoint program)
        equations (dialect/equations fused)]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 2 (count equations)) "the scalar extent and one fused map remain")
    (is (= '[a] (nth (second equations) 2)))
    (is (= 'rows (get-in (dialect/operation-parts (second equations))
                         [:attributes :extent])))
    (is (not-any? #{'h} (flatten equations))
        "alength of the producer is normalized to its certified result shape")
    (is (= fused (dialect/validate! fused)))))

(deftest effectful-equations-do-not-fuse
  (let [program (source-program map-map-source {'x :float})
        facts (-> (dialect/facts program)
                  (update-in [:equations 0 :effects] conj :memory/write)
                  (assoc :effects #{:memory/write}))
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

(deftest vertical-fusion-does-not-recompute-a-read-after-an-intervening-write
  (let [source '(let* [producer (raster.par/pmap i n float
                                                 (clojure.core/aget x i))
                       middle (raster.par/map! x j n float
                                               (+ (clojure.core/aget x j) 1.0))
                       consumer (raster.par/pmap k n float
                                                 (* (clojure.core/aget producer k) 2.0))]
                      consumer)
        program (frontend/form->program source {:dtype :float :array-types {'x :float}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 3 (count (dialect/equations result))))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (some #{'producer} (flatten (dialect/equations result)))
        "the consumer must read the value captured before x is mutated")))

(deftest horizontal-fusion-does-not-move-a-read-before-an-intervening-write
  (let [source '(let* [left (raster.par/pmap i n float
                                             (+ (clojure.core/aget a i) 1.0))
                       middle (raster.par/map! x j n float
                                               (+ (clojure.core/aget x j) 1.0))
                       right (raster.par/pmap k n float
                                              (* (clojure.core/aget x k) 2.0))]
                      [left right])
        program (frontend/form->program
                 source {:dtype :float :array-types {'a :float 'x :float}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 3 (count (dialect/equations result))))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= '[left right] (dialect/outputs result))
        "the right-hand read consumes middle's physical result, not another public output")
    (is (= '[left middle right] (mapv #(first (nth % 2)) (dialect/equations result)))
        "the intervening write remains ordered between the two reads")
    (is (= #{:memory/write} (get-in (dialect/facts result) [:equations 1 :effects])))
    (is (= [{:destination 'x :access :read-write :host-return :buffer}]
           (get-in (dialect/facts result) [:equations 1 :attributes :result-storage])))))

(deftest aliased-equations-decline-unproved-fusion
  (let [program (source-program map-map-source {'x :float})
        facts (assoc-in (dialect/facts program) [:equations 0 :aliases] {'y 'x})
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

(deftest vertical-fusion-composes-local-ssa-regions
  (let [program (source-program map-map-source {'x :float})
        equations (dialect/equations program)
        equations (mapv (fn [equation]
                          (let [operation (dialect/operation-parts equation)
                                {:keys [parameters body-results]}
                                (dialect/lambda-parts (:lambda operation))]
                            (list '= (second equation) (nth equation 2)
                                  (list 'map (:attributes operation) (:arrays operation)
                                        (:captures operation)
                                        (dialect/lambda-form
                                         parameters
                                         [(dialect/local-value
                                           'shared :float (first body-results))]
                                         '[shared])))))
                        equations)
        program (dialect/make (dialect/facts program)
                              equations
                              (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 1 (count (dialect/equations result))))
    (is (= [{:id 'rstr_producer_local_0
             :dtype :float
             :init '(clojure.core/float (* %element0 %element0))}
            {:id 'rstr_consumer_local_0
             :dtype :float
             :init '(clojure.core/float (+ rstr_producer_local_0 1.0))}]
           (mapv #(update % :init projection/scalar-folds->source) locals)))
    (is (= '[rstr_consumer_local_0] body-results))
    (is (= result (dialect/validate! result)))))

(deftest horizontal-fusion-alpha-renames-and-composes-local-ssa-regions
  (let [program (source-program horizontal-map-source {'a :float 'b :float})
        equations
        (mapv (fn [equation]
                (let [operation (dialect/operation-parts equation)
                      {:keys [parameters body-results]}
                      (dialect/lambda-parts (:lambda operation))]
                  (list '= (second equation) (nth equation 2)
                        (list 'map (:attributes operation) (:arrays operation)
                              (:captures operation)
                              (dialect/lambda-form
                               parameters
                               [(dialect/local-value 'shared :float (first body-results))]
                               '[shared])))))
              (dialect/equations program))
        program (dialect/make (dialect/facts program) equations (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= {:vertical 0 :horizontal 1 :iterations 2} stats))
    (is (= ['rstr_left_local_0 'rstr_right_local_0] (mapv :id locals)))
    (is (= '[(clojure.core/float (clojure.core/unchecked-float (* %element0 2.0)))
             (clojure.core/float (clojure.core/unchecked-float (+ %element1 1.0)))]
           (mapv (comp projection/scalar-folds->source :init) locals)))
    (is (= '[rstr_left_local_0 rstr_right_local_0] body-results))
    (is (= result (dialect/validate! result)))))

(deftest fusion-preserves-constituent-equation-facts
  (let [program (source-program map-map-source {'x :float})
        facts (-> (dialect/facts program)
                  (assoc-in [:equations 0 :attributes] {:source :producer})
                  (assoc-in [:equations 1 :attributes] {:source :consumer}))
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result _] (typed-fusion/fusion-fixpoint program)
        constituents (get-in (dialect/facts result)
                             [:equations 1 :attributes :fusion/constituents])]
    (is (= #{0 1} (set (keys constituents))))
    (is (= {:source :producer} (get-in constituents [0 :attributes])))
    (is (= {:source :consumer} (get-in constituents [1 :attributes])))))

(deftest horizontal-fusion-does-not-move-across-a-late-producer
  (let [tensor (av/tensor {:dtype :float :shape '[n]})
        extent (av/tensor {:dtype :long :shape []})
        equation-facts {'left (dialect/default-equation-facts)
                        'middle (assoc (dialect/default-equation-facts)
                                       :effects #{:memory/read})
                        'right (dialect/default-equation-facts)}
        program
        (dialect/make
         (dialect/default-program-facts
          {:values {'x tensor 'z tensor 'left-value tensor 'middle-value tensor
                    'right-value tensor 'n extent}
           :inputs '[n x z]
           :equations equation-facts
           :effects #{:memory/read}})
         [(list '= 'left '[left-value]
                (list 'map {:index 'i :extent 'n} '[x] []
                      (dialect/lambda-form '[x-element] '[(* x-element 2.0)])))
          (list '= 'middle '[middle-value]
                (list 'map {:index 'i :extent 'n} '[z] []
                      (dialect/lambda-form '[z-element] '[(+ z-element 1.0)])))
          (list '= 'right '[right-value]
                (list 'map {:index 'i :extent 'n} '[middle-value] []
                      (dialect/lambda-form '[middle-element]
                                           '[(* middle-element 3.0)])))]
         '[left-value right-value])
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))))

(defn- contract-result-map-program
  ([map-expression] (contract-result-map-program map-expression 'D))
  ([map-expression destination] (contract-result-map-program map-expression destination {}))
  ([map-expression destination options]
  (frontend/form->program
   (list 'let*
         ['contract-step
          '(raster.par/contract
            C [[i 4] [j 8]] [[l 16]]
            (* (clojure.core/aget A (+ (* i 16) l))
               (clojure.core/aget B (+ (* l 8) j))))
          'map-step
          (list 'raster.par/map! destination 't 32 nil map-expression)]
         'map-step)
   (merge {:dtype :float
    :array-types '{A :float B :float C :float D :float bias :float residual :float}
    :scalar-types '{scale :float}} options))))

(deftest result-map-stable-storage-requires-the-complete-producer-axis-map
  (doseq [[index fused?] [['t true] ['(mod t 8) false] ['(inc t) false]]]
    (let [read (with-meta (list 'clojure.core/aget 'C index)
                         {:raster.type/tag 'Float :raster.op/original 'clojure.core/aget})
          program (contract-result-map-program
                   (list 'max '(float 0.0) read) 'D
                   {:values {'C (av/tensor {:dtype :float :shape '[(extent C)]})}})
          [result stats] (typed-fusion/fusion-fixpoint program)]
      (is (= (if fused? 1 0) (:vertical stats)))
      (is (= (if fused? 1 2) (count (dialect/equations result))))
      (is (= result (dialect/validate! result)))))
  (let [read (with-meta '(clojure.core/aget bias t)
                       {:raster.type/tag 'Float :raster.op/original 'clojure.core/aget})
        rewritten (#'typed-fusion/erase-axis-mapped-load-indices read #{'bias})]
    (is (= '(clojure.core/aget bias 0) rewritten))
    (is (= (meta read) (meta rewritten)))))

(defn- map-initialized-contract-program
  ([initializer-expression]
   (map-initialized-contract-program initializer-expression
                                     '(* (clojure.core/aget A (+ (* i 16) l))
                                         (clojure.core/aget B (+ (* l 8) j)))
                                     32))
  ([initializer-expression fold-element initializer-extent]
   (frontend/form->program
    (list 'let*
          ['init (list 'raster.par/map! 'C 't initializer-extent nil
                       initializer-expression)
           'product
           (list 'raster.par/contract 'C '[[i 4] [j 8]] '[[l 16]]
                 fold-element
                 :epilogue
                 '{:acc acc
                   :expr (+ acc (clojure.core/aget C (+ (* i 8) j)))
                   :operands [{:sym C
                               :map {:groups [[[i 4]] [[j 8]]]}
                               :dtype :float}]
                   :scalars []
                   :dtype :float})]
          'product)
    {:dtype :float
     :array-types '{A :float B :float C :float bias :float}})))

(deftest result-transform-capacity-maps-own-their-axis-extents-and-storage-types
  (let [[program _] (typed-fusion/fusion-fixpoint
                     (map-initialized-contract-program
                      '(clojure.core/aget bias (clojure.core/rem t 8))))
        rewrite (fn [path value]
                  (let [equation (first (dialect/equations program))
                        operation (nth equation 3)
                        attributes (assoc-in (second operation) path value)]
                    (list 'soac-program (dialect/facts program)
                          [(apply list (assoc (vec equation) 3
                                             (apply list (assoc (vec operation) 1 attributes))))]
                          (dialect/outputs program))))]
    (doseq [[path value expected]
            [[[:result-transform :operands 0 :map :groups 0 0 1]
              1 :typed-soac-result-transform-axis-extent]
             [[:result-transform :operands 0 :dtype]
              :double :typed-soac-result-transform-operand-type]]]
      (is (= expected
             (try (dialect/validate! (rewrite path value)) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest dense-initializer-map-becomes-a-contraction-result-transform
  (let [program (map-initialized-contract-program
                 '(clojure.core/aget bias (clojure.core/rem t 8)))
        fold-before (-> program dialect/equations second typed-fusion/equation-info :body-results)
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        operation (dialect/operation-parts equation)
        transform (get-in operation [:attributes :result-transform])
        facts (dialect/facts result)]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 1 (count (dialect/equations result))))
    (is (= fold-before (:body-results (dialect/lambda-parts (:lambda operation))))
        "initializer fusion does not rewrite the contraction fold")
    (is (= '[bias] (mapv :value (:operands transform))))
    (is (= '#{j} (-> transform :operands first :map axis-map/axes set)))
    (is (empty? (:scalars transform)))
    (is (not-any? #{'C 't} (flatten (:lambda transform))))
    (is (= :write (get-in facts [:equations 1 :attributes :result-storage 0 :access])))
    (is (= result (dialect/validate! result)))))

(deftest initializer-fusion-refuses-unproved-addresses-and-extents
  (doseq [program [(map-initialized-contract-program
                    '(clojure.core/aget bias (clojure.core/rem (+ t 1) 8)))
                   (map-initialized-contract-program
                    '(clojure.core/aget bias (clojure.core/rem t 8))
                    '(* (clojure.core/aget A (+ (* i 16) l))
                        (clojure.core/aget B (+ (* l 8) j)))
                    31)]]
    (let [[result stats] (typed-fusion/fusion-fixpoint program)]
      (is (= program result))
      (is (zero? (:vertical stats))))))

(deftest initializer-fusion-refuses-a-destination-read-in-the-fold
  (let [program
        (map-initialized-contract-program
         '(clojure.core/aget bias (clojure.core/rem t 8))
         '(+ (* (clojure.core/aget A (+ (* i 16) l))
                (clojure.core/aget B (+ (* l 8) j)))
             (clojure.core/aget C (+ (* i 8) j)))
         32)
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (zero? (:vertical stats)))))

(deftest same-destination-result-map-fuses-without-reading-old-output
  (let [program (contract-result-map-program '(max (float 0.0) (clojure.core/aget C t)) 'C)
        [result stats] (typed-fusion/fusion-fixpoint program)
        facts (dialect/facts result)
        operation (dialect/operation-parts (first (dialect/equations result)))]
    (is (= 1 (:vertical stats)))
    (is (= 1 (count (dialect/equations result))))
    (is (= :write (get-in facts [:equations 0 :attributes :result-storage 0 :access])))
    (is (= 'C (get-in facts [:equations 0 :attributes :result-storage 0 :destination])))
    (is (empty? (get-in operation [:attributes :result-transform :operands])))
    (is (= result (dialect/validate! result)))))

(defn- dynamic-result-map-program [extent-expression placement]
  (let [producer ['contract-step
                  '(raster.par/contract C [[i m] [j n]] [[p k]]
                     (* (clojure.core/aget A (+ (* i k) p))
                        (clojure.core/aget B (+ (* p n) j))))]
        extent ['size extent-expression]
        consumer ['map-step '(raster.par/map! C t size nil
                               (max (float 0.0) (clojure.core/aget C t)))]]
    (frontend/form->program
     (list 'let* (vec (concat (if (= :before placement) (concat extent producer)
                                 (concat producer extent)) consumer)) 'map-step)
     {:dtype :float :array-types '{A :float B :float C :float}
      :scalar-types '{m :long n :long k :long size :long}})))

(defn- dynamic-aliased-result-map-program
  [read-value]
  (frontend/form->program
   (list 'let*
         ['size (with-meta '(clojure.core/* m n) {:raster.type/tag 'long})
          'product '(raster.par/contract C [[i m] [j n]] [[p k]]
                      (* (clojure.core/aget A (+ (* i k) p))
                         (clojure.core/aget B (+ (* p n) j))))
          'map-step (list 'raster.par/map! 'C 't 'size nil
                          (list 'max '(float 0.0)
                                (list 'clojure.core/aget read-value 't)))]
         'map-step)
   {:dtype :float
    ;; The production walker retains the local contraction binding's array element type. State
    ;; that same source fact explicitly in this raw frontend fixture; it is type evidence only,
    ;; not alias evidence.
    :array-types '{A :float B :float C :float product :float unattested :float}
    :scalar-types '{m :long n :long k :long size :long}}))

(deftest result-map-fusion-accepts-only-the-producers-attested-host-binding
  (let [program (dynamic-aliased-result-map-program 'product)
        [result stats] (typed-fusion/fusion-fixpoint program)
        equations (dialect/equations result)
        equation (second equations)
        operation (dialect/operation-parts equation)]
    (is (= 1 (:vertical stats)))
    (is (= 2 (count equations))
        "the checked size equation remains and the product map becomes the contraction epilogue")
    (is (= 'C (get-in (dialect/facts result)
                      [:equations (second equation) :attributes :result-storage 0 :destination])))
    (is (empty? (get-in operation [:attributes :result-transform :operands])))
    (is (= result (dialect/validate! result))))
  (let [program (dynamic-aliased-result-map-program 'unattested)
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (zero? (:vertical stats)))
    (is (= program result))
    (is (= 3 (count (dialect/equations result)))
        "an unrelated same-typed array is not inferred to alias the producer destination")))

(deftest dominating-typed-product-proves-dynamic-result-fusion
  (doseq [product ['(clojure.core/* m n) '(clojure.core/* n m)]]
    (let [expression (with-meta product {:raster.type/tag 'long})
          program (dynamic-result-map-program expression :before)
          [result stats] (typed-fusion/fusion-fixpoint program)
          equations (dialect/equations result)
          operation (dialect/operation-parts (second equations))]
      (is (= 1 (:vertical stats)))
      (is (= 2 (count equations)))
      (is (= (first (dialect/equations program)) (first equations))
          "the checked scalar definition remains before the output write")
      (is (some #{'size} (:captures operation))
          "the proof witness remains a dependency even though the epilogue need not read it")
      (is (= ['size] (mapv :value (get-in operation [:attributes :result-transform :scalars]))))
      (is (= result (dialect/validate! result))))))

(deftest dynamic-result-fusion-does-not-invent-extent-proofs
  (doseq [[expression placement]
          [[(with-meta '(clojure.core/* m n) {:raster.type/tag 'long}) :after]
           ['(clojure.core/* m n) :before]
           [(with-meta '(clojure.core/* m n) {:raster.type/tag 'int}) :before]
           [(with-meta '(unchecked-multiply m n) {:raster.type/tag 'long}) :before]
           [(with-meta '(clojure.core/* m k) {:raster.type/tag 'long}) :before]
           [(with-meta '(clojure.core/* (clojure.core/int m) n) {:raster.type/tag 'long}) :before]]]
    (let [program (dynamic-result-map-program expression placement)
          [result stats] (typed-fusion/fusion-fixpoint program)]
      (is (= program result) (pr-str [expression placement]))
      (is (zero? (:vertical stats))))))

(deftest same-destination-fusion-refuses-neighbor-reads
  (doseq [expression ['(clojure.core/aget C (mod (+ t 1) 32))
                      '(+ (clojure.core/aget C t) (clojure.core/aget C (mod (+ t 1) 32)))]]
    (let [program (contract-result-map-program expression 'C)
          [result stats] (typed-fusion/fusion-fixpoint program)]
      (is (= program result))
      (is (zero? (:vertical stats))))))

(deftest segmented-reduce-result-map-becomes-a-typed-result-transform
  (let [program (contract-result-map-program
                 '(* (+ (clojure.core/aget C t)
                        (clojure.core/aget bias (mod t 8))
                        (clojure.core/aget residual t))
                     scale))
        fold-before (-> program dialect/equations first typed-fusion/equation-info :body-results)
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        operation (dialect/operation-parts equation)
        transform (get-in operation [:attributes :result-transform])
        transform-region (dialect/lambda-parts (:lambda transform))
        facts (dialect/facts result)]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 1 (count (dialect/equations result))))
    (is (= '[map-step] (nth equation 2)))
    (is (= '[map-step] (dialect/outputs result)))
    (is (= '[A B bias residual scale] (:inputs facts)))
    (is (= 'D (get-in facts [:equations 0 :attributes :result-storage 0 :destination])))
    (is (= {'map-step 'D} (get-in facts [:equations 0 :aliases])))
    (is (not-any? #{'C [:effect-map 0 0]} (keys (:values facts))))
    (is (= '[4 8] (get-in facts [:values 'map-step :shape])))
    (is (= fold-before (:body-results (dialect/lambda-parts (:lambda operation))))
        "post-reduction fusion must never rewrite the reduction fold")
    (is (= '[residual bias] (mapv :value (:operands transform))))
    (is (= '#{i j}
           (-> transform :operands first :map axis-map/axes set)))
    (is (= '#{j}
           (-> transform :operands second :map axis-map/axes set)))
    (is (= '[scale] (mapv :value (:scalars transform))))
    (is (not-any? #{'C 't} (flatten (:body-results transform-region))))
    (is (= result (dialect/validate! result)))))

(deftest segmented-reduce-result-map-refuses-an-unproved-operand-index
  (let [program (contract-result-map-program
                 '(+ (clojure.core/aget C t)
                     (clojure.core/aget bias (+ t 1))))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations result))))))

(deftest segmented-reduce-result-map-retains-an-externally-live-intermediate
  (let [program (contract-result-map-program
                 '(+ (clojure.core/aget C t) 1.0))
        produced (-> program dialect/equations first (nth 2) first)
        program (dialect/make (dialect/facts program)
                              (dialect/equations program)
                              [produced 'map-step])
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (= {:vertical 0 :horizontal 0 :iterations 1} stats))
    (is (= 2 (count (dialect/equations result))))))

(defn- reduction-scalar-program []
  (frontend/form->program
         '(let* [total (raster.par/reduce acc 0.0 i n
                                          (+ acc (clojure.core/aget x i)))
                 ^double scaled (* ^double total ^double scale)]
                scaled)
         {:dtype :double
          :array-types {'x :double}
          :scalar-types {'scale :double}}))

(deftest completed-reduction-fuses-a-pure-scalar-consumer
  (let [program (reduction-scalar-program)
        fold-before (-> program dialect/equations first typed-fusion/equation-info :body-results)
        [result stats] (typed-fusion/fusion-fixpoint program)
        equation (first (dialect/equations result))
        operation (dialect/operation-parts equation)
        transform (get-in operation [:attributes :result-transform])
        transform-region (dialect/lambda-parts (:lambda transform))
        remapped (dialect/remap-values result {'scale 'gain})
        remapped-attributes (:attributes (dialect/operation-parts
                                          (first (dialect/equations remapped))))]
    (is (= {:vertical 1 :horizontal 0 :iterations 2} stats))
    (is (= 1 (count (dialect/equations result))))
    (is (= 'reduce (:kind operation)))
    (is (= '[scaled] (nth equation 2)))
    (is (= '[scaled] (dialect/outputs result)))
    (is (= '#{n scale x} (set (:inputs (dialect/facts result)))))
    (is (= fold-before (:body-results (dialect/lambda-parts (:lambda operation))))
        "the scalar epilogue must never enter the element or partial-reduction fold")
    (is (= [] (:operands transform)))
    (is (= '[scale] (mapv :value (:scalars transform))))
    (is (not-any? #{'total} (flatten (:body-results transform-region))))
    (is (= '[gain] (mapv :value (get-in remapped-attributes
                                        [:result-transform :scalars]))))
    (is (not (contains? remapped-attributes :segment-axes))
        "remapping a full reduction must not invent segmented axes")
    (is (= result (dialect/validate! result)))))

(defn- with-required-scalar [base equation-id]
  (let [source-id (get-in (dialect/facts base)
                          [:equations equation-id :provenance :source-binding-id])
        obligation {:source-binding-id source-id :flags #{:preserve-source-evaluation}
                    :ordering :source}
        facts (-> (dialect/facts base)
                  (assoc-in [:equations equation-id :attributes :required-evaluation] obligation)
                  (assoc-in [:attributes :required-evaluations equation-id] obligation))
        program (dialect/make facts (dialect/equations base) (dialect/outputs base))]
    program))

(deftest required-scalar-completion-is-not-a-reduction-epilogue
  (let [base (reduction-scalar-program)
        equation-id (second (second (dialect/equations base)))
        program (with-required-scalar base equation-id)
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (zero? (:vertical stats)))
    (is (= program result))
    (is (= 2 (count (dialect/equations result))))
    (is (= (dialect/required-evaluation program equation-id)
           (dialect/required-evaluation result equation-id)))))

(deftest maps-do-not-fuse-across-required-scalar-completion
  (let [base (frontend/form->program
              '(let* [y (raster.par/pmap i n double (* (clojure.core/aget x i) 2.0))
                      ^double marker (* ^double gain ^double gain)
                      z (raster.par/pmap j n double (+ (clojure.core/aget y j) marker))] z)
              {:dtype :double :array-types {'x :double}
               :scalar-types {'n :long 'gain :double}})
        program (with-required-scalar base 1)
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= program result))
    (is (zero? (:vertical stats)))
    (is (= 3 (count (dialect/equations result))))))

(deftest required-scalar-prefix-does-not-disable-later-map-fusion
  (let [base (frontend/form->program
              '(let* [^double marker (* ^double gain ^double gain)
                      y (raster.par/pmap i n double (* (clojure.core/aget x i) marker))
                      z (raster.par/pmap j n double (+ (clojure.core/aget y j) 1.0))] z)
              {:dtype :double :array-types {'x :double}
               :scalar-types {'n :long 'gain :double}})
        program (with-required-scalar base 0)
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 1 (:vertical stats)))
    (is (= 2 (count (dialect/equations result))))
    (is (= (first (dialect/equations program)) (first (dialect/equations result))))
    (is (= (dialect/required-evaluation program 0) (dialect/required-evaluation result 0)))))

(deftest fused-prefix-projection-rebases-all-source-coordinates
  (let [base (frontend/form->program
              '(let* [y (raster.par/pmap i n double (* (clojure.core/aget x i) 2.0))
                      z (raster.par/pmap j n double (+ (clojure.core/aget y j) 1.0))
                      ^double marker (* ^double gain ^double gain)
                      out (raster.par/pmap k n double (+ (clojure.core/aget z k) marker))] out)
              {:dtype :double :array-types {'x :double}
               :scalar-types {'n :long 'gain :double}})
        [program stats] (typed-fusion/fusion-fixpoint (with-required-scalar base 2))
        positions (into {} (map-indexed (fn [position equation] [(second equation) position]))
                        (dialect/equations program))
        projected (dialect/rebase-source-bindings program positions)]
    (is (= 1 (:vertical stats)))
    (is (= 3 (count (dialect/equations projected))))
    (is (= 1 (:source-binding-id (dialect/required-evaluation projected 2))))
    (is (every? zero? (map #(get-in % [:provenance :source-binding-id])
                          (vals (get-in (dialect/facts projected)
                                        [:equations 1 :attributes :fusion/constituents])))))
    (is (seq (:body (#'route/host-projection program [] (dialect/outputs program)))))
    (is (= projected (dialect/validate! projected)))))

(deftest completed-reduction-conversion-preserves-the-accumulator-precision
  (let [program (frontend/form->program
                 '(let* [total (raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
                         ^float result (float total)] result)
                 {:dtype :double :array-types {'x :double} :scalar-types {'n :long}})
        [result stats] (typed-fusion/fusion-fixpoint program)
        operation (dialect/operation-parts (first (dialect/equations result)))
        transform (get-in operation [:attributes :result-transform])]
    (is (= 1 (:vertical stats)))
    (is (= 1 (count (dialect/equations result))))
    (is (= [:double] (get-in operation [:attributes :dtypes])))
    (is (= :float (:result-dtype transform)))
    (is (= :float (get-in (dialect/facts result) [:values 'result :dtype])))
    (is (= result (dialect/validate! result)))))

(deftest adjacent-mixed-width-arithmetic-preserves-its-terminal-region
  (let [program (frontend/form->program
                 '(let* [total (raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
                         ^float result (float (+ total 0.25))] result)
                 {:dtype :double :array-types {'x :double} :scalar-types {'n :long}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 1 (:vertical stats)))
    (is (= 1 (count (dialect/equations result))))
    (let [operation (dialect/operation-parts (first (dialect/equations result)))]
      (is (= [:double] (get-in operation [:attributes :dtypes])))
      (is (= :float (get-in operation [:attributes :result-transform :result-dtype]))))
    (is (= result (dialect/validate! result)))))

(defn- separated-reduction-conversion [intervening terminal]
  (frontend/form->program
   (list 'let*
         ['total '(raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
          (with-meta 'unchanged {:tag 'double}) '(double total)
          'middle intervening
          (with-meta 'result {:tag 'float}) terminal]
         'result)
   {:dtype :double :array-types {'x :double 'out :double}
    :scalar-types {'n :long}}))

(deftest terminal-conversion-crosses-only-pure-ssa-equations
  (let [program (separated-reduction-conversion
                 '(raster.par/pmap j n double (* (aget x j) 2.0))
                 '(float unchanged))
        [result stats] (typed-fusion/fusion-fixpoint program)
        operation (dialect/operation-parts (first (dialect/equations result)))]
    (is (= 2 (:vertical stats)))
    (is (= ['reduce 'map] (mapv dialect/operation-kind (dialect/equations result))))
    (is (= [:double] (get-in operation [:attributes :dtypes])))
    (is (= :float (get-in operation [:attributes :result-transform :result-dtype])))
    (is (= result (dialect/validate! result)))))

(deftest terminal-conversion-does-not-cross-effects-or-generalize-arithmetic
  (doseq [[intervening terminal]
          [['(raster.par/map! out j n double (* (aget x j) 2.0)) '(float unchanged)]
           ['(raster.par/pmap j n double (* (aget x j) 2.0)) '(float (+ unchanged 0.25))]]]
    (let [program (separated-reduction-conversion intervening terminal)
          [result stats] (typed-fusion/fusion-fixpoint program)]
      (is (= 1 (:vertical stats)) "only the adjacent identity cast fuses")
      (is (= 3 (count (dialect/equations result)))))))

(deftest terminal-conversion-retains-existing-nonidentity-transforms
  (let [program (frontend/form->program
                 '(let* [total (raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
                         ^double scaled (* total 2.0)
                         middle (raster.par/pmap j n double (* (aget x j) 2.0))
                         ^float result (float scaled)] result)
                 {:dtype :double :array-types {'x :double} :scalar-types {'n :long}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 1 (:vertical stats)))
    (is (= 3 (count (dialect/equations result))))
    (is (= :double (get-in (dialect/operation-parts (first (dialect/equations result)))
                          [:attributes :result-transform :result-dtype])))))

(deftest terminal-conversion-retains-intermediate-narrowing
  (let [program (frontend/form->program
                 '(let* [total (raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
                         ^float narrowed (float total)
                         middle (raster.par/pmap j n double (* (aget x j) 2.0))
                         ^double result (double narrowed)] result)
                 {:dtype :double :array-types {'x :double} :scalar-types {'n :long}})
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 1 (:vertical stats)))
    (is (= 3 (count (dialect/equations result))))
    (is (= :float (get-in (dialect/operation-parts (first (dialect/equations result)))
                         [:attributes :result-transform :result-dtype])))))

(deftest terminal-conversion-retains-opaque-host-boundaries
  (let [program (separated-reduction-conversion
                 '(raster.par/pmap j n double (* (aget x j) 2.0)) '(float unchanged))
        facts (assoc-in (dialect/facts program) [:attributes :host-binding-ids] #{2})
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 1 (:vertical stats)))
    (is (= 3 (count (dialect/equations result))))))

(deftest terminal-conversion-retains-an-intervening-alias-contract
  (let [program (separated-reduction-conversion
                 '(raster.par/pmap j n double (aget x j)) '(float unchanged))
        middle-id (second (nth (dialect/equations program) 2))
        facts (assoc-in (dialect/facts program) [:equations middle-id :aliases] {'middle 'x})
        program (dialect/make facts (dialect/equations program) (dialect/outputs program))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (empty? (get-in facts [:equations middle-id :effects])))
    (is (= 1 (:vertical stats)))
    (is (= 3 (count (dialect/equations result))))))

(deftest terminal-conversion-retains-a-shared-reduction-result
  (let [program (separated-reduction-conversion
                 '(raster.par/pmap j n double (+ (aget x j) unchanged)) '(float unchanged))
        [result stats] (typed-fusion/fusion-fixpoint program)]
    (is (= 1 (:vertical stats)))
    (is (= 3 (count (dialect/equations result))))))

(deftest a-map-feeding-a-fold-map-keeps-the-fold-map-intact
  ;; The fold-map reads the map's result as a stable capture, so it is not a vertical fusion
  ;; consumer; reading its references must not re-emit it through the single-lambda builder,
  ;; which used to drop its folds and leave a malformed operation for the next iteration.
  (let [source '(let* [y (raster.par/pmap i n float (* (aget x i) 2.0))
                       eff (raster.par/segmented-fold-map!
                            [out] [[b m]] j n
                            [[mx -1.0E38 :element n
                              (max mx (aget y (+ (* b n) j)))]]
                            [(- (aget y (+ (* b n) j)) mx)])]
                      eff)
        program (source-program source {'x :float 'out :float 'y :float}
                                {'n :long 'm :long})
        [fused stats] (typed-fusion/fusion-fixpoint program)
        fold-map (first (filter #(= 'segmented-fold-map (dialect/operation-kind %))
                                (dialect/equations fused)))]
    (is (= ['map 'segmented-fold-map]
           (mapv dialect/operation-kind (dialect/equations program))))
    (is (zero? (:vertical stats)))
    (is (some? fold-map))
    (is (= 1 (count (:folds (dialect/operation-parts fold-map)))))
    (is (= fused (dialect/validate! fused)))))
