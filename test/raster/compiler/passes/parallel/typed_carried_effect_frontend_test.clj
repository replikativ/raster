(ns raster.compiler.passes.parallel.typed-carried-effect-frontend-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.segop-opencl :as gpu]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def source
  '(let* [effect
          (raster.par/map-void!
           row rows
           (do (aset scales row (float seed))
               (let* [^double sum
                      (loop* [^long k 0 ^double acc 0.25]
                        (if (< k width)
                          (let* [^float v (aget x (+ (* row width) k))]
                            (aset packed (+ (* row width) k) v)
                            (recur (inc k) (+ acc (double v))))
                          acc))]
                 (aset sums row (float sum)))))]
     effect))

(defn attempt [form]
  (route/attempt form :float {'x :float 'packed :float 'scales :float 'sums :float}
                 {:scalar-types {'rows :long 'width :long 'seed :float}}))

(deftest typed-result-valued-store-loop-is-an-ordered-effect-not-a-pure-local
  (let [result (attempt source)
        program (:program result)
        algorithm (-> program :equations first :algorithm)
        equation (first (dialect/equations algorithm))
        parts (-> equation dialect/operation-parts :lambda dialect/lambda-parts)
        effects (mapv dialect/effect-parts (:body-results parts))
        scheduled (:form (segop-lower/segop-lower-pass program {:target-device :ze:0 :dtype :float}))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x packed scales sums rows width seed] (:form jvm)))
        packed (float-array (repeat 6 -77)) scales (float-array 2) sums (float-array 2)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= 'effect-map (dialect/operation-kind equation)))
    (is (empty? (:locals parts)))
    (is (= [false true false] (mapv #(boolean (:loop %)) effects)))
    (is (some? (:carry (second effects))))
    (is (= :independent (-> equation dialect/operation-parts :attributes :iteration-order)))
    (is (nil? (execute (float-array [1 2 3 4 5 6]) packed scales sums 2 3 (float 2.5))))
    (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (vec packed)))
    (is (= [2.5 2.5] (vec scales)))
    (is (= [6.25 15.25] (vec sums)))))

(defn- replace-form [form before after]
  (walk/postwalk #(if (= before %) after %) form))

(deftest a-carried-result-scopes-over-subsequent-store-loops
  (let [form (replace-form
              (replace-form source '(+ acc (double v))
                            (with-meta '(+ acc (double v)) {:raster.type/tag 'double}))
              '(aset sums row (float sum))
              '(do (loop* [j 0]
                     (if (< j width)
                       (do (aset packed (+ (* row width) j)
                                 (float ^{:raster.type/tag double}
                                        (+ (double (aget packed (+ (* row width) j))) sum)))
                           (recur (inc j)))
                       nil))
                   (aset sums row (float sum))))
        result (attempt form)
        program (:program result)
        equation (-> program :equations first :algorithm dialect/equations first)
        effects (-> equation dialect/operation-parts :lambda dialect/lambda-parts
                    :body-results)
        scheduled (:form (segop-lower/segop-lower-pass program {:target-device :ze:0 :dtype :float}))
        execute (eval (list 'fn '[x packed scales sums rows width seed]
                            (:form (par-simd/simd-pass scheduled :min-elements 1))))]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= [false true true false]
           (mapv #(boolean (:loop (dialect/effect-parts %))) effects)))
    (doseq [target [:opencl-portable :cuda :hip]]
      (let [operation (first (soac-lower/lower-typed-effect-map
                              (-> program :equations first :algorithm) :ze:0))
            emitted (gpu/generate-scheduled-segmap-kernel
                      operation :dtype :float :target-dialect target
                      :array-types {'x :float 'packed :float 'scales :float 'sums :float}
                      :scalar-types {'rows :long 'width :long 'seed :float})]
        (is (= :kernel-body (get-in emitted [:attributes :emission-route])))))
    (doseq [width [0 1 3 8]]
      (let [n (* 2 width)
            x (float-array (range n))
            packed (float-array (repeat (+ n 2) -77))
            scales (float-array 2)
            sums (float-array 2)
            expected (mapv (fn [row]
                             (+ 0.25 (reduce + (range (* row width) (* (inc row) width)))))
                           (range 2))]
        (execute x packed scales sums 2 width (float 2.5))
        (is (= expected (vec sums)))
        (is (= (into (vec (mapcat (fn [row sum]
                                   (map #(+ % sum) (range (* row width) (* (inc row) width))))
                                 (range 2) expected))
                     [-77.0 -77.0])
               (vec packed)))
        (is (= [2.5 2.5] (vec scales)))))))

(deftest unsupported-recurrences-decline-before-typed-admission
  (doseq [variant [(replace-form source '(inc k) '(int (inc k)))
                   (replace-form source '(< k width) '(< (int k) width))
                   (replace-form source '(< k width) '(< k (+ width (long acc))))
                   (replace-form source '(< k width) '(< k (aget scales row)))
                   (replace-form source '(< k width) '(< k (do (aset scales row 0.0) width)))
                   (replace-form source 0.25 '(double k))
                   (replace-form source 0.25 '(do (aset sums row 1.0) 0.25))
                   (replace-form source '(+ acc (double v))
                                 '(do (aset scales row 1.0) (+ acc (double v))))
                   (walk/postwalk #(if (and (symbol? %) (= 'acc %)) (with-meta % nil) %) source)
                   (walk/postwalk #(if (and (symbol? %) (= 'sum %)) (with-meta % nil) %) source)]]
    (is (not= :typed-soac (get-in (attempt variant) [:stats :route])))))

(deftest continuation-loop-ordinals-preserve-interleaved-direct-stores
  (let [form (replace-form
              source '(aset sums row (float sum))
              '(do (loop* [j 0]
                     (if (< j width)
                       (do (aset packed (+ (* row width) j) (float sum))
                           (recur (inc j))) nil))
                   (aset sums row (aget packed (* row width)))
                   (loop* [j 0]
                     (if (< j width)
                       (do (aset packed (+ (* row width) j) (float seed))
                           (recur (inc j))) nil))))
        result (attempt form)
        program (:program result)
        equation (-> program :equations first :algorithm dialect/equations first)
        effects (-> equation dialect/operation-parts :lambda dialect/lambda-parts :body-results)
        scheduled (:form (segop-lower/segop-lower-pass program {:target-device :ze:0 :dtype :float}))
        execute (eval (list 'fn '[x packed scales sums rows width seed]
                            (:form (par-simd/simd-pass scheduled :min-elements 1))))
        packed (float-array 6) scales (float-array 2) sums (float-array 2)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= [false true true false true]
           (mapv #(boolean (:loop (dialect/effect-parts %))) effects)))
    (execute (float-array [1 2 3 4 5 6]) packed scales sums 2 3 (float 2.5))
    (is (= [6.25 15.25] (vec sums)))
    (is (= (vec (repeat 6 2.5)) (vec packed)))))

(def normalization-source
  (replace-form
   (replace-form source '(+ acc (double v))
                 (with-meta '(+ acc (double v)) {:raster.type/tag 'double}))
   '(aset sums row (float sum))
   '(let* [^double inverse ^{:raster.type/tag double} (/ 1.0 sum)]
      (do (loop* [j 0]
            (if (< j width)
              (do (aset packed (+ (* row width) j)
                        (float ^{:raster.type/tag double}
                               (* (double (aget packed (+ (* row width) j))) inverse)))
                  (recur (inc j))) nil))
          (aset sums row (float inverse))))))

(deftest post-carry-local-bindings-retain-their-execution-point
  (let [result (attempt normalization-source)
        program (:program result)
        algorithm (-> program :equations first :algorithm)
        equation (first (dialect/equations algorithm))
        parts (-> equation dialect/operation-parts :lambda dialect/lambda-parts)
        effects (mapv dialect/effect-parts (:body-results parts))
        scheduled (:form (segop-lower/segop-lower-pass program {:target-device :ze:0 :dtype :float}))
        execute (eval (list 'fn '[x packed scales sums rows width seed]
                            (:form (par-simd/simd-pass scheduled :min-elements 1))))]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (empty? (:locals parts)))
    (is (:region (last effects)))
    (is (= :sequential (-> equation dialect/operation-parts :attributes :iteration-order)))
    (is (= :read-write (:access (first (filter #(= 'packed (:destination %))
                                              (dialect/result-storage (dialect/facts algorithm)
                                                                      (second equation)))))))
    (doseq [width [0 1 3 8]]
      (let [n (* 2 width)
            packed (float-array (repeat (+ n 2) -77))
            scales (float-array 2) sums (float-array 2)
            inverses (mapv (fn [row] (/ 1.0 (+ 0.25 (reduce + (range (* row width)
                                                                      (* (inc row) width))))))
                           (range 2))]
        (execute (float-array (range n)) packed scales sums 2 width (float 2.5))
        (is (= (mapv float inverses) (vec sums)))
        (is (= (into (vec (mapcat (fn [row inverse]
                                    (map #(float (* % inverse))
                                         (range (* row width) (* (inc row) width))))
                                  (range 2) inverses)) [-77.0 -77.0]) (vec packed)))
        (is (= [2.5 2.5] (vec scales)))))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (= :kernel-body
             (get-in (gpu/generate-scheduled-segmap-kernel
                      (first (soac-lower/lower-typed-effect-map algorithm :ze:0))
                      :dtype :float :target-dialect target
                      :array-types {'x :float 'packed :float 'scales :float 'sums :float}
                      :scalar-types {'rows :long 'width :long 'seed :float})
                     [:attributes :emission-route]))))))

(deftest post-carry-local-reads-see-the-preceding-loop-writes
  (let [form (replace-form normalization-source '(/ 1.0 sum)
                           '(double (aget packed (* row width))))
        result (attempt form)
        scheduled (:form (segop-lower/segop-lower-pass (:program result)
                                                      {:target-device :ze:0 :dtype :float}))
        execute (eval (list 'fn '[x packed scales sums rows width seed]
                            (:form (par-simd/simd-pass scheduled :min-elements 1))))
        packed (float-array (repeat 6 -77)) scales (float-array 2) sums (float-array 2)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (execute (float-array [1 2 3 4 5 6]) packed scales sums 2 3 (float 2.5))
    (is (= [1.0 4.0] (vec sums)))
    (is (= [1.0 2.0 3.0 16.0 20.0 24.0] (vec packed)))))

(deftest post-carry-locals-still-require-retained-types
  (let [untyped (walk/postwalk #(if (and (symbol? %) (= 'inverse %))
                                 (with-meta % nil) %) normalization-source)
        untyped (replace-form untyped '(/ 1.0 sum) (with-meta '(/ 1.0 sum) nil))]
    (is (not= :typed-soac (get-in (attempt untyped) [:stats :route])))))

(deftest shadowed-comparison-casts-are-not-erased
  (doseq [test-form ['(< (long k) width) '(< k (long width))]]
    (is (not= :typed-soac
              (get-in (route/attempt (replace-form source '(< k width) test-form)
                                    :float {'x :float 'packed :float 'scales :float 'sums :float}
                                    {:scalar-types {'rows :long 'width :long 'seed :float
                                                    'long :object}})
                      [:stats :route])))))

(deftest shadowed-recurrence-operators-are-not-fixed-unit-steps
  (doseq [[step shadow] [['(inc k) 'inc] ['(+ k 1) '+] ['(inc (long k)) 'long]]]
    (is (not= :typed-soac
              (get-in (route/attempt (replace-form source '(inc k) step)
                                    :float {'x :float 'packed :float 'scales :float 'sums :float}
                                    {:scalar-types (assoc {'rows :long 'width :long 'seed :float}
                                                         shadow :object)})
                      [:stats :route])))))

(deftest carry-only-destination-reads-constrain-storage-and-iteration
  (doseq [variant [(replace-form source 0.25 '(double (aget sums row)))
                   (replace-form source '(+ acc (double v)) '(+ acc (double (aget scales row))))]]
    (let [result (attempt variant)
          algorithm (-> result :program :equations first :algorithm)
          equation (first (dialect/equations algorithm))
          facts (dialect/facts algorithm)
          storage (dialect/result-storage facts (second equation))
          read-write (set (map :destination (filter #(= :read-write (:access %)) storage)))]
      (is (= :typed-soac (get-in result [:stats :route])))
      (is (contains? (get-in facts [:equations (second equation) :effects]) :memory/read))
      (is (= :sequential (-> equation dialect/operation-parts :attributes :iteration-order)))
      (is (seq read-write)))))

(deftest floating-or-unknown-symbolic-bounds-are-not-integral-iteration-contracts
  (doseq [scalar-types [{'rows :long 'width :float 'seed :float}
                       {'rows :long 'seed :float}]]
    (is (not= :typed-soac
              (get-in (route/attempt source :float
                                    {'x :float 'packed :float 'scales :float 'sums :float}
                                    {:scalar-types scalar-types})
                      [:stats :route])))))

(deftest carry-dependent-indices-cannot-be-invariant-injectivity-factors
  (let [direct (replace-form source '(aset packed (+ (* row width) k) v)
                             '(aset packed (+ (* row width) (long acc)) v))
        indirect (walk/postwalk
                  (fn [form]
                    (if (and (seq? form) (= 'let* (first form)) (= 'v (first (second form))))
                      (let [[head bindings & body] form]
                        (list* head (into [(with-meta 'slot {:tag 'long}) '(long acc)] bindings)
                               (map #(replace-form % '(aset packed (+ (* row width) k) v)
                                                    '(aset packed (+ (* row width) slot) v)) body)))
                      form))
                  source)]
    (doseq [variant [direct indirect]]
      (let [result (attempt variant)
            equation (-> result :program :equations first :algorithm dialect/equations first)]
        (is (= :typed-soac (get-in result [:stats :route])))
        (is (= :sequential (-> equation dialect/operation-parts :attributes :iteration-order)))))))
