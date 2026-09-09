(ns raster.compiler.passes.parallel.typed-carried-effect-frontend-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
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
