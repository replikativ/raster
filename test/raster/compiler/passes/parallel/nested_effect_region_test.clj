(ns raster.compiler.passes.parallel.nested-effect-region-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.jvm.segop-simd :as jvm]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-route]
            [raster.compiler.passes.parallel.carried-effect-loop-fixture
             :refer [scheduled-normalization artifact typed-program]
             :rename {scheduled-normalization normalization}]))

(deftest nested-locals-execute-after-the-carry-before-the-next-loop
  (doseq [trips [0 1 3]]
    (let [operation (normalization trips)
          execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
          x (float-array (range 16)) words (float-array (repeat 16 -77)) totals (float-array 2)
          inverses (mapv (fn [row] (float (/ 1.0 (+ 0.25 (reduce + (take trips (drop (* row 8) x)))))))
                         (range 2))]
      (execute x words totals 2)
      (is (= inverses (vec totals)))
      (is (= (mapv (fn [i]
                    (if (< (mod i 8) trips)
                      (float (* (aget x i) (nth inverses (quot i 8))))
                      (float -77)))
                  (range 16))
             (vec words)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [compiled (artifact operation target)
              operations (get-in compiled [:attributes :kernel-body :operations])
              divisions (keep-indexed #(when (= :div (get-in %2 [:expression :op])) %1) operations)
              loops (keep-indexed #(when (:iter-args %2) %1) operations)]
          (is (= :kernel-body (get-in compiled [:attributes :emission-route])))
          (is (= 1 (count divisions)))
          (is (= 2 (count loops)))
          (is (< (first loops) (first divisions) (second loops))))))))

(deftest canonical-nested-continuations-share-host-and-scheduled-realization
  (let [program (walk/postwalk
                 #(if (= % '(effect sums :unique i true sum))
                    '(effect-region [(let-value inverse :double
                                               ^{:raster.type/tag double} (/ 1.0 sum))]
                                    [(effect sums :unique i true inverse)]) %)
                 (typed-program 3))
        program (dialect/validate! program)
        equation (first (dialect/equations program))
        realized ((ns-resolve 'raster.compiler.passes.parallel.typed-soac-route 'realize-equation)
                  program equation)
        scheduled (first (lower/lower-typed-effect-map program :ze:0))]
    (doseq [form [(:source realized) (jvm/compile-effect-segmap scheduled)]]
      (let [execute (eval (list 'fn '[x words totals seed rows] form))
            words (float-array (repeat 16 -77)) totals (float-array 2)]
        (execute (float-array (range 16)) words totals (float 0.25) 2)
        (is (= [(float (/ 1.0 3.25)) (float (/ 1.0 27.25))] (vec totals)))
        (is (= [0.0 1.0 2.0 -77.0] (subvec (vec words) 0 4)))))))

(deftest nested-local-reads-observe-preceding-stores
  (let [operation (assoc-in (normalization 1) [:scalar-region :effects 1 :region :locals 0 :init]
                           '(double (aget words (* i 8))))
        execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
        words (float-array (repeat 16 -77)) totals (float-array 2)]
    (execute (float-array (range 16)) words totals 2)
    (is (= [0.0 8.0] (vec totals)))))

(deftest nested-initializer-failure-does-not-move-before-preceding-stores
  (let [operation (-> (normalization 1)
                      (update :scalars conj 'limit)
                      (assoc-in [:scalar-region :effects 1 :region :locals]
                                [{:id 'inverse :dtype :long
                                  :init (with-meta '(clojure.core/+ limit 1) {:raster.type/tag 'long})}]))
        execute (eval (list 'fn '[x words totals rows limit] (jvm/compile-effect-segmap operation)))
        words (float-array [-77]) totals (float-array [-77])]
    (is (thrown? ArithmeticException
                 (execute (float-array [9]) words totals 1 Long/MAX_VALUE)))
    (is (= [9.0] (vec words)))
    (is (= [-77.0] (vec totals)))))

(deftest nested-locals-do-not-escape-their-continuation
  (let [operation (update-in (normalization 1) [:scalar-region :effects]
                              conj {:destination 'totals :dtype :float :conflict :unique
                                    :destination-index 'i :predicate true :value 'inverse})]
    (doseq [lower [jvm/compile-effect-segmap #(artifact % :opencl-portable)]]
      (is (= :scheduled-effect-carry
             (try (lower operation) :accepted
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest uncarried-nested-regions-retain-checked-source-arithmetic
  (let [operation (-> (normalization 0)
                      (update :scalars conj 'limit)
                      (update-in [:scalar-region :effects] #(vec (rest %)))
                      (assoc-in [:scalar-region :effects 0 :region :locals 0 :init]
                                (with-meta '(clojure.core/+ limit 1) {:raster.type/tag 'long})))
        execute (eval (list 'fn '[x words totals rows limit] (jvm/compile-effect-segmap operation)))
        totals (float-array [-77])]
    (is (not (dialect/scheduled-effect-carries? (get-in operation [:scalar-region :effects]))))
    (is (thrown? ArithmeticException
                 (execute (float-array 1) (float-array 1) totals 1 Long/MAX_VALUE)))
    (is (= [-77.0] (vec totals)))
    (is (= :kernel-body-c-trap-unsupported
           (try (artifact operation :opencl-portable :scalar-types {'rows :long 'limit :long})
                :accepted
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (doseq [target [:opencl-intel :cuda :hip]]
      (let [compiled (artifact operation target :scalar-types {'rows :long 'limit :long})
            checked (filter #(= :+ (get-in % [:expression :op]))
                            (get-in compiled [:attributes :kernel-body :operations]))]
        (is (= 1 (count checked)))
        (is (= :long (get-in (first checked) [:result :type])))
        (is (= :trap (get-in (first checked) [:expression :options :overflow])))))))

(deftest uncarried-nested-symbol-initializers-use-explicit-ieee-storage-conversion
  (let [operation (-> (normalization 0)
                      (assoc-in [:scalar-region :locals]
                                [{:id 'seed :dtype :double :init Double/MAX_VALUE}])
                      (update-in [:scalar-region :effects] #(vec (rest %)))
                      (assoc-in [:scalar-region :effects 0 :region :locals 0 :init] 'seed))
        execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
        totals (float-array 1)]
    (execute (float-array 1) (float-array 1) totals 1)
    (is (= Float/POSITIVE_INFINITY (aget totals 0)))
    (doseq [target [:opencl-portable :cuda :hip]]
      (let [operations (get-in (artifact operation target) [:attributes :kernel-body :operations])]
        (is (some #(and (= :cast (get-in % [:expression :op]))
                        (= :float (get-in % [:result :type]))
                        (= {:rounding :nearest-even :overflow :ieee}
                           (get-in % [:expression :options]))) operations))))))
