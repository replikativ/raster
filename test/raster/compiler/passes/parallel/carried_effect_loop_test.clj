(ns raster.compiler.passes.parallel.carried-effect-loop-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.segop-simd :as jvm]
            [raster.compiler.passes.parallel.carried-effect-loop-fixture :refer [scheduled-loop artifact]]))

(defn- operations [ops]
  (mapcat (fn [op] (cons op (operations (concat (:operations op)
                                               (:then-operations op) (:else-operations op))))) ops))

(deftest carried-effects-preserve-zero-trip-results-and-loop-local-values
  (doseq [trips [0 1 8]]
    (let [operation (scheduled-loop trips)
          execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
          x (float-array (range 16)) words (float-array (repeat 16 -77.0)) totals (float-array 2)]
      (execute x words totals 2)
      (doseq [row (range 2)]
        (is (= (+ 0.25 (reduce + (take trips (drop (* row 8) x)))) (double (aget totals row)))))
      (is (= (vec (map-indexed (fn [i x] (if (< (mod i 8) trips) x (float -77))) x))
             (vec words)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [compiled (artifact operation target)
              all (operations (get-in compiled [:attributes :kernel-body :operations]))
              loop (first (filter #(seq (:iter-args %)) all))
              inner (:operations loop)]
          (is (= :kernel-body (get-in compiled [:attributes :emission-route])))
          (is (= ['sum] (mapv :id (:results loop))))
          (is (= [:float] (mapv #(get-in % [:binding :type]) (:iter-args loop))))
          (is (< (first (keep-indexed #(when (= 'words (:buffer %2)) %1) inner))
                 (first (keep-indexed #(when (= :+ (get-in %2 [:expression :op])) %1) inner)))
              "store occurs before recurrence computation"))))))

(deftest checked-recurrence-fails-after-its-store-not-before
  (let [operation (-> (scheduled-loop 1)
                      (assoc-in [:scalar-region :effects 0 :loop :carry]
                                {:parameter 'acc :result 'sum :dtype :long
                                 :init Long/MAX_VALUE :update (with-meta '(clojure.core/+ acc 1)
                                                               {:raster.type/tag 'long})})
                      (assoc-in [:scalar-region :effects 1 :value] '(float sum)))
        execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
        x (float-array [9]) words (float-array [-77]) totals (float-array [-77])]
    (is (thrown? ArithmeticException (execute x words totals 1)))
    (is (= [9.0] (vec words)))
    (is (= [-77.0] (vec totals)))
    (let [all (operations (get-in (artifact operation :opencl-intel)
                                  [:attributes :kernel-body :operations]))
          recurrence (first (filter #(= :+ (get-in % [:expression :op])) all))]
      (is (= :trap (get-in recurrence [:expression :options :overflow]))))))

(deftest carried-effects-reject-bad-scope-before-target-lowering
  (let [base (scheduled-loop 8)
        carry-path [:scalar-region :effects 0 :loop :carry]
        variants [(assoc-in base (conj carry-path :dtype) :bogus)
                  (assoc-in base (conj carry-path :parameter) 'seed)
                  (assoc-in base (conj carry-path :result) 'k)
                  (assoc-in base (conj carry-path :init) 'sum)
                  (assoc-in base (conj carry-path :update) 'unknown)
                  (assoc-in base [:scalar-region :effects 0 :loop :extent] 'acc)
                  (assoc-in base [:scalar-region :effects 1 :value] 'loaded)
                  (update-in base [:scalar-region :effects] #(vec (reverse %)))]]
    (doseq [operation variants lower [jvm/compile-effect-segmap #(artifact % :opencl-portable)]]
      (is (= :scheduled-effect-carry
             (try (lower operation) :accepted
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest carried-loop-boundaries-use-typed-conversions-and-lexical-substitution
  (doseq [parameter ['acc 'float 'long]]
    (let [operation (-> (scheduled-loop 2)
                        (assoc-in [:scalar-region :locals]
                                  [{:id 'seed :dtype :long :init 3}
                                   {:id 'start :dtype :long :init 1}])
                        (assoc-in [:scalar-region :effects 0 :loop :lower] 'start)
                        (assoc-in [:scalar-region :effects 0 :loop :carry :parameter] parameter)
                        (assoc-in [:scalar-region :effects 0 :loop :carry :update] 'k))
          execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
          words (float-array (repeat 16 -77)) totals (float-array 2)]
      (execute (float-array (range 16)) words totals 2)
      (is (= [1.0 1.0] (vec totals)))
      (is (= [-77.0 1.0 -77.0 -77.0 -77.0 -77.0 -77.0 -77.0
              -77.0 9.0 -77.0 -77.0 -77.0 -77.0 -77.0 -77.0] (vec words)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [all (operations (get-in (artifact operation target) [:attributes :kernel-body :operations]))
              loop (first (filter #(seq (:iter-args %)) all))
              types (into {} (keep #(when (:result %) [(get-in % [:result :id])
                                                       (get-in % [:result :type])]) all))]
          (is (= :float (get types (get-in loop [:iter-args 0 :initial]))))
          (is (= :float (get types (get-in (last (:operations loop)) [:values 0])))))))))

(deftest carry-conversion-preserves-checked-source-operation-width
  (doseq [field [:init :update]]
    (let [expression (with-meta '(clojure.core/+ n 1) {:raster.type/tag 'long})
          operation (-> (scheduled-loop 1)
                        (update-in [:scalar-region :locals] conj {:id 'n :dtype :long :init Long/MAX_VALUE})
                        (assoc-in [:scalar-region :effects 0 :loop :carry field] expression))
          execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
          words (float-array [-77]) totals (float-array [-77])]
      (is (thrown? ArithmeticException (execute (float-array [9]) words totals 1)))
      (is (= [(if (= field :init) -77.0 9.0)] (vec words)))
      (is (= [-77.0] (vec totals)))
      (doseq [target [:opencl-intel :cuda :hip]]
        (let [all (operations (get-in (artifact operation target) [:attributes :kernel-body :operations]))
              checked (filter #(and (= :+ (get-in % [:expression :op]))
                                     (= :long (get-in % [:result :type]))) all)]
          (is (= 1 (count checked)))
          (is (= :trap (get-in (first checked) [:expression :options :overflow])))))
      (let [untyped (assoc-in operation [:scalar-region :effects 0 :loop :carry field]
                              (with-meta expression nil))]
        (is (= :scalar-source-type
               (try (artifact untyped :opencl-portable) :accepted
                    (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e))))))))))

(deftest carry-conversions-do-not-inherit-device-integer-wrapping
  (let [operation (-> (scheduled-loop 0)
                      (assoc-in [:scalar-region :locals] [{:id 'seed :dtype :long :init 2147483648}])
                      (assoc-in [:scalar-region :effects 0 :loop :carry :dtype] :int)
                      (assoc-in [:scalar-region :effects 1 :value] '(float sum)))
        execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))]
    (is (thrown? ArithmeticException (execute (float-array 1) (float-array 1) (float-array 1) 1)))
    (is (= :cast-policy
           (try (artifact operation :opencl-portable) :accepted
                (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e))))))))

(deftest typed-float-carries-use-ieee-storage-conversion
  (doseq [trips [0 1]]
    (let [operation (-> (scheduled-loop trips)
                        (assoc-in [:scalar-region :locals] [{:id 'seed :dtype :double :init Double/MAX_VALUE}])
                        (assoc-in [:scalar-region :effects 0 :loop :carry :update] 'seed))
          execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
          totals (float-array 1)]
      (execute (float-array [9]) (float-array 1) totals 1)
      (is (= Float/POSITIVE_INFINITY (aget totals 0)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [all (operations (get-in (artifact operation target) [:attributes :kernel-body :operations]))]
          (is (some #(and (= :cast (get-in % [:expression :op]))
                          (= {:rounding :nearest-even :overflow :ieee}
                             (get-in % [:expression :options]))) all)))))))

(deftest sequential-carried-loops-export-only-their-results
  (let [base (scheduled-loop 1)
        loop (get-in base [:scalar-region :effects 0])
        second-loop (-> loop
                        (assoc-in [:loop :carry] {:parameter 'acc2 :result 'sum2 :dtype :float
                                                :init 'sum :update (with-meta '(+ acc2 loaded)
                                                                     {:raster.type/tag 'double})}))
        operation (assoc-in base [:scalar-region :effects]
                            [loop second-loop
                             (assoc (get-in base [:scalar-region :effects 1]) :value 'sum2)])
        execute (eval (list 'fn '[x words totals rows] (jvm/compile-effect-segmap operation)))
        words (float-array 16) totals (float-array 2)]
    (execute (float-array (range 16)) words totals 2)
    (is (= [0.25 16.25] (vec totals)))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (= :kernel-body (get-in (artifact operation target) [:attributes :emission-route])))))
  (let [base (scheduled-loop 1)
        nested (assoc-in base [:scalar-region :effects 0 :loop :effects]
                         [(get-in base [:scalar-region :effects 0])])]
    (doseq [lower [jvm/compile-effect-segmap #(artifact % :opencl-portable)]]
      (is (= :scheduled-effect-carry
             (try (lower nested) :accepted
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))
