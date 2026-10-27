(ns raster.compiler.ir.extent-proof-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.extent-proof :as proof]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]))

(def environment
  {'m {:dtype :long :product {:const 1 :factors ['m]}}
   'n {:dtype :long :product {:const 1 :factors ['n]}}
   'i {:dtype :int :product {:const 1 :factors ['i]}}})

(deftest parallel-storage-equality-requires-the-exact-predecessor-context
  (let [values (into {} (map #(vector % (av/tensor {:dtype :long :shape []}))) '[m n total])
        scalar '(= product [total]
                   (scalar {:dtypes [:long]} [m n]
                           (lambda [a b] (region [] [(clojure.core/* a b)]))))
        algorithm (dialect/make
                   (dialect/default-program-facts
                    {:values values :inputs '[m n]
                     :equations {'product (dialect/default-equation-facts)}})
                   [scalar] '[total])
        predecessor {:id :product :operands '[m n] :results '[total]
                     :algorithm algorithm :effects #{} :operations [] :attributes {:host-only true}}
        consumer {:id :consumer :operands '[total m n] :results [] :algorithm nil :effects #{}}
        program {:values values :inputs '[m n] :equations [predecessor consumer]}
        equal? (fn [context]
                 (proof/same-volume? (proof/parallel-environment-before context consumer)
                                     'total '[(clojure.core/* m n)]))]
    (is (equal? program))
    (is (not (equal? (assoc program :equations [consumer predecessor]))))
    (is (not (equal? (assoc program :equations [predecessor consumer consumer]))))
    (is (not (equal? (assoc-in program [:values 'n :dtype] :int))))
    (is (not (equal? (update program :values dissoc 'n))))
    (is (not (equal? (assoc-in program [:equations 0 :effects] #{:memory/write}))))
    (is (not (equal? (assoc-in program [:equations 0 :operations] [:forged]))))
    (is (not (equal? (assoc-in program [:values 'n :representation]
                              {:kind :resident-scalar-buffer :elements 1}))))
    (is (not (proof/same-volume? (proof/parallel-environment-before program consumer)
                                 'total '[(clojure.core/* m n 2)])))
    (is (= {} (proof/parallel-environment-before program (assoc consumer :operands '[m]))))
    (is (= {} (proof/parallel-environment-before program (assoc consumer :algorithm (Object.))))
        "unsupported descriptive objects cannot supply an equality witness")
    (is (= {} (proof/parallel-environment-before program (assoc consumer :id :absent))))))

(defn- outer-equation
  [subprogram]
  (let [equation (first (dialect/equations subprogram))]
    {:id (second equation)
     :operands (:inputs (dialect/facts subprogram))
     :results (dialect/outputs subprogram)
     :algorithm subprogram
     :effects #{}
     :operations (dialect/equations subprogram)}))

(deftest component-predecessor-proof-needs-one-exact-contiguous-boundary
  (let [scalar-value (fn [dtype] (av/tensor {:dtype dtype :shape []}))
        values {'m (scalar-value :long) 'n (scalar-value :long)
                'total (scalar-value :long) 'i (scalar-value :int)
                'current (scalar-value :long) 'after (scalar-value :long)
                'future (scalar-value :long)}
        product '(= product [total]
                   (scalar {:dtypes [:long]} [m n]
                           (lambda [a b] (region [] [(clojure.core/* a b)]))))
        product-program
        (dialect/make
         (dialect/default-program-facts
          {:values values :inputs '[m n]
           :equations {'product (dialect/default-equation-facts)}})
         [product] '[total])
        first-equation '(= component-first [current]
                          (scalar {:dtypes [:long]} [total i]
                                  (lambda [size ordinal] (region [] [size]))))
        second-equation '(= component-second [after]
                           (scalar {:dtypes [:long]} [current]
                                   (lambda [value] (region [] [value]))))
        component
        (dialect/make
         (dialect/default-program-facts
          {:values values :inputs '[total i]
           :equations {'component-first (dialect/default-equation-facts)
                       'component-second (dialect/default-equation-facts)}})
         [first-equation second-equation] '[after])
        component-outers (mapv outer-equation (dialect/equation-subprograms component))
        predecessor {:id :product :operands '[m n] :results '[total]
                     :algorithm product-program :effects #{}
                     :operations (dialect/equations product-program)}
        future {:id :future :operands '[m] :results '[future]
                :algorithm (dialect/make
                            (dialect/default-program-facts
                             {:values values :inputs '[m]
                              :equations {'future-equation (dialect/default-equation-facts)}})
                            ['(= future-equation [future]
                                (scalar {:dtypes [:long]} [m]
                                        (lambda [value] (region [] [value]))))]
                            '[future])
                :effects #{} :operations []}
        parallel {:values values :inputs '[m n i]
                  :equations (vec (concat [predecessor] component-outers [future]))}
        context {:parallel-program parallel :component component}
        environment (proof/component-environment-before context)]
    (is (proof/same-volume? environment 'total '[m n]))
    (is (= :int (:dtype (get environment 'i))))
    (is (= {:const 1 :factors ['i]} (:product (get environment 'i))))
    (is (not (proof/same-volume? environment 'i '[m n])))
    (is (not-any? #(contains? environment %) '[current after future])
        "current and future definitions never enter the predecessor environment")
    (doseq [changed
            [(assoc context :component (dialect/remap-values component {'after 'foreign-after}))
             (assoc-in context [:parallel-program :equations]
                       (vec (concat [predecessor] (take 1 component-outers) [future])))
             (assoc-in context [:parallel-program :equations]
                       (vec (concat [predecessor] component-outers component-outers [future])))
             (assoc-in context [:parallel-program :equations]
                       (vec (concat [predecessor] (reverse component-outers) [future])))
             (assoc-in context [:parallel-program :values 'total :dtype] :int)
             (update-in context [:parallel-program :equations 1 :algorithm]
                        dialect/remap-values {'current 'copied-current})]]
      (is (= {} (proof/component-environment-before changed))))))

(deftest extent-equality-is-a-proof-not-cast-stripping
  (doseq [extent ['(* m n) '(* n m) '(* (long m) n) '(* m (* n 1))]]
    (is (proof/same-volume? environment extent '[m n])))
  (doseq [extent ['(* m n 2) '(* (int m) n) '(* (float m) n)
                  '(unchecked-multiply m n) '(raster.numeric/* i i)
                  '(* m unknown) '(* -1 m) '(* 9223372036854775807 2)]]
    (is (not (proof/same-volume? environment extent '[m n]))))
  (testing "opaque wrapping products cannot equal their mathematical expansion"
    (is (not (proof/same-volume? environment '(raster.numeric/* i i) '[i i]))))
  (testing "unknown/missing shapes and padding do not prove full coverage"
    (is (not (proof/same-volume? environment 1 nil)))
    (is (not (proof/same-volume? environment 9 [8])))
    (is (proof/same-volume? environment 8 [2 4]))))

(deftest guarded-rectangular-products-require-nonnegative-factor-proofs
  (let [guarded '(if (< m 1) 0 (if (< n 1) 0 (* m n)))
        nonnegative (update-vals environment #(assoc % :nonnegative true))]
    (is (proof/same-volume? nonnegative guarded '[m n]))
    (is (not (proof/same-volume? environment guarded '[m n]))
        "arbitrary public Long dimensions may be negative")))

(deftest array-length-results-carry-a-nonnegative-extent-witness
  (let [equation '(= length [n]
                     (scalar {:dtypes [:long]} [input]
                             (lambda [array] (region [] [(clojure.core/alength array)]))))
        facts (dialect/default-program-facts
               {:inputs '[input]
                :values {'input (av/tensor {:dtype :float :shape ['?]})
                         'n (av/tensor {:dtype :long :shape []})}
                :equations {'length (dialect/default-equation-facts {})}})
        environment (proof/advance (proof/initial-environment
                                    (dialect/make facts [equation] '[n]))
                                   facts equation)]
    (is (true? (:nonnegative (get environment 'n))))
    (is (proof/same-volume? environment 'n '[n]))))

(deftest scalar-witnesses-become-available-only-after-their-definition
  (let [equation '(= product [p]
                     (scalar {:dtypes [:long]} [m n]
                             (lambda [x y] (region [] [(* x y)]))))
        facts (dialect/default-program-facts
               {:inputs '[m n]
                :values (zipmap '[m n p] (repeat (av/tensor {:dtype :long :shape []})))
                :equations {'product (dialect/default-equation-facts {})}})
        p (dialect/make facts [equation] '[p])
        before (proof/initial-environment p)
        after (proof/advance before facts equation)]
    (is (not (contains? before 'p)))
    (is (not (proof/same-volume? before 'p '[p])))
    (is (not (proof/same-volume? before 'p '[m n])))
    (is (proof/same-volume? after 'p '[m n]))
    (is (= after (proof/environment p)))))
