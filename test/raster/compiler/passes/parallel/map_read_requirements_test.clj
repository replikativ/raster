(ns raster.compiler.passes.parallel.map-read-requirements-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.index-algebra :as algebra]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.map-read-requirements :as requirements]))

(deftest graph-minimum-requires-structural-capacity-or-the-exact-checked-guard
  (is (requirements/graph-capacity-covers? 8 4 []))
  (is (requirements/graph-capacity-covers? 832 (launch/product 13 32) []))
  (is (requirements/graph-capacity-covers? (launch/product 26 32) 416 []))
  (is (not (requirements/graph-capacity-covers? 415 (launch/product 13 32) [])))
  (is (not (requirements/graph-capacity-covers? 832 (launch/product 'n 32) [])))
  (is (not (requirements/graph-capacity-covers?
            (launch/product Long/MAX_VALUE 2) (launch/product Long/MAX_VALUE 3) [])))
  (is (requirements/graph-capacity-covers? 'capacity (launch/product 13 32)
                                           [{:expression 'capacity :op :>= :value 416}]))
  (is (not (requirements/graph-capacity-covers? 'capacity (launch/product 13 32)
                                                [{:expression 'capacity :op :>= :value 415}])))
  (is (requirements/graph-capacity-covers? 'capacity 'n
                                           [{:expression 'capacity :op :>= :value 'n}]))
  (doseq [conditions [[]
                      [{:expression 'other :op :>= :value 'n}]
                      [{:expression 'capacity :op :<= :value 'n}]
                      [{:expression 'capacity :op :>= :value 'other}]]]
    (is (not (requirements/graph-capacity-covers? 'capacity 'n conditions)))))

(deftest source-read-rewriting-keeps-embedded-compiler-records-opaque
  (let [certificate (scan/->AssociativeScan 'acc 0.0 '+ '(aget input i) 0.0 :float)
        expression (list 'pair certificate '(aget input i))
        rewritten (descriptor/rewrite-aget-indices expression {'input 'projected})]
    (is (identical? certificate (second rewritten)))
    (is (= '(aget input projected) (nth rewritten 2)))))

(defn- packed-head-map
  [coordinate]
  (segop/map->SegMap
   {:id :packed-heads
    :space (segop/make-seg-space 'idx 'total)
    :inputs #{'input} :outputs #{'output} :scalars '#{rows heads width total}
    :dtype :float :out-sym 'output
    :scalar-region
    {:locals
     '[{:id column :dtype :long :init (rem idx width)}
       {:id q :dtype :long :init (quot idx width)}
       {:id head :dtype :long :init (quot q rows)}
       {:id row :dtype :long :init (rem q rows)}]
     :result (list 'aget 'input coordinate)}}))

(defn- phased-fold-map [fold-expression map-expression]
  (segop/map->SegFoldMap
   {:id :read-phases
    :space (segop/make-seg-space-nd [{:name 'row :bound 3} {:name 'column :bound 17}])
    :index 'column :extent 17
    :folds [{:accumulator 'sum :dtype :float :identity 0.0 :extent 17
             :locals [] :step fold-expression :association :ordered}]
    :map-results [map-expression]
    :inputs '#{x gamma beta} :outputs '[out] :dtypes [:float] :scalars #{}}))

(deftest fold-map-read-minima-cover-all-phases-not-the-output-size
  (let [operation (phased-fold-map
                   '(+ sum (aget x (+ (* row 17) column)))
                   '(+ (* sum (aget gamma column)) (aget beta column)))
        minima (requirements/segfoldmap-read-requirements operation {})]
    (is (requirements/capacity-covers? 51 (get minima 'x)))
    (is (requirements/capacity-covers? 17 (get minima 'gamma)))
    (is (requirements/capacity-covers? 17 (get minima 'beta)))
    (is (not (requirements/capacity-covers? 50 (get minima 'x))))
    (is (not (requirements/capacity-covers? 16 (get minima 'beta))))
    (is (= minima (requirements/segfoldmap-read-requirements operation {})))))

(deftest fold-map-unknown-read-is-not-masked-by-a-known-phase
  (let [operation (assoc (phased-fold-map '(+ sum (aget x column))
                                         '(aget x (aget offsets row)))
                         :inputs '#{x offsets})
        minima (requirements/segfoldmap-read-requirements operation {})]
    (is (not (contains? minima 'x)))
    (is (requirements/capacity-covers? 3 (get minima 'offsets))))
  (let [operation (-> (phased-fold-map '(+ sum (aget x column)) '(aget x column))
                      (assoc :inputs #{'x})
                      (assoc-in [:folds 0 :identity] '(aget x (aget x row))))]
    (is (not (contains? (requirements/segfoldmap-read-requirements operation {}) 'x)))))

(deftest fold-map-multiple-read-phases-preserve-the-largest-minimum
  (let [operation (-> (phased-fold-map '(+ sum (aget x column)) '(aget x column))
                      (assoc :inputs #{'x})
                      (assoc-in [:folds 0 :extent] 23))
        required (get (requirements/segfoldmap-read-requirements operation {}) 'x)]
    (is (requirements/capacity-covers? 23 required))
    (is (not (requirements/capacity-covers? 22 required)))))

(deftest fold-map-local-and-bound-reads-participate-in-the-proof
  (let [operation (-> (phased-fold-map '(+ sum value) '(aget x column))
                      (assoc :inputs '#{x lengths})
                      (assoc-in [:folds 0 :locals]
                                '[{:id value :dtype :float :init (aget x column)}])
                      (assoc-in [:folds 0 :extent] '(aget lengths row)))
        minima (requirements/segfoldmap-read-requirements operation {})]
    (is (not (contains? minima 'x)))
    (is (requirements/capacity-covers? 3 (get minima 'lengths)))))

(deftest fold-map-empty-step-does-not-erase-identity-or-map-reads
  (let [operation (-> (phased-fold-map '(+ sum (aget x column)) '(aget beta column))
                      (assoc :inputs '#{x beta gamma})
                      (assoc-in [:folds 0 :extent] 0)
                      (assoc-in [:folds 0 :identity] '(aget gamma row)))
        minima (requirements/segfoldmap-read-requirements operation {})]
    (is (= 0 (get minima 'x)))
    (is (requirements/capacity-covers? 3 (get minima 'gamma)))
    (is (requirements/capacity-covers? 17 (get minima 'beta)))))

(deftest symbolic-packing-read-retains-its-exact-runtime-capacity
  (let [coordinate '(+ (* row (* heads width)) (+ (* head width) column))
        certificate (requirements/symbolic-read-certificate
                     (packed-head-map coordinate)
                     {:scalar-definitions {'total (launch/product 'rows 'heads 'width)}})
        result (:requirements certificate)]
    (is (= {:const 1 :factors '[heads rows width]}
           (algebra/monomial (get result 'input))))
    (is (= coordinate (get-in certificate [:reads 0 :source-coordinate])))
    (is (= {'total (launch/product 'rows 'heads 'width)}
           (:scalar-definitions certificate)))
    (is (= (get-in (packed-head-map coordinate) [:scalar-region :locals])
           (:source-locals certificate)))))

(deftest symbolic-read-proof-declines-padding-translation-and-indirection
  (doseq [coordinate ['(+ base (* row (* heads width)) (+ (* head width) column))
                      '(+ (* row stride) column)
                      '(aget indices idx)]]
    (is (nil? (requirements/symbolic-read-requirements
               (packed-head-map coordinate)
               {:scalar-definitions {'total (launch/product 'rows 'heads 'width)}})))))

(defn- scalar-scan [element inputs bound]
  (segop/map->SegScan
   {:id :scan-read-proof :space (segop/make-seg-space 'i bound)
    :scan-op {:element element} :inputs inputs :outputs #{'out}
    :scalars #{'n} :phase :single :dtype :float}))

(deftest fold-read-minima-are-complete-per-buffer
  (is (= {'labels 'n}
         (requirements/fold-read-requirements
          (scalar-scan '(+ (aget labels i) (aget logits (aget indices i)))
                       #{'labels 'logits} 'n) {})))
  (doseq [element ['(aget input (+ i 1))
                   '(aget input (aget indices i))
                   '(+ (aget input i) (aget input (+ i 1)))]]
    (is (= {} (requirements/fold-read-requirements
               (scalar-scan element #{'input} 'n) {}))))
  (is (= {'input 0}
         (requirements/fold-read-requirements
          (scalar-scan '(aget input i) #{'input} 0) {}))))

(deftest resident-fold-minima-only-name-current-phase-inputs
  (let [operation (assoc (scalar-scan '(aget input i) #{'input 'capture} 0)
                         :reduction {:attributes
                                     {:resident-scalar-captures #{'capture 'absent}}})]
    (is (= {'input 0 'capture 1}
           (requirements/fold-read-requirements operation {})))
    (is (= {'input 'n 'capture 1}
           (requirements/fold-read-requirements
            (assoc operation :space (segop/make-seg-space 'i 'n)) {})))))

(deftest ordered-loop-read-span-includes-the-counted-axis
  (let [operation
        (segop/map->SegMap
         {:id :row-dot :space (segop/make-seg-space 'row 'rows)
          :inputs #{'input} :outputs #{'output} :scalars '#{rows width}
          :dtype :float :out-sym 'output
          :scalar-region
          {:locals []
           :result
           '(loop* [d 0 acc 0.0]
              (if (< (long d) (long width))
                (recur (inc (long d))
                       (+ acc (aget input (+ (* row width) d))))
                acc))}})
        certificate (requirements/symbolic-read-certificate operation {})]
    (is (= {'input (launch/product 'rows 'width)} (:requirements certificate)))
    (is (= {'d 'width} (get-in certificate [:reads 0 :loop-indices])))
    (is (= '(clojure.core/+ (clojure.core/* row width) d)
           (get-in certificate [:reads 0 :projected-coordinate])))))

(deftest triangular-fold-uses-the-enclosing-digit-radix-as-a-safe-loop-bound
  (let [operation
        (segop/map->SegMap
         {:id :triangular-row :space (segop/make-seg-space 'bi '(clojure.core/* batch seq-len))
          :inputs #{'input} :outputs #{'output} :scalars '#{batch seq-len}
          :dtype :float :out-sym 'output
          :scalar-region
          {:locals
           '[{:id b :dtype :long :init (quot bi seq-len)}
             {:id i :dtype :long :init (rem bi seq-len)}
             {:id base :dtype :long
              :init (+ (* b (* seq-len seq-len)) (* i seq-len))}
             {:id sum :dtype :float
              :init (fold {:accumulator acc, :index j, :identity 0.0, :lower 0,
                           :dtype :float, :extent i, :association :ordered,
                           :upper-bound :inclusive}
                          (lambda [acc j]
                            (region [] [(+ acc (aget input (+ base j)))])))}]
           :result 'sum}})
        certificate (requirements/symbolic-read-certificate operation {})]
    (is (= {'input (launch/product 'batch 'seq-len 'seq-len)}
           (:requirements certificate)))
    (is (= {:const 1 :factors '[seq-len]}
           (get-in certificate [:reads 0 :loop-indices 'j])))))

(deftest triangular-fold-range-uses-retained-host-scalar-definitions
  (let [operation
        (segop/map->SegMap
         {:id :batched-head-triangle :space (segop/make-seg-space 't 'total)
          :inputs #{'weights} :outputs #{'output}
          :scalars '#{batch heads rows width slab square total}
          :dtype :float :out-sym 'output
          :scalar-region
          {:locals
           '[{:id bq :dtype :long :init (quot t slab)}
             {:id within :dtype :long :init (rem t slab)}
             {:id row :dtype :long :init (quot within width)}
             {:id column :dtype :long :init (rem within width)}
             {:id base :dtype :long :init (+ (* bq square) (* row rows))}
             {:id sum :dtype :float
              :init (fold {:accumulator acc, :index j, :identity 0.0, :lower 0,
                           :dtype :float, :extent row, :association :ordered,
                           :upper-bound :inclusive}
                          (lambda [acc j]
                            (region [] [(+ acc (aget weights (+ base j)))])))}]
           :result 'sum}})
        certificate
        (requirements/symbolic-read-certificate
         operation
         {:scalar-definitions
          {'slab (launch/product 'rows 'width)
           'square (launch/product 'rows 'rows)
           'total (launch/product 'batch 'heads 'rows 'width)}})]
    (is (= {'weights (launch/product 'batch 'heads 'rows 'rows)}
           (:requirements certificate)))
    (is (= {:const 1 :factors '[rows]}
           (get-in certificate [:reads 0 :loop-indices 'j])))))

(deftest address-projection-expands-only-coordinate-dependencies
  (let [noise-locals
        (mapv (fn [index]
                {:id (symbol (str "noise" index))
                 :dtype :long
                 :init (if (zero? index)
                         1
                         (let [prior (symbol (str "noise" (dec index)))]
                           (list '+ prior prior)))})
              (range 20))
        operation
        (segop/map->SegMap
         {:id :irrelevant-large-scalar-region
          :space (segop/make-seg-space 'row 'rows)
          :inputs #{'input} :outputs #{'output} :scalars '#{rows}
          :dtype :float :out-sym 'output
          :scalar-region {:locals noise-locals :result '(aget input row)}})
        certificate (requirements/symbolic-read-certificate operation {})]
    (is (= {'input 'rows} (:requirements certificate)))
    (is (= 'row (get-in certificate [:reads 0 :projected-coordinate])))
    (is (empty? (get-in certificate [:reads 0 :address-locals])))))
