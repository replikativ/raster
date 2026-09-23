(ns raster.compiler.passes.parallel.map-read-requirements-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.index-algebra :as algebra]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.map-read-requirements :as requirements]))

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
