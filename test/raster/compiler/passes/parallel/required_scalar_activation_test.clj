(ns raster.compiler.passes.parallel.required-scalar-activation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.required-scalar :as fixture]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as graphs]
            [raster.compiler.passes.parallel.segop-lower-pass :as schedule]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def target-descriptor
  {:subgroup-size 16 :max-workgroup-size 256 :max-warps-per-sm 16
   :max-blocks-per-sm 8 :sm-count 8 :machine-lanes 2048})

(deftest unused-required-scalar-retains-its-own-ordered-resident-stage
  (let [original (fixture/algorithm)
        [realized stats] (resident/realize original)
        original-scalar (second (soac/equations original))
        envelope (route/program-envelope realized)
        scheduled (:form (schedule/segop-lower-pass
                          envelope {:target-device :test:0 :dtype :double
                                    :target-descriptor target-descriptor}))
        scalar (second (:equations scheduled))
        {graph :graph graph-algorithm :algorithm body :body}
        (graphs/make-for-equations scheduled (:equations scheduled))
        nodes (:nodes graph)]
    (is (= 1 (:resident-reductions stats)))
    (is (= 1 (:resident-required-scalars stats)))
    (is (zero? (:inlined-scalars stats)))
    (is (= original-scalar (second (soac/equations realized))))
    (is (= (soac/required-evaluation original 1) (soac/required-evaluation realized 1)))
    (is (= [0 1 2] (mapv :id (:equations scheduled))))
    (is (= '[out] (:outputs scheduled)))
    (is (= {:kind :resident-scalar-buffer :elements 1}
           (get-in scheduled [:values 'unused-loss :representation])))
    (is (= 1 (count (:operations scalar))))
    (is (not (get-in scalar [:attributes :host-only])))
    (is (= #{'total} (get-in scalar [:operations 0 :inputs])))
    (is (= #{'divisor} (get-in scalar [:operations 0 :scalars])))
    (is (resident/validated-required-resident-output envelope 'unused-loss))
    (is (= 4 (count nodes)) "two reduction phases, scalar, map")
    (is (every? (fn [[earlier later]]
                  (some #{(:id earlier)} (:dependencies later)))
                (partition 2 1 nodes))
        "every source fence is represented in the portable DAG")
    (is (thrown? clojure.lang.ExceptionInfo
                 (graphs/validate-projection!
                  (assoc-in graph [:nodes 3 :dependencies] []) graph-algorithm body)))))

(defn- with-scalar-body [captures parameters expression]
  (let [original (fixture/algorithm)
        equations (assoc (vec (soac/equations original)) 1
                         (list '= 1 '[unused-loss]
                               (list 'scalar {:dtypes [:double]} captures
                                     (soac/lambda-form parameters [expression]))))]
    (soac/make (soac/facts original) equations (soac/outputs original))))

(deftest required-placement-does-not-grant-checked-arithmetic-permission
  (doseq [algorithm
          [(with-scalar-body '[total divisor n] '[sum denominator count]
             '(clojure.core/+ sum
                (clojure.core/double (clojure.core/+ 9223372036854775807 count))))
           (with-scalar-body '[total divisor] '[sum denominator]
             '(clojure.core/double (clojure.core/float sum)))]]
    (let [[realized stats] (resident/realize algorithm)]
      (is (zero? (:resident-required-scalars stats)))
      (is (= (second (soac/equations algorithm))
             (second (soac/equations realized)))))))
