(ns raster.compiler.ir.invocation-shape-closure-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-source :as source]
            [raster.compiler.passes.parallel.structured-control-route :as route]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.core :refer [deftm]]
            [raster.par :as par]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- allocation-plan [bindings]
  (let [integer (av/tensor {:dtype :long :shape []})
        buffer (av/tensor {:dtype :double :shape ['count]})]
    (invocation/from-prefix
     {:id :core-name-shape :parameters '[n] :parameter-values {'n integer}
      :bindings bindings :binding-values {'count integer 'out buffer}
      :program-values {'out buffer} :program-inputs '[out] :program-outputs '[out]})))

(deftest dimensions-and-closures-retain-lexical-core-names
  (is (= #{'count} (source/dimension-references 'count #{})))
  (is (= #{'count 'n}
         (source/dimension-references '(clojure.core/* count n) #{'count 'n})))
  (is (= #{'count} (source/dimension-references '(extent count) #{'count})))
  (is (= #{'count} (source/dimension-references '(extent count) #{})))
  (is (= #{'count} (source/dimension-references '(value count) #{})))
  (is (= #{'extent} (source/dimension-references 'extent #{})))
  (is (= #{'extent 'count 'n}
         (source/dimension-references '(extent count n) #{'count 'n})))
  (is (= #{'unknown-shape 'count}
         (source/dimension-references '(unknown-shape count) #{'count})))
  (is (= #{'count 'size}
         (#'frontend/scalar-dependency-closure
          {'count {:expr '(clojure.core/* n n)} 'size {:expr '(clojure.core/inc count)}}
          #{'size})))
  (is (= #{'count 'size 'out}
         (#'route/required-prefix-symbols
          '[[count (clojure.core/* n n)] [size (clojure.core/inc count)]
            [out (clojure.core/double-array size)]] #{'out}))))

(deftest allocation-plan-closes-shape-over-an-actual-earlier-producer
  (let [plan (allocation-plan '[[count (clojure.core/* n n)]
                                [out (clojure.core/double-array count)]])
        [producer allocation] (:steps plan)]
    (is (invocation/scalar-compute? producer))
    (is (= [{:symbol 'count :value (:id producer)}] (:operands allocation)))
    (is (= :invocation-buffer-allocation
           (reason #(invocation/validate! (assoc-in plan [:steps 1 :operands] [])))))
    (is (= :invocation-step-use-before-definition
           (reason #(invocation/validate!
                     (assoc-in plan [:steps 1 :operands 0 :value] :future))))))
  (is (= :invocation-prefix-free-value
         (reason #(allocation-plan '[[out (clojure.core/double-array count)]]))))
  (is (= :invocation-prefix-free-value
         (reason #(allocation-plan '[[out (clojure.core/double-array count)]
                                    [count (clojure.core/* n n)]])))))

(deftm write-before-core-named-shape
  [early :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! early i 1 double (+ (aget early i) 1.0))
        ^long count (* (long n) (long n))
        out (double-array count)]
    (par/map! out i count double 1.0)
    out))

(deftest core-named-shape-has-one-source-ordered-checked-producer
  (let [compilation (equation-first/compile #'write-before-core-named-shape
                                           {:target :ocl:0 :dtype :double})]
    (doseq [n [0 2 Long/MAX_VALUE]]
      (let [initial (invocation-link/begin-ordered-realization
                     compilation [(double-array [7.0]) n] nil)
            position (first (keep-indexed #(when (= 'count (:binding %2)) %1)
                                          (:source-order initial)))
            entry (nth (:source-order initial) position)
            at-host (assoc initial :source-cursor position)
            calls (atom 0)
            evaluate (fn [step operands]
                       (swap! calls inc)
                       (scalar/evaluate-invocation-step (:source-ns compilation) step operands))
            duplicate (fn [& _] (throw (AssertionError. "duplicate host evaluation")))]
        (is (pos? position))
        (is (some? (:invocation-step entry)))
        (is (some? (:equation entry)))
        (is (empty? (:acquired-storage initial)))
        (is (= :invocation-ordered-next-host
               (reason #(invocation-link/advance-host-source initial evaluate duplicate nil))))
        (is (zero? @calls))
        (if (= n Long/MAX_VALUE)
          (is (thrown? ArithmeticException
                       (invocation-link/advance-host-source at-host evaluate duplicate nil)))
          (let [advanced (invocation-link/advance-host-source at-host evaluate duplicate nil)
                result (get-in advanced [:host-scalars 'count])
                prepared (invocation-link/prepare-source-allocation
                          advanced
                          (fn [expression scalars]
                            (scalar/evaluate-array-size (:source-ns compilation)
                                                        (update-vals scalars :value) expression))
                          nil)
                materialized (:materialization (:state prepared))]
            (is (= {:type :long :value (* n n)} result))
            (is (= result (get-in advanced [:materialization :values (:invocation-step entry)])))
            (is (= (* n n) (:elements prepared)))
            (is (= materialized (materialization/validate-state! materialized)))
            (is (= :source-allocation (get-in prepared [:state :pending :kind])))
            (is (= (:source-cursor advanced) (get-in prepared [:state :source-cursor])))
            (is (empty? (get-in prepared [:state :acquired-storage])))))
        (is (= 1 @calls))
        (is (zero? (get-in initial [:materialization :next-step])))))))
