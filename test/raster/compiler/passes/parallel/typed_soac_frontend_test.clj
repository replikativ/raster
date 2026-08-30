(ns raster.compiler.passes.parallel.typed-soac-frontend-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.soac-dialect-adapter :as adapter]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def ^:private source
  '(let* [^long n (clojure.core/alength x)
          y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
          total (raster.par/reduce acc 0.0 j n (+ acc (clojure.core/aget y j)))]
         total))

(deftest direct-front-end-matches-the-compatibility-projection-on-the-overlap
  (let [[_ bindings] source
        nodes (legacy/let-bindings->nodes (partition 2 bindings))
        projected (adapter/legacy-nodes->program
                   nodes {:outputs '[total] :dtype :float :array-types {'x :float}
                          :include-scalar-bindings? true})
        direct (frontend/form->program source {:dtype :float :array-types {'x :float}})]
    (is (= (dialect/equations projected) (dialect/equations direct)))
    (is (= (:values (dialect/facts projected)) (:values (dialect/facts direct))))
    (is (= (:inputs (dialect/facts projected)) (:inputs (dialect/facts direct))))
    (is (= (dialect/outputs projected) (dialect/outputs direct)))
    (is (= :analyzed-source (get-in (dialect/facts direct) [:provenance :front-end])))
    (is (every? #(contains? (:provenance %) :source-binding-id)
                (vals (:equations (dialect/facts direct)))))))

(deftest production-route-does-not-load-the-record-adapter-as-its-front-door
  (let [aliases (ns-aliases 'raster.compiler.passes.parallel.typed-soac-route)]
    (is (= 'raster.compiler.passes.parallel.typed-soac-frontend
           (ns-name (get aliases 'frontend))))
    (is (not (contains? aliases 'legacy)))
    (is (not (contains? aliases 'adapter)))
    (is (= :analyzed-source
           (get-in (route/attempt source :float {'x :float}) [:stats :front-end])))))

(deftest parallel-semantics-enter-only-their-exact-typed-operation
  (testing "a certified inclusive scan is represented directly, with destination facts"
    (let [program (frontend/form->program
                   '(let* [result (raster.par/scan target acc 0.0 i n float
                                                   (+ acc (clojure.core/aget x i)))]
                          result)
                   {:dtype :float :array-types {'x :float 'target :float}})
          equation (first (dialect/equations program))]
      (is (= 'scan (dialect/operation-kind equation)))
      (is (= :inclusive (get-in (dialect/operation-parts equation) [:attributes :mode])))
      (is (= '{result target} (get-in (dialect/facts program) [:equations 0 :aliases])))
      (is (= #{:memory/write} (:effects (dialect/facts program))))))
  (testing "exclusive scan is the same certified operation with a distinct result mode"
    (let [program (frontend/form->program
                   '(let* [result (raster.par/scan-exclusive target acc 0.0 i n float
                                                             (+ acc (clojure.core/aget x i)))]
                          result)
                   {:dtype :float :array-types {'x :float 'target :float}})
          equation (first (dialect/equations program))]
      (is (= 'scan (dialect/operation-kind equation)))
      (is (= :exclusive (get-in (dialect/operation-parts equation) [:attributes :mode])))
      (is (= '[(clojure.core/inc n)]
             (:shape (get-in (dialect/facts program) [:values 'result]))))))
  (testing "a general recurrence cannot be mislabeled as an associative scan"
    (try
      (frontend/form->program
       '(let* [result (raster.par/scan target h 0.0 i n float
                                       (Math/tanh (+ h (clojure.core/aget x i))))]
              result)
       {:dtype :float :array-types {'x :float 'target :float}})
      (is false "an uncertified recurrence must decline")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :scan-not-associative (:reason (ex-data exception)))))))
  (testing "destination-writing map identity and effects are explicit compiler facts"
    (let [program (frontend/form->program
                   '(let* [step (raster.par/map! target i n float
                                                 (+ (clojure.core/aget x i) 1.0))]
                          step)
                   {:dtype :float :array-types {'x :float 'target :float}})
          facts (dialect/facts program)]
      (is (= 'map (dialect/operation-kind (first (dialect/equations program)))))
      (is (= '[x] (dialect/operation-inputs (first (dialect/equations program))))
          "a write-only destination is not duplicated as a semantic read operand")
      (is (= '{step target} (get-in facts [:equations 0 :aliases])))
      (is (= :buffer (get-in facts [:equations 0 :attributes :destination-return])))
      (is (some? (get-in facts [:values 'target]))
          "the physical output boundary retains its own value contract")
      (is (= #{:memory/write} (:effects facts)))))
  (testing "a source binder and destination still require distinct SSA identities"
    (is (nil? (frontend/form->program
               '(let* [target (raster.par/map! target i n float
                                               (+ (clojure.core/aget x i) 1.0))]
                      target)
               {:dtype :float :array-types {'x :float 'target :float}}))))
  (testing "a read/write destination is one explicit typed operand"
    (let [program (frontend/form->program
                   '(let* [step (raster.par/map! target i n float
                                                 (+ (clojure.core/aget target i) 1.0))]
                          step)
                   {:dtype :float :array-types {'target :float}})
          equation (first (dialect/equations program))
          facts (dialect/facts program)]
      (is (= '[target] (dialect/operation-inputs equation)))
      (is (= :read-write
             (get-in facts [:equations 0 :attributes :destination-access])))
      (is (= '{step target} (get-in facts [:equations 0 :aliases]))))))

(deftest destination-shapes-refine-across-ordered-maps
  (let [program (frontend/form->program
                 '(let* [first-step (raster.par/map! first-out i n float
                                                     (+ (clojure.core/aget x i) 1.0))
                         second-step (raster.par/map! second-out j n float
                                                      (* (clojure.core/aget first-out j) 2.0))]
                        second-step)
                 {:dtype :float
                  :array-types {'x :float 'first-out :float 'second-out :float}})]
    (is (= '[n] (get-in (dialect/facts program) [:values 'first-out :shape])))
    (is (= 2 (count (dialect/equations program))))))
