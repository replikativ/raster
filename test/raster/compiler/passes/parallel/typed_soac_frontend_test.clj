(ns raster.compiler.passes.parallel.typed-soac-frontend-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
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

(deftest compound-parallel-extents-become-typed-scalar-ssa
  (let [source '(let* [step (raster.par/map! target i
                                              (clojure.core/* nrows width)
                                              float (clojure.core/aget x i))]
                        step)
        normalized (frontend/normalize-source source)
        program (frontend/form->program
                 normalized
                 {:dtype :float
                  :array-types {'x :float 'target :float}
                  :scalar-types {'nrows :long 'width :long}})
        equations (dialect/equations program)]
    (is (= '[rstr_extent_0 step] (mapv (comp first #(nth % 2)) equations)))
    (is (= ['scalar 'map] (mapv dialect/operation-kind equations)))
    (is (= 'rstr_extent_0 (dialect/operation-extent (second equations))))
    (is (= :long (:dtype (get-in (dialect/facts program) [:values 'rstr_extent_0]))))
    (is (= :analyzed-source
           (get-in (route/attempt source :float {'x :float 'target :float}
                                  {:scalar-types {'nrows :long 'width :long}})
                   [:stats :front-end])))))

(deftest guarded-dense-write-is-an-explicit-inout-map
  (let [program
        (frontend/form->program
         '(let* [step (raster.par/map-void! i n
                                             (if (< i limit)
                                               (clojure.core/aset
                                                out i (clojure.core/aget src i))))]
                step)
         {:dtype :float
          :array-types {'src :float 'out :float}
          :scalar-types {'n :long 'limit :long}})
        equation (first (dialect/equations program))
        facts (dialect/facts program)]
    (is (= 'map (dialect/operation-kind equation)))
    (is (= '[out src limit] (dialect/operation-inputs equation)))
    (is (= [{:destination 'out :access :read-write :host-return :effect}]
           (get-in facts [:equations 0 :attributes :result-storage])))
    (is (= '(if (< i %capture0) %element1 %element0)
           (first (:body-results
                   (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))))))))

(deftest unique-indexed-write-is-a-typed-scatter
  (let [scatter '(raster.par/map-void!
                  i n
                  (clojure.core/aset out
                                     (raster.par/unique-index
                                      (clojure.core/aget indices i))
                                     (clojure.core/aget src i)))
        source (list 'let* ['step scatter] 'step)
        program (frontend/form->program
                 source {:dtype :float
                         :array-types {'indices :int 'src :float 'out :float}
                         :scalar-types {'n :long}})
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        write (dialect/write-parts
               (first (:body-results (dialect/lambda-parts (:lambda operation)))))]
    (is (= 'scatter (:kind operation)))
    (is (= :unique (get-in operation [:attributes :conflict])))
    (is (= '[indices src out] (dialect/operation-inputs equation)))
    (is (= {:destination-index '%element0 :predicate 1 :value '%element1} write))
    (is (= [{:destination 'out :access :read-write :host-return :effect}]
           (get-in (dialect/facts program) [:equations 0 :attributes :result-storage])))
    (is (= :analyzed-source
           (get-in (route/attempt source :float
                                  {'indices :int 'src :float 'out :float}
                                  {:scalar-types {'n :long}})
                   [:stats :front-end])))))

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
      (is (= [{:destination 'target :access :write :host-return :buffer}]
             (get-in facts [:equations 0 :attributes :result-storage])))
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
             (get-in facts [:equations 0 :attributes :result-storage 0 :access])))
      (is (= '{step target} (get-in facts [:equations 0 :aliases]))))))

(deftest contraction-enters-as-a-general-typed-segmented-reduction
  (let [source
        '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                       (* (clojure.core/aget A (+ (* i k) l))
                          (clojure.core/aget B (+ (* l n) j))))]
               step)
        options {:dtype :float
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m :long 'n :long 'k :long}}
        program (frontend/form->program source options)
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        routed (route/attempt source :float (:array-types options)
                              {:scalar-types (:scalar-types options)})
        scheduled
        (with-redefs [contraction-facts/contraction-facts
                      (fn [& _]
                        (throw (ex-info "typed scheduling reparsed source" {})))]
          ((requiring-resolve
            'raster.compiler.passes.parallel.segop-lower-pass/segop-lower-pass)
           (:program routed) {:target-device :ze:0 :dtype :float}))
        segcontract (-> scheduled :form :equations first :operations first)]
    (is (= 'segmented-reduce (:kind operation)))
    (is (= '[[i m] [j n]] (get-in operation [:attributes :segment-axes])))
    (is (= '[m n k] (dialect/operation-extents equation)))
    (is (= [{:destination 'C :access :write :host-return :effect}]
           (get-in (dialect/facts program) [:equations 0 :attributes :result-storage])))
    (is (= '[m n] (:shape (get-in (dialect/facts program)
                                  [:values (first (nth equation 2))]))))
    (is (= :analyzed-source (get-in routed [:stats :front-end])))
    (is (instance? raster.compiler.ir.segop.SegContract segcontract))
    (is (= 'C (get-in segcontract [:facts :out])))
    (is (nil? (some #(when (instance? raster.compiler.ir.soac.SoacContract %) %)
                    (tree-seq coll? seq (:form scheduled)))))
    (is (= :typed-soac (get-in (-> scheduled :form :equations first)
                               [:attributes :algorithm-dialect])))))

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

(deftest effect-only-pointwise-writes-have-logical-results-and-physical-storage
  (doseq [[label expression]
          [["map2"
            '(raster.par/map2! a b i n float
                               (+ (clojure.core/aget x i) 1.0)
                               (* (clojure.core/aget y i) 2.0))]
           ["independent multi-store map-void"
            '(raster.par/map-void!
              i n
              (do (clojure.core/aset a i (float (+ (clojure.core/aget x i) 1.0)))
                  (clojure.core/aset b i (float (* (clojure.core/aget y i) 2.0)))))]]]
    (testing label
      (let [program (frontend/form->program
                     (list 'let* ['effect expression] 'effect)
                     {:dtype :float
                      :array-types {'x :float 'y :float 'a :float 'b :float}})
            equation (first (dialect/equations program))
            facts (dialect/facts program)
            results (vec (nth equation 2))]
        (is (= [[:effect-map 0 0] [:effect-map 0 1]] results))
        (is (= ['a 'b] (dialect/physical-results program equation)))
        (is (= [:write :write]
               (mapv :access (dialect/result-storage program 0))))
        (is (= #{:memory/write} (:effects facts)))
        (is (= [] (dialect/outputs program))
            "the host nil result is not mislabeled as a tensor result")))))

(deftest typed-shared-locals-become-one-region-ssa-spine
  (let [expression
        '(raster.par/map-void!
          i n
          (let* [^float shifted (+ (clojure.core/aget x i) 1.0)
                 ^float squared (* shifted shifted)]
                (clojure.core/aset a i (float shifted))
                (clojure.core/aset b i (float squared))))
        program (frontend/form->program
                 (list 'let* ['effect expression] 'effect)
                 {:dtype :float :array-types {'x :float 'a :float 'b :float}})
        equation (first (dialect/equations program))
        {:keys [locals body-results]}
        (dialect/lambda-parts (:lambda (dialect/operation-parts equation)))]
    (is (= [{:id 'rstr_local_0 :dtype :float
             :init '(+ %element0 1.0)}
            {:id 'rstr_local_1 :dtype :float
             :init '(* rstr_local_0 rstr_local_0)}]
           locals))
    (is (= '[(float rstr_local_0) (float rstr_local_1)] body-results))
    (is (= '[n x] (:inputs (dialect/facts program))))
    (is (not-any? #{'rstr_local_0 'rstr_local_1}
                  (keys (:values (dialect/facts program))))
        "region-local SSA values are lexical, not fake program inputs")))

(deftest ordered-void-bodies-decline-the-functional-tuple-map
  (doseq [[label body]
          [["one destination written twice"
            '(do (clojure.core/aset a i (float 1.0))
                 (clojure.core/aset a i (float 2.0)))]
           ["a later store observes an earlier sibling write"
            '(do (clojure.core/aset a i (float (clojure.core/aget x i)))
                 (clojure.core/aset b i (float (clojure.core/aget a i))))]
           ["a typed local cannot hide a sibling destination dependency"
            '(let* [^float observed (clojure.core/aget a i)]
                   (clojure.core/aset a i (float (clojure.core/aget x i)))
                   (clojure.core/aset b i (float observed)))]
           ["an untyped shared local cannot enter a typed region"
            '(let* [v (+ (clojure.core/aget x i) 1.0)]
                   (clojure.core/aset a i (float v))
                   (clojure.core/aset b i (float (* v v))))]
           ["transitively shared locals without retained types also decline"
            '(let* [u (+ (clojure.core/aget x i) 1.0)
                    v (* u 2.0)]
                   (clojure.core/aset a i (float v))
                   (clojure.core/aset b i (float u)))]
           ["an indexed write without an explicit conflict contract"
            '(clojure.core/aset a (clojure.core/aget indices i)
                                (float (clojure.core/aget x i)))]]]
    (testing label
      (is (nil? (frontend/form->program
                 (list 'let* ['effect (list 'raster.par/map-void! 'i 'n body)] 'effect)
                 {:dtype :float
                  :array-types {'x :float 'a :float 'b :float 'indices :int}}))))))
