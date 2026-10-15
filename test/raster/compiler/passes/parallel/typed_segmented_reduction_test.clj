(ns raster.compiler.passes.parallel.typed-segmented-reduction-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.dialects :as dialects]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-pass]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-contraction-context :as contraction-context]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]))

(defn- tensor [dtype shape]
  (av/tensor {:dtype dtype :shape shape :representation {:kind :plain}}))

(defn- program []
  (let [body '(+ acc
                 (* (aget lhs (+ (* i inner) l))
                    (aget rhs (+ (* l width) j))))
        algebra (scan/certify-reassociation
                 {:acc 'acc :init 0.0 :lambda body} :float)
        equation
        (list '= 'contract-equation ['C]
              (list 'segmented-reduce
                    {:segment-axes '[[i m] [j n]]
                     :index 'l :extent 'k
                     :accumulators ['acc] :identities [0.0]
                     :dtypes [:float] :algebra [algebra]
                     :attributes {:stable-array-captures ['A 'B]}}
                    [] ['A 'B 'n 'k]
                    (dialect/lambda-form ['acc 'lhs 'rhs 'width 'inner] [body])))
        facts
        (assoc (dialect/default-program-facts
                {:front-end :typed-segmented-reduction-test})
               :values {'A (tensor :float '[m k])
                        'B (tensor :float '[k n])
                        'm (tensor :long [])
                        'n (tensor :long [])
                        'k (tensor :long [])
                        'C (tensor :float '[m n])}
               :inputs '[A B m n k]
               :equations {'contract-equation (dialect/default-equation-facts
                                                {:source :test})})]
    (dialect/make facts [equation] '[C])))

(defn- with-result-map [source result-map]
  (let [[equation] (dialect/equations source)
        operation (nth equation 3)
        attributes (assoc (second operation) :result-map result-map)
        operation (apply list (assoc (vec operation) 1 attributes))
        equation (apply list (assoc (vec equation) 3 operation))
        result-shape (axis-map/shape result-map)
        facts (assoc-in (dialect/facts source) [:values 'C :shape] result-shape)]
    (dialect/make facts [equation] '[C])))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest segmented-projection-and-lowering-use-the-same-validated-policy-scope
  (let [retained numerics/retained-source-arithmetic
        blas (numerics/blas-source-arithmetic :float)
        scoped (fn [local global]
                 (let [base (program) [equation] (dialect/equations base)
                       operation (nth equation 3)
                       attributes (cond-> (assoc-in (second operation) [:attributes :source-operation]
                                                   :raster.par/contract)
                                    (some? local) (assoc :source-arithmetic local))
                       operation (apply list (assoc (vec operation) 1 attributes))
                       facts (cond-> (dialect/facts base)
                               (some? global) (assoc-in [:attributes :source-arithmetic] global))]
                   (dialect/make facts [(apply list (assoc (vec equation) 3 operation))]
                                 (dialect/outputs base))))]
    (doseq [[local global expected] [[nil nil retained] [nil retained retained] [nil blas blas]
                                    [retained blas retained] [blas retained blas]]]
      (testing (str "local=" local ", global=" global)
        (let [source (scoped local global) [equation] (dialect/equations source)
              lowered (:operation (lower/project-typed-segmented-reduce source))]
          (is (= expected (:source-arithmetic
                           (projection/segmented-reduce-contract-components source equation))))
          (is (segop/seg-red? lowered))
          (is (= expected (:source-arithmetic lowered))))))
    (is (= :source-arithmetic
           (reason #(let [source (scoped blas :invalid)]
                      (projection/segmented-reduce-contract-components source (first (dialect/equations source)))))))
    (is (= :source-arithmetic
           (reason #(lower/project-typed-segmented-reduce (scoped blas :invalid)))))))

(deftest segmented-result-map-is-an-explicit-regrouping-witness
  (let [result-map {:groups '[[[i m] [j n]]]}
        source (with-result-map (program) result-map)
        attributes (:attributes (dialect/operation-parts
                                 (first (dialect/equations source))))
        projected (:operation (lower/project-typed-segmented-reduce source))]
    (is (= result-map (:result-map attributes)))
    (is (= result-map (:result-map projected)))
    (is (= '[(clojure.core/* m n)]
           (dialect/segmented-reduce-result-shape attributes)))
    (is (= '(clojure.core/+ (clojure.core/* i n) j)
           (dialect/segmented-reduce-result-index attributes)))
    (is (= result-map
           (:result-map (:operation
                         (lower/project-typed-segmented-reduce source)))))
    (let [rank-three (assoc attributes
                            :segment-axes '[[i 2] [j 3] [k 4]]
                            :result-map {:groups '[[[i 2]] [[j 3] [k 4]]]})]
      (is (= [2 12] (dialect/segmented-reduce-result-shape rank-three)))
      (is (= '(clojure.core/+ (clojure.core/* i 12)
                              (clojure.core/* j 4) k)
             (dialect/segmented-reduce-result-index rank-three))))))

(deftest segmented-result-map-is-part-of-the-scheduled-semantic-certificate
  (let [result-map {:groups '[[[i m] [j n]]]}
        grouped (with-result-map (program) result-map)
        [equation] (dialect/equations grouped)
        operation-form (nth equation 3)
        attributes (assoc-in (second operation-form) [:attributes :source-operation]
                             :raster.par/contract)
        operation-form (apply list (assoc (vec operation-form) 1 attributes))
        equation (apply list (assoc (vec equation) 3 operation-form))
        source (dialect/make (dialect/facts grouped) [equation] (dialect/outputs grouped))
        operation (first (lower/lower-typed-segmented-reduce source :ze:0 :dtype :float))]
    (is (some? (:schedule operation)))
    (is (= source (:program (contraction-context/validate-semantic! source operation))))
    (doseq [tampered [(dissoc operation :result-map)
                      (assoc operation :result-map {:groups '[[[i m]] [[j n]]]})]]
      (is (= :typed-contraction-semantic-operation
             (reason #(contraction-context/validate-semantic! source tampered)))))))

(deftest segmented-result-map-refuses-non-regrouping-layouts
  (let [attributes (:attributes (dialect/operation-parts
                                 (first (dialect/equations (program)))))
        rejected [{:groups '[[[j n] [i m]]]}
                  {:groups '[[[i m] [i m] [j n]]]}
                  {:groups '[[[i m]]]}
                  {:groups '[[[i m]] [] [[j n]]]}
                  {:groups '[[[i m] [j (+ n 1)]]]}
                  {:groups '[[[i m] [j n]]] :padding 1}]]
    (doseq [result-map rejected]
      (is (= :typed-soac-segmented-reduce-result-map
             (reason #(dialect/segmented-reduce-result-map
                       (assoc attributes :result-map result-map))))))
    (is (= [] (dialect/segmented-reduce-result-shape {:segment-axes []})))
    (is (= :typed-soac-segmented-reduce-result-map
           (reason #(dialect/segmented-reduce-result-shape
                     {:segment-axes [] :result-map {:groups [[]]}}))))
    (let [product-attributes
          {:segment-axes '[[i 2]] :index 'l :extent 3 :attributes {}
           :component-ids [:sum] :result-components [0]
           :accumulators '[acc] :identities [0.0] :dtypes [:double]
           :algebra {:associative? true}
           :result-map {:groups '[[[i 2]]]}}
          fold-map-attributes
          {:segment-axes '[[i 2]] :index 'l :extent 3 :attributes {}
           :association :ordered :dtypes [:double]
           :result-map {:groups '[[[i 2]]]}}]
      (is (true? (dialect/product-reduce-attributes?
                  (dissoc product-attributes :result-map))))
      (is (false? (dialect/product-reduce-attributes? product-attributes)))
      (is (true? (dialect/segmented-fold-map-attributes?
                  (dissoc fold-map-attributes :result-map))))
      (is (false? (dialect/segmented-fold-map-attributes? fold-map-attributes))))))

(deftest segmented-result-map-remaps-only-extent-identities
  (let [source (with-result-map (program) {:groups '[[[i m] [j n]]]})
        remapped (dialect/remap-values source {'i 'renamed-value
                                               'm [:extent :m]
                                               'n [:extent :n]})
        attributes (:attributes (dialect/operation-parts
                                 (first (dialect/equations remapped))))]
    (is (= '[[i [:extent :m]] [j [:extent :n]]]
           (:segment-axes attributes)))
    (is (= {:groups '[[[i [:extent :m]] [j [:extent :n]]]]}
           (:result-map attributes)))))

(deftest segmented-semantic-projection-is-device-free-and-exact
  (let [source (program)
        operation (first (lower/lower-typed-segmented-reduce source :ze:0 :dtype :float))
        forbidden (fn [& _] (throw (ex-info "semantic projection discovered target hardware" {})))]
    (with-redefs [hardware/descriptor-for forbidden]
      (let [projected (:operation (lower/project-typed-segmented-reduce
                                  source :dtype :float
                                  :flat-idx (get-in operation [:space :flat-idx])))]
        (is (= operation projected))
        (is (= projected (:operation (lower/project-typed-segmented-reduce
                                      source :dtype :float
                                      :flat-idx (get-in operation [:space :flat-idx])))))
        (is (= (:reduction operation) (:reduction projected)))))))

(deftest semantic-projection-retains-core-shadowing-shape-operands
  (let [source (dialect/remap-values (program) {'m 'seq 'n 'count 'k 'first})]
    (binding [util/*shadowing-locals* #{}]
      (let [projected (:operation (lower/project-typed-segmented-reduce source))]
        (is (= '#{seq count first} (:scalars projected)))
        (is (= '[[i seq] [j count] [l first]]
               (mapv (juxt :name :bound) (get-in projected [:space :dims]))))
        (binding [util/*shadowing-locals* '#{seq count first}]
          (is (= projected (:operation (lower/project-typed-segmented-reduce
                                        source :flat-idx (get-in projected [:space :flat-idx]))))))))))

(deftest segmented-reduction-is-a-typed-functional-equation
  (let [program (program)
        equation (first (dialect/equations program))
        attributes (:attributes (dialect/operation-parts equation))]
    (is (= 'segmented-reduce (dialect/operation-kind equation)))
    (is (= '[[i m] [j n]] (:segment-axes attributes)))
    (is (= '[m n k] (dialect/operation-extents equation)))
    (is (= '[A B n k] (dialect/operation-inputs equation)))
    (is (= '[C] (dialect/outputs program)))))

(deftest generic-fusion-keeps-an-unmatched-segmented-reduction-intact
  (let [source (program)
        [result stats] (fusion/fusion-fixpoint source)]
    (is (= source result))
    (is (= :segmented-reduce
           (:kind (fusion/equation-info (first (dialect/equations result))))))
    (is (= 0 (:vertical stats)))
    (is (= 0 (:horizontal stats)))))

(deftest generic-fusion-does-not-drop-an-explicit-result-layout
  (let [source (with-result-map (program) {:groups '[[[i m] [j n]]]})
        [result stats] (fusion/fusion-fixpoint source)]
    (is (= source result))
    (is (= {:groups '[[[i m] [j n]]]}
           (get-in (dialect/operation-parts (first (dialect/equations result)))
                   [:attributes :result-map])))
    (is (= 0 (:vertical stats)))
    (is (= 0 (:horizontal stats)))))

(deftest segmented-reduction-lowers-directly-to-canonical-segred
  (with-redefs [legacy/->SoacContract
                (fn [& _]
                  (throw (ex-info "legacy SoacContract was constructed" {})))]
    (let [operation (first (lower/lower-typed-segmented-reduce
                            (program) :ze:0 :dtype :float))
          dims (:dims (:space operation))
          product (:reduction operation)
          step (first (:results (reduction/fold-region product)))]
      (is (instance? raster.compiler.ir.segop.SegRed operation))
      (is (= '[[i m] [j n] [l k]]
             (mapv (juxt :name :bound) dims)))
      (is (= '#{A B} (:inputs operation)))
      (is (= '#{m n k} (:scalars operation)))
      (is (= '#{C} (:outputs operation)))
      (is (= :segmented (:phase operation)))
      (is (nil? (:schedule operation)))
      (is (reduction/product-reduction? product))
      (is (= :float (first (reduction/dtypes product))))
      (is (= 'C (first (reduction/results product))))
      (is (re-find #"A" (pr-str step)))
      (is (re-find #"B" (pr-str step)))
      (is (not (re-find #"lhs|rhs|width|inner" (pr-str step)))))))

(deftest value-remapping-includes-every-segment-extent
  (let [remapped (dialect/remap-values
                  (program) {'m [:binding 'm] 'n [:binding 'n] 'C [:binding 'C]})
        equation (first (dialect/equations remapped))]
    (is (= [['i [:binding 'm]] ['j [:binding 'n]]]
           (get-in (dialect/operation-parts equation) [:attributes :segment-axes])))
    (is (= [[:binding 'm] [:binding 'n] 'k]
           (dialect/operation-extents equation)))
    (is (= [[:binding 'C]] (dialect/outputs remapped)))))

(deftest typed-parallel-program-schedules-the-equation-without-compatibility-relowering
  (let [result-map {:groups '[[[i m] [j n]]]}
        algorithm (with-result-map (program) result-map)
        values (:values (dialect/facts algorithm))
        source '(raster.par/contract C [[i m] [j n]] [[l k]] body)
        equation
        (parallel-program/->ProgramEquation
         'contract-equation [:binding 'C] source '[A B m n k] '[C]
         algorithm [(first (dialect/equations algorithm))] #{}
         {:source-dialect :typed-soac} {})
        envelope
        (parallel-program/make
         {:dialect :typed-soac :source (list 'let* ['C source] 'C)
          :values values :inputs '[A B m n k] :equations [equation] :outputs '[C]
          :effects #{} :operation? (constantly true)
          :algorithm? (fn [_ candidate] (= candidate (dialect/validate! candidate)))})
        lowered (segop-pass/segop-lower-pass
                 envelope {:target-device :ze:0 :dtype :float})
        scheduled-equation (first (:equations (:form lowered)))
        operation (first (:operations scheduled-equation))]
    (is (dialects/valid-typed-soac-program? envelope)
        "the pipeline boundary admits the TypedSOAC segmented-reduce operation")
    (is (= :segop (:dialect (:form lowered))))
    (is (= 1 (get-in lowered [:stats :typed-soac-reused])))
    (is (= 1 (get-in lowered [:stats :segops-lowered])))
    (is (instance? raster.compiler.ir.segop.SegRed operation))
    (is (= result-map (:result-map operation)))
    (is (= :typed-soac (get-in scheduled-equation [:provenance :source-dialect])))
    (is (empty? (:diagnostics (:form lowered))))))
