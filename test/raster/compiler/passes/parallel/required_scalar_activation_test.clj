(ns raster.compiler.passes.parallel.required-scalar-activation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.fixtures.required-scalar :as fixture]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as graphs]
            [raster.compiler.passes.parallel.segop-lower-pass :as schedule]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(deftest declined-source-keeps-the-producer-namespace-without-granting-admission
  (let [quotient (with-meta '(clojure.core// (clojure.core/double n) 2.0)
                   {:raster.effect/preserve-evaluation true :raster.type/tag 'double})
        source (list 'let*
                     ['effect '(raster.par/map! out i n double (clojure.core/aget input i))
                      'quotient quotient]
                     'out)
        attempt (fn [source-ns]
                  (entry/call-with-definition-inputs
                   #(route/attempt source :double {'input :double 'out :double}
                                   {:scalar-types {'n :long} :source-ns source-ns
                                    :resident-reductions? true})))
        unknown (attempt nil)
        known (attempt (the-ns 'raster.compiler.passes.parallel.required-scalar-activation-test))
        refused (fn [result] (first (get-in result [:declined :bindings])))]
    (doseq [result [unknown known]]
      (is (= :typed-soac-source-coverage (get-in result [:declined :reason])))
      (is (not (contains? result :program)))
      (is (= #{:preserve-source-evaluation} (:source-evaluation-flags (refused result)))))
    (is (nil? (:normal-completion (refused unknown))))
    (is (= {:carrier "double" :presence :present :builtin-class "java.lang.Double"}
           (:normal-completion (refused known)))
        "normal completion is diagnostic evidence, not operand completion or source motion")
    (let [masked (create-ns (gensym "raster.test.masked-core."))
          producer (create-ns (gensym "raster.test.required-scalar-producer."))]
      (try
        (intern masked 'double (fn [_] nil))
        (binding [*ns* producer] (alias 'clojure.core (ns-name masked)))
        (is (= :intrinsic-source-definition-mismatch
               (try (attempt producer) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
            "the forwarded context must authenticate the actual qualifier lookup")
        (finally (remove-ns (ns-name producer)) (remove-ns (ns-name masked)))))))

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
