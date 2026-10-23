(ns raster.compiler.passes.parallel.device-scalar-substrate-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.backend.gpu.parallel-program-c-family :as target]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.compiler.passes.parallel.soac-lower :as lower]))

(def ^:private target-descriptor
  {:subgroup-size 16 :max-workgroup-size 256 :max-warps-per-sm 16
   :max-blocks-per-sm 8 :sm-count 8 :machine-lanes 2048})

(defn- resident-scalar [dtype]
  (av/tensor {:dtype dtype :shape [] :memory-space :device
              :representation {:kind :resident-scalar-buffer :elements 1}}))

(defn- scalar-program
  ([] (scalar-program {}))
  ([{:keys [result-value input-value bias-value operation-attributes equation-facts
            input-expression program-attributes]
     :or {result-value (resident-scalar :double)
          input-value (av/tensor {:dtype :float :shape '[(extent out)]})
          bias-value (av/tensor {:dtype :double :shape []})
          operation-attributes {:dtypes [:double]}
          equation-facts (dialect/default-equation-facts)
          input-expression '(clojure.core/aget array-value 0)
          program-attributes {}}}]
   (let [load (dialect/local-value 'loaded :float input-expression)
         widen (dialect/local-value
                'widened :double
                (dialect/scalar-convert
                 {:source-dtype :float :target-dtype :double
                  :rounding :exact :overflow :exact
                  :source-op 'clojure.core/double}
                 'loaded))
         equation
         (list '= 'terminal-scalar '[loss]
               (list 'scalar operation-attributes '[out bias]
                     (dialect/lambda-form '[array-value bias-value]
                                          [load widen]
                                          '[(clojure.core/+ widened bias-value)])))]
     (dialect/make
      (dialect/default-program-facts
       {:values {'out input-value
                 'bias bias-value
                 'loss result-value}
        :inputs '[out bias]
        :equations {'terminal-scalar equation-facts}
        :attributes program-attributes
        :effects (:effects equation-facts)})
      [equation] '[loss]))))

(defn- required-resident-payload-program []
  (let [obligation {:source-binding-id 0 :flags #{:preserve-source-evaluation}
                    :ordering :source}]
    (scalar-program
     {:input-value (resident-scalar :float)
      :input-expression 'array-value
      :equation-facts (assoc-in (dialect/default-equation-facts {:source-binding-id 0})
                               [:attributes :required-evaluation] obligation)
      :program-attributes {:required-evaluations {'terminal-scalar obligation}}})))

(defn- scheduled-program
  [algorithm operation]
  (let [facts (dialect/facts algorithm)
        equation
        (program/->ProgramEquation
         'terminal-scalar [:test :device-scalar] nil '[out bias] '[loss]
         algorithm [operation] #{} {:source :test}
         {:device :test:0})]
    (program/->ParallelProgram
     :scheduled-parallel nil (:values facts) '[out bias] [equation] '[loss]
     #{} [] {:source :test} {})))

(defn- reason-of [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest required-resident-payload-has-one-explicit-pointer-load
  (let [algorithm (required-resident-payload-program)
        operation (first (lower/lower-typed-scalar algorithm :test:0
                                                  :target-descriptor target-descriptor))
        reads (descriptor/aget-reads (:lambda operation))]
    (is (= #{'out} (:inputs operation)))
    (is (= #{'bias} (:scalars operation)))
    (is (= [{:sym 'out :idx 0}] (mapv #(select-keys % [:sym :idx]) reads)))
    (is (= (get-in (dialect/facts algorithm) [:equations 'terminal-scalar :attributes])
           (:algorithm-equation-attributes operation)))
    (is (= operation (lower/validate-typed-scalar-schedule! algorithm operation)))
    (doseq [target-dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (let [emitted (target/emit-program (scheduled-program algorithm operation)
                                        {:target-dialect target-dialect})]
        (is (= algorithm (get-in emitted [:program :equations 0 :algorithm])))))
    (doseq [forged [(assoc operation :inputs #{} :scalars #{'out 'bias})
                    (assoc-in operation [:scalar-region :locals 0 :init]
                              '(clojure.core/aget out 1))]]
      (is (= :typed-soac-device-scalar-correspondence
             (reason-of #(lower/validate-typed-scalar-schedule! algorithm forged)))))))

(deftest the-same-scalar-schedule-emits-through-every-existing-target-dialect
  (let [algorithm (scalar-program)
        operation (first (lower/lower-typed-scalar
                          algorithm :test:0 :target-descriptor target-descriptor))
        scheduled (scheduled-program algorithm operation)]
    (doseq [dialect [:opencl-intel :opencl-portable :cuda :hip]]
      (let [emitted (target/emit-program scheduled {:target-dialect dialect})]
        (is (= 1 (count (:kernels emitted))) (str dialect))
        (is (= {:kernel-body 1} (get-in emitted [:stats :emission-routes])))
        (is (zero? (get-in emitted [:stats :host-scalar-equations])))))))

(deftest scalar-ir-lowers-to-one-exact-extent-one-segmap
  (let [algorithm (scalar-program)
        operation (first (lower/lower-typed-scalar
                          algorithm :test:0 :target-descriptor target-descriptor))
        locals (get-in operation [:scalar-region :locals])]
    (is (segop/seg-map? operation))
    (is (= [{:name (get-in operation [:space :dims 0 :name]) :bound 1}]
           (get-in operation [:space :dims])))
    (is (= #{'out} (:inputs operation)))
    (is (= #{'bias} (:scalars operation)))
    (is (= #{'loss} (:outputs operation)))
    (is (= 'loss (:out-sym operation)))
    (is (= [:float :double] (mapv :dtype locals)))
    (is (= '(clojure.core/aget out 0) (:init (first locals))))
    (is (dialect/scalar-convert-form? (:init (second locals))))
    (is (identical? operation (lower/validate-typed-scalar-schedule! algorithm operation)))))

(deftest scalar-schedule-reuses-map-capacity-and-kernel-body-authorities
  (let [algorithm (scalar-program
                   {:result-value
                    (resident-scalar :double)})
        operation (first (lower/lower-typed-scalar
                          algorithm :test:0 :target-descriptor target-descriptor))
        scheduled (scheduled-program algorithm operation)
        equation (first (:equations scheduled))
        {:keys [body graph]} (equation-graph/make-for-equation scheduled equation)
        lowered-body (segmap-body/lower
                      operation {:array-types {'out :float 'loss :double}
                                 :scalar-types {'bias :double}})
        kernel (:kernel-body lowered-body)
        parameters (into {} (map (juxt :id identity)) (:parameters kernel))
        operations (tree-seq #(or (map? %) (sequential? %))
                             #(if (map? %) (vals %) (seq %)) (:operations kernel))]
    (is (= scheduled (program/validate! scheduled)))
    (is (= 1 (:elements (first (:inputs graph))))
        "the ordinary map read proof derives the exact aget[0] capacity")
    (is (= 1 (:elements (first (:outputs graph))))
        "rank-zero semantics has one physical result element")
    (is (= [] (get-in (dialect/facts algorithm) [:values 'loss :shape])))
    (is (= [1] (:shape (parameters 'out))))
    (is (= 1 (:bound lowered-body)))
    (is (= [(some #(when (= :bound (:role %)) (:id %)) (:parameters kernel))]
           (:shape (parameters 'loss)))
        "the existing map ABI bounds result storage by the exact extent-one launch binding")
    (is (some #(and (= :cast (get-in % [:expression :op]))
                    (= :double (get-in % [:expression :result-type]))
                    (= {:rounding :exact :overflow :exact}
                       (select-keys (get-in % [:expression :options])
                                    [:rounding :overflow])))
              operations)
        "the typed Float-to-Double checkpoint reaches the ordinary KernelBody lowerer")
    (is (= body (:body (equation-graph/make-for-equation scheduled equation))))))

(deftest scalar-schedule-correspondence-refuses-forged-segmap-semantics
  (let [algorithm (scalar-program)
        operation (first (lower/lower-typed-scalar
                          algorithm :test:0 :target-descriptor target-descriptor))
        forgeries
        [(assoc-in operation [:space :dims 0 :bound] 2)
         (assoc operation :inputs #{} :scalars #{'out 'bias})
         (assoc-in operation [:scalar-region :result] '(clojure.core/- widened bias))
         (update-in operation [:scalar-region :result]
                    #(with-meta % {:raster.type/tag 'float}))
         (assoc operation :dtype :float)
         (assoc operation :out-sym 'out)
         (assoc-in operation [:grid :block-size] 0)]]
    (doseq [forged forgeries]
      (is (= :typed-soac-device-scalar-correspondence
             (reason-of #(lower/validate-typed-scalar-schedule! algorithm forged)))))
    (let [forged (assoc-in operation [:scalar-region :result]
                           '(clojure.core/- widened bias))]
      (is (= :scheduled-device-scalar-correspondence
             (reason-of #(equation-graph/make-for-equation
                          (scheduled-program algorithm forged)
                          (first (:equations (scheduled-program algorithm forged))))))))))

(deftest scalar-lowering-refuses-unproved-logical-and-storage-boundaries
  (testing "a shaped logical result is not silently flattened"
    (is (= :typed-soac-scalar-result-type
           (reason-of #(lower/lower-typed-scalar
                       (scalar-program
                        {:result-value
                         (av/tensor {:dtype :double :shape [1] :memory-space :device
                                     :representation
                                     {:kind :resident-scalar-buffer :elements 1}})})
                       :test:0 :target-descriptor target-descriptor)))))
  (testing "a plain rank-zero result is not evidence of physical output storage"
    (is (= :typed-soac-device-scalar-subset
           (reason-of #(lower/lower-typed-scalar
                       (scalar-program
                        {:result-value (av/tensor {:dtype :double :shape []})})
                       :test:0 :target-descriptor target-descriptor)))))
  (testing "resident output storage is exact, not inferred from a representation name"
    (doseq [value [(av/tensor {:dtype :double :shape []
                               :representation
                               {:kind :resident-scalar-buffer :elements 1}})
                   (av/tensor {:dtype :double :shape [] :memory-space :device
                               :representation
                               {:kind :resident-scalar-buffer :elements 2}})]]
      (is (= :typed-soac-device-scalar-subset
             (reason-of #(lower/lower-typed-scalar
                          (scalar-program {:result-value value})
                          :test:0 :target-descriptor target-descriptor))))))
  (testing "a custom tensor representation is not treated as a raw pointer"
    (is (= :typed-soac-device-scalar-subset
           (reason-of #(lower/lower-typed-scalar
                       (scalar-program
                        {:input-value (av/tensor {:dtype :float :shape [1]
                                                 :representation {:kind :packed}})})
                       :test:0 :target-descriptor target-descriptor)))))
  (testing "rank zero with nondefault ownership is not silently physicalized"
    (is (= :typed-soac-device-scalar-subset
           (reason-of #(lower/lower-typed-scalar
                       (scalar-program
                        {:result-value (av/tensor {:dtype :double :shape []
                                                  :memory-space :device
                                                  :representation
                                                  {:kind :resident-scalar-buffer :elements 1}
                                                  :ownership :owned})})
                       :test:0 :target-descriptor target-descriptor)))))
  (testing "a resident rank-zero capture is not passed through the scalar ABI"
    (is (= :typed-soac-device-scalar-subset
           (reason-of #(lower/lower-typed-scalar
                       (scalar-program {:input-value (resident-scalar :float)})
                       :test:0 :target-descriptor target-descriptor)))))
  (testing "equation effects and aliases cannot be hidden by device scheduling"
    (doseq [facts [(assoc (dialect/default-equation-facts) :effects #{:memory/read})
                   (assoc (dialect/default-equation-facts) :aliases {'loss 'out})]]
      (is (= :typed-soac-device-scalar-subset
             (reason-of #(lower/lower-typed-scalar
                          (scalar-program {:equation-facts facts})
                          :test:0 :target-descriptor target-descriptor)))))))

(deftest scalar-scheduling-does-not-discard-value-or-equation-facets
  (doseq [facets [{:memory-space :host} {:ownership :borrowed}
                  {:effects #{:memory/read}} {:placement {:device :other}}
                  {:sharding {:axis 0}} {:attributes {:semantic-policy :custom}}]]
    (is (= :typed-soac-device-scalar-subset
           (reason-of #(lower/lower-typed-scalar
                        (scalar-program {:bias-value (av/tensor (merge {:dtype :double :shape []}
                                                                      facets))})
                        :test:0 :target-descriptor target-descriptor)))))
  (doseq [facets [{:memory-space :host} {:ownership :borrowed}
                  {:effects #{:memory/read}} {:placement {:device :other}}
                  {:sharding {:axis 0}} {:attributes {:semantic-policy :custom}}]]
    (is (= :typed-soac-device-scalar-subset
           (reason-of #(lower/lower-typed-scalar
                        (scalar-program {:input-value (av/tensor (merge {:dtype :float :shape [1]}
                                                                       facets))})
                        :test:0 :target-descriptor target-descriptor)))))
  (doseq [attributes [{:dtypes [:double] :attributes {:semantic-policy :custom}}
                      {:dtypes [:double] :unexpected-policy true}]]
    (is (= :typed-soac-device-scalar-subset
           (reason-of #(lower/lower-typed-scalar
                        (scalar-program {:operation-attributes attributes})
                        :test:0 :target-descriptor target-descriptor)))))
  (is (= :typed-soac-device-scalar-subset
         (reason-of #(lower/lower-typed-scalar
                      (scalar-program {:equation-facts
                                       (assoc (dialect/default-equation-facts)
                                              :attributes {:host-binding 'hidden})})
                      :test:0 :target-descriptor target-descriptor))))
  (is (segop/seg-map?
       (first (lower/lower-typed-scalar
               (scalar-program {:input-value (av/tensor {:dtype :float :shape [1]
                                                         :memory-space :device})})
               :test:0 :target-descriptor target-descriptor)))
      "an explicitly device-located plain tensor is admitted without changing its other facets"))
