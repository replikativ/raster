(ns raster.compiler.passes.parallel.device-scalar-activation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.write-coverage :as coverage]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def ^:private target-descriptor
  {:subgroup-size 16 :max-workgroup-size 256 :max-warps-per-sm 16
   :max-blocks-per-sm 8 :sm-count 8 :machine-lanes 2048})

(def ^:private terminal-source
  '(let* [out (clojure.core/float-array 1)
          effect (raster.par/map-void!
                  i 1
                  (clojure.core/aset out i
                                     (float (clojure.core/aget input i))))]
     (clojure.core/double (clojure.core/aget out 0))))

(defn- route-source [source array-types]
  (route/attempt source :float array-types
                 {:segmented-plans? true :resident-reductions? true}))

(defn- scalar-equation [program]
  (some #(when (and (dialect/program-form? (:algorithm %))
                    (= 'scalar
                       (dialect/operation-kind
                        (first (dialect/equations (:algorithm %))))))
           %)
        (:equations program)))

(deftest terminal-naming-reuses-frontend-authority-and-fails-closed
  (let [{:keys [form result description]}
        (frontend/materialize-terminal-scalar-result
         terminal-source {:dtype :float :array-types {'input :float}})]
    (is (= :scalar (:kind description)))
    (is (= :double (:scalar-dtype description)))
    (is (= 'double (:raster.type/tag (meta result))))
    (is (= (last terminal-source) (second (last (partition 2 (second form))))))
    (is (= result (last form)))
    (testing "an input read has no preceding typed writer"
      (is (nil? (frontend/materialize-terminal-scalar-result
                 (list 'let* (second terminal-source)
                       '(clojure.core/double (clojure.core/aget input 0)))
                 {:dtype :float :array-types {'input :float}}))))
    (testing "lexical call heads cannot manufacture cast authority"
      (is (nil? (frontend/materialize-terminal-scalar-result
                 (list 'let* (into ['double (float 1.0)] (second terminal-source))
                       '(double (clojure.core/aget out 0)))
                 {:dtype :float :array-types {'input :float}}))))
    (testing "an unknown terminal call remains outside source normalization"
      (is (nil? (frontend/materialize-terminal-scalar-result
                 (list 'let* (second terminal-source)
                       '(unknown-scalar (clojure.core/aget out 0)))
                 {:dtype :float :array-types {'input :float}}))))))

(deftest direct-terminal-scalar-is-realized-and-scheduled-once
  (let [{:keys [program stats declined]} (route-source terminal-source {'input :float})
        result (first (:outputs program))
        value (get-in program [:values result])
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program
                    {:target-device :test:0 :target-descriptor target-descriptor
                     :dtype :float :array-types {'out :float}
                     :scalar-types {}}))
        scheduled-equation (last (:equations scheduled))
        operation (first (:operations scheduled-equation))
        pairs (vec (partition 2 (second (:source program))))
        result-pair (some #(when (= result (first %)) %) pairs)
        scalar-store (some (fn [[_ expression]]
                             (when (and (seq? expression)
                                        (= 'clojure.core/aset (first expression))
                                        (= result (second expression)))
                               expression))
                           pairs)
        terminal-expression (last terminal-source)
        terminal-occurrences (count (filter #(= terminal-expression %)
                                            (tree-seq coll? seq (:source program))))
        execute (eval (list 'fn '[input]
                            (par-ir/expand-par-forms (:source program))))]
    (is (nil? declined))
    (is (= 1 (:terminal-scalars-materialized stats)))
    (is (= 1 (:resident-terminal-scalars stats)))
    (is (= [] (:shape value)))
    (is (= {:kind :resident-scalar-buffer :elements 1} (:representation value)))
    (is (= :device (:memory-space value)))
    (is (= 1 (count (:operations scheduled-equation))))
    (is (segop/seg-map? operation))
    (is (not (true? (get-in scheduled-equation [:attributes :host-only]))))
    (is (= 1 (get-in operation [:space :dims 0 :bound])))
    (is (= #{'out} (:inputs operation)))
    (is (= #{result} (:outputs operation)))
    (is (= {:source-prefix? false} (:algorithm-equation-attributes operation))
        "the schedule carries the exact validated source-order attribute")
    (is (= 'clojure.core/double-array (first (second result-pair))))
    (is (= 1 (second (second result-pair))))
    (is scalar-store)
    (is (= 1 terminal-occurrences) "the terminal conversion is evaluated exactly once")
    (is (= (list 'clojure.core/aget result 0) (last (:source program))))
    (is (= (double (float 1.234567890123))
           (execute (float-array [1.234567890123]))))
    (is (= [result]
           (:outputs (#'structured-route/validate-source-result!
                      program {:source-ns *ns* :public-parameters '[input]
                               :active-params '[input] :array-types {'input :float}
                               :return-tag 'double}))))))

(deftest scalar-complete-write-domain-requires-resident-storage-and-exact-capacity
  (let [program (:program (route-source terminal-source {'input :float}))
        equation (scalar-equation program)
        algorithm (:algorithm equation)
        result (first (:outputs program))
        inner-equation (first (dialect/equations algorithm))
        facts (dialect/facts algorithm)]
    (is (= [] (dialect/dense-functional-result-shape algorithm inner-equation result)))
    (is (coverage/dense-result-covers? algorithm result 1 identity))
    (doseq [capacity [0 2]]
      (is (not (coverage/dense-result-covers? algorithm result capacity identity))))
    (doseq [[facet value] [[:representation {:kind :plain}]
                          [:representation {:kind :resident-scalar-buffer :elements 2}]
                          [:memory-space nil]
                          [:logical-layout {:kind :strided}]
                          [:placement :host] [:sharding {:parts 2}] [:ownership :borrowed]
                          [:effects #{:memory/read}] [:attributes {:opaque true}]]]
      (is (nil? (dialect/dense-functional-result-shape
                 (assoc-in facts [:values result facet] value) inner-equation result))))))

(defn- named-terminal-program []
  (frontend/form->program
   (:form (frontend/materialize-terminal-scalar-result
           terminal-source {:dtype :float :array-types {'input :float}}))
   {:dtype :float :array-types {'input :float}}))

(deftest terminal-activation-requires-dominated-unobserved-unique-storage
  (let [program (named-terminal-program)
        facts (dialect/facts program)
        equations (dialect/equations program)
        [writer scalar] equations
        result (first (dialect/outputs program))
        rebuild #(dialect/make % equations [result])]
    (is (resident/terminal-device-scalar-projection program))
    (testing "prior host observation is not a device scalar continuation"
      (is (nil? (resident/terminal-device-scalar-projection
                 (rebuild (assoc-in facts [:attributes :host-read-values] ['out]))))))
    (testing "unsupported ScalarIR attributes remain unsupported, not compiler crashes"
      (is (nil? (resident/terminal-device-scalar-projection
                 (rebuild (assoc-in facts [:equations (second scalar) :attributes :opaque] true))))))
    (testing "a source allocation must dominate its owning writer"
      (is (nil? (resident/terminal-device-scalar-projection
                 (rebuild (assoc-in facts [:attributes :allocations 0 :source-binding-id] 2))))))
    (testing "an independent public shape alone is insufficient with a short allocator"
      (is (= :typed-soac-initialization-contract
             (try (resident/terminal-device-scalar-projection
                   (rebuild (-> facts
                                (assoc-in [:values 'out :shape] [1])
                                (assoc-in [:attributes :allocations 0 :extent] 0))))
                  nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (testing "every physical writer participates, not just the latest one"
      (let [logical [:second-writer 3]
            second-writer (list '= 3 [logical] (nth writer 3))
            writer-facts (get-in facts [:equations (second writer)])
            duplicate-facts (-> facts
                                (assoc-in [:values logical] (get-in facts [:values (first (nth writer 2))]))
                                (assoc-in [:equations 3] (-> writer-facts
                                                             (assoc :aliases {logical 'out})
                                                             (assoc-in [:provenance :source-binding-id] 1))))]
        (is (nil? (resident/terminal-device-scalar-projection
                   (dialect/make duplicate-facts [writer second-writer scalar] [result]))))))
    (testing "a future writer does not initialize an earlier scalar read"
      (is (nil? (resident/terminal-device-scalar-projection
                 (dialect/make facts [scalar writer] [result])))))))

(deftest isolated-resident-marker-cannot-activate-production-scheduling
  (let [marked (resident/terminal-device-scalar-projection (named-terminal-program))
        result (first (dialect/outputs marked))
        scalar (last (dialect/equations marked))
        isolated (route/program-envelope (dialect/equation-subprogram marked scalar))]
    (is (nil? (resident/validated-terminal-resident-output isolated result)))
    (is (nil? (#'structured-route/validated-public-resident-output isolated result)))
    (is (= :typed-soac-device-scalar-activation
           (try (segop-lower/segop-lower-pass isolated {:target-device :test:0})
                nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest enclosing-execution-spine-must-match-its-retained-algorithms
  (let [program (:program (route-source terminal-source {'input :float}))
        result (first (:outputs program))]
    (is (resident/validated-terminal-resident-output program result))
    (doseq [changed [(assoc-in program [:equations 0 :effects] #{})
                     (assoc-in program [:equations 0 :operations] [])
                     (assoc-in program [:equations 1 :operations] [])]]
      (is (nil? (resident/validated-terminal-resident-output changed result))))))

(deftest outer-shape-refinement-does-not-manufacture-singleton-capacity
  (let [program (:program (route-source terminal-source {'input :float}))
        result (first (:outputs program))
        forged (-> program
                   (assoc-in [:values 'out :shape] [1])
                   (assoc-in [:attributes :allocations] []))]
    (is (nil? (resident/validated-terminal-resident-output forged result)))))

(deftest unsupported-terminal-captures-restore-the-source-return
  (doseq [[source array-types]
          [[(list 'let* (second terminal-source)
                  '(clojure.core/double (clojure.core/aget out 1))) {'input :float}]
           [(list 'let* (second terminal-source)
                  '(clojure.core/+ (clojure.core/double (clojure.core/aget out 0))
                                  (clojure.core/double (clojure.core/aget extra 0))))
            {'input :float 'extra :float}]]]
    (let [routed (route-source source array-types)]
      (is (nil? (:declined routed)))
      (is (= 0 (get-in routed [:stats :terminal-scalars-materialized])))
      (is (= (last source) (last (get-in routed [:program :source])))))))

(deftest lost-post-optimization-residency-restores-original-return
  (with-redefs [resident/realize (fn [program]
                                 [program {:resident-reductions 0 :inlined-scalars 0
                                           :resident-terminal-scalars 0}])]
    (let [routed (route-source terminal-source {'input :float})]
      (is (nil? (:declined routed)))
      (is (= 0 (get-in routed [:stats :terminal-scalars-materialized])))
      (is (= (last terminal-source) (last (get-in routed [:program :source]))))
      (is (not-any? #(and (symbol? %) (= "rstr_terminal_scalar_result" (name %)))
                    (tree-seq coll? seq (get-in routed [:program :source])))))))

(deftest unsupported-terminal-storage-keeps-the-host-scalar-boundary
  (testing "zero capacity is never promoted to a physical singleton"
    (let [source
          '(let* [out (clojure.core/float-array 0)
                  effect (raster.par/map-void!
                          i 0 (clojure.core/aset out i (float 1.0)))]
             (clojure.core/double (clojure.core/aget out 0)))
          {:keys [program stats]} (route-source source {})]
      (is (= 0 (:resident-terminal-scalars stats)))
      (is (= 0 (:terminal-scalars-materialized stats)))
      (is (= (last source) (last (:source program))))))
  (testing "a guarded partial writer is not complete initialization evidence"
    (let [source
          '(let* [out (clojure.core/float-array 1)
                  effect (raster.par/map-void!
                          i 1
                          (when (= i 0)
                            (clojure.core/aset out i (float 1.0))))]
             (clojure.core/double (clojure.core/aget out 0)))
          routed (route-source source {})]
      (is (or (:declined routed)
              (zero? (get-in routed [:stats :resident-terminal-scalars] 0))))))
  (testing "a checked narrowing remains on the original refusal path"
    (let [source
          '(let* [out (clojure.core/long-array 1)
                  effect (raster.par/map-void!
                          i 1
                          (clojure.core/aset out i
                                             (long (clojure.core/aget input i))))]
             (clojure.core/int (clojure.core/aget out 0)))
          routed (route-source source {'input :long})]
      (is (or (:declined routed)
              (zero? (get-in routed [:stats :resident-terminal-scalars] 0))))))
  (testing "a trapping operation nested under control remains host-owned"
    (let [source
          '(let* [out (clojure.core/long-array 1)
                  effect (raster.par/map-void!
                          i 1
                          (clojure.core/aset out i
                                             (long (clojure.core/aget input i))))]
             (if (clojure.core/pos? (clojure.core/aget out 0))
               (clojure.core/+ (clojure.core/aget out 0) 1)
               (long 0)))
          routed (route-source source {'input :long})]
      (is (or (:declined routed)
              (zero? (get-in routed [:stats :resident-terminal-scalars] 0)))))))

(deftest unexpected-strict-lowering-errors-are-not-converted-to-ineligibility
  (with-redefs [scalar-expression/make-lowerer
                (fn [& _] (throw (IllegalStateException. "injected strict-lowering fault")))]
    (is (thrown-with-msg? IllegalStateException #"injected strict-lowering fault"
                          (route-source terminal-source {'input :float})))))
