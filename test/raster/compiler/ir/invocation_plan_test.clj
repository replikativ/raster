(ns raster.compiler.ir.invocation-plan-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.typed-scalar :as typed-scalar]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.ir.parallel-program :as program]))

(defn- scalar [dtype]
  (av/tensor {:dtype dtype :shape []}))

(deftest wrapping-casts-are-not-shape-identities
  (is (= 'xs (invocation-source/shape-projection-source '(clojure.core/long (alength xs)))))
  (is (nil? (invocation-source/shape-projection-source '(clojure.core/unchecked-int (alength xs))))))

(defn- array-value [dtype extent]
  (av/tensor {:dtype dtype :shape [extent] :representation {:kind :plain}}))

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception
         (:reason (ex-data exception)))))

(defn- terminal-program [body]
  ;; Structural producer correspondence only, deliberately not an emitted
  ;; reduction/export certificate or permission to execute this source.
  (program/make
   {:dialect :typed-parallel
    :source (list 'let* '[total (sum x)] body)
    :inputs '[x] :outputs '[total]
    :values {'x (array-value :double 4) 'total (scalar :double)}
    :equations [(program/map->ProgramEquation
                 {:id :sum :site [:binding 'total] :source '(sum x)
                  :operands '[x] :results '[total] :operations '[(reduce)]
                  :effects #{} :provenance {} :attributes {}})]}))

(def terminal-context
  {:source-policy {:unchecked-math false}
   :source-namespace 'raster.compiler.ir.invocation-plan-test})

(def terminal-effects (atom 0))

(deftest terminal-correspondence-is-not-execution-or-producer-certification
  (let [body '(do (clojure.core/swap! raster.compiler.ir.invocation-plan-test/terminal-effects
                                     clojure.core/inc)
                  total)
        p (terminal-program body)
        before @terminal-effects
        contract (invocation/terminal-source-result p terminal-context)
        changed (assoc-in p [:equations 0 :source] '(different-computation x))]
    (is (= body (:body contract)))
    (is (= before @terminal-effects) "describing a region must not execute it")
    (is (= contract (invocation/validate-terminal-source-result! contract changed terminal-context))
        "producer slot correspondence alone does not attest what that producer computes")))

(deftest terminal-ports-are-ordered-by-producers-not-the-result-tree
  (let [p (-> (terminal-program '[other total])
              (assoc-in [:values 'other] (scalar :long))
              (update :equations conj
                      (program/map->ProgramEquation
                       {:id :count :site [:binding 'other] :source '(count x)
                        :operands '[x] :results '[other] :operations '[(reduce)]
                        :effects #{} :provenance {} :attributes {}}))
              (assoc :source '(let* [total (sum x) other (count x)] [other total])))
        contract (invocation/terminal-source-result p terminal-context)]
    (is (= '[total other] (mapv :value (:parameters contract))))
    (is (= '[other total] (:body contract)))
    (is (= [[:sum 0] [:count 0]]
           (mapv (juxt :producer-equation :result-position) (:parameters contract))))))

(deftest terminal-correspondence-retains-conditional-completion-and-producer-ports
  (let [body (list 'if '(clojure.core/nil? total) nil
                   (list 'let* '[projected (clojure.core/float total)]
                         (completion/complete-value completion/object-input-contract 'projected)))
        p (terminal-program body)
        contract (invocation/terminal-source-result p terminal-context)]
    (is (= body (:body contract)))
    (is (= [{:symbol 'total :value 'total :dtype :double
             :producer-equation :sum :result-position 0}]
           (:parameters contract)))
    (is (= [:return] (:site contract)))
    (is (= (:source-policy terminal-context) (:source-policy contract)))
    (is (identical? contract
                    (invocation/validate-terminal-source-result! contract p terminal-context)))
    (doseq [changed [(assoc contract :body '(clojure.core/unchecked-float total))
                     (assoc-in contract [:parameters 0 :producer-equation] :another)
                     (assoc-in contract [:parameters 0 :result-position] 1)
                     (assoc-in contract [:parameters 0 :dtype] :float)
                     (assoc contract :source-policy {:unchecked-math true})
                     (assoc contract :source-namespace 'another.namespace)]]
      (is (= :invocation-terminal-correspondence
             (reason-of #(invocation/validate-terminal-source-result! changed p terminal-context)))))))

(deftest terminal-correspondence-uses-lexical-not-flat-free-variable-analysis
  (doseq [body ['(let* [copy total total 1.0] [copy total])
               '(let* [projected (clojure.core/float total)] projected)
               '(if (clojure.core/nil? total) (quote not-a-port) total)]]
    (is (= '[total]
           (mapv :value (:parameters (invocation/terminal-source-result
                                     (terminal-program body) terminal-context))))))
  (let [p (-> (terminal-program '(clojure.core/float count))
              (assoc :source '(let* [count (sum x)] (clojure.core/float count)))
              (assoc :outputs '[count])
              (update :values #(-> % (dissoc 'total) (assoc 'count (scalar :double))))
              (assoc-in [:equations 0 :results] '[count])
              (assoc-in [:equations 0 :site] [:binding 'count]))]
    (is (= '[count] (mapv :value (:parameters (invocation/terminal-source-result p terminal-context)))))))

(deftest terminal-correspondence-does-not-acquire-unrepresented-values-or-exports
  (doseq [body ['(aget x 0) '(+ total private-capture) '(let* [total 1.0] total)]]
    (is (= :invocation-terminal-free-values
           (reason-of #(invocation/terminal-source-result (terminal-program body) terminal-context)))))
  (doseq [value [(array-value :double 1) (scalar :float) (scalar :int)]]
    (is (= :invocation-terminal-carrier
           (reason-of #(invocation/terminal-source-result
                        (assoc-in (terminal-program 'total) [:values 'total] value)
                        terminal-context)))))
  (is (= :unclosed-source-policy
         (reason-of #(invocation/terminal-source-result (terminal-program 'total)
                                                      (dissoc terminal-context :source-policy)))))
  (is (= :invocation-terminal-source
         (reason-of #(invocation/terminal-source-result
                      (assoc (terminal-program 'total) :source '(do total)) terminal-context))))
  (is (= :invocation-terminal-source
         (reason-of #(invocation/terminal-source-result
                      (assoc (terminal-program 'total) :source '(let* [total (sum x)] total nil))
                      terminal-context)))))

(deftest public-shadowing-is-explicit-ssa
  (let [public-array (array-value :double '(extent u0))
        internal-array (array-value :double 'n)
        values {'u0 public-array 'nsteps (scalar :long)}
        binding-values {'n (scalar :int)
                        'u internal-array
                        'scratch internal-array
                        'nsteps (scalar :int)}
        program-values {'n (scalar :int) 'u internal-array 'scratch internal-array
                        'nsteps (scalar :int) 'result internal-array}
        plan (invocation/from-prefix
              {:id :test
               :parameters '[u0 nsteps]
               :parameter-values values
               :bindings '[[n (int (alength u0))]
                           [u (aclone u0)]
                           [scratch (double-array n)]
                           [nsteps (int nsteps)]]
               :binding-values binding-values
               :program-values program-values
               :program-inputs '[nsteps n u scratch]
               :program-outputs '[result]})
        public-nsteps (:id (second (:parameters plan)))
        narrowed (last (:steps plan))]
    (is (invocation/invocation-plan? plan))
    (is (invocation/shape-projection? (first (:steps plan))))
    (is (invocation/buffer-clone? (second (:steps plan))))
    (is (invocation/buffer-allocation? (nth (:steps plan) 2)))
    (is (invocation/scalar-compute? narrowed))
    (is (= '(lambda [nsteps] (region [] [(int nsteps)])) (:region narrowed)))
    (is (not (contains? narrowed :expression)))
    (is (not= public-nsteps (:id narrowed)))
    (is (= public-nsteps (-> narrowed :operands first :value)))
    (is (= (:id narrowed) (get-in plan [:bindings 0 :invocation-value])))))

(deftest impure-prefix-scalar-is-rejected
  (is (= :invocation-scalar-effect
         (reason-of
          #(invocation/from-prefix
            {:id :impure
             :parameters '[x]
             :parameter-values {'x (scalar :double)}
             :bindings '[[y (example.effects/mystery-effect! x)]]
             :binding-values {'y (scalar :double)}
             :program-values {'y (scalar :double)}
             :program-inputs '[y]
             :program-outputs '[y]})))))

(deftest core-named-public-scalars-remain-closed-operands
  (doseq [dimension '[seq count first]]
    (let [plan (invocation/from-prefix
                {:id [:core-local dimension]
                 :parameters [dimension]
                 :parameter-values {dimension (scalar :long)}
                 :bindings [['n (list 'clojure.core/long dimension)]]
                 :binding-values {'n (scalar :long)}
                 :program-values {'n (scalar :long)}
                 :program-inputs '[n] :program-outputs '[n]})
          step (first (:steps plan))]
      (is (= [dimension] (mapv :symbol (:operands step))))
      (is (= {:type :long :value 4}
             (typed-scalar/evaluate-invocation-step
              *ns* step {dimension {:type :long :value 4}})))
      ;; A corrupt closed region must not pass verification just because the missing local
      ;; shares a name with clojure.core. Its declared public lexical boundary remains known.
      (is (= :invocation-scalar-free-symbol
             (reason-of #(invocation/validate!
                          (assoc plan :steps
                                 [(assoc step :operands []
                                         :region (list 'lambda []
                                                       (list 'region []
                                                             [(list 'clojure.core/long dimension)])))]))))))))

(deftest scalar-compute-is-a-closed-rank-zero-region
  (is (= :invocation-scalar-operands
         (reason-of
          #(invocation/from-prefix
            {:id :buffer-scalar
             :parameters '[x]
             :parameter-values {'x (array-value :double 4)}
             :bindings '[[y (int x)]]
             :binding-values {'y (scalar :int)}
             :program-values {'y (scalar :int)}
             :program-inputs '[y]
             :program-outputs '[y]}))))
  (let [plan (invocation/from-prefix
              {:id :closed-scalar
               :parameters '[x]
               :parameter-values {'x (scalar :long)}
               :bindings '[[y (int x)]]
               :binding-values {'y (scalar :int)}
               :program-values {'y (scalar :int)}
               :program-inputs '[y]
               :program-outputs '[y]})
        open-step (assoc (first (:steps plan))
                         :region '(lambda [x] (region [] [(+ x rogue)])))
        open-plan (assoc plan :steps [open-step])]
    (is (= :invocation-scalar-free-symbol
           (reason-of #(invocation/validate! open-plan))))))
