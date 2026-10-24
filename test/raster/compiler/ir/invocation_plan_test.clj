(ns raster.compiler.ir.invocation-plan-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.typed-scalar :as typed-scalar]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.passes.parallel.resident-program-projection :as resident-projection]))

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

(deftest normalized-reference-identities-retain-their-exact-source
  (let [binder (with-meta 'saved {:tag 'java.lang.Object :raster.type/tag 'doubles})
        expression (vary-meta (completion/complete-value completion/object-input-contract 'x)
                              assoc :raster.effect/preserve-evaluation true)
        value (array-value :double 4)
        options {:id :reference-identity :parameters '[x] :parameter-values {'x value}
                 :bindings [[binder expression]] :binding-values {binder value}
                 :program-values {binder value} :program-inputs [] :program-outputs []}
        plan (invocation/from-prefix options)
        step (first (:steps plan))
        source (list 'let* [binder expression] nil)
        parallel (program/make {:dialect :typed-parallel :source source :values {'x value binder value}})]
    (is (invocation/value-alias? step))
    (is (= expression (:source-expression step)))
    (is (= (meta expression) (meta (:source-expression step))))
    (is (= 'x (invocation-source/reference-identity-source (:symbol step) (:source-expression step))))
    (is (identical? plan (invocation/validate! plan)))
    (is (string? (semantic/fingerprint plan)) "normalized identities contain no process-local Class proof")
    (is (not= (semantic/fingerprint plan)
              (semantic/fingerprint (assoc plan :steps [(update step :source-expression
                                                                vary-meta dissoc :raster.effect/preserve-evaluation)]))))
    (is (= (:id step) (:invocation-step (first (invocation/source-order plan parallel))))
        "even a numerically unused identity retains its ordered invocation step")
    (is (= #{binder} (#'resident-projection/required-prefix-symbols plan [])))
    (is (= :resident-projection-retained-reference-identity
           (reason-of #(#'resident-projection/validate-descriptor-prefix! [step])))
        "the old descriptor adapter must not erase the ordered step")
    (let [input (double-array [1.0 2.0 3.0 4.0])
          initial (materialization/begin-materialization plan [input])
          public-id (:id (first (:parameters plan)))
          before (get-in initial [:values public-id])
          advanced (materialization/materialize-next initial
                                                     (fn [& _] (throw (AssertionError. "alias evaluated scalar code"))))]
      (is (= 0 (:next-step initial)))
      (is (= 1 (:next-step advanced)))
      (is (identical? before (get-in advanced [:values (:id step)])))
      (is (identical? input (:source (get-in advanced [:values (:id step)]))))
      (is (some? (materialization/finish-materialization advanced))))
    (doseq [changed-source [(list 'let* [] nil)
                           (list 'let* [binder (vary-meta expression dissoc :raster.effect/preserve-evaluation)] nil)
                           (list 'let* [(vary-meta binder assoc :tag 'String) expression] nil)]]
      (is (some? (reason-of #(invocation/source-order plan (assoc parallel :source changed-source))))
          "removing or changing the retained binding cannot replay as the original plan"))
    (doseq [[bad-id bad-expression]
            [[(vary-meta binder assoc :tag 'String) expression]
             [(vary-meta binder dissoc :tag) expression]
             [binder (vary-meta expression assoc :raster.effect/source-reference-cast true)]
             [binder (vary-meta expression assoc :raster.effect/preserve-evaluation :unknown)]
             [binder (completion/complete-value
                      (completion/bytecode-boxed-parameter-contract (.arrayType Double/TYPE)) 'x)]
             [binder (completion/complete-value completion/object-input-contract '(do x))]]]
      (is (= :invocation-prefix-unsupported
             (reason-of #(invocation/from-prefix (assoc options :bindings [[bad-id bad-expression]]))))))
    (is (= :invocation-prefix-free-value
           (reason-of #(invocation/from-prefix
                        (assoc options :bindings [[binder (completion/complete-value
                                                         completion/object-input-contract 'future)]])))))
    (is (= :invocation-value-alias-source
           (reason-of #(invocation/from-prefix (assoc options :parameter-values {'x (scalar :double)}))))
        "a primitive/scalar source is not a buffer reference identity")
    (is (= :invocation-value-alias-source
           (reason-of #(invocation/from-prefix
                        (assoc options :bindings [[(vary-meta binder assoc :raster.type/tag 'floats)
                                                   expression]])))))
    (is (= :invocation-value-alias-source
           (reason-of #(invocation/validate!
                        (assoc plan :steps [(assoc step :source-expression
                                                  (completion/complete-value completion/object-input-contract 'other))])))))))

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
