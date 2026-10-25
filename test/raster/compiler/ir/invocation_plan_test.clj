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

(defn- result-import-fixture []
  (let [snapshot (completion/complete-value completion/object-input-contract 'total)
        reduction '(numerical/reduce xs)
        values {'xs (array-value :double 4) 'total (scalar :double)}
        opaque (av/make {:kind :opaque :memory-space :host :ownership :borrowed})
        equation (program/map->ProgramEquation
                  {:id :reduce :site [:binding 'total] :source reduction
                   :operands '[xs] :results '[total] :operations [{}]
                   :effects #{} :provenance {} :attributes {}})
        parallel (program/make {:dialect :typed-parallel
                                :source (list 'let* ['total reduction 'saved snapshot] 'total)
                                :values values :inputs '[xs] :outputs '[total] :equations [equation]})
        options {:id :result-import :parameters '[xs] :parameter-values (select-keys values '[xs])
                 :bindings [['saved snapshot]] :binding-values {'saved opaque}
                 :result-imports {'saved {:equation :reduce :result 'total}}
                 :program-values values :program-inputs '[xs] :program-outputs '[total]}]
    {:options options :plan (invocation/from-prefix options) :program parallel
     :snapshot snapshot :equation equation :opaque opaque}))

(deftest program-result-import-is-an-explicit-source-ordered-opaque-demand
  (let [{:keys [plan program snapshot]} (result-import-fixture)
        step (first (:steps plan))
        order (invocation/source-order plan program)]
    (is (invocation/program-result-import? step))
    (is (invocation/invocation-step? step))
    (is (= [] (:operands step)))
    (is (= {:equation :reduce :result 'total} (select-keys step [:equation :result])))
    (is (= snapshot (:source-expression step)))
    (is (= :opaque (get-in step [:value :kind])))
    (is (= :host (get-in step [:value :memory-space])))
    (is (nil? (get-in step [:value :dtype])))
    (is (identical? plan (invocation/validate-against! plan program)))
    (is (= :reduce (:equation (first order))))
    (is (= {:equation :reduce :result 'total} (:program-result-import (second order))))
    (is (= (:id step) (:invocation-step (second order))))))

(deftest result-imports-cannot-guess-or-relabel-a-source-producer
  (let [{:keys [plan program options equation snapshot opaque]} (result-import-fixture)
        step (first (:steps plan))
        check-both (fn [plan program]
                     (is (some? (reason-of #(invocation/validate-against! plan program))))
                     (is (some? (reason-of #(invocation/source-order plan program)))))]
    (doseq [changed [(assoc step :equation :wrong)
                     (assoc step :result 'xs)
                     (assoc step :operands [{:symbol 'xs :value (:id (first (:parameters plan)))}])
                     (assoc step :value (assoc opaque :dtype :double))]]
      (check-both (assoc plan :steps [changed]) program))
    (doseq [source [(list 'let* ['saved snapshot 'total (:source equation)] 'total)
                    (list 'let* ['total '(numerical/other xs) 'saved snapshot] 'total)
                    (list 'let* ['total (:source equation) 'saved
                                (completion/complete-value completion/object-input-contract '(aget xs 0))] 'total)
                    (list 'let* ['total (:source equation) 'saved 'total] 'total)
                    nil]]
      (check-both plan (assoc program :source source)))
    (check-both plan (update program :equations conj (assoc equation :id :duplicate)))
    (check-both plan (assoc program :equations []))
    (check-both plan (assoc program :equations [(assoc equation :operations [] :attributes {:host-only true})]))
    (check-both (assoc-in plan [:values 'total] (scalar :float)) program)
    (let [source-free (program/make
                       {:dialect :typed-parallel :source nil :values (:values program)
                        :inputs '[xs] :outputs '[xs] :equations []
                        :provenance {:front-end :explicit-typed-algorithm :pass :source-free-typed-invocation}
                        :attributes {:host-control :typed-invocation}})
          source-free-plan (assoc plan :program-outputs '[xs]
                                 :attributes {:source-dialect :typed-soac :algorithm-dialect :typed-soac
                                              :target-dialect :typed-invocation})]
      (is (invocation/source-free-program? source-free))
      (check-both source-free-plan source-free))
    (is (= :invocation-prefix-unsupported
           (reason-of #(invocation/from-prefix (dissoc options :result-imports)))))
    (is (= :invocation-program-result-import
           (reason-of #(invocation/from-prefix
                        (assoc options :result-imports {'unknown {:equation :reduce :result 'total}})))))
    (doseq [declaration [nil false {:equation :reduce} {:equation :reduce :result 'total :extra true}]]
      (is (= :invocation-program-result-import
             (reason-of #(invocation/from-prefix (assoc options :result-imports {'saved declaration}))))))
    (is (= :invocation-program-result-import
           (reason-of #(invocation/from-prefix
                        (assoc options :bindings [['saved (completion/complete-value
                                                          completion/object-input-contract '(aget xs 0))]])))))
    (is (= :invocation-opaque-program-value
           (reason-of #(invocation/from-prefix
                        (assoc options :program-values (assoc (:program-values options) 'saved opaque)
                               :program-inputs '[xs saved])))))))

(deftest result-import-keeps-its-producer-across-an-intervening-numerical-stage
  (let [{:keys [plan program snapshot]} (result-import-fixture)
        reduction (:source (first (:equations program)))
        other-source '(numerical/other xs)
        other (program/map->ProgramEquation
               {:id :other :site [:binding 'other] :source other-source
                :operands '[xs] :results '[other] :operations [{}]
                :effects #{} :provenance {} :attributes {}})
        changed (-> program
                    (assoc :source (list 'let* ['total reduction 'other other-source 'saved snapshot] 'total))
                    (assoc-in [:values 'other] (scalar :double))
                    (update :equations conj other))
        order (invocation/source-order plan changed)]
    (is (= [:reduce :other nil nil] (mapv :equation order)))
    (is (= {:equation :reduce :result 'total} (:program-result-import (nth order 2))))
    (is (nil? (:program-result-import (nth order 1))))))

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
