(ns raster.compiler.ir.invocation-plan-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.typed-scalar :as typed-scalar]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]))

(defn- scalar [dtype]
  (av/tensor {:dtype dtype :shape []}))

(deftest wrapping-casts-are-not-shape-identities
  (is (= 'xs (#'invocation/shape-source '(clojure.core/long (alength xs)))))
  (is (nil? (#'invocation/shape-source '(clojure.core/unchecked-int (alength xs))))))

(defn- array-value [dtype extent]
  (av/tensor {:dtype dtype :shape [extent] :representation {:kind :plain}}))

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception
         (:reason (ex-data exception)))))

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
