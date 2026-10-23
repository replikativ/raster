(ns raster.compiler.ir.invocation-materialization-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- scalar [dtype] (av/tensor {:dtype dtype :shape []}))
(defn- array-value [dtype extent]
  (av/tensor {:dtype dtype :shape [extent] :representation {:kind :plain}}))

(deftest source-alias-check-retains-every-distinct-storage-in-a-read-only-group
  (let [source (double-array 1)
        a {:source source :storage :a :mutable? false}
        b {:source source :storage :b :mutable? false}
        mutable-a (assoc a :mutable? true)]
    (is (nil? (materialization/conflicting-source-alias [a b])))
    (is (nil? (materialization/conflicting-source-alias [a mutable-a])))
    (is (nil? (materialization/conflicting-source-alias
               [mutable-a (assoc b :source (double-array 1))])))
    (is (= [mutable-a b] (materialization/conflicting-source-alias [mutable-a b])))
    (is (= [b mutable-a] (materialization/conflicting-source-alias [a b mutable-a])))
    (is (= [mutable-a b] (materialization/conflicting-source-alias [a mutable-a b])))))

(deftest integral-scalar-admission-is-not-an-implicit-conversion
  (doseq [[dtype values] [[:byte [Byte/MIN_VALUE 0 Byte/MAX_VALUE]]
                         [:int [Integer/MIN_VALUE 0 Integer/MAX_VALUE]]
                         [:long [Long/MIN_VALUE 0 Long/MAX_VALUE]]]
          value values
          argument [value {:type dtype :value value}]]
    (is (= {:type dtype :value value}
           (materialization/checked-scalar-value 'n (scalar dtype) argument))))
  (doseq [[dtype values] [[:byte [-129 128 1.5 Double/NaN Double/POSITIVE_INFINITY]]
                         [:int [-2147483649 2147483648 1.5]]
                         [:long [-9223372036854775809N 9223372036854775808N 1.5]]]
          value values
          argument [value {:type dtype :value value}]]
    (is (= :invocation-materialization-scalar
           (try (materialization/checked-scalar-value 'n (scalar dtype) argument)
                nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (doseq [value [Double/NEGATIVE_INFINITY Double/POSITIVE_INFINITY Double/NaN]]
    (let [actual (materialization/checked-scalar-value 'x (scalar :double) value)]
      (is (= :double (:type actual)))
      (is (= (Double/doubleToRawLongBits value)
             (Double/doubleToRawLongBits (:value actual)))))))

(deftest invalid-integral-parameter-refuses-before-the-scalar-evaluator
  (let [plan (invocation/from-prefix
              {:id :checked-integral-entry
               :parameters '[n] :parameter-values {'n (scalar :long)}
               :bindings '[[m (inc n)]] :binding-values {'m (scalar :long)}
               :program-values {'m (scalar :long)}
               :program-inputs '[m] :program-outputs '[m]})
        calls (atom 0)
        evaluate (fn [_ _] (swap! calls inc) {:type :long :value 2})]
    (doseq [value [1.5 9223372036854775808N {:type :long :value 1.5}]]
      (is (= :invocation-materialization-scalar
             (try (materialization/materialize plan [value] evaluate)
                  nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (zero? @calls))
    (let [result (materialization/materialize plan [1] evaluate)
          parameter-id (:id (first (:parameters plan)))]
      (is (= {:type :long :value 2} (get-in result [:program-scalars 'm])))
      (doseq [value [1.5 9223372036854775808N]]
        (is (= :invocation-materialization-scalar
               (try (materialization/validate!
                     (assoc-in result [:values parameter-id :value] value))
                    nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
            "a retained parameter's domain still holds when it is not a program input")))
    (is (= 1 @calls))))

(defn- fixture-plan []
  (let [public-array (array-value :double '(extent u0))
        internal-array (array-value :double 'n)]
    (invocation/from-prefix
     {:id :materialization-test
      :parameters '[u0 dt nsteps]
      :parameter-values {'u0 public-array 'dt (scalar :double) 'nsteps (scalar :long)}
      :bindings '[[n (int (alength u0))]
                  [u (aclone u0)]
                  [scratch (double-array n)]
                  [half-dt (* 0.5 dt)]
                  [nsteps (int nsteps)]]
      :binding-values {'n (scalar :int) 'u internal-array 'scratch internal-array
                       'half-dt (scalar :double) 'nsteps (scalar :int)}
      :program-values {'n (scalar :int) 'u internal-array 'scratch internal-array
                       'half-dt (scalar :double) 'nsteps (scalar :int)
                       'result internal-array}
      :program-inputs '[nsteps n half-dt u scratch]
      :program-outputs '[result]})))

(defn- evaluate-scalar [step operands]
  (let [{:keys [body-results]} (soac/lambda-parts (:region step))
        expression (first body-results)
        value
        (case (first expression)
          int (int (:value (get operands (second expression))))
          inc (inc (:value (get operands (second expression))))
          * (* (:value (get operands (nth expression 2))) (double (second expression))))]
    {:type (get-in step [:value :dtype]) :value value}))

(defn- materialization-reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- deferred-parameter-plan []
  (let [left (array-value :double '(extent left))
        right (array-value :double '(extent right))]
    (invocation/from-prefix
     {:id :deferred-parameter-state :parameters '[left n right]
      :parameter-values {'left left 'n (scalar :long) 'right right}
      :bindings [] :binding-values {}
      :program-values {'left left 'n (scalar :long) 'right right}
      :program-inputs '[left n right] :program-outputs '[left]})))

(deftest deferred-parameters-preserve-ordinals-and-borrowed-reference-identity
  (let [plan (deferred-parameter-plan)
        [left-id n-id right-id] (mapv :id (:parameters plan))]
    (doseq [left [(double-array [1.0 2.0]) (double-array [4.0])]]
      (let [right (double-array [3.0])
            arguments [left 7 right]
            initial (materialization/begin-materialization plan arguments #{left-id right-id})
            partial (materialization/materialize-demanded-parameters initial #{right-id})
            realized (materialization/materialize-demanded-parameters partial #{left-id})
            eager (materialization/begin-materialization plan arguments)]
        (is (= {n-id {:type :long :value 7}} (:values initial)))
        (is (= {'left left-id 'n n-id 'right right-id} (:lexical initial)))
        (is (= 0 (get-in initial [::materialization/deferred-parameters left-id :parameter-position])))
        (is (= 2 (get-in initial [::materialization/deferred-parameters right-id :parameter-position])))
        (is (identical? left (get-in initial [::materialization/deferred-parameters left-id :source])))
        (is (identical? right (get-in initial [::materialization/deferred-parameters right-id :source])))
        (is (empty? (:program-buffers (materialization/prefix-bindings initial))))
        (is (= #{'right} (set (keys (:program-buffers (materialization/prefix-bindings partial))))))
        (is (not (contains? (:values initial) left-id)) "no nil or dummy buffer placeholder")
        (is (= :invocation-materialization-pending-parameters
               (materialization-reason #(materialization/finish-materialization partial))))
        (is (= eager realized))
        (is (identical? left (get-in realized [:values left-id :source])))
        (is (= (materialization/finish-materialization eager)
               (materialization/finish-materialization realized)))
        (is (= realized (materialization/materialize-demanded-parameters realized #{left-id})))))))

(deftest deferred-reference-is-not-a-read-or-storage-proof
  (let [plan (deferred-parameter-plan)
        [left-id n-id right-id] (mapv :id (:parameters plan))
        right (double-array 1)
        initial (materialization/begin-materialization plan [nil 7 right] #{left-id})]
    (is (= :invocation-materialization-buffer
           (materialization-reason #(materialization/begin-materialization plan [nil 7 right]))))
    (is (identical? initial (materialization/validate-state! initial)))
    (is (nil? (get-in initial [::materialization/deferred-parameters left-id :source])))
    (is (not (contains? (:values initial) left-id)))
    (is (= :invocation-materialization-pending-parameters
           (materialization-reason #(materialization/finish-materialization initial))))
    (is (= :invocation-materialization-buffer
           (materialization-reason #(materialization/materialize-demanded-parameters initial #{left-id}))))
    (is (identical? right (get-in initial [:values right-id :source])))
    (let [redirected (assoc-in initial [::materialization/deferred-parameters left-id :source]
                              (double-array 2))]
      (is (identical? redirected (materialization/validate-state! redirected))
          "structural validation is not raw-reference ownership authentication")
      (is (= :invocation-materialization-pending-parameters
             (materialization-reason #(materialization/finish-materialization redirected)))))
    (doseq [selection [#{n-id} #{:unknown} [left-id] nil]]
      (is (= :invocation-materialization-deferred-parameters
             (materialization-reason #(materialization/begin-materialization plan [nil 7 right] selection))))
      (is (= :invocation-materialization-parameter-demand
             (materialization-reason #(materialization/materialize-demanded-parameters initial selection)))))
    (doseq [pending [nil []
                     {left-id {:source nil :parameter-position 2}}
                     {left-id {:source nil :parameter-position Long/MAX_VALUE}}
                     {left-id {:source nil :parameter-position 0 :claimed-read-proof true}}
                     {n-id {:source 7 :parameter-position 1}}]]
      (is (= :invocation-materialization-deferred-parameters
             (materialization-reason
              #(materialization/validate-state! (assoc initial ::materialization/deferred-parameters pending))))))
    (is (= :invocation-materialization-prefix
           (materialization-reason
            #(materialization/validate-state! (assoc-in initial [:values left-id] nil)))))))

(deftest deferred-input-cannot-feed-shape-clone-or-alias-steps
  (let [array (array-value :double '(extent x))]
    (doseq [[expression expected]
            [['(int (alength x)) :invocation-materialization-shape-source]
             ['(aclone x) :invocation-materialization-clone-source]
             ['x :invocation-materialization-step]]]
      (let [value (if (seq? expression)
                    (if (= 'int (first expression)) (scalar :int) array) array)
            plan (invocation/from-prefix
                  {:id :deferred-consumer :parameters '[x] :parameter-values {'x array}
                   :bindings [['result expression]] :binding-values {'result value}
                   :program-values {'result value} :program-inputs '[result] :program-outputs '[result]})
            id (:id (first (:parameters plan)))
            initial (materialization/begin-materialization plan [(double-array 3)] #{id})
            evaluations (atom 0)]
        (is (= expected (materialization-reason
                         #(materialization/materialize-next initial (fn [& _] (swap! evaluations inc))))))
        (is (zero? @evaluations))
        (let [ready (materialization/materialize-demanded-parameters initial #{id})]
          (is (some? (materialization/finish-materialization
                      (materialization/materialize-next ready nil)))))))))

(deftest demand-keeps-existing-dtype-shape-and-projection-validation
  (let [base (deferred-parameter-plan)
        id (:id (first (:parameters base)))
        projection {:public-parameters '[left n right]
                    :physical-parameters [{:symbol 'left :binding 'left}
                                          {:symbol 'n :binding 'n}
                                          {:symbol 'right :binding 'right}]}
        projected (assoc-in base [:attributes :parameter-projection] projection)]
    (is (some? (invocation/validate! projected)))
    (is (some? (materialization/begin-materialization projected [(double-array 2) 7 (double-array 1)])))
    (is (= :invocation-materialization-deferred-parameters
           (materialization-reason
            #(materialization/begin-materialization projected [nil 7 (double-array 1)] #{id}))))
    (let [pending (materialization/begin-materialization base [(float-array 2) 7 (double-array 1)] #{id})]
      (is (= :invocation-materialization-buffer-dtype
             (materialization-reason #(materialization/materialize-demanded-parameters pending #{id}))))))
  (let [array (array-value :double 3)
        plan (invocation/from-prefix
              {:id :deferred-static-shape :parameters '[x] :parameter-values {'x array}
               :bindings [] :binding-values {} :program-values {'x array}
               :program-inputs '[x] :program-outputs '[x]})
        id (:id (first (:parameters plan)))
        pending (materialization/begin-materialization plan [(double-array 2)] #{id})]
    (is (= :invocation-materialization-buffer-shape
           (materialization-reason #(materialization/materialize-demanded-parameters pending #{id}))))))

(deftest incremental-materialization-shares-the-eager-interpreter
  (let [plan (fixture-plan)
        arguments [(double-array [1.0 2.0 3.0]) 0.25 7]
        calls (atom [])
        evaluate (fn [step operands]
                   (swap! calls conj (:symbol step))
                   (evaluate-scalar step operands))
        initial (materialization/begin-materialization plan arguments)]
    (is (zero? (:next-step initial)))
    (is (= (set (map :id (:parameters plan))) (set (keys (:values initial)))))
    (is (empty? @calls))
    (is (= :invocation-materialization-incomplete
           (materialization-reason #(materialization/finish-materialization initial))))
    (let [complete (reduce (fn [state _] (materialization/materialize-next state evaluate))
                           initial (:steps plan))
          result (materialization/finish-materialization complete)]
      (is (= '[half-dt nsteps] @calls))
      (is (= result (materialization/materialize plan arguments evaluate-scalar)))
      (is (= :invocation-materialization-complete
             (materialization-reason #(materialization/materialize-next complete evaluate))))
      (is (= '[half-dt nsteps] @calls))
      (is (zero? (:next-step initial)) "an advance does not mutate an earlier state"))))

(defn- staged-invocation-plan []
  (invocation/from-prefix
   {:id :deferred-square
    :parameters '[n] :parameter-values {'n (scalar :long)}
    :bindings '[[prefix (double-array 1)] [square (* n n)]
                [out (double-array square)] [tail (inc square)]]
    :binding-values {'prefix (array-value :double 1) 'square (scalar :long)
                     'out (array-value :double 'square) 'tail (scalar :long)}
    :program-values {'prefix (array-value :double 1) 'square (scalar :long)
                     'out (array-value :double 'square) 'tail (scalar :long)}
    :program-inputs '[prefix square out tail] :program-outputs '[out]}))

(deftest incremental-check-failure-does-not-materialize-the-suffix
  (let [plan (staged-invocation-plan)]
    (doseq [n [3 Long/MAX_VALUE]]
      (let [events (atom [])
            initial (materialization/begin-materialization plan [n])
            prefix (materialization/materialize-next initial nil)
            evaluate (fn [step operands]
                       (swap! events conj (:symbol step))
                       {:type :long
                        :value (case (:symbol step)
                                 square (let [v (:value (get operands 'n))] (* v v))
                                 tail (inc (:value (get operands 'square))))})]
        (is (empty? @events) "prefix preparation has not evaluated the later check")
        ;; The caller's real graph completion will occupy this boundary. This
        ;; interpreter test does not claim to submit or await a device event.
        (swap! events conj :device-completed)
        (if (= n Long/MAX_VALUE)
          (do
            (is (thrown? ArithmeticException (materialization/materialize-next prefix evaluate)))
            (is (= [:device-completed 'square] @events))
            (is (= 1 (:next-step prefix)))
            (is (= (inc (count (:parameters plan))) (count (:values prefix))))
            (is (= :invocation-materialization-incomplete
                   (materialization-reason #(materialization/finish-materialization prefix)))))
          (let [square (materialization/materialize-next prefix evaluate)
                allocated (materialization/materialize-next square nil)
                complete (materialization/materialize-next allocated evaluate)
                result (materialization/finish-materialization complete)]
            (is (= [:device-completed 'square 'tail] @events))
            (is (= [9] (get-in result [:program-buffers 'out :shape])))
            (is (= {:type :long :value 9} (get-in result [:program-scalars 'square])))
            (is (= {:type :long :value 10} (get-in result [:program-scalars 'tail])))))))))

(deftest incremental-prefix-validation-does-not-run-the-evaluator
  (let [state (materialization/begin-materialization (fixture-plan)
                                                    [(double-array 3) 0.25 7])
        calls (atom 0)
        evaluate (fn [& _] (swap! calls inc))
        parameter (:id (last (:parameters (:plan state))))]
    (doseq [forged [(assoc state :next-step -1)
                    (assoc state :next-step 100)
                    (assoc state :lexical {})
                    (assoc-in state [:values :unavailable] {:type :long :value 1})
                    (assoc-in state [:values parameter] 7)
                    (assoc-in state [:values parameter] {:type :long :value 1.5})]]
      (is (some? (materialization-reason #(materialization/materialize-next forged evaluate)))))
    (is (zero? @calls))))

(deftest incremental-structured-prefix-values-retain-their-producers
  (let [plan (staged-invocation-plan)
        prefix (materialization/materialize-next
                (materialization/begin-materialization plan [3]) nil)
        id (:id (first (:steps plan)))
        calls (atom 0)
        evaluate (fn [& _] (swap! calls inc))]
    (doseq [[field value] [[:shape [2]] [:id :different-storage] [:initialization :unspecified]]]
      (is (= :invocation-materialization-state-producer
             (materialization-reason
              #(materialization/materialize-next (assoc-in prefix [:values id field] value)
                                                  evaluate)))))
    (is (zero? @calls)))
  (let [plan (fixture-plan)
        initial (materialization/begin-materialization plan [(double-array 3) 0.25 7])
        projected (materialization/materialize-next initial nil)
        id (:id (first (:steps plan)))]
    (is (= :invocation-materialization-state-producer
           (materialization-reason
            #(materialization/materialize-next (assoc-in projected [:values id :value] 4) nil)))))
  (let [array (array-value :double '(extent x))
        plan (invocation/from-prefix
              {:id :retained-alias :parameters '[x] :parameter-values {'x array}
               :bindings '[[alias x]] :binding-values {'alias array}
               :program-values {'alias array} :program-inputs '[alias] :program-outputs '[alias]})
        initial (materialization/begin-materialization plan [(double-array 3)])
        aliased (materialization/materialize-next initial nil)
        id (:id (first (:steps plan)))
        forged (assoc-in aliased [:values id :source] (double-array 3))]
    (is (= :invocation-materialization-state-producer
           (materialization-reason #(materialization/finish-materialization forged))))))

(deftest retained-buffers-cannot-change-the-plan-storage-contract
  (let [array (array-value :double '(extent x))
        plan (invocation/from-prefix
              {:id :retained-buffer-contract :parameters '[x] :parameter-values {'x array}
               :bindings [] :binding-values {} :program-values {'x array}
               :program-inputs '[x] :program-outputs '[x]})
        state (materialization/begin-materialization plan [(double-array 3)])
        id (:id (first (:parameters plan)))
        original (get-in state [:values id])
        different (assoc original :value (array-value :float '(extent x)) :source (float-array 3))
        complete (materialization/finish-materialization state)]
    (is (= :invocation-materialization-buffer-value
           (materialization-reason
            #(materialization/finish-materialization (assoc-in state [:values id] different)))))
    (is (= :invocation-materialization-buffer-value
           (materialization-reason
            #(materialization/validate! (-> complete
                                           (assoc-in [:values id] different)
                                           (assoc-in [:program-buffers 'x] different))))))))

(deftest checked-dimension-algebra-shares-graph-extent-semantics
  (let [make-plan
        (fn [dimension scalar-type]
          (invocation/from-prefix
           {:id :checked-dimension
            :parameters '[x n m]
            :parameter-values {'x (array-value :float '(extent x))
                               'n (scalar scalar-type) 'm (scalar :long)}
            :bindings [] :binding-values {}
            :program-values {'x (array-value :float dimension)
                             'n (scalar scalar-type) 'm (scalar :long)}
            :program-inputs '[x n m] :program-outputs '[x]}))
        plan (make-plan '(clojure.core/* n m) :long)]
    (is (= [6] (get-in (materialization/materialize plan [(float-array 6) 2 3] nil)
                       [:program-buffers 'x :shape])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"producer shape differs"
                          (materialization/materialize plan [(float-array 5) 2 3] nil)))
    (is (thrown? ArithmeticException
                 (materialization/materialize plan [(float-array 1) Long/MAX_VALUE 2] nil)))
    (doseq [[dimension scalar-type arguments]
            [['(clojure.core/* n m) :float [(float-array 6) 2.0 3]]
             ['(clojure.core/* missing m) :long [(float-array 6) 2 3]]
             ['(do n m) :long [(float-array 6) 2 3]]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (materialization/materialize (make-plan dimension scalar-type)
                                                arguments nil))))))

(deftest ordered-public-values-materialize-the-exact-program-boundary
  (let [source (double-array [1.0 2.0 3.0])
        result (materialization/materialize (fixture-plan) [source 0.25 7] evaluate-scalar)
        buffers (:program-buffers result)
        scalars (:program-scalars result)]
    (is (materialization/materialized-invocation? result))
    (is (= '[nsteps n half-dt u scratch]
           (mapv :program-value (:bindings (:plan result)))))
    (is (= #{'u 'scratch} (set (keys buffers))))
    (is (= #{'nsteps 'n 'half-dt} (set (keys scalars))))
    (is (= {:type :int :value 7} (get scalars 'nsteps)))
    (is (= {:type :int :value 3} (get scalars 'n)))
    (is (= {:type :double :value 0.125} (get scalars 'half-dt)))
    (is (= [3] (:shape (get buffers 'u))))
    (is (= [3] (:shape (get buffers 'scratch))))
    (is (identical? source (:source (get buffers 'u))))
    (is (= :copy (:initialization (get buffers 'u))))
    (is (= :zero (:initialization (get buffers 'scratch))))
    (is (= {:source-inspected false :driver-allocations 0} (:attributes result)))))

(deftest public-buffer-dtype-and-arity-are-checked
  (let [plan (fixture-plan)]
    (testing "argument order is exact"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"parameter order"
                            (materialization/materialize plan [(double-array 3) 0.25]
                                                         evaluate-scalar))))
    (testing "raw storage cannot silently change dtype"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dtype differs"
                            (materialization/materialize plan [(float-array 3) 0.25 7]
                                                         evaluate-scalar))))))

(deftest caller-owned-write-only-storage-is-materialized-outside-logical-inputs
  (let [value (array-value :float 4)
        plan (invocation/from-prefix
              {:id :write-only-storage
               :parameters '[x y]
               :parameter-values {'x value 'y value}
               :bindings [] :binding-values {}
               :program-values {'x value 'y value}
               :program-inputs '[x]
               :program-storage '[y]
               :program-outputs []})
        x (float-array 4)
        y (float-array 4)
        materialized (materialization/materialize plan [x y] evaluate-scalar)]
    (is (= '[x] (mapv :program-value (:bindings plan))))
    (is (= '[y] (mapv :program-value (:storage-bindings plan))))
    (is (= #{'x 'y} (set (keys (:program-buffers materialized)))))
    (is (identical? y (get-in materialized [:program-buffers 'y :source])))
    (is (empty? (:program-scalars materialized)))))

(deftest concrete-public-and-program-shapes-are-checked
  (testing "a static public shape cannot silently accept a different array length"
    (let [value (array-value :double 3)
          plan (invocation/from-prefix
                {:id :static-shape
                 :parameters '[x]
                 :parameter-values {'x value}
                 :bindings [] :binding-values {}
                 :program-values {'x value}
                 :program-inputs '[x] :program-outputs '[x]})]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"length differs"
           (materialization/materialize plan [(double-array 4)] evaluate-scalar)))))
  (testing "a producer and program input cannot disagree after symbolic extents resolve"
    (let [source-value (array-value :double '(extent x0))
          produced-value (array-value :double 'n)
          program-value (array-value :double 'm)
          plan (invocation/from-prefix
                {:id :symbolic-shape
                 :parameters '[x0]
                 :parameter-values {'x0 source-value}
                 :bindings '[[n (int (alength x0))]
                             [m (inc n)]
                             [x (aclone x0)]]
                 :binding-values {'n (scalar :int) 'm (scalar :int) 'x produced-value}
                 :program-values {'n (scalar :int) 'm (scalar :int)
                                  'x program-value 'result program-value}
                 :program-inputs '[n m x]
                 :program-outputs '[result]})]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"producer shape differs"
           (materialization/materialize plan [(double-array 3)] evaluate-scalar))))))
