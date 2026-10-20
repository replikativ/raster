(ns raster.compiler.ir.invocation-source-order-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.contractions :as contractions]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-source :as source]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.passes.parallel.auxiliary-semantic-witness-test :as auxiliary]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.par :as par]))

(deftm write-before-shape
  [x :- (Array double), first-output :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! first-output i 1 double (+ (aget x i) 1.0))
        ^long square (* (long n) (long n))
        out (double-array square)]
    (par/map! out i square double 1.0)
    out))

(deftm shape-before-write
  [x :- (Array double), n :- Long] :- (Array double)
  (let [^long square (* (long n) (long n))
        out (double-array square)]
    (par/map! out i square double (+ (aget x i) 1.0))
    out))

(deftm helper-source-zero-observed
  [x :- (Array double), early :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! early i 1 double (+ (aget x i) 1.0))
        out (arrays/zeros-like x n)
        result (par/map! out i n double (+ (aget out i) 1.0))]
    result))

(deftm write-then-functional-map
  [x :- (Array double), early :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! early i 1 double (+ (aget x i) 1.0))
        result (par/map [i n] (aget x i))]
    result))

(defn- compilation [v dtype]
  (let [program (:semantic (equation-first/compile v {:target :ocl:0 :dtype dtype}))]
    [program (get-in program [:attributes :invocation-plan])]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest prefix-source-ownership-is-distinct-from-late-routing
  (let [[program plan] (compilation #'shape-before-write :double)
        all (invocation/source-host-actions plan program)]
    (is (some #(= 'square (:binding %)) all))
    (is (empty? (invocation/ordered-host-actions plan program)))
    (is (= all (invocation/source-host-actions
                (assoc-in plan [:attributes :host-actions] []) program))
        "caller-provided action lists cannot remove the reconstructed obligation")
    (is (= :invocation-source-order-mismatch
           (reason #(invocation/source-host-actions
                     (assoc-in plan [:attributes :source-order]
                               (vec (reverse (invocation/source-order plan program)))) program))))))

(deftest helper-allocation-correspondence-is-retained-and-independently-rederived
  (let [[program plan] (compilation #'helper-source-zero-observed :double)
        order (invocation/source-order plan program)
        step-index (first (keep-indexed (fn [index step] (when (:source-projection step) index)) (:steps plan)))
        projection (get-in plan [:steps step-index :source-projection])]
    (is (vector? order))
    (is (some? step-index))
    (is (= 'raster.arrays/zeros-like (get-in projection [:helper :operation])))
    (is (= 'clojure.core/double-array (first (:constructor projection))))
    (doseq [changed [(update-in plan [:steps step-index] dissoc :source-projection)
                     (assoc-in plan [:steps step-index :source-projection :constructor]
                               '(clojure.core/double-array 0))
                     (assoc-in plan [:steps step-index :source-projection :helper :implementation]
                               'unrelated/constructor)]]
      (is (= :invocation-source-step-mismatch (reason #(invocation/source-order changed program)))))))

(deftest literal-helper-allocation-retains-an-independent-source-witness
  (let [compiled (equation-first/compile #'storage/double-fold-float-storage
                                        (merge storage/policy {:target :ocl:0 :dtype :double}))
        program (:semantic compiled)
        plan (get-in program [:attributes :invocation-plan])
        allocation (first (get-in program [:attributes :allocations]))]
    (is (= '(clojure.core/float-array 1) (get-in allocation [:source-projection :constructor])))
    (is (vector? (invocation/source-order plan program)))
    (doseq [changed [(assoc-in program [:attributes :allocations 0 :initialization] :unspecified)
                     (update-in program [:attributes :allocations 0] dissoc :source-projection)
                     (assoc-in program [:attributes :allocations 0 :source-projection :constructor]
                               '(clojure.core/float-array 2))]]
      (is (= :invocation-source-allocation (reason #(invocation/source-order plan changed)))))))

(deftest source-position-is-not-the-dependency-pruned-prefix-ordinal
  (let [[program plan] (compilation #'write-before-shape :double)
        order (invocation/source-order plan program)
        by-binding (into {} (map (juxt :binding identity)) order)
        step (first (:steps plan))
        square (get by-binding 'square)
        written (get by-binding 'written)]
    (is (= 0 (nth (:id step) 3)) "the invocation prefix contains only square")
    (is (= 1 (:position square)) "square belongs after the first numerical write")
    (is (= 0 (:position written)))
    (is (= (:id step) (:invocation-step square)))
    (is (some? (:equation square)) "one host definition has two representations")
    (is (= (mapv :id (:equations program)) (into [] (keep :equation) order)))
    (is (= (count (:steps plan)) (count (keep :invocation-step order))))))

(deftest eager-lowering-cannot-run-a-late-checked-source-action
  (let [compilation (equation-first/compile #'write-before-shape {:target :ocl:0 :dtype :double})
        program (:semantic compilation)
        plan (get-in program [:attributes :invocation-plan])
        actions (invocation/ordered-host-actions plan program)
        materializations (atom 0)
        first-output (double-array [0.0])]
    (is (some #(= 'square (:binding %)) actions))
    (is (every? #(pos? (:position %)) actions))
    (with-redefs [materialization/materialize
                  (fn [& _] (swap! materializations inc)
                    (throw (AssertionError. "late host calculation reached eager materialization")))]
      (is (= :ordered-source-execution-required
             (reason #(equation-first/lower compilation
                                            [(double-array [1.0]) first-output Long/MAX_VALUE])))))
    (is (zero? @materializations))
    (is (= [0.0] (vec first-output)))
    ;; Independent source oracle: the write happens BEFORE checked overflow.
    ;; The eager path must neither execute the overflow early nor claim success.
    (is (thrown? ArithmeticException
                 (write-before-shape (double-array [1.0]) first-output Long/MAX_VALUE)))
    (is (= [2.0] (vec first-output)))
    (let [changed (assoc-in plan [:attributes :source-order]
                            (vec (reverse (invocation/source-order plan program))))]
      (is (= :invocation-source-order-mismatch
             (reason #(invocation/ordered-host-actions changed program)))))))

(deftest late-source-allocation-still-requires-ordered-execution
  (let [compilation (equation-first/compile #'helper-source-zero-observed
                                           {:target :ocl:0 :dtype :double})
        program (:semantic compilation)
        plan (get-in program [:attributes :invocation-plan])
        allocations (filter :storage (invocation/ordered-host-actions plan program))
        first-output (double-array [0.0])
        materializations (atom 0)]
    (is (some #(= 'out (:binding %)) allocations))
    (with-redefs [materialization/materialize
                  (fn [& _] (swap! materializations inc)
                    (throw (AssertionError. "late source allocation reached eager materialization")))]
      (is (= :ordered-source-execution-required
             (reason #(equation-first/lower compilation
                                            [(double-array [1.0]) first-output -1])))))
    (is (zero? @materializations))
    (is (= [0.0] (vec first-output)))
    (is (thrown? NegativeArraySizeException
                 (helper-source-zero-observed (double-array [1.0]) first-output -1)))
    (is (= [2.0] (vec first-output)))))

(deftest generated-result-allocation-without-a-host-step-is-an-ordering-obligation
  (let [compilation (equation-first/compile #'write-then-functional-map
                                           {:target :ocl:0 :dtype :double})
        program (:semantic compilation)
        plan (get-in program [:attributes :invocation-plan])
        actions (invocation/ordered-host-actions plan program)
        generated (filter #(and (:realization %) (nil? (:invocation-step %))) actions)]
    (is (seq generated))
    (is (every? #(= :result-allocation (:kind (source/realization-action program %))) generated))
    (is (= :ordered-source-execution-required
           (reason #(equation-first/lower compilation
                                          [(double-array [1.0]) (double-array 1) 1]))))))

(deftest source-order-attests-typed-producers-not-just-local-names
  (let [[program plan] (compilation #'write-before-shape :double)
        order (invocation/source-order plan program)
        attested (assoc-in plan [:attributes :source-order] order)]
    (is (= attested (invocation/validate-against! attested program)))
    (doseq [changed [(assoc-in attested [:attributes :source-order 0 :position] 99)
                     (update-in attested [:attributes :source-order] #(vec (reverse %)))
                     (update-in attested [:attributes :source-order] pop)]]
      (is (= :invocation-source-order-mismatch
             (reason #(invocation/validate-against! changed program)))))
    (let [changed (assoc-in plan [:steps 0 :region]
                            '(lambda [n] (region [] [(clojure.core/inc n)])))]
      (is (= :invocation-source-step-mismatch
             (reason #(invocation/source-order changed program)))))))

(deftest source-order-cannot-legitimize-reordered-equations
  (let [[program plan] (compilation #'write-before-shape :double)
        [write shape & suffix] (:equations program)
        changed (assoc program :equations (into [shape write] suffix))]
    (is (= :invocation-source-order-equations
           (reason #(invocation/source-order plan changed))))))

(deftest physical-result-allocation-has-its-own-source-position
  (let [[program plan] (compilation #'contractions/fixed-matmul :float)
        order (invocation/source-order plan program)
        allocation (some #(when (= 'output (:binding %)) %) order)
        numerical (last (filter :equation order))]
    (is (= 'output (:storage allocation)))
    (is (< (:position allocation) (:position numerical)))
    (is (nil? (:equation allocation)) "allocation is not another numerical execution")
    (is (= (mapv :id (:equations program)) (into [] (keep :equation) order)))))

(defn- allocation-boundary []
  (let [values {'n (av/tensor {:dtype :long :shape []})
                'out (av/tensor {:dtype :double :shape '[n]})}
        program (parallel-program/make
                 {:dialect :typed-parallel :values values :inputs '[out] :outputs '[out]
                  :source '(let* [out (clojure.core/double-array n)] out)})
        plan (invocation/from-prefix
              {:id :allocation-source :parameters '[n] :parameter-values (select-keys values '[n])
               :bindings '[[out (clojure.core/double-array n)]]
               :binding-values (select-keys values '[out]) :program-values values
               :program-inputs '[out] :program-outputs '[out]})]
    [program plan]))

(deftest structured-allocation-cannot-erase-changed-constructor-arguments
  (let [[program plan] (allocation-boundary)]
    (is (vector? (invocation/source-order plan program)))
    (doseq [expression '[(clojure.core/double-array (clojure.core/inc n))
                        (clojure.core/float-array n)
                        (clojure.core/double-array [1.0])
                        (clojure.core/double-array n 1.0)]]
      (let [changed (assoc program :source (list 'let* ['out expression] 'out))]
        (is (= :invocation-source-allocation
               (reason #(invocation/source-order plan changed))) (str expression))))))

(deftest clone-correspondence-preserves-arity-and-operand-evaluation
  (let [values {'xs (av/tensor {:dtype :double :shape '[n]})
                'out (av/tensor {:dtype :double :shape '[n]})}
        program (parallel-program/make
                 {:dialect :typed-parallel :values values :inputs '[xs out] :outputs '[out]
                  :source '(let* [out (clojure.core/aclone xs)] out)})
        plan (invocation/from-prefix
              {:id :clone-source :parameters '[xs] :parameter-values (select-keys values '[xs])
               :bindings '[[out (clojure.core/aclone xs)]]
               :binding-values (select-keys values '[out]) :program-values values
               :program-inputs '[xs out] :program-outputs '[out]})]
    (is (vector? (invocation/source-order plan program)))
    (doseq [expression '[(clojure.core/aclone (clojure.core/long xs))
                        (clojure.core/aclone xs 0)]]
      (is (= :invocation-source-clone
             (reason #(invocation/source-order
                       plan (assoc program :source (list 'let* ['out expression] 'out)))))
          (str expression)))))

(deftest projection-unwrapping-is-not-a-narrowing-proof
  (is (source/projection-corresponds? '(clojure.core/long (clojure.core/alength xs))
                                       (av/tensor {:dtype :long :shape []})))
  (doseq [dtype [:byte :long]]
    (is (not (source/projection-corresponds? '(clojure.core/byte (clojure.core/alength xs))
                                             (av/tensor {:dtype dtype :shape []}))))))

(deftest allocation-claims-and-auxiliary-owners-are-unambiguous
  (let [[program plan] (allocation-boundary)
        claim {:destination 'out :source-expression '(clojure.core/double-array n)}]
    (is (= :invocation-source-allocation-claims
           (reason #(invocation/source-order
                     plan (assoc-in program [:attributes :allocations] [claim claim]))))))
  (let [program (#'auxiliary/functional-program)
        equation (first (:equations program))
        {:keys [binding expression]} (first (get-in equation [:attributes :realization-bindings]))]
    (is (= equation (source/realization-equation program binding expression)))
    (is (= :invocation-source-ambiguous-realization
           (reason #(source/realization-equation
                     (update program :equations conj (assoc equation :id :duplicate-owner))
                     binding expression))))))

(deftest auxiliary-actions-are-recovered-only-at-their-exact-source-position
  (doseq [program [(#'auxiliary/functional-program) (#'auxiliary/aliased-program)]
          equation (:equations program)
          {:keys [binding] :as witness} (get-in equation [:attributes :realization-bindings])
          :let [pairs (mapv vec (partition 2 (second (:source program))))
                position (first (keep-indexed #(when (= binding (first %2)) %1) pairs))
                entry {:position position :binding binding :realization (:id equation)}]]
    (is (= (assoc witness :equation (:id equation)) (source/realization-action program entry)))
    (is (= :invocation-source-realization-action
           (reason #(source/realization-action program (assoc entry :realization :wrong-owner)))))
    (is (= :invocation-source-realization-action
           (reason #(source/realization-action program (assoc entry :binding 'unrelated)))))))
