(ns raster.compiler.passes.parallel.auxiliary-semantic-witness-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.typed-soac-route :as typed]
            [raster.compiler.passes.parallel.structured-control-route :as structured]
            [raster.compiler.passes.parallel.segop-lower-pass :as lowering]))

(defn- functional-program []
  (:program (typed/attempt
             '(let* [result (raster.par/pmap i 8 double (clojure.core/aget x i))] result)
             :double {'x :double})))

(defn- aliased-program []
  (:program (typed/attempt
             '(let* [left (raster.par/map! a i 8 double (clojure.core/aget x i))
                     right (raster.par/map! b j 8 double (clojure.core/aget x j))]
                [left right])
             :double {'x :double 'a :double 'b :double})))

(defn- replace-binding [program binding expression]
  (update program :source
          (fn [source]
            (with-meta (list* (first source)
                              (vec (mapcat (fn [[id value]]
                                             [id (if (= id binding) expression value)])
                                           (partition 2 (second source))))
                              (nnext source))
              (meta source)))))

(defn- semantic-variants [expression]
  (cond-> [(with-meta expression (assoc (meta expression) :tag 'float))
           (with-meta expression (assoc (meta expression) :raster.type/tag 'float))
           (with-meta expression (assoc (meta expression) :raster.op/original 'raster.numeric/-))]
    (seq? expression)
    (conj (with-meta (list* (with-meta (first expression)
                             (assoc (meta (first expression)) :raster.op/original 'raster.numeric/-))
                           (rest expression))
            (meta expression)))))

(defn- relocated [expression]
  (with-meta expression (assoc (meta expression) :file "relocated.clj" :line 90
                              :column 7 :end-line 91 :end-column 9)))

(deftest auxiliary-allocation-and-alias-witnesses-use-semantic-source-equality
  (let [programs [(functional-program) (aliased-program)]
        all-witnesses (mapcat #(mapcat (fn [equation] (get-in equation [:attributes :realization-bindings]))
                                      (:equations %)) programs)]
    (is (every? some? programs))
    (is (= #{:result-allocation :storage-alias} (set (map :kind all-witnesses))))
    (doseq [program programs
            :let [witnesses (vec (mapcat #(get-in % [:attributes :realization-bindings]) (:equations program)))]]
    (doseq [{:keys [binding expression]} witnesses]
      (is (typed/realization-binding? program binding expression))
      (let [moved (relocated expression)]
        (is (typed/realization-binding? program binding moved))
        (is (typed/realization-binding? (replace-binding program binding moved) binding expression)))
      (doseq [changed (semantic-variants expression)]
        (is (= expression changed) "ordinary equality hides the semantic metadata mutation")
        (is (false? (typed/realization-binding? program binding changed)))
        (is (false? (typed/realization-binding? (replace-binding program binding changed)
                                               binding expression)))))
    ;; Updating both the source and advisory allocation witness cannot override its independently
    ;; reconstructed typed allocation. Alias destination membership remains its separate owner check.
    (doseq [{:keys [binding expression]} (filter #(= :result-allocation (:kind %)) witnesses)
            changed (semantic-variants expression)]
      (let [forged (-> (replace-binding program binding changed)
                       (update :equations
                               #(mapv (fn [equation]
                                        (update-in equation [:attributes :realization-bindings]
                                                   (fn [entries]
                                                     (mapv (fn [entry]
                                                             (cond-> entry (= binding (:binding entry))
                                                               (assoc :expression changed))) entries)))) %)))]
        (is (false? (typed/realization-binding? forged binding changed))))))))

(deftest host-prefix-and-allocation-executors-retain-semantic-expression-witnesses
  (let [check (ns-resolve 'raster.compiler.passes.parallel.structured-control-route 'validate-host-coverage!)
        expression '(clojure.core/double-array n)
        program {:source (list 'let* ['buffer expression] 'buffer) :equations []}]
    (doseq [allocation? [false true]
            :let [owner (cond-> program allocation?
                          (assoc :attributes {:allocations [{:destination 'buffer
                                                            :source-expression expression}]}))
                  prefix (if allocation? [] [['buffer expression]])]]
      (is (= owner (check owner prefix #{})))
      (let [moved (replace-binding owner 'buffer (relocated expression))]
        (is (= moved (check moved prefix #{}))))
      (doseq [changed (semantic-variants expression)]
        (is (= :structured-control-host-continuation
               (try (check (replace-binding owner 'buffer changed) prefix #{}) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest singleton-packet-source-admission-preserves-semantic-metadata
  (let [packet (functional-program)
        equation (first (:equations packet))
        binding (second (:site equation))
        expression (:source equation)
        admit (ns-resolve 'raster.compiler.passes.parallel.segop-lower-pass 'singleton-packet)
        run (fn [candidate]
              (with-redefs [typed/attempt (fn [& _] {:program candidate})
                            ;; Keep this test at packet admission, before independent scheduling.
                            lowering/segop-lower-pass (fn [p _] {:form p})]
                (admit (first (:results equation)) expression {:dtype :double} (atom #{}) 'map)))]
    (is (some? (:primary (run packet))))
    (is (some? (:primary (run (replace-binding packet binding (relocated expression))))))
    (doseq [changed (semantic-variants expression)]
      (is (= :typed-map-source
             (try (run (replace-binding packet binding changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:stage (ex-data error)))))))))

(deftest analyzed-allocation-source-retains-its-declared-host-scalar-semantics
  (let [source '(let* [size (clojure.core/* nrows width)
                       out (clojure.core/float-array size)
                       effect (raster.par/map-void! i nrows
                                (clojure.core/aset out i (float (clojure.core/aget x i))))]
                  effect)
        options {:dtype :float :public-parameters '[x nrows width]
                 :array-types {'x :float}
                 :scalar-types {'nrows :long 'width :long 'size :long}}
        program (:program (typed/attempt source :float {'x :float}
                                        {:scalar-types (:scalar-types options)}))
        allocation (first (get-in program [:attributes :allocations]))
        expression (:source-expression allocation)
        retained (get (into {} (map vec (partition 2 (second (:source program))))) 'out)
        promote #(structured/promote-program % options)]
    (is (= 'out (:destination allocation)))
    (is (= 'long (:tag (meta (second expression)))))
    (is (= 'long (:raster.type/tag (meta (second retained)))))
    (is (some? (promote program)))
    (doseq [changed (semantic-variants expression)]
      (is (= :structured-control-host-continuation
             (try (promote (replace-binding program 'out changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= :structured-control-host-continuation
             (try (promote (assoc-in program [:attributes :allocations 0 :source-expression] changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (let [changed (with-meta (list (first expression)
                                  (with-meta (second expression) {:tag 'int :raster.type/tag 'int}))
                    (meta expression))]
      (is (= :structured-control-host-continuation
             (try (promote (replace-binding program 'out changed)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest analyzed-binding-transport-requires-exact-unique-original-ownership
  (let [transport (ns-resolve 'raster.compiler.passes.parallel.typed-soac-route
                              'retain-analyzed-binding-expressions)
        binding (with-meta 'out {:line 1})
        form (with-meta (list 'let* (with-meta [binding '(clojure.core/float-array size)] {:line 2})
                              'out) {:line 3})
        expression '(clojure.core/float-array ^long size)
        descriptor {:id 0 :sym 'out :expr expression}
        projected (transport form [descriptor] :allocation-source-binding-mismatch)]
    (is (identical? binding (first (second projected))))
    (is (= (meta form) (meta projected)))
    (is (= (meta (second form)) (meta (second projected))))
    (is (= (nnext form) (nnext projected)))
    (doseq [invariant [:allocation-source-binding-mismatch :protected-source-binding-mismatch]
            descriptions [[(dissoc descriptor :id)] [(assoc descriptor :id -1)]
                          [(assoc descriptor :id 1)] [(assoc descriptor :sym 'foreign)]
                          [(dissoc descriptor :expr)] [descriptor descriptor]]]
      (is (= invariant
             (try (transport form descriptions invariant) nil
                  (catch clojure.lang.ExceptionInfo error (:invariant (ex-data error)))))))))
