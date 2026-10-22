(ns raster.compiler.backend.jvm.nested-invocation-prefix-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-prefix :as prefix]
            [raster.compiler.core.source-prefix-test :as fixture]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.fixtures.distributed-training :as training]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link]))

(defn- capture []
  (#'fixture/capture #'training/local-loss
                     {:dtype :float :schedule {:precision :f32-scalar}}))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- reclose [arguments]
  (entry/call-with-definition-inputs #(apply prefix/close arguments)))

(def unrelated-input 7)

(deftest nested-prefix-joins-original-first-initializer-and-final-adjacent-allocation
  (let [{:keys [frontend inputs construction correspondence compilation]} (capture)
        program (:semantic compilation)
        arguments (conj inputs construction)
        step (first (get-in program [:attributes :invocation-plan :steps]))
        original (prefix/original-step correspondence frontend program step)]
    (is (some? correspondence))
    (is (= :nested-invocation-input (:kind original)))
    (is (identical? step (:step original)))
    (is (identical? (second (get-in program [:attributes :invocation-plan :steps])) (:allocation original)))
    (is (some? (reclose arguments)))
    (is (nil? (reclose inputs)) "the root alone cannot authorize newly inserted work")
    (is (nil? (reclose (assoc arguments 5 {}))))
    (is (nil? (prefix/original-step correspondence frontend (assoc program :copied true) step)))
    (is (nil? (prefix/original-step correspondence frontend program (into {} step))))
    (entry/call-with-definition-inputs
     (fn []
       (let [fresh (prefix/original-step correspondence frontend program step)]
         (is (= 2 (count (:boundaries fresh))))
         (doseq [{:keys [receiver]} (:boundaries fresh)]
           (is (contains? (:definition-proofs @entry/*emission-inputs*) receiver)
               "a new owner reobserves both actual dispatch/return boundaries")))))
    (let [bridge entry/frontend-reference-return-bridge]
      (with-redefs [entry/frontend-reference-return-bridge
                    (fn [& args] (when-let [result (apply bridge args)]
                                   (assoc result :parameter-carriers [])))]
        (is (nil? (prefix/original-step correspondence frontend program step)))))
    (let [view transport/source-construction-view]
      (with-redefs [transport/source-construction-view
                    (fn [& args] (when-let [result (apply view args)]
                                   (update result :builder-expansions #(into % %))))]
        (is (nil? (reclose arguments)) "duplicate source occurrences cannot pick a first matching frame")))
    (let [edge entry/frontend-invocation-input-source]
      (with-redefs [entry/frontend-invocation-input-source
                    (fn [& args] (when-let [result (apply edge args)]
                                   (assoc result :operand (Object.))))]
        (is (nil? (reclose arguments)) "the actual emitted root operand must join its input-origin edge")))
    (let [represented (nth inputs 3)
          [id expression & remaining] (second represented)
          change-prefix (fn [bindings] (with-meta (list* (first represented) bindings (nnext represented))
                                         (meta represented)))]
      (doseq [changed [(change-prefix (vec (concat [id '(clojure.core/+ rows 1)] remaining)))
                       (change-prefix (vec (concat (take 2 remaining) [id expression] (drop 2 remaining))))]]
        (is (nil? (reclose (assoc arguments 3 changed))))))
    (let [observe inline/nested-original-call-frame]
      (doseq [change [(fn [frame] (when frame (update frame :argument-slots #(vec (reverse %)))))
                      (fn [frame] (when frame (assoc frame :parent-frame nil)))]]
        (with-redefs [inline/nested-original-call-frame (fn [& args] (change (apply observe args)))]
          (is (nil? (reclose arguments)) "slot order and imported ancestry cannot be inferred from names"))))))

(deftest nested-prefix-executes-all-original-arguments-once-in-source-order
  (let [calls (atom []) prepare entry/prepare-frontend-invocation-input!
        {:keys [compilation frontend correspondence]}
        (with-redefs [entry/prepare-frontend-invocation-input!
                      (fn [frontend call ordinal]
                        (when-let [region (prepare frontend call ordinal)]
                          (let [invoke (:invoke region) op (descriptor/semantic-op call)]
                            (assoc region :invoke (fn [& arguments]
                                                    (swap! calls conj [op ordinal])
                                                    (apply invoke arguments))))))]
          (capture))
        step (first (get-in compilation [:semantic :attributes :invocation-plan :steps]))
        theta (float-array 17) x (float-array 34) y (float-array 2)
        arguments [theta x y (long 2) (long 17)]
        operands {'rows {:type :long :value (long 2)}}
        execute #(owner/evaluate-ordered-invocation-step compilation step operands %)
        forbidden (fn [& _] (throw (AssertionError. "prepared source tried to acquire storage or recompile")))]
    (is (empty? @calls) "preparation executes no arguments")
    (with-redefs [gpu/alloc! forbidden entry/prepare-frontend-invocation-input! forbidden
                  scalar/evaluate-invocation-step forbidden]
      (is (= {:type :long :value (long 2)} (execute arguments))))
    (is (= (vec (concat (map #(vector 'raster.dl.nn/linear-nb %) (range 5))
                       (map #(vector 'raster.arrays/alloc-like %) (range 2)))) @calls)
        "even the unused width argument completes before entering the allocator")
    (doseq [bad [nil (pop arguments) (assoc arguments 4 (int 17))
                 (assoc arguments 3 (long 3)) (assoc arguments 0 (double-array 17))]]
      (reset! calls [])
      (is (= :ordered-nested-source-arguments (reason #(execute bad))))
      (is (empty? @calls)))
    (let [receiver (get-in (prefix/original-step correspondence frontend (:semantic compilation) step)
                          [:boundaries 0 :receiver])
          metadata (meta receiver)]
      (is (var? receiver))
      (try
        (alter-meta! receiver assoc ::changed true)
        (reset! calls [])
        (is (= :stale-definition-proof-input (reason #(execute arguments))))
        (is (empty? @calls) "stale selection refuses before executing any original argument")
        (finally (reset-meta! receiver metadata))))
    (reset! calls [])
    (is (= {:type :long :value (long 2)} (execute arguments)))
    (is (= 7 (count @calls)))))

(deftest nested-execution-never-rereads-the-mutable-compilation-collector
  (let [prepare-var (ns-resolve 'raster.compiler.backend.jvm.source-owner 'prepare-scalar-steps)
        prepare @prepare-var collector (atom nil)
        {:keys [compilation]}
        (with-redefs-fn {prepare-var (fn [& arguments]
                                      (reset! collector (nth arguments 5))
                                      (apply prepare arguments))} capture)
        history (binding [entry/*emission-inputs* (atom {:static [] :values []})]
                  (entry/observe-definition-input!
                   #'unrelated-input (the-ns 'raster.compiler.backend.jvm.nested-invocation-prefix-test)
                   'unrelated-input)
                  @entry/*emission-inputs*)
        _ (entry/assert-definition-inputs-current! history)
        metadata (meta #'unrelated-input)
        step (first (get-in compilation [:semantic :attributes :invocation-plan :steps]))]
    (is (some? @collector))
    (try
      (alter-meta! #'unrelated-input assoc ::changed true)
      ;; Poison ONLY the obsolete compilation atom, not the immutable owner or
      ;; the authentic dependencies used to emit the seven argument regions.
      (reset! @collector history)
      (is (= {:type :long :value (long 2)}
             (owner/evaluate-ordered-invocation-step
              compilation step {'rows {:type :long :value (long 2)}}
              [(float-array 17) (float-array 34) (float-array 2) (long 2) (long 17)])))
      (finally (reset-meta! #'unrelated-input metadata)))))

(deftest nested-allocation-keeps-constructor-narrowing-and-refuses-incompatible-extents
  (let [{:keys [compilation]} (capture)
        forbidden (fn [& _] (throw (AssertionError. "allocation preparation acquired device resources")))]
    (with-redefs [gpu/alloc! forbidden gpu/make-session forbidden]
      (doseq [[rows expected] [[(long -1) NegativeArraySizeException]
                              [(inc (long Integer/MAX_VALUE)) NegativeArraySizeException]
                              [(long 4294967296) :invocation-ordered-allocation-shape]]]
        (let [arguments [(float-array 17) (float-array 0) (float-array 0) rows (long 17)]
              initial (owner/begin-ordered-realization compilation arguments nil)
              _ (owner/assert-ordered-executor-coverage! compilation initial arguments)
              counted (invocation-link/advance-host-source
                       initial (fn [step operands]
                                 (owner/evaluate-ordered-invocation-step compilation step operands arguments))
                       nil nil)
              actual (try
                       (invocation-link/prepare-source-allocation
                        counted #(scalar/evaluate-array-size (:source-ns compilation) (update-vals %2 :value) %1) nil)
                       :unexpected-success
                       (catch NegativeArraySizeException error (class error))
                       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))]
          (is (= expected actual))
          (if (= expected NegativeArraySizeException)
            (is (thrown? NegativeArraySizeException (float-array rows)))
            (is (zero? (alength (float-array rows)))
                "wrapped source counts cannot silently become a huge native shape")))))))

(deftest nested-public-array-and-scalar-admission-precedes-session-acquisition
  (let [{:keys [compilation]} (capture)
        arguments [(float-array 17) (float-array 34) (float-array 2) (long 2) (long 17)]
        forbidden (fn [& _] (throw (AssertionError. "invalid public arguments opened a device session")))]
    (with-redefs [gpu/make-session forbidden gpu/alloc! forbidden]
      (doseq [bad [(assoc arguments 0 nil) (assoc arguments 1 nil)
                   (assoc arguments 2 nil) (assoc arguments 0 (double-array 17))
                   (assoc arguments 4 (int 17))]]
        (is (instance? clojure.lang.ExceptionInfo
                       (try (gpu-link/instantiate-host-result! compilation bad) nil
                            (catch clojure.lang.ExceptionInfo error error))))))))
