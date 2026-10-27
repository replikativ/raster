(ns raster.compiler.core.inline-call-plan-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.inline-call-plan :as plan]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.pipeline :as pipeline]))

(deftm sequential-array-helper [input :- (Array double) n :- Long] :- (Array double) input)
(deftm sequential-array-owner [input :- (Array double) n :- Long] :- (Array double) input)
(deftm sequential-long-helper [n :- Long] :- Long n)
(deftm sequential-long-owner [n :- Long] :- Long n)

(defn- options [owner]
  (let [source (core/resolve-deftm-var owner)
        facts (pipeline/deftm-frontend-facts source :double)]
    {:source-var source :source-ns (:source-ns facts)
     :active-params (:parameters facts) :public-parameters (:parameters facts)}))

(defn- call [helper argument return-tag]
  (let [source (core/resolve-deftm-var helper)
        source-meta (meta source)
        operation (symbol (str (ns-name (:ns source-meta)))
                          (if (= helper #'sequential-array-helper)
                            "sequential-array-helper" "sequential-long-helper"))
        implementation (symbol (str (ns-name (:ns source-meta)))
                               (str (:name source-meta) "-impl"))
        arguments (if (= helper #'sequential-array-helper) [argument 'n] [argument])
        selection (entry/selected-definition operation implementation (count arguments))]
    (with-meta
      (list* '.invk (with-meta implementation
                    {:raster.type/tag (get-in selection [:entry :typed-iface])
                     :raster.type/ret-tag return-tag}) arguments)
      {:raster.op/original operation})))

(deftest sequential-sites-use-actual-emitter-contexts
  (entry/call-with-definition-inputs
   (fn []
     (let [scope (plan/source-context (options #'sequential-array-owner))
           first-call (call #'sequential-array-helper 'input 'doubles)
           second-call (call #'sequential-array-helper 'input 'doubles)]
       (is (some? scope))
       (doseq [candidate [(list 'do first-call second-call)
                          (list 'let* ['ignored first-call] second-call)]]
         (let [plans (plan/call-plans scope candidate)
               first-plan (.get plans first-call)
               second-plan (.get plans second-call)]
           (is (= 2 (.size plans)))
           (is (identical? candidate (:candidate first-plan)))
           (is (identical? candidate (:candidate second-plan)))
           (is (identical? (nth first-call 2) (get-in first-plan [:inputs 0 :operand])))
           (is (true? (get-in second-plan [:caller-context :object-call-result?])))
           (is (nil? (get-in second-plan [:plan :primitive-result-class])))))
       (let [candidate (list 'let* ['alias first-call]
                             (call #'sequential-array-helper 'alias 'doubles))
             plans (plan/call-plans scope candidate)
             tail-plan (.get plans (nth candidate 2))]
         (is (= 2 (.size plans)))
         (is (= :ref (get-in tail-plan [:inputs 0 :raw-carrier])))
         (is (= :checkcast (get-in tail-plan [:inputs 0 :adaptation])))
         (is (= 'alias (get-in tail-plan [:inputs 0 :operand]))))))))

(deftest sequential-sites-do-not-cross-unknown-or-ambiguous-boundaries
  (entry/call-with-definition-inputs
   (fn []
     (let [scope (plan/source-context (options #'sequential-array-owner))
           before (call #'sequential-array-helper 'input 'doubles)
           after (call #'sequential-array-helper 'input 'doubles)
           visits (atom 0)
           quoted (list 'quote (lazy-seq (swap! visits inc) (list before)))]
       (doseq [unknown ['(unknown-callback input) '(if true input input)
                        '(loop* [i 0] input) quoted
                        (list '.invk (second before) '(unknown-callback input))]]
         (let [plans (plan/call-plans scope (list 'do before unknown after))]
           (is (= 1 (.size plans)))
           (is (some? (.get plans before)))
           (is (nil? (.get plans after)))))
       (is (zero? @visits))
       (is (zero? (.size (plan/call-plans scope (list 'do before before)))))
       (doseq [opaque [(list 'if true before 'input)
                       (list 'loop* [] before)
                       (list 'unknown-callback before)]]
         (is (zero? (.size (plan/call-plans scope (list 'do before opaque))))))
       (let [equal-copy (with-meta (apply list before) (meta before))]
         (is (= before equal-copy))
         (is (not (identical? before equal-copy)))
         (is (= 2 (.size (plan/call-plans scope (list 'do before equal-copy))))))
       (is (zero? (.size (plan/call-plans scope
                                         (list 'let* ['input before] after)))))
       (is (zero? (.size (plan/call-plans scope
                                         (list 'let* ['x '(unknown-callback input)] after)))))))))

(deftest primitive-caller-results-remain-outside-object-publication-plans
  (entry/call-with-definition-inputs
   (fn []
     (let [scope (plan/source-context (options #'sequential-long-owner))
           initializer (call #'sequential-long-helper 'n 'long)
           tail (call #'sequential-long-helper 'n 'long)
           candidate (list 'let* ['answer initializer] tail)
           plans (plan/call-plans scope candidate)]
       (is (= 1 (.size plans)))
       (is (some? (.get plans initializer)))
       (is (nil? (.get plans tail)))
       (is (nil? (get-in (.get plans initializer) [:plan :primitive-result-class])))))))

(deftest shared-sequential-context-factory-preserves-the-existing-context-contract
  (let [base {:object-result? true :void-context true :object-call-result? true
              :preserve-source-call-results? false :source-ns *ns*}]
    (is (= (assoc (dissoc base :void-context :object-result?) :preserve-source-call-results? true)
           (bytecode/source-evaluation-context base :initializer)))
    (is (= (assoc (dissoc base :object-result?) :void-context true)
           (bytecode/source-evaluation-context base :statement)))
    (is (identical? base (bytecode/source-evaluation-context base :body-result)))
    (is (= (assoc (dissoc base :object-result? :void-context :object-call-result?)
                  :preserve-source-call-results? true :object-call-result? true)
           (bytecode/source-evaluation-context base :object-call)))))
