(ns raster.compiler.backend.jvm.binding-context-owner-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [clojure.set :as set]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.dl.loss :as loss]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm response [a :- Double k :- Double x :- Double] :- Double
  (n/* a (n// x (n/+ k x))))
(deftm array-tail [theta :- (Array double) x :- Double] :- Double
  (let [a (aget theta 0) k (aget theta 1)] (response a k x)))
(deftm shadowed [theta :- (Array double) x :- Double] :- Double
  (let [a (aget theta 0) a (aget theta 1)] (response a a x)))
(deftm inferred-binding [x :- Double] :- Double
  (let [power (n/* x x)] power))
(deftm weighted-reduction [xs :- (Array double) coefficient :- Double cnt :- Long] :- Double
  (let [gain (n/* coefficient coefficient)]
    (par/reduce acc 0.0 i cnt (n/+ acc (n/* gain (aget xs i))))))
(deftm float-reduced [xs :- (Array float) cnt :- Long] :- Float
  (par/reduce acc (float 0.0) i cnt (n/+ acc (aget xs i))))
(deftm reduction-input [xs :- (Array double) coefficient :- Double cnt :- Long] :- Double
  (n/* coefficient (par/reduce acc 0.0 i cnt (n/+ acc (aget xs i)))))
(deftm initialized-reduction [xs :- (Array double) events :- (Array long) cnt :- Long] :- Double
  (par/reduce acc (double (do (aset events 0 (long 1)) (aget xs 0))) i cnt (n/+ acc 1.0)))
(deftm root-context-reduction [xs :- (Array float) cnt :- Long] :- Double
  (clojure.core//
    (par/reduce acc (float 0.0) i cnt (clojure.core/+ acc (float (aget xs i))))
    (double cnt)))

(deftm counted-steps [cnt :- Long] :- Float
  (par/reduce acc (float 0.0) i cnt (n/+ acc (float 1.0))))

(deftm count-before-initialization [events :- (Array long) cnt :- Long] :- Float
  (par/reduce acc (float (do (aset events 1 (inc (aget events 1))) 0.0))
    i (do (aset events 0 (inc (aget events 0))) cnt)
    (do (aset events 2 (inc (aget events 2))) (n/+ acc (float 1.0)))))

(defonce count-global-calls (atom 0))
(defn count-sibling [_] (swap! count-global-calls inc) 0)
(defn _plus__m_long_long [_ _] (swap! count-global-calls inc) 0)
(defn input [value] (swap! count-global-calls inc) value)

(use-fixtures :once (fn [run] (response 0.8 0.7 1.0) (run)))

(defn- emitted [v]
  (let [source (core/resolve-deftm-var v)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))]
    (core/ensure-bytecode-implementation! receiver)
    (:raster.core/bytecode-emission (meta receiver))))

(defn- invoke-region [region environment]
  (apply (:invoke region) (mapv environment (:parameters region))))

(defn- thrown-result [run]
  (try (run) :returned
       (catch Throwable error [(class error) (.getMessage error)])))

(deftest reduction-count-outline-preserves-checked-source-completion
  (let [emission (emitted #'counted-steps)
        original (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission original)
        region (bytecode/compile-reduction-count-region emission original)]
    (is (= :owned-bytecode-reduction-count-region (:kind region)))
    (is (identical? (:bound observed) (:bound-context region)))
    (is (identical? (get-in observed [:bound :initializer])
                    (get-in region [:input :operand])))
    (is (contains? (:bound observed) :element-type))
    (is (= (get-in observed [:bound :element-type]) (get-in region [:spec :element-type])))
    (is (= Integer/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
    (is (= ['cnt] (:parameters region)))
    (is (= Long/TYPE (first (.getParameterTypes ^java.lang.reflect.Method (:method region)))))
    (doseq [cnt [0 1 3 -1 (long Integer/MIN_VALUE) (long Integer/MAX_VALUE)]]
      (is (= cnt (invoke-region region {'cnt cnt}))))
    (doseq [cnt [4294967297 Long/MAX_VALUE Long/MIN_VALUE]]
      (let [expected (thrown-result #(counted-steps cnt))]
        (is (not= :returned expected))
        (is (= expected (thrown-result #(invoke-region region {'cnt cnt}))))))
    (doseq [cnt [0 1 3 -1]]
      (is (= (float (max 0 cnt)) (counted-steps cnt))))))

(deftest reduction-count-outline-executes-only-count-effects-once
  (let [emission (emitted #'count-before-initialization)
        original (first (vals (:reduction-catalog emission)))
        region (bytecode/compile-reduction-count-region emission original)]
    (is (= #{'events 'cnt} (set (:parameters region))))
    (is (not-any? #(contains? region %) [:pure? :replayable? :source-position? :domain-proved?]))
    (doseq [cnt [-1 0 2]]
      (let [events (long-array 3)]
        (is (= cnt (invoke-region region {'events events 'cnt cnt})))
        (is (= [1 0 0] (vec events)))))
    (let [events (long-array 3)]
      (is (= (float 2.0) (count-before-initialization events 2)))
      (is (= [1 1 2] (vec events))))
    (let [events (long-array 1)]
      (is (= 2 (invoke-region region {'events events 'cnt 2})))
      (is (= [1] (vec events)))
      (is (thrown? ArrayIndexOutOfBoundsException (count-before-initialization events 2)))
      (is (= [2] (vec events)) "the primal reaches its separate seed only after counting"))
    (is (= (first (thrown-result #(count-before-initialization nil 2)))
           (first (thrown-result #(invoke-region region {'events nil 'cnt 2})))))
    (let [expected-events (long-array 3)
          actual-events (long-array 3)
          expected (thrown-result #(count-before-initialization expected-events Long/MAX_VALUE))]
      (is (not= :returned expected))
      (is (= expected (thrown-result #(invoke-region region {'events actual-events 'cnt Long/MAX_VALUE}))))
      (is (= [1 0 0] (vec expected-events) (vec actual-events))))))

(deftest reduction-count-outline-requires-joined-original-evidence
  (let [emission (emitted #'counted-steps)
        original (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission original)
        bound-key [(get-in observed [:expansion :witness]) :bound-scope]
        path [:reduction-contexts bound-key]
        binder (get-in observed [:bound :binder])
        changed [(with-meta (apply list original) (meta original))]]
    (is (= :unowned-bytecode-reduction-count-region
           (try (bytecode/compile-reduction-count-region (emitted #'float-reduced) original) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (doseq [source changed]
      (is (= :unowned-bytecode-reduction-count-region
             (try (bytecode/compile-reduction-count-region emission source) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (doseq [mutated [(assoc-in emission (conj path :status) :ambiguous)
                     (assoc-in emission (conj path :initializer) '(int 7))
                     (assoc-in emission (conj path :binder) (with-meta (symbol (str binder)) (meta binder)))
                     (assoc-in emission (into path [:before 'cnt :type]) :double)
                     (update-in emission (conj path :context) dissoc :void-context)
                     (assoc-in emission (into path [:context :object-result?]) nil)]]
      (is (nil? (bytecode/retained-reduction-emission mutated original)))
      (is (= :unowned-bytecode-reduction-count-region
             (try (bytecode/compile-reduction-count-region mutated original) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest reduction-count-outline-refuses-incomplete-compiler-context
  (let [emission (emitted #'counted-steps)
        original (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission original)
        path [:reduction-contexts [(get-in observed [:expansion :witness]) :bound-scope]]]
    (doseq [mutated [(update-in emission path dissoc :element-type)
                     (update-in emission path dissoc :unretained-sibling-calls?)
                     (assoc-in emission (conj path :unretained-sibling-calls?) nil)]]
      (is (= :unowned-bytecode-reduction-count-region
             (try (bytecode/compile-reduction-count-region mutated original) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest reduction-count-outline-does-not-rebind-original-siblings-to-globals
  (doseq [[bound siblings]
          [['(count-sibling cnt)
            [{:name 'count-sibling :params [(with-meta 'cnt {:tag 'long})]
              :return-tag 'long :walked-body '[3]
              :source-ns (the-ns 'raster.compiler.backend.jvm.binding-context-owner-test)}]]
           ;; Including the real mangled deftm seeds the typed sibling table.
           ;; Its lookup uses name, so the other namespace is not a distinction.
           ['(if (clojure.core/< cnt 0)
               (raster.numeric/_plus__m_long_long 0 3)
               (raster.compiler.backend.jvm.binding-context-owner-test/_plus__m_long_long cnt 3))
            []]]]
    (let [before @count-global-calls
          body (list 'raster.par/reduce 'acc '(float 0.0) 'i bound
                     '(clojure.core/+ acc (float 1.0)))
          interface (types/ensure-fn-interface! '[long])
          emitter (fn [& args] (apply bytecode/compile-typed-impl! (concat args [siblings])))
          {:keys [emission implementation]}
          (#'core/emit-owned-bytecode!
           emitter "raster.compiled.ReductionCountSibling" '[cnt] '[long] 'float [body]
           (the-ns 'raster.compiler.backend.jvm.binding-context-owner-test)
           (:iface-name interface) 'count-sibling-original)
          original (first (vals (:reduction-catalog emission)))
          observed (bytecode/retained-reduction-emission emission original)]
      (is (= (float 3.0) (implementation 0)))
      (is (true? (get-in observed [:bound :unretained-sibling-calls?])))
      (is (= :unsupported-bytecode-input-callees
             (try (bytecode/compile-reduction-count-region emission original) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= before @count-global-calls) "neither original execution nor outlining calls the colliding global"))))

(deftest method-local-count-callee-denial-respects-quoted-data
  (let [candidate? #'bytecode/method-local-callee-candidate?
        context {:typed-sibling-methods {"helper" {}} :sibling-methods {"static-helper" {}}}]
    (is (true? (candidate? '(other/helper 1) context)))
    (is (true? (candidate? '(fn* [] (helper 1)) context)))
    (is (true? (candidate? '(static-helper 1) context)))
    (is (false? (candidate? '(quote (helper 1)) context)))
    (is (false? (candidate? '(clojure.core/quote (helper 1)) context)))
    (is (false? (candidate? '(other/static-helper 1) context)))))

(deftest owned-count-outline-does-not-capture-a-global-input-function
  (let [body '(raster.par/reduce acc (float 0.0) i (input cnt)
                (clojure.core/+ acc (float 1.0)))
        interface (types/ensure-fn-interface! '[long])
        {:keys [emission implementation]}
        (#'core/emit-owned-bytecode!
         bytecode/compile-typed-impl! "raster.compiled.GlobalInputCount"
         '[cnt] '[long] 'float [body]
         (the-ns 'raster.compiler.backend.jvm.binding-context-owner-test)
         (:iface-name interface) 'global-input-count)
        original (first (vals (:reduction-catalog emission)))
        before @count-global-calls
        region (bytecode/compile-reduction-count-region emission original)]
    (is (= before @count-global-calls) "outlining must not execute the input")
    (is (not= "input" (.getName ^java.lang.reflect.Method (:method region))))
    (is (= (str (get-in region [:spec :name])) (.getName ^java.lang.reflect.Method (:method region))))
    ;; Guard the regression run against the old recursively self-capturing code.
    (when (not= "input" (.getName ^java.lang.reflect.Method (:method region)))
      (is (= 3 (invoke-region region {'cnt 3})))
      (is (= (inc before) @count-global-calls))
      (is (= (float 3.0) (implementation 3)))
      (is (= (+ before 2) @count-global-calls)))))

(deftest outlined-method-name-avoids-qualified-and-bound-source-names
  (let [choices (atom ['taken 'bound 'fresh])
        allocate (fn [& _] (let [choice (first @choices)] (swap! choices subvec 1) choice))
        chosen (with-redefs [clojure.core/gensym allocate]
                 (#'bytecode/fresh-input-method-name
                  {:params '[parameter]
                   :walked-body '[(let* [bound 1] (other/taken bound))]}))]
    (is (= 'fresh chosen))
    (is (empty? @choices))))

(deftest reduction-count-outline-retains-the-entire-emission-history
  (let [emission (emitted #'counted-steps)
        original (first (vals (:reduction-catalog emission)))
        region (bytecode/compile-reduction-count-region emission original)
        dependency (some #(when (= #'n/_plus__m_float_float (:var %)) (:var %))
                         (:dependency-snapshots emission))]
    (is (var? dependency))
    (with-redefs-fn {dependency (fn [& _] nil)}
      #(is (thrown? clojure.lang.ExceptionInfo (invoke-region region {'cnt 1}))))
    (is (= 2 (invoke-region region {'cnt 2})))))

(deftest sequential-binding-facts-come-from-the-actual-owner
  (let [emission (emitted #'array-tail)
        scope (first (:body emission))
        first-binding (bytecode/retained-binding-context emission scope 0)
        second-binding (bytecode/retained-binding-context emission scope 1)]
    (is (= 'a (:binder first-binding)))
    (is (= 'k (:binder second-binding)))
    (is (= :double (get-in first-binding [:after :type])))
    (is (= (:after first-binding) (get-in second-binding [:before 'a])))
    (is (= :ref (get-in first-binding [:before 'theta :type])))
    (is (identical? (class (double-array 0)) (get-in first-binding [:before 'theta :reference-class])))
    (is (nil? (bytecode/retained-binding-context emission scope -1)))
    (is (nil? (bytecode/retained-binding-context emission scope 2)))
    (is (nil? (bytecode/retained-binding-context
               emission (with-meta (apply list scope) (meta scope)) 0)))))

(deftest whole-body-entry-and-return-facts-belong-to-the-original-body
  (let [emission (emitted #'array-tail)
        context (bytecode/retained-body-context emission)
        scope (first (:body emission))
        first-binding (bytecode/retained-binding-context emission scope 0)]
    (is (identical? (:body emission) (:body context)))
    (is (= (:locals context) (:before first-binding)))
    (is (= :double (:return-stack-type context)))
    (is (= {:object-result? false :object-call-result? false
            :preserve-source-call-results? false :void-context false} (:result-context context)))
    (is (nil? (bytecode/retained-body-context
               (assoc emission :body (with-meta (into [] (:body emission)) (meta (:body emission)))))))))

(deftest same-spelling-is-not-a-binding-identity
  (let [emission (emitted #'shadowed)
        scope (first (:body emission))
        first-binding (bytecode/retained-binding-context emission scope 0)
        second-binding (bytecode/retained-binding-context emission scope 1)]
    (is (= (:binder first-binding) (:binder second-binding)))
    (is (not= (get-in first-binding [:after :slot]) (get-in second-binding [:after :slot])))
    (is (= (:after first-binding) (get-in second-binding [:before 'a])))
    (let [call (first (filter #(= 'raster.compiler.backend.jvm.binding-context-owner-test/response
                                  (:raster.op/original (meta %)))
                             (vals (:invocation-catalog emission))))
          context (bytecode/retained-invocation-context emission call)]
      (is (= (:after second-binding) (get-in context [:inputs 0 :ports 'a]))))))

(deftest new-emission-does-not-reuse-lexical-occurrence-authority
  (let [first-owner (bytecode/retain-invocation-occurrences '[(let* [x 1.0] x)])
        second-owner (bytecode/retain-invocation-occurrences (:body first-owner))]
    (is (= (:body first-owner) (:body second-owner)))
    (is (empty? (set/intersection (set (keys (:binding-catalog first-owner)))
                                         (set (keys (:binding-catalog second-owner))))))))

(deftest binding-layout-is-part-of-the-occurrence-proof
  (let [emission (emitted #'inferred-binding)
        scope (first (:body emission))
        record (bytecode/retained-binding-context emission scope 0)
        key (first (keys (:binding-contexts emission)))]
    (is (some? record))
    (is (= 'power (:binder record)))
    (is (= 1 (:binding-count record)))
    (is (nil? (bytecode/retained-binding-context
               (assoc-in emission [:binding-contexts key :binding-count] 2) scope 0)))
    (is (nil? (bytecode/retained-binding-context
               (assoc-in emission [:binding-contexts key :binder] 'different) scope 0)))))

(deftest reduction-observations-retain-the-actual-loop-and-backedge-carriers
  (let [emission (emitted #'weighted-reduction)
        source (first (vals (:reduction-catalog emission)))
        {:keys [expansion bound loop recur exit] :as observed}
        (bytecode/retained-reduction-emission emission source)]
    (is (some? observed))
    (is (identical? source (:source expansion)))
    (is (= [:int :double] (mapv #(get-in loop [:locals % :type]) (:variables loop))))
    (is (= [:int :double] (mapv #(get-in % [:after :type]) (:initializers loop))))
    (is (= [:int :double] (mapv :type (:values recur))))
    (is (= {:object-result? false :object-call-result? false
            :preserve-source-call-results? true :void-context false}
           (:context bound)))
    (is (every? #(= (:context bound) (:context %)) (:initializers loop)))
    (is (identical? (:label-id loop) (:label-id recur)))
    (is (= :double (get-in exit [:local :type])))
    (is (= :ref (:result-type exit)))
    (is (= (second (:variables loop)) (:value exit)))
    (is (some #(identical? (:expression exit) %) (:body loop)))
    (is (identical? (nth (:expression exit) 2) (:expression recur)))
    (is (= (:after bound) (get-in loop [:outer-locals (:binder bound)])))
    (is (= :ref (get-in loop [:outer-locals 'gain :type])))
    (is (identical? Object (get-in loop [:outer-locals 'gain :reference-class])))
    (is (not (contains? emission :reduction-labels)))
    (is (nil? (bytecode/retained-reduction-emission
               emission (with-meta (apply list source) (meta source)))))
    (is (nil? (bytecode/retained-reduction-emission
               (assoc-in emission [:reduction-contexts [(:witness expansion) :recur] :label-id] (Object.))
               source)))
    (is (nil? (bytecode/retained-reduction-emission
               (assoc-in emission [:reduction-contexts [(:witness expansion) :exit] :local :slot] -1)
               source)))
    (is (nil? (bytecode/retained-reduction-emission
               (update-in emission [:reduction-contexts [(:witness expansion) :exit] :expression]
                          #(with-meta (apply list %) (meta %))) source)))
    (is (nil? (bytecode/retained-reduction-emission
               (update emission :reduction-contexts dissoc [(:witness expansion) :exit]) source)))))

(deftest reduction-observations-do-not-assume-double-carries
  (let [emission (emitted #'float-reduced)
        source (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission source)]
    (is (some? observed))
    (is (= [:int :float] (mapv #(get-in observed [:loop :locals % :type])
                              (get-in observed [:loop :variables]))))))

(deftest reduction-root-context-is-observed-at-the-whole-region
  (let [emission (emitted #'root-context-reduction)
        source (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission source)
        root (:root-result observed)
        path [:reduction-contexts [(get-in observed [:expansion :witness]) :bound-scope] :root-result]]
    (is (some? root))
    (is (= :ref (:result-type root)))
    ;; This real quotient distinguishes the premise the consumer needs: the
    ;; loop emitter strengthens its inner context, but the root did not start
    ;; with that flag. Copying the exit's context would be wrong.
    (is (false? (get-in root [:context :preserve-source-call-results?])))
    (is (true? (get-in observed [:exit :context :preserve-source-call-results?])))
    (is (= #{:object-result? :object-call-result? :preserve-source-call-results? :void-context}
           (set (keys (:context root)))))
    (is (every? boolean? (vals (:context root))))
    (is (some #(identical? (get-in observed [:loop :expression]) %) (:body root)))
    (is (not (contains? root :result-fact)))
    ;; Exit facts alone cannot manufacture root evidence. Equal copied regions
    ;; and a mismatched physical result likewise fail the root relation.
    (doseq [changed [(update-in emission (pop path) dissoc :root-result)
                     (assoc-in emission (conj path :body)
                               (mapv #(with-meta (apply list %) (meta %)) (:body root)))
                     (assoc-in emission (conj path :result-type) :float)
                     (update-in emission (conj path :context) dissoc :void-context)
                     (assoc-in emission (conj path :context :void-context) nil)
                     (assoc-in emission (conj path :context :invented) false)]]
      (let [result (bytecode/retained-reduction-emission changed source)]
        (is (some? result))
        (is (nil? (:root-result result)))))
    (is (= 2.0 (root-context-reduction (float-array [1.0 3.0]) 2)))))

(deftest observed-root-context-replays-original-count-and-initializer-effects
  (let [emission (emitted #'initialized-reduction)
        source (first (vals (:reduction-catalog emission)))
        root (:root-result (bytecode/retained-reduction-emission emission source))
        contract (completion/bytecode-result-context-contract (:context root) (:result-type root))
        artifact (bytecode/compile-specialized-class!
                   "raster.compiled.ReductionRootContext"
                   [{:name 'run :params [(with-meta 'xs {:tag 'doubles})
                                        (with-meta 'events {:tag 'longs})
                                        (with-meta 'cnt {:tag 'long})]
                     :return-tag 'Object
                     :walked-body [(completion/complete-value contract source)]
                     :source-ns (the-ns 'raster.compiler.backend.jvm.binding-context-owner-test)}])
        method (get-in artifact [:methods 'run])
        replay (fn [xs events cnt]
                 (try (.invoke ^java.lang.reflect.Method method nil (object-array [xs events cnt]))
                      (catch java.lang.reflect.InvocationTargetException e (throw (.getCause e)))))
        outcome (fn [f xs cnt]
                  (let [events (long-array [0])]
                    [(try [:value (Double/doubleToRawLongBits (double (f xs events cnt)))]
                          (catch Exception e [:error (class e)]))
                     (vec events)]))]
    (is (some? root))
    (doseq [xs [(double-array [3.0]) (double-array 0)]
            count [-1 0 2 Long/MAX_VALUE]]
      (is (= (outcome initialized-reduction xs count) (outcome replay xs count))))))

(deftest reduction-completion-retains-the-object-carrier-and-exact-fp-payload
  (doseq [[function payload] [[#'weighted-reduction Double] [#'float-reduced Float]]]
    (let [emission (emitted function)
          source (first (vals (:reduction-catalog emission)))
          observed (bytecode/retained-reduction-emission emission source)
          witness (get-in observed [:expansion :witness])
          exit-path [:reduction-contexts [witness :exit]]]
      (is (= {:carrier Object :presence :present :builtin-class payload}
             (:result-fact observed)))
      ;; These are malformed/copy observations, not alternative source routes.
      ;; Neither a logical annotation nor an impossible boxing path creates facts.
      (doseq [changed [(assoc-in emission (conj exit-path :result-type) :double)
                       (assoc-in emission (conj exit-path :input-type) :long)
                       (-> emission
                           (assoc-in (conj exit-path :input-type) :ref)
                           (assoc-in (conj exit-path :context :object-result?) false))]]
        (is (nil? (:result-fact (bytecode/retained-reduction-emission changed source)))))
      (is (nil? (:result-fact
                 (bytecode/retained-reduction-emission
                  emission (with-meta (apply list source) (meta source))))))))
  (doseq [count [0 1 3]]
    (is (instance? Double (weighted-reduction (double-array [1.0 2.0 3.0]) 2.0 count)))
    (is (instance? Float (float-reduced (float-array [1.0 2.0 3.0]) count)))))

(deftest unchanged-mse-count-outlines-only-the-original-prediction-length
  (let [source-var (core/resolve-deftm-var #'loss/mse-loss {:dtype :float})
        receiver (ns-resolve (.ns source-var) (symbol (str (:name (meta source-var)) "-impl")))
        _ (core/ensure-bytecode-implementation! receiver)
        original (:raster.core/bytecode-emission (meta receiver))
        interface (types/ensure-fn-interface! (:tags original))
        ;; Emit the unchanged stdlib body, not a hand-written scalar-count
        ;; surrogate. A fresh module also avoids lending new context fields to
        ;; an older implementation already resident in a development REPL.
        {:keys [emission]}
        (#'core/emit-owned-bytecode!
         bytecode/compile-typed-impl! "raster.compiled.MseCountComponent"
         (:params original) (:tags original) (:return-tag original)
         (:body original) (:source-ns original) (:iface-name interface) 'mse-count-component)
        source (first (vals (:reduction-catalog emission)))
        region (bytecode/compile-reduction-count-region emission source)]
    (is (= :owned-bytecode-reduction-count-region (:kind region)))
    (is (= '[pred] (:parameters region))
        "neither target nor the mean denominator is a count dependency")
    (is (= Integer/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
    (doseq [size [0 1 17]]
      (is (= size (invoke-region region {'pred (float-array size)}))))
    (is (thrown? NullPointerException (invoke-region region {'pred nil})))
    (is (nil? (bytecode/retained-reduction-emission original source))
        "the fresh module does not authenticate an older emission's source")))

(deftest mse-source-completion-does-not-confuse-the-float-checkpoint-and-double-quotient
  (let [source-var (core/resolve-deftm-var #'loss/mse-loss {:dtype :float})
        receiver (ns-resolve (.ns source-var) (symbol (str (:name (meta source-var)) "-impl")))
        _ (core/ensure-bytecode-implementation! receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        source (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission source)]
    (is (some? observed))
    (is (= {:carrier Object :presence :present :builtin-class Float}
           (:result-fact observed)))
    (is (instance? Double (loss/mse-loss (float-array [4096.0 1.0 1.0 1.0]) (float-array 4) 4)))
    (is (Double/isNaN (double (loss/mse-loss (float-array 0) (float-array 0) 0))))
    ;; Completion is conditional: bad reads must still fail in source order.
    (is (thrown? ArrayIndexOutOfBoundsException
                 (loss/mse-loss (float-array [1.0]) (float-array 0) 1)))))

(deftest reduction-completion-keeps-the-historical-definition-guard
  (let [emission (emitted #'weighted-reduction)
        source (first (vals (:reduction-catalog emission)))
        dependency (:var (first (:dependency-snapshots emission)))
        metadata (meta dependency)]
    (is (var? dependency))
    (try
      (alter-meta! dependency assoc ::completion-probe (Object.))
      (is (thrown? clojure.lang.ExceptionInfo
                   (bytecode/retained-reduction-emission emission source)))
      (finally (reset-meta! dependency metadata)))
    (is (= {:carrier Object :presence :present :builtin-class Double}
           (:result-fact (bytecode/retained-reduction-emission emission source))))))

(deftest observed-reduction-retains-count-before-initializer-order
  (doseq [cnt [-1 0 2]]
    (let [events (long-array [0])]
      (is (= (+ 3.0 (max 0 cnt)) (initialized-reduction (double-array [3.0]) events cnt)))
      (is (= [1] (vec events)))))
  (let [events (long-array [0])
        expected (try (int Long/MAX_VALUE) nil (catch Exception error (class error)))
        actual (try (initialized-reduction (double-array 0) events Long/MAX_VALUE) nil
                    (catch Exception error (class error)))]
    (is (= expected actual))
    (is (= [0] (vec events)))))

(deftest reduction-input-edge-requires-the-actual-emitted-region
  (let [emission (emitted #'reduction-input)
        source (first (vals (:reduction-catalog emission)))
        parent (some (fn [call]
                       (when (some #(identical? source %) (util/call-args call)) call))
                     (vals (:invocation-catalog emission)))
        input-index (first (keep-indexed #(when (identical? source %2) %1)
                                         (util/call-args parent)))
        edge (bytecode/retained-input-source-context emission parent input-index)
        token (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta parent))
        actual (:operand edge)]
    (is (some? parent))
    (is (some? edge))
    (is (identical? source (:source-original edge)))
    (is (identical? actual (get-in (bytecode/retained-reduction-emission emission source)
                                  [:bound :expression])))
    (is (nil? (bytecode/retained-input-source-context
               (assoc-in emission [:invocation-contexts token :inputs input-index :operand]
                         (with-meta (apply list actual) (meta actual)))
               parent input-index)))))
