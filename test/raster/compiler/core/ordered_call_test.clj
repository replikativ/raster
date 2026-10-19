(ns raster.compiler.core.ordered-call-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.ordered-call :as ordered]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion])
  (:import [java.lang.reflect InvocationHandler Method Proxy]))

(def ^:dynamic *target* nil)
(def policy {:unchecked-math false})

(deftest qualified-method-source-cannot-be-redirected-through-a-namespace-alias
  (let [source (create-ns (gensym "raster.test.method_source_"))
        shadow (create-ns (gensym "raster.test.method_shadow_"))
        method (.getMethod java.util.Objects "isNull" (into-array Class [Object]))
        head (with-meta 'java.util.Objects/isNull {:param-tags [Object]})]
    (try
      (intern shadow 'isNull (fn [_] :shadowed))
      (binding [*ns* source]
        (alias 'java.util.Objects (ns-name shadow))
        (is (= :shadowed ((eval (list 'fn* '[x] (list head 'x))) nil))
            "the actual source compiler resolves the alias, not the claimed static Method")
        (is (= :source-invocation-namespace-shadow
               (try (ordered/verify-source-method! method) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (finally (remove-ns (ns-name source)) (remove-ns (ns-name shadow))))))

(defn- compile-source [params body source-policy]
  (binding [*unchecked-math* (:unchecked-math source-policy)]
    (eval (list 'fn* params body))))

(defn- compile-plan [params plan]
  (let [normalized (ordered/normalize-resolved-call plan gensym (set params))
        ports (:class-ports normalized)
        class-bindings (into {} (map (juxt :class :port)) ports)
        body (completion/project-to-jvm-source
              (list 'let* (:bindings normalized) (:invocation normalized))
              (:source-policy normalized) class-bindings)
        factory (compile-source (mapv :port ports)
                                (list 'fn* params body) (:source-policy normalized))]
    (apply factory (map :class ports))))

(defn- outcome [f input]
  (let [events (atom [])]
    (try {:value (f input events) :events @events}
         (catch Throwable error {:exception (class error) :events @events}))))

(deftest normalized-call-exposes-completed-inputs-in-formal-order
  (let [method (.getMethod Math "pow" (into-array Class [Double/TYPE Double/TYPE]))
        normalized (ordered/normalize-resolved-call
                    (ordered/plan-resolved-call method nil ['x 'y] policy) gensym '#{x y})
        values (:input-values normalized)
        bindings (into {} (map vec (partition 2 (:bindings normalized))))]
    (is (= 2 (count values)))
    (is (= values (vec (rest (:invocation normalized)))))
    (is (= '[x y] (mapv #(-> (get bindings %) completion/canonical-parts :operand) values)))))

(defn- typed-fixture [tags]
  (let [entry (types/ensure-fn-interface! tags)
        carrier (:iface-class entry)
        method (first (filter #(= "invk" (.getName ^Method %)) (.getDeclaredMethods ^Class carrier)))
        target (Proxy/newProxyInstance
                (.getClassLoader ^Class carrier) (into-array Class [carrier])
                (reify InvocationHandler
                  (invoke [_ _ _ args] (aget ^objects args 1))))]
    {:class carrier :method method :target target :hint (:iface-name entry)}))

(deftest typed-input-completion-happens-before-later-arguments
  (doseq [[tags input expected]
          [[['double 'double] nil {:exception NullPointerException :events []}]
           [['doubles 'double] 42 {:exception ClassCastException :events []}]
           [['doubles 'double] nil {:value 2.0 :events [:later]}]
           [['double 'double] 3.0 {:value 2.0 :events [:later]}]]]
    (let [{:keys [method target hint]} (typed-fixture tags)
          receiver (with-meta 'target {:tag hint})
          arguments '[x (do (swap! events conj :later) 2.0)]
          source (list* '.invk receiver arguments)
          plan (ordered/plan-resolved-call method receiver arguments policy)
          original (partial (compile-source [receiver 'x 'events] source policy) target)
          transformed (partial (compile-plan [receiver 'x 'events] plan) target)]
      (is (= expected (outcome original input)))
      (is (= (outcome original input) (outcome transformed input))))))

(deftest host-conversion-happens-before-later-arguments
  (let [method (.getMethod Math "pow" (into-array Class [Double/TYPE Double/TYPE]))
        arguments '[x (do (swap! events conj :later) 2.0)]
        original (compile-source '[x events] (list* 'Math/pow arguments) policy)
        transformed (compile-plan '[x events] (ordered/plan-resolved-call method nil arguments policy))]
    (is (= {:exception NullPointerException :events []} (outcome original nil)))
    (is (= {:value 9.0 :events [:later]} (outcome original 3.0)))
    (doseq [input [nil 3.0]]
      (is (= (outcome original input) (outcome transformed input))))))

(deftest runtime-target-selection-precedes-arguments
  (let [method (.getMethod clojure.lang.IFn "invoke" (into-array Class [Object]))
        receiver 'raster.compiler.core.ordered-call-test/*target*
        arguments '[(do (set! raster.compiler.core.ordered-call-test/*target*
                              (fn* [x] [:after x])) 7)]
        original (compile-source [] (list* receiver arguments) policy)
        transformed (compile-plan [] (ordered/plan-resolved-call method receiver arguments policy))
        observe (fn [f] (binding [*target* (fn [x] [:before x])] (f)))]
    (is (= [:before 7] (observe original)))
    (is (= (observe original) (observe transformed)))))

(deftest invalid-receiver-and-nil-invocation-keep-distinct-failure-points
  (let [method (.getMethod clojure.lang.IFn "invoke" (into-array Class [Object]))
        arguments '[(do (swap! events conj :later) 7)]
        original (compile-source '[target events] (list* 'target arguments) policy)
        transformed (compile-plan '[target events] (ordered/plan-resolved-call method 'target arguments policy))]
    (is (= {:exception ClassCastException :events []} (outcome original 42)))
    (is (= {:exception NullPointerException :events [:later]} (outcome original nil)))
    (doseq [input [42 nil vector]]
      (is (= (outcome original input) (outcome transformed input))))))

(deftest object-inputs-keep-source-materialization-identity
  (let [method (.getMethod clojure.lang.IFn "invoke" (into-array Class [Object Object]))]
    (doseq [value [1000 5.125]]
      (let [receiver (with-meta 'target {:tag 'clojure.lang.IFn})
            arguments [value '(do nil)]
            source (list* receiver arguments)
            original (compile-source [receiver] (list 'identical? (list 'first source) value) policy)
            plan (ordered/plan-resolved-call method receiver arguments policy)
            normalized (ordered/normalize-resolved-call plan gensym #{receiver})
            ports (:class-ports normalized)
            body (completion/project-to-jvm-source
                  (list 'let* (:bindings normalized)
                        (list 'identical? (list 'first (:invocation normalized)) value))
                  policy (into {} (map (juxt :class :port)) ports))
            factory (compile-source (mapv :port ports) (list 'fn* [receiver] body) policy)
            transformed (apply factory (map :class ports))]
        (is (true? (original vector)))
        (is (= (original vector) (transformed vector)))))))

(deftest missing-descriptor-policy-and-unsupported-reference-adaptation-refuse
  (let [method (.getMethod Math "pow" (into-array Class [Double/TYPE Double/TYPE]))]
    (is (thrown? clojure.lang.ExceptionInfo (ordered/plan-resolved-call nil nil '[x y] policy)))
    (is (thrown? clojure.lang.ExceptionInfo (ordered/plan-resolved-call method nil '[x y] nil)))
    (is (thrown? clojure.lang.ExceptionInfo (ordered/plan-resolved-call method nil '[x] policy)))
    (let [plan (ordered/plan-resolved-call method nil '[x y] policy)]
      (is (thrown? clojure.lang.ExceptionInfo
                   (ordered/normalize-resolved-call (assoc plan :input-contracts []) gensym #{})))))
  ;; Reference parameters with adaptation are NOT blanket Class.cast inputs.
  (let [method (.getMethod java.util.concurrent.CompletableFuture "supplyAsync"
                           (into-array Class [java.util.function.Supplier]))]
    (is (= :unclosed-call-input-conversion
           (try (ordered/plan-resolved-call method nil '[f] policy) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest normalization-rejects-capturing-and-repeated-identifiers
  (let [method (.getMethod Math "pow" (into-array Class [Double/TYPE Double/TYPE]))
        plan (ordered/plan-resolved-call method nil '[x (let* [nested 2.0] nested)] policy)]
    (doseq [id ['x 'nested 'unused-owner-parameter 'elsewhere/value 42]]
      (is (= :nonfresh-ordered-call-identifier
             (try (ordered/normalize-resolved-call plan (constantly id) #{'unused-owner-parameter}) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (= :nonfresh-ordered-call-identifier
           (try (ordered/normalize-resolved-call plan (constantly 'same) #{}) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (let [normalized (ordered/normalize-resolved-call
                      plan (fn [prefix] (with-meta (gensym prefix) {:tag 'double :ownership :bogus})) #{})]
      (is (every? (comp nil? meta) (take-nth 2 (:bindings normalized)))))
    (is (= :unclosed-ordered-call-scope
           (try (ordered/normalize-resolved-call plan gensym nil) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
