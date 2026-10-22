(ns raster.compiler.backend.jvm.reference-return-bridge-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.dl.loss]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(defonce body-events (atom []))

(defn reference-sibling [value]
  (swap! body-events conj :unrelated-global)
  value)

(defn- fixture
  ([return-tag] (fixture ['Object] return-tag '[value]))
  ([tags return-tag body]
   (let [interface (types/ensure-fn-interface! tags)]
     (#'core/emit-owned-bytecode!
      bytecode/compile-typed-impl! "raster.compiled.ReferenceReturnBridge"
      '[value] tags return-tag body *ns* (:iface-name interface) 'reference-return))))

(defn- invoke [implementation value]
  (clojure.lang.Reflector/invokeInstanceMethod implementation "invk" (object-array [value])))

(defn- failure [run]
  (try (run) :accepted
       (catch java.lang.reflect.InvocationTargetException e (class (.getCause e)))
       (catch Throwable e (class e))))

(defn- training-mse-receiver []
  ;; Request the public specialization without invoking the numeric function.
  ;; Looking up a mangled Var alone accidentally depends on earlier warm tests.
  (let [source (core/resolve-deftm-var #'raster.dl.loss/mse-loss {:dtype :float})]
    (ns-resolve (.ns ^clojure.lang.Var source)
                (symbol (str (:name (meta source)) "-impl")))))

(deftest actual-reference-wrapper-preserves-identity-including-nil
  (doseq [[tag carrier value]
          [['Object Object (Object.)]
           ['java.lang.String String (String. "reference")]
           ['floats (class (float-array 0)) (float-array [1.0 2.0])]
           ['doubles (class (double-array 0)) (double-array [1.0 2.0])]
           ['ints (class (int-array 0)) (int-array [1 2])]]]
    (let [{:keys [implementation emission]} (fixture tag)
          bridge (bytecode/retained-reference-return-bridge emission)]
      (is (= :reference-static-return (:kind bridge)))
      (is (identical? carrier (:body-return-carrier bridge)))
      (is (identical? Object (:result-carrier bridge)))
      (is (= :ref (get-in bridge [:body-context :emitted-result-type])))
      (is (completion/bytecode-operand-context? (get-in bridge [:body-context :result-context])))
      (is (nil? (bytecode/retained-typed-return-bridge emission))
          "reference pass-through is not primitive payload correspondence")
      (is (not-any? #(contains? bridge %) [:executor :non-null? :payload-carrier]))
      (doseq [input [nil value]]
        (is (identical? input (invoke implementation input)))
        (is (identical? input (.invoke ^java.lang.reflect.Method (:compute-method bridge)
                                      nil (object-array [input]))))))))

(deftest primitive-array-body-return-keeps-its-exact-wrapper-and-body-context
  (doseq [tag '[floats doubles ints longs bytes shorts booleans chars]]
    (let [{:keys [emission]} (fixture [tag] 'double '[1.25])
          bridge (bytecode/retained-primitive-body-return-bridge emission)
          saved (first (:typed-return-bridges emission))]
      (is (= :boxed-primitive-static-return (:kind bridge)))
      (is (identical? Double/TYPE (:payload-carrier bridge)))
      (is (= (:parameter-carriers bridge) (:compute-parameter-carriers bridge)))
      (is (= :double (get-in bridge [:body-context :emitted-result-type])))
      (is (nil? (bytecode/retained-typed-return-bridge emission))
          "A primitive body result does not make an array consumer a scalar leaf")
      (doseq [changed [(assoc emission :body (vec (map identity (:body emission))))
                       (assoc emission :typed-return-bridges [(assoc saved :parameter-carriers [Double/TYPE])])
                       (assoc emission :typed-return-bridges [(assoc saved :payload-carrier Float/TYPE)])
                       (assoc emission :typed-return-bridges [(assoc saved :compute-method nil)])]]
        (is (nil? (bytecode/retained-primitive-body-return-bridge changed))))))
  (doseq [tag '[Object objects java.lang.String]]
    (let [{:keys [emission]} (fixture [tag] 'double '[1.25])]
      (is (nil? (bytecode/retained-primitive-body-return-bridge emission))
          "Arbitrary Object/reference adaptations are outside this boundary"))))

(deftest unchanged-training-mse-body-replay-matches-the-actual-helper
  ;; Exercise the real training helper, including its reduction and array reads.
  ;; This verifies physical replay only, not GPU placement/source-import closure.
  (entry/call-with-definition-inputs
   (fn []
     (let [receiver (training-mse-receiver)
           _ (core/ensure-current-owned-bytecode! receiver)
           emission (:raster.core/bytecode-emission (meta receiver))
           bridge (bytecode/retained-primitive-body-return-bridge emission)
           _ (is (some? bridge))
           context (:body-context bridge)
           result-contract (completion/bytecode-result-context-contract
                            (:result-context context) (:emitted-result-type context))
           body (if (= 1 (count (:body emission))) (first (:body emission))
                    (list* 'do (:body emission)))
           replay (completion/complete-value
                   completion/object-input-contract
                   (completion/complete-value result-contract body))
           params (mapv (fn [parameter tag] (with-meta parameter (assoc (meta parameter) :tag tag)))
                        (:params emission) (:tags emission))
           {:keys [methods]} (bytecode/compile-specialized-class!
                              "raster.compiled.TrainingMseBodyReplay"
                              [{:name 'run :params params :return-tag 'Object
                                :source-ns (:source-ns emission) :walked-body [replay]}])
           observe (fn [^java.lang.reflect.Method method target arguments]
                     (try
                       [:value (Double/doubleToRawLongBits
                                (.invoke method target (object-array arguments))) ]
                       (catch java.lang.reflect.InvocationTargetException error
                         [:error (class (.getCause error))])))]
       (doseq [[pred target n] [[(float-array [1.0 2.0]) (float-array [0.0 1.0]) 2]
                                [(float-array [-1.5 2.25 0.5]) (float-array [0.0 -1.0 0.75]) 3]
                                [(float-array []) (float-array []) 1]
                                [(float-array [1.0]) (float-array [0.0]) 0]
                                [(float-array [1.0 2.0]) (float-array [0.0]) 2]
                                [nil (float-array [0.0]) 1]]]
         (let [arguments [pred target n]]
           (is (= (observe (:invocation-method bridge) (:implementation emission) arguments)
                  (observe (get methods 'run) nil arguments)))))))))

(deftest primitive-body-frontend-observation-retains-history-not-copied-evidence
  (entry/call-with-definition-inputs
   (fn []
     (let [receiver (training-mse-receiver)
           _ (core/ensure-current-owned-bytecode! receiver)
           emission (:raster.core/bytecode-emission (meta receiver))
           source (get-in emission [:source-entry :source-var])
           frontend (entry/original-source-frontend source)
           parent (atom {:static [] :values []})
           before @parent]
       (binding [entry/*emission-inputs* parent]
         (doseq [changed [(assoc frontend :emission (into {} emission))
                          (assoc-in frontend [:emission :implementation] (Object.))
                          (assoc-in frontend [:emission :typed-return-bridges] [])]]
           (is (nil? (entry/frontend-primitive-body-return-bridge changed)))
           (is (identical? before @parent)))
         (is (some? (entry/frontend-primitive-body-return-bridge frontend)))
         (is (= 1 (count (:producer-inputs @parent))))
         (is (satisfies? entry/RetainedProducerInput (first (:producer-inputs @parent)))))
       (let [dependency (:var (first (:dependency-snapshots emission)))
             metadata (meta dependency)]
         (is (var? dependency))
         (is (entry/assert-definition-inputs-current! @parent))
         (try
           (alter-meta! dependency assoc ::changed-after-body-observation true)
           (is (thrown? clojure.lang.ExceptionInfo (entry/assert-definition-inputs-current! @parent)))
           (finally (reset-meta! dependency metadata)))
         (is (entry/assert-definition-inputs-current! @parent)))))))

(deftest concrete-static-return-check-remains-an-obligation
  (let [{:keys [implementation emission]} (fixture 'floats)
        bridge (bytecode/retained-reference-return-bridge emission)
        carrier (:body-return-carrier bridge)]
    (is (= "[F" (.getName ^Class carrier)))
    (is (= ClassCastException (failure #(invoke implementation "not a float array"))))
    (is (= ClassCastException
           (failure #(.invoke ^java.lang.reflect.Method (:compute-method bridge)
                              nil (object-array ["not a float array"])))))
    (is (nil? (invoke implementation nil)) "CHECKCAST does not imply non-nullness")))

(deftest whole-body-context-retains-original-method-local-callee-denial
  (let [source-ns (the-ns 'raster.compiler.backend.jvm.reference-return-bridge-test)
        interface (types/ensure-fn-interface! '[Object])
        sibling {:name 'reference-sibling :params [(with-meta 'value {:tag 'Object})]
                 :return-tag 'Object :walked-body '[nil] :source-ns source-ns}
        emitter (fn [& args] (apply bytecode/compile-typed-impl! (concat args [[sibling]])))
        {:keys [implementation emission]}
        (#'core/emit-owned-bytecode!
         emitter "raster.compiled.ReferenceSiblingBody" '[value] '[Object] 'Object
         '[(reference-sibling value)] source-ns (:iface-name interface) 'reference-sibling-body)
        bridge (bytecode/retained-reference-return-bridge emission)
        value (Object.)
        before @body-events]
    (is (some? bridge) "the reference wrapper edge itself remains valid")
    (is (true? (get-in bridge [:body-context :unretained-sibling-calls?])))
    (is (nil? (invoke implementation value)) "original call selects its same-class sibling")
    (is (= before @body-events) "neither query nor original invokes the colliding global"))
  (is (false? (get-in (bytecode/retained-reference-return-bridge (:emission (fixture 'Object)))
                     [:body-context :unretained-sibling-calls?]))))

(deftest missing-foreign-and-mutated-return-observations-abstain
  (let [{:keys [emission]} (fixture 'Object)
        bridge (first (:typed-return-bridges emission))
        foreign (first (:typed-return-bridges (:emission (fixture 'Object))))
        token (:raster.compiler.backend.jvm.bytecode/body-occurrence (meta (:body emission)))]
    (is (some? (bytecode/retained-reference-return-bridge emission)))
    (is (not= (:compute-method bridge) (:compute-method foreign)))
    (is (= (vec (.getParameterTypes ^java.lang.reflect.Method (:compute-method bridge)))
           (vec (.getParameterTypes ^java.lang.reflect.Method (:compute-method foreign)))))
    (is (= (.getReturnType ^java.lang.reflect.Method (:compute-method bridge))
           (.getReturnType ^java.lang.reflect.Method (:compute-method foreign))))
    (doseq [changed [(dissoc emission :typed-return-bridges)
                     (assoc emission :implementation (Object.))
                     (assoc emission :body (with-meta (mapv identity (:body emission))
                                                     (meta (:body emission))))
                     (update emission :typed-return-bridges #(into % %))
                     (assoc emission :typed-return-bridges [(assoc bridge :kind :boxed-primitive-static-return)])
                     (assoc emission :typed-return-bridges [(assoc bridge :body-return-carrier String)])
                     (assoc emission :typed-return-bridges [(assoc bridge :result-carrier String)])
                     (assoc emission :typed-return-bridges [(assoc bridge :parameter-carriers [Long/TYPE])])
                     (assoc emission :typed-return-bridges [(assoc bridge :compute-method nil)])
                     (assoc emission :typed-return-bridges [(assoc bridge :compute-method (:compute-method foreign))])
                     (assoc emission :typed-return-bridges [(assoc bridge :invocation-method nil)])
                     (assoc-in emission [:body-contexts token :method] nil)
                     (assoc-in emission [:body-contexts token :emitted-result-type] :long)
                     (assoc-in emission [:body-contexts token :return-descriptor] "Ljava/lang/String;")]]
      (is (nil? (bytecode/retained-reference-return-bridge changed))))))

(deftest queries-do-not-execute-or-prepare-the-body
  (reset! body-events [])
  (let [{:keys [implementation emission]}
        (fixture ['Object] 'Object
                 '[(do (clojure.core/swap!
                        raster.compiler.backend.jvm.reference-return-bridge-test/body-events
                        clojure.core/conj :body) value)])]
    (with-redefs [core/ensure-bytecode-implementation!
                  (fn [& _] (throw (AssertionError. "return query must not compile")))]
      (is (some? (bytecode/retained-reference-return-bridge emission)))
      (is (empty? @body-events)))
    (is (nil? (invoke implementation nil)))
    (is (= [:body] @body-events))))

(deftest primitive-boxing-and-guarded-wrapper-edges-are-not-reference-recipes
  (let [{:keys [emission]} (fixture ['long] 'long '[value])]
    (is (some? (bytecode/retained-typed-return-bridge emission)))
    (is (nil? (bytecode/retained-reference-return-bridge emission))))
  (let [{:keys [emission]} (fixture ['long] 'Object '[value])]
    (is (= :reference-static-return (:kind (first (:typed-return-bridges emission)))))
    (is (true? (get-in (bytecode/retained-reference-return-bridge emission)
                      [:body-context :result-context :object-result?]))
        "replaying this whole body must preserve its original boxing context")
    (is (nil? (bytecode/retained-typed-return-bridge emission))
        "a boxed body is not an independent primitive payload recipe"))
  (let [callback (:iface-name (types/ensure-fn-interface! ['long]))
        {:keys [emission]} (fixture [callback] 'Object '[value])]
    (is (nil? (bytecode/retained-reference-return-bridge emission))
        "an instanceof/fallback wrapper is not unguarded pass-through")))
