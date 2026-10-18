(ns raster.compiler.backend.jvm.typed-entrypoint-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.types :as types]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]))

(defn- compile-entry [params tags return-tag body]
  (let [{:keys [iface-name]} (types/ensure-fn-interface! tags)]
    (bytecode/compile-typed-entrypoint!
     (str "raster.compiled.Entrypoint" (clojure.lang.RT/nextID))
     params tags return-tag body *ns* iface-name)))

(deftest actual-entrypoints-retain-the-emitted-primitive-boxing-recipe
  (doseq [[tag value primitive wrapper]
          [['double (double -0.0) Double/TYPE Double]
           ['float (float -0.0) Float/TYPE Float]
           ['long (long 1000) Long/TYPE Long]
           ['int (int 1000) Integer/TYPE Integer]
           ['boolean true Boolean/TYPE Boolean]
           ;; Existing int-stack wrapper behavior, not a claim that the source
           ;; Byte/Short/Char boxing semantics have been repaired.
           ['byte (byte 7) Byte/TYPE Integer]
           ['short (short 7) Short/TYPE Integer]
           ['char (char 7) Character/TYPE Integer]]]
    (let [{:keys [implementation invocation-method compute-method return-completion] :as entry}
          (compile-entry ['x] [tag] tag ['x])
          result (implementation value)]
      (is (identical? entry (method-entry/assert-typed-entrypoint! entry implementation)))
      (is (identical? (class implementation) (.getDeclaringClass invocation-method)))
      (is (identical? primitive (.getReturnType compute-method)))
      (is (= Object (.getReturnType invocation-method)))
      (is (= :boxed (:kind return-completion)))
      (is (= :present (:presence return-completion)))
      (is (identical? primitive (:primitive-carrier return-completion)))
      (is (identical? wrapper (:object-carrier return-completion)))
      (is (identical? wrapper (class result)))
      (when (= tag 'double)
        (is (= (Double/doubleToRawLongBits value) (Double/doubleToRawLongBits result))))
      (when (= tag 'float)
        (is (= (Float/floatToRawIntBits value) (Float/floatToRawIntBits result)))))))

(deftest reference-and-void-results-do-not-acquire-false-presence
  (let [{:keys [implementation return-completion] :as entry}
        (compile-entry ['x] ['Object] 'Object ['x])
        token (Object.)]
    (is (identical? entry (method-entry/assert-typed-entrypoint! entry implementation)))
    (is (= {:kind :reference :source-carrier Object :presence :unknown} return-completion))
    (is (identical? token (implementation token)))
    (is (nil? (implementation nil))))
  (let [{:keys [implementation return-completion] :as entry}
        (compile-entry [] [] 'void ['(java.lang.Thread/onSpinWait)])]
    (is (identical? entry (method-entry/assert-typed-entrypoint! entry implementation)))
    (is (= {:kind :absent :primitive-carrier Void/TYPE :presence :absent} return-completion))
    (is (nil? (implementation)))))

(deftest guarded-invk-keeps-the-fallback-result-unknown
  (let [{inner-tag :iface-name} (types/ensure-fn-interface! ['double])
        {outer-interface :iface-name} (types/ensure-fn-interface! [inner-tag])
        fallback-name (gensym "entrypoint-fallback-")
        calls (atom [])
        fallback (intern *ns* fallback-name (fn [value] (swap! calls conj value) nil))]
    (try
      (let [{:keys [implementation invocation-method return-completion] :as entry}
            (bytecode/compile-typed-entrypoint!
             (str "raster.compiled.GuardedEntrypoint" (clojure.lang.RT/nextID))
             ['f] [inner-tag] 'double [1.0] *ns* outer-interface
             (str (ns-name *ns*)) (str fallback-name))
            typed (:implementation (compile-entry ['x] ['double] 'double ['x]))
            ordinary (fn [x] x)
            invoke (fn [value]
                     (.invoke ^java.lang.reflect.Method invocation-method implementation
                              (object-array [value])))]
        (is (identical? entry (method-entry/assert-typed-entrypoint! entry implementation)))
        (is (= :guarded (:kind return-completion)))
        (is (= :unknown (:presence return-completion)))
        (is (= [0] (:guarded-indices return-completion)))
        (is (= Double (get-in return-completion [:compute-return :object-carrier])))
        (is (= 1.0 (invoke typed)))
        (is (empty? @calls))
        (is (nil? (invoke ordinary)))
        (is (= 1 (count @calls)))
        (is (identical? ordinary (first @calls))))
      (finally (ns-unmap (.ns ^clojure.lang.Var fallback) fallback-name)))))

(deftest entrypoint-methods-and-receiver-must-have-the-same-actual-owner
  (let [[a b] (binding [method-entry/*compilation-classloader*
                       (clojure.lang.DynamicClassLoader. (clojure.lang.RT/baseLoader))]
                [(compile-entry ['x] ['double] 'double ['x])
                 (compile-entry ['x] ['float] 'float ['x])])
        input (:implementation a)
        reason (fn [entry impl]
                 (try (method-entry/assert-typed-entrypoint! entry impl) :accepted
                      (catch clojure.lang.ExceptionInfo e (:rule (ex-data e)))))]
    (is (= :typed-entrypoint (reason a (:implementation b))))
    (is (= :typed-entrypoint (reason (assoc a :invocation-method (:invocation-method b)) input)))
    (is (= :typed-entrypoint (reason (assoc a :compute-method (:compute-method b)) input)))
    (is (= :typed-entrypoint (reason (assoc-in a [:return-completion :primitive-carrier] Float/TYPE) input)))
    (is (= :typed-entrypoint (reason (assoc-in a [:return-completion :object-carrier] Float) input)))
    (let [foreign (binding [method-entry/*compilation-classloader*
                           (clojure.lang.DynamicClassLoader. (clojure.lang.RT/baseLoader))]
                    (compile-entry ['x] ['double] 'double ['x]))]
      (is (= :typed-entrypoint (reason (assoc a :compute-method (:compute-method foreign)) input))))))

(defn- with-owned-source [annotation expression check]
  (let [scope (create-ns (gensym "raster.entrypoint-source."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval (list 'raster.core/deftm 'owned ['x ':- annotation] ':- annotation expression)))
      (let [generic (ns-resolve scope 'owned)
            source (core/resolve-deftm-var generic)
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            _ (core/ensure-bytecode-implementation! receiver)
            operation (symbol (str (ns-name scope)) "owned")
            receiver-name (symbol (str (ns-name scope)) (str (:name (meta receiver))))
            call (with-meta (list '.invk receiver-name 'x) {:raster.op/original operation})]
        (check {:call call :source source :receiver receiver}))
      (finally (remove-ns (ns-name scope))))))

(deftest strict-scalar-consumer-uses-the-published-actual-return-contract
  (doseq [[annotation expression dtype carrier]
          [['Double '(clojure.core/* x x) :double Double]
           ['Float '(clojure.core/float (clojure.core/* x x)) :float Float]
           ;; Unlike the old nullable source-presence query, this proof starts
           ;; from an actual primitive input port. Returning that port is safe.
           ['Double 'x :double Double]]]
    (with-owned-source annotation expression
      (fn [{:keys [call source receiver]}]
        (is (nil? (scalar-body/checked-selected-object-result call))
            "an unowned inspection may not publish a proof from fresh observations")
        (let [inputs (atom {:static [] :values []})
              proof (binding [method-entry/*emission-inputs* inputs]
                      (scalar-body/checked-selected-object-result call))]
          (is (= :boxed-scalar (:kind proof)))
          (is (= dtype (:type proof)))
          (is (identical? carrier (:object-carrier proof)))
          (is (identical? @receiver (get-in proof [:owner :entrypoint :implementation])))
          (is (= 3 (count (:definition-proofs @inputs))))
          (is (true? (method-entry/assert-definition-inputs-current! @inputs)))
          (with-redefs-fn {source (fn [& _] :changed-source)}
            #(is (nil? (binding [method-entry/*emission-inputs* (atom {:static [] :values []})]
                         (scalar-body/checked-selected-object-result call)))))
          (with-redefs-fn {receiver (fn [& _] :changed-receiver)}
            #(is (nil? (binding [method-entry/*emission-inputs* (atom {:static [] :values []})]
                         (scalar-body/checked-selected-object-result call))))))))))

(deftest scalar-provenance-does-not-invent-a-primitive-region-from-a-return-annotation
  (doseq [[annotation expression]
          [['Object 'x]
           ['Double '(clojure.core/identity x)]
           ;; The emitter's primitive return conversion is real, but the
           ;; retained computation has a different width. Discharging it needs
           ;; an explicit conversion proof, not just the method's return tag.
           ['Double '(clojure.core/float x)]]]
    (with-owned-source annotation expression
      (fn [{:keys [call]}]
        (is (nil? (binding [method-entry/*emission-inputs* (atom {:static [] :values []})]
                    (scalar-body/checked-selected-object-result call)))
            "unknown/opaque bodies and implicit width changes remain unproved")))))

(deftest scalar-provenance-keeps-historical-inputs-after-the-query
  (with-owned-source 'Double '(clojure.core/* x x)
    (fn [{:keys [call source receiver]}]
      (let [inputs (atom {:static [] :values []})
            proof (binding [method-entry/*emission-inputs* inputs]
                    (scalar-body/checked-selected-object-result call))]
        (is (= :boxed-scalar (:kind proof)))
        (doseq [changed [source receiver]]
          (with-redefs-fn {changed (fn [& _] :changed-after-proof)}
            #(is (thrown? clojure.lang.ExceptionInfo
                          (method-entry/assert-definition-inputs-current! @inputs))
                 "a completed query cannot bless later replacement of its producer")))
        (is (true? (method-entry/assert-definition-inputs-current! @inputs)))))))
