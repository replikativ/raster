(ns raster.compiler.backend.jvm.generated-body-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.call-boundary :as boundary]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.compiler.passes.scalar.simplify :as simplify]
            [raster.compiler.fixtures.distributed-training :as training]))

(deftm unused-argument-helper [value :- (Array float) unused :- Double] :- (Array float)
  (aset value 0 (float (+ 1.0 (aget value 0))))
  value)

(deftm ordered-body-helper [state :- (Array float) x :- Double] :- (Array float)
  (let [alias state
        rounded (int x)
        negative? (< x 0.0)]
    (aset alias 0 (float rounded))
    (double x)
    (long rounded)
    (int rounded)
    false
    nil
    (when negative? (aset alias 1 (float -1.0)))
    alias))

(defn- interface-method [tags]
  (let [interface (types/ensure-fn-interface! tags)]
    (first (filter #(= "invk" (.getName ^java.lang.reflect.Method %))
                   (.getDeclaredMethods ^Class (:iface-class interface))))))

(defn- capture! [operation implementation arity method]
  (boundary/capture-selected-emission! operation implementation arity method
                                      core/ensure-current-owned-bytecode!))

(defn- emit [program]
  (let [interface (types/ensure-fn-interface! (:tags program))]
    (let [emitted (core/emit-owned-bytecode!
      bytecode/compile-typed-impl! "raster.compiled.GeneratedBodyTest"
      (:params program) (:tags program) 'Object (:walked-body program)
      (:source-ns program) (:iface-name interface) 'generated-body-test nil)]
      (generated/validate-program-emission! program (:emission emitted))
      ;; These controls are independently emitted derived programs, just like
      ;; the production generated-entry consumer. Close their actual callees
      ;; before invocation; a promotion must restart the surrounding test owner,
      ;; not invalidate an already prepared sibling on the first numeric call.
      (bytecode/prepare-emitted-global-invocations! (:emission emitted))
      (:implementation emitted))))

(defn- invoke [function arguments]
  (clojure.lang.Reflector/invokeInstanceMethod function "invk" (object-array arguments)))

(defn- helper-program [source]
  (let [operation (symbol (str (ns-name (.ns source))) (str (:name (meta source))))
        source (core/resolve-deftm-var source)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        _ (core/ensure-current-owned-bytecode! receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        receiver-symbol (with-meta (symbol (str (ns-name (.ns receiver))) (str (:name (meta receiver))))
                          {:tag (:typed-interface emission)})]
    {:params '[state x] :tags '[Object Object] :source-ns *ns*
     :walked-body [(list 'let* ['result (with-meta (list '.invk receiver-symbol 'state 'x)
                                         {:raster.op/original operation})]
                         'result)]}))

(deftest anchored-statement-spines-preserve-discard-context-and-local-facts
  (entry/call-with-definition-inputs
   (fn []
     (let [program (helper-program #'ordered-body-helper)
           method (interface-method (:tags program))
           normalized (generated/normalize-program program method capture!)
           original (emit program)
           compiled (emit normalized)
           pairs (partition 2 (second (first (:walked-body normalized))))]
       (is (some? normalized))
       (is (< 3 (count pairs)))
       (let [statements (filter (fn [[_ value]]
                                  (when-let [{:keys [contract operand]} (completion/canonical-parts value)]
                                    (and (= :jvm-bytecode-result-context (:source-context contract))
                                         (seq? operand) (= 'do (first operand))
                                         (= 3 (count operand)) (nil? (last operand))))) pairs)]
         (is (seq statements))
         (is (every? #(true? (:raster.effect/effectful (meta (first %)))) statements)
             "the producer carries the existing statement contract, not a guessed numerical dtype"))
       (doseq [x [2.75 -3.25 0.0]]
         (let [left (float-array [10.0 20.0]) right (aclone left)]
           (is (identical? left (invoke original [left x])))
           (is (identical? right (invoke compiled [right x])))
           (is (= (vec left) (vec right)))))
       (let [before @entry/*emission-inputs*]
         (with-redefs [bytecode/retained-terminal-context (fn [& _] nil)]
           (is (nil? (generated/normalize-program program method capture!))))
         (is (identical? before @entry/*emission-inputs*)))))))

(deftest generated-publication-rejects-missing-context-and-changed-local-facts
  (entry/call-with-definition-inputs
   (fn []
     (let [program (helper-program #'ordered-body-helper)
           normalized (generated/normalize-program program (interface-method (:tags program)) capture!)
           interface (types/ensure-fn-interface! (:tags normalized))
           emitted (core/emit-owned-bytecode!
                    bytecode/compile-typed-impl! "raster.compiled.GeneratedValidationTest"
                    (:params normalized) (:tags normalized) 'Object (:walked-body normalized)
                    (:source-ns normalized) (:iface-name interface) 'generated-validation nil)
           check (fn [] (try (generated/validate-program-emission! normalized (:emission emitted))
                            :accepted
                            (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
           observe bytecode/retained-input-binding-context]
       (is (= :accepted (check)))
       (with-redefs [bytecode/retained-input-completion-context (fn [& _] nil)]
         (is (= :generated-body-completion-context (check))))
       (doseq [[field value] [[:hint 'Object] [:verified :invalid] [:reference-class String]]]
         (with-redefs [bytecode/retained-input-binding-context
                       (fn [& arguments]
                         (when-let [context (apply observe arguments)]
                           (assoc-in context [:after field] value)))]
           (is (= :generated-body-local-facts (check)))))
       (let [completion-observation bytecode/retained-input-completion-context]
         (with-redefs [bytecode/retained-input-completion-context
                       (fn [& arguments]
                         (when-let [context (apply completion-observation arguments)]
                           (update-in context [:operand-context :void-context] not)))]
           (is (= :generated-body-completion-context (check)))))
       (is (= :generated-body-completion-context
              (try
                (generated/validate-program-emission!
                 (update normalized :walked-body #(with-meta (apply list %) nil)) (:emission emitted))
                :accepted
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(defn- observe [function arguments]
  (try
    (let [values (invoke function arguments)]
      {:value-bits (Double/doubleToRawLongBits (double (first values)))
       :gradient (vec (second values)) :inactive (vec (drop 2 values))})
    (catch java.lang.reflect.InvocationTargetException error [:error (class (.getCause error))])
    (catch Throwable error [:error (class error)])))

(deftest unchanged-training-whole-helpers-preserve-values-and-failure-domains
  (entry/call-with-definition-inputs
   (fn []
     (let [raw-input (atom nil)
           normalize generated/normalize-program
           ;; Capture the actual new-program producer input before physical
           ;; normalization. Do not try to normalize an already lowered entry
           ;; again, nor reconstruct a smaller hand-written gradient fixture.
           prepared (with-redefs [generated/normalize-program
                                  (fn [program method capture]
                                    (reset! raw-input program)
                                    (normalize program method capture))]
                      ((deref (ns-resolve 'raster.ad.reverse 'build-grad-walked-body))
                       #'training/local-loss [0]))
           program @raw-input
           captures (atom [])
           normalized (generated/normalize-program
                       program (interface-method (:tags program))
                       (fn [& arguments]
                         (swap! captures conj (first arguments))
                         (apply capture! arguments)))
           original (emit program)
           compiled (emit normalized)
           runtime (:implementation (reverse/prepared-bytecode-entry prepared))]
       (is (some? normalized))
       (is (= (:raster.compiler.backend.jvm.generated-body/entry-method prepared)
              (interface-method (:tags prepared))))
       (is (identical? program (:raster.ad.reverse/producer-core prepared))
           "the retained core is the actual input observed before helper/JVM adaptation")
       (is (identical? program (:raster.compiler.backend.jvm.generated-body/normalization-input prepared)))
       (is (identical? program (:raster.compiler.backend.jvm.generated-body/normalization-input normalized)))
       (doseq [key [:params :tags :source-ns :result-sym :param-adj-syms]]
         (is (identical? (get program key) (get prepared key))))
       (is (identical? (last (first (:walked-body program)))
                       (last (first (:walked-body prepared))))
           "adaptation keeps the exact padded result-slot vector")
       (is (identical? (:walked-body prepared)
                       (:input-body (reverse/prepared-bytecode-entry prepared))))
       (is (identical? (.getDeclaringClass ^java.lang.reflect.Method
                                         (:raster.compiler.backend.jvm.generated-body/entry-method prepared))
                       (:interface-class (reverse/prepared-bytecode-entry prepared))))
       (is (every? (set @captures)
                   '[raster.dl.nn/linear-nb raster.dl.loss/mse-loss raster.dl.loss/mse-grad
                     raster.dl.nn/linear-dx raster.dl.nn/linear-dW]))
       (let [before @entry/*emission-inputs*]
         (is (nil? (binding [util/*fresh-binder* (constantly 'x)]
                     (generated/normalize-program program (interface-method (:tags program)) capture!))))
         (is (identical? before @entry/*emission-inputs*)))
       (let [observe bytecode/retained-invocation-context
             before @entry/*emission-inputs*]
         (with-redefs [bytecode/retained-invocation-context
                       (fn [& arguments]
                         (when-let [context (apply observe arguments)]
                           (update context :receiver
                                   #(when % (with-meta % (assoc (meta %) :tag 'Object))))))]
           (is (nil? (generated/normalize-program program (interface-method (:tags program)) capture!))))
         (is (identical? before @entry/*emission-inputs*)))
       (let [observe bytecode/retained-binding-context
             before @entry/*emission-inputs*]
         (with-redefs [bytecode/retained-binding-context
                       (fn [& arguments]
                         (when-let [context (apply observe arguments)]
                           (update context :input-context dissoc :object-call-result?)))]
           (is (nil? (generated/normalize-program program (interface-method (:tags program)) capture!))))
         (is (identical? before @entry/*emission-inputs*))
         (let [completions (filter raster.compiler.core.value-completion/canonical-form?
                                   (tree-seq coll? seq (:walked-body normalized)))
               anchors (filter #(= :jvm-bytecode-result-context (get (second %) :source-context))
                               completions)]
           (is (< 6 (count anchors)) "helper initializers, not only helper returns, retain context")
           (is (every? #(raster.compiler.core.value-completion/bytecode-operand-context?
                         (:operand-context (second %))) anchors))))
       (doseq [rows [0 1 3 5] changed? [false true]]
         (reverse/assert-prepared-program-source! prepared)
         (let [theta (if changed?
                       (float-array (map #(+ 0.125 (* 0.75 %)) (training/initial-theta)))
                       (training/initial-theta))
               {:keys [x y]} (first (training/batches [rows]))
               x (if changed? (float-array (map #(+ 0.0625 (* -0.75 %)) x)) x)
               y (if changed? (float-array (map #(- (* 0.5 %) 0.125) y)) y)
               arguments [theta x y rows 17]
               actual (observe compiled arguments)
               producer (observe original arguments)
               owned (observe runtime arguments)
               expected (training/rounded-gradient-sum theta {:rows rows :x x :y y})
               weighted (mapv #(float (* (float rows) %)) (:gradient actual))]
           (is (= producer actual owned))
           (when (zero? rows)
             (is (Double/isNaN (Double/longBitsToDouble (:value-bits actual)))
                 "the empty mean retains its source NaN loss, not a fabricated zero"))
           (is (= (mapv #(Float/floatToRawIntBits %) (:gradient producer))
                  (mapv #(Float/floatToRawIntBits %) (:gradient actual))
                  (mapv #(Float/floatToRawIntBits %) (:gradient owned)))
               "the producer, normalized helper body and actual owned runtime preserve FP32 bits")
           ;; Compare the gradient sum, not a division by rows: this also checks
           ;; the real zero-row behavior without replacing its NaN loss.
           (is (< (training/max-error expected weighted) 1e-7))))
       (doseq [arguments [[nil (float-array 17) (float-array 1) 1 17]
                          [(float-array 17) nil (float-array 1) 1 17]
                          [(float-array 17) (float-array 17) nil 1 17]
                          [(float-array 17) (float-array 1) (float-array 1) 1 17]
                          [(float-array 17) (float-array 17) (float-array 0) 1 17]
                          [(float-array 17) (float-array 0) (float-array 0) 0 17]]]
         (is (= (observe original arguments) (observe compiled arguments))))))))

(deftest unused-helper-arguments-are-converted-before-body-effects
  (entry/call-with-definition-inputs
   (fn []
     (let [source (core/resolve-deftm-var #'unused-argument-helper)
           receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
           _ (core/ensure-current-owned-bytecode! receiver)
           emission (:raster.core/bytecode-emission (meta receiver))
           receiver-symbol (with-meta (symbol (str (ns-name (.ns receiver))) (str (:name (meta receiver))))
                             {:tag (:typed-interface emission) :raster.type/ret-tag 'floats})
           program {:params '[value unused] :tags '[Object Object] :source-ns *ns*
                    :walked-body [(list 'let* ['result (with-meta (list '.invk receiver-symbol 'value 'unused)
                                                        {:raster.op/original
                                                         'raster.compiler.backend.jvm.generated-body-test/unused-argument-helper})]
                                        'result)]}
           normalized (generated/normalize-program program (interface-method (:tags program)) capture!)
           original (emit program) compiled (emit normalized)
           optimized (emit (update normalized :walked-body
                                   #(mapv (fn [body] (simplify/simplify (pe/pe body))) %)))
           result (fn [function arguments]
                    (let [value (first arguments)
                          value (if (instance? (class (float-array 0)) value) (aclone ^floats value) value)
                          arguments (assoc arguments 0 value)]
                      [(try (let [result (invoke function arguments)]
                              {:same-reference? (identical? value result)})
                            (catch java.lang.reflect.InvocationTargetException error (class (.getCause error)))
                            (catch Throwable error (class error)))
                       (when (instance? (class (float-array 0)) value) (vec value))]))]
       (is (some? normalized))
       (is (nil? (generated/normalize-program program (interface-method (:tags program))
                                              (fn [& _] nil))))
       (is (nil? (generated/normalize-program program (interface-method '[floats double])
                                              (fn [& _] (throw (ex-info "wrong ABI captured a helper" {}))))))
       (doseq [implementation [compiled optimized]
               arguments [[(float-array [1.0]) 2.0]
                          [nil 2.0]
                          [(float-array [1.0]) nil]
                          [(float-array [1.0]) "bad-number"]
                          ["bad-array" "bad-number"]]]
         (let [[expected events] (result original arguments)
               [actual actual-events] (result implementation arguments)]
           (is (= expected actual))
           (is (= events actual-events))))))))

(deftest unsupported-new-program-shapes-do-not-invent-physical-facts
  (let [method (interface-method '[Object Object])
        parameters [(with-meta 'x {:tag 'double}) (with-meta 'y {:tag 'float})]
        locals (bytecode/resolved-parameter-locals method parameters)]
    (is (= :ref (get-in locals ['x :type])))
    (is (= :ref (get-in locals ['y :type])))
    (is (identical? Object (get-in locals ['x :reference-class])))
    (is (nil? (get-in locals ['x :hint])))
    (doseq [params ['[x] '[x x] '[x 1]]]
      (is (nil? (bytecode/resolved-parameter-locals method params))))
    (entry/call-with-definition-inputs
     (fn []
       (doseq [body '[(if x y x)
                     (let* [result (unknown x)] [result nil nil])
                     (let* [result y result x] [result nil nil])]]
         (is (nil? (generated/normalize-program
                    {:params '[x y] :tags '[Object Object] :source-ns *ns* :walked-body [body]}
                    method (fn [& _] (throw (ex-info "unsupported input requested helper capture" {})))))))))))

(deftest original-and-new-namespace-lookups-remain-in-the-emitted-owner
  (let [source-scope (create-ns (gensym "raster.generated-source."))
        target-scope (create-ns (gensym "raster.generated-target."))
        decoy (create-ns (gensym "raster.generated-decoy."))]
    (try
      (binding [*ns* source-scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm checked-helper [x :- Double] :- Double
                 (let [rounded (float x)] (double rounded)))))
      (intern decoy 'float identity)
      (entry/call-with-definition-inputs
       (fn []
         (let [source (core/resolve-deftm-var (ns-resolve source-scope 'checked-helper))
               receiver (ns-resolve source-scope (symbol (str (:name (meta source)) "-impl")))
               _ (core/ensure-current-owned-bytecode! receiver)
               emission (:raster.core/bytecode-emission (meta receiver))
               operation (symbol (str (ns-name source-scope)) "checked-helper")
               receiver-symbol (with-meta (symbol (str (ns-name source-scope)) (str (:name (meta receiver))))
                                 {:tag (:typed-interface emission) :raster.type/ret-tag 'double})
               program {:params '[x] :tags '[double] :source-ns target-scope
                        :walked-body [(list 'let* ['result (with-meta (list '.invk receiver-symbol 'x)
                                                            {:raster.op/original operation})] 'result)]}
               normalized (generated/normalize-program program (interface-method (:tags program)) capture!)
               _ (is (some? normalized))
               interface (types/ensure-fn-interface! (:tags program))
               emitted (core/emit-owned-bytecode!
                        bytecode/compile-typed-impl! "raster.compiled.LookupOwnerTest"
                        (:params program) (:tags program) 'Object (:walked-body normalized) target-scope
                        (:iface-name interface) 'lookup-owner @entry/*emission-inputs*)
               guard (entry/prepare-emission-input-guard! (:emission emitted))
               current? (fn [] (try (guard) true (catch clojure.lang.ExceptionInfo _ false)))]
           (is (= 1.25 (invoke (:implementation emitted) [1.25])))
           (is (current?))
           ;; The helper's ORIGINAL bare cast lookup changes, without replacing
           ;; the core Var itself. Its old body must not acquire new authority.
           (try
             (ns-unmap source-scope 'float)
             (intern source-scope 'float identity)
             (is (not (current?)))
             (finally
               (ns-unmap source-scope 'float)
               (binding [*ns* source-scope] (refer 'clojure.core :only '[float]))))
           (is (current?))
           ;; Independently change only the NEW canonical lookup. The original
           ;; helper namespace still resolves its historical cast normally.
           (try
             (binding [*ns* target-scope] (alias 'clojure.core (ns-name decoy)))
             (is (not (current?)))
             (finally (ns-unalias target-scope 'clojure.core)))
           (is (current?)))))
      (finally
        (doseq [scope [source-scope target-scope decoy]] (remove-ns (ns-name scope)))))))
