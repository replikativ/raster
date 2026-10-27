(ns raster.compiler.backend.jvm.generated-body-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.ad.reverse :as reverse]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.call-boundary :as boundary]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.compiler.passes.scalar.simplify :as simplify]
            [raster.compiler.passes.scalar.source-import :as source-import]
            [raster.compiler.fixtures.distributed-training :as training]
            [raster.compiler.fixtures.mixed-storage :as storage]))

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

(deftm unused-terminal-array [values :- (Array double) unused :- (Array double)] :- Double
  (par/reduce total 0.0 i (alength values) (+ total (aget values i))))

(deftm transparent-double-energy [coefficient :- Double values :- (Array double)] :- Double
  (par/dot-product (par/scale coefficient values) values))

(defn- interface-method [tags]
  (let [interface (types/ensure-fn-interface! tags)]
    (first (filter #(= "invk" (.getName ^java.lang.reflect.Method %))
                   (.getDeclaredMethods ^Class (:iface-class interface))))))

(defn- capture! [operation implementation arity method]
  (boundary/capture-selected-emission! operation implementation arity method
                                      core/ensure-current-owned-bytecode!))

(defn- emit [program]
  (let [interface (types/ensure-fn-interface! (:tags program))]
    (let [emitted (source-policy/call-with-owner
                   (cond-> {} (:source-policy program)
                     (assoc :raster.core/source-policy (:source-policy program)))
                   #(core/emit-owned-bytecode!
      bytecode/compile-typed-impl! "raster.compiled.GeneratedBodyTest"
      (:params program) (:tags program) 'Object (:walked-body program)
      (:source-ns program) (:iface-name interface) 'generated-body-test nil))]
      (generated/validate-program-emission! program (:emission emitted))
      ;; These controls are independently emitted derived programs, just like
      ;; the production generated-entry consumer. Close their actual callees
      ;; before invocation; a promotion must restart the surrounding test owner,
      ;; not invalidate an already prepared sibling on the first numeric call.
      (bytecode/prepare-emitted-global-invocations! (:emission emitted))
      (:implementation emitted))))

(defn- invoke [function arguments]
  (clojure.lang.Reflector/invokeInstanceMethod function "invk" (object-array arguments)))

(deftest source-casts-retain-producer-policy-before-helper-adaptation
  (doseq [policy [false true :warn-on-boxed]
          operation ['clojure.core/float 'clojure.core/unchecked-float]]
    (entry/call-with-definition-inputs
     (fn []
       (let [outputs '[result]
             program {:params '[value] :tags '[double] :source-ns *ns*
                      :source-policy {:unchecked-math policy}
                      :walked-body [(list 'let* ['result (list operation 'value)] outputs)]}
             normalized (generated/normalize-program program (interface-method (:tags program)) capture!)
             term (nth (second (first (:walked-body normalized))) 1)
             parts (completion/canonical-parts term)
             original (emit program)
             compiled (emit normalized)]
         (is (= :source-floating-cast (get-in parts [:contract :source-context])))
         (is (= (:source-policy program) (get-in parts [:contract :source-policy])))
         (is (identical? program (:raster.compiler.backend.jvm.generated-body/normalization-input normalized)))
         (is (identical? outputs (last (first (:walked-body normalized)))))
         (doseq [value [0.0 -0.0 1.0e40 Double/NaN]]
           (let [outcome (fn [f]
                           (try [:value (Float/floatToRawIntBits (first (invoke f [value])))]
                                (catch java.lang.reflect.InvocationTargetException error
                                  [:error (class (.getCause error))])
                                (catch Throwable error [:error (class error)])))]
             (is (= (outcome original) (outcome compiled))))))))))

(deftest canonical-only-adapter-survives-helper-abstention-without-claiming-imports
  (entry/call-with-definition-inputs
   (fn []
     (let [program {:params '[value] :tags '[double] :source-ns *ns*
                    :source-policy {:unchecked-math false}
                    :walked-body ['(let* [result (clojure.core/float value)] [result])]}
           adapter (with-redefs-fn
                     {(ns-resolve 'raster.compiler.backend.jvm.generated-body 'normalize-helper-program)
                      (fn [& _] nil)}
                     #(generated/normalize-program program (interface-method (:tags program)) capture!))]
       (is (identical? program (:raster.compiler.backend.jvm.generated-body/normalization-input adapter)))
       (is (= {} (:raster.compiler.backend.jvm.generated-body/binding-facts adapter)))
       (is (= [(float 0.25)] (invoke (emit adapter) [0.25])))
       (is (nil? (generated/normalize-program program nil capture!)))))))

(deftest source-cast-construction-respects-lexical-and-code-owner-boundaries
  (entry/call-with-definition-inputs
   (fn []
     (let [normalize (ns-resolve 'raster.compiler.backend.jvm.generated-body 'canonical-source-casts)
           existing (completion/complete-value
                     (completion/source-floating-cast-contract :float {:unchecked-math true})
                     '(clojure.core/float value))
           bodies ['(let* [float identity result (float value)] [result])
                   '(fn* [value] (clojure.core/float value))
                   '(quote (clojure.core/float value)) existing]
           program {:params '[value] :tags '[Object] :source-ns *ns*
                    :source-policy {:unchecked-math false} :walked-body bodies}
           result (normalize program)]
       (is (false? (:changed? result)))
       (is (= bodies (:body result)))
       (is (identical? existing (last (:body result))))
       (is (false? (:changed? (normalize (dissoc program :source-policy)))))
       (is (= :unclosed-source-policy
              (try (normalize (assoc program :source-policy {})) nil
                   (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest canonical-source-casts-preserve-nullable-branches-and-completion-owner-scope
  (entry/call-with-definition-inputs
   (fn []
     (let [program {:params '[value] :tags '[Object] :source-ns *ns*
                    :source-policy {:unchecked-math false}
                    :walked-body ['(let* [result (if (nil? value) nil (clojure.core/float value))]
                                     [result])]}
           normalized (generated/normalize-program program (interface-method (:tags program)) capture!)
           compiled (emit normalized)
           term (completion/complete-value
                 (completion/source-floating-cast-contract :float {:unchecked-math false}) 'value)
           nested (list 'fn* '[value] term)
           owner-check {:walked-body [nested]
                        :raster.compiler.backend.jvm.generated-body/binding-facts {}}]
       (is (= [nil] (invoke compiled [nil])))
       (is (= [(float 0.25)] (invoke compiled [0.25])))
       (is (identical? owner-check (generated/validate-program-emission! owner-check {})))
       (doseq [current-owner-expression [term (list 'def 'unexecuted-test-definition term)]]
         (is (= :generated-body-completion-context
                (try (generated/validate-program-emission!
                      (assoc owner-check :walked-body [current-owner-expression]) {}) nil
                     (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))))

(deftest generated-source-adaptation-cannot-change-its-emission-policy
  (is (= :ad-generated-source-policy-mismatch
         (try ((ns-resolve 'raster.ad.reverse 'emit-generated-program-entry!)
               {:source-policy {:unchecked-math true}}
               {:source-entry {:source-policy {:unchecked-math false}}}) nil
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

(deftest source-cast-construction-refuses-a-replaced-bootstrap-intrinsic
  (entry/call-with-definition-inputs
   (fn []
     (let [normalize (ns-resolve 'raster.compiler.backend.jvm.generated-body 'canonical-source-casts)
           program {:params '[value] :source-ns *ns*
                    :source-policy {:unchecked-math false}
                    :walked-body ['(clojure.core/float value)]}]
       (is (= :intrinsic-source-definition-mismatch
              (with-redefs [clojure.core/float identity]
                (try (normalize program) nil
                     (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))))

(deftest mixed-gradient-retains-owned-float-completion-and-separate-numerical-core
  (let [prepared (reverse/prepare-value+grad #'storage/mixed-scale-energy [0])
        {:keys [emission input-body]} (reverse/prepared-bytecode-entry prepared)
        core (:raster.ad.reverse/producer-core prepared)
        source-casts (fn [body]
                       (filter #(= :source-floating-cast
                                   (get-in (completion/canonical-parts %) [:contract :source-context]))
                               (tree-seq coll? seq body)))
        terms (vec (source-casts input-body))
        gradient (reverse/value+grad #'storage/mixed-scale-energy :wrt [0])
        result (gradient (float 2) (double-array [1 2 3]))]
    (is (= 1 (count terms)))
    (is (empty? (source-casts (:walked-body core))))
    (is (identical? core (:raster.compiler.backend.jvm.generated-body/normalization-input prepared)))
    (is (= (:source-policy core) (:source-policy prepared)))
    (doseq [term terms]
      (let [observed (bytecode/retained-input-completion-context emission input-body term)]
        (is (= :ref (:input-carrier observed)))
        (is (= :float (:completion-carrier observed)))
        (is (= (:source-policy prepared) (get-in observed [:contract :source-policy])))))
    (is (= [28.0 (float 14.0) nil] result))
    (is (instance? Float (second result)))
    (is (= IllegalArgumentException
           (try (gradient (float 2) (double-array [1.0e20])) nil
                (catch Throwable error (class error)))))))

(defn- fp-output-bits [values]
  (mapv (fn [value]
          (cond (instance? Float value) [:float (Float/floatToRawIntBits value)]
                (instance? Double value) [:double (Double/doubleToRawLongBits value)]
                :else value)) values))

(defn- projected-program [prepared projection]
  {:params '[events coefficient values] :tags '[Object Object Object]
   :source-ns *ns* :source-policy (:source-policy prepared)
   ;; Actual completion contexts are audited by the common test emitter. No
   ;; pre-projection binder expectation is asserted after whole-form hygiene.
   :raster.compiler.backend.jvm.generated-body/binding-facts {}
   :walked-body [(list 'let* (vec (mapcat identity (:bindings projection)))
                       (:result-form projection))]})

(deftest scoped-derived-ad-source-keeps-the-actual-prepared-runtime-entry
  (entry/call-with-definition-inputs
   (fn []
     (let [callable (reverse/value+grad #'storage/mixed-scale-energy :wrt [0])
           prepared (:prepared (reverse/captured-prepared-callable callable (meta callable)))
           physical (reverse/prepared-bytecode-entry prepared)
           core (:raster.ad.reverse/producer-core prepared)
           core-body (:walked-body core)
           prepared-body (:walked-body prepared)
           arguments ['(do (clojure.core/swap! events conj :coefficient) coefficient)
                      '(do (clojure.core/swap! events conj :values) values)]
           calls (atom 0)
           consumer (fn [request]
                      (swap! calls inc)
                      (source-import/construct-contextual-terminal-return request))
           baseline (reverse/project-prepared-value+grad-call prepared arguments :value+grad {})
           normalize generated/normalize-program
           normalization (atom nil)
           derived (with-redefs [generated/normalize-program
                                 (fn [& arguments]
                                   (let [result (apply normalize arguments)]
                                     (when (= 4 (count arguments))
                                       (reset! normalization {:program (first arguments)
                                                              :options (last arguments) :result result}))
                                     result))]
                     (reverse/call-with-generated-source-construction
                      consumer
                      #(reverse/project-prepared-value+grad-call prepared arguments :value+grad {})))
           original (emit (projected-program prepared baseline))
           compiled (emit (projected-program prepared derived))
           after (reverse/project-prepared-value+grad-call prepared arguments :value+grad {})
           unchanged (emit (projected-program prepared after))
           outcome (fn [function coefficient values]
                     (let [events (atom [])
                           value (try [:value (fp-output-bits (invoke function [events coefficient values]))]
                                      (catch java.lang.reflect.InvocationTargetException error
                                        [:error (class (.getCause error))])
                                      (catch Throwable error [:error (class error)]))]
                       [value @events]))]
       (is (= 2 @calls) "the mixed Float continuation constructs both selected terminal helpers")
       (is (true? (get-in @normalization [:options :terminal-only?])))
       (is (identical? core (:program @normalization)))
       (let [original-pairs (partition 2 (second (first (:walked-body core))))
             normalized-pairs (partition 2 (second (first (get-in @normalization [:result :walked-body]))))
             ordinary (filter #(= 'raster.par/scale
                                   (:semantic-op (descriptor/call-description (second %)))) original-pairs)
             contextual (filter (fn [[_ expression]]
                                  (let [{:keys [contract operand]} (completion/canonical-parts expression)]
                                    (and (= :jvm-bytecode-result-context (:source-context contract))
                                         (= :ref (:result-carrier contract))
                                         (par-ir/par-reduce-form? operand)))) normalized-pairs)]
         (is (seq ordinary) "the actual mixed gradient contains an ordinary array scale helper")
         (doseq [[binder initializer] ordinary]
           (let [retained (first (filter #(identical? binder (first %)) normalized-pairs))]
             (is (identical? initializer (second retained))
                 "ordinary array helpers remain literal calls for the normal GPU inliner")))
         (is (seq contextual) "only authenticated terminal reductions acquire protected return bindings"))
       (is (< (count (:bindings baseline)) (count (:bindings derived))))
       (is (= 2 (count (filter #(= :jvm-object-input
                                    (get-in (completion/canonical-parts (second %))
                                            [:contract :source-context]))
                              (take 2 (:bindings derived)))))
           "every caller argument is snapshotted before wrapper conversions")
       (is (identical? physical (reverse/prepared-bytecode-entry prepared)))
       (is (identical? core (:raster.ad.reverse/producer-core prepared)))
       (is (identical? core-body (:walked-body core)))
       (is (identical? prepared-body (:walked-body prepared)))
       (is (identical? prepared (:prepared (reverse/captured-prepared-callable callable (meta callable)))))
       (is (= [28.0 (float 14.0) nil] (callable (float 2) (double-array [1 2 3]))))
       (is (= [[:value (fp-output-bits [28.0 (float 14.0) nil])] [:coefficient :values]]
              (outcome compiled (float 2) (double-array [1 2 3]))))
       (doseq [[coefficient values] [[(float 2) (double-array [1 2 3])]
                                     [(float -0.0) (double-array [1 2 3])]
                                     [(float 2) (double-array [Double/NaN])]
                                     [(float 2) (double-array [1.0e20])]
                                     [nil (double-array [1 2 3])]
                                     ["wrong coefficient" (double-array [1 2 3])]
                                     [(float 2) (float-array [1 2 3])]]]
         (is (= (outcome original coefficient values)
                (outcome compiled coefficient values)
                (outcome unchanged coefficient values))))
       (let [before @calls]
         (reverse/project-prepared-value+grad-call prepared arguments :grad {})
         (is (= before @calls) "the scoped consumer is absent from ordinary projection"))
       (let [projection (reverse/call-with-generated-source-construction
                         consumer #(reverse/project-prepared-value+grad-call prepared arguments :grad {}))]
         (is (= [[:value (fp-output-bits [(float 14.0) nil])] [:coefficient :values]]
                (outcome (emit (projected-program prepared projection))
                         (float 2) (double-array [1 2 3])))))))))

(deftest transparent-double-ad-keeps-its-original-source-projection
  (entry/call-with-definition-inputs
   (fn []
     (let [callable (reverse/value+grad #'transparent-double-energy :wrt [0])
           prepared (:prepared (reverse/captured-prepared-callable callable (meta callable)))
           physical (reverse/prepared-bytecode-entry prepared)
           core (:raster.ad.reverse/producer-core prepared)
           arguments '[coefficient values]
           baseline (reverse/project-prepared-value+grad-call prepared arguments :value+grad {})
           project-flat @#'reverse/project-flat-prepared-call
           projected-source (atom nil)
           derived (with-redefs-fn
                     {#'generated/normalize-program
                      (fn [& _] (throw (AssertionError. "transparent Double AD must not query terminal normalization")))
                      #'reverse/project-generated-float-source
                      (fn [& _] (throw (AssertionError. "transparent Double AD must not enter derived-core construction")))
                      #'reverse/project-flat-prepared-call
                      (fn [source & args]
                        (reset! projected-source source)
                        (apply project-flat source args))}
                     #(reverse/call-with-generated-source-construction
                       (fn [& _] (throw (AssertionError. "transparent Double AD must not construct terminal returns")))
                       (fn [] (reverse/project-prepared-value+grad-call prepared arguments :value+grad {}))))
           topology (fn [projection]
                      (frequencies
                       (keep (fn [node]
                               (or (when-let [contract (:contract (completion/canonical-parts node))]
                                     [:completion (:source-context contract)])
                                   (when-let [operation (:semantic-op (descriptor/call-description node))]
                                     [:operation operation])))
                             (tree-seq coll? seq [(:bindings projection) (:result-form projection)]))))
           original (emit (projected-program prepared baseline))
           compiled (emit (projected-program prepared derived))
           values (double-array [1 2 3])]
       (is (identical? (first (:walked-body prepared)) @projected-source)
           "the exact existing adapter, not a reconstructed numerical core, supplies source")
       (is (= (topology baseline) (topology derived))
           "ordinary kernel calls and return contexts have unchanged topology")
       (is (= (fp-output-bits [28.0 14.0 nil])
              (fp-output-bits (invoke original [(atom []) 2.0 values]))
              (fp-output-bits (invoke compiled [(atom []) 2.0 values]))
              (fp-output-bits (callable 2.0 values))))
       (is (identical? physical (reverse/prepared-bytecode-entry prepared)))
       (is (identical? core (:raster.ad.reverse/producer-core prepared)))
       (is (identical? prepared (:prepared (reverse/captured-prepared-callable callable (meta callable)))))
       (let [reason (fn [project]
                      (try (project) nil
                           (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
             invalid #(reverse/project-prepared-value+grad-call prepared '[coefficient] :value+grad {})
             before @entry/*emission-inputs*]
         (is (= (reason invalid)
                (reason #(reverse/call-with-generated-source-construction
                          (fn [& _] (throw (AssertionError. "must not derive"))) invalid))))
         (is (identical? before @entry/*emission-inputs*)
             "the common default arity refusal adopts no new history"))))))

(deftest original-source-projection-fence-preserves-unrelated-contexts-and-restores-on-errors
  (let [context {:project-prepared-call (fn [& _] :unrelated)}
        failure (Exception. "original construction failed")]
    (with-bindings {#'reverse/*generated-product-context* context}
      (is (identical? context (reverse/call-with-original-source-projection
                              #(deref #'reverse/*generated-product-context*))))
      (is (identical? failure
                      (try (reverse/call-with-original-source-projection #(throw failure))
                           (catch Exception error error))))
      (is (identical? context @#'reverse/*generated-product-context*)))
    (entry/call-with-definition-inputs
     #(reverse/call-with-generated-source-construction
       source-import/construct-contextual-terminal-return
       (fn []
         (let [target-context @#'reverse/*generated-product-context*]
           (is (nil? (reverse/call-with-original-source-projection
                      (fn [] @#'reverse/*generated-product-context*))))
           (is (identical? failure
                           (try (reverse/call-with-original-source-projection (fn [] (throw failure)))
                                (catch Exception error error))))
           (is (identical? target-context @#'reverse/*generated-product-context*))))))))

(deftest cold-original-ad-publication-does-not-inherit-target-local-construction
  (let [scope (create-ns (gensym "raster.original-ad-publication."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm cold-gradient
                 [coefficient :- Float values :- (Array double)] :- Float
                 (let [gradient ((raster.ad.reverse/value+grad
                                  (var raster.compiler.fixtures.mixed-storage/mixed-scale-energy) :wrt [0])
                                 coefficient values)]
                   (nth gradient 1)))))
      (let [generic (ns-resolve scope 'cold-gradient)
            source (core/resolve-deftm-var generic)
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            calls (atom 0)
            consumer (fn [request]
                       (swap! calls inc)
                       (source-import/construct-contextual-terminal-return request))]
        (is (nil? (:raster.core/bytecode-emission (meta receiver))) "this original publisher starts cold")
        (entry/call-with-definition-inputs
         #(reverse/call-with-generated-source-construction
           consumer
           (fn []
             (core/ensure-current-owned-bytecode! receiver)
             (is (zero? @calls) "JIT publication uses only the original source projector")
             (let [implementation @receiver
                   emission (:raster.core/bytecode-emission (meta receiver))
                   contextual (filter (fn [node]
                                        (let [{:keys [contract operand]} (completion/canonical-parts node)]
                                          (and (= :jvm-bytecode-result-context (:source-context contract))
                                               (par-ir/par-reduce-form? operand))))
                                      (tree-seq coll? seq (:body emission)))
                   prepared (reverse/prepare-value+grad #'storage/mixed-scale-energy [0])]
               (is (some? emission))
               (is (seq (:body emission)))
               (is (empty? contextual) "original cached source contains no target-local reduction contexts")
               (is (some? (reverse/project-prepared-value+grad-call
                           prepared '[coefficient values] :value+grad {})))
               (is (= 2 @calls) "the surrounding target-local projector remains active after publication")
               (core/ensure-current-owned-bytecode! receiver)
               (is (identical? implementation @receiver))
               (is (identical? emission (:raster.core/bytecode-emission (meta receiver))))))))
        (is (= (float 14.0) (@generic (float 2) (double-array [1 2 3])))))
      (finally (remove-ns (ns-name scope))))))

(deftest derived-ad-source-abstention-and-failure-do-not-replace-existing-owners
  (entry/call-with-definition-inputs
   (fn []
     (let [prepared (reverse/prepare-value+grad #'storage/mixed-scale-energy [0])
           arguments '[coefficient values]
           consumer source-import/construct-contextual-terminal-return
           project #(reverse/project-prepared-value+grad-call prepared arguments :value+grad {})
           project-flat @#'reverse/project-flat-prepared-call
           actual-source (atom nil)]
       (with-redefs-fn
         {#'generated/normalize-program (fn [& _] nil)
          #'reverse/project-flat-prepared-call
          (fn [source & args] (reset! actual-source source) (apply project-flat source args))}
         #(is (some? (reverse/call-with-generated-source-construction consumer project))))
       (is (identical? (first (:walked-body prepared)) @actual-source)
           "normalization abstention projects the actual prepared adapter, not an unadapted core")
       (let [before @entry/*emission-inputs*
             failure (Exception. "consumer rejected after projection")]
         (is (nil? (reverse/call-with-generated-source-construction
                    consumer #(do (project) nil))))
         (is (identical? before @entry/*emission-inputs*))
         (is (identical? failure
                         (try (reverse/call-with-generated-source-construction
                               consumer #(do (project) (throw failure))) nil
                              (catch Exception error error))))
         (is (identical? before @entry/*emission-inputs*)))
       (is (= :ad-generated-source-construction-context
              (try (reverse/call-with-generated-source-construction
                    consumer #(reverse/call-with-generated-source-construction consumer project)) nil
                   (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
       (let [conflict {:project-prepared-call (fn [& _] (throw (AssertionError. "must not override")))}]
         (with-bindings {#'reverse/*generated-product-context* conflict}
           (is (= :ad-generated-source-construction-context
                  (try (reverse/call-with-generated-source-construction consumer project) nil
                       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
       (is (= :ad-prepared-program-mismatch
              (try (reverse/call-with-generated-source-construction
                    consumer #(reverse/project-prepared-value+grad-call
                               (assoc prepared :raster.ad.reverse/producer-core
                                      (into {} (:raster.ad.reverse/producer-core prepared)))
                               arguments :value+grad {})) nil
                   (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

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

(deftest helper-application-refuses-a-valid-but-foreign-selected-capture
  (entry/call-with-definition-inputs
   (fn []
     (let [program (helper-program #'ordered-body-helper)
           foreign (helper-program #'unused-argument-helper)
           foreign-call (second (second (first (:walked-body foreign))))
           {:keys [semantic-op implementation-op]} (descriptor/call-description foreign-call)
           captures (atom 0)]
       (is (nil?
            (generated/normalize-program
             program (interface-method (:tags program))
             (fn [_ _ arity method]
               (swap! captures inc)
               ;; Both helpers have the same physical interface. Matching a
               ;; Method signature does not authorize substituting its source.
               (capture! semantic-op implementation-op arity method)))))
       (is (pos? @captures))))))

(deftest helper-application-validation-retains-construction-and-original-capture-guard
  (entry/call-with-definition-inputs
   (fn []
     (let [program (helper-program #'ordered-body-helper)
           expression (second (second (first (:walked-body program))))
           locals (bytecode/resolved-parameter-locals
                   (interface-method (:tags program)) (:params program))
           context (bytecode/binding-initializer-context {:source-ns *ns* :object-result? true})
           current? (atom true)
           application
           (source-import/instantiate-emitted-helper
            expression locals context
            (fn [& arguments]
              (let [owner (apply capture! arguments)
                    original (:guard owner)]
                (assoc owner :guard
                       #(do (when-not @current?
                              (throw (ex-info "original capture is no longer current"
                                              {:reason ::stale-application-capture})))
                            (original))))) nil false)]
       (is (some? application))
       (is (nil? (source-import/instantiate-emitted-helper
                  (with-meta (cons 'not-an-invocation (rest expression)) (meta expression))
                  locals context
                  (fn [& _] (throw (AssertionError. "non-call reached helper capture"))) nil false)))
       (is (identical? application (source-import/validate-helper-application! application)))
       (doseq [altered [(assoc application :result-form 7.0)
                        (update application :application assoc :selected-method nil)
                        (dissoc application :receipt)]]
         (is (= :helper-application-correspondence
                (try (source-import/validate-helper-application! altered) nil
                     (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
       (reset! current? false)
       (try
         (is (= ::stale-application-capture
                (try (source-import/validate-helper-application! application) nil
                     (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
         ;; The enclosing compiler owner independently validates on exit too.
         ;; Restore this test-only guard so that boundary can close normally.
         (finally (reset! current? true)))))))

(deftest helper-applications-do-not-capture-prior-sibling-receipts
  (entry/call-with-definition-inputs
   (fn []
     (let [program (helper-program #'ordered-body-helper)
           expression (second (second (first (:walked-body program))))
           locals (bytecode/resolved-parameter-locals
                   (interface-method (:tags program)) (:params program))
           context (bytecode/binding-initializer-context {:source-ns *ns* :object-result? true})
           applications (mapv (fn [_]
                                (source-import/instantiate-emitted-helper
                                 expression locals context capture! nil false))
                              (range 4))]
       (is (every? some? applications))
       (doseq [[index application] (map-indexed vector applications)]
         (is (identical? application (source-import/validate-helper-application! application)))
         (doseq [earlier (take index applications)]
           (is (not-any? #(identical? (:receipt earlier) %)
                         (get-in application [:application :dependencies :producer-inputs]))
               "a sibling receipt is not part of this application's own construction history")))))))

(defn- terminal-helper-program [generic]
  (let [operation (symbol (str (ns-name (.ns generic))) (str (:name (meta generic))))
        source (core/resolve-deftm-var generic)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        _ (core/ensure-current-owned-bytecode! receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        parameters (mapv #(symbol (str "argument" %)) (range (count (:params emission))))
        receiver-symbol (with-meta (symbol (str (ns-name (.ns receiver))) (str (:name (meta receiver))))
                          {:tag (:typed-interface emission)})
        outputs '[result]]
    {:params parameters :tags (vec (repeat (count parameters) 'Object)) :source-ns *ns*
     :walked-body [(list 'let* ['result (with-meta (list* '.invk receiver-symbol parameters)
                                         {:raster.op/original operation})] outputs)]}))

(defn- consume-terminal-for-jvm
  [{:keys [return-contract argument-bindings instantiated-body binding-facts]}]
  ;; This test consumer keeps the complete original result-context recipe.
  ;; In particular it does not extract the reduction or erase its Object context.
  {:expression (list 'let* argument-bindings
                     (util/subst-syms {(:value-port return-contract) instantiated-body}
                                     (:recipe return-contract)))
   :binding-facts binding-facts})

(deftest terminal-static-return-needs-an-explicit-constructing-consumer
  (entry/call-with-definition-inputs
   (fn []
     (doseq [helper [#'par/sum #'unused-terminal-array]]
       (let [program (terminal-helper-program helper)
             method (interface-method (:tags program))
             seen (atom [])
             events (atom [])
             query source-import/terminal-reduction-return-contract
             allocate util/*fresh-binder*
             before @entry/*emission-inputs*
             _ (is (nil? (generated/normalize-program program method capture!)))
             _ (is (identical? before @entry/*emission-inputs*))
             normalized
             (with-redefs [source-import/terminal-reduction-return-contract
                           (fn [request] (swap! events conj :authenticate) (query request))]
               (binding [util/*fresh-binder* (fn [id] (swap! events conj :hygiene) (allocate id))]
                 (generated/normalize-program
                  program method capture!
                  {:consume-terminal-reduction
                   (fn [request]
                     (swap! seen conj request)
                     (consume-terminal-for-jvm request))})))
             request (first @seen)
             returned (:return-contract request)
             original (emit program)
             compiled (emit normalized)]
         (is (some? normalized))
         (is (= 1 (count @seen)))
         (is (= :authenticate (first @events)))
         (is (some #{:hygiene} @events))
         (is (identical? returned (source-import/validate-terminal-reduction-return! returned)))
         (is (identical? (:frontend request) (:frontend returned)))
         (is (identical? (nth (second (first (:walked-body program))) 1) (:expression request)))
         (is (instance? java.lang.reflect.Method (get-in request [:invocation :method])))
         (is (= (count (:params program)) (count (:inputs request)) (count (:input-values request))))
         (is (every? #(= :ref (:type %)) (vals (:locals request))))
         (is (every? #(identical? Object (:reference-class %)) (vals (:locals request))))
         (is (every? #(identical? (.arrayType Double/TYPE) (get-in % [:contract :target-carrier]))
                     (:inputs request))
             "every actual formal array CHECKCAST remains in its positional recipe")
         (is (= (bytecode/retained-input-body-context (:emission returned) (:input-body returned))
                (:saved-return-context request)))
         (is (identical? (last (first (:walked-body program)))
                         (last (first (:walked-body normalized))))
             "the actual caller's output object survives construction")
         (is (identical? program (:raster.compiler.backend.jvm.generated-body/normalization-input normalized)))
         (doseq [arguments (if (= 1 (count (:params program)))
                             [[(double-array [1 2 3])] [(float-array 1)]]
                             [[(double-array [1 2 3]) (double-array 0)]
                              [(double-array [1 2 3]) (float-array 0)]])]
           (let [observe (fn [function]
                           (try [:value (invoke function arguments)]
                                (catch java.lang.reflect.InvocationTargetException error
                                  [:error (class (.getCause error))])
                                (catch Throwable error [:error (class error)])))]
             (is (= (observe original) (observe compiled)))))
         (is (= [6.0] (invoke compiled (vec (repeat (count (:params program)) (double-array [1 2 3])))))))))))

(deftest terminal-consumer-cannot-erase-arguments-or-activate-a-foreign-frontend
  (entry/call-with-definition-inputs
   (fn []
     (let [program (terminal-helper-program #'par/sum)
           _ (terminal-helper-program #'unused-terminal-array)
           foreign (entry/original-source-frontend (core/resolve-deftm-var #'unused-terminal-array))
           method (interface-method (:tags program))
           calls (atom 0)
           consumer (fn [request] (swap! calls inc) (consume-terminal-for-jvm request))]
       (with-redefs [entry/original-source-frontend (fn [_] foreign)]
         (is (= :unowned-checked-result-frontend
                (try (generated/normalize-program program method capture!
                                                  {:consume-terminal-reduction consumer}) nil
                     (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
             "selected capture rejects the foreign frontend before optional construction"))
       (is (zero? @calls))
       (let [before @entry/*emission-inputs*]
         (is (nil? (generated/normalize-program
                    program method capture!
                    {:consume-terminal-reduction
                     (fn [request]
                       (assoc (consume-terminal-for-jvm request) :expression
                              (list 'let* [] (:instantiated-body request))))})))
         (is (identical? before @entry/*emission-inputs*)))
       (is (= :generated-body-unsupported
              (try (generated/normalize-program program method capture!
                                                {:consume-terminal-reduction true}) nil
                   (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

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
       :gradient-class (class (second values))
       :gradient (vec (second values)) :inactive (vec (drop 2 values))})
    (catch java.lang.reflect.InvocationTargetException error [:error (class (.getCause error))])
    (catch Throwable error [:error (class error)])))

(defn- bitwise-gradient-observation [observation]
  (if (map? observation)
    (update observation :gradient
            (fn [gradient]
              (mapv (fn [value]
                      [(class value)
                       (cond
                         (instance? Float value) (Float/floatToRawIntBits value)
                         (instance? Double value) (Double/doubleToRawLongBits value)
                         :else value)])
                    gradient)))
    observation))

(deftest gradient-observation-preserves-nan-payload-zero-and-storage-class
  (let [observe-array (fn [gradient]
                        (bitwise-gradient-observation
                         {:gradient-class (class gradient) :gradient (vec gradient)}))
        nan (Float/intBitsToFloat 0x7fc00001)
        other-nan (Float/intBitsToFloat 0x7fc00002)]
    (is (= (observe-array (float-array [nan]))
           (observe-array (float-array [nan]))))
    (is (not= (observe-array (float-array [nan]))
              (observe-array (float-array [other-nan]))))
    (is (not= (observe-array (float-array [0.0]))
              (observe-array (float-array [-0.0]))))
    (is (not= (observe-array (float-array [1.0]))
              (observe-array (double-array [1.0]))))
    (is (= [:error NullPointerException]
           (bitwise-gradient-observation [:error NullPointerException])))))

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
         ;; Numeric equality rejects separately boxed NaNs. Keep their exact
         ;; payload bits and actual storage/element Classes in this oracle.
         (is (= (bitwise-gradient-observation (observe original arguments))
                (bitwise-gradient-observation (observe compiled arguments)))))))))

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

(deftest terminal-only-keeps-ordinary-calls-and-rolls-back-refused-capture-history
  (entry/call-with-definition-inputs
   (fn []
     (let [program (helper-program #'ordered-body-helper)
           method (interface-method (:tags program))
           [binder initializer] (take 2 (second (first (:walked-body program))))
           checks (atom 0)
           marker (reify entry/RetainedProducerInput
                    (assert-producer-current! [this] (swap! checks inc) this))
           captures (atom 0)
           normalized
           (generated/normalize-program
            program method
            (fn [& arguments]
              (swap! captures inc)
              (entry/adopt-definition-inputs! {:producer-inputs [marker]})
              (apply capture! arguments))
            {:terminal-only? true
             :consume-terminal-reduction
             (fn [_] (throw (AssertionError. "ordinary reference helper is not a terminal reduction")))})
           [actual-binder actual-initializer] (take 2 (second (first (:walked-body normalized))))
           original (emit program)
           compiled (emit normalized)]
       (is (some? normalized))
       (is (pos? @captures))
       (is (identical? binder actual-binder))
       (is (identical? initializer actual-initializer))
       (is (not-any? #(identical? marker %) (:producer-inputs @entry/*emission-inputs*)))
       (reset! checks 0)
       (entry/assert-definition-inputs-current! @entry/*emission-inputs*)
       (is (zero? @checks) "a preserved call does not retain a rejected import's capture history")
       (doseq [x [2.75 -3.25]]
         (let [left (float-array [10 20]) right (aclone left)]
           (is (identical? left (invoke original [left x])))
           (is (identical? right (invoke compiled [right x])))
           (is (= (vec left) (vec right)))))
       (is (= :generated-body-unsupported
              (try (generated/normalize-program program method capture! {:terminal-only? true}) nil
                   (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

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
