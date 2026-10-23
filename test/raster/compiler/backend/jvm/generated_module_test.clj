(ns raster.compiler.backend.jvm.generated-module-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.fixtures.distributed-training :as training]))

(defn- emit [parameters tags body]
  (let [interface (types/ensure-fn-interface! tags)
        template (try (generated/construct-value-source-template body parameters)
                      (catch clojure.lang.ExceptionInfo error
                        (when-not (= :generated-body-unsupported (:reason (ex-data error)))
                          (throw error))))
        emitted (core/emit-owned-bytecode!
                 bytecode/compile-typed-impl! "raster.compiled.GeneratedModuleTest"
                 parameters tags 'Object body
                 (the-ns 'raster.compiler.backend.jvm.generated-module-test)
                 (:iface-name interface) 'generated-module-test)]
    (generated/retain-value-source-template
     (cond-> {:params parameters} template (assoc ::generated/value-source-template-candidate template))
     (assoc emitted :input-body body :interface-class (:iface-class interface)))))

(deftest pre-emission-template-does-not-reconstruct-numerics-from-physical-code
  (let [parameters '[x]
        context (completion/bytecode-operand-context {})
        contract (completion/planned-bytecode-input-contract Float/TYPE context :double)
        input (completion/complete-value contract 'x)
        boxed (completion/complete-value completion/object-input-contract 'converted)
        arithmetic '(clojure.core/* converted converted)
        body [(list 'let* ['converted input 'boxed boxed 'square arithmetic] '[square boxed])]
        template (with-redefs [bytecode/retained-input-completion-context
                              (fn [& _] (throw (AssertionError. "physical observation during construction")))]
                   (generated/construct-value-source-template body parameters))
        obligations (:completion-obligations template)
        candidate (second (:source template))]
    (is (identical? body (:input-body template)))
    (is (identical? parameters (:parameters template)))
    (is (identical? arithmetic (nth candidate 5)))
    (is (false? (:admitted? template)))
    (is (conversion/canonical-form? (second candidate)))
    (is (= [:double :float] ((juxt :source-dtype :target-dtype) (first obligations))))
    (is (= [:scalar-conversion :object-materialization] (mapv :classification obligations)))
    (is (every? #(and (= :pending (:status %)) (nil? (:observed %))) obligations))
    (is (not (completion/contains-completion? (:source template))))))

(deftest template-holes-belong-to-occurrences-not-source-object-identities
  (let [parameters '[x]
        contract (completion/bytecode-floating-input-contract
                  Float/TYPE (completion/bytecode-operand-context {}))
        shared (completion/complete-value contract 'x)
        quoted (list 'quote shared)
        body [(list 'let* ['left shared 'right shared 'quoted quoted] '[left right quoted])]
        template (generated/construct-value-source-template body parameters)
        [left right :as obligations] (:completion-obligations template)]
    (is (= 2 (count obligations)))
    (is (identical? shared (:source left)))
    (is (identical? shared (:source right)))
    (is (not (identical? (:token left) (:token right))))
    (is (not (identical? (:hole left) (:hole right))))
    (is (= [[1 1] [1 3]] (mapv :path obligations)))
    (is (every? #(and (= :numeric-boundary-hole (:classification %))
                     (nil? (:source-dtype %))
                     (completion/canonical-form? (:hole %))) obligations))
    (is (identical? quoted (nth (second (:source template)) 5)))
    (is (every? #(= (meta shared) (meta (:hole %))) obligations)
        "fresh holes carry only the existing completion constructor's metadata")))

(deftest malformed-template-completion-cannot-become-an-erased-wrapper
  (is (= :invalid-value-completion
         (try (generated/construct-value-source-template
               [(list 'let* ['value (list completion/completion-symbol {:source-context :unknown} 'x)]
                      '[value])] '[x])
              nil
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

(deftest templates-require-lexical-parameter-and-binding-identities
  (doseq [[parameters bindings] [['[x/y] '[value 1.0]]
                                 [[1] '[value 1.0]]
                                 ['[x] '[scope/value 1.0]]]]
    (is (= :generated-body-unsupported
           (try (generated/construct-value-source-template [(list 'let* bindings '[value])] parameters)
                nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest retained-templates-join-nested-shared-completions-in-annotation-order
  (let [parameters '[x]
        inner (completion/complete-value completion/object-input-contract 'x)
        outer (completion/complete-value completion/object-input-contract inner)
        reference (completion/complete-value (completion/bytecode-boxed-parameter-contract Object) 'x)
        bytecode-reference (completion/complete-value
                            (completion/bytecode-reference-input-contract
                             Object (completion/bytecode-operand-context {})) 'x)
        body [(list 'let* ['left outer 'right inner 'reference reference 'bytecode-reference bytecode-reference]
                    '[left right reference bytecode-reference])]
        physical (emit parameters '[Object] body)
        template (::generated/value-source-template physical)
        proposal (generated/propose-value-source physical parameters)]
    (is (identical? template (:template proposal)))
    (is (= 5 (count (:completion-obligations proposal))))
    (is (every? true? (map identical? [inner outer inner reference bytecode-reference]
                          (map :source (:completion-obligations proposal)))))
    (is (= [:object-materialization :object-materialization :object-materialization
            :reference-identity :reference-identity]
           (mapv :classification (:completion-obligations proposal))))
    (is (identical? (:source template) (:source proposal)) "no numerical spine rebuild when there are no holes")
    (is (= :generated-body-unsupported
           (try (generated/propose-value-source (dissoc physical ::generated/value-source-template) parameters)
                nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest unavailable-completion-spines-do-not-disable-the-original-cpu-entry
  (let [parameters '[x]
        contract (completion/planned-bytecode-input-contract
                  Double/TYPE (completion/bytecode-operand-context {}) :double)
        divergence (completion/complete-value contract '(throw x))
        later (completion/complete-value completion/object-input-contract 'x)
        body [(list 'let* ['value (list 'do divergence later)] '[value])]
        physical (emit parameters '[Object] body)]
    (is (nil? (::generated/value-source-template physical)))
    (is (thrown? IllegalStateException ((:implementation physical) (IllegalStateException. "stop"))))
    (is (= :generated-body-unsupported
           (try (generated/propose-value-source physical parameters) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
  (let [parameters '[x]
        contract (completion/planned-bytecode-input-contract
                  Double/TYPE (completion/bytecode-operand-context {}) :double)
        body [(list 'let* ['value (completion/complete-value contract '(throw x))] '[value])]
        physical (emit parameters '[Object] body)]
    (is (some? (::generated/value-source-template physical))
        "a divergent operand has no normal carrier to contradict the planned conversion")
    (is (thrown? IllegalStateException ((:implementation physical) (IllegalStateException. "stop"))))))

(deftest template-boundary-specialization-preserves-floating-checkpoints
  (let [parameters '[x]
        context (completion/bytecode-operand-context {})
        inner (completion/complete-value
               (completion/planned-bytecode-input-contract Float/TYPE context :double) 'x)
        outer (completion/complete-value
               (completion/bytecode-floating-input-contract Double/TYPE context) inner)
        body [(list 'let* ['value outer] '[value])]
        physical (emit parameters '[double] body)
        proposal (generated/propose-value-source physical parameters)
        projected (conversion/project-canonical-to-source (:source proposal))
        compiled (emit parameters '[double] [projected])]
    (is (= 1 (count (filter :hole (get-in proposal [:template :completion-obligations])))))
    (is (not (completion/contains-completion? (:source proposal))))
    (doseq [x [0.0 -0.0 1.25 (/ 1.0 3.0) Double/MIN_VALUE Double/MAX_VALUE
               Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN]]
      (is (= (Double/doubleToRawLongBits (double (first ((:implementation physical) x))))
             (Double/doubleToRawLongBits (double (first ((:implementation compiled) x)))))
          (str "float checkpoint followed by double conversion: " x)))))

(deftest a-proposal-retains-every-context-and-is-not-admission
  (let [parameters '[x]
        context (completion/bytecode-operand-context {})
        input (completion/complete-value (completion/bytecode-floating-input-contract Float/TYPE context) 'x)
        boxed (completion/complete-value completion/object-input-contract 'converted)
        body [(list 'let* ['converted input 'boxed boxed] '[boxed nil])]
        physical (emit parameters '[double] body)
        proposal (generated/propose-value-source physical parameters)
        conversion (second (second (:source proposal)))
        obligations (:completion-obligations proposal)]
    (is (false? (:admitted? proposal)))
    (is (identical? (first body) (:original proposal)))
    (is (identical? physical (:physical proposal)))
    (is (not (completion/contains-completion? (:source proposal))))
    (is (conversion/canonical-form? conversion))
    (is (= {:source-dtype :double :target-dtype :float :rounding :nearest-even
            :overflow :ieee :source-op 'clojure.core/unchecked-float}
           (second conversion)))
    (is (= 2 (count obligations)))
    (is (every? #(= :pending (:status %)) obligations))
    (is (identical? input (:source (first obligations))))
    (is (identical? boxed (:source (second obligations))))
    (is (= '[boxed nil] (last (:source proposal))))
    (is (= Float/POSITIVE_INFINITY (first ((:implementation physical) Double/MAX_VALUE))))))

(deftest a-result-fact-is-not-the-inputs-builtin-number-proof
  (let [parameters '[x]
        context (completion/bytecode-operand-context {})
        input (completion/complete-value (completion/bytecode-floating-input-contract Double/TYPE context) 'x)
        body [(list 'let* ['converted input] '[converted])]
        physical (emit parameters '[Object] body)]
    (is (= [2.0] ((:implementation physical) (float 2.0))))
    (is (= :generated-body-unsupported
           (try (generated/propose-value-source physical parameters) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest proposals-require-every-original-completion-record
  (let [parameters '[x]
        term (completion/complete-value completion/object-input-contract 'x)
        body [(list 'let* ['value term] '[value])]
        physical (emit parameters '[double] body)
        missing (assoc-in physical [:emission :completion-contexts] {})]
    (is (= :unclosed-generated-module-completion
           (try (generated/propose-value-source missing parameters) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (generated/propose-value-source
                  (assoc physical :input-body (mapv identity body)) parameters)))))

(deftest root-spines-must-be-ssa-and-shadowed-locals-cannot-borrow-root-facts
  (doseq [bindings ['[value 1.0 value 2.0] '[x 1.0]]]
    (let [parameters '[x]
          body [(list 'let* bindings [(last (take-nth 2 bindings))])]
          physical (emit parameters '[double] body)]
      (is (thrown? clojure.lang.ExceptionInfo (generated/propose-value-source physical parameters)))))
  (let [parameters '[x other]
        term (completion/complete-value
              (completion/bytecode-floating-input-contract Double/TYPE (completion/bytecode-operand-context {})) 'x)
        body [(list 'let* ['result (list 'let* '[x other] term)] '[result])]
        physical (emit parameters '[double Object] body)]
    (is (= [2.0] ((:implementation physical) 1.0 (float 2.0))))
    (is (thrown? clojure.lang.ExceptionInfo (generated/propose-value-source physical parameters)))))

(deftest physical-entry-counts-cannot-be-truncated-into-a-proposal
  (let [parameters '[x] body ['(let* [value x] [value])]
        physical (emit parameters '[double] body)
        bridge (bytecode/retained-boxed-entry-bridge (:emission physical))]
    (with-redefs [bytecode/retained-boxed-entry-bridge
                  (constantly (assoc-in bridge [:boxed-entry :parameter-carriers] []))]
      (is (thrown? clojure.lang.ExceptionInfo (generated/propose-value-source physical parameters))))))

(deftest quoted-completion-shaped-data-is-not-a-module-operation
  (let [parameters '[x]
        term (completion/complete-value completion/object-input-contract 'x)
        body [(list 'let* ['value (list 'quote term)] '[value])]
        physical (emit parameters '[double] body)
        proposal (generated/propose-value-source physical parameters)]
    (is (empty? (:completion-obligations proposal)))
    (is (= (first body) (:source proposal)))
    (is (false? (:admitted? proposal)))))

(deftest a-call-proposal-stages-all-arguments-before-pending-wrapper-conversions
  (let [parameters '[value scratch]
        term (completion/complete-value completion/object-input-contract 'value)
        body [(list 'let* ['answer term] '[answer nil])]
        physical (emit parameters '[float doubles] body)
        arguments '[(draw) reference]
        array-class (class (double-array 0))
        facts [{:carrier Double/TYPE :presence :present}
               {:carrier Object :presence :unknown}]
        call (generated/instantiate-value-source physical parameters arguments facts)
        {:keys [argument-plan input-obligations bindings outputs]} call
        [floating reference] input-obligations
        converted (second (:binding floating))]
    (is (false? (:admitted? call)))
    (is (= arguments (mapv second (subvec bindings 0 2))))
    (is (= (mapv :binding input-obligations) (subvec bindings 2 4)))
    (is (every? #(nil? (meta %)) (map first bindings)))
    (is (= :scalar-conversion (:classification floating)))
    (is (conversion/canonical-form? converted))
    (is (= {:source-dtype :double :target-dtype :float :rounding :nearest-even
            :overflow :ieee :source-op 'clojure.core/unchecked-float} (second converted)))
    (is (= (get-in argument-plan [:slots 0 :object-binder]) (nth converted 2)))
    (is (= :reference-check (:classification reference)))
    (is (identical? array-class (get-in reference [:slot :parameter-class])))
    (is (identical? (first facts) (:input-fact floating)))
    (is (= :double (get-in call [:normalization-requirements :scalar-types
                                 (get-in floating [:slot :object-binder])])))
    (is (= :float (get-in call [:normalization-requirements :scalar-types
                                (get-in floating [:slot :input-binder])])))
    (is (= :double (get-in call [:normalization-requirements :array-types
                                 (get-in reference [:slot :input-binder])])))
    (is (every? #(= :pending (:status %)) input-obligations))
    (is (every? completion/canonical-form? (mapcat (juxt :object-context :conversion-context)
                                                  input-obligations)))
    (is (= [false true] (mapv nil? outputs)))
    (let [root (first (:root-bindings call))]
      (is (identical? term (:original-expression root)))
      (is (= 'value (:candidate-expression root)))
      (is (= (first outputs) (:id root))))
    (let [qualification (last (:qualification-construction call))
          alpha (last (:alpha-construction call))
          substitution (last (:substitution-construction call))]
      (is (identical? (get-in call [:proposal :source]) (:original qualification)))
      (is (identical? (:qualified-source call) (:constructed qualification)))
      (is (identical? (:qualified-source call) (:original alpha)))
      (is (identical? (:hygienic-source call) (:constructed alpha)))
      (is (identical? (:hygienic-source call) (:original substitution)))
      (is (identical? (:projected-source call) (:constructed substitution))))
    (is (not (completion/contains-completion? (:projected-source call))))
    (is (= [Float/POSITIVE_INFINITY nil]
           ((:implementation physical) Double/MAX_VALUE nil)))))

(deftest unknown-numeric-inputs-refuse-but-object-identity-remains-pending
  (let [parameters '[x] body ['(let* [result x] [result])]
        floating (emit parameters '[float] body)]
    (doseq [facts [[{:carrier Object}] [{:carrier Boolean/TYPE}]
                  [{:carrier Float/TYPE :builtin-class Double}]
                  [{:carrier Double/TYPE :builtin-class Float}] []]]
      (is (= :generated-body-unsupported
             (try (generated/instantiate-value-source floating parameters '[value] facts) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (let [object (emit parameters '[Object] body)
          call (generated/instantiate-value-source object parameters '[value] [nil])]
      (is (= :reference-identity (get-in call [:input-obligations 0 :classification])))
      (is (= :pending (get-in call [:input-obligations 0 :status])))
      (is (false? (:admitted? call))))))

(deftest call-construction-refuses-partial-spines-and-detached-namespace-contexts
  (let [parameters '[x] body ['(let* [a x b a] [b])]
        physical (emit parameters '[double] body)
        facts [{:carrier Double/TYPE :presence :present}]
        construct #(generated/instantiate-value-source physical parameters '[value] facts)]
    (doseq [transform [#(with-redefs [util/alpha-convert (fn [& _] '(let* [b 1.0] [b]))] (%))
                      #(with-redefs [util/alpha-convert (fn [& _] '(let* [a 1.0 b a] []))] (%))
                      #(with-redefs [util/alpha-convert (fn [& _] '(let* [a 1.0 a 2.0] [a]))] (%))
                      #(with-redefs [inference/qualify-body-symbols (fn [& _] '(let* [] []))] (%))]]
      (is (= :generated-body-unsupported
             (try (transform construct) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (let [detached (create-ns (gensym "raster.detached-module."))]
      (remove-ns (ns-name detached))
      (is (= :generated-body-unsupported
             (try (generated/instantiate-value-source
                   (assoc-in physical [:emission :source-ns] detached) parameters '[value] facts) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest the-real-training-body-produces-one-complete-provisional-spine
  (entry/call-with-definition-inputs
   (fn []
     (let [prepared (reverse/prepare-value+grad #'training/local-loss [0])
           _ (reverse/adopt-prepared-program-inputs! prepared)
           physical (reverse/prepared-bytecode-entry prepared)
           proposal (generated/propose-value-source physical (:params prepared))
           original (first (:input-body physical))
           source (:source proposal)
           obligations (:completion-obligations proposal)
           expected (filter completion/canonical-form? (tree-seq coll? seq original))]
       (is (some? (::generated/value-source-template-candidate prepared)))
       (is (identical? (::generated/value-source-template-candidate prepared)
                       (::generated/value-source-template physical)))
       (is (identical? (::generated/value-source-template physical) (:template proposal)))
       (is (false? (:admitted? proposal)))
       (is (= (count expected) (count obligations)))
       (is (pos? (count obligations)))
       (is (not (completion/contains-completion? source)))
       (is (= (take-nth 2 (second original)) (take-nth 2 (second source))))
       (is (= (last original) (last source)))
       (is (every? #(= :pending (:status %)) obligations))
       (is (identical? prepared (reverse/assert-prepared-program-source! prepared)))))))

(deftest nested-completion-origins-follow-the-staged-argument-not-the-root
  (entry/call-with-definition-inputs
   (fn []
     (let [parameters '[ref n divisor]
           context (completion/bytecode-operand-context {:preserve-source-call-results? true})
           contract (completion/planned-bytecode-input-contract Long/TYPE context :long)
           size (completion/complete-value contract '(clojure.core/quot n divisor))
           interface (types/ensure-fn-interface! '[floats long])
           receiver (with-meta 'raster.arrays/alloc-like_m_floats_long-impl
                      {:tag (:iface-name interface) :raster.type/tag (:iface-name interface)})
           allocation (with-meta (list '.invk receiver 'ref size)
                        {:raster.op/original 'raster.arrays/alloc-like})
           root (completion/complete-value
                 (completion/bytecode-result-context-contract context :ref) allocation)
           physical (emit parameters '[Object long long] [(list 'let* ['out root] '[out])])
           call (generated/instantiate-value-source
                 physical parameters parameters
                 [{:carrier Object :presence :unknown}
                  {:carrier Long/TYPE :presence :present}
                  {:carrier Long/TYPE :presence :present}])
           layout (generated/value-source-binding-layout call)
           root (first (:root-bindings call))
           [reference argument] (get-in root [:argument-construction :arguments])
           source-call (get-in root [:argument-construction :source-call])
           indices (:body-obligation-indices layout)
           obligations (get-in call [:proposal :completion-obligations])
           by-source (fn [source]
                       (keep-indexed #(when (identical? source (:source %2)) %1) obligations))
           staged (emit parameters '[Object long long]
                        [(list 'let* (vec (mapcat identity (:bindings call))) (:outputs call))])
           observe (fn [physical arguments]
                     (try {:lengths (mapv alength (apply (:implementation physical) arguments))}
                          (catch Throwable e {:error (class e)})))]
       (is (some? layout))
       (is (= 1 (count (by-source size))))
       (is (= (:call-binding-index argument) (nth indices (first (by-source size)))))
       (is (= :operand (nth (:template-source-path argument) 2)))
       (is (identical? size
                       (:source-input (bytecode/retained-input-invocation-input-source
                                       (:emission physical) (:input-body physical) source-call 1))))
       (is (nil? (bytecode/retained-input-invocation-input-source
                  (assoc (:emission physical) :completion-catalog {})
                  (:input-body physical) source-call 1)))
       (is (nil? (bytecode/retained-input-invocation-input-source
                  (update (:emission physical) :completion-contexts
                          #(into {} (map (fn [[id record]] [id (assoc record :status :ambiguous)])) %))
                  (:input-body physical) source-call 1)))
       (is (nil? (bytecode/retained-input-completion-context
                  (:emission physical) (:input-body physical)
                  (with-meta (apply list size) (meta size)))))
       (is (< (:call-binding-index reference) (:call-binding-index argument) (:call-binding-index root)))
       (doseq [actual [[(float-array 0) 6 2] [nil 6 2]
                       ["not an array" 6 0] [nil 6 0] [(float-array 0) -6 2]]]
         (is (= (observe physical actual) (observe staged actual))))
       (is (= {:error ClassCastException} (observe staged ["not an array" 6 0])))
       (is (= {:error ArithmeticException} (observe staged [nil 6 0])))
       (is (nil? (generated/value-source-binding-layout
                  (assoc-in call [:proposal :completion-obligations (first (by-source size)) :path]
                            [1 1]))))
       (is (nil? (generated/value-source-binding-layout
                  (assoc-in call [:root-bindings 0 :argument-construction :arguments 1 :template-source-path]
                            (:template-source-path reference)))))))))

(deftest allocation-layout-rejects-detached-source-and-observation-joins
  (entry/call-with-definition-inputs
   (fn []
     (let [prepared (reverse/prepare-value+grad #'training/local-loss [0])
           _ (reverse/adopt-prepared-program-inputs! prepared)
           physical (reverse/prepared-bytecode-entry prepared)
           call (generated/instantiate-value-source
                 physical (:params prepared) '[theta x y rows width]
                 [{:carrier Object :presence :unknown} {:carrier Object :presence :unknown}
                  {:carrier Object :presence :unknown} {:carrier Long/TYPE :presence :present}
                  {:carrier Long/TYPE :presence :present}])
           root-index (first (keep-indexed #(when (:argument-construction %2) %1) (:root-bindings call)))
           path [:root-bindings root-index :argument-construction]
           copy-form (fn [form] (with-meta (apply list form) (meta form)))
           changes [(fn [c] (assoc-in c [:observed :method] nil))
                    (fn [c] (update-in c [:observed :inputs] #(mapv identity %)))
                    (fn [c] (update-in c [:observed :input-completions] #(mapv identity %)))
                    (fn [c] (assoc-in c [:observed :invocation-mode] :local-typed))
                    (fn [c] (update-in c [:observed :receiver] #(with-meta % {:tag 'Object})))
                    (fn [c] (update-in c [:observed :source-input] copy-form))
                    (fn [c] (update c :source-call copy-form))
                    (fn [c] (update c :candidate-call copy-form))
                    (fn [c] (update-in c [:arguments 0 :original-operand] #(with-meta % {:tag 'Object})))
                    (fn [c] (update-in c [:arguments 1 :operand] copy-form))
                    (fn [c] (update-in c [:arguments 1 :completion] #(assoc % :adaptation :identity)))
                    (fn [c] (assoc-in c [:arguments 1 :input-index] 0))
                    (fn [c] (update-in c [:arguments 1 :binding 1] copy-form))
                    (fn [c] (update c :constructed-call
                                    #(util/remake-from % (assoc (vec %) 1 'other/implementation))))]]
       (is (some? (generated/value-source-binding-layout call)))
       (doseq [change changes]
         (is (nil? (generated/value-source-binding-layout (update-in call path change)))))
       (doseq [change [(fn [o] (assoc o :token (Object.)))
                       (fn [o] (update o :source copy-form))
                       (fn [o] (update o :contract #(into {} %)))
                       (fn [o] (update o :path conj :detached))]]
         (is (nil? (generated/value-source-binding-layout
                    (update-in call [:proposal :completion-obligations 0] change)))))))))

(deftest ordered-allocation-inputs-preserve-the-generated-jvm-program
  (entry/call-with-definition-inputs
   (fn []
     (let [prepared (reverse/prepare-value+grad #'training/local-loss [0])
           _ (reverse/adopt-prepared-program-inputs! prepared)
           physical (reverse/prepared-bytecode-entry prepared)
           arguments '[theta x y rows width]
           facts [{:carrier Object :presence :unknown}
                  {:carrier Object :presence :unknown}
                  {:carrier Object :presence :unknown}
                  {:carrier Long/TYPE :presence :present}
                  {:carrier Long/TYPE :presence :present}]
           call (generated/instantiate-value-source physical (:params prepared) arguments facts)
           body [(list 'let* (vec (mapcat identity (:bindings call))) (:outputs call))]
           staged (emit arguments (:tags prepared) body)
           number-value (fn [value]
                          [(class value)
                           (cond (instance? Double value) (Double/doubleToRawLongBits (double value))
                                 (instance? Float value) (Float/floatToRawIntBits (float value))
                                 :else value)])
           observe (fn [physical arguments]
                     (try
                       {:values
                        (mapv (fn [value]
                                (cond
                                  (instance? Number value) (number-value value)
                                  (and value (.isArray (class value))) [(class value) (mapv number-value value)]
                                  :else value))
                              (apply (:implementation physical) arguments))}
                       (catch Throwable e {:error (class e)})))]
       (is (some? (generated/value-source-binding-layout call)))
       (doseq [rows [1 3] change [0 1]
               :let [{:keys [x y]} (first (training/batches [rows]))
                     theta (float-array (map #(+ (double %) (* change 0.125)) (training/initial-theta)))
                     x (float-array (map #(+ (double %) (* change 0.0625)) x))
                     actual [theta x y rows 17]]]
         (is (= (observe physical actual) (observe staged actual))))
       (doseq [actual [[nil (float-array 17) (float-array 1) 1 17]
                       [(float-array 17) nil (float-array 1) 1 17]
                       [(float-array 17) (float-array 17) nil 1 17]
                       [(float-array 17) (float-array 1) (float-array 1) 1 17]
                       [(float-array 17) (float-array 17) (float-array 0) 1 17]
                       [(float-array 17) (float-array 0) (float-array 0) 0 17]]]
         (is (= (observe physical actual) (observe staged actual))))))))

(deftest the-real-training-caller-inserts-the-whole-pending-call-at-its-owned-cut
  (entry/call-with-definition-inputs
   (fn []
     (let [root (core/resolve-deftm-var #'training/local-gradient-sum)
           _ (core/prepare-deftm-frontend! root)
           frontend (entry/original-source-frontend root)
           caller (entry/frontend-body-entry-carriers frontend root)
           normalized (transport/normalize-frontend-bindings frontend (:parameters caller))
           environment (zipmap (:parameters caller) (get-in frontend [:source-entry :tags]))
           seen (atom nil)
           expanded
           (binding [transport/*prepared-program-consumer*
                     (fn [event placement]
                       (let [prepared (:prepared event)
                             physical (reverse/prepared-bytecode-entry prepared)
                             _ (reverse/adopt-prepared-program-inputs! prepared)
                             call (generated/instantiate-value-source
                                   physical (:params prepared) (vec (:arguments event))
                                   (mapv (:parameter-facts caller) (:arguments event)))
                             import {:event event :call call :status :pending :admitted? false}]
                         (reset! seen {:call call :event event :placement placement :import import})
                         {:ports (:outputs call) :bindings (:bindings call) :import import}))]
             (transport/with-source-construction-history
              (:source-normalization normalized)
              #(inline/inline-ad-applications (first (:walked-body normalized)) environment)))
           {:keys [call event placement import]} @seen
           view (transport/source-construction-view (:source-construction expanded)
                                                    (:source-normalization normalized) (:form expanded))
           cut (first (:prepared-program-cuts view))]
       (is (= 1 (count (:prepared-program-cuts view))))
       (is (identical? event (:event cut)))
       (is (identical? placement (:placement cut)))
       (is (identical? import (:import cut)))
       (is (identical? (:bindings call) (:canonical-bindings cut)))
       (is (= 5 (count (:input-obligations call))))
       (is (every? #(= :pending (:status %)) (:input-obligations call)))
       (is (every? #(= :pending (:status %)) (get-in call [:proposal :completion-obligations])))
       (is (false? (:admitted? call)))
       (let [expected (mapcat #(map (comp second :binding)
                                   (get-in % [:argument-construction :arguments]))
                              (:root-bindings call))
             actual (filter completion/canonical-form? (tree-seq coll? seq (:form expanded)))]
         (is (= 6 (count expected) (count actual)))
         (is (every? (fn [node] (some #(identical? node %) expected)) actual)
             "only the exact new pending argument completions remain in the constructed source"))
       (is (some #(and (= :prepared-reverse-ad (first %))
                       (identical? (:occurrence event) (nth % 2 nil)))
                 (:unclosed-source-imports @entry/*emission-inputs*))
           "complete construction still confers no execution or marker closure")))))
