(ns raster.compiler.passes.scalar.checked-input-capacity-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays]
            [raster.core :as core :refer [deftm defvalue]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.checked-inputs :as checked]
            [raster.compiler.core.inline-call-plan :as call-plan]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.scalar.host-abstract-value :as host]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.pipeline :as pipeline]))

(defn- length-call [argument]
  (let [implementation 'raster.arrays/alength_m_doubles-impl
        selection (entry/selected-definition 'raster.arrays/alength implementation 1)]
    (with-meta (list '.invk (with-meta implementation {:raster.type/tag (get-in selection [:entry :typed-iface])}) argument)
      {:raster.op/original 'raster.arrays/alength})))

(defn- source [checked-first?]
  (let [field (with-meta 'field {:raster.type/tag 'doubles})
        read (with-meta '(.payload container) {:raster.type/tag 'doubles})
        write (with-meta '(raster.par/map! field i 1 nil (clojure.core/aget field i)) {:elem-type :double})
        later ['size (length-call 'field)]
        effect ['written write]]
    (list 'let* (into [field read] (if checked-first? (into later effect) (into effect later))) 'written)))

(defn- reason [construct]
  (try (construct) nil (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))

(deftest checked-array-inputs-use-actual-interface-and-never-known-or-future-seeds
  (entry/call-with-definition-inputs
   (fn []
     (let [expression (length-call 'field)
           fact (get (checked/array-refinements expression #{'field} #{} {}) 'field)
           select! entry/selected-parameter-carriers calls (atom 0)]
       (is (= :double (:dtype fact)))
       (is (= '[(extent field)] (:shape fact)))
       (is (not (true? (get-in fact [:attributes :present]))))
       (with-redefs [entry/selected-parameter-carriers (fn [& args] (swap! calls inc) (apply select! args))]
         (doseq [[available excluded values]
                 [[#{} #{} {}] [#{'field} #{'field} {}]
                  [#{'field} #{} {'field (av/tensor {:dtype :float :shape [3]})}]
                  [#{'field} #{} {'field (av/tensor {:dtype :long :shape []})}]]]
           (is (empty? (checked/array-refinements expression available excluded values))))
         (is (zero? @calls) "no physical signature queries without a new unknown actual"))
       (is (empty? (checked/array-refinements (with-meta expression {:raster.op/lexical-callee true})
                                             #{'field} #{} {})))
       (is (empty? (checked/array-refinements expression #{'field (second expression)} #{} {})))
       (let [original-find find-ns]
         (with-redefs [clojure.core/find-ns
                       (fn [name] (when-not (= name 'raster.compiler.backend.jvm.bytecode) (original-find name)))]
           (is (empty? (checked/array-refinements expression #{'field} #{} {})))
           (is (nil? (find-ns 'raster.compiler.backend.jvm.bytecode)))))))))

(deftest source-capacity-refines-only-after-the-dominating-checked-call
  (entry/call-with-definition-inputs
   (fn []
     (let [options {:lexical-inputs '[container]
                    :array-types {'field :double}
                    :reference-aliases {'written 'field}}
           analysis (host/analyze-capacities (source true) options)]
       (is (nil? (get-in analysis [:definition-values 'field]))
           "the Object field result is NOT retroactively an array definition")
       (is (= :double (get-in analysis [:values 'field :dtype])))
       (is (= ['size] (get-in analysis [:definition-values 'written :shape])))
       (is (= :host-abstract-value-reference-alias
              (reason #(host/analyze-capacities (source false) options))))
       (let [conditional (list 'let*
                               ['field '(.payload container)
                                'size (list 'if 'condition (length-call 'field) 0)
                                'written '(unknown/write! field)] 'written)]
         (is (= :host-abstract-value-reference-alias
                (reason #(host/analyze-capacities conditional options)))))))))

(deftest frontend-retains-precheck-and-successor-boundaries-without-global-seeding
  (entry/call-with-definition-inputs
   (fn []
     (let [options {:dtype :double :lexical-inputs '[container] :array-types {'field :double}}
           construction (frontend/form->program-components (source true) (assoc options :segmented-plans? true))
           before (:preceding-values construction)]
       (is (some? (:typed-program construction)))
       (is (nil? (get-in before [0 'field])))
       (is (nil? (get-in before [1 'field])))
       (is (= :double (get-in before [2 'field :dtype])))
       (is (= '[(extent field)] (get-in before [2 'field :shape])))
       (is (= :host-abstract-value-reference-alias
              (reason #(frontend/form->program (source false) options))))))))

(deftest retagged-or-alias-interface-hints-never-enter-the-resolved-class-planner
  (entry/call-with-definition-inputs
   (fn []
     (let [expression (length-call 'field) calls (atom 0)
           receiver (second expression)]
       (with-redefs [bytecode/resolved-typed-invocation-plan
                     (fn [& _] (swap! calls inc) (throw (Exception. "untrusted Class hint was resolved")))]
         (doseq [hint ['raster.fn.IFn__MaliciousInitializer 'alias/IFn__doubles]]
           (is (empty? (checked/array-refinements
                        (list '.invk (with-meta receiver {:raster.type/tag hint}) 'field)
                        #{'field} #{} {}))))
         (is (zero? @calls)))))))

(deftest duplicate-actuals-share-one-observed-call-and-one-refinement
  (entry/call-with-definition-inputs
   (fn []
     (let [implementation (symbol "raster.arrays" (str (types/mangle 'acopy! '[doubles long doubles long long]) "-impl"))
           selection (entry/selected-definition 'raster.arrays/acopy! implementation 5)
           receiver (with-meta implementation {:raster.type/tag (get-in selection [:entry :typed-iface])})
           expression (list '.invk receiver 'field 0 'field 0 0)
           select! entry/selected-parameter-carriers calls (atom 0)]
       (with-redefs [entry/selected-parameter-carriers (fn [& args] (swap! calls inc) (apply select! args))]
         (is (= #{'field} (set (keys (checked/array-refinements expression #{'field} #{} {})))))
         (is (= 1 @calls)))))))

(defvalue CheckedFieldContainer [payload :- (Array double)])

(deftype ObjectFieldContainer [payload])

(deftm nullable-array-helper [values :- (Array double)] :- Long 7)

(deftm bounded-reference-count-helper [values :- (Array double)] :- Long 7)

(deftm bounded-reference-count-caller [values :- (Array double)] :- Long
  (let [result (bounded-reference-count-helper values)] result))

(deftest whole-call-plan-really-expands-a-proved-builtin-reference-call
  (let [values (double-array [1.0 2.0])
        _ (is (= 7 (bounded-reference-count-caller values)))
        source-var (core/resolve-deftm-var #'bounded-reference-count-caller)
        facts (pipeline/deftm-frontend-facts source-var :double)
        source (first (pipeline/get-walked-body source-var :double))
        options {:source-var source-var :source-ns (:source-ns facts)
                 :active-params (:parameters facts)}
        successes (atom 0) whole! call-plan/whole-call]
    (with-redefs [call-plan/whole-call
                  (fn [& args]
                    (let [plan (apply whole! args)]
                      (when plan (swap! successes inc)) plan))]
      (entry/call-with-definition-inputs
       #(inline/expand-for-backends source 0 (:param-env facts) options))
      (is (pos? @successes) "recursive invk route acquired genuine full entry/body/exit evidence")
      (reset! successes 0)
      (let [compiled (pipeline/compile-aot #'bounded-reference-count-caller :simd? false)]
        (is (pos? @successes) "public pipeline did not merely keep every call")
        (is (= 7 (compiled values)))
        (is (= [1.0 2.0] (vec values)))))))

(defmacro define-object-field-source [function-name operation implementation & [numerical-consumer?]]
  (let [selection (entry/selected-definition operation implementation 1)
        receiver (with-meta implementation {:raster.type/tag (get-in selection [:entry :typed-iface])})
        call (list '.invk receiver 'values)]
    (list 'raster.core/deftm function-name
          (cond-> ['container ':- 'ObjectFieldContainer 'events ':- '(Array long)]
            numerical-consumer? (into ['count ':- 'Long])) ':- 'Long
          (list 'let (into ['values '(.payload container)
                           'before '(clojure.core/aset events 0 (long 1))
                           'result call]
                          (concat
                           (when numerical-consumer?
                             ['written (with-meta
                                         (list 'raster.par/map!
                                               (with-meta 'values {:raster.type/tag 'doubles})
                                               'i 'count 'double '(double i))
                                         {:elem-type :double})])
                           ['after '(clojure.core/aset events 1 (long 1))]))
                '(long result)))))

(declare object-field-order! object-nullable-order!)

(define-object-field-source object-field-order! raster.arrays/alength raster.arrays/alength_m_doubles-impl)

(define-object-field-source object-nullable-order!
  raster.compiler.passes.scalar.checked-input-capacity-test/nullable-array-helper
  raster.compiler.passes.scalar.checked-input-capacity-test/nullable-array-helper_m_doubles-impl)

(declare object-nullable-zero-map-order!)

(define-object-field-source object-nullable-zero-map-order!
  raster.compiler.passes.scalar.checked-input-capacity-test/nullable-array-helper
  raster.compiler.passes.scalar.checked-input-capacity-test/nullable-array-helper_m_doubles-impl true)

(deftest forged-length-metadata-cannot-turn-a-nullable-helper-into-capacity
  (entry/call-with-definition-inputs
   (fn []
     (let [operation 'raster.compiler.passes.scalar.checked-input-capacity-test/nullable-array-helper
           implementation 'raster.compiler.passes.scalar.checked-input-capacity-test/nullable-array-helper_m_doubles-impl
           selection (entry/selected-definition operation implementation 1)
           receiver (with-meta implementation {:raster.type/tag (get-in selection [:entry :typed-iface])})
           forged (with-meta (list '.invk receiver 'field) {:raster.op/original 'raster.arrays/alength})]
       (is (empty? (checked/array-refinements forged #{'field} #{} {})))
       (is (= :host-abstract-value-reference-alias
              (reason #(host/analyze-capacities
                        (list 'let* ['field '(.payload container) 'size forged 'written '(unknown/write! field)] 'written)
                        {:lexical-inputs '[container] :reference-aliases {'written 'field}}))))))))

(deftm checked-field-order! [container :- CheckedFieldContainer events :- (Array long)] :- Long
  (let [values (.payload container)
        before (clojure.core/aset events 0 (long 1))
        size (raster.arrays/alength values)
        after (clojure.core/aset events 1 (long 1))]
    (long size)))

(deftm nullable-helper-order! [container :- CheckedFieldContainer events :- (Array long)] :- Long
  (let [values (.payload container)
        before (clojure.core/aset events 0 (long 1))
        result (nullable-array-helper values)
        after (clojure.core/aset events 1 (long 1))]
    (long result)))

(deftest checked-field-null-and-wrong-payload-keep-source-effect-order
  (let [compiled (pipeline/compile-aot #'checked-field-order! :simd? false)
        ctor (first (.getConstructors CheckedFieldContainer))]
    (doseq [[payload exception expected-events]
            [[nil NullPointerException [1 0]] ["not an array" ClassCastException [0 0]]]]
      (let [container (.newInstance ^java.lang.reflect.Constructor ctor (object-array [payload]))
            original-events (long-array 2) compiled-events (long-array 2)]
        (is (= exception (try (checked-field-order! container original-events) nil
                             (catch Throwable failure (class failure)))))
        (is (= exception (try (compiled container compiled-events) nil
                             (catch Throwable failure (class failure)))))
        (is (= expected-events (vec original-events) (vec compiled-events)))))
    (let [events (long-array 2)]
      (is (thrown? NullPointerException (compiled nil events)))
      (is (= [0 0] (vec events))))
    (is (= 7 (nullable-array-helper nil)) "an array invocation formal alone is NOT nonnull proof")
    (let [nullable (pipeline/compile-aot #'nullable-helper-order! :simd? false :inline? false)
          container (.newInstance ^java.lang.reflect.Constructor ctor (object-array [nil]))
          original-events (long-array 2) compiled-events (long-array 2)]
      (is (= 7 (nullable-helper-order! container original-events) (nullable container compiled-events)))
      (is (= [1 1] (vec original-events) (vec compiled-events))))))

(deftest original-object-field-effects-precede-the-actual-typed-input-check
  (let [compiled (pipeline/compile-aot #'object-field-order! :simd? false :inline? false)
        nullable (pipeline/compile-aot #'object-nullable-order! :simd? false :inline? false)]
    (doseq [[payload exception] [[nil NullPointerException] ["wrong" ClassCastException]]]
      (let [value (ObjectFieldContainer. payload)
            original-events (long-array 2) compiled-events (long-array 2)]
        (is (= exception (try (object-field-order! value original-events) nil
                             (catch Throwable failure (class failure)))))
        (is (= exception (try (compiled value compiled-events) nil
                             (catch Throwable failure (class failure)))))
        (is (= [1 0] (vec original-events) (vec compiled-events)))))
    (let [value (ObjectFieldContainer. nil)
          original-events (long-array 2) compiled-events (long-array 2)]
      (is (= 7 (object-nullable-order! value original-events) (nullable value compiled-events)))
      (is (= [1 1] (vec original-events) (vec compiled-events))))
    (let [value (ObjectFieldContainer. "wrong")
          original-events (long-array 2) compiled-events (long-array 2)]
      (is (= ClassCastException
             (try (object-nullable-order! value original-events) nil
                  (catch Throwable failure (class failure)))))
      (is (= ClassCastException
             (try (nullable value compiled-events) nil
                  (catch Throwable failure (class failure)))))
      (is (= [1 0] (vec original-events) (vec compiled-events))))))

(deftest nullable-checked-input-does-not-evaluate-capacity-for-a-zero-iteration-consumer
  ;; Count is a runtime argument, so compilation cannot simply fold a literal
  ;; empty loop away. An accepted null reference needs no array access at count0.
  (let [value (ObjectFieldContainer. nil)
        original-events (long-array 2)
        compiled-events (long-array 2)
        compiled (pipeline/compile-aot #'object-nullable-zero-map-order! :simd? false :inline? false)]
    (is (= 7 (object-nullable-zero-map-order! value original-events 0)))
    (is (= [1 1] (vec original-events)))
    (is (= 7 (compiled value compiled-events 0)))
    (is (= [1 1] (vec compiled-events)))))
