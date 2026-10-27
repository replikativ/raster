(ns raster.compiler.passes.scalar.closed-prefix-effects-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.arrays :as arrays]
            [raster.numeric :as n]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.par]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed-frontend]
            [raster.compiler.passes.scalar.source-import :as source-import]))

(deftm prefix-array [values :- (Array double) n :- Long] :- (Array double)
  values)

(deftm prefix-owner [coefficient :- Float values :- (Array double) n :- Long] :- Object
  values)

(deftm prefix-object-owner [coefficient :- Object values :- (Array double) n :- Long] :- Object
  values)

(deftm prefix-allocation-frame [values :- (Array float) width :- Long] :- (Array float)
  (let [span (n/* width width)
        out (arrays/alloc-like values (n/* width span))]
    out))

(deftm direct-array-count-frame [width :- Long] :- (Array float)
  (let [out (float-array (n/* width width))]
    out))

(defn- prepared-source [function]
  (let [source (core/resolve-deftm-var function)]
    (core/ensure-current-owned-bytecode!
     (ns-resolve (.ns ^clojure.lang.Var source)
                 (symbol (str (:name (meta source)) "-impl"))))
    source))

(defn- prefix-request [function unknown-prefix?]
  (let [source (prepared-source function)
        helper (prepared-source #'prefix-array)
        operation 'raster.compiler.passes.scalar.closed-prefix-effects-test/prefix-array
        implementation (symbol (str (ns-name (.ns ^clojure.lang.Var helper)))
                               (str (:name (meta helper)) "-impl"))
        selected (entry/selected-definition operation implementation 2)
        receiver (with-meta implementation {:raster.type/tag (get-in selected [:entry :typed-iface])})
        call (with-meta (list '.invk receiver 'values 'n) {:raster.op/original operation})
        boxed (completion/complete-value completion/object-input-contract 'coefficient)
        unboxed (completion/complete-value (completion/bytecode-boxed-parameter-contract Float/TYPE) 'boxed)
        bindings (cond-> ['boxed boxed 'unboxed unboxed]
                   unknown-prefix? (into ['unknown '(clojure.core/identity unboxed)]))
        form (list 'let* (into bindings ['result call]) 'result)
        options {:source-var source :source-ns (.ns ^clojure.lang.Var source)
                 :lexical-inputs '[coefficient values n]}]
    {:source form :call call :options options}))

(defn- with-witness [request consume]
  (entry/call-with-definition-inputs
   #(consume (source-import/prepare-contextual-source-witness (:source request) (:options request)))))

(deftest compound-long-argument-frame-keeps-call-order-and-original-allocation
  (let [source (prepared-source #'prefix-allocation-frame)]
    (entry/call-with-definition-inputs
     (fn []
       (let [frontend (entry/original-source-frontend source)
             body (first (:body frontend))
             options {:source-var source :source-ns (.ns ^clojure.lang.Var source)
                      :lexical-inputs (get-in frontend [:emission :params])}
             witness (source-import/prepare-contextual-source-witness body options)
             projected (source-import/project-contextual-invocation-arguments witness {:required? true})]
         (is (= 1 (count (:frames projected))))
         (when projected
           (let [frame (first (:frames projected))
                 before (:original frame)
                 after (:call frame)
                 pairs (mapv vec (partition 2 (second (:form projected))))
                 current (source-import/prepare-contextual-source-witness (:form projected) options)
                 observe (fn [implementation reference width]
                           (try
                             (let [result (implementation reference width)]
                               {:class (class result) :values (vec result)})
                             (catch Throwable error
                               {:error (class (if (instance? java.lang.reflect.InvocationTargetException error)
                                                (.getCause error) error))})))]
             (is (= (take 2 before) (take 2 after)) "the original global call/receiver remain")
             (is (= 2 (count (get-in frame [:frame :bindings]))))
             (is (= [1] (:exported-ordinals frame)))
             (is (= 1 (count (:exported-bindings frame))))
             (is (identical? (nth before 2) (nth after 2))
                 "the inert array argument retains its original source view")
             (is (every? symbol? (drop 2 after)))
             (is (every? completion/canonical-form? (map second (get-in frame [:frame :bindings]))))
             (is (some #(contains? (meta %) :raster.compiler.backend.jvm.bytecode/call-occurrence)
                       (tree-seq coll? seq (get-in frame [:frame :bindings])))
                 "private frame operands retain the original invocation occurrence")
             (is (not-any? #(contains? (meta %) :raster.compiler.backend.jvm.bytecode/call-occurrence)
                           (tree-seq coll? seq (:form projected)))
                 "new source does not borrow private occurrences from its evidence")
             (is (semantic/retained-source-equivalent? (:form projected) (:form projected)))
             (let [observe bytecode/retained-input-source-context]
               (with-redefs [bytecode/retained-input-source-context (fn [& _] nil)]
                 (is (nil? (source-import/project-contextual-invocation-arguments witness))))
               (with-redefs [bytecode/retained-input-source-context
                             (fn [& args]
                               (some-> (apply observe args)
                                       (update :operand #(with-meta (apply list %) (meta %)))))]
                 (is (nil? (source-import/project-contextual-invocation-arguments witness)))
                 "a structurally equal copied operand cannot establish the source/physical edge"))
             (is (= 1 (count (filter #(= 'span (first %)) pairs)))
                 "normalization does not re-evaluate the earlier boxed Long producer")
             (doseq [[reference width] [[(float-array 0) 0] [(float-array 0) 2]
                                      [nil 2] [(float-array 0) -1]
                                      [(float-array 0) Long/MAX_VALUE]
                                      [(float-array 0) 65536]]]
               ;; 65536^3 narrows to zero at JVM newarray: no huge allocation.
               ;; This is source equivalence, not a claim of GPU wrap support.
               (is (= (observe (:implementation witness) reference width)
                      (observe (:implementation current) reference width))))
             (is (nil? (source-import/project-contextual-invocation-arguments current))
                 "completed leaf arguments make this normalization idempotent"))))))))

(deftest direct-array-count-frame-preserves-owned-int-completion-and-constructor-order
  (let [source (prepared-source #'direct-array-count-frame)]
    (entry/call-with-definition-inputs
     (fn []
       (let [frontend (entry/original-source-frontend source)
             body (first (:body frontend))
             options {:source-var source :source-ns (.ns ^clojure.lang.Var source)
                      :lexical-inputs (get-in frontend [:emission :params])}
             witness (source-import/prepare-contextual-source-witness body options)
             projected (source-import/project-contextual-invocation-arguments
                        witness {:required? true})]
         (is (= 1 (count (:frames projected))))
         (when projected
           (let [artifact (first (:frames projected))
                 before (:original artifact)
                 after (:call artifact)
                 physical (:physical-count artifact)
                 rederived (bytecode/retained-input-array-count-context
                            (:emission witness) (:input-body witness) before)
                 [[count-id count-expression]] (get-in artifact [:frame :bindings])
                 projected-count (some (fn [[id expression]]
                                         (when (identical? id count-id) expression))
                                       (partition 2 (second (:form projected))))
                 contract (:contract (completion/canonical-parts count-expression))
                 current (source-import/prepare-contextual-source-witness
                          (:form projected) options)
                 observe (fn [implementation width]
                           (try
                             (let [result (implementation width)]
                               {:class (class result) :length (alength result)})
                             (catch Throwable error
                               {:error (class (if (instance? java.lang.reflect.InvocationTargetException error)
                                                (.getCause error) error))})))]
             (is (= (first before) (first after) 'clojure.core/float-array))
             (is (= 2 (count after)))
             (is (identical? count-id (second after)))
             (is (= :jvm-bytecode-integral-input (:source-context contract)))
             (is (identical? Integer/TYPE (:target-carrier contract)))
             (is (contains? #{:int :long :ref} (:expected-input-carrier contract)))
             (is (= :raw-jvm-int (get-in artifact [:physical-count :completion])))
             (is (= :jvm-zero (get-in artifact [:allocation :initialization])))
             (is (identical? (second before) (:source-original physical)))
             (is (identical? (:source-original physical) (nth count-expression 2)))
             (is (identical? (:operand physical)
                             (get-in artifact [:allocation :count :operand])))
             (is (and (identical? (:source-original physical) (:source-original rederived))
                      (identical? (:owned-source physical) (:owned-source rederived))
                      (identical? (:operand physical) (:operand rederived)))
                 "the frame authority stays in the exact current retained count query")
             (is (and projected-count
                      (not-any? #(contains? (meta %)
                                            :raster.compiler.backend.jvm.bytecode/call-occurrence)
                                (tree-seq coll? seq projected-count)))
                 "only the NEW source completion is detached from old occurrence tokens")
             (doseq [width [0 2 -1 46341 4294967297 65536 Long/MAX_VALUE]]
               (is (= (observe (:implementation witness) width)
                      (observe (:implementation current) width))
                   "zero, negative and wrapped Int counts retain JVM constructor behavior"))
             (is (nil? (source-import/project-contextual-invocation-arguments current))
                 "the named Int count makes direct constructor projection idempotent")
             (let [query bytecode/retained-input-array-count-context
                   attempt (fn []
                             (try
                               (if (nil? (source-import/project-contextual-invocation-arguments witness))
                                 :declined
                                 :accepted)
                               (catch clojure.lang.ExceptionInfo error
                                 (:reason (ex-data error)))))]
               (with-redefs [bytecode/retained-input-array-count-context
                             (fn [& arguments]
                               (some-> (apply query arguments)
                                       (update :operand #(with-meta (apply list %) (meta %)))))]
                 (is (contains? #{:declined :source-input-array-count-correspondence}
                                (attempt))
                     "a copied count expression has no current occurrence authority"))
               (with-redefs [bytecode/retained-input-array-count-context
                             (fn [& arguments]
                               (when-let [record (apply query arguments)]
                                 (let [policy (update (:source-policy record) :unchecked-math not)]
                                   (-> record
                                       (assoc :source-policy policy)
                                       (assoc-in [:allocation :count :source-policy] policy)))))]
                 (is (contains? #{:declined :source-input-array-count-correspondence}
                                (attempt))
                     "a foreign count policy cannot replace the current source policy")))
             (let [metadata (meta source)]
               (try
                 (alter-meta! source assoc ::stale-direct-count true)
                 (is (thrown? clojure.lang.ExceptionInfo
                              (entry/assert-producer-current! (:receipt projected))))
                 (finally (reset-meta! source metadata)))))))))))

(deftest object-number-array-count-does-not-acquire-a-callback-free-long-frame
  (let [source (prepared-source #'prefix-object-owner)]
    (entry/call-with-definition-inputs
     (fn []
       (let [frontend (entry/original-source-frontend source)
             candidate '(let* [out (clojure.core/float-array
                                     (clojure.core/int coefficient))]
                          out)
             options {:source-var source :source-ns (.ns ^clojure.lang.Var source)
                      :lexical-inputs (get-in frontend [:emission :params])}
             witness (source-import/prepare-contextual-source-witness
                      candidate options)]
         (is (nil? (source-import/project-contextual-invocation-arguments witness))
             "Number.intValue may execute user code and is not a physical Long input frame"))))))

(deftest actual-builtin-box-prefix-can-import-a-reference-call-without-erasing-checks
  (let [request (prefix-request #'prefix-owner false)]
    (with-witness
      request
      (fn [witness]
        (let [projected (source-import/project-contextual-reference-helpers witness)
              application (first (:applications projected))
              new-owner (when projected
                          (source-import/prepare-contextual-source-witness (:form projected) (:options request)))]
          (is (= 2 (:prefix-endpoint projected)))
          (is (identical? (:call request) (get-in application [:application :caller-occurrence])))
          (is (= 2 (count (:argument-bindings application))))
          (is (source-import/validate-helper-application! application))
          (when new-owner
            (doseq [array [nil (double-array [1.0 2.0])]]
              (is (identical? array ((:implementation witness) (float 0.25) array 0))))
            (doseq [array [nil (double-array [3.0])]]
              (is (identical? array ((:implementation new-owner) (float 0.25) array 0))))))))))

(deftest object-number-conversion-and-unknown-prefix-do-not-authorize-a-later-helper
  (doseq [[function unknown?] [[#'prefix-object-owner false] [#'prefix-owner true]]]
    (let [request (prefix-request function unknown?)]
      (with-witness request #(is (nil? (source-import/project-contextual-reference-helpers %)))))))

(deftest current-prefix-projection-rejects-copied-witness-source
  (let [request (prefix-request #'prefix-owner false)]
    (with-witness
      request
      (fn [witness]
        (let [copied (with-meta (apply list (:scope witness)) (meta (:scope witness)))]
          (is (= copied (:scope witness)))
          (is (not (identical? copied (:scope witness))))
          (is (thrown? clojure.lang.ExceptionInfo
                       (source-import/project-contextual-reference-helpers (assoc witness :scope copied)))))))))

(deftest repeated-call-object-does-not-acquire-a-unique-prefix-occurrence
  (let [request (prefix-request #'prefix-owner false)
        source (:source request)
        repeated (assoc request :source
                        (list 'let* (into (second source) ['again (:call request)]) 'again))]
    (with-witness repeated #(is (nil? (source-import/project-contextual-reference-helpers %))))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(deftm prefix-mapped [coefficient :- Double values :- (Array double) n :- Long] :- (Array double)
  (par/pmap i n double (* coefficient (aget values i))))

(deftm prefix-callback [values :- (Array double) n :- Long] :- (Array double)
  (do (clojure.core/identity n) values))

(defn- typed-helper-call [function operation arguments]
  (let [helper (prepared-source function)
        implementation (symbol (str (ns-name (.ns ^clojure.lang.Var helper)))
                               (str (:name (meta helper)) "-impl"))
        selected (entry/selected-definition operation implementation (count arguments))
        receiver (with-meta implementation {:raster.type/tag (get-in selected [:entry :typed-iface])})]
    (with-meta (list* '.invk receiver arguments) {:raster.op/original operation})))

(deftest owned-map-crossing-preserves-checks-and-copied-scopes-abstain
  (let [request (prefix-request #'prefix-owner false)
        mapped (typed-helper-call #'prefix-mapped
                                  'raster.compiler.passes.scalar.closed-prefix-effects-test/prefix-mapped
                                  '[unboxed values n])
        bindings (into (subvec (second (:source request)) 0 4) ['mapped mapped 'result (:call request)])
        request (assoc request :source (list 'let* bindings 'result))]
    (with-witness
      request
      (fn [witness]
        (let [projected (source-import/project-contextual-reference-helpers witness)
              application (first (:applications projected))
              frontend (get-in application [:application :frontend])
              emission (entry/observed-frontend-emission frontend)
              spine (bytecode/retained-pmap-spine emission (first (:body emission)))
              scope (nth (get-in spine [:expansion :map]) 5)
              observed (bytecode/retained-binding-context emission scope 0)
              projector (#'source-import/closed-source-effect-projector (java.util.IdentityHashMap.))
              copied (with-meta (apply list scope) (meta scope))
              new-owner (source-import/prepare-contextual-source-witness (:form projected) (:options request))]
          (is (= 2 (count (:applications projected))))
          (is (some? ((:value projector) emission (:before observed) {} scope)))
          (is (= scope copied))
          (is (nil? ((:value projector) emission (:before observed) {} copied)))
          (is (nil? ((:value projector) emission (:before observed) {} '(let* [] values))))
          (doseq [implementation [(:implementation witness) (:implementation new-owner)]]
            (let [array (double-array [1.0 2.0])]
              (is (identical? array (implementation (float 0.25) array 2))))
            (is (thrown? NegativeArraySizeException (implementation (float 0.25) nil -1)))
            (is (nil? (implementation (float 0.25) nil 0)))
            (is (thrown? NullPointerException (implementation (float 0.25) nil 1)))))))))

(deftest unknown-callee-body-cannot-authorize-a-later-helper
  (let [request (prefix-request #'prefix-owner false)
        callback (typed-helper-call #'prefix-callback
                                    'raster.compiler.passes.scalar.closed-prefix-effects-test/prefix-callback
                                    '[values n])
        bindings (into (subvec (second (:source request)) 0 4) ['callback callback 'result (:call request)])
        request (assoc request :source (list 'let* bindings 'result))]
    (with-witness request
      (fn [witness]
        (let [projected (source-import/project-contextual-reference-helpers witness)]
          (is (= 1 (count (:applications projected)))))))))

(deftest standalone-map-consumed-recipe-preserves-order-without-freshness
  (let [request (prefix-request #'prefix-owner false)
        body '(let* [ordinal (clojure.core/long index)]
                (clojure.core/* unboxed (clojure.core/aget values ordinal)))
        mapped (list 'raster.par/map! 'values 'index 'n 'clojure.core/double body)
        request (assoc request :source
                       (list 'let* (into (subvec (second (:source request)) 0 4)
                                        ['mapped mapped 'result (:call request)]) 'result))]
    (with-witness
      request
      (fn [witness]
        (let [emission (:emission witness)
              record (bytecode/retained-input-map-effect-expansion emission (:input-body witness) mapped)
              projector (#'source-import/closed-source-effect-projector (java.util.IdentityHashMap.))
              projected (source-import/project-contextual-reference-helpers witness)
              projected-owner (when projected
                                (source-import/prepare-contextual-source-witness (:form projected) (:options request)))
              actual (:actual-expanded record)
              copied (when actual (with-meta (apply list actual) (meta actual)))]
          (is (= :map-effect (:source-kind record)))
          (is (nil? (:parent-pmap record)))
          (is (not-any? #(contains? record %) [:fresh? :allocation :initialization :array-class]))
          (is (= 1 (count (:applications projected))))
          (is (nil? (bytecode/retained-input-map-effect-expansion
                     emission (:input-body witness) (with-meta (apply list mapped) (meta mapped)))))
          (is (nil? ((:value projector) emission (get-in record [:bound-observation :before]) {} copied)))
          (doseq [implementation (keep :implementation [witness projected-owner])]
            (doseq [array [nil (double-array [2.0 4.0])]]
              (is (identical? array (implementation (float 0.5) array 0))))
            (let [array (double-array [2.0 4.0])]
              (is (identical? array (implementation (float 0.5) array 2)))
              (is (= [1.0 2.0] (vec array))))
            (is (nil? (implementation (float 0.5) nil -1)))
            (is (thrown? NullPointerException (implementation (float 0.5) nil 1)))
            (is (thrown? ArithmeticException (implementation (float 0.5) nil Long/MAX_VALUE))))
          (doseq [bad [(assoc-in emission [:pmap-expansions
                                          (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta (:source record)))
                                          :body] '(clojure.core/identity 1))
                       (assoc-in emission [:pmap-expansions
                                          (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta (:source record)))
                                          :parent-pmap] (Object.))]]
            (is (nil? (bytecode/retained-input-map-effect-expansion bad (:input-body witness) mapped)))))))))

(deftest standalone-map-unknown-body-and-repeated-origin-remain-unproved
  (doseq [unknown? [false true]]
    (let [request (prefix-request #'prefix-owner false)
          body (if unknown?
                 '(let* [ordinal (clojure.core/long index)]
                    (clojure.core/identity (clojure.core/aget values ordinal)))
                 '(let* [ordinal (clojure.core/long index)]
                    (clojure.core/aget values ordinal)))
          mapped (list 'raster.par/map! 'values 'index 'n 'clojure.core/double body)
          bindings (cond-> (into (subvec (second (:source request)) 0 4) ['mapped mapped])
                     (not unknown?) (into ['again mapped]))
          request (assoc request :source (list 'let* (into bindings ['result (:call request)]) 'result))]
      (with-witness request
        (fn [witness]
          (if unknown?
            (do (is (nil? (source-import/project-contextual-reference-helpers witness)))
                ;; Zero trips do not justify unconditional callback freedom.
                (is (nil? ((:implementation witness) (float 0.5) nil 0))))
            (is (nil? (bytecode/retained-input-map-effect-expansion
                       (:emission witness) (:input-body witness) mapped)))))))))

(deftest standard-map-role-observation-does-not-change-expansion-or-count-order
  (let [bound '(do (swap! events inc) n)
        body '(do (swap! events + 10) (aget values index))
        source (list 'raster.par/map! 'values 'index bound 'double body)
        recipe (atom nil)
        [ordinary observed]
        (with-redefs [clojure.core/gensym (fn [_] 'private-count)]
          [(raster.compiler.ir.par/expand-par-map! source)
           (raster.compiler.ir.par/expand-par-map! source identity #(reset! recipe %))])
        execute (binding [*unchecked-math* false]
                  (eval (list 'fn '[values n events] observed)))]
    (is (= ordinary observed))
    (is (identical? observed (:expanded @recipe)))
    (is (identical? source (:source @recipe)))
    (is (identical? bound (:bound @recipe)))
    (is (identical? body (:body @recipe)))
    (doseq [n [-1 0 2]]
      (let [events (atom 0) values (double-array [2.0 3.0])]
        (is (identical? values (execute values n events)))
        (is (= (if (= n 2) 21 1) @events))))
    (let [events (atom 0)]
      (is (thrown? ArithmeticException (execute nil Long/MAX_VALUE events)))
      (is (= 1 @events)))))

(deftm prefix-opaque-array-owner [coefficient :- Float values :- Object n :- Long] :- Object
  values)

(defn- checked-length-request [owner unknown-prefix?]
  (let [request (prefix-request owner unknown-prefix?)
        call (typed-helper-call (ns-resolve 'raster.arrays 'alength_m_doubles)
                                'raster.arrays/alength '[values])
        bindings (vec (drop-last 2 (second (:source request))))]
    (assoc request :source (list 'let* (into bindings ['length (list 'clojure.core/int call)]) 'length))))

(deftest owned-final-int-length-source-retains-array-checks-and-whole-chain
  (let [request (checked-length-request #'prefix-owner false)]
    (with-witness request
      (fn [witness]
        (let [projected (source-import/project-contextual-array-lengths witness)
              replacement (get-in projected [:rewrites 0 :expression])
              transformed (when projected
                            (source-import/prepare-contextual-source-witness (:form projected) (:options request)))]
          (is (= 1 (count (:rewrites projected))))
          (is (= '(clojure.core/int (clojure.core/long (clojure.core/alength values))) replacement))
          (is (identical? (:scope witness) (:source projected)))
          (when projected
            (is (thrown? clojure.lang.ExceptionInfo
                         (#'source-import/validate-owned-import!
                          (assoc projected :form (with-meta (apply list (:form projected)) (meta (:form projected))))
                          :form :copied-length-source)))
            (let [receiver (ns-resolve 'raster.arrays 'alength_m_doubles-impl)]
              (with-redefs-fn {receiver (fn [_] 0)}
                #(is (thrown? clojure.lang.ExceptionInfo
                              (#'source-import/validate-owned-import! projected :form :stale-length-source))))))
          (doseq [implementation (keep :implementation [witness transformed])]
            (is (= 0 (implementation (float 0.25) (double-array 0) 0)))
            (is (= 3 (implementation (float 0.25) (double-array 3) 0)))
            (is (thrown? NullPointerException (implementation (float 0.25) nil 0)))
            (is (thrown? ClassCastException (implementation (float 0.25) (long-array 3) 0)))))))))

(deftest unknown-prefix-or-origin-cannot-canonicalize-a-mutable-length-call
  (doseq [[owner unknown?] [[#'prefix-owner true] [#'prefix-opaque-array-owner false]]]
    (with-witness (checked-length-request owner unknown?)
      #(is (nil? (source-import/project-contextual-array-lengths %))))))

(deftest checked-length-admission-requires-causal-full-capacity-and-keeps-source-action
  (let [expression '(clojure.core/int (clojure.core/long (clojure.core/alength input)))
        scalar (av/tensor {:dtype :int :shape []})
        view (av/tensor {:dtype :double :shape '[(extent input)]})
        capacities {:definition-values {'input view} :available #{'input}}
        description {:id 0 :sym 'length :expr expression :kind :scalar
                     :scalar-dtype :int :before-parallel? false}
        admit (fn [pairs descriptions capacities storage inputs]
                (#'typed-frontend/checked-source-shape-descriptions pairs descriptions capacities storage inputs))]
    (is (= 'input (invocation-source/checked-int-array-length-projection expression scalar {'input view})))
    (is (not (invocation-source/projection-corresponds? expression scalar)))
    (is (identical? expression
                    (invocation-source/canonicalize-shape-projection {'input view 'length scalar} #{} 'length expression)))
    (is (= #{0} (admit [['length expression]] [description] capacities {'input view} '[input])))
    (is (not (#'typed-frontend/supported-description? #{} description)))
    (is (#'typed-frontend/admitted-description? #{} description #{0}))
    (let [normalized-expression (apply list expression)
          normalized (assoc description :expr normalized-expression)
          origins {0 {:source expression :description normalized}}
          project #(#'typed-frontend/checked-source-shape-descriptions
                    [['length expression]] [%] capacities {'input view} '[input] %2)]
      (is (= #{0} (project normalized origins)))
      (is (empty? (project (into {} normalized) origins)))
      (is (empty? (project normalized (assoc-in origins [0 :source] (apply list expression)))))
      (is (empty? (project normalized (assoc-in origins [0 :description] (into {} normalized))))))
    (doseq [[pairs descriptions cap storage inputs]
            [[[['length expression]] [(assoc description :id 1)] capacities {'input view} '[input]]
             [[['length expression]] [(assoc description :expr (apply list expression))] capacities {'input view} '[input]]
             [[['length expression]] [description] capacities {'input view} []]
             [[['length expression]] [description] (assoc capacities :available #{}) {'input view} '[input]]
             [[['length expression]] [description] capacities {'input (assoc view :shape [1])} '[input]]]]
      (is (empty? (admit pairs descriptions cap storage inputs))))
    (doseq [bad ['(clojure.core/int (clojure.core/double (clojure.core/alength input)))
                 '(clojure.core/int (clojure.core/byte (clojure.core/alength input)))
                 '(clojure.core/unchecked-int (clojure.core/long (clojure.core/alength input)))
                 '(clojure.core/int (.invk opaque/length input))]]
      (is (nil? (invocation-source/checked-int-array-length-projection bad scalar {'input view}))))
    (is (nil? (invocation-source/checked-int-array-length-projection
               expression scalar {'input (assoc view :shape [1 2])})))))

(deftest checked-final-int-shape-production-checks-visible-domain-without-allocation
  (let [scalar (av/tensor {:dtype :int :shape []})
        view (av/tensor {:dtype :double :shape '[(extent input)]})
        expression '(clojure.core/int (clojure.core/long (clojure.core/alength input)))
        plan (invocation/from-prefix
              {:id :checked-length-domain :parameters '[input] :parameter-values {'input view}
               :bindings [['length expression]] :binding-values {'length scalar}
               :program-values {'length scalar} :program-inputs '[length] :program-outputs '[length]})
        parameter-id (get-in plan [:parameters 0 :id])
        step-id (get-in plan [:steps 0 :id])
        initial (materialization/source-free-parameter-state
                 (materialization/begin-parameter-materialization plan [(double-array 0)]) #{parameter-id})
        advance (fn [size]
                  (materialization/materialize-next
                   (assoc-in initial [:values parameter-id :shape] [size])
                   (fn [& _] (throw (AssertionError. "shape projection executed scalar source")))))]
    ;; These are conditional view descriptors, not physical storage/readiness
    ;; claims. In particular MAX is checked without allocating a huge buffer.
    (doseq [size [0 2 3 Integer/MAX_VALUE]]
      (is (= size (get-in (advance size) [:values step-id :value]))))
    (is (= :invocation-materialization-scalar
           (try (advance (inc (long Integer/MAX_VALUE))) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (invocation/from-prefix
                  {:id :logical-value-is-not-a-source-view :parameters '[input] :parameter-values {'input view}
                   :bindings [['length '(clojure.core/int (clojure.core/long (clojure.core/alength logical)))]]
                   :binding-values {'length scalar} :program-values {'logical (assoc view :shape [1]) 'length scalar}
                   :program-inputs '[length] :program-outputs '[length]})))))

(deftest actual-reference-return-identity-preserves-only-the-same-array-view-origin
  (let [request (prefix-request #'prefix-owner false)
        array-class (class (double-array 0))
        published (completion/complete-value
                   (completion/bytecode-result-context-contract
                    (completion/bytecode-operand-context {:object-result? true}) :ref) 'fresh)
        checked (completion/complete-value (completion/reference-check-contract array-class) 'published)
        call (typed-helper-call (ns-resolve 'raster.arrays 'alength_m_doubles)
                                'raster.arrays/alength '[checked])
        source (list 'let* ['count '(clojure.core/int n)
                           'fresh '(clojure.core/double-array count)
                           'published published 'checked checked
                           'length (list 'clojure.core/int call)] 'length)
        request (assoc request :source source)]
    (with-witness request
      (fn [witness]
        (let [projected (source-import/project-contextual-array-lengths witness)
              transformed (when projected (source-import/prepare-contextual-source-witness (:form projected) (:options request)))]
          (is (= 1 (count (:rewrites projected))))
          (is (= '(clojure.core/int (clojure.core/long (clojure.core/alength checked)))
                 (get-in projected [:rewrites 0 :expression])))
          (doseq [implementation (keep :implementation [witness transformed])]
            (is (= 0 (implementation (float 0.5) nil 0)))
            (is (= 3 (implementation (float 0.5) nil 3)))
            (is (thrown? NegativeArraySizeException (implementation (float 0.5) nil -1)))
            (is (thrown? ArithmeticException (implementation (float 0.5) nil Long/MAX_VALUE)))))))))

(deftest owned-reduction-crossing-requires-whole-current-control-and-update
  (let [request (prefix-request #'prefix-owner false)
        root '(raster.par/reduce acc 0.0 i n (clojure.core/+ acc (clojure.core/aget values i)))
        reduced (completion/complete-value
                 (completion/bytecode-result-context-contract (completion/bytecode-operand-context {}) :ref)
                 root)
        bindings (into (subvec (second (:source request)) 0 4) ['reduced reduced 'result (:call request)])
        request (assoc request :source (list 'let* bindings 'result))]
    (with-witness
      request
      (fn [witness]
        (let [projection (source-import/project-contextual-reference-helpers witness)
              emission (:emission witness)
              observed (bytecode/retained-input-reduction-emission emission (:input-body witness) root)
              node (get-in observed [:bound :expression])
              locals (get-in observed [:bound :before])
              project (fn [expression]
                        ((:value (#'source-import/closed-source-effect-projector (java.util.IdentityHashMap.)))
                         emission locals {} expression))
              new-owner (source-import/prepare-contextual-source-witness (:form projection) (:options request))
              observe bytecode/retained-reduction-emission]
          (is (= 1 (count (:applications projection))))
          (is (some? (project node)))
          (is (nil? (project (with-meta (apply list node) (meta node)))))
          ;; Negative controls alter independently supplied observation facets;
          ;; no unchecked record may classify the source loop by shape alone.
          (doseq [change [#(assoc-in % [:exit :control :kind] :unknown)
                          #(assoc-in % [:loop :initializers 1 :initializer] '(clojure.core/identity 0.0))
                          #(assoc-in % [:recur :values 1 :operand] '(clojure.core/identity acc))
                          #(assoc % :root-result nil)]]
            (with-redefs [bytecode/retained-reduction-emission
                          (fn [module source] (some-> (observe module source) change))]
              (is (nil? (project node)))))
          (doseq [implementation [(:implementation witness) (:implementation new-owner)]]
            (let [values (double-array [1.0 -2.0 3.0])]
              (is (identical? values (implementation (float 0.25) values 3))))
            (is (nil? (implementation (float 0.25) nil 0)))
            (is (thrown? NullPointerException (implementation (float 0.25) nil 1)))))))))
