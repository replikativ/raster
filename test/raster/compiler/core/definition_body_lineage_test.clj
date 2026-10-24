(ns raster.compiler.core.definition-body-lineage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.passes.scalar.normal-return :as normal]))

(deftm typed-null [x :- Double] :- Object nil)
(deftm plain-null [x :- Object] :- Object nil)
(deftm plain-array [x :- Object] :- Object (clojure.core/double-array 0))
(deftm plain-borrow [x :- Object] :- Object x)
(deftm bare-array [n :- Long] :- (Array double) (double-array n))

(defn- selected [generic]
  (let [method (first (get (core/methods-of generic) 1))
        source-name (types/mangle (:name (meta generic)) (:tags method))
        ns-name (str (:mangled-ns method))
        source (find-var (symbol ns-name (str source-name)))
        receiver (when (:typed-impl method)
                   (find-var (symbol ns-name (str source-name "-impl"))))
        operation (symbol (str (:ns (meta generic))) (str (:name (meta generic))))]
    {:source source :receiver receiver
     :call (if receiver
             (with-meta (list '.invk (symbol ns-name (str source-name "-impl")) 'input)
               {:raster.op/original operation})
             (list (symbol ns-name (str source-name)) 'input))}))

(defn- fact [expression & [env]]
  (entry/call-with-definition-inputs #(normal/result-fact expression env)))

(defn- stale-source-data [operation]
  (try (operation) nil
       (catch clojure.lang.ExceptionInfo failure
         (select-keys (ex-data failure) [:reason :source]))))

(defn- product-outcome [call]
  (let [body (list 'let* ['result call]
                   '(if (clojure.core/nil? result) :absent :present))
        lowered (entry/call-with-definition-inputs #(products/lower-local-products body))
        run (fn [expression] ((eval (list 'fn* '[input] expression)) 2.0))]
    [(run body) (run lowered)]))

(deftest changed-source-metadata-cannot-certify-old-code
  (doseq [generic [#'typed-null #'plain-null]]
    (let [{:keys [source receiver call]} (selected generic)
          original (meta source)]
      (when receiver (core/ensure-bytecode-implementation! receiver))
      (is (= [:absent :absent] (product-outcome call)))
      (try
        (alter-meta! source assoc :raster.core/deftm-source-body
                     '[(clojure.core/double-array 0)])
        (if receiver
          (do
            (is (= {:reason :stale-definition-proof-input :source source}
                   (stale-source-data #(fact call))))
            (is (= {:reason :stale-definition-proof-input :source source}
                   (stale-source-data #(product-outcome call)))))
          (do
            ;; Plain Clojure's optional lineage reader abstains rather than
            ;; consuming a contradictory typed frontend owner.
            (is (= {:presence :unknown} (fact call)))
            (is (= [:absent :absent] (product-outcome call)))))
        (is (nil? ((eval (list 'fn* '[input] call)) 2.0))
            "the unchanged executable still returns nil; only its claimed source changed")
        (finally (reset-meta! source original))))))

(deftest mutable-walked-metadata-is-not-the-executable-body
  (doseq [generic [#'typed-null #'plain-null]]
    (let [{:keys [source receiver call]} (selected generic)
          original (meta source)]
      (when receiver (core/ensure-bytecode-implementation! receiver))
      (try
        (alter-meta! source assoc :raster.core/deftm-walked-body
                     '[(clojure.core/double-array 0)])
        (is (= {:presence :absent} (fact call)))
        (is (= [:absent :absent] (product-outcome call)))
        (finally (reset-meta! source original))))))

(deftest direct-declarations-preserve-real-allocation-and-alias-proofs
  (is (= {:presence :present :array-tag 'doubles}
         (fact (:call (selected #'plain-array)))))
  (let [{:keys [source call]} (selected #'plain-borrow)
        original @source]
    (is (= {:presence :present} (fact call {'input {:presence :present}})))
    (is (= {:presence :absent} (fact call {'input {:presence :absent}})))
    (is (= {:presence :unknown} (fact call)))
    (try
      (alter-var-root source (constantly (fn [_] (double-array 0))))
      (is (= {:presence :unknown} (fact call)))
      (finally (alter-var-root source (constantly original))))))

(deftest missing-or-mismatched-plain-lineage-abstains
  (let [{:keys [source call]} (selected #'plain-array)
        original (meta source)]
    (try
      (doseq [metadata [(dissoc original :raster.core/clojure-source-entry)
                        (assoc-in original [:raster.core/clojure-source-entry :source-root]
                                  (fn [_] (double-array 0)))
                        (assoc-in original [:raster.core/clojure-source-entry :source-var]
                                  #'plain-null)]]
        (reset-meta! source metadata)
        (is (= {:presence :unknown} (fact call))))
      (finally (reset-meta! source original)))))

(deftest bare-constructor-proofs-do-not-depend-on-a-mutable-walk-cache
  (let [scope (create-ns (gensym "raster.cold-constructor-lineage."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm bare-array [n :- Long] :- (Array double) (double-array n))))
      (let [{:keys [source receiver call]} (selected (ns-resolve scope 'bare-array))
            source-before (meta source) receiver-before (meta receiver)
            implementation (var-get receiver)]
        (is (= {:presence :unknown} (fact call)) "a query does not prepare its producer")
        (is (identical? source-before (meta source)))
        (is (identical? receiver-before (meta receiver)))
        (core/ensure-walked-body! source)
        (is (= {:presence :unknown} (fact call)) "a mutable walk cache is not a frozen producer")
        (with-redefs [core/ensure-bytecode-implementation!
                      (fn [& _] (throw (AssertionError. "source query invoked the JIT")))]
          (core/prepare-deftm-frontend! source)
          (is (= {:presence :present :array-tag 'doubles} (fact call))))
        (is (identical? implementation (var-get receiver)))
        (let [original (meta source)]
          (try
            (alter-meta! source assoc :raster.core/deftm-walked-body '[nil])
            (is (= {:presence :present :array-tag 'doubles} (fact call))
                "the authenticated body does not come from the mutable cache")
            (finally (reset-meta! source original))))
        (core/ensure-bytecode-implementation! receiver)
        (is (= {:presence :present :array-tag 'doubles} (fact call))
            "receiver publication retains the same frozen semantic body"))
      (finally (remove-ns (ns-name scope)))))
  (let [operation 'raster.arrays/zeros-like
        signature (inference/registered-call-signature operation '[doubles long])
        source (find-var (:mangled-sym signature))
        receiver-name (symbol (namespace (:mangled-sym signature))
                              (str (name (:mangled-sym signature)) "-impl"))
        receiver (find-var receiver-name)
        call (with-meta (list '.invk receiver-name 'input 'count)
               {:raster.op/original operation})]
    (core/prepare-deftm-frontend! source)
    (is (= {:presence :present :array-tag 'doubles} (fact call)))
    (core/ensure-walked-body! source)
    (is (= {:presence :present :array-tag 'doubles} (fact call)))
    (core/ensure-bytecode-implementation! receiver)
    (is (= {:presence :present :array-tag 'doubles} (fact call)))
    (is (zero? (alength (arrays/zeros-like (double-array 0) 0))))))

(deftest nested-cold-dependencies-retain-real-dispatch-syntax-and-values
  (binding [*ns* (the-ns 'raster.compiler.core.definition-body-lineage-test)]
    (eval '(deftm cold-leaf [n :- Long] :- (Array double) (double-array n)))
    (eval '(deftm cold-parent [n :- Long] :- (Array double)
             (let [first-array (cold-leaf n) second-array (cold-leaf n)] second-array))))
  (let [{:keys [receiver call]} (selected (find-var
                                         'raster.compiler.core.definition-body-lineage-test/cold-parent))
        lookup entry/observed-body-operation]
    (core/ensure-bytecode-implementation! receiver)
    (with-redefs [entry/observed-body-operation
                  (fn [context operation locals]
                    (when (= '.invk operation)
                      (throw (AssertionError. "dispatch is not an ordinary source lookup")))
                    (lookup context operation locals))]
      (is (= {:presence :present :array-tag 'doubles} (fact call))))
    (doseq [n [0 3 1]]
      (let [actual (clojure.lang.Reflector/invokeInstanceMethod
                    @receiver "invk" (object-array [(long n)]))]
        (is (= (vec (double-array n)) (vec actual)))))
    (is (thrown? NegativeArraySizeException
                 (clojure.lang.Reflector/invokeInstanceMethod
                  @receiver "invk" (object-array [(long -1)]))))))

(deftest delayed-macro-expansion-retains-its-own-bodies
  (binding [*ns* (the-ns 'raster.compiler.core.definition-body-lineage-test)]
    (doseq [annotation '[Object Double]]
      (let [name (if (= annotation 'Object) 'delayed-plain 'delayed-typed)
            declaration (fn [body]
                          (list 'raster.core/deftm name ['x :- annotation] :- 'Object body))
            original (macroexpand-1 (declaration nil))
            ;; Same mangled name, expanded before A is evaluated. This must
            ;; not replace A's raw body, Clojure body or lazy JIT specification.
            _ (macroexpand-1 (declaration '(clojure.core/double-array 0)))]
        (eval original)
        (let [{:keys [source receiver call]} (selected (ns-resolve *ns* name))]
          (when receiver (core/prepare-deftm-frontend! source))
          (is (= {:presence :absent} (fact call)))
          (is (= [:absent :absent] (product-outcome call)))
          (when receiver
            (core/ensure-bytecode-implementation! receiver)
            (is (= {:presence :absent} (fact call)))
            (is (= [:absent :absent] (product-outcome call)))))))))

(deftest declarations-with-the-same-name-in-distinct-namespaces-do-not-share-bodies
  (let [left (or (find-ns 'raster.internal.lineage-left)
                 (create-ns 'raster.internal.lineage-left))
        right (or (find-ns 'raster.internal.lineage-right)
                  (create-ns 'raster.internal.lineage-right))
        expand (fn [source-ns body]
                 (binding [*ns* source-ns]
                   (refer 'clojure.core)
                   (macroexpand-1
                    (list 'raster.core/deftm 'same-name '[x :- Object] :- 'Object body))))
        original (expand left nil)
        _ (expand right '(clojure.core/double-array 0))]
    (binding [*ns* left] (eval original))
    (let [call (:call (selected (ns-resolve left 'same-name)))]
      (is (= {:presence :absent} (fact call)))
      (is (= [:absent :absent] (product-outcome call))))))
