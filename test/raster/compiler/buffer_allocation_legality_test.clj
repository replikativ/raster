(ns raster.compiler.buffer-allocation-legality-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.arrays :as arrays]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.hoist :as hoist]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.scalar.buffer-fuse :as buffer-fuse]))

(defn- with-source [f]
  (let [source (create-ns (gensym "raster.test.buffer_legality_"))]
    (try
      (binding [*ns* source]
        (refer 'clojure.core)
        (eval '(def calls (atom 0)))
        (eval '(def failure (ex-info "size failure" {:test :size})))
        (eval '(defn ^{:tag 'long} tick [n] (swap! calls inc) n))
        (eval '(defn ^{:tag 'long} fail [n] (swap! calls inc) (throw failure)))
        (f source))
      (finally (remove-ns (ns-name source))))))

(defn- operation [source name tags]
  (symbol (str (ns-name source)) (str name "_m_" tags)))

(defn- define-allocator [name params body]
  (eval (list 'raster.core/deftm name params ':- '(Array double) body)))

(defn- call-form [op arguments]
  ;; These fixture operations return the supplied Long, and global-size is
  ;; published only as a Long. Retain that source fact as the typed pipeline
  ;; does; unknown carriers are tested separately below.
  (let [arguments (map #(if (or (= 'global-size %)
                               (and (seq? %) (contains? '#{tick record-value record-failure} (first %))))
                         (with-meta % (assoc (meta %) :raster.type/tag 'long)) %) arguments)]
    (list 'let* ['result (apply list op arguments)] 'result)))

(deftest allocation-literal-evidence-retains-physical-carriers
  (doseq [[value expected] [[(long 1) :long] [(int 1) :int] [(byte 1) :int]
                           [(short 1) :int] [(float 1) :float] [(double 1) :double]]]
    (is (= expected (:dtype (descriptor/allocation-literal-argument value)))))
  (doseq [value [(bigint 1) (biginteger 1) 1/3 (bigdec 1)]
          expression [value (list 'clojure.core/long value)]]
    (is (nil? (descriptor/allocation-literal-argument expression)))))

(deftest direct-allocation-projection-is-derived-from-the-selected-definition
  (with-source
    (fn [source]
      (define-allocator 'pure-like '[ref :- (Array double) n :- Long]
        '(double-array (clojure.core/long n)))
      (let [op (operation source "pure-like" "doubles_long")
            implementation (symbol (namespace op) (str (name op) "-impl"))
            make-call (fn [arguments]
                        (with-meta (apply list '.invk implementation arguments)
                          {:raster.op/original (symbol (namespace op) "pure-like")}))
            call (make-call '[x cnt])
            values {'x {:dtype :double :shape ['m]} 'cnt {:dtype :long :shape []}}
            projection (method-entry/call-with-definition-inputs
                        #(invocation-source/required-allocation-source! call values))]
        (is (= '(clojure.core/double-array (clojure.core/long cnt)) (:constructor projection))
            "projection retains the conversion authored in the actual helper body")
        (is (= implementation (get-in projection [:helper :implementation])))
        (is (= ['x 'cnt] (get-in projection [:helper :arguments])))
        (doseq [literal [(long 1) '(clojure.core/long 1) '(clojure.core/long (clojure.core/int 1))
                         (list 'clojure.core/long (int 1))
                         (list 'clojure.core/long (byte 1))
                         (list 'clojure.core/long (short 1))]]
          (let [literal-projection (descriptor/direct-allocation-projection (make-call ['x literal]) values)]
            (is (= (list 'clojure.core/double-array (list 'clojure.core/long literal))
                   (:constructor literal-projection)))
            (is (= ['x literal] (get-in literal-projection [:helper :arguments])))))
        (doseq [unsupported [(int 1) (byte 1) (short 1) (float 1.0) (double 1.0)
                             (bigint 1) (biginteger 1)
                             (list 'clojure.core/long (bigint 1))
                             (list 'clojure.core/long (biginteger 1))
                             '(clojure.core/int 1) '(long 1)
                             '(clojure.core/long (tick 1)) '(clojure.core/+ 0 1)
                             '(clojure.core/long 1.0E100)
                             (with-meta '(clojure.core/long 1)
                               {:raster.effect/preserve-evaluation true})]]
          (is (nil? (descriptor/direct-allocation-projection (make-call ['x unsupported]) values))
              "literal admission neither adapts the actual carrier nor executes an open expression"))
        (doseq [selected [(ns-resolve source (symbol (name implementation)))
                          (ns-resolve source (symbol (name op)))]]
          (with-redefs-fn {selected (fn [& _] (double-array 0))}
            #(is (nil? (descriptor/direct-allocation-projection call values))
                 "a replaced receiver or delegated source is not the registered helper definition")))
        (is (nil? (descriptor/direct-allocation-projection (make-call '[(do (tick 1) x) cnt]) values))
            "an unused argument's effects cannot disappear during constructor projection")
        (is (nil? (descriptor/direct-allocation-projection (make-call '[x (tick cnt)]) values))
            "size argument checks must be evaluated once in an explicit prefix first")
        (is (nil? (descriptor/direct-allocation-projection call (assoc-in values ['x :dtype] :float))))
        (is (nil? (descriptor/direct-allocation-projection call (assoc-in values ['cnt :dtype] :double))))
        (is (nil? (descriptor/direct-allocation-projection call (dissoc values 'x))))
        (is (nil? (descriptor/direct-allocation-projection (with-meta call {:raster.op/lexical-callee true}) values)))
        (define-allocator 'pure-like '[ref :- (Array double) n :- Long]
          '(let [out (double-array n)] (aset out 0 9.0) out))
        (is (nil? (descriptor/direct-allocation-projection call values))
            "an alloc/compute helper is not a pure constructor, even after a previous projection"))
      (define-allocator 'bare-like '[ref :- (Array double) n :- Long] '(double-array n))
      (let [op (operation source "bare-like" "doubles_long")
            implementation (symbol (namespace op) (str (name op) "-impl"))
            call (with-meta (list '.invk implementation 'x 'cnt)
                   {:raster.op/original (symbol (namespace op) "bare-like")})
            int-values {'x {:dtype :double :shape ['m]} 'cnt {:dtype :int :shape []}}
            _ (method-entry/call-with-definition-inputs
               #(invocation-source/required-allocation-source! call int-values))
            project #(binding [method-entry/*emission-inputs* nil]
                       (descriptor/direct-allocation-projection call int-values))]
        (is (= '(clojure.core/double-array cnt)
               (:constructor (descriptor/direct-allocation-projection
                              call {'x {:dtype :double :shape ['m]}
                                    'cnt {:dtype :long :shape []}})))
            "semantic constructor projection does not invent a physical entry adapter")
        (let [projection (project)
              signature (method-entry/live-selected-parameter-carriers
                         (symbol (namespace op) "bare-like") implementation 2)]
          (is (= Long/TYPE (second (:parameter-carriers signature))))
          (is (= '(clojure.core/double-array (clojure.core/long cnt)) (:constructor projection)))
          (is (= '[x cnt] (get-in projection [:helper :arguments])))
          (is (= :int (get-in int-values ['cnt :dtype]))))
        (doseq [count [Integer/MIN_VALUE -1 0 1 Integer/MAX_VALUE]]
          (is (= (long count)
                 (eval (list 'let* ['cnt count] (second (:constructor (project))))))
              "check boundary conversions without allocating boundary-sized arrays"))
        (binding [*ns* source]
          (is (= 1 (alength (eval (list '.invk implementation '(double-array 0) '(int 1)))))))
        (is (= '(clojure.core/double-array (clojure.core/long cnt)) (:constructor (project)))
            "replay rechecks a warmed selected interface without an active definition owner")
        (is (nil? (descriptor/direct-allocation-projection (list op 'x 'cnt) int-values))
            "a direct IFn entry does not inherit typed invk adaptation")
        (let [signature (method-entry/live-selected-parameter-carriers
                         (symbol (namespace op) "bare-like") implementation 2)
              iface (get-in signature [:definition :entry :typed-iface])]
          (with-redefs [types/fn-interface-by-name
                        (atom (assoc-in @types/fn-interface-by-name [iface :param-tags] '[doubles double]))]
            (is (nil? (project)) "mismatched live interface evidence cannot authorize widening"))))
      (let [op (symbol (str (ns-name source)) "tick")
            fake {:allocates? true :auto-detected? true
                  :direct-allocation {:parameters '[n] :parameter-tags '[long]
                                      :constructor '(clojure.core/double-array n)}}]
        (with-redefs [descriptor/resolve-buffer-semantics (fn [_] [fake op])]
          (is (nil? (descriptor/direct-allocation-projection (list op 'cnt) {'cnt {:dtype :long :shape []}})))))
      (is (= {:constructor '(clojure.core/double-array cnt)
              :public-initialization :zero
              :realized-initialization :zero}
             (descriptor/direct-allocation-projection '(clojure.core/double-array cnt) {})))
      (doseq [call ['(double-array cnt) '(clojure.core/double-array cnt)]
              :let [lexical (with-meta call {:raster.op/lexical-callee true})]]
        (is (nil? (descriptor/array-allocation-size lexical)))
        (is (nil? (descriptor/direct-allocation-projection lexical {})))))))

(defn- compiled-allocation-call [generic dtype arguments]
  (let [source (core/resolve-deftm-var generic {:dtype dtype})
        implementation (symbol (str (:ns (meta source)))
                               (str (:name (meta source)) "-impl"))
        receiver (ns-resolve (:ns (meta source)) (symbol (name implementation)))]
    (core/ensure-bytecode-implementation! receiver)
    {:source source :receiver receiver
     :call (with-meta (list* '.invk implementation arguments)
             {:raster.op/original
              (symbol (str (:ns (meta generic))) (str (:name (meta generic))))})}))

(defn- compiled-source-call [source helper-name tags arguments]
  (let [operation (operation source helper-name tags)
        implementation (symbol (namespace operation) (str (name operation) "-impl"))
        receiver (ns-resolve source (symbol (name implementation)))]
    (core/ensure-bytecode-implementation! receiver)
    (with-meta (list* '.invk implementation arguments)
      {:raster.op/original (symbol (str (ns-name source)) (str helper-name))})))

(defn- cold-source-call [source helper-name tags arguments]
  (let [operation (operation source helper-name tags)
        implementation (symbol (namespace operation) (str (name operation) "-impl"))
        source-var (ns-resolve source (symbol (name operation)))
        receiver (ns-resolve source (symbol (name implementation)))]
    {:source source-var :receiver receiver
     :call (with-meta (list* '.invk implementation arguments)
             {:raster.op/original (symbol (str (ns-name source)) (str helper-name))})}))

(deftest required-allocation-construction-prepares-only-the-exact-cold-forwarder
  (with-source
    (fn [source-ns]
      (define-allocator 'cold-leaf '[ref :- (Array double) n :- Long]
        '(double-array n))
      (define-allocator 'cold-semantic '[ref :- (Array double) n :- Long]
        '(double-array n))
      (define-allocator 'alloc-like '[ref :- (Array double) n :- Long]
        '(cold-leaf ref n))
      (let [values {'reference {:dtype :double :shape ['capacity]}
                    'count {:dtype :long :shape []}}
            leaf (cold-source-call source-ns "cold-leaf" "doubles_long"
                                   '[reference count])
            semantic (cold-source-call source-ns "cold-semantic" "doubles_long"
                                       '[reference count])
            forward (cold-source-call source-ns "alloc-like" "doubles_long"
                                      '[reference count])
            forward-metadata (meta (:receiver forward))]
        (is (descriptor/alloc-op? (descriptor/semantic-op (:call forward)))
            "the cold candidate survives allocation classification before proof")
        (is (map? (:raster.core/jit-spec forward-metadata))
            "the fixture must exercise a genuinely pending selected implementation")
        (is (nil? (invocation-source/allocation-source (:call leaf) values))
            "optional correspondence does not populate a cold semantic source")
        (is (map? (:raster.core/jit-spec (meta (:receiver leaf))))
            "the optional direct constructor query does not acquire a physical helper module")
        (alter-meta! (:source semantic) dissoc :raster.core/deftm-walked-body)
        (alter-meta! (:receiver semantic) update :raster.core/jit-spec
                     dissoc :frontend-source-owner)
        (let [attempts (atom 0)
              projection
              (method-entry/call-with-definition-inputs
               #(do (swap! attempts inc)
                    (invocation-source/required-allocation-source! (:call semantic) values)))]
          (is (= 2 @attempts) "missing semantic source causes one bounded fresh restart")
          (is (= '(clojure.core/double-array count) (:constructor projection)))
          (is (map? (:raster.core/jit-spec (meta (:receiver semantic))))
              "semantic preparation does not acquire a physical module"))
        (is (nil? (invocation-source/allocation-source (:call forward) values))
            "optional correspondence never prepares a missing physical source")
        (is (identical? forward-metadata (meta (:receiver forward))))
        (let [attempts (atom 0)
              projection
              (method-entry/call-with-definition-inputs
               #(do (swap! attempts inc)
                    (invocation-source/required-allocation-source! (:call forward) values)))]
          (is (< 1 @attempts) "the required owner restarts after bounded preparation")
          (is (= '(clojure.core/double-array count) (:constructor projection)))
          (is (= 'alloc-like (-> projection :helper :operation name symbol)))
          (is (= 2 (count (take-while some?
                                      (iterate #(get-in % [:delegation :definition])
                                               (get-in projection [:helper :definition]))))))
          (is (= :unspecified (:initialization
                               (invocation-source/allocation-source
                                (:call forward) values))))
          (is (= '(clojure.core/double-array count)
                 (:constructor (invocation-source/allocation-source (:call leaf) values))))
          (alter-meta! (:source forward) dissoc :raster.core/deftm-walked-body)
          (is (= (:constructor projection)
                 (:constructor (invocation-source/allocation-source
                                (:call forward) values)))
              "a ready physical module reuses its authenticated original body when the cache is absent")
          (let [preparations (atom 0)]
            (with-redefs [bytecode/retained-input-invocation-context (constantly nil)
                          core/prepare-owned-source-frontend!
                          (fn [& _]
                            (swap! preparations inc)
                            (throw (AssertionError. "a ready unsupported source was re-prepared")))]
              (is (nil? (method-entry/call-with-definition-inputs
                         #(invocation-source/required-allocation-source!
                           (:call forward) values))))
              (is (zero? @preparations)
                  "ready-but-unsupported source is a final decline, not a renewal request"))))))))

(deftest delegated-allocation-projection-retains-the-outer-public-contract
  (doseq [[dtype array-value expected-constructor]
          [[:float (float-array 0) 'clojure.core/float-array]
           [:double (double-array 0) 'clojure.core/double-array]]]
    (let [{:keys [source receiver call]}
          (compiled-allocation-call #'arrays/alloc-like dtype '[reference count])
          values {'reference {:dtype dtype :shape ['capacity]}
                  'count {:dtype :long :shape []}}
          [projection observations]
          (method-entry/call-with-definition-inputs
           (fn [] [(descriptor/direct-allocation-projection call values)
                   @method-entry/*emission-inputs*]))
          chain (take-while some?
                            (iterate #(get-in % [:delegation :definition])
                                     (get-in projection [:helper :definition])))]
      (is (= (list expected-constructor 'count) (:constructor projection)))
      (is (= :unspecified (:public-initialization projection)))
      (is (= :zero (:realized-initialization projection)))
      (is (= {:public :unspecified :realized :zero}
             (descriptor/allocation-projection-initializations projection)))
      (doseq [changed [(assoc projection :public-initialization :zero)
                       (assoc projection :public-initialization :copy)
                       (assoc projection :realized-initialization :unspecified)
                       (assoc-in projection [:helper :operation] 'raster.arrays/zeros-like)
                       (update projection :helper dissoc :definition)]]
        (is (nil? (descriptor/allocation-projection-initializations changed))))
      (is (= 2 (count chain)) "alloc-like delegates exactly through zeros-like")
      (is (= :unspecified
             (:initialization
              (method-entry/call-with-definition-inputs
               #(invocation-source/allocation-source call values)))))
      (is (true?
           (method-entry/call-with-definition-inputs
            #(invocation-source/allocation-corresponds?
              call {:dtype dtype :shape ['count]} :unspecified values))))
      (is (false?
           (method-entry/call-with-definition-inputs
            #(invocation-source/allocation-corresponds?
              call {:dtype dtype :shape ['count]} :zero values)))
          "a zeroing leaf does not promote the public allocator's readiness")
      (is (not-any? #(or (fn? %) (instance? java.lang.reflect.Method %)
                         (instance? clojure.lang.Namespace %))
                    (tree-seq coll? seq (:helper projection))))
      (doseq [reference [nil array-value]
              count [-1 0 3]]
        (let [observe (fn [invoke]
                        (try (let [array (invoke)] [(class array) (vec array)])
                             (catch Throwable error {:error (class error)})))]
          (is (= (observe #(source reference count))
                 (observe #(eval (list 'let* ['count count] (:constructor projection))))))))
      (is (contains? (:definition-proofs observations) source))
      (is (contains? (:definition-proofs observations) receiver))
      (with-redefs-fn {receiver (fn [& _] (if (= dtype :float)
                                           (float-array 0) (double-array 0)))}
        #(is (thrown? clojure.lang.ExceptionInfo
                      (method-entry/assert-definition-inputs-current! observations)))))))

(deftest delegated-allocation-projection-requires-its-exact-emitted-call-edge
  (let [{:keys [call]} (compiled-allocation-call #'arrays/alloc-like :float '[reference count])
        values {'reference {:dtype :float :shape ['capacity]}
                'count {:dtype :long :shape []}}
        observe bytecode/retained-input-invocation-context
        project #(method-entry/call-with-definition-inputs
                  (fn [] (descriptor/direct-allocation-projection call values)))]
    (is (some? (project)))
    (doseq [edit [(constantly nil)
                  #(assoc % :invocation-mode :unproved)
                  #(assoc % :receiver 'foreign/implementation)
                  #(assoc % :method nil)
                  #(assoc-in % [:receiver-input :ports] {'foreign {:type :ref}})
                  #(assoc-in % [:inputs 0 :context] {:operand-context :forged})
                  #(assoc-in % [:inputs 0 :unretained-sibling-calls?] true)
                  #(assoc-in % [:input-completions 1 :adaptation] :coerce)
                  #(assoc-in % [:input-completions 0 :parameter-carrier] Object)]]
      (with-redefs [bytecode/retained-input-invocation-context
                    (fn [& arguments] (some-> (apply observe arguments) edit))]
        (is (nil? (project)))))))

(deftest delegated-allocation-projection-refuses-nonidentity-forwarding
  (with-source
    (fn [source]
      (define-allocator 'storage-helper '[ref :- (Array double) n :- Long]
        '(double-array n))
      (define-allocator 'forwarding-helper '[ref :- (Array double) n :- Long]
        '(storage-helper ref n))
      (define-allocator 'two-hop-helper '[ref :- (Array double) n :- Long]
        '(forwarding-helper ref n))
      (let [values {'reference {:dtype :double :shape ['capacity]}
                    'other {:dtype :double :shape ['other-capacity]}
                    'count {:dtype :long :shape []}
                    'extra {:dtype :long :shape []}}
            project (fn [name tags arguments]
                      (let [call (compiled-source-call source name tags arguments)]
                        (method-entry/call-with-definition-inputs
                         #(descriptor/direct-allocation-projection call values))))
            ;; A projection query observes already emitted forwarding edges; it
            ;; never compiles a hidden intermediate merely to complete proof.
            _leaf (compiled-source-call source "storage-helper" "doubles_long"
                                        '[reference count])
            _middle (compiled-source-call source "forwarding-helper" "doubles_long"
                                          '[reference count])
            projection (project "two-hop-helper" "doubles_long" '[reference count])
            chain (take-while some?
                              (iterate #(get-in % [:delegation :definition])
                                       (get-in projection [:helper :definition])))]
        (is (= '(clojure.core/double-array count) (:constructor projection)))
        (is (= 3 (count chain)))
        ;; Define and check each refusal before another namespace mutation can
        ;; invalidate the exact source lookup history it is meant to test.
        (define-allocator 'computed-forward '[ref :- (Array double) n :- Long]
          '(storage-helper ref (inc n)))
        (is (nil? (project "computed-forward" "doubles_long" '[reference count])))
        (define-allocator 'constant-forward '[ref :- (Array double) n :- Long]
          '(storage-helper ref 1))
        (is (nil? (project "constant-forward" "doubles_long" '[reference count])))
        (define-allocator 'dropped-forward
          '[ref :- (Array double) n :- Long unused :- Long]
          '(storage-helper ref n))
        (is (nil? (project "dropped-forward" "doubles_long_long"
                           '[reference count extra])))
        (eval '(raster.core/deftm two-array-helper
                 [left :- (Array double) right :- (Array double) n :- Long] :- (Array double)
                 (double-array n)))
        (eval '(raster.core/deftm duplicate-forward
                 [left :- (Array double) right :- (Array double) n :- Long] :- (Array double)
                 (two-array-helper left left n)))
        (is (nil? (project "duplicate-forward" "doubles_doubles_long"
                           '[reference other count])))
        (eval '(raster.core/deftm permuted-forward
                 [left :- (Array double) right :- (Array double) n :- Long] :- (Array double)
                 (two-array-helper right left n)))
        (is (nil? (project "permuted-forward" "doubles_doubles_long"
                           '[reference other count])))))))

(deftest specialization-carrier-admission-is-not-removed
  (with-source
    (fn [source]
      (define-allocator 'carrier '[n :- Long]
        '(let [out (double-array n)] (aset out 0 (double n)) out))
      (let [original (call-form (operation source "carrier" "long") [3.5])
            result (buffer-fuse/fuse-let original)]
        (is (= original (:form result)))
        (is (= [3.0 0.0 0.0] (vec (eval original))))
        (is (= (vec (eval original)) (vec (eval (:form result))))))
      (define-allocator 'array-carrier '[a :- (Array double)] '(double-array (alength a)))
      (define-allocator 'reference-carrier '[unused :- String n :- Long] '(double-array n))
      (doseq [[name tags arguments]
              [["array-carrier" "doubles" '[(float-array 3)]]
               ["reference-carrier" "String_long" [3 3]]]]
        (let [op (operation source name tags)
              original (call-form op arguments)
              result (buffer-fuse/fuse-let original)]
          (is (some? (descriptor/resolve-buffer-semantics op)))
          (is (= original (:form result)))
          (let [before (try (eval original) nil (catch Throwable error error))
                after (try (eval (:form result)) nil (catch Throwable error error))]
            (is (some? before))
            (is (= (class before) (class after))))))
      (let [op (operation source "carrier" "long")
            source-form (list 'let* ['n 3 'alias 'n 'result (list op 'alias)] 'result)
            result (buffer-fuse/fuse-let source-form)]
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (is (= (vec (eval source-form)) (vec (eval (:form result))))))
      (let [original (call-form (operation source "carrier" "long") '[unknown])]
        (is (= original (:form (buffer-fuse/fuse-let original)))))
      (let [original (list 'let* ['n '(if flag 3 3.5)
                                  'result (list (operation source "carrier" "long") 'n)] 'result)]
        (is (= original (:form (buffer-fuse/fuse-let original :param-env {'flag 'boolean})))
            "all reachable carriers must agree, including earlier locals"))
      (let [original (call-form (operation source "carrier" "long") '[n])
            result (buffer-fuse/fuse-let original :param-env {'n 'long})]
        (is (= 1 (get-in result [:stats :fresh-allocs])))))))

(defn- error-of [f]
  (try (f) nil (catch Throwable error error)))

(deftest effectful-and-throwing-size-prefixes-are-not-replayed
  (with-source
    (fn [source]
      (doseq [helper '[tick fail]]
        (define-allocator 'allocate '[n :- Long]
          (list 'let ['m (list helper 'n) 'out '(double-array m)] 'out))
        (let [op (operation source "allocate" "long")
              original (call-form op [3])
              fused (:form (buffer-fuse/fuse-let original))
              calls @(ns-resolve source 'calls)
              failure @(ns-resolve source 'failure)]
          (is (nil? (descriptor/resolve-buffer-semantics op)))
          (is (= original fused))
          (doseq [form [original fused]]
            (reset! calls 0)
            (if (= helper 'fail)
              (is (identical? failure (try (eval form) nil (catch clojure.lang.ExceptionInfo e e))))
              (is (= 3 (alength ^doubles (eval form)))))
            (is (= 1 @calls))))))))

(deftest unrelated-preallocation-effects-and-open-helpers-retain-order
  (with-source
    (fn [source]
      (define-allocator 'allocate '[n :- Long]
        '(let [_ (tick n) out (double-array n)] out))
      (let [op (operation source "allocate" "long")
            original (call-form op [3])]
        (is (nil? (descriptor/resolve-buffer-semantics op)))
        (is (= original (:form (buffer-fuse/fuse-let original)))))
      (eval '(defn ^{:tag 'long} helper [n] n))
      (define-allocator 'open '[n :- Long] '(let [m (helper n) out (double-array m)] out))
      (let [op (operation source "open" "long")]
        (is (nil? (descriptor/resolve-buffer-semantics op)) "current purity is not a dependency certificate")
        (eval '(defn ^{:tag 'long} helper [n] (swap! calls inc) n))
        (is (nil? (descriptor/resolve-buffer-semantics op)))
        (let [form (:form (buffer-fuse/fuse-let (call-form op [3])))]
          (eval form)
          (is (= 1 @@(ns-resolve source 'calls))))))))

(deftest actual-argument-effects-are-neither-duplicated-nor-dropped
  (with-source
    (fn [source]
      (doseq [[name params body tags arguments]
              [['unused '[n :- Long extra :- Long] '(double-array n) "long_long" '[3 (tick 1)]]
               ['repeated '[n :- Long] '(let [out (double-array n)] (aset out 0 (double n)) out)
                "long" '[(tick 3)]]]]
        (define-allocator name params body)
        (let [op (operation source (str name) tags)
              original (call-form op arguments)
              fused (:form (buffer-fuse/fuse-let original))
              calls @(ns-resolve source 'calls)]
          (is (some? (descriptor/resolve-buffer-semantics op)) "safe body still has an auto contract")
          (is (= 1 (get-in (buffer-fuse/fuse-let original) [:stats :fresh-allocs]))
              "actuals are transported in evaluate-once bindings")
          (doseq [form [original fused]]
            (reset! calls 0)
            (is (= 3 (alength ^doubles (eval form))))
            (is (= 1 @calls))))))))

(deftest constructor-identity-and-overload-are-not-guessed
  (with-source
    (fn [source]
      (eval '(defn bogus-array [n] (double-array n)))
      (doseq [[name params body tags]
              [['impostor '[n :- Long] '(bogus-array n) "long"]
               ['initializer '[n :- Long] '(double-array n 4.0) "long"]
               ['copy-input '[a :- (Array double)] '(double-array a) "doubles"]]]
        (define-allocator name params body)
        (is (nil? (descriptor/resolve-buffer-semantics (operation source (str name) tags))))))))

(deftest namespaced-core-constructor-name-is-not-a-core-constructor
  (with-source
    (fn [source]
      (let [foreign (create-ns (gensym "raster.test.foreign_allocator_"))]
        (try
          (intern foreign (with-meta 'double-array {:tag 'doubles})
                  (fn [n] (double-array n)))
          (define-allocator 'foreign '[n :- Long]
            (list (symbol (str (ns-name foreign)) "double-array") 'n))
          (is (nil? (descriptor/resolve-buffer-semantics (operation source "foreign" "long"))))
          (finally (remove-ns (ns-name foreign))))))))

(deftest stable-size-and-harmless-prefix-still-fuse
  (with-source
    (fn [source]
      (define-allocator 'stable '[n :- Long] '(let [m n out (double-array m)] out))
      (let [op (operation source "stable" "long")
            original (call-form op [3])
            result (buffer-fuse/fuse-let original)]
        (is (some? (descriptor/resolve-buffer-semantics op)))
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (is (= (vec (eval original)) (vec (eval (:form result)))))))))

(deftest lexical-substitution-preserves-shadowing-capture-and-quoted-data
  (with-source
    (fn [source]
      (doseq [[name body expected]
              [['shadow '(let [out (double-array n)]
                           (let [n 2] (aset out 0 (double n))) out) 2.0]
               ['capture '(let [out (double-array n)]
                            (let [m 2] (aset out 0 (double n))) out) 3.0]
               ['quoted '(let [out (double-array n)]
                           (aset out 0 (if (= 'n (symbol "n")) 1.0 2.0)) out) 1.0]]]
        (define-allocator name '[n :- Long] body)
        (let [op (operation source (str name) "long")
              original (list 'let* ['m 3 'result (list op 'm)] 'result)
              result (buffer-fuse/fuse-let original)]
          (is (= 1 (get-in result [:stats :fresh-allocs])))
          (is (= expected (aget ^doubles (eval (:form result)) 0)))
          (is (= (vec (eval original)) (vec (eval (:form result))))))))))

(deftest global-actuals-are-captured-by-the-original-call
  (with-source
    (fn [source]
      (eval '(def global-size 3))
      (eval '(defn change-global [] (alter-var-root #'global-size (constantly 7))))
      (define-allocator 'global '[n :- Long]
        '(let [out (double-array n)] (change-global) (aset out 0 (double n)) out))
      (let [original (call-form (operation source "global" "long") '[global-size])
            result (buffer-fuse/fuse-let original)]
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (doseq [form [original (:form result)]]
          (alter-var-root (ns-resolve source 'global-size) (constantly 3))
          (is (= [3.0 0.0 0.0] (vec (eval form)))))))))

(deftest caller-cast-names-cannot-capture-helper-operations
  (with-source
    (fn [source]
      (define-allocator 'casted '[n :- Long]
        '(let [m (long n) out (double-array m)] (aset out 0 (double n)) out))
      (let [original (list 'let* ['long '(fn [x] (tick x))
                                 'double '(fn [x] (tick x))
                                 'result (list (operation source "casted" "long") 3)] 'result)
            result (buffer-fuse/fuse-let original)
            calls @(ns-resolve source 'calls)]
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (doseq [form [original (:form result)]]
          (reset! calls 0)
          (is (= [3.0 0.0 0.0] (vec (eval form))))
          (is (zero? @calls)))))))

(deftest core-named-caller-values-preserve-lexical-identity
  (with-source
    (fn [source]
      (define-allocator 'local-name '[n :- Long]
        '(let [out (double-array n)] (let [long 2] (aset out 0 (double n))) out))
      (let [op (operation source "local-name" "long")
            original (list 'let* ['long 3 'result (list op 'long)] 'result)
            result (buffer-fuse/fuse-let original)]
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (is (= [3.0 0.0 0.0] (vec (eval original))))
        (is (= (vec (eval original)) (vec (eval (:form result))))))
      (let [original (list 'let* ['x 3 'result
                                  (list (operation source "local-name" "long") '(long x))] 'result)
            result (buffer-fuse/fuse-let original)]
        (is (= 1 (get-in result [:stats :fresh-allocs])))
        (is (= (vec (eval original)) (vec (eval (:form result)))))))))

(deftest bare-extents-execute-once-without-hoisting-permission
  (with-source
    (fn [source]
      (eval '(raster.core/deftm extent-tick [n :- Long] :- Long (do (swap! calls inc) n)))
      (eval '(raster.core/deftm extent-fail [n :- Long] :- Long (do (swap! calls inc) (throw failure))))
      (doseq [helper '[extent-tick extent-fail]]
        (define-allocator 'bare '[n :- Long] (list 'double-array (list helper 'n)))
        (let [original (call-form (operation source "bare" "long") [3])
              result (buffer-fuse/fuse-let original)
              calls @(ns-resolve source 'calls)
              failure @(ns-resolve source 'failure)
              remarked (hoist/infer-hoistable (:form result) #{})
              allocation (first (filter #(-> % first meta :raster.buffer/no-hoist)
                                        (partition 2 (second remarked))))]
          (is (= 1 (get-in result [:stats :fresh-allocs])))
          (is (some? allocation))
          (is (not (hoist/hoist-safe-pair? #{} allocation)))
          (doseq [form [original (:form result)]]
            (reset! calls 0)
            (if (= helper 'extent-fail)
              (is (identical? failure (error-of #(eval form))))
              (is (= 3 (alength ^doubles (eval form)))))
            (is (= 1 @calls))))))))

(deftest replay-admission-requires-local-and-unshadowed-operation-evidence
  (is (not (effects/replay-safe-value? 'global-value)))
  (is (effects/replay-safe-value? 'local-value {'local-value nil}))
  (is (not (effects/replay-safe-value? '(long n) {'long nil 'n 'long})))
  (is (effects/replay-safe-value? '(long n) {'n 'long})))

(deftest argument-frames-retain-order-unused-arguments-and-failures
  (with-source
    (fn [source]
      (eval '(def order (atom [])))
      (eval '(defn record-value [n] (swap! order conj n) n))
      (eval '(defn record-failure [n] (swap! order conj n) (throw failure)))
      (define-allocator 'ordered '[n :- Long unused :- Long] '(double-array n))
      (doseq [second-argument '[(record-value 1) (record-failure 1)]]
        (let [original (call-form (operation source "ordered" "long_long")
                                  ['(record-value 3) second-argument])
              result (buffer-fuse/fuse-let original)
              order @(ns-resolve source 'order)
              failure @(ns-resolve source 'failure)]
          (is (= 1 (get-in result [:stats :fresh-allocs])))
          (doseq [form [original (:form result)]]
            (reset! order [])
            (if (= 'record-failure (first second-argument))
              (is (identical? failure (error-of #(eval form))))
              (is (= 3 (alength ^doubles (eval form)))))
            (is (= [3 1] @order))))))))

(deftest global-extent-identity-is-not-caller-namespace-identity
  (with-source
    (fn [source]
      (eval '(def global-width 3))
      (define-allocator 'open-global '[unused :- Long] '(double-array (long global-width)))
      ;; Some frontend paths already qualify the read. Exercise the still-open
      ;; retained-source boundary explicitly, as cache tests do for source facts.
      (alter-meta! (ns-resolve source 'open-global_m_long)
                   assoc :raster.core/deftm-walked-body
                   '[(double-array (long global-width))])
      (is (nil? (descriptor/resolve-buffer-semantics (operation source "open-global" "long"))))
      (define-allocator 'closed-global '[unused :- Long]
        (list 'double-array (list 'long (symbol (str (ns-name source)) "global-width"))))
      (let [caller (create-ns (gensym "raster.test.allocation_caller_"))
            original (call-form (operation source "closed-global" "long") [0])
            result (buffer-fuse/fuse-let original)]
        (try
          (binding [*ns* caller]
            (refer 'clojure.core)
            (eval '(def global-width 7))
            (is (= 1 (get-in result [:stats :fresh-allocs])))
            (is (= 3 (alength ^doubles (eval original))))
            (is (= 3 (alength ^doubles (eval (:form result))))))
          (finally (remove-ns (ns-name caller))))))))

(deftest argument-frame-does-not-replace-arity-refusal
  (with-source
    (fn [source]
      (define-allocator 'one '[n :- Long] '(double-array n))
      (doseq [arguments [[] [3 1]]]
        (let [original (call-form (operation source "one" "long") arguments)
              result (buffer-fuse/fuse-let original)]
          (is (= original (:form result)))
          (let [original-error (error-of #(eval original))
                transformed-error (error-of #(eval (:form result)))]
            (is (some? original-error))
            (is (= (class original-error) (class transformed-error)))))))))

(deftest length-dimensions-require-the-actual-global-owner
  (let [argument (with-meta 'a {:raster.type/tag 'doubles})
        expression-meta {:raster.type/tag 'long :line 42}
        global (with-meta (list '.invk 'raster.arrays/alength_m_doubles-impl argument)
                 expression-meta)
        retained (with-meta (list '.invk 'other-implementation argument)
                   (assoc expression-meta :raster.op/original 'raster.arrays/alength))
        foreign (list '.invk 'campaign.foreign/alength_m_doubles-impl argument)
        lexical (with-meta global (assoc expression-meta :raster.op/lexical-callee true))]
    (doseq [expression [(with-meta (list 'clojure.core/alength argument) expression-meta)
                        global retained]]
      (let [dimensions (hoist/build-dim-env (list 'let* ['size expression] 'size) #{'a})
            dimension (get dimensions 'size)]
        (is (= (list 'clojure.core/alength argument) dimension))
        (is (= expression-meta (select-keys (meta dimension) (keys expression-meta))))
        (is (identical? argument (second dimension)))))
    (doseq [expression [foreign lexical
                        (with-meta global {:raster.op/original 'campaign.foreign/alength})]]
      (is (= {} (hoist/build-dim-env (list 'let* ['size expression] 'size) #{'a}))))
    ;; Wrong arity/unknown intermediate carries no length proof.
    (is (= {} (hoist/build-dim-env '(let* [size (clojure.core/alength a b)] size) #{'a 'b})))
    (is (= {} (hoist/build-dim-env '(let* [size (clojure.core/alength unknown)] size) #{'a})))))

(deftest dimension-rules-obey-ordered-lexical-masking
  (let [lookup descriptor/get-op-descriptor
        dimension-rule {:shape {:dim-rule (fn [_ _ _] '(clojure.core/alength a))}}]
    (with-redefs [descriptor/get-op-descriptor
                  (fn [operation]
                    (if (= 'length-like operation) dimension-rule (lookup operation)))]
      (is (= {'size '(clojure.core/alength a)}
             (hoist/build-dim-env '(let* [size (length-like a)] size) #{'a})))
      (is (= {} (hoist/build-dim-env '(let* [size (length-like a)] size) #{'a 'length-like})))
      (is (= {} (hoist/build-dim-env '(let* [length-like nil size (length-like a)] size) #{'a})))
      (is (= {} (hoist/build-dim-env '(let* [length-like nil size (.invk length-like a)] size) #{'a}))))
    ;; A later unknown rebinding cannot reuse the earlier dimension fact.
    (is (= {} (hoist/build-dim-env '(let* [size (clojure.core/alength a) size nil] size) #{'a})))
    (is (= {'size '(clojure.core/alength a)}
           (hoist/build-dim-env '(let* [alength nil size (clojure.core/alength a)] size) #{'a})))))

(deftest foreign-length-facts-cannot-change-allocation-capacity
  (let [output (with-meta 'out {:raster.buffer/hoistable true})
        foreign '(.invk campaign.foreign/alength_m_doubles-impl a)
        pairs [['size foreign] [output '(double-array size)]]
        dimensions (hoist/build-dim-env (list 'let* (vec (mapcat identity pairs)) output) #{'a})
        rewritten (hoist/rewrite-alloc-exprs pairs dimensions #{'a})]
    (is (= pairs rewritten))
    (is (identical? foreign (second (first rewritten))))
    ;; The independently supplied helper result is 2 while the backing array
    ;; has 5 elements. Rewriting to core alength would allocate the wrong size.
    (is (= 2 (alength ^doubles
                     (eval (list 'let* ['a '(double-array 5) 'size 2]
                                 (second (second rewritten)))))))
    (let [global-pairs [['size '(.invk raster.arrays/alength_m_doubles-impl a)]
                        [output '(double-array size)]]
          dimensions (hoist/build-dim-env (list 'let* (vec (mapcat identity global-pairs)) output) #{'a})
          rewritten (hoist/rewrite-alloc-exprs global-pairs dimensions #{'a})]
      (is (= '(double-array (clojure.core/alength a)) (second (second rewritten)))))))
