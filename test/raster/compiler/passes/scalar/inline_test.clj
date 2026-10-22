(ns raster.compiler.passes.scalar.inline-test
  "Guards the deftm inliner's call/callee arity contract.

  The inliner substitutes callee params for call args POSITIONALLY — `(nth args i)`
  over the param index. Nothing checked that the two had the same length, so a stale
  call site — a deftm that grew a parameter while one of its callers was not
  updated — died as a bare `IndexOutOfBoundsException: null` out of PersistentVector,
  thrown from deep inside the AD-prep inliner (inline.clj:1108), naming neither the
  callee nor the arities. It read like a compiler bug; it was a wrong-arity CALL.

  (The same stale call against a PARAMETRIC (All [T]) callee fails differently and
  even more obscurely: no overload resolves, the call is never inlined, stays
  symbolic, and reverse-AD reports `No AD template for <callee>` — a message that
  points at the AD registry rather than at the call. Both shapes were hit in the same
  finetune commit, where the gemma block gained a `bs` batch param and two reference
  losses were not updated.)

  A deftm has no variadic or multi-arity form, so a wrong-arity call can never
  dispatch: it is always a bug, and the inliner must say so AT THE CALL — naming the
  callee and both arities."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.par]
            [raster.numeric]
            [raster.arrays]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.pipeline :as pipeline]
            [raster.ad.reverse :as rev]))

(deftest no-op-ad-hoisting-preserves-exact-source-occurrences
  (let [leaf (with-meta '(helper x) {:source-marker :leaf})
        tuple (with-meta [leaf 'x] {:source-marker :tuple})
        source (with-meta (list 'outer leaf tuple) {:source-marker :root})
        scoped '(let* [x ((raster.ad.reverse/value+grad (var f)) y)] x)
        result (#'inline/hoist-nested-vg ['answer source 'scoped scoped] [source tuple])]
    (is (false? (:changed? result)))
    (is (identical? source (second (:bindings result))))
    (is (identical? scoped (nth (:bindings result) 3)))
    (is (identical? source (first (:body result))))
    (is (identical? tuple (second (:body result))))
    (is (identical? leaf (second (first (:body result)))))
    (is (identical? (meta source) (meta (first (:body result)))))))

(deftest real-ad-hoisting-replaces-only-the-changed-source-path
  (let [application '((raster.ad.reverse/value+grad (var f)) x)
        leaf '(untouched x)
        source (with-meta (list 'outer leaf application) {:source-marker :root})
        result (#'inline/hoist-nested-vg ['answer source] [])
        [lifted call answer transformed] (:bindings result)]
    (is (true? (:changed? result)))
    (is (symbol? lifted))
    (is (identical? application call))
    (is (= 'answer answer))
    (is (not (identical? source transformed)))
    (is (identical? leaf (second transformed)))
    (is (= lifted (nth transformed 2)))
    (is (identical? (meta source) (meta transformed))))
  (let [application '((raster.ad.reverse/value+grad (var f)) x)
        result (#'inline/hoist-nested-vg ['answer application] [])]
    (is (false? (:changed? result)))
    (is (identical? application (second (:bindings result))))))

(deftest ad-hoisting-never-evaluates-quoted-data
  (let [application '((raster.ad.reverse/value+grad (var f)) x)]
    (doseq [head '[quote clojure.core/quote]
            data [application [application]]]
      (let [quoted (with-meta (list head data) {:source-marker :data})
            source (list 'outer quoted [quoted])
            result (#'inline/hoist-nested-vg ['answer source 'data quoted] [quoted])]
        (is (false? (:changed? result)))
        (is (identical? source (second (:bindings result))))
        (is (identical? quoted (nth (:bindings result) 3)))
        (is (identical? quoted (first (:body result))))))
    (let [quoted (list 'quote application)
          result (#'inline/hoist-nested-vg ['answer (list 'outer quoted application)] [])]
      (is (true? (:changed? result)))
      (is (= 4 (count (:bindings result))))
      (is (identical? quoted (second (nth (:bindings result) 3)))))))

(deftest actual-inliner-retains-formal-class-import-history
  (let [name (gensym "raster.formal.qualification.")
        source (create-ns name)]
    (try
      (binding [*ns* source]
        (clojure.core/refer 'clojure.core)
        (.importClass ^clojure.lang.Namespace source java.util.Date)
        (eval '(raster.core/deftm helper [unused :- Date] :- Long (let [one 1] one))))
      (binding [entry/*emission-inputs* (atom {:static [] :values []})]
        (let [helper (ns-resolve source 'helper_m_Date)
              info (#'inline/try-resolve-walked-body helper)
              inputs @entry/*emission-inputs*]
          (is (= '[java.util.Date] (:tags info)))
          (is (seq (:walked-body info)))
          (is (entry/assert-definition-inputs-current! inputs))
          (ns-unmap source 'Date)
          (.importClass ^clojure.lang.Namespace source java.sql.Date)
          (is (= :stale-source-resolution
                 (try (entry/assert-definition-inputs-current! inputs) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
      (finally (remove-ns name)))))

(deftest anf-bindings-carry-existing-result-types-without-reinference
  (let [result (with-meta 'rhs {:raster.type/tag 'doubles})
        effect (with-meta 'effect {:raster.effect/effectful true})
        source (with-meta (list 'let* (with-meta [effect result] {:bindings-marker :kept}) result)
                 {:source-marker :kept})
        flattened (inline/flatten-nested-lets source)
        binder (first (second flattened))]
    (is (= source flattened) "metadata-only propagation must survive structural equality")
    (is (= 'doubles (:raster.type/tag (meta binder))))
    (is (:raster.effect/effectful (meta binder)))
    (is (= {:source-marker :kept} (meta flattened)))
    (is (= {:bindings-marker :kept} (meta (second flattened))))
    (is (nil? (:tag (meta binder))) "compiler type evidence is not a JVM primitive hint")
    (let [already-typed (with-meta 'value {:raster.type/tag 'double})
          init (with-meta 'input {:raster.type/tag 'float})
          form (inline/flatten-nested-lets (list 'let* [already-typed init] already-typed))]
      (is (= 'double (:raster.type/tag (meta (first (second form)))))))
    (let [form (inline/flatten-nested-lets '(let* [unknown (unregistered-call)] unknown))]
      (is (nil? (:raster.type/tag (meta (first (second form)))))))))

(deftest sequential-wrapper-normalization-preserves-order-and-hygiene
  (doseq [source ['(let* [x 7 result (let* [x (+ x 1)] x)] result)
                  '(let* [result (let* [x 7] x) x 11] [result x])
                  '(let* [result (do (record! :prefix)
                                    (let* [x (record! :body)] x)
                                    (record! :suffix))] result)]]
    (let [normalized (inline/flatten-nested-lets source)
          run (fn [expression]
                (let [trace (atom [])
                      f (eval (list 'fn '[record!] expression))
                      value (f (fn [event] (swap! trace conj event) event))]
                  [value @trace]))]
      (is (= (run source) (run normalized)))))
  (doseq [opaque ['(f (record! :first) (let* [x (record! :second)] x))
                  '[(record! :first) (let* [x (record! :second)] x)]
                  '(if c (let* [x 1] x) 0)
                  '(loop* [i 0] (let* [x i] x))
                  '(fn* [a] (let* [x a] x))]]
    (let [source (list 'let* ['result opaque] 'result)]
      (is (= source (inline/flatten-nested-lets source))))))

(deftest wrapper-normalization-does-not-discard-evaluation-denials
  (let [wrapped (with-meta '(let* [unused (clojure.core// x 0)] 7.0)
                  {:raster.effect/preserve-evaluation true})
        source (list 'let* ['result wrapped] 'result)]
      (let [normalized (inline/flatten-nested-lets source)
            initializer (nth (second normalized) 1)]
        (is (= wrapped initializer))
        (is (true? (:raster.effect/preserve-evaluation (meta initializer))))
        (is (false? (effects/removable-expr? initializer))))))

(defn- run-normalization [source]
  (let [trace (atom [])
        callee (atom (fn [& arguments] [:original arguments]))
        record! (fn [event] (swap! trace conj event) event)
        replace-callee! (fn []
                          (swap! trace conj :replacement)
                          (reset! callee (fn [& arguments] [:replacement arguments]))
                          2)
        execute (eval (list 'fn '[record! callee replace-callee!] source))]
    (try {:value (execute record! callee replace-callee!) :trace @trace}
         (catch Throwable error {:error (class error) :trace @trace}))))

(deftest default-wrapper-normalization-preserves-operand-and-callee-order
  (doseq [source
          ['(let* [answer [(record! :first) (let* [value (record! :second)] 2)]] answer)
           '(let* [answer (vector (record! :first) (let* [value (record! :second)] 2))] answer)
           '(let* [answer (@callee 1 (let* [value (replace-callee!)] value))] answer)
           '(let* [answer (do (record! :first)
                             (let* [value (record! :second)] value)
                             (record! :third))] answer)]]
    (is (= (run-normalization source)
           (run-normalization (inline/flatten-nested-lets source)))))
  (let [source '(let* [answer [(record! :first) (let* [value (record! :second)] 2)]] answer)
        result (run-normalization (inline/flatten-nested-lets source))]
    (is (= {:value [:first 2] :trace [:first :second]} result)))
  (let [source '(let* [answer (@callee 1 (let* [value (replace-callee!)] value))] answer)
        result (run-normalization (inline/flatten-nested-lets source))]
    (is (= {:value [:original '(1 2)] :trace [:replacement]} result))))

(deftest default-wrapper-normalization-preserves-first-failure-and-formal-conversion
  (doseq [source
          ['(let* [answer [(throw (IllegalStateException. "first"))
                          (let* [value (record! :second)] value)]] answer)
           '(let* [answer (vector (aget (double-array 0) 0)
                                 (let* [value (record! :second)] value))] answer)
           ;; The ^long hint selects the primitive formal. Its implicit
           ;; unboxing must fail BEFORE the second argument's lexical region.
           '(let* [answer (Math/addExact ^long (identity nil)
                                        (let* [value (record! :second)] (long 1)))] answer)]]
    (let [original (run-normalization source)
          actual (run-normalization (inline/flatten-nested-lets source))]
      (is (class? (:error original)))
      (is (empty? (:trace original)))
      (is (= original actual)))))

(deftest value-gradient-projection-indices-use-checked-constant-evidence
  (let [elements {'vg ['primal 'gradient]}]
    (doseq [index [1 '(long 1) '(clojure.core/long 1) '(clojure.core/int (long 1))]]
      (is (= 'gradient (#'inline/known-vg-element elements 'vg index))))
    (doseq [index [-1 2 'n '(long n) '(int 4294967296) '(float 1)]]
      (is (nil? (#'inline/known-vg-element elements 'vg index))))))

(deftest pure-unary-spines-expose-the-parallel-leaf-once
  (let [leaf '(raster.par/reduce acc 0.0 i 3 (+ acc (double i)))
        wrapper (with-meta (list 'Math/sqrt (list 'double leaf)) {:source-marker :kept})
        source (list 'let* ['value wrapper] 'value)
        exposed (inline/expose-parallel-unary-spines source #{})
        initializer (nth (second exposed) 1)
        reference (eval (list 'fn [] source))
        transformed (eval (list 'fn [] exposed))]
    (is (= 'let* (first initializer)))
    (is (= leaf (nth (second initializer) 1)))
    (is (= (reference) (transformed)))
    (is (= {:source-marker :kept} (meta (last initializer)))))
  (let [leaf (with-meta '(raster.par/reduce acc (float 0.0) i 1 (float i))
               {:tag 'float :raster.type/tag 'float})
        wrapper (with-meta (list '.invk 'raster.numeric/sqrt_m_float-impl leaf)
                  {:raster.op/original 'raster.numeric/sqrt :raster.type/tag 'float})
        exposed (inline/expose-parallel-unary-spines (list 'let* ['value wrapper] 'value) #{})
        initializer (nth (second exposed) 1)
        binder (first (second initializer))
        result (last initializer)]
    (is (= 'float (:raster.type/tag (meta binder))))
    (is (nil? (:tag (meta binder))) "do not introduce a JVM primitive let hint")
    (is (= '.invk (first result)))
    (is (= 'raster.numeric/sqrt_m_float-impl (second result)))
    (is (= (meta wrapper) (meta result)))))

(deftest unary-exposure-stops-at-unproved-calls-and-scopes
  (let [leaf '(raster.par/reduce acc 0.0 i 3 (+ acc (double i)))
        retained (with-meta (list '.invk 'raster.numeric/sqrt_m_float-impl leaf)
                   {:raster.op/original 'raster.numeric/sqrt})
        lexical (vary-meta retained assoc :raster.op/lexical-callee true)]
    (doseq [[initializer locals]
            [[(list '+ '(record! :first) leaf) #{}]
             [(list 'double (list 'if 'condition leaf 0.0)) #{}]
             [(list 'double (list 'let* ['local leaf] 'local)) #{}]
             [(list 'double [leaf]) #{}]
             [(list 'unregistered-helper leaf) #{}]
             [(list 'double leaf) #{'double}]
             [retained #{'raster.numeric/sqrt_m_float-impl}]
             [lexical #{}]
             [(list '.invk 'raster.numeric/sqrt_m_float-impl leaf) #{}]]]
      (let [source (list 'let* ['result initializer] 'result)]
        (is (= source (inline/expose-parallel-unary-spines source locals)))))
    (let [source (list 'let* ['double '(fn* [x] x)
                            'result (list 'double leaf)] 'result)]
      (is (= source (inline/expose-parallel-unary-spines source #{}))))))

(deftest unary-exposure-requires-purity-and-keeps-checked-wrapper-order
  (let [leaf '(raster.par/reduce acc 0.0 i 3 (+ acc (double i)))
        wrapper (list 'int leaf)
        source (list 'let* ['value wrapper] 'value)]
    (with-redefs [effects/analyze-effect (constantly :io)]
      (is (= source (inline/expose-parallel-unary-spines source #{}))))
    (let [exposed (inline/expose-parallel-unary-spines source #{})
          initializer (nth (second exposed) 1)
          [result parallel] (second initializer)]
      (is (= leaf parallel))
      (is (= (list 'int result) (last initializer)))
      (is (= 1 (count (filter #(= leaf %) (tree-seq coll? seq exposed)))))
      (is (= ((eval (list 'fn [] source))) ((eval (list 'fn [] exposed))))))))

(deftm option-loss [x :- Double y :- Double] :- Double
  (+ (* x x) (* x y)))

(deftm option-one-loss [x :- Double] :- Double (* x x))

(deftm argument-effect-loss [x :- Double unused :- Double] :- Double
  (* x x))

(deftest ad-arguments-evaluate-once-in-source-order-even-when-unused
  (doseq [mode [:value+grad :grad]]
    (let [expanded (inline/inline-value+grad-call
                    {:var-sym 'raster.compiler.passes.scalar.inline-test/argument-effect-loss
                     :args '[(draw :first) (draw :unused)] :mode mode})
          f (eval (list 'fn '[draw]
                        (list 'let* (vec (mapcat identity (:bindings expanded)))
                              (:elements expanded))))
          trace (atom [])
          draw (fn [id] (swap! trace conj id) (if (= id :first) 3.0 7.0))]
      (is (= (if (= mode :value+grad)
               ((rev/value+grad #'argument-effect-loss) 3.0 7.0)
               ((rev/grad #'argument-effect-loss) 3.0 7.0))
             (f draw)))
      (is (= [:first :unused] @trace))
      (reset! trace [])
      (let [failure (ex-info "argument failed" {:argument :unused})]
        (is (identical? failure
                        (try (f (fn [id]
                                  (swap! trace conj id)
                                  (if (= id :unused) (throw failure) 3.0)))
                             nil (catch Throwable error error))))
        (is (= [:first :unused] @trace)))
      (reset! trace [])
      (let [failure (ex-info "first argument failed" {:argument :first})]
        (is (identical? failure
                        (try (f (fn [id] (swap! trace conj id) (throw failure)))
                             nil (catch Throwable error error))))
        (is (= [:first] @trace)))))
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"Arity mismatch"
       (inline/inline-value+grad-call
        {:var-sym 'raster.compiler.passes.scalar.inline-test/argument-effect-loss
         :args '[x] :mode :value+grad})))
  (let [expanded (inline/inline-value+grad-call
                  {:var-sym 'raster.compiler.passes.scalar.inline-test/option-one-loss
                   :args '[(clojure.core/float (draw))] :mode :value+grad})
        [argument expression] (first (:bindings expanded))]
    (is (= '(clojure.core/float (draw)) expression))
    (is (= 'float (:raster.type/tag (meta argument)))
        "the Double formal must not widen the lifted Float producer")))

(deftm option-overloaded-loss [x :- Float n :- Long] :- Float
  (raster.numeric/* x x))

(deftm option-overloaded-loss [x :- Double n :- Long] :- Double
  (raster.numeric/* x x))

(deftest optionless-ad-retains-concrete-overload-types-and-gradient-slots
  (doseq [[tag argument] [['float (float 2.0)] ['double 2.0]]]
    (let [callee 'raster.compiler.passes.scalar.inline-test/option-overloaded-loss
          resolved (#'inline/resolve-deftm-for-ad callee [tag 'long])
          result (inline/inline-value+grad-call
                  {:var-sym callee :args '[x n] :mode :value+grad}
                  :param-env {'x tag 'n 'long})
          f (eval (list 'fn '[x n]
                        (list 'let* (vec (mapcat identity (:bindings result)))
                              (:elements result))))
          actual (f argument (long 3))]
      (is (= ((rev/value+grad (:var resolved)) argument (long 3)) actual))
      (is (= [4.0 4.0 nil] actual))
      ;; The primal aliases a typed retained expression; its result reference need not
      ;; duplicate binder evidence. Gradient binders already carry the tangent tag.
      (is (= tag (:raster.type/tag
                  (meta (get (into {} (:bindings result)) (:loss-sym result))))))
      (is (= tag (:raster.type/tag (meta (first (:grad-syms result))))))
      (is (nil? (second (:grad-syms result)))))))

(deftm option-one-grad [x :- Double] :- Double
  ((raster.ad.reverse/grad
     (var raster.compiler.passes.scalar.inline-test/option-one-loss) :wrt [0]) x))

(deftm option-one-grad-bound [x :- Double] :- Double
  (let [gradient (raster.ad.reverse/grad
                  (var raster.compiler.passes.scalar.inline-test/option-one-loss) :wrt [0])]
    (gradient x)))

(deftest static-ad-options-use-the-public-gradient-body
  (doseq [source
          ['(let* [vg ((raster.ad.reverse/value+grad
                         (var raster.compiler.passes.scalar.inline-test/option-loss)
                         :wrt [0]) x y)] vg)
           '(let* [gradient (raster.ad.reverse/value+grad
                             (var raster.compiler.passes.scalar.inline-test/option-loss)
                             :mode :reverse :wrt [0])
                   vg (gradient x y)] vg)]]
    (let [expanded (inline/inline-deftm-calls source)
          f (eval (list 'fn '[x y] expanded))]
      (is (= [10.0 7.0 nil] (f 2.0 3.0)))
      (is (= ((rev/value+grad #'option-loss :wrt [0]) 2.0 3.0) (f 2.0 3.0)))))
  (let [source '(let* [gradient ((raster.ad.reverse/grad
                                  (var raster.compiler.passes.scalar.inline-test/option-loss)
                                  :wrt [0]) x y)] gradient)
        expanded (inline/inline-deftm-calls source)
        f (eval (list 'fn '[x y] expanded))]
    (is (= [7.0 nil] (f 2.0 3.0))))
  (doseq [source ['(let* [g ((raster.ad.reverse/grad
                             (var raster.compiler.passes.scalar.inline-test/option-one-loss)
                             :wrt [0]) x)] g)
                  '(let* [gradient (raster.ad.reverse/grad
                                    (var raster.compiler.passes.scalar.inline-test/option-one-loss)
                                    :wrt [0])
                          g (gradient x)] g)]]
    (let [expanded (inline/inline-deftm-calls source)
          f (eval (list 'fn '[x] expanded))]
      (is (= ((rev/grad #'option-one-loss :wrt [0]) 2.0) (f 2.0)))
      (is (= 4.0 (f 2.0)))))
  (let [source '(let* [vg ((raster.ad.reverse/value+grad
                            (var raster.compiler.passes.scalar.inline-test/option-loss)
                            :wrt []) x y)] vg)
        expanded (inline/inline-deftm-calls source)
        f (eval (list 'fn '[x y] expanded))]
    (is (= [10.0 nil nil] (f 2.0 3.0))))
  (doseq [wrt [[-1] [2] [0 0]]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (inline/inline-value+grad-call
                    {:var-sym 'raster.compiler.passes.scalar.inline-test/option-loss
                     :args '[x y] :mode :value+grad :options [:wrt wrt]}))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"literal :wrt"
                        (#'inline/value+grad-call?
                         '((raster.ad.reverse/value+grad (var example/loss) :wrt runtime-ports)
                           x y)))))

(deftest compiled-single-parameter-grad-preserves-the-public-scalar-result
  (doseq [source [#'option-one-grad #'option-one-grad-bound]]
    (let [f (pipeline/compile-aot source)]
      (is (= 4.0 (f 2.0)))
      (is (= -6.0 (f -3.0))))))

(deftest lifted-arguments-retain-source-types-not-formal-consumer-types
  (doseq [[argument environment] [['(float (aget a 0)) {'a 'doubles}]
                                 ['(clojure.core/aget a 0) {'a 'floats}]]]
    (let [bindings (atom [])
          substitution (#'inline/argument-substitution
                        ['x] [argument] ['double] environment #(swap! bindings conj %))
          id (get substitution 'x)]
      (is (= 'float (:raster.type/tag (meta id))))
      (is (= [[id argument]] @bindings)))))

(deftest checked-constant-evidence-avoids-unnecessary-argument-bindings
  (doseq [argument ['(float 0.0) '(float 1.0) '(long 4)]]
    (is (false? (#'inline/needs-arg-lift? argument))))
  (doseq [argument ['(int 4294967296) '(float (observe!))]]
    (is (#'inline/needs-arg-lift? argument))))

(deftest leaf-bodies-are-inlinable
  (doseq [body ['x 'java.lang.Float/NEGATIVE_INFINITY 42 1.5 nil true]]
    (is (#'inline/inlinable-body? body))))

(deftest lifted-tail-effect-retains-the-statement-binder-contract
  (let [result (:form (#'inline/inline-one-pass
                       '(let* [] (raster.par/map-void! i n nil))))
        binder (first (second result))]
    (is (:raster.effect/effectful (meta binder)))
    (is (= 'raster.par/map-void! (-> result second second first)))))

(deftest inlining-preserves-unused-checked-arguments
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['int] :walked-body [42]})}
    (fn []
      (let [inlined (#'inline/inline-invk '(.invk example/constant (int 4294967296)))]
        (is (thrown? ArithmeticException (eval inlined))
            "an unused checked argument still throws before entering the callee"))
      (let [inlined (#'inline/inline-invk
                    '(.invk example/constant (int (do (swap! calls inc) 3))))]
        (is (= [42 1] (eval (list 'let ['calls '(atom 0)]
                                 [inlined '(deref calls)]))))))))

(deftest direct-inline-arity-is-checked-before-substitution
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['long] :walked-body [42]})}
    (fn []
      (doseq [call ['(.invk example/constant)
                   '(.invk example/constant 1 (swap! calls inc))]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Arity mismatch"
                             (#'inline/inline-invk call)))))))

(deftest both-inline-entrypoints-share-ordered-expansion-construction
  (let [build @#'inline/build-inline-expansion
        modes (atom [])
        info {:params '[unused value] :tags '[int long]
              :walked-body '[(let* [copy value]
                               (swap! calls conj :callee)
                               (+ copy copy))]}]
    (with-redefs-fn
      {#'inline/try-resolve-deftm (fn [& _] info)
       #'inline/build-inline-expansion
       (fn [callee args environment substitute result caller]
         (swap! modes conj (if result :binding :nested))
         (build callee args environment substitute result caller))}
      (fn []
        (doseq [mode [:binding :nested]
                [first-value expected-events expected-result]
                [[3 [:first :second :callee] [:value 14]]
                 [4294967296 [:first] [:error ArithmeticException]]]]
          (let [arguments [(list 'int (list 'do '(swap! calls conj :first) first-value))
                           '(do (swap! calls conj :second) 7)]
                expression (if (= :binding mode)
                             (:form (#'inline/inline-one-pass
                                     (list 'let* ['result (list* 'example/helper arguments)] 'result)))
                             (#'inline/inline-invk (list* '.invk 'example/helper arguments)))
                actual (eval (list 'let ['calls '(atom [])]
                                   [(list 'try [:value expression]
                                          (list 'catch 'Throwable 'error
                                                [:error '(class error)]))
                                    '(deref calls)]))]
            (is (= [expected-result expected-events] actual))))))
    (is (= [:binding :binding :nested :nested] @modes))))

(deftest both-inline-entrypoints-retain-every-body-form
  (with-redefs-fn
    {#'inline/try-resolve-deftm
     (fn [& _] {:params [] :tags []
               :walked-body '[(swap! calls conj :first) (swap! calls conj :second) 42]})}
    (fn []
      (doseq [expression [(:form (#'inline/inline-one-pass
                                 '(let* [result (example/helper)] result)))
                          (#'inline/inline-invk '(.invk example/helper))]]
        (is (= [42 [:first :second]]
               (eval (list 'let ['calls '(atom [])] [expression '(deref calls)])))))
      (binding [inline/*inline-size-limit* 6]
        (is (= '(.invk example/helper)
               (#'inline/inline-invk '(.invk example/helper)))
            "the size budget must cover the whole body, not just its first form")))))

(deftest unused-collection-arguments-still-evaluate-their-elements
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :walked-body [42]})}
    (fn []
      (doseq [argument ['[(swap! calls inc)] '{:key (swap! calls inc)}
                       '#{(swap! calls inc)}]]
        (let [inlined (#'inline/inline-invk (list '.invk 'example/constant argument))]
          (is (= [42 1] (eval (list 'let ['calls '(atom 0)]
                                   [inlined '(deref calls)])))))))))

(deftest local-function-inlining-preserves-call-by-value
  (doseq [body [nil false 42 '(+ x x)]
          argument ['(int (do (swap! calls inc) 3))]]
    (let [source (list 'let* ['f (list 'ftm ['x] body)
                             'result (list 'f argument)] 'result)
          inlined (:form (#'inline/inline-one-pass source))
          ;; The local function declaration remains for later DCE; use a host fn
          ;; solely to execute this intermediate compiler form.
          executable (clojure.walk/postwalk
                      #(if (and (seq? %) (= 'ftm (first %)))
                         (cons 'fn (rest %)) %) inlined)]
      (is (= [(if (seq? body) 6 body) 1]
             (eval (list 'let ['calls '(atom 0)] [executable '(deref calls)])))))))

(deftest binding-inline-accepts-constant-leaves
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [& _] {:params ['x] :tags ['long] :walked-body [42]})}
    (fn []
      (let [inlined (:form (#'inline/inline-one-pass
                           '(let* [result (example/constant (int 4294967296))] result)))]
        (is (thrown? ArithmeticException (eval inlined)))))))

(deftest nested-argument-calls-are-expanded-before-lifting
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['long] :walked-body ['x]})}
    (fn []
      (let [inlined (#'inline/inline-invk
                    '(.invk example/identity (.invk example/identity (+ 1 2))))]
        (is (not-any? #(and (seq? %) (= '.invk (first %)))
                      (tree-seq coll? seq inlined)))
        (is (= 3 (eval inlined)))))))

(deftest inlining-visits-loop-initializers-without-changing-binders
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['long] :walked-body ['x]})}
    (fn []
      (let [inlined (#'inline/inline-invk
                    '(loop* [i 0 acc (.invk example/identity (long (+ i 2)))]
                       (if (< i 3) (recur (inc i) (+ acc i)) acc)))]
        (is (= ['i 'acc] (vec (take-nth 2 (second inlined)))))
        (is (not-any? #(and (seq? %) (= '.invk (first %)))
                      (tree-seq coll? seq inlined)))
        (is (= 5 (eval inlined)))))))

(deftest late-dispatch-resolution-preserves-segmented-fold-schedule-requests
  (let [request {:association :implementation-defined}
        source
        `(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[row rows]] index width
                 [[sum 0.0 :float width
                   (clojure.core/+ sum (clojure.core/aget values index))
                   ~request]]
                 [(clojure.core/float sum)])]
           effect)
        resolved (inline/resolve-generic-deftm-calls source)
        fold (-> resolved second second (nth 5) first)]
    (is (= 6 (count fold)))
    (is (= request (nth fold 5)))))

;; Monomorphic callee taking THREE args (mirrors the concrete-float finetune.train/gblock,
;; which takes 38 and was called with 37).
(deftm arity-callee
  [a :- (Array double) n :- Long s :- Double] :- Double
  (raster.par/reduce acc 0.0 i n
                     (raster.numeric/+ acc (raster.numeric/* (raster.arrays/aget a i) s))))

;; The STALE caller: arity-callee takes (a n s); this passes only (a n).
;; deftm walking is lazy, so the bad call is only reached when value+grad forces the
;; AD-prep inline — exactly where the finetune chain died.
(deftm arity-caller-stale
  [a :- (Array double) n :- Long] :- Double
  (raster.compiler.passes.scalar.inline-test/arity-callee a n))

;; The correct caller — the guard must not over-fire.
(deftm arity-caller-good
  [a :- (Array double) n :- Long s :- Double] :- Double
  (raster.compiler.passes.scalar.inline-test/arity-callee a n s))

(deftest wrong-arity-deftm-call-fails-loud-in-ad-inline
  (testing "value+grad over a stale (wrong-arity) deftm call names the callee and both arities"
    (let [t (try (let [vg (rev/value+grad #'arity-caller-stale)]
                   ;; if grad-body construction is deferred, force it with a call
                   (vg (double-array [1.0 2.0]) 2)
                   nil)
                 (catch Throwable e e))]
      (is (some? t) "a wrong-arity deftm call must throw, not silently mis-inline")
      (is (not (instance? IndexOutOfBoundsException t))
          "must not surface as a bare IndexOutOfBoundsException from PersistentVector")
      (is (instance? clojure.lang.ExceptionInfo t)
          (str "must be a NAMED ex-info, got: " (some-> t class str)))
      (when (instance? clojure.lang.ExceptionInfo t)
        (let [d (ex-data t)]
          (is (= 'raster.compiler.passes.scalar.inline-test/arity-callee (:callee d))
              "names the callee")
          (is (= 3 (:expected-arity d)) "reports the callee's declared arity")
          (is (= 2 (:actual-arity d)) "reports the call's actual arity")
          (is (re-find #"Arity mismatch" (ex-message t))
              (str "message must say what is wrong, got: " (ex-message t))))))))

(deftest correct-arity-call-still-inlines-and-differentiates
  (testing "the guard must not over-fire: a well-formed call differentiates through the inline"
    (let [a (double-array [1.0 2.0])
          [v da] ((rev/value+grad #'arity-caller-good) a 2 3.0)]
      ;; L = s * sum(a) = 3 * (1+2) = 9 ; dL/da_i = s = 3
      (is (= 9.0 (double v)))
      (is (= [3.0 3.0] (vec da))))))
