(ns raster.compiler.pipeline-test
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest testing is]]
            [pattern.nanopass.dialect :as nanopass]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.simplify :as simplify]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.soa-lower :as soa-lower]
            [raster.compiler.passes.scalar.local-products :as local-products]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed-frontend]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.compiler.ad.flatten :as flat]
            [raster.compiler.ir.dialects :as dialects]
            [raster.ad.reverse :as rad]))

(deftest boxed-entry-construction-transports-types-through-the-existing-census
  (let [floating (completion/complete-value (completion/bytecode-boxed-parameter-contract Float/TYPE) 'input)
        array (completion/complete-value
               (completion/bytecode-boxed-parameter-contract (class (double-array 0))) 'array)
        source (list 'let* ['floating floating 'storage array] '[floating storage])
        tagged (#'pipeline/tag-expr-types source {'input 'Object 'array 'Object})
        declarations (mapv #(get (meta %) :raster.type/tag) (take-nth 2 (second tagged)))
        census (#'pipeline/record-fixpoint-census! :final tagged
                {:target-device :cuda:equation-first-compile-gate})]
    (is (= '[float doubles] declarations))
    (is (zero? (:non-exempt-untagged census)))
    (is (zero? (:undevirtualized census)))
    (is (completion/contains-completion? tagged)
        "typedness does not erase the conversion or discharge its owner")))

(defn- isolated-fixpoint [input expand]
  ;; Isolate termination from producer behavior. Real compiled workloads remain
  ;; covered by pipeline-quality-test; these controls exercise metadata invisible
  ;; to Clojure = without asking another pass to invent that metadata for us.
  (with-redefs-fn
    {#'inline/expand-for-backends (fn [form & _] (expand form))
     #'soa-lower/lower-local-constructors (fn [form & _] form)
     #'local-products/lower-local-products identity
     #'pipeline/tag-binding-types (fn [form & _] form)
     #'pipeline/carry-void-binding-contracts identity
     #'pipeline/record-fixpoint-census! (fn [& _])}
    #(#'pipeline/pass-fixpoint input {})))

(deftest fixpoint-consumes-metadata-only-repairs
  (let [input '(let* [value 1.0] value)
        repaired (list 'let* (second input)
                       (with-meta 'value {:raster.type/tag 'double}))
        seen (atom [])
        result (isolated-fixpoint input (fn [form] (swap! seen conj form) repaired))]
    (is (= input repaired) "ordinary equality misses the retained reference type")
    (is (= 2 (count @seen)) "the next iteration must consume newly retained facts")
    (is (= 'double (:raster.type/tag (meta (last (second @seen))))))
    (is (= 'double (:raster.type/tag (meta (last (:form result))))))
    (is (= 1 (get-in result [:stats :fixpoint-iterations])))))

(deftest fixpoint-stability-retains-the-latest-form
  (let [input (with-meta '(let* [value 1.0] value) {:line 1})
        latest (with-meta input {:line 900 :column 12})
        seen (atom 0)
        result (isolated-fixpoint input (fn [_] (swap! seen inc) latest))]
    (is (= 1 @seen) "source locations alone do not require another iteration")
    (is (= {:line 900 :column 12} (meta (:form result))))))

(deftest fixpoint-compares-numeric-carriers-and-semantic-metadata
  (doseq [[left right] [[0.0 -0.0] [(float 1.0) (double 1.0)]
                       [(with-meta 'value {:raster.type/tag 'float})
                        (with-meta 'value {:raster.type/tag 'double})]
                       [(with-meta '[x] {:raster.type/element 'float})
                        (with-meta '[x] {:raster.type/element 'double})]]]
    (is (not (#'pipeline/fixpoint-equivalent? left right))))
  (is (#'pipeline/fixpoint-equivalent? Double/NaN Double/NaN)))

(deftest unsupported-fixpoint-evidence-does-not-certify-stability
  (let [input (with-meta '(let* [value 1.0] value) {:tag Double})
        seen (atom 0)
        result (isolated-fixpoint input
                 (fn [form] (vary-meta form assoc :line (swap! seen inc))))]
    (is (= 5 @seen) "unsupported runtime metadata uses the existing iteration cap")
    (is (= 5 (get-in result [:stats :fixpoint-iterations])))
    (is (= 5 (:line (meta (:form result)))))
    (is (identical? Double (:tag (meta (:form result))))))
  (let [failure (ex-info "unexpected fingerprint failure" {:reason :different-error})]
    (with-redefs [semantic-fingerprint/equivalent? (fn [& _] (throw failure))]
      (is (identical? failure
                     (try (#'pipeline/fixpoint-equivalent? 'x 'x)
                          (catch clojure.lang.ExceptionInfo error error)))))))

(defn- form-contains?
  "True if sym appears anywhere in a nested form (symbols, lists, vectors)."
  [form sym]
  (cond
    (= form sym) true
    (and (symbol? form) (symbol? sym)
         (= (name form) (name sym))) true
    (seq? form) (some #(form-contains? % sym) form)
    (vector? form) (some #(form-contains? % sym) form)
    :else false))

;; ================================================================
;; AD template path verification
;; ================================================================

(deftest ad-transform-uses-templates-test
  (testing "AD transform produces flat code with templates (not closure pullback lookup)"
    (let [form '(let* [r (* x y)] r)
          transformed (rad/transform-body form '[x y])]
      ;; Template path: should NOT contain get-pullback-factory runtime lookups
      (is (not (form-contains? transformed 'get-pullback-factory))
          "Template path should be used, not closure pullback lookup"))))

;; ================================================================
;; Flatten + simplify integration
;; ================================================================

(deftest flatten-simplify-integration-test
  (testing "flatten + simplify cleans up AD noise"
    (let [;; f(x,y) = x*y -> df/dx = y, df/dy = x
          form '(let* [r (raster.numeric/* x y)
                       result__1 r]
                      [result__1
                       (fn* [dy__rad]
                            (let* [d_r__2 dy__rad
                                   dg_x__3 (raster.numeric/* d_r__2 y)
                                   dg_y__4 (raster.numeric/* d_r__2 x)]
                                  [dg_x__3 dg_y__4]))])
          ;; Flatten AD output (no optimizer — that's now user code)
          flat-result (flat/flatten-ad-form form)
          flat-form (:form flat-result)
          simplified (simplify/simplify-derivative flat-form)]
      ;; After simplification, (* 1.0 y) should become y
      ;; The form should not have closure anymore
      (is (not (form-contains? simplified 'fn*))
          "No closures should remain after flattening")
      ;; Should have the primal result as body
      (is (= 'let* (first simplified))
          "Should be a flat let* form"))))

;; ================================================================
;; Pipeline function existence
;; ================================================================

(deftest pipeline-api-test
  (testing "pipeline functions exist"
    (is (fn? pipeline/compile-aot))
    (is (fn? pipeline/show-pipeline))))

(deftest fixpoint-typedness-distinguishes-statements-from-untyped-values
  (let [effect (with-meta '_effect {:raster.effect/effectful true})]
    (testing "an explicit statement contract is independent of the RHS control spelling"
      (is (#'pipeline/census-exempt-binding?
           [effect '(if predicate (raster.par/map-void! i n body) nil)])))
    (testing "the same untyped conditional is not exempt when its value could be observed"
      (is (not (#'pipeline/census-exempt-binding?
                ['result '(if predicate then-value else-value)]))))
    (testing "an unmarked parallel call is not admitted by an operation-name whitelist"
      (is (not (#'pipeline/census-exempt-binding?
                ['result '(raster.par/map-void! i n body)]))))))

(deftest binding-tagging-carries-the-void-statement-contract
  (doseq [expression [(with-meta '(dotimes [i n] (clojure.core/aset output i 0.0))
                         {:raster.type/tag 'float})
                      '(raster.par/segmented-fold-map!
                        [output] [[segment segment-count]] index width
                        [[sum 0.0 :float width sum]] [(float sum)])]]
    (let [tagged (#'pipeline/tag-binding-types
                  (list 'let* ['effect expression] 'output) {})
          binder (first (second tagged))]
      (is (:raster.effect/effectful (meta binder))
          (str "missing statement contract for " (first expression))))))

(deftest void-contract-carrying-reaches-bindings-inside-opaque-loops
  (let [form '(let* [outer
                     (dotimes [i n]
                       (let* [_ (dotimes [j i] (clojure.core/aset output j 0.0))]
                         nil))]
                output)
        carried (#'pipeline/carry-void-binding-contracts form)
        inner-binder (-> carried second second (nth 2) second first)]
    (is (:raster.effect/effectful (meta (first (second carried)))))
    (is (:raster.effect/effectful (meta inner-binder)))
    (is (identical? carried (#'pipeline/carry-void-binding-contracts carried)))))

(deftest void-contract-analysis-preserves-value-regions-and-quoted-data
  (doseq [form ['(let* [x (outer (inner value))] x)
                '(let* [x (let* [saved value] (outer saved))] x)
                '(quote (let* [_ (dotimes [i n] (clojure.core/aset output i 0.0))] nil))
                '(clojure.core/quote
                   (let* [_ (dotimes [i n] (clojure.core/aset output i 0.0))] nil))]]
    (is (identical? form (#'pipeline/carry-void-binding-contracts form)))))

(deftest void-contract-carrying-marks-only-unused-effect-loop-results
  (let [effect-loop '(loop* [i 0]
                       (if (< i n)
                         (do (clojure.core/aset output i 0.0) (recur (inc i)))
                         nil))
        unused (#'pipeline/carry-void-binding-contracts
                (list 'let* ['_ effect-loop 'answer 1] 'answer))
        consumed (#'pipeline/carry-void-binding-contracts
                  (list 'let* ['result effect-loop 'answer '(if result 1 2)] 'answer))]
    (is (:raster.effect/effectful (meta (first (second unused)))))
    (is (not (:raster.effect/effectful (meta (first (second consumed)))))
        "an observable nil result is not silently reclassified as a GPU statement")))

;; ================================================================
;; Pass registry
;; ================================================================

(deftest pass-rename-test
  (testing "core pipeline pass specs exist"
    (let [specs @#'pipeline/pass-specs]
      (is (contains? specs :lower))
      (is (contains? specs :fixpoint))
      (is (contains? specs :dce))
      (is (contains? specs :expand))))
  (testing "removed pass keys are gone"
    (let [specs @#'pipeline/pass-specs]
      (is (not (contains? specs :inline)))
      (is (not (contains? specs :post-ad-inline)))
      (is (not (contains? specs :normalize)))
      (is (not (contains? specs :cse)))
      (is (not (contains? specs :simplify)))
      (is (not (contains? specs :peephole)))
      (is (not (contains? specs :par-fuse))))))

(deftest semantic-admission-shares-ordered-late-wrapper-normalization
  ;; Buffer/representation passes can introduce this whole-let initializer
  ;; after earlier expansion. Both admission attempts must see the SAME safe
  ;; canonicalization, not regain the unsafe operand-hoisting implementation.
  (let [source '(let* [size (let* [rows (record! :size)] (* rows 2))
                      output (allocate! size)] output)
        seen (atom [])
        capture (fn [kind]
                  (fn [candidate & _]
                    (swap! seen conj [kind candidate])
                    {:declined {:reason ::unsupported-fixture}}))
        run (fn [expression rows]
              (let [trace (atom [])
                    execute (binding [*unchecked-math* false]
                              (eval (list 'fn '[record! allocate!] expression)))
                    record! (fn [event] (swap! trace conj event) rows)
                    allocate! (fn [size] (swap! trace conj :allocation) [:buffer size])]
                (try {:value (execute record! allocate!) :trace @trace}
                     (catch ArithmeticException _ {:failure :overflow :trace @trace}))))]
    (with-redefs [structured-route/attempt (capture :structured)
                  typed-route/attempt (capture :typed)]
      (#'pipeline/semantic-program-attempt source {:target-device :ocl:late-wrapper :dtype :float} nil)
      (is (= [:structured :typed] (mapv first @seen)))
      (let [normalized (second (first @seen))]
        (is (identical? normalized (second (second @seen))))
        (is (= 3 (quot (count (second normalized)) 2)))
        (is (= {:value [:buffer 4] :trace [:size :allocation]} (run normalized 2)))
        (doseq [rows [2 Long/MAX_VALUE]]
          (is (= (run source rows) (run normalized rows))))
        (is (= {:failure :overflow :trace [:size]} (run normalized Long/MAX_VALUE))))
      (reset! seen [])
      (#'pipeline/semantic-program-attempt source {:target-device :cpu:late-wrapper :dtype :float} nil)
      (is (= [:typed] (mapv first @seen)))
      (is (identical? source (second (first @seen))))
      (with-redefs-fn {#'pipeline/abstract-machine-for-target (fn [& _] nil)}
        #(let [refused (#'pipeline/pass-soac-fuse source {:target-device :ocl:late-wrapper :dtype :float})]
           (is (= source (:form refused)) "refusal keeps the original nested source")
           (is (= :disabled (get-in refused [:stats :compatibility-fusion]))))))))

(deftest normalized-semantic-admission-does-not-rewrite-or-normalize-source
  (let [source '(let* [result (raster.par/rng-fill! seeds n base-seed)] result)
        options {:target-device :ocl:normalized-source :dtype :long
                 :array-types {'seeds :long} :scalar-types {'n :int 'base-seed :long}}
        normalized (typed-frontend/normalize-source source options)
        seen (atom [])
        structured-seen (atom [])
        discover structured-route/attempt
        project @#'typed-route/source-projection
        result
        (with-redefs-fn
          {#'pipeline/prepare-semantic-source (fn [& _] (throw (AssertionError. "representation replay")))
           #'typed-frontend/normalize-source (fn [& _] (throw (AssertionError. "normalization replay")))
           #'typed-route/attempt (fn [& _] (throw (AssertionError. "ordinary route replay")))
           #'structured-route/attempt (fn [form options]
                                       (swap! structured-seen conj form)
                                       (discover form options))
           #'typed-route/source-projection (fn [form options]
                                             (swap! seen conj form)
                                             (project form options))}
          #(pipeline/normalized-semantic-program-attempt normalized options source))]
    (is (= 1 (count @structured-seen)))
    (is (identical? source (first @structured-seen)))
    (is (identical? normalized (first @seen)))
    (is (some? (:program result)))
    (is (true? (get-in result [:stats :typed-validated])))
    (is (= :analyzed-source (get-in result [:stats :front-end])))))

(deftest certified-butterfly-enters-the-typed-effect-vertical
  (let [source '(let* [step (raster.par/butterfly! re im i half wr wi base)] step)
        options {:dtype :float
                 :array-types {'re :float 'im :float 'wr :float 'wi :float}
                 :scalar-types {'half :long 'base :long}}
        result (#'pipeline/pass-soac-fuse source options)
        bare '(raster.par/butterfly! re im i half wr wi base)
        bare-result (#'pipeline/pass-soac-fuse bare options)]
    (testing "the binding form is certified instead of entering compatibility fusion"
      (is (= :typed-soac (get-in result [:stats :route])))
      (is (nil? (get-in result [:stats :typed-soac-declined])))
      (is (= :typed-soac (get-in result [:form :dialect]))))
    (testing "the bare surface form is normalized into the same typed dialect"
      (is (= :typed-soac (get-in bare-result [:form :dialect]))))))

;; ================================================================
;; Dialect validation
;; ================================================================

(deftest dialect-checkers-test
  (testing "dialect-checkers registry has all pipeline stages"
    (let [checkers (set (keys dialects/dialect-checkers))
          pass-targets (into #{} (map :to) (vals @#'pipeline/pass-specs))
          missing (set/difference pass-targets checkers)]
      (is (empty? missing)
          (str "Missing dialect checkers for pipeline stages: " (sort missing))))))

(deftest diagnostic-runner-validates-pass-output-test
  (testing "diagnostic runner enforces declared output dialects"
    (let [invalid-pass :test-invalid-flattened
          invalid-form '(+ x y)
          original-specs @#'pipeline/pass-specs]
      (alter-var-root #'pipeline/pass-specs
                      assoc
                      invalid-pass
                      {:from :walked
                       :to :flattened
                       :fn (fn [_form _opts] invalid-form)})
      (try
        (try
          (#'pipeline/run-passes-diagnostic '(let* [x 1.0] x) [invalid-pass] {})
          (is false "Expected diagnostic runner to reject invalid pass output")
          (catch clojure.lang.ExceptionInfo ex
            (let [data (ex-data ex)]
              (is (= invalid-pass (:pass data)))
              (is (= :flattened (:dialect data)))
              (is (:validation-error data)))))
        (finally
          (alter-var-root #'pipeline/pass-specs (constantly original-specs)))))))

(defn- runner-probe [runner transform validate observer opts]
  (with-redefs-fn
    {#'pipeline/pass-specs {:probe {:from :walked :to :lowered :fn transform}}
     #'pipeline/validate-dialect! validate}
    #(binding [pipeline/*pass-observer* observer]
       (runner [:input] [:probe] opts))))

(defn- caught-runner-error [f]
  (try (f) nil (catch Throwable error error)))

(deftest pass-runners-share-execution-without-leaking-results-to-observers
  (doseq [runner [pipeline/run-passes #'pipeline/run-passes-diagnostic]
          wrapped? [false true]]
    (let [calls (atom []) events (atom [])
          result (runner-probe
                  runner
                  (fn [source _]
                    (swap! calls conj :transform)
                    (if wrapped? {:form (conj source :output) :stats {:private (Object.)}}
                        (conj source :output)))
                  (fn [& _] (swap! calls conj :validation))
                  #(swap! events conj %) {})
          event (first @events)]
      (is (= [:input :output] (if (= runner pipeline/run-passes) result (:form result))))
      (is (= [:transform :validation] @calls))
      (is (= 1 (count @events)))
      (is (= #{:pass :from :to :status :timing-source :timing-scope
               :transform-ns :validation-ns} (set (keys event))))
      (is (= {:pass :probe :from :walked :to :lowered :status :success
              :timing-source :host-monotonic :timing-scope :inclusive}
             (dissoc event :transform-ns :validation-ns)))
      (is (every? #(and (integer? %) (<= 0 %))
                  ((juxt :transform-ns :validation-ns) event))))))

(deftest shared-runner-keeps-diagnostic-metadata-and-stage-overwrite-semantics
  (with-redefs-fn
    {#'pipeline/pass-specs
     {:first {:from :custom :to :lowered
              :fn (fn [source _] {:form (conj source :first) :stats {:count 1}
                                  :backend :gpu :kernels [:kernel]
                                  :emitted-program :program :cuda-result :cuda})}
      :second {:from :* :to :lowered :fn (fn [source _] (conj source :second))}}
     #'pipeline/validate-dialect! (fn [& _])}
    #(let [result (#'pipeline/run-passes-diagnostic [:input] [:first :second] {} :custom)]
       (is (= [:input :first :second] (:form result)))
       (is (= (:form result) (get-in result [:stages :lowered])))
       (is (= {:lowered {:count 1}} (:stats result)))
       (is (= {:backend-type :gpu :kernels [:kernel]
               :emitted-program :program :cuda-result :cuda}
              (dissoc (:stages result) :lowered)))
       (is (= (:form result)
              (pipeline/run-passes [:input] [:first :second] {} :custom))))))

(deftest shared-runner-retains-transform-only-inline-binding-scope
  (doseq [runner [pipeline/run-passes #'pipeline/run-passes-diagnostic]
          observer [nil (fn [_])]
          [opts inherited expected]
          [[{} false [false false]]
           [{:owned-jvm-calls? true} false [true false]]
           [{:target-device :ocl:probe} false [false true]]
           [{} true [false true]]]]
    (let [seen (atom [])]
      (binding [inline/*owned-jvm-calls?* true
                inline/*inline-scalar-bodies?* inherited]
        (runner-probe runner
                      (fn [source _]
                        (swap! seen conj [inline/*owned-jvm-calls?* inline/*inline-scalar-bodies?*])
                        source)
                      (fn [& _]
                        (swap! seen conj [inline/*owned-jvm-calls?* inline/*inline-scalar-bodies?*]))
                      observer opts))
      (is (= [expected [true inherited]] @seen)))))

(deftest pass-observer-preserves-original-failures-and-the-failing-phase
  (doseq [runner [pipeline/run-passes #'pipeline/run-passes-diagnostic]
          phase [:transform :validation]
          reason [:probe-failure (Object.)]]
    (let [primary (ex-info "compiler failure" {:reason reason :private (Object.)})
          events (atom []) calls (atom [])
          error (caught-runner-error
                 #(runner-probe
                   runner
                   (fn [source _]
                     (swap! calls conj :transform)
                     (if (= phase :transform) (throw primary) source))
                   (fn [& _] (swap! calls conj :validation) (throw primary))
                   (fn [event] (swap! events conj event)) {}))
          event (first @events)]
      (is (identical? primary error))
      (is (= (if (= phase :transform) [:transform] [:transform :validation]) @calls))
      (is (= 1 (count @events)))
      (is (= (cond-> {:status :failure :failure-phase phase
                      :error-class "clojure.lang.ExceptionInfo"}
               (keyword? reason) (assoc :reason reason))
             (select-keys event [:status :failure-phase :reason :error-class])))
      (is (not (contains? event :private)))
      (when (= phase :transform) (is (zero? (:validation-ns event)))))))

(deftest pass-observer-errors-never-mask-compiler-errors-or-notify-twice
  (doseq [runner [pipeline/run-passes #'pipeline/run-passes-diagnostic]
          situation [:compiler-fails :compiler-succeeds :same-throwable]]
    (let [primary (ex-info "compiler" {})
          reporting (if (= situation :same-throwable) primary (ex-info "observer" {}))
          notifications (atom 0)
          error (caught-runner-error
                 #(runner-probe runner
                                (fn [source _]
                                  (if (= situation :compiler-succeeds) source (throw primary)))
                                (fn [& _])
                                (fn [_] (swap! notifications inc) (throw reporting)) {}))]
      (is (identical? (if (= situation :compiler-succeeds) reporting primary) error))
      (is (= 1 @notifications))
      (is (= (if (= situation :compiler-fails) [reporting] [])
             (vec (.getSuppressed ^Throwable primary)))))))

(deftest pass-arrow-preconditions-run-before-transformation-and-observation
  (doseq [runner [pipeline/run-passes #'pipeline/run-passes-diagnostic]]
    (let [calls (atom [])]
      (with-redefs-fn
        {#'pipeline/pass-specs {:probe {:from :custom :to :lowered
                                       :fn (fn [& _] (swap! calls conj :transform))}}}
        #(binding [pipeline/*pass-observer* (fn [_] (swap! calls conj :observer))]
           (is (instance? AssertionError
                          (caught-runner-error (fn [] (runner [:input] [:probe] {})))))))
      (is (empty? @calls)))))

(deftest valid-walked-test
  (testing "walked forms are recognized"
    (is (dialects/valid-walked?
         '(let* [a (* x y)
                 b (+ a 1.0)]
                b)))
    (is (dialects/valid-walked? '(let* [] x)))
    (is (dialects/valid-walked?
         '(let* [i (double x)] i))))
  (testing "terminals are valid walked expressions"
    ;; Numbers, symbols, strings are valid terminals in the Walked dialect
    (is (dialects/valid-walked? 42))
    (is (dialects/valid-walked? 'x))))

(deftest walked-binding-validation-short-circuits-the-formal-grammar
  (testing "the already-authoritative structural fallback does not traverse a large program"
    (with-redefs [nanopass/valid? (fn [& _]
                                    (throw (ex-info "formal grammar was invoked" {})))]
      (is (dialects/valid-walked?
           '(let* [a (* x y)
                   b (+ a 1.0)]
              b)))))
  (testing "non-binding expressions retain formal dialect validation"
    (let [calls (atom [])]
      (with-redefs [nanopass/valid? (fn [dialect value]
                                      (swap! calls conj [dialect value])
                                      true)]
        (is (dialects/valid-walked? 42))
        (is (= [[dialects/Walked 42]] @calls))))))

(deftest valid-flattened-test
  (testing "valid flattened forms"
    (is (dialects/valid-flattened?
         '(let* [a (* x y)] a)))
    (is (dialects/valid-flattened?
         '(let* [a (* x y)
                 b (+ a 1.0)]
                [b a]))))
  (testing "non-flattened rejected"
    (is (not (dialects/valid-flattened? '(+ x y))))
    (is (not (dialects/valid-flattened? '[x y])))))

(deftest valid-rewalked-test
  (testing "rewalked forms reject raw scalar calls in walkable bindings"
    (is (not (dialects/valid-rewalked?
              '(let* [a (raster.numeric/+ x y)] a))))
    (let [result (dialects/validate-rewalked
                  '(let* [a (raster.numeric/+ x y)] a))]
      (is (= :pe-rewalked (:fail result)))
      (is (= '(raster.numeric/+ x y) (get-in result [:details :offending])))))
  (testing "rewalked forms allow effect bindings to stay direct"
    (is (dialects/valid-rewalked?
         '(let* [a (.invk _plus__m_double_double-impl x y)
                 _effect (println a)]
                a)))))

(deftest valid-ad-transformed-test
  (testing "let*-wrapped AD output"
    (is (dialects/valid-ad-transformed?
         '(let* [r (* x y)]
                [r (fn* [dy] [dy dy])]))))
  (testing "raw 2-vector AD output is no longer canonical at this stage"
    (is (not (dialects/valid-ad-transformed?
              '[x (fn* [dy] [dy])])))
    (is (= :ad-transformed
           (:fail (dialects/validate-ad-transformed '[x (fn* [dy] [dy])])))))
  (testing "non-AD forms rejected"
    (is (not (dialects/valid-ad-transformed? '(+ x y))))
    (is (not (dialects/valid-ad-transformed? '[x y z])))))

(deftest validate-walked-test
  (testing "valid-walked? accepts walked forms"
    (is (dialects/valid-walked?
         '(let* [a (* x y)] a))))
  (testing "valid-walked? accepts terminals"
    (is (dialects/valid-walked? 42)))
  (testing "validate-walked returns map for structurally invalid"
    (let [result (dialects/validate-walked {:not "an expr"})]
      (is (map? result))
      (is (= :walked (:fail result)))))
  (testing "valid-let*? rejects non-let forms"
    (is (not (dialects/valid-let*? 42)))
    (is (not (dialects/valid-let*? '(+ x y))))
    (is (dialects/valid-let*? '(let* [a 1] a)))))
