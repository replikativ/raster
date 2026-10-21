(ns raster.compiler.backend.jvm.ordered-scalar-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.backend.jvm.terminal-source :as projection]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.arrays :as arrays]
            [raster.numeric :as numeric]
            [raster.core :as core]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.par :as par]))

(core/deftm shadowed-dimension [n :- Long] :- (Array double)
  (let [^long m (+ (long n) 1)
        ^long n (* (long m) 2)
        out (double-array n)]
    (par/map! out i n double 1.0)))

(core/deftm object-then-overflow [x :- Object n :- Long] :- Double
  (+ (double x) (double (* (long n) (long n)))))

(core/deftm typed-long-division [n :- Long] :- Double (/ (long n) 2))

(core/deftm typed-unchecked-narrow [n :- Long] :- Long
  (long (clojure.core/unchecked-int n)))

(core/deftm selected-prefix-dimension [n :- Long] :- (Array double)
  (let [size (numeric/* n 2)
        out (double-array size)]
    (par/map! out i size double 1.0)))

(core/deftm typed-checked-narrow [n :- Long] :- Long
  (long (clojure.core/int n)))

(defn plain-checked-narrow ^long [^long n] (long (int n)))

(core/deftm float-checkpoint [x :- Double] :- Double
  (let [rounded (float x)] (double rounded)))

(core/deftm unprepared-late-dimension
  [x :- (Array double), early :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! early i 1 double (+ (aget x i) 1.0))
        ^long half (long (/ (long n) 2))
        out (double-array half)]
    (par/map! out i half double 1.0)))

(core/deftm direct-length-dimension [xs :- (Array double)] :- (Array double)
  (let [n (clojure.core/long (clojure.core/alength xs))
        m (+ (long n) 1)
        out (double-array m)]
    (par/map! out i m double 1.0)))

(core/deftm selected-length-dimension [xs :- (Array double)] :- (Array double)
  (let [n (clojure.core/long (arrays/alength xs))
        m (+ (long n) 1)
        out (double-array m)]
    (par/map! out i m double 1.0)))

(defn- outcome [f]
  (try (let [result (f)] [:value (class result)
                         (if (instance? Double result) (Double/doubleToLongBits result) result)])
       (catch Exception error [:error (class error)])))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest prepared-source-occurrence-preserves-the-actual-ordered-dimension
  (let [inputs [0 1 3 -3 Long/MAX_VALUE Long/MIN_VALUE]
        ;; Establish the independent actual JVM oracles BEFORE freezing source
        ;; ownership: a cold entry may legitimately publish its implementation.
        oracles (mapv (fn [n]
                        (let [early (double-array [0.0])
                              result (outcome #(let [out (fixture/write-before-shape
                                                         (double-array [7.0]) early n)]
                                                (long (alength out))))]
                          (is (= [8.0] (vec early)))
                          result)) inputs)
        compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        step (first (get-in compilation [:semantic :attributes :invocation-plan :steps]))
        run #(get (owner/evaluate-ordered-invocation-step
                   compilation step {'n {:type :long :value %}}) :value)]
    (doseq [[n source] (map vector inputs oracles)]
      (is (= source (outcome #(run n)))))
    (is (= :ordered-source-owner-mismatch
           (reason #(owner/evaluate-ordered-invocation-step
                     (dissoc compilation :ordered-source-owner) step {'n {:type :long :value 3}}))))
    (is (= :unclosed-ordered-scalar-executor
           (reason #(owner/evaluate-ordered-invocation-step
                     compilation (assoc step :symbol 'changed) {'n {:type :long :value 3}}))))
    (doseq [value [nil 3.0 (int 3)]]
      (is (= :ordered-scalar-source-arguments
             (reason #(owner/evaluate-ordered-invocation-step
                       compilation step {'n {:type :long :value value}})))))))

(deftest typed-projection-does-not-borrow-clojure-long-division-or-logical-dtypes
  ;; Ordinary Clojure returns Ratio here; the typed bytecode path has a different
  ;; division plan. Neither logical Double nor primitive Long ports close that gap.
  (is (= 1/2 ((fn [^long n] (/ n 2)) 1)))
  (let [actual (typed-long-division 1)]
    (is (= Double (class actual)))
    (is (= 0.5 actual)))
  (is (= -1 (typed-unchecked-narrow Long/MAX_VALUE)))
  (is (thrown? ArithmeticException (typed-checked-narrow Long/MAX_VALUE)))
  (is (thrown? ArithmeticException (plain-checked-narrow Long/MAX_VALUE)))
  (doseq [[expression ports policy]
          [['(clojure.core// n 2) [{:symbol 'n :carrier Long/TYPE}] {:unchecked-math false}]
           ['(clojure.core/* n n) [{:symbol 'n :carrier Object}] {:unchecked-math false}]
           ['(clojure.core/long n) [{:symbol 'n :carrier Double/TYPE}] {:unchecked-math false}]
           ['(clojure.core/int n) [{:symbol 'n :carrier Object}] {:unchecked-math false}]
           ['(clojure.core/int n) [{:symbol 'n :carrier Double/TYPE}] {:unchecked-math false}]
           ['(clojure.core/int n) [{:symbol 'n :carrier Float/TYPE}] {:unchecked-math false}]
           ['(clojure.core/int n) [{:symbol 'n :carrier Boolean/TYPE}] {:unchecked-math false}]
           ['(clojure.core/unchecked-int n) [{:symbol 'n :carrier Long/TYPE}] {:unchecked-math false}]
           ['(clojure.core/double true) [] {:unchecked-math false}]
           ['(if true n 2.0) [{:symbol 'n :carrier Long/TYPE}] {:unchecked-math false}]
           ['(clojure.core/* n n) [{:symbol 'n :carrier Long/TYPE}] {:unchecked-math true}]
           ['(clojure.core/* n n) [{:symbol 'n :dtype :long}] {:unchecked-math false}]]]
    (is (= :unclosed-terminal-source-projection
           (reason #(entry/call-with-definition-inputs
                     (fn [] (projection/project-scalar expression ports *ns* policy))))))))

(deftest integral-projection-matches-plain-and-typed-primitive-oracles
  (let [method (.getMethod (class @#'plain-checked-narrow) "invokeStatic" (into-array Class [Long/TYPE]))
        ports [{:symbol 'n :carrier (first (.getParameterTypes method))}]
        values [0 -1 Integer/MAX_VALUE (inc (long Integer/MAX_VALUE)) Long/MAX_VALUE]]
    (entry/call-with-definition-inputs
     (fn []
       (let [projected (projection/project-scalar '(clojure.core/long (clojure.core/int n))
                                                  ports *ns* {:unchecked-math false})
             execute (owner/compile-function
                      (:params projected) (:source projected)
                      (fn [params body] (list 'fn* params body))
                      (merge projected {:source-policy {:unchecked-math false} :source-ns *ns*
                                        :guard #(entry/assert-definition-inputs-current! @entry/*emission-inputs*)}))]
         (is (= Long/TYPE (:carrier (first ports))))
         (doseq [n values]
           (is (= (outcome #(plain-checked-narrow n))
                  (outcome #(typed-checked-narrow n))
                  (outcome #(execute [n]))))))))))

(deftest eager-owned-prefix-reuses-each-source-result-instead-of-reinterpreting-it
  (let [compilation (equation-first/compile #'shadowed-dimension {:target :ocl:0 :dtype :double})
        original owner/evaluate-ordered-invocation-step
        calls (atom [])
        forbidden (fn [& _] (throw (AssertionError. "owned source fell back to typed interpretation")))]
    (with-redefs [scalar/evaluate-invocation-step forbidden scalar/evaluate-host-equation forbidden
                  owner/evaluate-ordered-invocation-step
                  (fn [c step values]
                    (swap! calls conj (:id step))
                    (original c step values))]
      (is (map? (equation-first/lower compilation [1])))
      (is (= 2 (count @calls)))
      (is (= 2 (count (distinct @calls)))))))

(deftest eager-owned-prefix-preflights-all-executors-before-materialization
  (let [compilation (equation-first/compile #'selected-prefix-dimension {:target :ocl:0 :dtype :double})
        started (atom 0)]
    (is (some? (:ordered-source-owner compilation)))
    (with-redefs [materialization/materialize
                  (fn [& _] (swap! started inc) (throw (AssertionError. "missing executor ran source")))]
      (is (= :unclosed-ordered-source-executor
             (reason #(equation-first/lower compilation [1])))))
    (is (zero? @started))))

(deftest eager-owned-prefix-checks-original-history-before-materialization
  (let [compilation (equation-first/compile #'shadowed-dimension {:target :ocl:0 :dtype :double})
        other (equation-first/compile #'shadowed-dimension {:target :ocl:0 :dtype :double})
        source (core/resolve-deftm-var #'shadowed-dimension {:dtype :double})
        metadata (meta source)
        started (atom 0)]
    (with-redefs [materialization/materialize
                  (fn [& _] (swap! started inc) (throw (AssertionError. "stale source was evaluated")))]
      (is (= :ordered-source-owner-mismatch
             (reason #(equation-first/lower
                       (assoc compilation :ordered-source-owner (:ordered-source-owner other)) [1]))))
      (let [parameters (mapv :symbol (get-in compilation [:semantic :attributes :invocation-plan :parameters]))
            projection {:public-parameters parameters
                        :physical-parameters (mapv (fn [id] {:symbol id :binding id}) parameters)}
            changed (assoc-in compilation
                              [:semantic :attributes :invocation-plan :attributes :parameter-projection]
                              projection)]
        (is (= :ordered-source-owner-mismatch
               (reason #(equation-first/lower changed [1])))
            "even structurally valid projection metadata cannot select another family's evaluator"))
      (try
        (alter-meta! source assoc ::changed-source true)
        (is (= :stale-definition-proof-input
               (reason #(equation-first/lower compilation [1]))))
        (finally (reset-meta! source metadata))))
    (is (zero? @started))))

(deftest eager-evaluator-authority-cannot-be-removed-copied-or-redirected
  (let [compilation (equation-first/compile #'shadowed-dimension {:target :ocl:0 :dtype :double})
        other (equation-first/compile #'shadowed-dimension {:target :ocl:0 :dtype :double})
        started (atom 0)
        forbidden (fn [& _] (swap! started inc) (throw (AssertionError. "unowned eager preparation ran source")))]
    (is (some? (:eager-source-owner compilation)))
    (with-redefs [materialization/materialize forbidden scalar/evaluate-invocation-step forbidden
                  scalar/evaluate-host-equation forbidden owner/evaluate-ordered-invocation-step forbidden]
      (doseq [[changed expected]
              [[(dissoc compilation :eager-source-owner) :eager-source-owner-mismatch]
               [(assoc compilation :eager-source-owner (:eager-source-owner other)) :eager-source-owner-mismatch]
               [(dissoc compilation :ordered-source-owner) :eager-source-owner-mismatch]
               [(assoc compilation :source-ns 'clojure.core) :eager-source-owner-mismatch]
               [(assoc compilation :source-ns nil) :eager-source-owner-mismatch]
               [(assoc (dissoc compilation :eager-source-owner :ordered-source-owner) :source-ns nil)
                :eager-source-owner-mismatch]
               ;; Removing a defrecord basis field already invalidates the
               ;; compilation type; it cannot reach evaluator selection.
               [(dissoc compilation :source-ns) :equation-first-compilation]
               [(dissoc compilation :source-ns :eager-source-owner :ordered-source-owner)
                :equation-first-compilation]]]
        (is (= expected (reason #(equation-first/lower changed [1]))))))
    (is (zero? @started))))

(deftest prepared-steps-follow-invocation-identities-through-shadowed-bindings
  (let [expected (mapv #(vec (shadowed-dimension %)) [0 2 5])
        compilation (equation-first/compile #'shadowed-dimension {:target :ocl:0 :dtype :double})
        plan (get-in compilation [:semantic :attributes :invocation-plan])]
    (doseq [[n result] (map vector [0 2 5] expected)]
      (let [realized (materialization/materialize
                      plan [n] (partial owner/evaluate-ordered-invocation-step compilation))]
        (is (= (count result) (get-in realized [:values (:id (last (:steps plan))) :value])))
        (is (zero? (get-in realized [:attributes :driver-allocations])))))
    (let [source (core/resolve-deftm-var #'shadowed-dimension {:dtype :double})
          metadata (meta source)
          first-step (first (:steps plan))]
      (try
        (alter-meta! source assoc ::changed-source true)
        (is (= :stale-definition-proof-input
               (reason #(owner/evaluate-ordered-invocation-step
                         compilation first-step {'n {:type :long :value 2}}))))
        (finally (reset-meta! source metadata))))))

(deftest late-integral-executors-retain-actual-ports-and-transitive-int-results
  (let [compilation (equation-first/compile #'fixture/write-before-checked-integral-shape
                                           {:target :ocl:0 :dtype :double})
        steps (get-in compilation [:semantic :attributes :invocation-plan :steps])
        first-step (first (filter #(= 'elements (:symbol %)) steps))
        next-step (first (filter #(= 'count (:symbol %)) steps))
        operands {'rows {:type :long :value 2} 'width {:type :long :value 3}}
        result (owner/evaluate-ordered-invocation-step compilation first-step operands)]
    (is (= Integer (class (:value result))))
    (is (= 6 (:value result)))
    (is (= 18 (:value (owner/evaluate-ordered-invocation-step
                       compilation next-step {'elements result 'width {:type :long :value 3}}))))
    (doseq [wrong [nil 2.0 (int 2)]]
      (is (= :ordered-scalar-source-arguments
             (reason #(owner/evaluate-ordered-invocation-step
                       compilation first-step (assoc-in operands ['rows :value] wrong))))))
    (is (= :ordered-scalar-source-arguments
           (reason #(owner/evaluate-ordered-invocation-step
                     compilation next-step {'elements {:type :int :value (long 6)}
                                            'width {:type :long :value 3}}))))
    (let [effects (atom 0)]
      (with-redefs [gpu/make-session (fn [& _] (swap! effects inc))
                    materialization/materialize (fn [& _] (swap! effects inc))]
        (is (= :ordered-source-owner-mismatch
               (reason #(link/instantiate-ordered-invocation!
                         (dissoc compilation :ordered-source-owner)
                         [(double-array [7.0]) (double-array 1) 2 3] {}))))
        (is (zero? @effects))))))

(deftest nullable-object-ports-preserve-explicit-conversion-and-operand-order
  ;; Both oracles are actual source executions, not results from the new walker.
  (let [inputs [[nil Long/MAX_VALUE] [2.0 Long/MAX_VALUE] [2.5 3] [(float 0.5) 2]]
        actual (mapv (fn [[x n]] (outcome #(object-then-overflow x n))) inputs)
        clojure-source (fn [x ^long n] (+ (double x) (double (* n n))))]
    (is (= actual (mapv (fn [[x n]] (outcome #(clojure-source x n))) inputs)))
    (entry/call-with-definition-inputs
     (fn []
       (let [source (core/resolve-deftm-var #'object-then-overflow {:dtype :double})
             frontend (entry/original-source-frontend source)
             definition (:definition frontend)
             facts (:parameter-facts
                    (entry/selected-source-entry-carriers
                     (:operation definition) (:implementation-op definition) 2 source))
             ports (mapv (fn [symbol] (assoc (get facts symbol) :symbol symbol)) '[x n])
             projected (projection/project-scalar
                        '(clojure.core/+ (clojure.core/double x)
                                         (clojure.core/double
                                          (clojure.core/* (clojure.core/long n) (clojure.core/long n))))
                        ports *ns* {:unchecked-math false})
             execute (owner/compile-function
                      (:params projected) (:source projected)
                      (fn [params body] (list 'fn* params body))
                      (merge projected {:source-policy {:unchecked-math false} :source-ns *ns*
                                        :guard #(entry/assert-definition-inputs-current! @entry/*emission-inputs*)}))]
         (is (= Object (get-in facts ['x :carrier])))
         (is (= Long/TYPE (get-in facts ['n :carrier])))
         (is (= actual (mapv #(outcome (fn [] (execute %))) inputs))))))))

(deftest closed-fragment-keeps-explicit-float-checkpoints
  (let [inputs [0.0 -0.0 16777217.0 Double/NaN Double/MAX_VALUE Double/POSITIVE_INFINITY]
        expected (mapv #(outcome (fn [] (float-checkpoint %))) inputs)]
    (entry/call-with-definition-inputs
     (fn []
       (let [source (core/resolve-deftm-var #'float-checkpoint {:dtype :double})
             frontend (entry/original-source-frontend source)
             definition (:definition frontend)
             facts (:parameter-facts
                    (entry/selected-source-entry-carriers
                     (:operation definition) (:implementation-op definition) 1 source))
             projected (projection/project-scalar
                        '(let* [rounded (clojure.core/float x)] (clojure.core/double rounded))
                        [(assoc (get facts 'x) :symbol 'x)] *ns* {:unchecked-math false})
             execute (owner/compile-function
                      (:params projected) (:source projected)
                      (fn [params body] (list 'fn* params body))
                      (merge projected {:source-policy {:unchecked-math false} :source-ns *ns*
                                        :guard #(entry/assert-definition-inputs-current! @entry/*emission-inputs*)}))]
         (is (= Double/TYPE (get-in projected [:result-facts :carrier])))
         (is (= expected (mapv #(outcome (fn [] (execute [%]))) inputs))))))))

(deftest executor-coverage-is-complete-before-any-capture-or-source-mutation
  (let [compilation (equation-first/compile #'unprepared-late-dimension
                                           {:target :ocl:0 :dtype :double})
        early (double-array [0.0])
        sessions (atom 0)
        evaluations (atom 0)]
    (with-redefs [gpu/make-session (fn [& _] (swap! sessions inc)
                                   (throw (Exception. "unexpected session")))
                  owner/evaluate-ordered-invocation-step
                  (fn [& _] (swap! evaluations inc) (throw (Exception. "unexpected source")))]
      (is (= :unclosed-ordered-source-executor
             (reason #(link/instantiate-ordered-invocation!
                       compilation [(double-array [7.0]) early 6] {}))))
      (is (= :unclosed-ordered-source-executor
             (reason #(#'compiled/prepare-ordered-equation-first
                       compilation [(double-array [7.0]) early 6] {} nil))))
      (is (zero? @sessions))
      (is (zero? @evaluations))
      (is (= [0.0] (vec early))))))

(deftest public-carrier-preflight-precedes-source-and-driver-work
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})]
    (doseq [n [(int 3)]]
      (let [arguments [(double-array [7.0]) (double-array [0.0]) n]
            state (invocation-link/begin-ordered-realization compilation arguments {})]
        (is (= :unclosed-ordered-source-executor
               (reason #(owner/assert-ordered-executor-coverage! compilation state arguments))))))
    (let [arguments [(double-array [7.0]) (double-array [0.0]) 3]
          state (invocation-link/begin-ordered-realization compilation arguments {})]
      (is (identical? state (owner/assert-ordered-executor-coverage! compilation state arguments))))))

(deftest length-carriers-come-from-owned-source-not-logical-shape-dtypes
  (doseq [source [#'direct-length-dimension #'selected-length-dimension]]
    (let [inputs (mapv double-array [0 1 5])
          expected (mapv #(long (alength (source %))) inputs)
          compilation (equation-first/compile source {:target :ocl:0 :dtype :double})
          plan (get-in compilation [:semantic :attributes :invocation-plan])]
      (doseq [[xs count] (map vector inputs expected)]
        (let [state (invocation-link/begin-ordered-realization compilation [xs] {})
              _ (is (identical? state (owner/assert-ordered-executor-coverage! compilation state [xs])))
              realized (materialization/materialize
                        plan [xs] (partial owner/evaluate-ordered-invocation-step compilation))
              length-value (get-in realized [:values (:id (first (:steps plan))) :value])
              result (get-in realized [:values (:id (last (:steps plan))) :value])]
          (is (identical? Long (class length-value)))
          (is (= count result)))))))

(deftest selected-length-needs-its-actual-computation-and-historical-lookup
  (let [compilation (equation-first/compile #'selected-length-dimension
                                           {:target :ocl:0 :dtype :double})
        program (:semantic compilation)
        plan (get-in program [:attributes :invocation-plan])
        step (first (:steps plan))
        expression (second (second (:source program)))
        source-facts {(:source step) {:carrier Object :required-array-class (class (double-array 0))}}
        project @(ns-resolve 'raster.compiler.backend.jvm.source-owner 'owned-long-length-facts)
        body entry/selected-definition-body]
    (entry/call-with-definition-inputs
     (fn []
       (is (= Long/TYPE (:carrier (project expression step source-facts *ns*))))
       (doseq [alter [(fn [context] (assoc context :body ['(clojure.core/long 7)]))
                      (fn [context] (assoc context :body ['(throw (Exception. "not a projection"))]))
                      (fn [context] (assoc context :pending-source? false :dependency-snapshots []))]]
         (with-redefs [entry/selected-definition-body
                       (fn [definition]
                         (let [context (body definition)]
                           (if (= 'raster.arrays/alength (:operation definition))
                             (alter context) context)))]
           (is (nil? (project expression step source-facts *ns*)))))))))
