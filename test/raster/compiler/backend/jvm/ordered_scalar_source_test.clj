(ns raster.compiler.backend.jvm.ordered-scalar-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.backend.jvm.terminal-source :as projection]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.core :as core]
            [raster.par :as par]))

(core/deftm shadowed-dimension [n :- Long] :- (Array double)
  (let [^long m (+ (long n) 1)
        ^long n (* (long m) 2)
        out (double-array n)]
    (par/map! out i n double 1.0)))

(core/deftm object-then-overflow [x :- Object n :- Long] :- Double
  (+ (double x) (double (* (long n) (long n)))))

(core/deftm typed-long-division [n :- Long] :- Double (/ (long n) 2))

(core/deftm float-checkpoint [x :- Double] :- Double
  (let [rounded (float x)] (double rounded)))

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
  (doseq [[expression ports policy]
          [['(clojure.core// n 2) [{:symbol 'n :carrier Long/TYPE}] {:unchecked-math false}]
           ['(clojure.core/* n n) [{:symbol 'n :carrier Object}] {:unchecked-math false}]
           ['(clojure.core/long n) [{:symbol 'n :carrier Double/TYPE}] {:unchecked-math false}]
           ['(clojure.core/double true) [] {:unchecked-math false}]
           ['(if true n 2.0) [{:symbol 'n :carrier Long/TYPE}] {:unchecked-math false}]
           ['(clojure.core/* n n) [{:symbol 'n :carrier Long/TYPE}] {:unchecked-math true}]
           ['(clojure.core/* n n) [{:symbol 'n :dtype :long}] {:unchecked-math false}]]]
    (is (= :unclosed-terminal-source-projection
           (reason #(entry/call-with-definition-inputs
                     (fn [] (projection/project-scalar expression ports *ns* policy))))))))

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
