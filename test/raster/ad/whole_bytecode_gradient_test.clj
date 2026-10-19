(ns raster.ad.whole-bytecode-gradient-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.source-owner :as source-owner]))

(deftm response [a :- Double k :- Double x :- Double] :- Double
  (n/* a (n// x (n/+ k x))))
(deftm tail-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [amplitude (aget theta 0) kappa (aget theta 1)]
    (n/* 2.0 (response amplitude kappa exposure))))
(deftm nested-tail-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [amplitude (aget theta 0) kappa (aget theta 1)]
    (n/* 2.0 (n/- (response amplitude kappa exposure) 0.3))))
(deftm shadow-tail-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [amplitude (aget theta 0) amplitude (aget theta 1) kappa (aget theta 2)]
    (n/* 2.0 (response amplitude kappa exposure))))
(deftm widened-return [x :- Float] :- Double x)
(deftm constant-return [x :- Double] :- Double 7)
(deftm intermediate-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [a (aget theta 0) k (aget theta 1)
        power (n/* a a)
        scaled (n/+ power 0.5)]
    (n/* 2.0 (response scaled k exposure))))
(deftm unused-intermediate [theta :- (Array double) exposure :- Double] :- Double
  (let [a (aget theta 0) k (aget theta 1)
        third (aget theta 2) unused (n/* third exposure)]
    (n/* 2.0 (response a k exposure))))
(deftm response-local [theta :- (Array double) exposure :- Double] :- Double
  (let [a (aget theta 0) k (aget theta 1)
        contribution (response a k exposure)]
    (n/* 2.0 contribution)))
(deftm float-helper [x :- Float] :- Float (n/* x x))
(deftm widened-object-local [x :- Float] :- Double
  (let [contribution (float-helper x)] contribution))
(deftm scoped-input-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [a (aget theta 0) k (aget theta 1)]
    (n/* 2.0 (response (let [a (n/* a a)] (n/+ a 0.5)) k exposure))))
(deftm array-input-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [a (aget theta 0) k (aget theta 1)]
    (n/* 2.0 (response a k (aget theta 2)))))
(deftm count-input-factor [theta :- (Array double) cnt :- Long] :- Double
  (let [a (aget theta 0) k (aget theta 1)]
    (n/* 2.0 (response a k (double cnt)))))
(deftm explicit-float [x :- Double] :- Float (float x))
(deftm scalar-carry-region [x :- Double cnt :- Long] :- Double
  (par/reduce acc 1.0 i cnt (n/* acc x)))
(deftm active-seed-region [x :- Double seed :- Double cnt :- Long] :- Double
  (par/reduce acc seed i cnt (n/* acc x)))
(deftm overwritten-carry-region [x :- Double cnt :- Long] :- Double
  (par/reduce acc x i cnt x))
(deftm packed-observation-region [x :- Double data :- (Array double) cnt :- Long
                                  stride :- Long offset :- Long] :- Double
  (par/reduce acc 0.0 i cnt
    (n/+ acc (n/* x (aget data (+ (* (long i) stride) offset))))))
(def index-events (atom []))
(deftm effectful-index-region [x :- Double data :- (Array double)] :- Double
  (n/* x (aget data (do (swap! index-events conj :index) 0))))
(deftm array-carry-region [xs :- (Array double) coefficient :- Double cnt :- Long] :- Double
  (let [gain (n/* coefficient coefficient)]
    (par/reduce acc 0.0 i cnt (n/+ acc (n/* gain (aget xs i))))))
(deftm mismatched-array-alias [xs :- (Array double) x :- Double] :- Double
  (let [^floats unused xs] (n/* x x)))

(use-fixtures :once (fn [run] (response 0.8 0.7 1.0) (run)))

(defn- prepared [v]
  (reverse/prepare-owned-bytecode-value+grad v [0]))

(deftest owned-ordered-reductions-execute-with-active-seeds-and-zero-trips
  (let [prepared (reverse/prepare-owned-bytecode-value+grad #'active-seed-region [0 1])
        invoke (#'reverse/compile-prepared-value+grad prepared)]
    (doseq [x [0.7 1.2] seed [0.4 -0.3] cnt [-1 0 1 3]]
      (let [[value dx ds inactive] (invoke x seed cnt)
            trips (max 0 cnt)]
        (is (nil? inactive))
        (is (< (Math/abs (- value (* seed (Math/pow x trips)))) 1e-12))
        (is (< (Math/abs (- dx (if (zero? trips) 0.0 (* seed trips (Math/pow x (dec trips)))))) 1e-12))
        (is (< (Math/abs (- ds (Math/pow x trips))) 1e-12))))
    (is (thrown? ArithmeticException (invoke 0.7 0.4 Long/MAX_VALUE)))))

(deftest overwritten-carry-does-not-require-a-carry-cotangent
  (let [invoke (#'reverse/compile-prepared-value+grad (prepared #'overwritten-carry-region))]
    (doseq [cnt [-1 0 1 3]]
      (is (= [0.7 1.0 nil] (invoke 0.7 cnt))))))

(deftest packed-inactive-observations-use-primitive-owned-index-siblings
  (let [p (prepared #'packed-observation-region)
        invoke (#'reverse/compile-prepared-value+grad p)
        regions (get-in p [:source-owner :entry :input-regions])]
    (is (= 1 (count regions)))
    (is (= 'int (get-in regions [0 :spec :return-tag])))
    (is (= :ad-unowned-bytecode-input-spec
           (try (#'reverse/compile-prepared-value+grad
                  (update-in p [:source-owner :entry :input-regions 0 :spec] assoc :return-tag 'long))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (doseq [x [0.7 1.2] cnt [0 1 3] stride [0 1 2] offset [0 1]]
      (let [data (double-array [0.2 0.8 1.1 0.4 0.6 0.9])
            total (reduce + 0.0 (map #(aget data (+ (* % stride) offset)) (range cnt)))
            [value dx inactive-data inactive-count inactive-stride inactive-offset]
            (invoke x data cnt stride offset)]
        (is (< (Math/abs (- value (* x total))) 1e-12))
        (is (< (Math/abs (- dx total)) 1e-12))
        (is (every? nil? [inactive-data inactive-count inactive-stride inactive-offset]))))
    (is (= [1.4 2.0 nil nil nil nil] (invoke 0.7 (double-array [2.0]) 1 1 4294967296)))
    (is (thrown? ArrayIndexOutOfBoundsException (invoke 0.7 (double-array [2.0]) 1 1 Long/MAX_VALUE)))
    (is (thrown? ArrayIndexOutOfBoundsException (invoke 0.7 (double-array [2.0]) 1 1 -1)))
    (is (thrown? NullPointerException (invoke 0.7 nil 1 1 0)))
    (is (= [0.0 0.0 nil nil nil nil] (invoke 0.7 nil 0 1 0)))))

(deftest inactive-index-ports-do-not-prove-effect-closure
  (let [before @index-events]
    (is (= :ad-unclosed-bytecode-body-region
           (try (prepared #'effectful-index-region)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= before @index-events))))

(deftest actual-reduction-step-uses-the-same-owned-region-producer
  (let [source (core/resolve-deftm-var #'scalar-carry-region)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        _ (core/ensure-current-owned-bytecode! receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        original (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission original)
        locals (get-in observed [:loop :locals])
        env (into {} (map (fn [[id facts]]
                           [id {:value (vary-meta id assoc :raster.type/tag (symbol (name (:type facts))))
                                :facts facts}])) locals)
        carry (second (get-in observed [:loop :variables]))
        context {:body [(nth original 5)]
                 :return-stack-type (get-in locals [carry :type])
                 :result-context (get-in observed [:recur :values 1 :context])}
        program (with-bindings {#'reverse/*gensym-counter* (atom 0)}
                  (#'reverse/owned-bytecode-region-program
                    emission (mapv :value (vals env))
                    [(get-in env [carry :value]) (get-in env ['x :value])]
                    context env))]
    (is (some? observed))
    (is (seq (:bindings program)))
    (is (= 'double (:raster.type/tag (meta (:result program)))))
    (is (pos? (.size ^java.util.IdentityHashMap (:calls program))))))

(deftest actual-step-entry-preserves-object-captures-and-inactive-indexed-data
  (let [source (core/resolve-deftm-var #'array-carry-region)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        _ (core/ensure-current-owned-bytecode! receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        original (first (vals (:reduction-catalog emission)))
        body-context (bytecode/retained-body-context emission)
        params (mapv #(vary-meta %1 assoc :raster.type/tag %2) (:params emission) (:tags emission))
        env (into {} (map (fn [id value] [id {:value value :facts (get-in body-context [:locals id])}])
                         (:params emission) params))
        binding-record (bytecode/retained-binding-context emission (first (:body emission)) 0)
        gain (with-meta 'captured_gain {:tag 'Object :raster.type/tag 'double})
        env (assoc env (:binder binding-record) {:value gain :facts (:after binding-record)})
        produce (fn [original env active]
                  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
                    (#'reverse/owned-bytecode-reduction-step-program emission original params active env)))
        program (produce original env [gain])
        reads (filter #(and (seq? %) (= 'clojure.core/aget (first %)))
                      (tree-seq coll? seq (:bindings program)))
        completions (keep completion/canonical-parts (tree-seq coll? seq (:bindings program)))]
    (is (= :ref (get-in binding-record [:after :type])))
    (is (identical? Object (get-in binding-record [:after :reference-class])))
    (is (= 'Object (:tag (meta gain))))
    (is (= 'int (:raster.type/tag (meta (:index program)))))
    (is (= 'double (:raster.type/tag (meta (:carry program)))))
    (is (some #(= (:index program) (nth % 2 nil)) reads))
    (is (some #(= gain (:operand %)) completions))
    (is (= :ad-unclosed-bytecode-reduction-entry
           (try (produce (with-meta (apply list original) (meta original)) env [gain]) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= :ad-unclosed-bytecode-reduction-entry
           (try (produce original (update-in env [(:binder binding-record) :facts :slot] inc) [gain]) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= :ad-unclosed-bytecode-body-region
           (try (produce original env [gain (first params)]) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(defn- check [v values offset zero-prefix]
  (let [p (prepared v)
        f (#'reverse/compile-prepared-value+grad p)]
    (is (= :jvm-bytecode (get-in p [:source-owner :backend])))
    (is (= :unprojected-jvm-source-owner
           (try (source-owner/assert-portable-source! p)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (doseq [[a k x] [[0.8 0.7 1.0] [1.2 0.4 0.2] [-0.2 1.1 2.0]]]
      (let [theta (double-array (values a k))
            [value gradient inactive] (f theta x)
            den (+ k x)
            expected (- (* 2.0 (/ (* a x) den)) offset)
            expected-gradient (into (vec (repeat zero-prefix 0.0))
                                    [(* 2.0 (/ x den)) (* -2.0 (/ (* a x) (* den den)))])]
        (is (nil? inactive))
        (is (< (Math/abs (- value expected)) 1e-12))
        (is (instance? (class (double-array 0)) gradient))
        (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-12) expected-gradient gradient))) ))))

(deftest whole-caller-keeps-the-array-read-scatter-chain
  (check #'tail-factor vector 0.0 0))
(deftest whole-caller-composes-nested-tail-calls
  (check #'nested-tail-factor vector 0.6 0))
(deftest whole-caller-respects-sequential-lexical-shadowing
  (check #'shadow-tail-factor (fn [a k] [17.0 a k]) 0.0 1))

(deftest original-method-return-completes-before-public-loss-and-seed
  (let [p (reverse/prepare-owned-bytecode-value+grad #'widened-return [0])
        f (#'reverse/compile-prepared-value+grad p)]
    (doseq [x [(float 0.25) (float -0.0) Float/POSITIVE_INFINITY Float/NaN]]
      (let [[value gradient] (f x)]
        (is (instance? Double value))
        (is (= (Double/doubleToRawLongBits (widened-return x))
               (Double/doubleToRawLongBits value)))
        (is (instance? Float gradient))
        (is (= (float 1.0) gradient)))))
  (let [p (reverse/prepare-owned-bytecode-value+grad #'constant-return [0])
        f (#'reverse/compile-prepared-value+grad p)]
    (is (= [7.0 0.0] (f 2.0)))))

(deftest unused-array-binder-cannot-discard-its-original-checkcast
  (is (thrown? ClassCastException (mismatched-array-alias (double-array 2) 2.0)))
  (is (= :ad-unclosed-bytecode-body-region
         (try (reverse/prepare-owned-bytecode-value+grad #'mismatched-array-alias [1])
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))

(deftest object-valued-numeric-locals-complete-only-at-their-consumers
  (let [p (prepared #'intermediate-factor)
        f (#'reverse/compile-prepared-value+grad p)]
    (let [object-helpers (filter #(identical? Object (:result-carrier %)) (:helpers p))
          physical-bindings (filter #(= 'Object (:tag (meta (first %))))
                                    (partition 2 (get-in p [:program :forward-bindings])))]
      (is (seq object-helpers))
      (is (seq physical-bindings))
      ;; Declared arithmetic rules retain the original selected call. A
      ;; derived helper instead produces an Object-completed paired primal.
      ;; Neither path may introduce an eager FP completion at this local.
      (is (not-any? (fn [[_ init]]
                      (= :jvm-bytecode-floating-input
                         (get-in (completion/canonical-parts init) [:contract :source-context])))
                    physical-bindings)))
    (doseq [[a k x] [[0.8 0.7 1.0] [1.2 0.4 0.2] [-0.2 1.1 2.0]]]
      (let [theta (double-array [a k])
            [value gradient] (f theta x)
            den (+ k x)
            expected-k (- (/ (* 2.0 (+ (* a a) 0.5) x) (* den den)))]
        (is (= (intermediate-factor theta x) value))
        (is (< (Math/abs (- (aget gradient 0) (/ (* 4.0 a x) den))) 1e-12))
        (is (< (Math/abs (- (aget gradient 1) expected-k)) 1e-12))))))

(deftest unused-object-valued-numerical-initializers-are-still-evaluated
  (is (thrown? ArrayIndexOutOfBoundsException
               (unused-intermediate (double-array [0.8 0.7]) 1.0)))
  (let [p (prepared #'unused-intermediate)
        f (#'reverse/compile-prepared-value+grad p)]
    (is (thrown? ArrayIndexOutOfBoundsException (f (double-array [0.8 0.7]) 1.0)))))

(deftest derived-helper-object-result-keeps-its-paired-primal-boxed
  (let [p (prepared #'response-local)
        f (#'reverse/compile-prepared-value+grad p)
        bindings (partition 2 (get-in p [:program :forward-bindings]))]
    (is (some (fn [[id init]]
                (and (= 'Object (:tag (meta id)))
                     (= completion/object-input-contract
                        (:contract (completion/canonical-parts init))))) bindings))
    (let [[value gradient] (f (double-array [0.8 0.7]) 1.0)]
      (is (= (response-local (double-array [0.8 0.7]) 1.0) value))
      (is (< (Math/abs (- (aget gradient 0) (/ 2.0 1.7))) 1e-12)))))

(deftest object-float-local-retains-helper-rounding-before-method-widening
  (let [p (prepared #'widened-object-local)
        f (#'reverse/compile-prepared-value+grad p)]
    (doseq [x [(float 0.3) (float -0.0) Float/POSITIVE_INFINITY Float/NaN]]
      (let [[value gradient] (f x)]
        (is (instance? Double value))
        (is (instance? Float gradient))
        (is (= (Double/doubleToRawLongBits (widened-object-local x))
               (Double/doubleToRawLongBits value)))))))

(deftest compound-input-regions-compose-with-the-existing-lexical-producer
  (doseq [v [#'scoped-input-factor #'array-input-factor]]
    (let [p (prepared v)
          f (#'reverse/compile-prepared-value+grad p)
          values [0.8 0.7 1.1]
          primal #(v (double-array %) 1.1)
          [value gradient] (f (double-array values) 1.1)]
      (is (= (primal values) value))
      (doseq [index (range 3)]
        (let [fd (/ (- (primal (update values index + 1e-6))
                       (primal (update values index - 1e-6))) 2e-6)]
          (is (< (Math/abs (- fd (aget gradient index))) 1e-8)))))))

(deftest compound-input-source-correspondence-requires-the-original-occurrence
  (let [p (prepared #'scoped-input-factor)
        emission (get-in p [:source-owner :emission])
        original (first (filter #(= 'raster.ad.whole-bytecode-gradient-test/response
                                    (:raster.op/original (meta %)))
                                (vals (:invocation-catalog emission))))
        context (bytecode/retained-input-source-context emission original 0)]
    (is (some? context))
    (is (identical? (nth original 2) (:source-original context)))
    (is (nil? (bytecode/retained-input-source-context
               emission (with-meta (apply list original) (meta original)) 0)))
    (is (nil? (bytecode/retained-input-source-context emission original -1)))
    (is (nil? (bytecode/retained-input-source-context emission original 3)))))

(deftest explicit-primitive-casts-use-actual-emitted-conversion-evidence
  (let [p (prepared #'count-input-factor)
        f (#'reverse/compile-prepared-value+grad p)
        emission (get-in p [:source-owner :emission])
        original (first (filter #(bytecode/retained-floating-conversion emission %)
                                (vals (:invocation-catalog emission))))
        context (bytecode/retained-floating-conversion emission original)]
    (is (= :long (:input-type context)))
    (is (identical? Double/TYPE (:target-carrier context)))
    (is (nil? (bytecode/retained-floating-conversion
               emission (with-meta (apply list original) (meta original)))))
    (doseq [cnt [1 3 19]]
      (let [[value gradient inactive] (f (double-array [0.8 0.7]) cnt)
            den (+ 0.7 cnt)]
        (is (= (count-input-factor (double-array [0.8 0.7]) cnt) value))
        (is (nil? inactive))
        (is (< (Math/abs (- (aget gradient 0) (/ (* 2.0 cnt) den))) 1e-12)))))
  (let [p (prepared #'explicit-float)
        f (#'reverse/compile-prepared-value+grad p)]
    (doseq [x [0.1 -0.0 Double/MAX_VALUE Double/NaN]]
      (let [[value gradient] (f x)]
        (is (instance? Float value))
        (is (= (Float/floatToRawIntBits (explicit-float x))
               (Float/floatToRawIntBits value)))
        (is (= 1.0 gradient))))))

(deftest producer-renewal-replaces-code-without-revalidating-old-artifacts
  (let [scope (create-ns (gensym "raster.ad.owned-renewal."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm leaf [x :- Double] :- Double (raster.numeric/* x x)))
        (eval '(raster.core/deftm caller [x :- Double] :- Double
                 (raster.numeric/* 2.0 (leaf x))))
        (let [v (ns-resolve scope 'caller)
              old (prepared v)
              f (#'reverse/compile-prepared-value+grad old)
              source (core/resolve-deftm-var v)
              receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
              old-emission (get-in old [:source-owner :emission])
              dependency (core/resolve-deftm-var (ns-resolve scope 'leaf))]
          (is (= [18.0 12.0] (f 3.0)))
          ;; A historical observation really changes. Never patch its snapshot
          ;; or execute a numerical warm-up to conceal that fact.
          (alter-meta! dependency assoc ::renewal-trigger (Object.))
          (is (thrown? clojure.lang.ExceptionInfo (f 3.0)))
          (let [fresh (core/ensure-current-owned-bytecode! receiver)]
            (is (not (identical? old-emission fresh)))
            (is (not (identical? (:implementation old-emission) (:implementation fresh))))
            (is (identical? (:implementation fresh) @receiver)))
          (is (thrown? clojure.lang.ExceptionInfo (f 3.0)))
          (let [p (prepared v)
                g (#'reverse/compile-prepared-value+grad p)]
            (is (= [18.0 12.0] (g 3.0))))
          ;; Legacy/manual metadata without a usable source plan cannot be
          ;; renewed from walked-body=nil. Refuse before entering the emitter.
          (alter-meta! dependency assoc ::renewal-trigger (Object.))
          (alter-meta! source assoc :raster.core/deftm-annotations nil)
          (alter-meta! receiver assoc-in
                       [:raster.core/bytecode-emission :source-entry :annotations] nil)
          (is (= :unusable-bytecode-renewal-source
                 (with-redefs [core/do-bytecode-upgrade!
                               (fn [& _] (throw (AssertionError. "invalid renewal reached emission")))]
                   (try (core/ensure-current-owned-bytecode! receiver)
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
      (finally (remove-ns (ns-name scope))))))
