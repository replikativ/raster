(ns raster.ad.whole-bytecode-gradient-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
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
(deftm mismatched-array-alias [xs :- (Array double) x :- Double] :- Double
  (let [^floats unused xs] (n/* x x)))

(use-fixtures :once (fn [run] (response 0.8 0.7 1.0) (run)))

(defn- prepared [v]
  (reverse/prepare-owned-bytecode-value+grad v [0]))

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
