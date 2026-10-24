(ns raster.ad.paired-helper-emission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.core :as core]
            [raster.arrays]
            [raster.par]
            [raster.numeric :as n]))

(deftest paired-primal-keeps-original-nested-floating-call-context
  (let [scope (create-ns (gensym "raster.paired-floating-context."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm identity-value [x :- Double] :- Double x))
        (eval '(raster.core/deftm ratio [numerator :- Double denominator :- Double] :- Double
                 (clojure.core// (identity-value numerator) (identity-value denominator)))))
      (let [ratio (ns-resolve scope 'ratio)
            source (core/resolve-deftm-var ratio)
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            _ (core/ensure-bytecode-implementation! receiver)
            emission (:raster.core/bytecode-emission (meta receiver))
            artifact (#'reverse/get-paired-helper-program emission [0 1])
            outcome (fn [f arguments]
                      (try {:bits (Double/doubleToRawLongBits (double (apply f arguments)))}
                           (catch Throwable e {:error (class e)})))]
        (doseq [arguments [[1.0 2.0] [1.0 0.0] [0.0 0.0] [-0.0 2.0]]]
          (is (= (outcome @ratio arguments)
                 (outcome (fn [& args] (first (apply (:companion artifact) args))) arguments))
              (str "the paired result must preserve the actual primal for " arguments))))
      (finally (remove-ns (ns-name scope))))))

(deftest paired-owner-retains-a-helper-premise-removed-by-scalar-projection
  (let [scope (create-ns (gensym "raster.projected-helper-premise."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm twice [x :- Double] :- Double
                 (let [left (double x) right (double x)]
                   (raster.ad.reverse/grad-acc left right)))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'twice))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            _ (core/ensure-bytecode-implementation! receiver)
            emission (:raster.core/bytecode-emission (meta receiver))
            prepared (binding [*ns* scope]
                       (reverse/ad-prepare (first (:body emission))
                                           (zipmap (:params emission) (:tags emission))))
            artifact (#'reverse/get-paired-helper-program emission [0])
            [value pullback] ((:companion artifact) 3.0)
            reason #(try ((:guard artifact)) nil
                         (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))]
        (is (= 6.0 value))
        (is (= [2.0] (pullback 1.0)))
        (is (not-any? #{'raster.ad.reverse/grad-acc} (tree-seq coll? seq prepared))
            "the source helper call really was projected away")
        ;; The entry guard captures the actual original program before
        ;; ad-prepare erases this call; scanning only the projected form could
        ;; not establish the dependency. Replacement must not execute.
        (with-redefs [reverse/grad-acc (fn [& _] (throw (AssertionError. "must not invoke replacement")))]
          (is (= :ad-normalized-execution-callee (reason))))
        (is (nil? (reason))))
      (finally (remove-ns (ns-name scope))))))

(deftest paired-helper-retains-primal-pullback-and-historical-dependencies
  (let [scope (create-ns (gensym "raster.paired-helper-emission."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double
                 (raster.numeric/* x x))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'square))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            _ (core/ensure-bytecode-implementation! receiver)
            emission (:raster.core/bytecode-emission (meta receiver))
            artifact (#'reverse/get-paired-helper-program emission [0])
            companion-emission (:companion-emission artifact)
            [value pullback] ((:companion artifact) 3.0)
            dependency (:var (first (:dependency-snapshots emission)))
            reason (fn [f] (try (f) nil
                                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
        (is (= :jvm-bytecode (:backend companion-emission)))
        (is (identical? emission (:emission artifact))
            "the companion does not renew or replace its original input module")
        (is (identical? (:companion artifact) (:implementation companion-emission)))
        (is (= (:params emission) (:params companion-emission)))
        (is (= (:tags emission) (:tags companion-emission)))
        (is (= 'Object (:return-tag companion-emission)))
        (is (some? (bytecode/retained-body-context companion-emission)))
        (is (identical? companion-emission
                        (method-entry/assert-emission-inputs-current! companion-emission))
            "executing the companion must not first materialize one of its guarded callees")
        (is (= 9.0 value))
        (is (= [6.0] (pullback 1.0)))
        (is (= [12.0] (pullback 2.0)))
        (is (= 16.0 (first ((:companion artifact) 4.0))))
        (is (identical? artifact (#'reverse/get-paired-helper-program emission [0])))
        (is (var? dependency))
        ;; Actual successful emitter input, not synthetic current metadata. The
        ;; root changes without a compiler revision; neither a new derivation
        ;; nor an owner's delayed-pullback guard may accept the stale snapshot.
        (with-redefs-fn {dependency (fn [& _] (throw (AssertionError. "must not invoke dependency")))}
          #(do
             (is (= :stale-emission-inputs
                    (reason (fn [] (#'reverse/get-paired-helper-program emission [0])))))
             (is (= :stale-emission-inputs (reason (:guard artifact))))))
        (is (nil? (reason (:guard artifact)))))
      (finally (remove-ns (ns-name scope))))))

(deftest selected-reduction-helper-retains-requested-active-ports
  (let [scope (create-ns (gensym "raster.selected-reduction-helper."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm helper
                 [a :- Double b :- Double xs :- (Array double) cnt :- Long] :- Double
                 (raster.par/reduce acc 0.0 i cnt
                   (raster.numeric/+ acc
                     (raster.numeric/* a (raster.numeric/+ b (raster.arrays/aget xs i))))))))
      (let [operation (symbol (str (ns-name scope)) "helper")
            source (core/resolve-deftm-var (ns-resolve scope 'helper))
            implementation (symbol (str (ns-name scope)) (str (:name (meta source)) "-impl"))
            interface-name (get-in (method-entry/selected-definition operation implementation 4)
                                   [:entry :typed-iface])
            interface-class (:iface-class (get @types/fn-interface-by-name interface-name))
            method (.getMethod ^Class interface-class "invk"
                               (into-array Class [Double/TYPE Double/TYPE (class (double-array 0)) Long/TYPE]))
            captured (with-redefs-fn {#'reverse/get-paired-helper-program
                                      (fn [& _] (throw (AssertionError. "capture must not derive AD")))}
                       #(#'reverse/capture-selected-helper-emission operation implementation 4 method))
            selected (#'reverse/get-selected-paired-helper-program operation implementation 4 method [0 1])
            all (#'reverse/get-selected-paired-helper-program operation implementation 4 method [0 1 2])]
        (is (= :jvm-bytecode (get-in captured [:emission :backend])))
        (is (nil? ((:guard captured))))
        (let [arguments '[caller-a caller-b observations length]
              visible (set arguments)
              region (#'reverse/paired-helper-memory-region selected arguments visible)
              reason (fn [artifact values]
                       (try (#'reverse/paired-helper-memory-region artifact arguments values) nil
                            (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
          (is (= {:kind :region :regions [{:writes #{} :replays '#{observations} :value-facts {}}]}
                 region))
          (is (= :ad-unprojectable-helper-replay (reason selected (disj visible 'observations))))
          (is (= :ad-unprojectable-helper-replay
                 (reason (update selected :replays conj 'private-storage) visible)))
          (is (= :ad-paired-helper-effects
                 (reason (assoc selected :writes '#{private-storage}) visible))))
        (doseq [[a b values] [[2.0 3.0 [1.0 2.0]] [-1.5 0.25 [4.0 -2.0 3.0]] [4.0 -3.0 []]]]
          (let [xs (double-array values) cnt (long (count values))
                expected-a (reduce + 0.0 (map #(+ b %) values))
                expected-b (* a cnt)
                [primal pullback] ((:companion selected) a b xs cnt)
                [all-primal all-pullback] ((:companion all) a b xs cnt)]
            (is (= (* a expected-a) primal all-primal))
            (doseq [seed [1.0 2.0]]
              ((:guard selected))
              ((:guard all))
              (let [expected [(* seed expected-a) (* seed expected-b)]
                    gradient (all-pullback seed)]
                (is (= expected (pullback seed)))
                (is (= expected (subvec gradient 0 2)))
                (is (= (vec (repeat cnt (* seed a))) (vec (nth gradient 2))))))
            (is (= values (vec xs))))))
      (finally (remove-ns (ns-name scope))))))

(deftest selected-paired-helper-is-bound-to-actual-source-and-receiver
  (let [scope (create-ns (gensym "raster.selected-paired-helper."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double
                 (raster.numeric/* x x))))
      (let [operation (symbol (str (ns-name scope)) "square")
            source (core/resolve-deftm-var (ns-resolve scope 'square))
            implementation (symbol (str (ns-name scope)) (str (:name (meta source)) "-impl"))
            receiver (ns-resolve scope (symbol (name implementation)))
            interface-name (get-in (method-entry/selected-definition operation implementation 1)
                                   [:entry :typed-iface])
            interface-class (:iface-class (get @types/fn-interface-by-name interface-name))
            method (.getMethod ^Class interface-class "invk" (into-array Class [Double/TYPE]))
            artifact (#'reverse/get-selected-paired-helper-program operation implementation 1 method [0])
            [value pullback] ((:companion artifact) 3.0)
            reason (fn [f] (try (f) nil
                                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
        (is (= 9.0 value))
        (is (= [6.0] (pullback 1.0)))
        (is (identical? @receiver (:implementation (:emission artifact))))
        (is (nil? (reason (:guard artifact))))
        (doseq [v [source receiver]]
          (with-redefs-fn {v (fn [& _] (throw (AssertionError. "must not execute replacement")))}
            #(is (= :ad-paired-helper-selected-definition (reason (:guard artifact))))))
        (is (nil? (reason (:guard artifact))))
        (is (= :ad-paired-helper-selected-definition
               (reason (fn [] (#'reverse/get-selected-paired-helper-program
                               operation implementation 1 nil [0]))))))
      (finally (remove-ns (ns-name scope))))))
