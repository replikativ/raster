(ns raster.compiler.passes.scalar.local-constructor-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [defvalue deftm]]
            [raster.sci.distributions :as dist]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.cpu.aot :as aot]
            [raster.compiler.passes.scalar.soa-lower :as sl]
            [raster.compiler.passes.scalar.dce :as dce]))

(defvalue LocalFloatPair [a :- Double b :- Double])
(defvalue LocalIntegerPair [a :- Long b :- Integer])
(def observations (atom []))
(def dynamic-value (atom 1.0))

(deftm density-buffer [x :- Double s :- Double] :- (Array double)
  (let [out (double-array 1)]
    (aset out 0 (dist/logpdf (dist/->Normal 0.0 s) x))
    out))

(defn- project [source]
  (sl/lower-local-constructors source {:source-ns (the-ns 'raster.compiler.passes.scalar.local-constructor-test)}))

(defn- run [source]
  (binding [*ns* (the-ns 'raster.compiler.passes.scalar.local-constructor-test)] (eval source)))

(deftest projections-preserve-values-and-read-once
  (doseq [source ['(let* [p (->LocalFloatPair 1.0 2.0)] (+ (.a p) (.b p)))
                  '(.a (->LocalFloatPair 3.0 4.0))
                  '(let* [p (->LocalFloatPair @dynamic-value 2.0)
                          ignored (reset! dynamic-value 7.0)] (+ (.a p) (.a p)))]]
    (reset! dynamic-value 1.0)
    (let [reference (run source)]
      (reset! dynamic-value 1.0)
      (is (= reference (run (project source)))))))

(deftest argument-evaluation-precedes-conversion
  (let [source '(let* [p (->LocalFloatPair "bad" (do (swap! observations conj :later) 2.0))]
                 (.b p))]
    (doseq [expression [source (project source) (:form (dce/eliminate-dead-bindings (project source)))]]
      (reset! observations [])
      (is (thrown? ClassCastException (run expression)))
      (is (= [:later] @observations)))))

(deftest unused-fields-still-evaluate-and-convert
  (let [source '(let* [p (->LocalFloatPair 1.0 "bad")] (.a p))]
    (is (thrown? ClassCastException (run source)))
    (is (thrown? ClassCastException (run (project source))))
    (is (thrown? ClassCastException (run (:form (dce/eliminate-dead-bindings (project source))))))))

(deftest escapes-foreign-and-redefined-factories-are-not-projected
  (let [escaping '(let* [p (->LocalFloatPair 1.0 2.0)] p)
        lexical '(let* [->LocalFloatPair (fn* [x y] [x y])] (->LocalFloatPair 1.0 2.0))
        loop-scope '(loop* [->LocalFloatPair other-factory]
                      (.a (->LocalFloatPair 1.0 2.0)))
        source '(let* [p (->LocalFloatPair 1.0 2.0)] (.a p))]
    (is (= escaping (project escaping)))
    (is (= lexical (project lexical)))
    (is (= loop-scope (project loop-scope)))
    (with-redefs [->LocalFloatPair (fn [x y] [x y])]
      (is (= source (project source))))))

(deftest projection-preserves-source-metadata
  (let [bindings (with-meta '[p (->LocalFloatPair 1.0 2.0)] {:line 17})
        source (with-meta (list 'let* bindings '(.a p)) {:line 16 :file "probe.clj"})
        result (project source)]
    (is (= (meta source) (meta result)))
    (is (= (meta bindings) (meta (second result))))))

(deftest numeric-field-conversion-contract
  (is (= 3 (run (project '(let* [p (->LocalIntegerPair 1 2)] (+ (.a p) (.b p)))))))
  (doseq [source ['(let* [p (->LocalIntegerPair 1 2147483648)] (.a p))
                  '(let* [p (->LocalIntegerPair 9223372036854775808N 1)] (.b p))]]
    (let [failure (fn [x] (try (run x) nil (catch Exception e (class e))))]
      (is (some? (failure source)))
      (is (= (failure source) (failure (project source))))
      (is (= (failure source) (failure (:form (dce/eliminate-dead-bindings (project source)))))))))

(deftest constructed-density-compiles-through-shared-projection
  (let [compiled (pipeline/compile-aot #'density-buffer :dtype :double)]
    (doseq [[x s] [[0.5 2.0] [-0.75 0.4]]]
      (let [expected (* -0.5 (+ (Math/log (* 2.0 Math/PI)) (* 2.0 (Math/log s))
                                (* (/ x s) (/ x s))))]
        (is (< (Math/abs (- expected (aget ^doubles (compiled x s) 0))) 1.0e-12))))))

(deftest shared-c-source-boundary-does-not-contain-a-record
  (let [source (:form (aot/fused-scalar-form #'density-buffer :double))]
    (is (not-any? #(and (seq? %) (= 'raster.sci.distributions/->Normal (first %)))
                  (tree-seq coll? seq source)))))
