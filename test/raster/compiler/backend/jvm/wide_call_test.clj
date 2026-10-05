(ns raster.compiler.backend.jvm.wide-call-test
  "Boxed dynamic calls must respect IFn's twenty-positional-argument boundary."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.pipeline :as pipeline]))

(defn positional [& args] (vec args))
(defn selected-function [f] f)

(defmacro define-call-boundaries []
  (cons 'do
        (for [arity [20 21 38]
              kind [:var :local :expression]
              :let [function-name (symbol (str "boxed-" (name kind) "-call-" arity))
                    signature (if (= kind :var) '[x :- Double] '[f :- Object x :- Double])
                    head (case kind
                           :var 'positional
                           :local 'f
                           :expression '(selected-function f))
                    arguments (cons 'x (range (dec arity)))]]
          (list 'deftm function-name signature ':- 'Object (cons head arguments)))))

(define-call-boundaries)

(defmacro define-typed-wide-target []
  (let [args (mapv #(symbol (str "a" %)) (range 37))]
    (list 'deftm 'typed-wide-target
          (into '[x :- Double] (mapcat #(vector % ':- 'Long) args))
          ':- 'Double (list 'clojure.core/+ 'x (list 'double (last args))))))

(define-typed-wide-target)

(deftest dynamic-calls-at-and-above-positional-boundary
  (doseq [arity [20 21 38] kind [:var :local :expression]
          :let [v (ns-resolve 'raster.compiler.backend.jvm.wide-call-test
                             (symbol (str "boxed-" (name kind) "-call-" arity)))
                compiled (pipeline/compile-aot v)
                expected (vec (cons 4.5 (range (dec arity))))]]
    (is (= expected (if (= kind :var) (compiled 4.5) (compiled positional 4.5)))
        (str kind " call with " arity " arguments"))))

(deftest wide-var-calls-observe-current-root
  (let [compiled (pipeline/compile-aot #'boxed-var-call-21)]
    (is (= (vec (cons 1.5 (range 20))) (compiled 1.5)))
    (with-redefs [positional (fn [& args] [:changed (count args) (first args)])]
      (is (= [:changed 21 2.5] (compiled 2.5))))
    (is (= (vec (cons 3.5 (range 20))) (compiled 3.5)))))

(deftest wide-calls-reach-typed-compiled-apply-to
  (let [target (pipeline/compile-aot #'typed-wide-target)
        caller (pipeline/compile-aot #'boxed-local-call-38)]
    (is (== 40.5 (apply target (cons 4.5 (range 37)))))
    (is (== 40.5 (caller target 4.5)))))

(deftest wide-argument-order-and-single-evaluation
  (let [compiled (pipeline/compile-aot #'boxed-expression-call-21)
        calls (atom [])
        target (fn [& args] (swap! calls conj args) (vec args))]
    (is (= (vec (cons 7.5 (range 20))) (compiled target 7.5)))
    (is (= [(vec (cons 7.5 (range 20)))] (mapv vec @calls)))))

(defn observed-function [events f] (swap! events conj :head) f)
(defn observed-argument [events index] (swap! events conj index) index)

(defmacro define-effectful-call []
  (list 'deftm 'effectful-call '[events :- Object f :- Object] ':- 'Object
        (cons '(observed-function events f)
              (for [index (range 21)] (list 'observed-argument 'events index)))))

(define-effectful-call)

(deftest wide-calls-preserve-head-and-argument-evaluation-order
  (let [compiled (pipeline/compile-aot #'effectful-call)
        events (atom [])
        target (fn [& args] (swap! events conj :call) (vec args))]
    (is (= (vec (range 21)) (compiled events target)))
    (is (= (vec (concat [:head] (range 21) [:call])) @events))))
