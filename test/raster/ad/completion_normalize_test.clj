(ns raster.ad.completion-normalize-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse.normalize :as anf]
            [raster.compiler.core.value-completion :as completion]))

(def policy {:unchecked-math false})

(deftm quoted-scale [x :- Double] :- Double
  (let [data '(let* [inner 7.0] inner)
        scale (double (count data))]
    (n/* x scale)))

(deftest source-ad-preserves-quoted-binding-data
  (doseq [x [0.0 2.0 -3.0]]
    (is (= (* 3.0 x) (quoted-scale x)))
    (is (= [(* 3.0 x) 3.0] ((rev/value+grad #'quoted-scale) x)))
    (is (= [(* 3.0 x) 6.0] ((jvp/jvp #'quoted-scale) x 2.0)))))

(deftest nested-let-hoisting-preserves-opaque-regions
  (let [data '(let* [tick (swap! events conj :must-not-run)] tick)
        quoted (list 'quote data)
        normalized (#'rev/hoist-nested-lets quoted)
        events (atom [])]
    (is (identical? quoted normalized))
    (is (= data ((eval (list 'fn '[events] normalized)) events)))
    (is (= [] @events)))
  (doseq [contract [completion/object-input-contract
                    (completion/primitive-input-contract Float/TYPE policy)]]
    (let [expression (completion/complete-value contract '(let* [input x] input))]
      (is (identical? expression (#'rev/hoist-nested-lets expression))))))

(deftest hoisting-does-not-admit-malformed-reserved-completions
  (doseq [source [(list completion/completion-symbol {:source-context :unknown}
                       '(let* [input 1.0] input))
                  (list completion/completion-symbol completion/object-input-contract
                        '(let* [input 1.0] input) :extra)]]
    (let [normalized (rev/call-with-shared-ad-gensym
                       #(#'rev/hoist-nested-lets source))]
      (is (= :invalid-value-completion
             (try (completion/project-to-jvm-source normalized policy) :accepted
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(defn- source-function [parameters source]
  (binding [*unchecked-math* false]
    (eval (list 'fn parameters (completion/project-to-jvm-source source policy)))))

(deftest completion-preserves-original-primitive-result-context
  (let [contract (completion/primitive-input-contract Float/TYPE policy)
        original (completion/complete-value contract '(Math/abs (double x)))
        [bindings normalized] (anf/anf-normalize-expr original gensym)
        source (source-function '[x] original)
        prepared (source-function '[x] (list 'let* bindings normalized))
        boxed (source-function '[x] (completion/complete-value contract 'x))]
    (is (= [] bindings))
    (is (identical? original normalized))
    (is (= Float/POSITIVE_INFINITY (source Double/MAX_VALUE) (prepared Double/MAX_VALUE)))
    (is (thrown? IllegalArgumentException (boxed Double/MAX_VALUE))
        "boxing the same mathematical value changes checked float input conversion")))

(deftest completed-first-input-still-fails-before-later-operand-effects
  (let [first-input (completion/complete-value
                     (completion/primitive-input-contract Double/TYPE policy) 'x)
        expression (list 'vector first-input '(do (swap! events conj :later) 2.0))
        [bindings result] (anf/normalize-for-ad [] [expression] gensym)
        prepared (source-function '[x events] (list 'let* bindings result))]
    (let [events (atom [])]
      (is (thrown? NullPointerException (prepared nil events)))
      (is (= [] @events)))
    (let [events (atom [])]
      (is (= [3.0 2.0] (prepared 3.0 events)))
      (is (= [:later] @events)))))

(deftest object-completion-does-not-lift-a-compound-operand
  (let [expression (completion/complete-value completion/object-input-contract
                                              '(do (swap! events conj :operand) x))
        [bindings normalized] (anf/anf-normalize-expr expression gensym)
        f (source-function '[x events] (list 'let* bindings normalized))
        input (Object.) events (atom [])]
    (is (= [] bindings))
    (is (identical? expression normalized))
    (is (identical? input (f input events)))
    (is (= [:operand] @events))))

(deftest lexical-initializers-flatten-at-their-original-binding-position
  (let [bindings '[outer 10
                   before (swap! events conj :before)
                   result (let* [outer (do (swap! events conj :initializer) 20)]
                            (swap! events conj :statement)
                            (let* [inner (do (swap! events conj :tail) outer)] inner))
                   after (swap! events conj :after)]
        [flat value] (anf/normalize-for-ad bindings ['(vector outer result)] gensym)
        source (source-function '[events] (list 'let* bindings '(vector outer result)))
        prepared (source-function '[events] (list 'let* flat value))
        a (atom []) b (atom [])]
    (is (= [10 20] (source a) (prepared b)))
    (is (= [:before :initializer :statement :tail :after] @a @b))
    (is (not-any? #(and (seq? %) (contains? #{'let 'let*} (first %)))
                  (map second (partition 2 flat)))
        "binding initializers are flat without moving a call operand's conversion")))

(deftest initializer-flattening-does-not-pass-an-earlier-completed-input
  (let [input (completion/complete-value
               (completion/primitive-input-contract Double/TYPE policy) 'x)
        expression (list 'vector input
                         '(let* [tick (swap! events conj :later)] 2.0))
        source (source-function '[x events] expression)]
    (doseq [normalize [identity (fn [form]
                                 (rev/call-with-shared-ad-gensym
                                   #(#'rev/hoist-nested-lets form)))]
            :let [[bindings value] (anf/normalize-for-ad [] [(normalize expression)] gensym)
                  prepared (source-function '[x events] (list 'let* bindings value))]
            f [source prepared]]
      (let [events (atom [])]
        (is (thrown? NullPointerException (f nil events)))
        (is (= [] @events)))
      (let [events (atom [])]
        (is (= [3.0 2.0] (f 3.0 events)))
        (is (= [:later] @events))))))

(deftest lexical-flattening-keeps-quoted-binders-opaque
  (let [data '(let* [tick (swap! events conj :must-not-run)] tick)
        bindings (vector 'result (list 'let* [] (list 'quote data)))
        [flat value] (anf/normalize-for-ad bindings ['result] gensym)
        prepared (source-function '[events] (list 'let* flat value))
        events (atom [])]
    (is (= data (prepared events)))
    (is (= [] @events))))

(deftest lexical-initializer-result-declarations-survive-without-becoming-casts
  (doseq [tag ['float 'double]]
    (let [expression (with-meta '(let* [] input)
                       {:raster.type/tag tag :raster.effect/writes '#{not-a-value-type}})
          flat (anf/anf-normalize-bindings ['out expression] gensym)
          out (first (filter #(= 'out %) (take-nth 2 flat)))
          fixed (with-meta 'out {:raster.type/tag 'double})
          explicit (anf/anf-normalize-bindings [fixed expression] gensym)]
      (is (= tag (:raster.type/tag (meta out))))
      (is (not (contains? (meta out) :raster.effect/writes)))
      (is (= 'double (:raster.type/tag (meta (last (take-nth 2 explicit))))))
      (let [input (Object.)]
        (is (identical? input ((source-function '[input] (list 'let* flat 'out)) input)))))))
