(ns raster.ad.completion-normalize-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse.normalize :as anf]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
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

(deftest lexical-value-declarations-reach-aliases-and-final-result
  (let [declared (with-meta 'value {:raster.type/tag 'double :tag 'Object
                                   :raster.effect/writes '#{not-a-type}})
        [flat result] (anf/normalize-for-ad [declared 'input 'alias 'value 'last-alias 'alias]
                                          ['last-alias] gensym)]
    (is (= ['double 'double 'double]
           (mapv #(-> % meta :raster.type/tag) (take-nth 2 flat))))
    (is (= ['double 'double]
           (mapv #(-> % meta :raster.type/tag) [(nth flat 3) (nth flat 5)])))
    (is (= 'double (:raster.type/tag (meta result))))
    (is (= 'Object (:tag (meta (first flat)))))
    (is (not-any? #(contains? (meta %) :tag) [(nth flat 2) (nth flat 3) result]))
    (is (not-any? #(contains? (meta %) :raster.effect/writes) [(nth flat 2) result]))
    (let [body (list 'let* flat result)
          interpreted (source-function '[input] body)
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.DeclarationTransport"
                    [{:name 'run :params '[input] :return-tag 'Object
                      :walked-body [body] :source-ns *ns*}])
          method (get-in artifact [:methods 'run])]
      (doseq [input [nil (Double. 1.25) (Object.) "not a number"]]
        (is (identical? input (interpreted input)))
        (is (identical? input (.invoke ^java.lang.reflect.Method method nil (object-array [input])))
            "a retained tangent declaration is not primitive carrier or presence evidence"))))
  (let [initializer (with-meta '(let* [] input) {:raster.type/tag 'float})
        [flat result] (anf/normalize-for-ad [] [initializer] gensym)]
    (is (= 'float (:raster.type/tag (meta (last (take-nth 2 flat))))))
    (is (= 'float (:raster.type/tag (meta result))))))

(deftest declarations-obey-initializer-scope-and-shadowing
  (let [declared (with-meta 'x {:raster.type/tag 'double})
        physical-only (with-meta 'x {:tag 'Object})
        flat (anf/anf-normalize-bindings [declared 'input physical-only 'x 'alias 'x] gensym)]
    (is (= 'double (:raster.type/tag (meta (nth flat 3))))
        "the second initializer still sees the preceding declaration")
    (is (nil? (:raster.type/tag (meta (nth flat 4)))))
    (is (nil? (:raster.type/tag (meta (nth flat 5))))))
  (let [declared (with-meta 'x {:raster.type/tag 'double})
        source (list 'fn* ['x] 'x)
        quoted '(quote x)
        flat (anf/anf-normalize-bindings [declared 'input 'f source 'data quoted] gensym)
        f (nth flat 3)]
    (is (= 'x (last f)))
    (is (nil? (:raster.type/tag (meta (last f)))))
    (is (identical? quoted (nth flat 5)))))

(deftest conflicting-lexical-declarations-require-a-refinement-owner
  (let [declared (with-meta 'x {:raster.type/tag 'double})
        conflicting (with-meta 'x {:raster.type/tag 'float})]
    (is (= :ad-conflicting-lexical-declaration
           (try (anf/anf-normalize-bindings [declared 'input 'use conflicting] gensym) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest declaration-transport-keeps-qualified-quotes-opaque
  (doseq [quoted ['(quote x) '(clojure.core/quote x)
                  '(clojure.core/quote (let* [x 1.0] x))]]
    (let [declared (with-meta 'x {:raster.type/tag 'double})
          flat (anf/anf-normalize-bindings [declared 'input 'data quoted] gensym)]
      (is (identical? quoted (nth flat 3)))
      (is (nil? (:raster.type/tag (meta (second (nth flat 3)))))))
    (let [source (list 'let* ['x 1.0] quoted)
          flat (anf/anf-normalize-bindings ['data source] gensym)]
      (is (identical? quoted (last flat))
          "flattening an initializer must not alpha-rename quoted data"))))

(deftest final-body-declaration-does-not-survive-untyped-shadow
  (let [[_ result] (anf/normalize-for-ad
                    [(with-meta 'x {:raster.type/tag 'double}) 'input
                     (with-meta 'x {:tag 'Object}) 'replacement]
                    ['x] gensym)]
    (is (= 'x result))
    (is (nil? (:raster.type/tag (meta result))))))

(deftest object-completion-retains-logical-declaration-alongside-physical-hint
  (let [declared (with-meta 'x {:raster.type/tag 'double :tag 'Object})
        snapshot (with-meta 'saved {:tag 'Object})
        expression (completion/complete-value completion/object-input-contract 'x)
        [flat result] (anf/normalize-for-ad [declared 'input snapshot expression 'alias 'saved]
                                          ['alias] gensym)
        body (list 'let* flat result)
        source (source-function '[input] body)
        artifact (bytecode/compile-specialized-class!
                   "raster.compiled.CompletedDeclarationTransport"
                   [{:name 'run :params '[input] :return-tag 'Object
                     :walked-body [body] :source-ns *ns*}])
        method (get-in artifact [:methods 'run])]
    (is (= {:tag 'Object :raster.type/tag 'double} (meta (nth flat 2))))
    (is (= 'double (:raster.type/tag (meta result))))
    (doseq [input [nil (Double. 2.5) (Object.) "not numeric"]]
      (is (identical? input (source input)))
      (is (identical? input (.invoke ^java.lang.reflect.Method method nil (object-array [input])))))))

(deftest object-completion-declaration-authority-is-exact
  (let [declared (with-meta 'x {:raster.type/tag 'double})
        completed (completion/complete-value completion/object-input-contract declared)]
    (is (= 'double (completion/object-value-declaration completed)))
    (is (nil? (completion/object-value-declaration
                (completion/complete-value completion/object-input-contract (with-meta 'x {:tag 'double})))))
    (is (nil? (completion/object-value-declaration
                (completion/complete-value (completion/primitive-input-contract Float/TYPE policy) declared))))
    (is (nil? (completion/object-value-declaration (concat completed [:extra]))))
    (is (= :conflicting-value-completion-declaration
           (try (completion/object-value-declaration (vary-meta completed assoc :raster.type/tag 'float)) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :ad-conflicting-lexical-declaration
           (try (anf/anf-normalize-bindings [(with-meta 'saved {:raster.type/tag 'float}) completed] gensym) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest initial-owner-declarations-flow-without-defeating-shadowing
  (let [[flat result] (anf/normalize-for-ad ['alias 'formal] ['alias] gensym {'formal 'double})]
    (is (= 'double (:raster.type/tag (meta (first flat)))))
    (is (= 'double (:raster.type/tag (meta (second flat)))))
    (is (= 'double (:raster.type/tag (meta result)))))
  (let [[flat result] (anf/normalize-for-ad [(with-meta 'formal {:tag 'Object}) 'formal 'alias 'formal]
                                          ['alias] gensym {'formal 'double})]
    (is (= 'double (:raster.type/tag (meta (second flat)))))
    (is (nil? (:raster.type/tag (meta result)))))
  (let [[flat _] (anf/normalize-for-ad ['f '(fn* [formal] formal)] ['f] gensym {'formal 'double})]
    (is (nil? (:raster.type/tag (meta (last (second flat)))))))
  (let [[_ result] (anf/normalize-for-ad [] ['formal] gensym {'formal 'double})]
    (is (= 'double (:raster.type/tag (meta result))))))

(deftest derivative-regions-import-only-visible-owner-declarations
  (let [declared (with-meta 'outer {:raster.type/tag 'double})
        context @#'rev/*source-memory-context*
        normalized (fn [visible]
                     (with-bindings {#'rev/*source-memory-context*
                                     (assoc context :values visible
                                            :value-facts {'outer {:value declared}})}
                       (:fwd-bindings (#'rev/let-gradient-pieces ['alias 'outer] ['alias] [] 'dy))))]
    (is (nil? (:raster.type/tag (meta (first (normalized []))))))
    (is (= 'double (:raster.type/tag (meta (first (normalized ['outer]))))))))

(deftest forward-presence-is-frozen-before-lexical-rebinding
  (let [facts (:value-facts (#'rev/forward-pass
                            ['original nil 'saved 'original 'original 1.0 'last 'saved] []))]
    (is (= :present (get-in facts ['original :presence])))
    (is (= :absent (get-in facts ['saved :presence])))
    (is (= :absent (get-in facts ['last :presence]))))
  (let [facts (:value-facts (#'rev/forward-pass ['p [nil] 'p 1.0] []))]
    (is (nil? (get-in facts ['p :product])))))

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

(deftest object-completion-distributes-before-normalizing-original-exits
  (let [expression (completion/complete-value completion/object-input-contract
                                              '(do (swap! events conj :operand) x))
        [bindings normalized] (anf/anf-normalize-expr expression gensym)
        f (source-function '[x events] (list 'let* bindings normalized))
        input (Object.) events (atom [])]
    (is (= 2 (count bindings)))
    (is (= (completion/complete-value completion/object-input-contract 'x) normalized))
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
