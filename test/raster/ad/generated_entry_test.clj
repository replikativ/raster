(ns raster.ad.generated-entry-test
  "Physical contracts of the shared generated reverse-AD entry, without a device."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.types :as types]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.ad.reverse :as reverse]))

(core/deftm captured-entry-loss [x :- Double] :- Double (raster.numeric/* x x))
(core/deftm captured-cubic-loss [x :- Double] :- Double
  (raster.numeric/* x (raster.numeric/* x x)))
(core/deftm unused-entry-input-loss [x :- Double unused :- Long] :- Double
  (raster.numeric/* x x))
(core/deftm nil-inner-entry-gradient [x :- Double] :- Double
  (let [values ((reverse/value+grad #'unused-entry-input-loss :wrt [0]) x nil)]
    (nth values 1)))
(core/deftm wrong-inner-entry-gradient [x :- Double] :- Double
  (let [values ((reverse/value+grad #'unused-entry-input-loss :wrt [0]) x "not a Number")]
    (nth values 1)))
(core/deftm float-inner-entry-gradient [x :- Float] :- Double
  (let [values ((reverse/value+grad #'captured-cubic-loss) x)]
    (nth values 1)))
(core/deftm checked-read-loss [x :- Double data :- (Array double)] :- Double
  (let [unused (aget data 9)] (raster.numeric/* x x)))

;; Define both source functions before retaining any compiler observations.
;; Alternating wide JVM slots exercise the real 20/21-argument entry boundary.
(doseq [arity [20 21]]
  (let [parameters (mapv #(symbol (str "value" %)) (range arity))
        tags (take arity (cycle '[Double Long]))
        declaration (vec (mapcat (fn [parameter tag] [parameter :- tag]) parameters tags))]
    (eval (list 'raster.core/deftm (symbol (str "wide-loss-" arity)) declaration :- 'Double
                (list 'raster.numeric/* (first parameters) (first parameters))))))

(deftest wide-prepared-runtime-uses-the-owned-entry-and-positional-gradient-slots
  (doseq [arity [20 21]]
    (let [function (ns-resolve 'raster.ad.generated-entry-test (symbol (str "wide-loss-" arity)))
          prepared (reverse/prepare-value+grad function [0])
          physical (reverse/prepared-bytecode-entry prepared)
          execute ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared)
          arguments (mapv #(if (even? %) 3.0 (long %)) (range arity))
          expected (into [9.0 6.0] (repeat (dec arity) nil))]
      (is (identical? physical (reverse/prepared-bytecode-entry prepared)))
      (is (= arity (count (:params prepared))))
      (is (= expected (apply (:implementation physical) arguments)))
      (is (= expected (apply execute arguments)))
      (is (= (into [6.25 -5.0] (repeat (dec arity) nil))
             (apply execute (assoc arguments 0 -2.5))))
      (doseq [index [0 (dec arity)]
              [value failure] [[nil NullPointerException] ["not a number" ClassCastException]]]
        (is (= failure (try (apply execute (assoc arguments index value)) nil
                            (catch Throwable error (class error))))))
      (doseq [invalid [(pop arguments) (conj arguments 1.0)]]
        (is (thrown? clojure.lang.ArityException (apply execute invalid)))))))

(deftest captured-ad-callables-retain-their-actual-preparation
  (let [value-gradient (reverse/value+grad #'captured-entry-loss)
        gradient (reverse/grad #'captured-entry-loss)]
    (doseq [[function mode expected] [[value-gradient :value+grad [9.0 6.0]]
                                    [gradient :grad 6.0]]]
      (let [metadata (meta function)
            {:keys [prepared]} (reverse/captured-prepared-callable function metadata)
            scope (create-ns (gensym "raster.captured-gradient."))]
        (try
          (is (= expected (function 3.0)))
          (is (= mode (:mode (reverse/captured-prepared-callable function metadata))))
          (is (identical? prepared (:raster.ad.reverse/prepared-program metadata)))
          (intern scope 'captured function)
          (entry/call-with-definition-inputs
           (fn []
             (binding [*ns* scope]
               (let [descriptor (#'inline/try-resolve-deftm
                                 (symbol (str (ns-name scope)) "captured") '[double])
                     expansion (#'inline/build-inline-expansion descriptor '[argument]
                                {'argument 'double} inline/subst-syms 'answer
                                (list (symbol (str (ns-name scope)) "captured") 'argument))]
                 (is (identical? prepared (:prepared-program descriptor)))
                 (is (= mode (:projection-mode descriptor)))
                 (is (seq (:bindings expansion)))
                 (let [owner (:raster.ad.reverse/generated-program-owner prepared)]
                   (is (satisfies? entry/RetainedProducerInput owner))
                   (is (some #(identical? owner %) (:producer-inputs @entry/*emission-inputs*))
                       "captured CPU calls retain their actual existing producer")
                   (is (identical? owner (entry/assert-producer-current! owner))))))))
          (doseq [changed [(with-meta (fn [& _] -99.0) metadata)
                           (with-meta function (assoc metadata :raster.ad.reverse/projection-mode
                                                    (if (= mode :grad) :value+grad :grad)))]]
            (is (= :ad-prepared-callable-mismatch
                   (try (reverse/captured-prepared-callable changed (meta changed)) nil
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
            (intern scope 'copied changed)
            (is (= :ad-prepared-callable-mismatch
                   (try (binding [*ns* scope]
                          (#'inline/try-resolve-deftm (symbol (str (ns-name scope)) "copied") '[double]))
                        nil
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
          (finally (remove-ns (ns-name scope))))))
    (let [revision ((requiring-resolve 'raster.ad.templates/registry-revision))]
      (with-redefs-fn {(requiring-resolve 'raster.ad.templates/registry-revision)
                      (constantly (inc revision))}
        (fn []
          (doseq [function [value-gradient gradient]]
            (is (= :ad-prepared-program-stale-rules
                   (try (function 3.0) nil
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))))

(deftest retained-producer-core-is-bound-to-the-existing-adapter-owner
  (let [prepared (reverse/prepare-value+grad #'captured-entry-loss [0])
        core (:raster.ad.reverse/producer-core prepared)
        input-key :raster.compiler.backend.jvm.generated-body/normalization-input
        core-key :raster.ad.reverse/producer-core
        physical (reverse/prepared-bytecode-entry prepared)
        copy (into {} core)
        execute (#'reverse/compile-prepared-value+grad prepared)]
    (is (map? core))
    (is (if (contains? prepared input-key)
          (identical? core (get prepared input-key))
          (identical? (:walked-body core) (:walked-body prepared)))
        "a real adaptation retains its input; abstention retains the unchanged core body")
    (is (identical? (:walked-body prepared) (:input-body physical)))
    (is (= [9.0 6.0] (execute 3.0) ((:implementation physical) 3.0)))
    (doseq [changed [(assoc prepared core-key copy)
                     (assoc prepared input-key copy)
                     (assoc prepared core-key copy input-key copy)
                     (update-in prepared [core-key :walked-body] #(vec (map identity %)))
                     (dissoc prepared core-key)]]
      (is (= :ad-prepared-program-mismatch
             (try (reverse/assert-prepared-program-source! changed) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    ;; Check the constructor relation itself, independently of the enclosing
    ;; frozen-map check. A copied input cannot be paired with the retained core.
    (doseq [changed (cond-> [(assoc prepared input-key copy)
                            (update prepared :tags #(vec (map identity %)))]
                     (contains? prepared input-key) (conj (dissoc prepared core-key)))]
      (is (= :ad-generated-producer-core-mismatch
             (try (#'reverse/assert-generated-producer-core! changed) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
  ;; If normalization abstains, retain the unchanged clean core;
  ;; it does not acquire a fictitious normalization edge or change execution.
  (let [prepared (with-redefs [generated/normalize-program (fn [& _] nil)]
                   (reverse/prepare-value+grad #'captured-entry-loss [0]))
        core (:raster.ad.reverse/producer-core prepared)]
    (is (map? core))
    (is (not (contains? prepared :raster.compiler.backend.jvm.generated-body/normalization-input)))
    (is (identical? (:walked-body core) (:walked-body prepared)))
    (is (identical? core (#'reverse/assert-generated-producer-core! prepared)))
    (is (= [9.0 6.0] ((#'reverse/compile-prepared-value+grad prepared) 3.0))))
  ;; Optional whole-program normalization may also decline before JVM helper
  ;; adaptation. The actual raw padded producer still exists at its constructor;
  ;; keeping that body is not permission to substitute an adapted historical body.
  (with-redefs-fn {(ns-resolve 'raster.ad.reverse 'normalize-generated-preparation)
                  (fn [& _] nil)}
    (fn []
      (let [prepared (reverse/prepare-value+grad #'captured-cubic-loss nil)
            core (:raster.ad.reverse/producer-core prepared)
            first-gradient (reverse/grad #'captured-cubic-loss)
            second-gradient (reverse/grad first-gradient)
            third-gradient (reverse/grad second-gradient)]
        (is (map? core))
        (is (identical? (:walked-body core) (:walked-body prepared)))
        (is (identical? core (#'reverse/assert-generated-producer-core! prepared)))
        (is (not (contains? prepared :raster.compiler.backend.jvm.generated-body/normalization-input)))
        (doseq [x [-2.0 0.0 3.0]]
          (is (= (* 6.0 x) (second-gradient x)))
          (is (= 6.0 (third-gradient x))))
        (is (thrown? NullPointerException (third-gradient nil)))
        (is (thrown? ClassCastException (third-gradient "not a Number")))))))

(deftest captured-gradient-remains-a-higher-order-ad-input
  (let [first-gradient (reverse/grad #'captured-entry-loss)
        second-gradient (reverse/grad first-gradient)]
    (is (= 6.0 (first-gradient 3.0)))
    (is (= 2.0 (second-gradient 3.0)))))

(deftest captured-numerical-producers-compose-through-third-order
  (let [first-gradient (reverse/grad #'captured-cubic-loss)
        second-gradient (reverse/grad first-gradient)
        third-gradient (reverse/grad second-gradient)]
    (doseq [x [-2.0 0.0 3.0]]
      (is (= (* 3.0 x x) (first-gradient x)))
      (is (= (* 6.0 x) (second-gradient x)))
      (is (= 6.0 (third-gradient x))))
    (doseq [function [first-gradient second-gradient third-gradient]
            [argument failure] [[nil NullPointerException] ["not a Number" ClassCastException]]]
      (is (= failure (try (function argument) nil (catch Throwable error (class error))))))
    (let [revision ((requiring-resolve 'raster.ad.templates/registry-revision))]
      (with-redefs-fn {(requiring-resolve 'raster.ad.templates/registry-revision)
                      (constantly (inc revision))}
        #(is (= :ad-prepared-program-stale-rules
                (try (reverse/grad second-gradient) nil
                     (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest duplicate-parent-imports-retain-distinct-construction-history
  (let [parent (reverse/grad #'captured-cubic-loss)
        scope (create-ns (gensym "raster.duplicate-parent-gradient."))]
    (try
      (intern scope 'parent parent)
      (binding [*ns* scope]
        (clojure.core/refer 'clojure.core)
        (eval '(raster.core/deftm twice [x :- Double] :- Double
                 (raster.numeric/+ (parent x) (parent x)))))
      (let [source (ns-resolve scope 'twice)
            prepared (reverse/prepare-value+grad source nil)
            history-key :raster.ad.reverse/entry-constructions
            history (get prepared history-key)
            execute (#'reverse/compile-prepared-value+grad prepared)
            gradient (reverse/grad source)
            second-gradient (reverse/grad gradient)
            other-parent (reverse/prepare-value+grad #'captured-entry-loss nil)
            other-bridge (:bridge (generated/physical-entry-view
                                   (reverse/prepared-bytecode-entry other-parent)
                                   (:params other-parent)))]
        (is (vector? history))
        (is (<= 2 (count history)) "both actual imports must survive preparation")
        (let [terms (vec (mapcat (fn [construction]
                                  (mapcat (juxt :object-contract :contract) (:slots construction)))
                                history))]
          (is (every? true? (for [left (range (count terms)) right (range left)]
                             (not (identical? (nth terms left) (nth terms right)))))
              "equal contract data does not conflate independently constructed imports"))
        (doseq [x [-2.0 0.0 3.0]]
          (is (= [(* 6.0 x x) (* 12.0 x)] (execute x)))
          (is (= (* 12.0 x) (gradient x)))
          (is (= 12.0 (second-gradient x))))
        (doseq [changed [(assoc prepared history-key (into [] history))
                         (assoc prepared history-key (conj history :forged-construction))
                         (assoc prepared history-key
                                (update-in history [0 :slots 0 :object-contract] #(into {} %)))
                         (assoc prepared history-key
                                (update-in history [0 :slots 0 :contract] #(into {} %)))
                         (assoc prepared history-key
                                (assoc-in history [0 :slots 0 :parameter-class] Float/TYPE))
                         (assoc prepared history-key (assoc-in history [0 :prepared] other-parent))
                         (assoc prepared history-key
                                (assoc-in history [0 :boxed-entry-bridge] other-bridge))
                         (dissoc prepared history-key)]]
          (is (= :ad-prepared-program-mismatch
                 (try (reverse/assert-prepared-program-source! changed) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest higher-order-numerical-projection-keeps-unused-entry-checks
  (doseq [[source failure] [[#'nil-inner-entry-gradient NullPointerException]
                            [#'wrong-inner-entry-gradient ClassCastException]]]
    (let [derivative (reverse/grad source)]
      (is (= failure (try (derivative 3.0) nil (catch Throwable error (class error)))))))
  (let [prepared (reverse/prepare-value+grad #'unused-entry-input-loss [0])
        policy (:source-policy (reverse/prepared-bytecode-entry prepared))]
    (doseq [[left right failure] [[nil 2 NullPointerException]
                                  [3.0 "not a Number" ClassCastException]]]
      (let [execute
            (entry/call-with-definition-inputs
             (fn []
               (let [arguments [(list 'do '(swap! events conj :left) left)
                                (list 'do '(swap! events conj :right) right)]
                     {:keys [bindings result-form]}
                     (reverse/project-prepared-value+grad-call prepared arguments :value+grad {})
                     expression (list 'let* (vec (mapcat identity bindings)) result-form)
                     interface (types/ensure-fn-interface! '[Object])]
                 (binding [*unchecked-math* (:unchecked-math policy)]
                   (:implementation
                    (core/emit-owned-bytecode!
                     bytecode/compile-typed-impl! "raster.compiled.EntryArgumentOrderTest"
                     '[events] '[Object] 'Object [expression]
                     (the-ns 'raster.ad.generated-entry-test) (:iface-name interface)
                     'entry-argument-order @entry/*emission-inputs*))))))
            events (atom [])]
        (is (= failure (try (execute events) nil (catch Throwable error (class error)))))
        (is (= [:left :right] @events)
            "all original Object operands precede the first entry conversion")))))

(deftest higher-order-entry-transport-retains-the-source-tangent-width
  (let [gradient (reverse/grad #'float-inner-entry-gradient)]
    (doseq [x [(float -2.0) (float 0.0) (float 1.25)]]
      (let [actual (gradient x)]
        (is (= Float (class actual)) "the Double callee does not retag its Float caller's tangent")
        (is (= (Float/floatToRawIntBits (float (* 6.0 (double x))))
               (Float/floatToRawIntBits (float actual))))))))

(deftest canonical-boxed-input-syntax-does-not-authorize-a-transpose
  (let [parameter (with-meta 'input {:raster.type/tag 'double})
        expression (list 'let*
                         ['value (completion/complete-value
                                  (completion/bytecode-boxed-parameter-contract Double/TYPE)
                                  parameter)]
                         '(raster.numeric/* value value))]
    (is (= :ad-unclosed-completion-transpose
           (try (reverse/grad-expr expression [parameter]) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "a canonical contract and declared tangent dtype are not an actual entry constructor")))

(deftest independently-compiled-ad-does-not-publish-another-entry-as-its-source
  ;; Model the documented pipeline distinction with an observable checked read:
  ;; this optimized artifact has discarded it; the generated entry has not.
  (let [optimized (fn [x _] [(* x x) (* 2.0 x) nil])
        function (with-redefs-fn
                   {(ns-resolve 'raster.ad.reverse 'compiled-value+grad-fn)
                    (fn [& _] {:fn optimized})}
                   #(reverse/value+grad #'checked-read-loss :wrt [0] :compile? true))
        ordinary (reverse/value+grad #'checked-read-loss :wrt [0])]
    (is (= [9.0 6.0 nil] (function 3.0 (double-array 0))))
    (is (thrown? ArrayIndexOutOfBoundsException (ordinary 3.0 (double-array 0))))
    (is (nil? (reverse/captured-prepared-callable function (meta function))))
    (is (nil? (:raster.core/deftm-walked-body (meta function))))
    (is (true? (:no-inline (meta function))))))
