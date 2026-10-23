(ns raster.ad.generated-entry-test
  "Physical contracts of the shared generated reverse-AD entry, without a device."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.ad.reverse :as reverse]))

(core/deftm captured-entry-loss [x :- Double] :- Double (raster.numeric/* x x))
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
                 (is (some #(= :prepared-reverse-ad (first %))
                           (:unclosed-source-imports @entry/*emission-inputs*))
                     "captured calls retain the same unclosed semantic import, not a fake helper frontend")))))
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

(deftest captured-gradient-remains-a-higher-order-ad-input
  (let [first-gradient (reverse/grad #'captured-entry-loss)
        second-gradient (reverse/grad first-gradient)]
    (is (= 6.0 (first-gradient 3.0)))
    (is (= 2.0 (second-gradient 3.0)))))

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
