(ns raster.compiler.ir.pure-map-expansion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.par]
            [raster.compiler.ir.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.inline-call-plan :as call-plan]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.materialize :as materialize]
            [raster.compiler.passes.scalar.inline-test :as inline-test]
            [raster.compiler.passes.scalar.normal-return :as normal-return]
            [raster.compiler.passes.scalar.source-import :as source-import]))

(deftm counted-float-map [counter :- (Array long) n :- Long] :- (Array float)
  (raster.par/pmap #_{:clj-kondo/ignore [:unresolved-symbol]} i
    (do (clojure.core/aset counter 0 (clojure.core/inc (clojure.core/aget counter 0))) n)
    float 1.25))

(deftm counted-float-owner [counter :- (Array long) n :- Long] :- (Array float)
  (counted-float-map counter n))

(deftm typed-descendant-map [input :- (Array float) n :- Long] :- (Array float)
  (raster.par/pmap i (arrays/alength input) float (arrays/aget input i)))

(defn- source-function [source]
  (binding [*ns* (the-ns 'raster.compiler.ir.pure-map-expansion-test)]
    (eval (list 'fn ['counter 'n 'out] source))))

(defn- outcome [f counter n out]
  (try
    (let [result (f counter n out)]
      {:class (class result) :values (vec result)})
    (catch Throwable failure {:exception (class failure)})))

(deftest shared-map-recipe-preserves-count-storage-and-source-ordinal
  (doseq [[cast array-class value] [['float (class (float-array 0)) 1.25]
                                  ['double (class (double-array 0)) 1.25]
                                  ['clojure.core/float (class (float-array 0)) 1.25]
                                  ['clojure.core/unchecked-float (class (float-array 0)) 1.25]
                                  ['clojure.core/unchecked-int (class (int-array 0)) 1]
                                  ['long (class (long-array 0)) 1]
                                  [nil (class (double-array 0)) 1.25]]]
    (let [source (with-meta
                   (list 'raster.par/pmap 'i
                         '(do (aset counter 0 (inc (aget counter 0))) n)
                         cast '(if (instance? Long i) 1.25 9.0))
                   {:raster.type/elem-type :long})
          original (source-function source)
          lowered (source-function (:form (materialize/materialize-pass source nil)))]
      (doseq [f [original lowered]
              [n expected] [[2 {:class array-class :values [value value]}]
                            [0 {:class array-class :values []}]
                            [-1 {:exception NegativeArraySizeException}]
                            [2147483648 {:exception ArithmeticException}]
                            [nil {:exception NullPointerException}]]]
        (let [counter (long-array 1)]
          (is (= expected (outcome f counter n nil)))
          (is (= 1 (aget counter 0))))))))

(deftest materialization-does-not-capture-outer-result-or-force-quoted-data
  (let [source '(let* [out (raster.par/pmap i 2 double (aget out i))] out)
        lowered (:form (materialize/materialize-pass source nil))
        input (double-array [4.0 5.0])]
    (is (= [4.0 5.0] (vec ((source-function lowered) nil nil input))))
    (is (= [4.0 5.0] (vec input))))
  (let [forced (atom 0)
        quoted (list 'quote (lazy-seq (swap! forced inc) (list 'never)))
        source (list 'raster.par/pmap 'i 0 'double quoted)]
    (par/pure-map-expansion 'result source)
    (materialize/materialize-pass source nil)
    (is (zero? @forced)))
  (is (= :present (normal-return/presence '(raster.par/pmap i 0 float (unknown i))))))

(deftest shared-map-storage-does-not-add-new-array-families
  (doseq [cast ['byte 'clojure.core/byte 'boolean 'unknown-cast nil]]
    (is (= :double (:element-type
                    (par/pure-map-expansion 'result (list 'raster.par/pmap 'i 0 cast 0)))))))

(deftest result-binding-check-stays-after-the-completed-map
  (let [binder (with-meta 'out {:tag 'floats})
        source (list 'let*
                     [binder '(raster.par/pmap i
                               (do (aset counter 0 (inc (aget counter 0))) n)
                               double (do (aset counter 1 (inc (aget counter 1))) 1.25))]
                     binder)
        materialized (:form (materialize/materialize-pass source nil))]
    (doseq [form [source materialized]]
      (let [[_ f] (@#'inline-test/owned-inline-function
                   '[counter n] {:tags '[longs long] :return-tag 'Object
                                 :source-ns (the-ns 'raster.compiler.ir.pure-map-expansion-test)}
                   (fn [] {:body form}))]
        (doseq [n [0 2]]
          (let [counter (long-array 2)]
            (is (= {:exception ClassCastException}
                   (outcome (fn [c n _] (f c n)) counter n nil)))
            (is (= [1 n] (vec counter)))))))))

(defn- prepared-application []
  (let [helper (core/resolve-deftm-var #'counted-float-map)
        owner (core/resolve-deftm-var #'counted-float-owner)]
    (entry/call-with-definition-inputs #(core/prepare-owned-source-frontend! helper))
    (let [facts (pipeline/deftm-frontend-facts owner :double)
          body (pipeline/get-walked-body owner :double)
          candidate (if (= 1 (count body)) (first body) (list* 'do body))
          options {:source-var owner :source-ns (:source-ns facts)
                   :active-params (:parameters facts) :public-parameters (:parameters facts)}]
      (@#'inline-test/owned-inline-function
       '[counter n] {:tags '[longs long] :return-tag 'floats
                     :source-ns (the-ns 'raster.compiler.ir.pure-map-expansion-test)}
       (fn []
         (let [plans (call-plan/call-plans (call-plan/source-context options) candidate)
               occurrence (first (keys plans))
               application (source-import/instantiate-ready-reference-helper candidate occurrence options)]
           (when-not application (throw (ex-info "missing owned pmap application" {})))
           {:projection application :body (:expression application)}))))))

(deftest owned-pmap-import-consumes-the-same-count-and-storage-recipe
  (let [[_ imported] (prepared-application)]
    (doseq [f [counted-float-map imported]
            [n expected] [[2 {:class (class (float-array 0)) :values [1.25 1.25]}]
                          [0 {:class (class (float-array 0)) :values []}]
                          [-1 {:exception NegativeArraySizeException}]
                          [2147483648 {:exception ArithmeticException}]]]
      (let [counter (long-array 1)]
        (is (= expected (outcome (fn [c n _] (f c n)) counter n nil)))
        (is (= 1 (aget counter 0)))))))

(deftest public-aot-keeps-the-owned-map-count-and-storage-contract
  (doseq [inline? [false true]]
    (let [compiled (pipeline/compile-aot #'counted-float-owner :simd? false :inline? inline?)]
      (doseq [[n expected] [[2 {:class (class (float-array 0)) :values [1.25 1.25]}]
                            [0 {:class (class (float-array 0)) :values []}]
                            [-1 {:exception NegativeArraySizeException}]
                            [2147483648 {:exception ArithmeticException}]]]
        (let [counter (long-array 1)]
          (is (= expected (outcome (fn [c n _] (compiled c n)) counter n nil)))
          (is (= 1 (aget counter 0))))))))

(deftest pmap-expansion-query-requires-the-consumed-exact-module-occurrence
  (prepared-application)
  (entry/call-with-definition-inputs
   (fn []
     (let [frontend (entry/original-source-frontend (core/resolve-deftm-var #'counted-float-map))
           emission (entry/observed-frontend-emission frontend)
           original (first (vals (:pmap-catalog emission)))
           copy (with-meta (apply list original) (meta original))
           record (bytecode/retained-pmap-expansion emission original)]
       (is (some? record))
       (is (identical? original (get-in record [:recipe :source])))
       (is (nil? (bytecode/retained-pmap-expansion emission copy)))
       (is (nil? (bytecode/retained-pmap-expansion (dissoc emission :pmap-expansions) original)))
       (is (nil? (bytecode/retained-pmap-expansion
                  (update emission :pmap-expansions
                          #(into {} (map (fn [[token _]] [token {:status :ambiguous}]) %))) original)))
       (with-redefs [par/pure-map-expansion (fn [& _] (throw (AssertionError. "must not re-expand")))]
         (is (thrown? clojure.lang.ExceptionInfo (entry/observed-frontend-emission frontend))))
       (with-redefs [descriptor/cast-ops (assoc-in descriptor/cast-ops ['float :result-tag] 'double)]
         (is (thrown? clojure.lang.ExceptionInfo (entry/observed-frontend-emission frontend))))))))

(deftest pmap-spine-requires-actual-binding-allocation-and-both-return-aliases
  (prepared-application)
  (entry/call-with-definition-inputs
   (fn []
     (let [frontend (entry/original-source-frontend (core/resolve-deftm-var #'counted-float-map))
           emission (entry/observed-frontend-emission frontend)
           [token original] (first (:pmap-catalog emission))
           spine (bytecode/retained-pmap-spine emission original)
           expansion (:expansion spine)
           scope (:scope spine)]
       (is (some? spine))
       (is (= (class (float-array 0)) (:array-class spine)))
       (is (nil? (bytecode/retained-pmap-spine emission (with-meta (apply list original) (meta original)))))
       (doseq [changed [(dissoc emission :binding-catalog)
                        (dissoc emission :pmap-results)
                        (assoc-in emission [:pmap-expansions token :expanded]
                                  (with-meta (apply list scope) (meta scope)))
                        (assoc-in emission [:pmap-expansions token :recipe :count] 'foreign-count)
                        (assoc-in emission [:pmap-expansions token :status] :ambiguous)]]
         (is (nil? (bytecode/retained-pmap-spine changed original))))
       (is (some? (:map-effect expansion)))
       (doseq [[role owner] [[:map-return (:map-effect expansion)]
                             [:root-return token]]
               changed [(assoc-in emission [:pmap-results [owner role] :status] :ambiguous)
                        (update-in emission [:pmap-results [owner role] :local :slot] inc)
                        (assoc-in emission [:pmap-results [owner role] :local :reference-class]
                                  (class (double-array 0)))
                        (assoc-in emission [:pmap-results [owner role] :expression]
                                  (with-meta (symbol (name (get expansion role)))
                                    (meta (get expansion role))))]]
         (is (nil? (bytecode/retained-pmap-spine changed original))))))))

(deftest generated-map-wrappers-do-not-remint-original-typed-descendants
  (let [source (core/resolve-deftm-var #'typed-descendant-map)]
    (entry/call-with-definition-inputs #(core/prepare-owned-source-frontend! source))
    (entry/call-with-definition-inputs
     (fn []
       (let [frontend (entry/original-source-frontend source)
             emission (entry/observed-frontend-emission frontend)
             original (first (vals (:pmap-catalog emission)))
             bound (nth original 2)
             body (nth original 4)
             spine (bytecode/retained-pmap-spine emission original)]
         (is (some? spine))
         (doseq [[child query] [[bound bytecode/retained-invocation-context]
                               [body bytecode/retained-array-read]]]
           (let [observed (query emission child)]
             (is (= :known (:status observed)))
             (is (identical? observed (query emission child)))
             (is (nil? (query
                        emission (with-meta (apply list child) (meta child)))))))
         (is (identical? bound (nth (nth (second (:scope spine)) 1) 1)))
         (is (identical? body (nth (nth (:map (:expansion spine)) 5) 2))))))))
