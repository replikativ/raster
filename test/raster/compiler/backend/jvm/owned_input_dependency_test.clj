(ns raster.compiler.backend.jvm.owned-input-dependency-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm identity-double [x :- Double] :- Double x)
(deftm identity-caller [x :- Double] :- Double (identity-double x))

(defonce global-calls (atom 0))
(defn local-value [_] (swap! global-calls inc) 100)

(defn- typed-call [operand]
  (let [source (core/resolve-deftm-var #'identity-caller)
        body (core/ensure-walked-body! source)
        call (first (filter #(and (seq? %) (= '.invk (first %)))
                            (tree-seq coll? seq body)))]
    (assert call)
    (with-meta (list (first call) (second call) operand) (meta call))))

(defn- original-module [family & [include-sibling?]]
  (let [[params tags return-tag body sibling-tag]
        (case family
          :invocation ['[x] '[double] 'double [(typed-call '(local-value x))] 'double]
          :index ['[values x] '[doubles long] 'double
                  '[(clojure.core/aget values (local-value x))] 'long]
          :allocation ['[x] '[long] 'doubles '[(double-array (local-value x))] 'long]
          :reduction ['[x] '[long] 'double
                      '[(raster.par/reduce sum 0.0 i (local-value x) (+ sum 1.0))] 'long]
          :terminal ['[x] '[double] 'double '[(let* [copy x] (local-value copy))] 'double]
          :terminal-control ['[x] '[double] 'double '[(let* [copy x] copy)] 'double])
        source-ns (the-ns 'raster.compiler.backend.jvm.owned-input-dependency-test)
        emitter
        (fn [class-name _params _tags _return body source-ns & _]
          (let [artifact
                (bytecode/compile-specialized-class!
                 class-name
                 (cond-> [{:name 'run :params (mapv #(with-meta %1 {:tag %2}) params tags)
                           :return-tag return-tag :walked-body body :source-ns source-ns}]
                   (not= false include-sibling?)
                   (conj {:name 'local-value :params [(with-meta 'x {:tag sibling-tag})]
                          :return-tag sibling-tag :walked-body '[(+ x 1)] :source-ns source-ns})))
                method (get-in artifact [:methods 'run])]
            (fn [& arguments]
              (try (.invoke ^java.lang.reflect.Method method nil (object-array arguments))
                   (catch java.lang.reflect.InvocationTargetException e (throw (.getCause e)))))))]
    (#'core/emit-owned-bytecode!
     emitter "raster.compiled.OriginalSiblingInput" params tags return-tag body
     source-ns nil 'original-sibling-input)))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest every-owned-input-family-refuses-unretained-original-siblings
  (doseq [family [:invocation :index :allocation :reduction :terminal]]
    (let [{:keys [implementation emission]} (original-module family)
          before @global-calls
          source (first (:body emission))
          [arguments expected observe outline flag-path]
          (case family
            :invocation [[-2.0] -1.0 bytecode/retained-invocation-context
                         #(bytecode/compile-invocation-input-region %1 %2 0)
                         [:inputs 0 :unretained-sibling-calls?]]
            :index [[(double-array [3.0 5.0 7.0]) 1] 7.0 bytecode/retained-array-read
                    bytecode/compile-array-index-region [:input :unretained-sibling-calls?]]
            :allocation [[2] 3 bytecode/retained-zero-array-allocation
                         bytecode/compile-array-count-region [:count :unretained-sibling-calls?]]
            :reduction [[2] 3.0 bytecode/retained-reduction-emission
                        bytecode/compile-reduction-count-region [:bound :unretained-sibling-calls?]]
            :terminal [[-2.0] -1.0 bytecode/retained-terminal-context
                       bytecode/compile-terminal-result-region [:unretained-sibling-calls?]])
          actual (apply implementation arguments)]
      (is (= expected (if (= :allocation family) (alength ^doubles actual) actual)))
      (is (= before @global-calls) "the original class selects its own sibling")
      (is (true? (get-in (observe emission source) flag-path)))
      (is (= :unsupported-bytecode-input-callees (reason #(outline emission source))))
      (is (= before @global-calls) "refusal cannot execute the unrelated global"))))

(deftest terminal-with-no-sibling-dependency-still-outlines
  (let [{:keys [implementation emission]} (original-module :terminal-control)
        scope (first (:body emission))
        observed (bytecode/retained-terminal-context emission scope)
        region (bytecode/compile-terminal-result-region emission scope)]
    (is (false? (:unretained-sibling-calls? observed)))
    (doseq [value [-2.0 -0.0 Double/POSITIVE_INFINITY Double/NaN]]
      (is (= (Double/doubleToRawLongBits (implementation value))
             (Double/doubleToRawLongBits ((:invoke region) value)))))))

(deftest ordinary-global-calls-still-outline-without-rebinding
  (let [{:keys [implementation emission]} (original-module :invocation false)
        source (first (:body emission))
        observed (bytecode/retained-invocation-context emission source)
        before @global-calls
        region (bytecode/compile-invocation-input-region emission source 0)]
    (is (false? (get-in observed [:inputs 0 :unretained-sibling-calls?])))
    (is (= before @global-calls) "outlining itself does not call the global")
    (is (= 100.0 (implementation -2.0) ((:invoke region) -2.0)))
    (is (= (+ before 2) @global-calls))))

(deftest dependency-denial-follows-the-original-emitter-lookup-keys
  (let [candidate? #'bytecode/method-local-callee-candidate?
        typed {:typed-sibling-methods {"local-value" :method}}
        ordinary {:sibling-methods {"local-value" :method}}]
    (is (true? (candidate? '(other/local-value x) typed)))
    (is (false? (candidate? '(other/local-value x) ordinary)))
    (is (true? (candidate? '(local-value x) ordinary)))
    (is (true? (candidate? '(fn* [] (local-value x)) typed)))
    (is (false? (candidate? '(quote (local-value x)) typed)))
    (is (false? (candidate? '(clojure.core/quote (local-value x)) typed)))))

(deftest legacy-input-records-cannot-acquire-method-dependency-closure
  (let [{:keys [emission]} (original-module :invocation)
        source (first (:body emission))
        token (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta source))]
    (doseq [value [nil :unknown]]
      (is (= :unowned-bytecode-input-dependencies
             (reason #(bytecode/compile-invocation-input-region
                       (assoc-in emission [:invocation-contexts token :inputs 0
                                           :unretained-sibling-calls?] value)
                       source 0)))))
    (is (= :unowned-bytecode-input-dependencies
           (reason #(bytecode/compile-invocation-input-region
                     (update-in emission [:invocation-contexts token :inputs 0]
                                dissoc :unretained-sibling-calls?) source 0))))))
