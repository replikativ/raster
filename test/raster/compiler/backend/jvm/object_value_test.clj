(ns raster.compiler.backend.jvm.object-value-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.core :refer [deftm]]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion])
  (:import [java.lang.reflect InvocationTargetException Proxy InvocationHandler]))

(deftm object-boundary-helper [x :- Double] :- Double x)

(defn- compile-value [params expression]
  (let [{:keys [methods] :as artifact}
        (bytecode/compile-specialized-class!
         "raster.compiled.ObjectValueContext"
         [{:name 'run :params params :return-tag 'Object
           :walked-body [expression] :source-ns *ns*}])
        method (get methods 'run)]
    (assoc artifact :run
           (fn [& arguments]
             (try (.invoke ^java.lang.reflect.Method method nil (object-array arguments))
                  (catch InvocationTargetException failure (throw (.getCause failure))))))))

(deftest product-fields-and-identity-observers-share-original-literal-references
  (doseq [value [(Long/valueOf 1000) (Double/valueOf 5.125) (float 1.25) \a]]
    (let [expression (list 'let* ['pair [value '(do nil)]]
                           (list 'clojure.core/identical? '(clojure.core/nth pair 0) value))
          source ((eval (list 'fn* [] expression)))
          execute (:run (compile-value [] expression))]
      (is (true? source))
      (is (= source (execute))))))

(deftest literal-pooling-uses-identity-not-numeric-equality
  (let [left (Double. 5.125) right (Double. 5.125)
        execute (:run (compile-value [] [left right left]))
        result (execute)]
    (is (not (identical? left right)))
    (is (identical? left (nth result 0)))
    (is (identical? right (nth result 1)))
    (is (identical? (nth result 0) (nth result 2)))
    (is (not (identical? (nth result 0) (nth result 1))))))

(deftest java-object-inputs-preserve-the-original-reference
  (let [token (Double. 5.125)
        accept (:run (compile-value [] (list 'java.util.Objects/requireNonNull token)))
        construct (:run (compile-value [] (list 'new 'java.util.concurrent.atomic.AtomicReference token)))
        receiver (with-meta 'receiver {:tag 'java.util.concurrent.atomic.AtomicReference})
        store (:run (compile-value [receiver] (list 'do (list '. receiver 'set token) nil)))
        cell (java.util.concurrent.atomic.AtomicReference.)]
    (is (identical? token (accept)))
    (is (identical? token (.get ^java.util.concurrent.atomic.AtomicReference (construct))))
    (store cell)
    (is (identical? token (.get cell)))))

(deftest source-deftm-object-results-use-the-original-callable
  (let [argument (list 'do '(swap! trace conj :argument) 2.0)
        expression (list 'raster.compiler.backend.jvm.object-value-test/object-boundary-helper argument)
        execute (:run (compile-value '[trace] expression))
        token (Double. 5.125)]
    ;; The real deftm stub remains callable before/after specialization.
    (is (= 2.0 (execute (atom []))))
    (is (= 2.0 (execute (atom []))))
    (with-redefs [object-boundary-helper (fn [x] (is (= 2.0 x)) token)]
      (let [trace (atom [])]
        (is (identical? token (execute trace)))
        (is (= [:argument] @trace))))
    (let [local (symbol "raster.compiler.backend.jvm.object-value-test/object-boundary-helper")
          shadow (:run (compile-value [local] (list local 2.0)))]
      (is (identical? token (shadow (fn [_] token)))))))

(deftest object-storage-consumes-original-references
  (let [token (Double. 5.125)
        construct (:run (compile-value [] (list 'clojure.core/object-array [token])))
        array (with-meta 'array {:tag 'objects})
        write-array (:run (compile-value [array] (list 'clojure.core/aset array 0 token)))
        box (with-meta 'box {:tag 'clojure.lang.Box})
        write-field (:run (compile-value [box] (list 'do (list 'set! (list '. box 'val) token) nil)))
        storage (object-array 1) cell (clojure.lang.Box. nil)]
    (is (identical? token (aget ^objects (construct) 0)))
    (is (identical? token (write-array storage)))
    (is (identical? token (aget storage 0)))
    (write-field cell)
    (is (identical? token (.val cell)))))

(deftest object-return-literals-have-owned-reference-lowering
  (doseq [value [(int 1000) (short 3) (byte 2) 1/3
                 (bigint "1208925819614629174706176") (bigdec "1.25")]]
    (let [execute (:run (compile-value [] value))]
      (is (identical? value (execute)))
      (is (= (class value) (class (execute)))))))

(deftest quoted-data-and-exception-results-retain-owned-references
  (let [token (Double. 5.125)
        data [token {:value token}]
        quoted (:run (compile-value [] (list 'quote data)))
        selected (:run (compile-value '[fail]
                                      (list 'try
                                            (list 'if 'fail '(throw (new IllegalStateException)) token)
                                            (list 'catch 'IllegalStateException 'failure token))))]
    (is (identical? data (quoted)))
    (is (identical? token (selected false)))
    (is (identical? token (selected true)))))

(deftest preparation-treats-quoted-programs-as-data
  (doseq [head '[quote clojure.core/quote]]
    (let [data '(clojure.core/dotimes [i 3] (raster.par/pmap i 3 i))
          expression (list head data)
          materialize (requiring-resolve 'raster.compiler.passes.parallel.materialize/materialize-pass)]
      (doseq [prepare [bytecode/desugar-invk
                       (ns-resolve 'raster.compiler.backend.jvm.bytecode 'pre-expand-par-forms)
                       (requiring-resolve 'raster.compiler.core.macroexpand/macroexpand-all-preserving)
                       #(:form (materialize % nil))]]
        (is (identical? expression (prepare expression))))
      (is (= 0 (get-in (materialize (list 'let* [] expression) nil) [:stats :materialized]))))))

(deftest literal-owner-method-name-is-not-a-public-function-slot
  (is (= :reserved-jvm-method-name
         (try
           (bytecode/compile-specialized-class!
            "raster.compiled.ReservedObjectOwner"
            [{:name '__raster_literal :params [] :return-tag 'Object
              :walked-body [nil] :source-ns *ns*}])
           nil
           (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))

(deftest object-context-stays-in-selected-result-regions
  (let [value (Double. 5.125)
        field (list 'let* ['ignored '(swap! trace conj :enter)]
                    (list 'if 'flag
                          (list 'do '(swap! trace conj :then) value)
                          (list 'loop* ['i '(int 0)]
                                (list 'if '(< i 2) '(recur (unchecked-inc-int i)) value))))
        execute (:run (compile-value '[flag trace] [field value '(do false) '(do nil)]))]
    (doseq [flag [true false]]
      (let [trace (atom []) result (execute flag trace)]
        (is (= [:enter] (take 1 @trace)))
        (is (= (if flag [:enter :then] [:enter]) @trace))
        (is (identical? value (nth result 0)))
        (is (identical? (nth result 0) (nth result 1)))
        (is (= [false nil] (subvec result 2)))))))

(deftest children-and-separate-owners-cannot-substitute-another-pool
  (let [value (Double. 5.125)
        expression [(list 'fn* [] [value value]) value]
        a (compile-value [] expression)
        other (Double. 5.125)
        b (compile-value [] [(list 'fn* [] [other other]) other])
        [child token] ((:run a)) [other-child other-token] ((:run b))]
    (is (not= (:class a) (:class b)))
    (is (not= (.getName ^Class (:class a)) (.getName ^Class (:class b))))
    (is (identical? (.getClassLoader ^Class (:class a)) (.getClassLoader (class child))))
    (is (identical? token (first (child))))
    (is (identical? token (second (child))))
    (is (identical? other-token (first (other-child))))
    (is (not (identical? token other-token)))))

(deftest explicit-object-completion-snapshots-a-primitive-once
  (let [input (with-meta 'input {:tag 'double})
        execute (:run (compile-value
                       [input]
                       (list 'let* ['saved (completion/complete-value completion/object-input-contract input)]
                             ['saved 'saved])))]
    (doseq [value [5.125 1000.0 -0.0]]
      (let [[a b] (execute value)]
        (is (= value a b))
        (is (identical? a b))))))

(deftest object-result-invk-keeps-a-cached-reference
  (let [{:keys [iface-class iface-name]} (types/ensure-fn-interface! '[double])
        token (Double. 5.125) calls (atom 0)
        implementation (Proxy/newProxyInstance
                        (.getClassLoader ^Class iface-class) (into-array Class [iface-class])
                        (reify InvocationHandler
                          (invoke [_ _ method _]
                            (when (= "invk" (.getName ^java.lang.reflect.Method method))
                              (swap! calls inc) token))))
        target (intern (the-ns 'raster.compiler.backend.jvm.object-value-test)
                       'cached-object-result implementation)
        receiver (with-meta 'raster.compiler.backend.jvm.object-value-test/cached-object-result
                   {:tag iface-name :raster.type/tag iface-name :raster.type/ret-tag 'double})
        input (with-meta 'input {:tag 'double})
        expression [(list '. receiver 'invk input) token]]
    (try
      (let [execute (:run (compile-value [input] expression))]
        (doseq [input [1.0 -2.0]]
          (let [[actual expected] (execute input)]
            (is (identical? token expected))
            (is (identical? expected actual)))))
      (is (= 2 @calls))
      (finally (ns-unmap (:ns (meta target)) (:name (meta target)))))))

(deftest initialization-failure-never-returns-a-callable-artifact
  (let [seen (atom nil) published (atom nil)
        refusal (with-redefs-fn
                  {#'bytecode/initialize-literals!
                   (fn [owner _]
                     (reset! seen owner)
                     (throw (ex-info "injected owner initialization failure" {:reason :injected-literal-init})))}
                  #(try (reset! published (compile-value [] [(Double. 5.125)])) :incorrectly-published
                        (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))]
    (is (= :injected-literal-init refusal))
    (is (nil? @published))
    (is (class? @seen))
    (let [getter (.getMethod ^Class @seen @#'bytecode/literal-getter (into-array Class [Integer/TYPE]))]
      (let [failure (try (.invoke getter nil (object-array [(int 0)])) nil
                         (catch InvocationTargetException failure (.getCause failure)))]
        (is (instance? IllegalStateException failure))
        (is (= "JVM literal owner is not initialized" (ex-message failure)))))))
