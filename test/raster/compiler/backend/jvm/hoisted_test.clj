(ns raster.compiler.backend.jvm.hoisted-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.hoisted :as hoisted]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.closure :as closure]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]])
  (:import [java.lang.reflect Modifier]
           [java.util.concurrent CountDownLatch TimeUnit]))

(deftest nullable-logical-scalars-keep-reference-capable-helper-boundaries
  (let [projected (with-meta 'projected {:raster.type/tag 'double})
        alias (with-meta 'alias {:raster.type/tag 'double})
        answer (with-meta 'answer {:raster.type/tag 'boolean})
        projection (with-meta '(if (clojure.core/nil? input) nil
                                 (clojure.core/double input))
                     {:raster.type/tag 'double})
        consumer (list 'if (list 'clojure.core/nil? alias) true
                       (list 'if (list 'clojure.core/identical? alias alias) false false))
        compile! (fn [bindings body result-tag]
                   (closure/hoist-and-compile
                    (list 'let* bindings body) '[input] '[input]
                    {:param-env {'input 'Object} :return-tag result-tag :dtype :double}))
        consumed (compile! [projected projection alias projected answer consumer] answer 'boolean)
        returned (compile! [projected projection alias projected] alias 'Object)
        unused (compile! [projected projection] 7.0 'double)]
    (doseq [input [nil 2.0 -3.0 -0.0]]
      (is (= (nil? input) ((:fn consumed) input)))
      (is (= 7.0 ((:fn unused) input)))
      (let [actual ((:fn returned) input)]
        (if (nil? input)
          (is (nil? actual))
          (is (= (Double/doubleToRawLongBits (double input))
                 (Double/doubleToRawLongBits (double actual)))))))
    (let [helpers (vals (:helper-map consumed))
          producer (first (filter #(= projected (:binding %)) helpers))
          consumer (first (filter #(= answer (:binding %)) helpers))]
      (is (= 'Object (:return-tag producer)))
      (is (= 'Object (:tag (meta (first (:params consumer)))))))
    (let [bindings (second (:compute-form consumed))
          id (first (filter #(= alias %) (take-nth 2 bindings)))]
      (is (= 'double (:raster.type/tag (meta id))))
      (is (= 'Object (:tag (meta id)))))))

(deftest actual-primitive-conversions-preserve-values-and-traps
  (let [id (with-meta 'converted {:raster.type/tag 'double})
        expression '(if flag (clojure.core/double input) (clojure.core/double input))
        compiled (closure/hoist-and-compile
                  (list 'let* [id expression] id) '[input flag] '[input flag]
                  {:param-env {'input 'Object 'flag 'boolean}
                   :return-tag 'double :dtype :double})]
    (doseq [flag [true false]]
      (is (= 2.0 ((:fn compiled) 2.0 flag)))
      (is (thrown? NullPointerException ((:fn compiled) nil flag))))))

(deftest helper-carriers-belong-to-lexical-binding-occurrences
  (let [x (with-meta 'x {:raster.type/tag 'double})
        saved (with-meta 'saved {:raster.type/tag 'boolean})
        projection '(if (clojure.core/nil? input) nil (clojure.core/double input))
        test-x (list 'if (list 'clojure.core/nil? x) true
                     (list 'if (list 'clojure.core/identical? x x) false false))
        rebound (list 'let* [x projection saved test-x x '(clojure.core/double 3.0)] saved)
        nested (list 'let* [x 3.0 saved (list 'let* [x projection] test-x)] saved)]
    (doseq [source [rebound nested]]
      (let [compiled (closure/hoist-and-compile
                      source '[input] '[input]
                      {:param-env {'input 'Object} :return-tag 'boolean :dtype :double})]
        (is (true? ((:fn compiled) nil)))
        (is (false? ((:fn compiled) 2.0)))))))

(deftest nullable-integer-helper-inputs-keep-exact-arithmetic
  (let [x (with-meta 'x {:raster.type/tag 'long})
        answer (with-meta 'answer {:raster.type/tag 'long})
        expression (list 'if (list 'clojure.core/nil? x) -1 (list 'clojure.core/- x 1))
        compiled (closure/hoist-and-compile
                  (list 'let* [x '(if (clojure.core/nil? input) nil input) answer expression] answer)
                  '[input] '[input]
                  {:param-env {'input 'Object} :return-tag 'long :dtype :double})]
    (is (= 9007199254740994 ((:fn compiled) 9007199254740995)))
    (is (= -1 ((:fn compiled) nil)))))

(defn- wrapper [initialize]
  (let [result (hoisted/compile-hoisted-class!
                (str (gensym "raster.compiled.Publication_"))
                [{:name 'compute :params (mapv #(with-meta % {:tag 'Object}) '[x left right])
                  :return-tag 'Object :walked-body [['left 'right]]
                  :element-type 'double :source-ns *ns*}]
                {:field-tags '[Object Object] :fn-param-tags '[Object]
                 :return-tag 'Object :compute-name 'compute
                 :iface-class (:iface-class (types/ensure-fn-interface! '[Object]))})
        cls ^Class (:wrapper-class result)]
    (.newInstance (.getConstructor cls (into-array Class [Object]))
                  (object-array [initialize]))))

(defn- call [f entry]
  (case entry
    :invoke (f :argument)
    :apply (apply f [:argument])
    :typed (.invoke (.getMethod (class f) "invk" (into-array Class [Object]))
                    f (object-array [:argument]))))

(defn- worker [f]
  (let [result (promise)
        thread (Thread. ^Runnable (fn [] (try (deliver result (f))
                                            (catch Throwable t (deliver result t)))))]
    (.setDaemon thread true)
    (.start thread)
    [thread result]))

(defn- await-blocked? [^Thread thread]
  (let [deadline (+ (System/nanoTime) 3000000000)]
    (loop []
      (cond (= Thread$State/BLOCKED (.getState thread)) true
            (or (not (.isAlive thread)) (> (System/nanoTime) deadline)) false
            :else (do (Thread/yield) (recur))))))

(deftest concurrent-entry-publishes-complete-fields-once
  (doseq [[first-entry second-entry] [[:invoke :typed] [:typed :apply] [:apply :invoke]]]
    (let [entered (CountDownLatch. 1) release (CountDownLatch. 1)
          calls (atom 0)
          f (wrapper (fn [_]
                       (swap! calls inc)
                       (.countDown entered)
                       (when-not (.await release 10 TimeUnit/SECONDS)
                         (throw (ex-info "Initializer release timed out" {})))
                       [:left :right]))
          [first-thread first-result] (worker #(call f first-entry))
          second-worker (atom nil)]
      (try
        (is (.await entered 5 TimeUnit/SECONDS))
        (reset! second-worker (worker #(call f second-entry)))
        (is (await-blocked? (first @second-worker))
            "The second entry must wait under doInit's monitor")
        (is (= 1 @calls))
        (finally
          (.countDown release)
          (.join ^Thread first-thread 5000)
          (when-let [[thread _] @second-worker] (.join ^Thread thread 5000))))
      (is (not (.isAlive ^Thread first-thread)))
      (is (not (.isAlive ^Thread (first @second-worker))))
      (is (= [:left :right] (deref first-result 1000 ::timeout)))
      (is (= [:left :right] (deref (second @second-worker) 1000 ::timeout)))
      (doseq [entry [:invoke :typed :apply]] (is (= [:left :right] (call f entry))))
      (is (= 1 @calls)))))

(deftest publication-flags-and-stale-initialization-entry
  (let [calls (atom 0) f (wrapper (fn [_] (swap! calls inc) [:left :right]))
        cls (class f)
        initialized (.getDeclaredField cls "initialized")
        init (.getDeclaredMethod cls "doInit" (into-array Class [Object]))]
    (is (Modifier/isVolatile (.getModifiers initialized)))
    (is (Modifier/isSynchronized (.getModifiers init)))
    (is (= [:left :right] (call f :invoke)))
    (.setAccessible init true)
    ;; Deterministically model an entry which read false before a completed init.
    (.invoke init f (object-array [:argument]))
    (is (= 1 @calls))
    (is (= [:left :right] (call f :apply)))))

(deftest failed-initializer-is-not-published-and-can-retry
  (let [calls (atom 0) failure (IllegalStateException. "first initialization fails")
        f (wrapper (fn [_] (if (= 1 (swap! calls inc)) (throw failure) [:left :right])))
        initialized (.getDeclaredField (class f) "initialized")]
    (.setAccessible initialized true)
    (is (identical? failure (try (call f :invoke) nil (catch Throwable t t))))
    (is (false? (.getBoolean initialized f)))
    (is (= [:left :right] (call f :typed)))
    (is (.getBoolean initialized f))
    (is (= [:left :right] (call f :apply)))
    (is (= 2 @calls))))

(deftm boolean-predicate [x :- Double] :- Boolean
  (clojure.core/< x 0.0))

(deftest public-boolean-result-uses-the-object-callable-abi
  (let [compiled (pipeline/compile-aot #'boolean-predicate :target :jvm)]
    (doseq [x [-2.0 0.0 3.0 -0.5]]
      (is (= (boolean-predicate x) (compiled x)))
      (is (instance? Boolean (compiled x))))))

(deftest typed-wrappers-implement-their-actual-object-return-interface
  (doseq [[tag carrier input] [['double Double/TYPE 5.125]
                              ['long Long/TYPE 1000]
                              ['float Float/TYPE (float 5.125)]
                              ['int Integer/TYPE (int 1000)]
                              ['boolean Boolean/TYPE true]
                              ['boolean Boolean/TYPE false]
                              ['Object Object :value]
                              ['Object Object (Object.)]
                              ['Object Object (double-array [1.0 2.0])]
                              ['void Object :ignored]]
          cold-entry [:invoke :apply :typed]]
    (let [parameter-tag (if (= 'void tag) 'Object tag)
          interface (:iface-class (types/ensure-fn-interface! [parameter-tag]))
          interface-method (.getMethod ^Class interface "invk" (into-array Class [carrier]))
          initialized (atom [])
          result (hoisted/compile-hoisted-class!
                  (str (gensym "raster.compiled.ObjectReturn_"))
                  [{:name 'compute
                    :params [(with-meta 'x {:tag parameter-tag}) (with-meta 'field {:tag 'Object})]
                    :return-tag tag :walked-body [(if (= 'void tag) '(do nil) 'x)]
                    :element-type 'double :source-ns *ns*}]
                  {:field-tags '[Object] :fn-param-tags [parameter-tag]
                   :return-tag tag :compute-name 'compute :iface-class interface})
          cls ^Class (:wrapper-class result)
          f (.newInstance (.getConstructor cls (into-array Class [Object]))
                          (object-array [(fn [argument] (swap! initialized conj argument) [:field])]))
          invoke (fn [entry]
                   (case entry
                     :invoke (f input)
                     :apply (apply f [input])
                     :typed (.invoke interface-method f (object-array [input]))))
          expected (when-not (= 'void tag) input)]
      (is (= Object (.getReturnType interface-method)))
      (is (= (if (= 'void tag) Void/TYPE carrier)
             (.getReturnType ^java.lang.reflect.Method (get (:methods result) 'compute))))
      (is (= expected (invoke cold-entry)))
      (doseq [entry [:invoke :apply :typed]] (is (= expected (invoke entry))))
      (when (= 'Object tag)
        (doseq [entry [:invoke :apply :typed]] (is (identical? input (invoke entry)))))
      (when (= 'boolean tag)
        (is (thrown? NullPointerException (f nil)))
        (is (thrown? ClassCastException (f 0)))
        (is (thrown? ClassCastException (apply f [:truthy-is-not-boolean]))))
      (is (= [input] @initialized) "every cold entry boxes its initializer input exactly once"))))

(deftest typed-wrapper-preserves-mixed-width-argument-slots
  (doseq [cold-entry [:invoke :apply :typed]]
    (let [tags '[double boolean long double floats]
          interface (:iface-class (types/ensure-fn-interface! tags))
          method (.getMethod ^Class interface "invk"
                             (into-array Class [Double/TYPE Boolean/TYPE Long/TYPE Double/TYPE
                                                (class (float-array 0))]))
          initialized (atom [])
          result (hoisted/compile-hoisted-class!
                  (str (gensym "raster.compiled.MixedWidth_"))
                  [{:name 'compute
                    :params (conj (mapv #(with-meta %1 {:tag %2}) '[left flag count right values] tags)
                                  (with-meta 'field {:tag 'Object}))
                    :return-tag 'double
                    :walked-body ['(if flag (clojure.core/+ left (double count) right
                                                          (double (clojure.core/aget values 0)))
                                       left)]
                    :element-type 'double :source-ns *ns*}]
                  {:field-tags '[Object] :fn-param-tags tags
                   :return-tag 'double :compute-name 'compute :iface-class interface})
          cls ^Class (:wrapper-class result)
          f (.newInstance (.getConstructor cls (into-array Class [Object]))
                          (object-array [(fn [& arguments] (swap! initialized conj arguments) [:field])]))
          values (float-array [4.0])
          arguments [1.25 true (long 1000) 2.5 values]
          invoke (fn [entry args]
                   (case entry
                     :invoke (let [[a b c d e] args] (f a b c d e))
                     :apply (apply f args)
                     :typed (.invoke method f (object-array args))))]
      (is (= 1007.75 (invoke cold-entry arguments)))
      (doseq [entry [:invoke :apply :typed]]
        (is (= 1007.75 (invoke entry arguments)))
        (is (= 1.25 (invoke entry (assoc arguments 1 false)))))
      (is (= [arguments] (mapv vec @initialized))))))
