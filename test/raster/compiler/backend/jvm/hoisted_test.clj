(ns raster.compiler.backend.jvm.hoisted-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.hoisted :as hoisted]
            [raster.compiler.core.types :as types])
  (:import [java.lang.reflect Modifier]
           [java.util.concurrent CountDownLatch TimeUnit]))

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
