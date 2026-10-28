(ns raster.runtime.callsites-test
  (:require [clojure.test :refer [deftest is]]
            [raster.runtime.callsites :as callsites])
  (:import [java.lang.invoke MethodHandles MethodType MutableCallSite MethodHandle]
           [java.lang.ref WeakReference]
           [java.util.concurrent CountDownLatch TimeUnit]))

(definterface DoubleTarget
  (^Object invk [^double value]))

(deftype OffsetTarget [^double offset]
  DoubleTarget
  (invk [_ value] (+ value offset)))

(definterface LongTarget
  (^Object invk [^long value]))

(deftype IncompatibleTarget []
  LongTarget
  (invk [_ value] value))

(deftype MultiTarget []
  DoubleTarget
  (invk [_ ^double value] (+ value 10.0))
  LongTarget
  (invk [_ ^long value] (+ value 20)))

(definterface WideTarget
  (^Object invk [^long a ^double b value ^float c ^boolean flag ^int d
                 p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15]))

(deftype WideIdentityTarget []
  WideTarget
  (invk [_ _a _b value _c _flag _d
         _p0 _p1 _p2 _p3 _p4 _p5 _p6 _p7 _p8 _p9 _p10 _p11 _p12 _p13 _p14 _p15]
    value))

(def ^:private method-type
  (MethodType/methodType Double/TYPE (into-array Class [Double/TYPE])))

(defn- with-target [initial check]
  (let [scope (create-ns (gensym "raster.callsite-lifecycle."))
        v (intern scope 'target initial)]
    (try (check v) (finally (remove-ns (ns-name scope))))))

(defn- link [v]
  (callsites/bootstrap (MethodHandles/lookup) "test" method-type
                       (str (ns-name (:ns (meta v)))) (str (:name (meta v)))))

(defn- invoke [^MutableCallSite site value]
  (.invokeWithArguments ^MethodHandle (.dynamicInvoker site) (object-array [(double value)])))

(deftest root-replacement-and-with-redefs-restore-update-already-linked-sites
  (with-target
    (OffsetTarget. 0.0)
    (fn [v]
      (let [site (link v)]
        (is (= 1.0 (invoke site 1.0)))
        (alter-var-root v (constantly (OffsetTarget. 10.0)))
        (is (= 11.0 (invoke site 1.0)))
        (with-redefs-fn {v (OffsetTarget. 20.0)}
          #(is (= 21.0 (invoke site 1.0))))
        (is (= 11.0 (invoke site 1.0)))))))

(deftest delayed-explicit-invalidation-cannot-reinstall-an-older-root
  (with-target
    (OffsetTarget. 0.0)
    (fn [v]
      (let [site (link v) old @v]
        (alter-var-root v (constantly (OffsetTarget. 4.0)))
        (callsites/invalidate! v old)
        (is (= 5.0 (invoke site 1.0)))
        (is (= 5.0 (.invk ^DoubleTarget @v 1.0)))))))

(deftest separate-registration-establishes-freshness-and-checks-the-fixed-abi
  (with-target
    (OffsetTarget. 6.0)
    (fn [v]
      (let [stale (callsites/create-target-handle (OffsetTarget. 0.0) method-type)
            site (MutableCallSite. ^MethodHandle stale)]
        (is (identical? site (callsites/register-callsite! v site method-type)))
        (is (= 7.0 (invoke site 1.0)))
        (is (= :typed-callsite-registration
               (try (callsites/register-callsite!
                     v site (MethodType/methodType Long/TYPE (into-array Class [Double/TYPE])))
                    nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (alter-var-root v (constantly (OffsetTarget. 8.0)))
        (is (= 9.0 (invoke site 1.0)))))))

(deftest incompatible-relink-fails-closed-and-the-live-site-can-recover
  (with-target
    (OffsetTarget. 0.0)
    (fn [v]
      (let [site (link v)]
        (alter-var-root v (constantly (IncompatibleTarget.)))
        (is (= :typed-callsite-relink
               (try (invoke site 1.0) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (alter-var-root v (constantly (OffsetTarget. 7.0)))
        (is (= 8.0 (invoke site 1.0)))))))

(deftest missing-initial-target-is-not-a-permanently-stale-bootstrap
  (doseq [initial [nil (IncompatibleTarget.)]]
    (with-target
      initial
      (fn [v]
        (let [site (link v)]
          (is (= :typed-callsite-relink
                 (try (invoke site 1.0) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (alter-var-root v (constantly (OffsetTarget. 2.0)))
          (is (= 3.0 (invoke site 1.0))))))))

(deftest a-failing-signature-does-not-leave-other-sites-stale-or-prevent-recovery
  (with-target
    (MultiTarget.)
    (fn [v]
      (let [long-type (MethodType/methodType Long/TYPE (into-array Class [Long/TYPE]))
            long-site (callsites/bootstrap (MethodHandles/lookup) "long-test" long-type
                                           (str (ns-name (:ns (meta v)))) (str (:name (meta v))))
            double-site (link v)
            invoke-long (fn [] (.invokeWithArguments ^MethodHandle (.dynamicInvoker long-site)
                                                     (object-array [(long 1)])))]
        (is (= 21 (invoke-long)))
        (alter-var-root v (constantly (OffsetTarget. 5.0)))
        (is (thrown? clojure.lang.ExceptionInfo (invoke-long)))
        (is (= 6.0 (invoke double-site 1.0)))
        (alter-var-root v (constantly (MultiTarget.)))
        (is (= 21 (invoke-long)))
        (is (= 11.0 (invoke double-site 1.0)))))))

(deftest number-conversion-updates-target-once-before-callsite-execution
  (with-target
    (OffsetTarget. 0.0)
    (fn [v]
      (let [site (link v) conversions (atom 0)
            value (proxy [Number] []
                    (doubleValue []
                      (swap! conversions inc)
                      (alter-var-root v (constantly (OffsetTarget. 9.0)))
                      (double 1.0))
                    (floatValue [] (float 1.0)) (longValue [] (long 1)) (intValue [] (int 1)))]
        (is (= 1.0 (invoke site 1.0)))
        (is (= 10.0 (invoke site value)))
        (is (= 1 @conversions))))))

(deftest explicit-invalidation-covers-nonnotifying-unbind-and-preserves-recovery
  (with-target
    (OffsetTarget. 0.0)
    (fn [^clojure.lang.Var v]
      (let [site (link v)]
        ;; Var.unbindRoot does not notify watches; this path remains explicit.
        (.unbindRoot v)
        (callsites/invalidate! v nil)
        (is (thrown? clojure.lang.ExceptionInfo (invoke site 1.0)))
        (.bindRoot v (OffsetTarget. 3.0))
        (is (= 4.0 (invoke site 1.0)))))))

(deftest failure-handles-preserve-reference-and-void-wide-mixed-signatures
  (doseq [result [Object Void/TYPE]]
    (with-target
      (WideIdentityTarget.)
      (fn [v]
        (let [parameters (into-array Class (concat [Long/TYPE Double/TYPE Object Float/TYPE Boolean/TYPE Integer/TYPE]
                                                   (repeat 16 Object)))
              mt (MethodType/methodType result parameters)
              site (callsites/bootstrap (MethodHandles/lookup) "wide-test" mt
                                        (str (ns-name (:ns (meta v)))) (str (:name (meta v))))
              payload (Object.)
              arguments (object-array (concat [(long 1) (double 2) payload (float 3) true (int 4)]
                                                (repeat 16 nil)))
              run (fn [] (.invokeWithArguments ^MethodHandle (.dynamicInvoker site) arguments))]
          (is (= mt (.type site)))
          (alter-var-root v (constantly (IncompatibleTarget.)))
          (let [failure (try (run) nil (catch clojure.lang.ExceptionInfo e e))]
            (is (= :typed-callsite-relink (:reason (ex-data failure))))
            (is (instance? Throwable (.getCause ^Throwable failure))))
          (alter-var-root v (constantly (WideIdentityTarget.)))
          (if (identical? Object result)
            (is (identical? payload (run)))
            (is (nil? (run)))))))))

(deftest dead-site-references-are-pruned-without-forcing-gc
  (with-target
    (OffsetTarget. 0.0)
    (fn [v]
      (let [first-site (link v)
            registry (var-get #'callsites/registry)
            first-reference (locking registry (first (get registry v)))]
        (is (identical? first-site (.get ^WeakReference first-reference)))
        (.clear ^WeakReference first-reference)
        (let [second-site (link v)
              entries (locking registry (vec (get registry v)))]
          (is (= 1 (count entries)))
          (is (identical? second-site (.get ^WeakReference (first entries))))
          (.clear ^WeakReference (first entries))
          (callsites/invalidate! v @v)
          (is (locking registry (nil? (get registry v)))))))))

(deftest recursive-root-notifications-follow-the-final-raw-root
  (with-target
    (OffsetTarget. 0.0)
    (fn [v]
      (let [site (link v) intermediate (OffsetTarget. 3.0) final (OffsetTarget. 7.0)]
        (add-watch v ::recursive-change
                   (fn [_ reference _ new]
                     (when (identical? intermediate new)
                       (alter-var-root reference (constantly final)))))
        (try
          (alter-var-root v (constantly intermediate))
          (is (identical? final @v))
          (is (= 8.0 (invoke site 1.0)))
          (finally (remove-watch v ::recursive-change)))))))

(deftest linking-cannot-miss-a-root-update-between-reading-and-registering
  (with-target
    (OffsetTarget. 0.0)
    (fn [v]
      (let [creating (CountDownLatch. 1) proceed (CountDownLatch. 1)
            publishing (CountDownLatch. 1) first? (atom true)
            original callsites/create-target-handle]
        (with-redefs [callsites/create-target-handle
                      (fn [impl mt]
                        (when (compare-and-set! first? true false)
                          (.countDown creating)
                          (when-not (.await proceed 5 TimeUnit/SECONDS)
                            (throw (IllegalStateException. "test link was not released"))))
                        (original impl mt))]
          (let [linked (future (link v))]
            (try
              (is (.await creating 5 TimeUnit/SECONDS))
              (let [updated (future (.countDown publishing)
                                    (alter-var-root v (constantly (OffsetTarget. 8.0))))]
                (is (.await publishing 5 TimeUnit/SECONDS))
                (is (= :pending (deref updated 30 :pending))
                    "root publication is serialized with initial target registration")
                (.countDown proceed)
                (is (not= :timeout (deref updated 5000 :timeout)))
                (let [site (deref linked 5000 :timeout)]
                  (is (not= :timeout site))
                  (when-not (= :timeout site) (is (= 9.0 (invoke site 1.0))))))
              (finally
                (.countDown proceed)
                (deref linked 5000 nil)))))))))
