(ns raster.compiler.backend.jvm.valhalla-detection-test
  "valhalla-available? must answer whether a value class can be defined in THIS
  JVM, not whether the JDK version number is high enough. JDK 27 started
  without --enable-preview refuses class files of version 71.65535, and before
  the probe every defval/defvalue then died with a ClassNotFoundException."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.valhalla :as valhalla]
            [raster.core :refer [defval defvalue]]))

(defn- preview-enabled? []
  (boolean (some #{"--enable-preview"}
                 (.getInputArguments (java.lang.management.ManagementFactory/getRuntimeMXBean)))))

(deftest availability-follows-the-running-jvm
  (testing "no preview features, no value classes"
    (when-not (preview-enabled?)
      (is (false? valhalla/valhalla-available?))))
  (testing "below JDK 27 there is never a value class"
    (when (< (.feature (Runtime/version)) 27)
      (is (false? valhalla/valhalla-available?)))))

(defval ProbeTag)
(defvalue ProbePoint [x :- Double, y :- Double])

(deftest defval-and-defvalue-load-on-every-jvm
  (testing "a singleton tag and a field value type both define and construct"
    (is (some? (->ProbeTag)))
    (is (= 2.0 (:y (->ProbePoint 1.0 2.0))))))
