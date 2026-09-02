(ns raster.compiler.passes.parallel.liveness-map-recursion-test
  "A write/read fusion access hidden in a map literal remains visible to liveness."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.write-read-fuse :as wrf]))

(deftest collect-aget-syms-sees-map-literal
  (testing "an aget read reachable only inside a map literal is collected, not dropped"
    (is (contains? (#'wrf/collect-aget-syms '(foo {:k (clojure.core/aget buf i)})) 'buf))))
