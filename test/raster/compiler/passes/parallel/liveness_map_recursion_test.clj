(ns raster.compiler.passes.parallel.liveness-map-recursion-test
  "Silently-ignored-information family: fusion-pass local liveness helpers must recurse
   into MAP literals (a `case*` clause map) the same way util/free-syms-flat does. A
   symbol / aget read reachable only inside a map literal used to look dead, so the
   sole-consumer / liveness checks could wrongly fuse away a still-live buffer.

   These pin the two remaining local copies (par-fusion/all-symbol-uses and
   write-read-fuse/collect-aget-syms). Before the fix each returned #{} for the map
   literal, so the symbol was invisible."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.par-fusion :as pf]
            [raster.compiler.passes.parallel.write-read-fuse :as wrf]))

(deftest all-symbol-uses-sees-map-literal
  (testing "a symbol reachable only inside a map literal is collected, not dropped"
    (is (contains? (#'pf/all-symbol-uses '(foo {:k barsym})) 'barsym))
    (is (contains? (#'pf/all-symbol-uses
                    '(case* x 0 default {0 [(quote lit) hidden-sym]}))
                   'hidden-sym))))

(deftest collect-aget-syms-sees-map-literal
  (testing "an aget read reachable only inside a map literal is collected, not dropped"
    (is (contains? (#'wrf/collect-aget-syms '(foo {:k (clojure.core/aget buf i)})) 'buf))))
