(ns raster.compiler.backend.cpu.host-entry-contract-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.compiler.backend.cpu.aot :as aot]))

;; Scalars deliberately precede arrays in the public signature: native pointer
;; positions must follow the retained array ABI, not public parameter offsets.
(deftm checked-write
  [x :- Double a :- (Array double) b :- (Array double)] :- Double
  (let [unused (if (clojure.core/identical? a b)
                 (throw (clojure.core/ex-info "different original objects" {:input (quote a)}))
                 nil)
        written (aset b 0 x)]
    (n/* x (aget a 0))))

(deftm checked-empty
  [x :- Double a :- (Array double) b :- (Array double)] :- Double
  (let [unused (if (clojure.core/identical? a b)
                 (throw (clojure.core/ex-info "different original objects" {})) nil)] x))

(deftest real-native-original-argument-contract
  (let [compiled (aot/compile-aot-c #'checked-write :double)
        a (double-array [3.0]) b (double-array [-1.0])]
    (is (= 6.0 (checked-write 2.0 (aclone a) (aclone b)) (compiled 2.0 a b)))
    (is (= [2.0] (vec b)))
    (is (= [:read :read-write] (:native-pointer-access (meta compiled))))
    (is (= 1 (count (:entry-preconditions (meta compiled)))))
    (let [shared (double-array [19.0])
          error (try (compiled 7.0 shared shared) nil
                     (catch clojure.lang.ExceptionInfo e e))]
      (is (= :native-object-alias (:reason (ex-data error))))
      (is (= [19.0] (vec shared)) "entry refusal must precede the store"))))

(deftest empty-object-is-not-a-range-proof
  (let [compiled (aot/compile-aot-c #'checked-empty :double)
        a (double-array 0) b (double-array 0)]
    (is (= 9.0 (compiled 9.0 a b)))
    (is (= :native-object-alias
           (:reason (ex-data (try (compiled 9.0 a a) nil
                                 (catch clojure.lang.ExceptionInfo e e))))))))
