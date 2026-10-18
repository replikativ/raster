(ns raster.ad.reduction-initializer-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]))

(defn- ordered-program []
  ;; Exercise the assembly itself: source preparation may already lift this
  ;; initializer, but an owned prepared step must not depend on that rewrite.
  (let [acc (with-meta 'acc {:raster.type/tag 'double})
        x (with-meta 'x {:raster.type/tag 'double})
        init (with-meta '(do (swap! events inc) 3.0) {:raster.type/tag 'double})
        step (with-meta (list 'clojure.core/* acc x) {:raster.type/tag 'double})
        record (with-bindings {#'reverse/*gensym-counter* (atom 0)}
                 (#'reverse/gen-reverse-ordered-reduce
                   (list 'raster.par/reduce acc init 'i 'size step) [x]))
        bindings (vec (concat (:forward-bindings record)
                              [(:d-acc-sym record) 1.0]
                              (mapcat identity (:backward-maps record))
                              (mapcat identity (:backward-reduces record))))]
    {:record record
     :invoke (eval (list 'fn ['x 'size 'events]
                         (list 'let* bindings
                               [(:reduce-result-sym record) (first (:d-scalar-syms record))])))}))

(deftest completed-initializer-is-not-replayed-in-backward
  (let [{:keys [record invoke]} (ordered-program)]
    (is (= :double-carry (:residual-kind record)))
    (doseq [size [-1 0 1 3]]
      (let [events (atom 0)
            [value gradient] (invoke 2.0 size events)
            trips (max 0 size)]
        (is (= (* 3.0 (Math/pow 2.0 trips)) value))
        (is (= (if (zero? trips) 0.0 (* 3.0 trips (Math/pow 2.0 (dec trips))))
               (double (or gradient 0.0))))
        (is (= 1 @events) "even a zero-trip reduction evaluates its initializer exactly once")))))

(deftest count-failure-precedes-initializer
  (let [{:keys [invoke]} (ordered-program)
        events (atom 0)]
    (is (thrown? ArithmeticException (invoke 2.0 Long/MAX_VALUE events)))
    (is (zero? @events))))
