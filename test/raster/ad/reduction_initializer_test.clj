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

(deftest assembly-does-not-repeat-source-read-or-carry-analysis
  (let [captured (atom nil)
        original @#'reverse/assemble-ordered-reduce]
    (with-redefs-fn {#'reverse/assemble-ordered-reduce
                    (fn [step active]
                      (reset! captured [step active])
                      (original step active))}
      ordered-program)
    (let [[step active] @captured
          reject (fn [& _] (throw (ex-info "source analysis repeated" {})))
          assembled (with-bindings {#'reverse/*gensym-counter* (atom 0)}
                      (with-redefs-fn {#'reverse/analyze-par-map-body reject
                                      #'reverse/carry-dtype-consistent? reject}
                        #(original step active)))]
      (is (= :double-carry (:residual-kind assembled)))
      (is (seq (:forward-bindings assembled)))
      (is (seq (:backward-reduces assembled))))))

(defn- prepared-step-spec []
  (let [acc (with-meta 'acc {:raster.type/tag 'double})
        x (with-meta 'x {:raster.type/tag 'double})
        result (with-meta 'step_value {:raster.type/tag 'double})
        program (#'reverse/derive-flat-product-program
                  [result (list 'clojure.core/* acc x)] result [acc x])]
    {:acc-sym acc :init-expr 3.0 :idx-sym 'i :bound-expr 'size
     :body-expr '(throw (ex-info "unowned source used" {}))
     :agets [] :scalar-bindings [] :body-result nil :free-syms #{x}
     :typed-double? false :replay-values #{} :step-program program}))

(deftest prepared-pullbacks-do-not-reparse-or-replay-forward-steps
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [spec (prepared-step-spec)
          original @#'reverse/assemble-ordered-reduce
          reject (fn [& _] (throw (ex-info "prepared step normalized again" {})))
          record (with-redefs-fn {#'reverse/normalize-for-ad reject
                                 #'reverse/gen-inline-step-gradients reject}
                   #(original spec ['x]))
          bindings (vec (concat (:forward-bindings record)
                                [(:d-acc-sym record) 1.0]
                                (mapcat identity (:backward-maps record))
                                (mapcat identity (:backward-reduces record))))
          invoke (eval (list 'fn ['x 'size]
                            (list 'let* bindings
                                  [(:reduce-result-sym record) (first (:d-scalar-syms record))])))]
      (is (= :closure-tape (:residual-kind record)))
      (is (= [6.0 3.0] (invoke 2.0 1)))
      (is (= [24.0 36.0] (invoke 2.0 3)))
      (is (= 3.0 (first (invoke 2.0 0)))))))

(deftest prepared-steps-refuse-unproved-carry-replay-and-port-changes
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [spec (prepared-step-spec)]
      (doseq [[path value] [[[:typed-double?] true]
                            [[:step-program :active-inputs] ['x]]
                            [[:step-program :writes] #{'observations}]]]
        (is (= :ad-prepared-reduction-step-ports
               (try (#'reverse/assemble-ordered-reduce (assoc-in spec path value) ['x]) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest prepared-step-captures-inactive-observation-values-before-overwrite
  (with-bindings {#'reverse/*gensym-counter* (atom 0)
                  #'reverse/*constant-gradient-arrays* #{'observations}}
    (let [spec (prepared-step-spec)
          acc (:acc-sym spec)
          x (with-meta 'x {:raster.type/tag 'double})
          observed (with-meta 'observed {:raster.type/tag 'double})
          product (with-meta 'product {:raster.type/tag 'double})
          result (with-meta 'observed_step {:raster.type/tag 'double})
          array (with-meta 'observations {:raster.type/tag 'doubles})
          program (#'reverse/derive-flat-product-program
                    [observed (list 'clojure.core/aget array 'i)
                     product (list 'clojure.core/* acc x)
                     result (list 'clojure.core/* product observed)] result [acc x])
          record (#'reverse/assemble-ordered-reduce (assoc spec :step-program program) [x])
          bindings (vec (concat (:forward-bindings record)
                                ['_overwrite '(when (pos? size) (aset observations 0 99.0))
                                 (:d-acc-sym record) 1.0]
                                (mapcat identity (:backward-maps record))
                                (mapcat identity (:backward-reduces record))))
          invoke (eval (list 'fn ['x 'size array]
                            (list 'let* bindings
                                  [(:reduce-result-sym record) (first (:d-scalar-syms record))])))
          observations (double-array [2.0 3.0])]
      (is (= :closure-tape (:residual-kind record)))
      (is (= [72.0 72.0] (invoke 2.0 2 observations)))
      (is (= [99.0 3.0] (vec observations)))
      (is (= 3.0 (first (invoke 2.0 0 (double-array 0))))))))

(deftest saved-prepared-pullbacks-do-not-repeat-forward-effects
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [spec (update-in (prepared-step-spec) [:step-program :form]
                          #(list 'do '(swap! events inc) %))
          record (#'reverse/assemble-ordered-reduce spec ['x])
          bindings (vec (concat (:forward-bindings record)
                                [(:d-acc-sym record) 1.0]
                                (mapcat identity (:backward-maps record))
                                (mapcat identity (:backward-reduces record))))
          invoke (eval (list 'fn ['x 'size 'events]
                            (list 'let* bindings (:reduce-result-sym record))))]
      (doseq [size [-1 0 1 3]]
        (let [events (atom 0)]
          (is (= (* 3.0 (Math/pow 2.0 (max 0 size))) (invoke 2.0 size events)))
          (is (= (max 0 size) @events)))))))
