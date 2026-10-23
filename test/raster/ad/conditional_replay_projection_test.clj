(ns raster.ad.conditional-replay-projection-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse.normalize :as anf]
            [raster.ad.reverse :as reverse]))

(deftm scalar-branches [x :- Double] :- Double
  (if (n/> x 0.0)
    (n/* x x)
    (n/* 5.0 (n/* x x))))

(deftest conditional-residual-ports-compose-with-linearization
  ;; Replay bookkeeping must not introduce non-ANF structured values into
  ;; the reverse program subsequently differentiated by HVP.
  (doseq [[x derivative curvature] [[4.0 8.0 2.0] [1.0 2.0 2.0] [-2.0 -20.0 10.0]]]
    (let [[gradient product] ((jvp/hvp #'scalar-branches) x 1.0)]
      (is (= [derivative] (vec gradient)))
      (is (= [curvature] (vec product))))))

(deftest residual-storage-is-the-ordered-free-projection-of-the-reverse-region
  (let [forward '[dead (aget observations 99) payload (* x x) count callback]
        piece {:fwd-bindings forward
               :rev-bindings '[answer (let* [local count] (local payload))]
               :param-adj-syms '[answer]}
        projected (#'reverse/retain-live-residual-values piece)]
    (is (identical? forward (:fwd-bindings projected)))
    (is (= '[payload count] (:saved-values projected)))
    (is (= (:rev-bindings piece) (:rev-bindings projected)))
    (is (= [] (:saved-values
               (#'reverse/retain-live-residual-values
                (assoc piece :rev-bindings '[answer (quote [payload count])])))))))

(deftm selected-forward-only-read
  [x :- Double observations :- (Array double) counts :- (Array long) flag :- Long] :- Double
  (if (= flag 1)
    (let [tick (aset counts 0 (inc (aget counts 0)))
          unused (aget observations 0)]
      (n/* x x))
    (n/* x 3.0)))

(deftest residual-liveness-is-not-permission-to-drop-forward-evaluation
  (doseq [compile? [false true]]
    (let [gradient (reverse/value+grad #'selected-forward-only-read :wrt [0] :compile? compile?)
          counts (long-array 1)]
      (when compile?
        (is (true? (:raster.ad.reverse/compiled? (meta gradient)))
            (str (:raster.ad.reverse/compile-failure (meta gradient)))))
      (is (thrown? ArrayIndexOutOfBoundsException
                   (gradient 2.0 (double-array 0) counts 1)))
      (is (= [1] (vec counts)) "selected forward effect precedes the failed unused read")
      (is (= [6.0 3.0 nil nil nil]
             (gradient 2.0 (double-array 0) counts 0)))
      (is (= [1] (vec counts)) "unselected branch is not executed")
      (is (= [4.0 4.0 nil nil nil]
             (gradient 2.0 (double-array [9.0]) counts 1)))
      (is (= [2] (vec counts)) "selected branch is evaluated once"))))

(deftest nonempty-replay-ports-are-flat-anf-values
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (with-redefs-fn
      {#'reverse/let-gradient-pieces
       (fn [& _] {:fwd-bindings '[value (* x x)] :body-sym 'value
                  :pullback-replay-conditions {'observations '(= selector 1)}})}
      (fn []
        (let [{:keys [record fwd-patch]}
              ((get-method @#'reverse/ad-record :if) :if 'result
               '(if branch (* x x) (* x 2.0)) {'x true})
              [_ conditional] (fwd-patch '[result ignored])
              tape (:tape record)]
          (is (= (list 'clojure.core/nth tape 2)
                 (get (:replay-conditions record) 'observations)))
          (doseq [arm (drop 2 conditional)]
            (let [ports (last arm) bindings (second arm)]
              (is (= 3 (count ports)))
              (is (every? anf/trivial-expr? ports))
              (is (= '(= selector 1)
                     (get (into {} (map vec (partition 2 bindings))) (last ports)))))))))))

(deftm nested-selected-replay
  [x :- Double observations :- (Array double) destination :- (Array double)
   counts :- (Array long) outer :- Long inner :- Long] :- Double
  (let [value (if (= outer 1)
                (let [tick (aset counts 0 (inc (aget counts 0)))]
                  (if (= inner 1)
                    (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget observations k))))
                    (n/* x 2.0)))
                (n/* x 3.0))
        changed (aset destination 0 99.0)]
    value))

(deftm nested-cancellation-checkpoint [x :- Double cnt :- Long] :- Double
  (loop [i 1 acc 0.0]
    (if (< i cnt)
      (let [value (loop [j 1 a 0.0 b 0.0]
                    (if (< j 2)
                      (recur (inc j) (n/* x 1.0e16) (n/* x -1.0e16))
                      (n/+ a b)))]
        (recur (inc i) (n/+ acc (n/+ value x))))
      acc)))

(deftest residual-condition-projection-does-not-change-adjoint-fold-order
  (is (= [1.0 0.0 nil]
         ((reverse/value+grad #'nested-cancellation-checkpoint :wrt [0]) 1.0 2))))

(deftest nested-selection-exports-evaluated-replay-obligations
  (doseq [compile? [false true]]
   (let [gradient (reverse/value+grad #'nested-selected-replay :wrt [0] :compile? compile?)]
    (doseq [[outer inner coefficient effects] [[0 0 3.0 0] [0 1 3.0 0] [1 0 2.0 1]]]
      (let [shared (double-array [3.0]) counts (long-array 1)]
        (is (= [(n/* 2.0 coefficient) coefficient nil nil nil nil nil]
               (gradient 2.0 shared shared counts outer inner)))
        (is (= [99.0] (vec shared)))
        (is (= [effects] (vec counts)))))
    (let [observations (double-array [3.0]) destination (double-array [7.0])
          counts (long-array 1)]
      (is (= [6.0 3.0 nil nil nil nil nil] (gradient 2.0 observations destination counts 1 1)))
      (is (= [3.0] (vec observations)))
      (is (= [99.0] (vec destination)))
      (is (= [1] (vec counts))))
    (let [shared (double-array [3.0]) counts (long-array 1)]
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 shared shared counts 1 1) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [3.0] (vec shared)))
      ;; The selected primal executes once before the later write is fenced.
      (is (= [1] (vec counts)))))))

(deftest conditional-records-cannot-fall-back-to-flattened-selector-summaries
  (is (= :ad-region-replay-projection
         (try (#'reverse/record-replay-conditions {:type :if-region :sym 'value} {} {}) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
