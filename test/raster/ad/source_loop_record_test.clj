(ns raster.ad.source-loop-record-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]))

(deftm scoped-loop
  [x :- Double counts :- (Array long) cnt :- Long selected :- Long] :- Double
  (if (= selected 1)
    (let [value (loop [i 1 acc x]
                  (if (< i cnt)
                    (let [tick (aset counts 0 (inc (aget counts 0)))]
                      (recur (inc i) (n/* acc x)))
                    acc))]
      (n/* value x))
    (n/+ x 1.0)))

(deftm continuation-cancellation [x :- Double cnt :- Long] :- Double
  (let [value (loop [i 1 a 0.0 b 0.0]
                (if (< i cnt)
                  (recur (inc i) (n/* x 1.0e16) (n/* x -1.0e16))
                  (n/+ a b)))]
    (n/+ value x)))

(deftm nested-step-cancellation [x :- Double cnt :- Long] :- Double
  (loop [i 1 acc 0.0]
    (if (< i cnt)
      (recur (inc i)
             (n/+ acc
                  (let [value (loop [j 1 a 0.0 b 0.0]
                                (if (< j 2)
                                  (recur (inc j) (n/* x 1.0e16) (n/* x -1.0e16))
                                  (n/+ a b)))]
                    (n/+ value x))))
      acc)))

(deftm prelude-loop-cancellation [x :- Double cnt :- Long] :- Double
  (loop [i 1 acc 0.0]
    (if (< i cnt)
      (let [value (loop [j 1 a 0.0 b 0.0]
                    (if (< j 2)
                      (recur (inc j) (n/* x 1.0e16) (n/* x -1.0e16))
                      (n/+ a b)))]
        (recur (inc i) (n/+ acc (n/+ value x))))
      acc)))

(deftm shared-effectful-prelude
  [x :- Double counts :- (Array long) cnt :- Long selected :- Long] :- Double
  (if (= selected 1)
    (let [value (loop [i 1 a x b x]
                  (if (< i cnt)
                    (let [tick (aset counts 0 (inc (aget counts 0)))]
                      (recur (inc i) (n/* a x) (n/+ b x)))
                    (n/+ a b)))]
      (n/* value x))
    x))

(deftm terminal-only-active
  [x :- Double counts :- (Array long) cnt :- Long] :- Double
  (loop [i 1]
    (if (< i cnt)
      (let [tick (aset counts 0 (inc (aget counts 0)))]
        (recur (inc i)))
      x)))

(deftm effectful-inactive-recur
  [x :- Double counts :- (Array long)] :- Double
  (loop [i 0 acc x]
    (if (< i 2)
      (recur (let [tick (aset counts 0 (inc (aget counts 0)))] (inc i))
             (n/* acc x))
      acc)))

(deftm unused-singular-initializer [x :- Double cnt :- Long] :- Double
  (loop [i 1 acc x unused (n/sqrt x)]
    (if (< i cnt) (recur (inc i) acc unused) acc)))

(deftest unused-loop-cotangents-remain-absent-not-explicit-zero
  ;; A nil/absent edge cannot be replaced by numeric zero: at x=0 the
  ;; unnecessary sqrt pullback would evaluate 0/0 and poison the live edge.
  (doseq [cnt [1 2]]
    (is (= [0.0 1.0 nil]
           ((reverse/value+grad #'unused-singular-initializer :wrt [0]) 0.0 cnt)))))

(deftest visited-inactive-recur-effects-have-a-checked-source-scope
  (let [counts (long-array 1)
        gradient (reverse/value+grad #'effectful-inactive-recur :wrt [0])]
    (is (= [8.0 12.0 nil] (gradient 2.0 counts)))
    (is (= [2] (vec counts)))))

(deftm successive-step-replay
  [x :- Double observations :- (Array double) destination :- (Array double)] :- Double
  (loop [i 1 acc x]
    (if (< i 3)
      (let [changed (aset destination 0 (n/+ (aget destination 0) 1.0))]
        (recur (inc i)
               (n/+ acc (par/reduce sum 0.0 k 1
                          (n/+ sum (n/* x (aget observations k)))))))
      acc)))

(deftm conditional-replay-then-write
  [x :- Double observations :- (Array double) destination :- (Array double) mode :- Long] :- Double
  (loop [i 0 acc x]
    (if (< i 2)
      (let [value (if (= i 0)
                    (if (= mode 1)
                      (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget observations k))))
                      (n/* x 2.0))
                    (let [changed (aset destination 0 (n/+ (aget destination 0) 1.0))]
                      (n/* x 3.0)))]
        (recur (inc i) (n/+ acc value)))
      acc)))

(deftm saved-scalar-across-writes
  [x :- Double observations :- (Array double) destination :- (Array double)] :- Double
  (loop [i 0 acc x]
    (if (< i 2)
      (let [value (aget observations 0)
            changed (aset destination 0 (n/+ (aget destination 0) 1.0))]
        (recur (inc i) (n/+ acc (n/* x value))))
      acc)))

(deftm replay-before-terminal-write
  [x :- Double observations :- (Array double) destination :- (Array double)] :- Double
  (loop [i 0 acc x]
    (if (< i 1)
      (recur (inc i)
             (n/+ acc (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget observations k))))))
      (let [changed (aset destination 0 99.0)] acc))))

(deftm nested-write-after-replay
  [x :- Double observations :- (Array double) destination :- (Array double)] :- Double
  (loop [i 0 acc x]
    (if (< i 2)
      (let [changed (loop [j 0]
                      (if (< j 1)
                        (let [tick (aset destination 0 (n/+ (aget destination 0) 1.0))]
                          (recur (inc j)))
                        nil))]
        (recur (inc i)
               (n/+ acc (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget observations k)))))))
      acc)))

(deftm nested-implicit-scan-after-replay
  [x :- Double observations :- (Array double) destination :- (Array double) mode :- Long] :- Double
  (loop [i 0 acc x]
    (if (< i 2)
      (let [changed (loop [j 1]
                      (if (< j 2)
                        (let [ignored (if (= mode 1)
                                        (par/scan destination carry (double i) k 1 double
                                          (n/+ carry 1.0))
                                        destination)]
                          (recur (inc j)))
                        nil))]
        (recur (inc i)
               (n/+ acc (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget observations k)))))))
      acc)))

(deftest implicit-soac-writes-in-inactive-nested-loops-check-the-frontier
  (let [gradient (reverse/value+grad #'nested-implicit-scan-after-replay :wrt [0])
        observations (double-array [3.0]) destination (double-array [99.0])]
    (is (= [14.0 7.0 nil nil nil] (gradient 2.0 observations destination 1)))
    (is (= [2.0] (vec destination)))
    (let [shared (double-array [3.0])]
      (is (= [14.0 7.0 nil nil nil] (gradient 2.0 shared shared 0)))
      (is (= [3.0] (vec shared)))
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 shared shared 1) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [1.0] (vec shared)) "second iteration's implicit write must be fenced"))))

(deftm changing-replay-object
  [x :- Double observations :- (Array double) alternate :- (Array double)
   destination :- (Array double)] :- Double
  (loop [i 0 acc x current observations]
    (if (< i 2)
      (let [changed (if (= i 1) (aset destination 0 42.0) nil)]
        (recur (inc i)
               (n/+ acc (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget current k)))))
               alternate))
      acc)))

(deftm zero-trip-replay-frontier
  [x :- Double observations :- (Array double) destination :- (Array double) cnt :- Long] :- Double
  (loop [i 0 acc x]
    (if (< i cnt)
      (recur (inc i)
             (n/+ acc (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget observations k))))))
      (let [changed (aset destination 0 42.0)] acc))))

(deftm private-double-replay-with-long-effects
  [x :- Double observations :- (Array double) counts :- (Array long) mode :- Long] :- Double
  (loop [i 0 acc x]
    (if (< i 2)
      (let [tick (aset counts 0 (inc (aget counts 0)))
            value (if (= mode 1)
                    (let [^doubles copy (aclone observations)]
                      (par/reduce sum 0.0 k 1 (n/+ sum (n/* x (aget copy k)))))
                    (n/* x 2.0))]
        (recur (inc i) (n/+ acc value)))
      acc)))

(deftest incompatible-storage-classes-do-not-create-a-private-replay-hazard
  (let [gradient (reverse/value+grad #'private-double-replay-with-long-effects :wrt [0])]
    (doseq [[mode value dx] [[1 14.0 7.0] [0 10.0 5.0]]]
      (let [counts (long-array 1)]
        (is (= [value dx nil nil nil]
               (gradient 2.0 (double-array [3.0]) counts mode)))
        (is (= [2] (vec counts)))))))

(deftest public-scoped-loop-shares-the-checked-loop-builder
  (let [gradient (reverse/value+grad #'scoped-loop :wrt [0])]
    (doseq [[cnt selected value dx effects]
            [[1 1 4.0 4.0 0] [3 1 16.0 32.0 2] [3 0 3.0 1.0 0]]]
      (let [counts (long-array 1)]
        (is (= value (scoped-loop 2.0 counts cnt selected)))
        (is (= [effects] (vec counts)))
        (aset counts 0 0)
        (is (= [value dx nil nil nil] (gradient 2.0 counts cnt selected)))
        (is (= [effects] (vec counts)))))))

(deftest returned-alternatives-are-may-edges-not-exact-aliases
  (let [xs (with-meta 'xs {:raster.type/tag 'doubles})
        ys (with-meta 'ys {:raster.type/tag 'doubles})
        result (with-meta 'result {:raster.type/tag 'doubles})]
    (with-bindings {#'reverse/*gensym-counter* (atom 0)
                    #'reverse/*source-memory-context*
                    {:values [xs ys] :value-facts {xs {:value xs} ys {:value ys}}}}
      (with-redefs-fn
        {#'reverse/ad-record (fn [& _]
                              {:record {:type :probe :sym result :result-aliases #{xs ys}}})}
        #(let [state (#'reverse/forward-pass [result (list 'identity xs)] [xs])]
           (is (contains? (get-in state [:aliases result]) xs))
           (is (contains? (get-in state [:aliases result]) ys))
           (is (nil? (get-in state [:value-facts result :alias]))))))))

(deftest a-shared-prelude-executes-once-not-once-per-active-carry
  (let [gradient (reverse/value+grad #'shared-effectful-prelude :wrt [0])
        counts (long-array 1)]
    (is (= 28.0 (shared-effectful-prelude 2.0 counts 3 1)))
    (is (= [2] (vec counts)))
    (aset counts 0 0)
    (is (= [28.0 44.0 nil nil nil] (gradient 2.0 counts 3 1)))
    (is (= [2] (vec counts)))))

(deftest inactive-steps-still-execute-before-an-active-terminal
  (let [gradient (reverse/value+grad #'terminal-only-active :wrt [0])]
    (doseq [cnt [1 3]]
      (let [counts (long-array 1)]
        (is (= [2.0 1.0 nil nil] (gradient 2.0 counts cnt)))
        (is (= [(dec cnt)] (vec counts)))))))

(deftest later-iterations-cannot-invalidate-earlier-live-replays
  (let [gradient (reverse/value+grad #'successive-step-replay :wrt [0])
        observations (double-array [3.0]) destination (double-array [3.0])
        call (fn [x] (let [shared (double-array [3.0])]
                       (successive-step-replay x shared shared)))
        h 1.0e-5]
    (is (= [14.0 7.0 nil nil] (gradient 2.0 observations destination)))
    (is (= 20.0 (call 2.0)))
    (is (< (Math/abs (- 10.0 (/ (- (call (+ 2.0 h)) (call (- 2.0 h))) (* 2.0 h)))) 1.0e-7))
    ;; Currently accepted as [20,11,...], not the independent derivative 10.
    ;; A runtime fence must precede the write invalidating the first pullback.
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 observations observations) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (<= (aget observations 0) 4.0))))

(deftest executed-replay-paths-not-current-selectors-own-the-frontier
  (let [gradient (reverse/value+grad #'conditional-replay-then-write :wrt [0])]
    (let [shared (double-array [3.0])]
      (is (= [12.0 6.0 nil nil nil] (gradient 2.0 shared shared 0)))
      (is (= [4.0] (vec shared))))
    (let [shared (double-array [3.0])]
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 shared shared 1) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [3.0] (vec shared))))))

(deftest taped-scalar-values-do-not-retain-the-source-array
  (let [gradient (reverse/value+grad #'saved-scalar-across-writes :wrt [0])
        shared (double-array [3.0])]
    (is (= [16.0 8.0 nil nil] (gradient 2.0 shared shared)))
    (is (= [5.0] (vec shared)))))

(deftest terminal-and-nested-writes-check-retained-replay-objects
  (doseq [[function unchanged] [[#'replay-before-terminal-write 3.0]
                                [#'nested-write-after-replay 4.0]]]
    (let [gradient (reverse/value+grad function :wrt [0])
          shared (double-array [3.0])]
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 shared shared) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [unchanged] (vec shared))))))

(deftest previous-object-identity-survives-a-changing-carry
  (let [gradient (reverse/value+grad #'changing-replay-object :wrt [0])
        observations (double-array [3.0]) alternate (double-array [7.0])
        destination (double-array [99.0])]
    (is (= [22.0 11.0 nil nil nil] (gradient 2.0 observations alternate destination)))
    (is (= [42.0] (vec destination)))
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 observations alternate observations) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [3.0] (vec observations)))))

(deftest zero-trip-has-no-retained-replay
  (let [gradient (reverse/value+grad #'zero-trip-replay-frontier :wrt [0])
        shared (double-array [3.0])]
    (is (= [2.0 1.0 nil nil nil] (gradient 2.0 shared shared 0)))
    (is (= [42.0] (vec shared)))))

(deftest pure-loops-do-not-allocate-replay-frontiers
  (let [code (reverse/grad-expr
              '(loop* [i 1 a x]
                 (if (< i 3) (recur (inc i) (* a x)) a)) '[x])]
    (is (not-any? #{'java.util.IdentityHashMap.} (tree-seq coll? seq code)))))

(deftest existing-continuation-checkpoints-are-not-reassociated
  ;; The restored reference produces zero, not one: its continuation contribution
  ;; enters before the cancelling large contributions. No reordering permission
  ;; was granted. The shared-step prototype must preserve this checkpoint too.
  (doseq [function [#'continuation-cancellation #'prelude-loop-cancellation]]
    (is (= [1.0 0.0 nil] ((reverse/value+grad function :wrt [0]) 1.0 2)))))
