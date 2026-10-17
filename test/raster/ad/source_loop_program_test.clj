(ns raster.ad.source-loop-program-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftm nested-replay-then-write
  [x :- Double ys :- (Array double) zs :- (Array double)] :- Double
  (let [out (double-array 1)]
    (par/map! out i 1 nil
      (loop [j 1 acc x]
        (if (< j 3) (recur (inc j) (n/* acc x))
            (par/reduce s 0.0 k 1 (n/+ s (n/* acc (aget ys k)))))))
    (aset zs 0 7.0)
    (aget out 0)))

(deftm write-before-nested-replay [x :- Double ys :- (Array double)] :- Double
  (let [out (double-array 1) ignored (aset ys 0 7.0)]
    (par/map! out i 1 nil
      (loop [j 1 acc x]
        (if (< j 3) (recur (inc j) (n/* acc x))
            (par/reduce s 0.0 k 1 (n/+ s (n/* acc (aget ys k)))))))
    (aget out 0)))

(deftm scalar-capture-then-write [x :- Double ys :- (Array double)] :- Double
  (let [out (double-array 1)]
    (par/map! out i 1 nil
      (loop [j 1 acc x]
        (if (< j 3) (recur (inc j) (n/* acc x)) (n/* acc (aget ys 0)))))
    (aset ys 0 7.0)
    (aget out 0)))

(deftm inactive-initializer-order
  [x :- Double divisor :- Long ys :- (Array double)] :- Double
  (loop [i (quot 3 divisor) observed (aget ys 0) acc x]
    (if (< i 3) (recur (inc i) observed (n/* acc x)) acc)))

(deftm inactive-initializer-before-origin
  [x :- Double divisor :- Long ys :- (Array double)] :- Double
  (loop [observed (aget ys 0) i (quot 3 divisor) acc x]
    (if (< i 3) (recur observed (inc i) (n/* acc x)) acc)))

(deftm map-step-changes-index
  [xs :- (Array double) indices :- (Array int)] :- Double
  (let [out (double-array 1)
        ignored (par/map! out i 1 nil
                  (let [value (aget xs (aget indices i))
                        changed (aset indices i 1)] value))]
    (aget out 0)))

(deftm float-step-writes-other-index-port
  [xs :- (Array float) indices :- (Array int) destination :- (Array int)] :- Float
  (par/reduce acc (float 1.0) i 1
    (let [value (aget xs (aget indices i))
          changed (aset destination i 1)]
      (n/* acc value))))

(deftm child-write-before-own-replay
  [x :- Double ys :- (Array double)] :- Double
  (let [out (double-array 1)
        ignored (par/map! out i 1 nil
                  (let [changed (aset ys 0 3.0)]
                    (par/reduce acc 0.0 k 1 (n/+ acc (n/* x (aget ys k))))))]
    (aget out 0)))

(deftest parent-scatter-indices-precede-child-effects
  (is (= :replayed-array-overwritten
         (try (rev/value+grad #'map-step-changes-index :wrt [0]) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (let [gradient (rev/value+grad #'float-step-writes-other-index-port :wrt [0])
        indices (int-array [0]) destination (int-array [0])
        [value dx] (gradient (float-array [2.0 7.0]) indices destination)]
    (is (= (float 2.0) value))
    (is (= [1.0 0.0] (vec dx)))
    (is (= [1] (vec destination)))
    (is (= :ad-replayed-array-alias
           (try (gradient (float-array [2.0 7.0]) indices indices) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [0] (vec indices)))))

(deftest child-owned-replays-may-follow-lawful-earlier-writes
  (let [ys (double-array [17.0])]
    (is (= [6.0 3.0 nil] ((rev/value+grad #'child-write-before-own-replay :wrt [0]) 2.0 ys)))
    (is (= [3.0] (vec ys)))))

(deftest prior-index-obligations-use-retained-root-and-visibility-authority
  (let [root (with-meta 'root {:raster.type/tag 'ints})
        alias (with-meta 'alias {:raster.type/tag 'ints})
        hidden (with-meta 'hidden {:raster.type/tag 'ints})
        offset (with-meta 'offset {:raster.type/tag 'long})
        seen (atom nil)]
    (with-bindings {#'rev/*source-memory-context*
                    {:values [root] :replayed {'earlier true}
                     :unprojectable {'earlier-private true}
                     :value-facts {root {:value root} alias {:value alias :alias root}
                                   hidden {:value hidden} offset {:value offset}}}}
      (with-redefs-fn {#'rev/gen-reverse-let-program
                      (fn [& _] (reset! seen @#'rev/*source-memory-context*)
                        {:writes #{} :aliases {}})}
        #(#'rev/gen-reverse-read-region [] [1.0] [] #{alias hidden offset})))
    (is (= #{'root 'earlier} (set (keys (:replayed @seen)))))
    (is (= #{'hidden 'earlier-private} (set (keys (:unprojectable @seen)))))
    (is (= [root] (:values @seen)))))

(deftest inactive-compound-initializers-keep-source-order
  (doseq [[source expected] [[#'inactive-initializer-order ArithmeticException]
                             [#'inactive-initializer-before-origin ArrayIndexOutOfBoundsException]]]
    (let [gradient (rev/value+grad source :wrt [0])
          exception-class (fn [f] (try (f 2.0 0 (double-array 0)) nil
                                      (catch Throwable e (class e))))]
      (is (= expected (exception-class @source)))
      (is (= expected (exception-class gradient)))))
  (let [gradient (rev/value+grad #'inactive-initializer-order :wrt [0])]
    ;; Both initializers still execute when the first test is false.
    (is (= [2.0 1.0 nil nil] (gradient 2.0 1 (double-array [3.0]))))
    (is (= [8.0 12.0 nil nil] (gradient 2.0 3 (double-array [3.0]))))))

(deftest unvisited-writing-loop-fragments-are-not-certified-by-summary
  ;; Exercise the parsed phase boundary directly: surface normalization can
  ;; move a recur argument into the checked prelude, which is a different case.
  (doseq [[reason phase loop-form]
          [[:unclosed-loop-effect :test '(loop* [i 0 acc x]
                     (if (do (aset ys 0 7.0) (< i 2))
                       (recur (inc i) (* acc x)) acc))]
           ;; The shared step now visits inactive recur arguments. This raw
           ;; private entry lacks a declared ys port, so it must still refuse
           ;; the write rather than certifying it through a summary.
           [:ad-unprojectable-write-root nil '(loop* [i 0 acc x]
                               (if (< i 2)
                                 (recur (let* [s (identity ys) ignored (aset s 0 7.0)] (inc i))
                                        (* acc x)) acc))]]]
    (with-bindings {#'rev/*gensym-counter* (atom 0)}
      (let [failure (try (#'rev/gen-reverse-loop-program* loop-form '[x]) nil
                         (catch clojure.lang.ExceptionInfo e (ex-data e)))]
        (is (= reason (:reason failure)))
        (is (= phase (:phase failure)))))))

(deftest nested-pullback-replay-survives-parent-map-boundary
  (let [h 1.0e-5 call #(let [shared (double-array [3.0])] (nested-replay-then-write % shared shared))
        gradient (rev/value+grad #'nested-replay-then-write :wrt [0])
        ys (double-array [3.0]) zs (double-array [11.0])]
    (is (= 24.0 (call 2.0)))
    (is (< (Math/abs (- 36.0 (/ (- (call (+ 2.0 h)) (call (- 2.0 h))) (* 2.0 h)))) 1.0e-7))
    (is (= [24.0 36.0 nil nil] (gradient 2.0 ys zs)))
    (is (= [7.0] (vec zs)))
    ;; The old accepted answer for aliased ports was [24,84,...]. The
    ;; caller's input must not be modified before the inherited fence fires.
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 ys ys) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [3.0] (vec ys)))))

(deftest lawful-earlier-writes-and-saved-scalar-observations-remain-valid
  (let [ys (double-array [3.0])]
    (is (= [56.0 84.0 nil] ((rev/value+grad #'write-before-nested-replay :wrt [0]) 2.0 ys)))
    (is (= [7.0] (vec ys))))
  (let [ys (double-array [3.0])]
    (is (= [24.0 36.0 nil] ((rev/value+grad #'scalar-capture-then-write :wrt [0]) 2.0 ys)))
    (is (= [7.0] (vec ys)))))

(deftest emitted-pullback-replays-are-not-all-forward-record-replays
  (let [sym (with-meta 'value {:raster.type/tag 'double})
        record {:type :probe :sym sym :replays '#{input}}]
    (with-bindings {#'rev/*gensym-counter* (atom 0)}
      (with-redefs-fn {#'rev/emit-backward (fn [_ _ _ context] context)}
        #(do
           (is (= #{} (:pullback-replays (#'rev/reverse-pass [record] {sym true} {}))))
           (is (= '#{input} (:pullback-replays (#'rev/reverse-pass [record] {sym true} {sym ['dy]}))))
           ;; Unused output writers still execute in the primal, but have no
           ;; backward demand merely because they declare a buffer write.
           (is (= #{} (:pullback-replays
                              (#'rev/reverse-pass [(assoc record :written-arrs '[out])]
                                                 {sym true} {})))))))))

(deftest checked-child-effects-do-not-erase-parent-output-stability
  (let [record {:written-arrs '[out]
                :effects {:kind :region :regions [{:writes '#{private out} :replays '#{input}
                                                   :value-facts {'private {:value 'private}}}]}}
        effects (#'rev/record-memory-effects record '(raster.par/map! out i 2 double (aget out 0)))]
    (is (= '#{private out} (:writes effects)))
    (is (= #{} (:local-writes effects)))
    (is (= '#{out} (:local-reads effects)))
    (is (contains? (:value-facts effects) 'private))))
