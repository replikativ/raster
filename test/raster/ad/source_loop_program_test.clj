(ns raster.ad.source-loop-program-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftest internal-ad-ports-follow-code-not-activity-map-order
  (let [parameters (mapv #(with-meta (symbol (str "p" %))
                           {:raster.type/tag 'double :port %}) (range 16))
        activity (zipmap parameters (repeat true))
        expression (list 'n/+ (nth parameters 11) (nth parameters 2)
                         (nth parameters 15) (nth parameters 11))
        expected (mapv parameters [11 2 15])
        ports (#'rev/ordered-active-inputs expression activity)]
    (is (= expected ports))
    (is (= (mapv meta expected) (mapv meta ports)))
    (is (= ports (#'rev/ordered-active-inputs expression
                  (into {} (reverse (seq activity))))))
    (let [renaming (zipmap parameters (mapv #(symbol (str "fresh_" %)) (range 16)))
          renamed (apply list 'n/+ (map renaming (rest expression)))]
      (is (= (mapv renaming ports)
             (#'rev/ordered-active-inputs renamed
               (into {} (map (fn [[p active]] [(renaming p) active]) activity))))))
    (with-redefs-fn {#'rev/gen-reverse-par-reduce
                    (fn [_ active] {:active-free active})}
      #(is (= expected (get-in (#'rev/ad-record :par-reduce 'result expression activity)
                              [:record :active-free]))))))

(deftest internal-ad-ports-respect-lexical-shadowing-and-initializers
  (let [activity {'p true 'q true 'r true 'count true 'inactive false}
        order #(#'rev/ordered-active-inputs % activity)]
    (is (= '[q r] (order '(let* [p q q r] (n/+ p q) p))))
    ;; p in the initializer is external; the body p is loop-local.
    (is (= '[p q] (order '(loop* [p p q q] (if q (recur p q) p)))))
    (is (= '[q count] (order '(fn* p [p] (n/+ p q count)))))
    (is (= '[q] (order '(fn* [p] (fn* [r] (n/+ p r q))))))
    (is (= '[r q] (order '(do (quote [p count]) (n/+ r inactive q)))))
    (is (= '[count p] (order '(if condition (n/* count count) (n/+ p count)))))))

(deftm deterministic-port-density
  [x :- Double y :- Double values :- (Array double) rows :- Long] :- Double
  (par/reduce acc 0.0 i rows
    (n/+ acc (n/+ (n/* y (aget values i)) (n/* x x)))))

(deftest deterministic-port-layout-preserves-public-gradient-slots
  (let [gradient (rev/value+grad #'deterministic-port-density)
        selected (rev/value+grad #'deterministic-port-density :wrt [1])
        values (double-array [1.0 3.0 5.0])]
    (is (= [30.0 12.0 9.0] (subvec (vec (gradient 2.0 2.0 values 3)) 0 3)))
    (is (= [30.0 nil 9.0 nil nil] (vec (selected 2.0 2.0 values 3))))))

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
  (doseq [[phase loop-form]
          [[:test '(loop* [i 0 acc x]
                     (if (do (aset ys 0 7.0) (< i 2))
                       (recur (inc i) (* acc x)) acc))]
           [:inactive-recur '(loop* [i 0 acc x]
                               (if (< i 2)
                                 (recur (let* [s (identity ys) ignored (aset s 0 7.0)] (inc i))
                                        (* acc x)) acc))]]]
    (with-bindings {#'rev/*gensym-counter* (atom 0)}
      (let [failure (try (#'rev/gen-reverse-loop-program* loop-form '[x]) nil
                         (catch clojure.lang.ExceptionInfo e (ex-data e)))]
        (is (= :unclosed-loop-effect (:reason failure)))
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
