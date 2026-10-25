(ns raster.compiler.passes.scalar.bare-value-grad-inline-test
  "A deftm whose whole body is `((value+grad #'f) args...)` (or `grad`) has its gradient
   inlined at compile time, through the same preparation as a let-bound value+grad.
   Covers: no runtime AD preparation in the compiled wrapper, repeated calls, the
   preparation lifecycle, redefinition of `f`, and an owned (JVM bytecode) preparation.
   Needs no native library."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as a]
            [raster.par :as p]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pipeline]))

(def ^:private this-ns (the-ns 'raster.compiler.passes.scalar.bare-value-grad-inline-test))

(defn- define-loss!
  "(Re)define `vgw-loss` at the REPL: :square → w·Σx², :cubic → w·Σx³."
  [kind]
  (binding [*ns* this-ns]
    (eval (list 'deftm 'vgw-loss '[x :- (Array double) w :- Double] :- 'Double
                (list 'p/reduce 'acc 0.0 'i '(a/alength x)
                      (list 'n/+ 'acc
                            (case kind
                              :square '(n/* w (n/* (a/aget x i) (a/aget x i)))
                              :cubic '(n/* (n/* w (a/aget x i)) (n/* (a/aget x i) (a/aget x i))))))))))

(define-loss! :square)

(deftm vgw-bare [x :- (Array double) w :- Double] :- Object
  ((rev/value+grad (var vgw-loss)) x w))

(deftm vgw-let [x :- (Array double) w :- Double] :- Object
  (let [result ((rev/value+grad (var vgw-loss)) x w)] result))

(deftm gw-bare [x :- (Array double) w :- Double] :- Object
  ((rev/grad (var vgw-loss)) x w))

;; A loss whose reverse program is owned by its JVM bytecode emission
;; (prepare-value+grad returns a :source-owner).
(deftm owned-response [amplitude :- Double kappa :- Double exposure :- Double] :- Double
  (n/* amplitude (n// exposure (n/+ kappa exposure))))

(deftm owned-factor [theta :- (Array double) data :- (Array double) cnt :- Long] :- Double
  (let [amplitude (a/aget theta 0) kappa (a/aget theta 1)]
    (n/* 2.0 (owned-response amplitude kappa
                             (p/reduce acc 0.0 i cnt
                                       (n/+ acc (owned-response amplitude kappa (a/aget data i))))))))

(deftm owned-vg-bare [theta :- (Array double) data :- (Array double) cnt :- Long] :- Object
  ((rev/value+grad (var owned-factor) :wrt [0]) theta data cnt))

(defn- plain [result]
  (mapv #(cond (nil? %) nil (number? %) (double %) :else (vec %)) result))

(defn- close? [actual expected]
  (let [actual (flatten (plain actual)) expected (flatten (plain expected))]
    (and (= (count actual) (count expected))
         (every? true? (map (fn [x y] (if (or (nil? x) (nil? y))
                                        (= x y)
                                        (<= (Math/abs (- (double x) (double y)))
                                            (* 1e-12 (max 1.0 (Math/abs (double y)))))))
                            actual expected)))))

(defn- inputs [seed]
  (let [r (java.util.Random. seed)]
    [(double-array (repeatedly (+ 3 seed) #(.nextGaussian r))) (+ 0.5 (.nextDouble r))]))

(defn- counting-preparations
  "Run `thunk` counting calls of rev/prepare-value+grad; returns [result count]."
  [thunk]
  (let [calls (atom 0) original rev/prepare-value+grad]
    (with-redefs [rev/prepare-value+grad (fn [& args] (swap! calls inc) (apply original args))]
      (let [result (thunk)] [result @calls]))))

(defn- refusing-runtime-ad [thunk]
  (with-redefs [rev/value+grad (fn [& _] (throw (ex-info "runtime value+grad" {})))
                rev/prepare-value+grad (fn [& _] (throw (ex-info "runtime AD preparation" {})))]
    (thunk)))

(deftest bare-wrapper-is-inlined-at-compile-time
  (doseq [[wrapper reference] [[#'vgw-bare (rev/value+grad #'vgw-loss)]
                               [#'gw-bare (rev/grad #'vgw-loss)]]]
    (testing (str wrapper)
      (let [f (pipeline/compile-aot wrapper)
            [x w] (inputs 1)]
        (is (close? (refusing-runtime-ad #(f x w)) (reference x w))
            "the compiled wrapper never re-enters runtime AD")))))

(deftest bare-wrapper-matches-let-bound-wrapper-on-repeated-calls
  (let [[bare preparations] (counting-preparations #(pipeline/compile-aot #'vgw-bare))
        let-bound (pipeline/compile-aot #'vgw-let)
        reference (rev/value+grad #'vgw-loss)
        argument-lists (mapv inputs (range 8))
        [results call-preparations]
        (counting-preparations #(doall (for [_ (range 3) args argument-lists] (apply bare args))))]
    (testing "preparation lifecycle: prepared while compiling, never while calling"
      (is (pos? preparations))
      (is (zero? call-preparations)))
    (testing "every call (inspected after all calls) has its own, correct result storage"
      (is (every? true? (map (fn [result args] (close? result (apply reference args)))
                             results (cycle argument-lists))))
      (is (every? true? (map (fn [args] (close? (apply bare args) (apply let-bound args)))
                             argument-lists))))))

(deftest redefinition-of-the-differentiated-function
  (let [[x w] (inputs 2)]
    (try
      (let [old-bare (pipeline/compile-aot #'vgw-bare)
            old-let (pipeline/compile-aot #'vgw-let)
            before (old-bare x w)
            _ (vgw-bare x w)
            _ (define-loss! :cubic)
            expected ((rev/value+grad #'vgw-loss) x w)]
        (is (not (close? before expected)) "the redefinition changes the gradient")
        (testing "an already compiled wrapper is a snapshot, exactly like a let-bound wrapper"
          (is (close? (old-bare x w) before))
          (is (close? (old-bare x w) (old-let x w))))
        (testing "calling the wrapper deftm directly sees the new definition"
          (is (close? (vgw-bare x w) expected)))
        (testing "recompiling re-prepares the AD program from the new definition"
          (let [[recompiled preparations] (counting-preparations #(pipeline/compile-aot #'vgw-bare))]
            (is (pos? preparations))
            (is (close? (refusing-runtime-ad #(recompiled x w)) expected))
            (is (close? (recompiled x w) ((pipeline/compile-aot #'vgw-let) x w))))))
      (finally
        (define-loss! :square)))))

(deftest owned-preparation-is-projected-into-the-bare-wrapper
  (owned-response 0.8 0.7 1.0)
  (is (= :jvm-bytecode (get-in (rev/prepare-value+grad #'owned-factor [0]) [:source-owner :backend])))
  (let [f (pipeline/compile-aot #'owned-vg-bare)
        reference (rev/value+grad #'owned-factor :wrt [0])
        data (double-array [0.2 0.8 1.1])]
    (doseq [[amplitude kappa] [[0.8 0.7] [1.2 0.4]]
            cnt [0 1 3]]
      (let [theta (double-array [amplitude kappa])]
        (is (close? (refusing-runtime-ad #(f theta data cnt)) (reference theta data cnt))
            [amplitude kappa cnt])))))
