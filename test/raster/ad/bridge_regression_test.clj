(ns raster.ad.bridge-regression-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]
            [raster.sci.distributions :as dist]))

(deftm normal-density-sum [mu :- Double, sigma :- Double,
                           ys :- (Array double), count :- Long] :- Double
  (loop [i 0 sum 0.0]
    (if (< i count)
      (recur (inc i)
             (n/+ sum (dist/logpdf (dist/->Normal mu sigma)
                                   (aget ys i))))
      sum)))

(deftm normal-density-reduce [mu :- Double, sigma :- Double,
                              ys :- (Array double), count :- Long] :- Double
  (par/reduce sum 0.0 i count
    (n/+ sum (dist/logpdf (dist/->Normal mu sigma) (aget ys i)))))

(deftm normal-density-seeded-loop [mu :- Double, sigma :- Double,
                                  ys :- (Array double), count :- Long] :- Double
  (let [prior (dist/logpdf (dist/->Normal 0.0 sigma) mu)]
    (loop [i 0 sum prior]
      (if (< i count)
        (recur (inc i)
               (n/+ sum (dist/logpdf (dist/->Normal mu sigma) (aget ys i))))
        sum))))

(deftm normal-density-seeded-reduce [mu :- Double, sigma :- Double,
                                    ys :- (Array double), count :- Long] :- Double
  (par/reduce sum (dist/logpdf (dist/->Normal 0.0 sigma) mu) i count
    (n/+ sum (dist/logpdf (dist/->Normal mu sigma) (aget ys i)))))

(deftm normal-prior-then-loop [mu :- Double, ys :- (Array double), count :- Long,
                              prior-sigma :- Double, observation-sigma :- Double] :- Double
  (let [prior (dist/logpdf (dist/->Normal 0.0 prior-sigma) mu)]
    (loop [i 0 sum prior]
      (if (< i count)
        (recur (inc i)
               (n/+ sum (dist/logpdf (dist/->Normal mu observation-sigma)
                                     (aget ys i))))
        sum))))

(deftm normal-strided-observations [mu :- Double, ys :- (Array double), count :- Long,
                                   sigma :- Double] :- Double
  (loop [i 0 sum 0.0]
    (if (< i count)
      (recur (inc i)
             (n/+ sum (dist/logpdf (dist/->Normal mu sigma)
                                   (aget ys (* 2 i)))))
      sum)))

(deftm normal-strided-reduce [mu :- Double, ys :- (Array double), count :- Long,
                             sigma :- Double] :- Double
  (par/reduce sum 0.0 i count
    (n/+ sum (dist/logpdf (dist/->Normal mu sigma) (aget ys (* 2 i))))))

(deftm normal-strided-let-reduce [mu :- Double, ys :- (Array double), count :- Long,
                                 sigma :- Double] :- Double
  (par/reduce sum 0.0 i count
    (let [y (aget ys (* 2 i))]
      (n/+ sum (dist/logpdf (dist/->Normal mu sigma) y)))))

(deftm normal-strided-let-loop [mu :- Double, ys :- (Array double), count :- Long,
                               sigma :- Double] :- Double
  (loop [i 0 sum 0.0]
    (if (< i count)
      (let [y (aget ys (* 2 i))]
        (recur (inc i) (n/+ sum (dist/logpdf (dist/->Normal mu sigma) y))))
      sum)))

(deftest let-prelude-scan-lift-requires-purity
  (is (some? ((var rev/carry-loop->scan)
              '(loop* [i 0 sum 0.0]
                 (if (< i count)
                   (let* [y (* 2.0 i)]
                     (recur (inc i) (+ sum y)))
                   sum)))))
  (is (nil? ((var rev/carry-loop->scan)
             '(loop* [i 0 sum 0.0]
                (if (< i count)
                  (let* [y (aget ys i)]
                    (recur (inc i) (+ sum y)))
                  sum)))))
  (is (nil? ((var rev/carry-loop->scan)
             '(loop* [i 0 sum 0.0]
                (if (< i count)
                  (let* [y (do (aset touched i 1.0) (aget ys i))]
                    (recur (inc i) (+ sum y)))
                  sum))))))

(deftest selected-gradient-inputs-allow-strided-observations
  (let [ys (double-array [0.1 9.0 0.7 9.0 -0.3 9.0])
        expected (reduce + (map #(dist/logpdf (dist/->Normal 0.2 1.4) %)
                                [0.1 0.7 -0.3]))]
    (doseq [source [#'normal-strided-observations
                    #'normal-strided-reduce
                    #'normal-strided-let-reduce
                    #'normal-strided-let-loop]]
      (let [[value dmu dys dcount dsigma]
            ((rev/value+grad source :wrt [0 3]) 0.2 ys 3 1.4)]
        (is (< (Math/abs (- value expected)) 1e-10))
        (is (< (Math/abs (- dmu (reduce + (map #(/ (- % 0.2) (* 1.4 1.4))
                                                  [0.1 0.7 -0.3])))) 1e-9))
        (is (nil? dys))
        (is (nil? dcount))
        (is (number? dsigma))))
    (let [[dmu dys dcount dsigma]
          ((rev/grad #'normal-strided-reduce :wrt [0 3]) 0.2 ys 3 1.4)]
      (is (number? dmu))
      (is (nil? dys))
      (is (nil? dcount))
      (is (number? dsigma)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-differentiable"
                          (rev/value+grad #'normal-strided-observations :wrt [2])))))

(deftest generated-gradient-helper-namespace-is-available
  (is (some? (find-ns 'raster.dl.nn))))

(deftest constructed-normal-in-reduction-has-gradient
  (let [ys (double-array [0.1 0.7 -0.3])
        [value dmu dsigma _] ((rev/value+grad #'normal-density-sum)
                               0.2 1.4 ys 3)
        expected (reduce + (map #(dist/logpdf (dist/->Normal 0.2 1.4) %) ys))
        expected-dmu (reduce + (map #(/ (- % 0.2) (* 1.4 1.4)) ys))
        expected-dsigma
        (reduce + (map #(let [delta (- % 0.2)]
                          (+ (- (/ 1.0 1.4))
                             (/ (* delta delta) (* 1.4 1.4 1.4)))) ys))]
    (doseq [[f [v dm ds _]]
            [["lifted loop" [value dmu dsigma nil]]
             ["explicit reduce" ((rev/value+grad #'normal-density-reduce)
                                  0.2 1.4 ys 3)]]]
      (is (< (Math/abs (- v expected)) 1e-10) f)
      (is (< (Math/abs (- dm expected-dmu)) 1e-9) f)
      (is (< (Math/abs (- ds expected-dsigma)) 1e-9) f))))

(deftest constructed-normal-in-reduction-initializer-has-gradient
  (let [ys (double-array [0.1 0.7 -0.3])
        mu 0.2 sigma 1.4
        prior (dist/logpdf (dist/->Normal 0.0 sigma) mu)
        expected (+ prior (reduce + (map #(dist/logpdf (dist/->Normal mu sigma) %) ys)))
        expected-dmu (+ (- (/ mu (* sigma sigma)))
                        (reduce + (map #(/ (- % mu) (* sigma sigma)) ys)))
        expected-dsigma
        (reduce + (map #(let [delta (- % mu)]
                          (+ (- (/ 1.0 sigma))
                             (/ (* delta delta) (* sigma sigma sigma))))
                       (cons 0.0 ys)))]
    (doseq [[label source] [["let-bound loop initializer" #'normal-density-seeded-loop]
                            ["direct reduction initializer" #'normal-density-seeded-reduce]]]
      (let [[value dmu dsigma _] ((rev/value+grad source) mu sigma ys 3)]
        (is (< (Math/abs (- value expected)) 1e-10) label)
        (is (< (Math/abs (- dmu expected-dmu)) 1e-9) label)
        (is (< (Math/abs (- dsigma expected-dsigma)) 1e-9) label)))))

(deftest separate-prior-and-observation-scales-seed-loop
  (let [ys (double-array [0.1 0.7 -0.3])
        mu 0.2 s0 2.7 s 1.4
        expected (+ (dist/logpdf (dist/->Normal 0.0 s0) mu)
                    (reduce + (map #(dist/logpdf (dist/->Normal mu s) %) ys)))
        [value dmu _ dcount ds0 ds] ((rev/value+grad #'normal-prior-then-loop)
                                       mu ys 3 s0 s)]
    (is (< (Math/abs (- value expected)) 1e-10))
    (is (< (Math/abs (- dmu (+ (- (/ mu (* s0 s0)))
                               (reduce + (map #(/ (- % mu) (* s s)) ys))))) 1e-9))
    (is (nil? dcount))
    (is (< (Math/abs (- ds0 (+ (- (/ 1.0 s0))
                                 (/ (* mu mu) (* s0 s0 s0))))) 1e-9))
    (is (< (Math/abs (- ds (reduce + (map #(let [delta (- % mu)]
                                             (+ (- (/ 1.0 s))
                                                (/ (* delta delta) (* s s s)))) ys)))) 1e-9))))
