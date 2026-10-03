(ns raster.ad.bridge-regression-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]
            [raster.sci.distributions :as dist]))

(deftm constant-latent-loop [b :- (Array double), count :- Long] :- Double
  (loop [i 0 acc 0.0]
    (if (< i count)
      (recur (inc i) (n/+ acc (aget b 0)))
      acc)))

(deftm two-constant-latents-loop [b :- (Array double), count :- Long] :- Double
  (loop [i 0 acc 0.0]
    (if (< i count)
      (recur (inc i)
             (n/+ (n/* 0.5 acc)
                  (n/+ (n/* (ra/aget b 0) (ra/aget b 0))
                       (n/* 3.0 (ra/aget b 1)))))
      acc)))

(deftest active-literal-index-reads-in-carry-loops
  (let [simple (rev/value+grad #'constant-latent-loop :wrt [0])
        two (rev/value+grad #'two-constant-latents-loop :wrt [0])]
    (doseq [count [0 1 5]]
      (let [[value gradient dcount] (simple (double-array [2.5 9.0]) count)]
        (is (= (* 2.5 count) value))
        (is (= [(double count) 0.0] (vec gradient)))
        (is (nil? dcount)))
      (let [b (double-array [1.25 -0.4])
            [value gradient] (two b count)
            factor (* 2.0 (- 1.0 (Math/pow 0.5 count)))]
        (is (< (Math/abs (- value (* factor (+ (* 1.25 1.25) (* 3.0 -0.4))))) 1e-12))
        (is (= [(* factor 2.5) (* factor 3.0)] (vec gradient)))
        (doseq [index [0 1]]
          (let [plus (aclone b) minus (aclone b) h 1e-5]
            (aset-double plus index (+ (aget b index) h))
            (aset-double minus index (- (aget b index) h))
            (is (< (Math/abs (- (aget ^doubles gradient index)
                                (/ (- (two-constant-latents-loop plus count)
                                      (two-constant-latents-loop minus count)) (* 2.0 h)))) 1e-8))))))
    ;; No primal read exists on a zero-trip loop; do not hoist b[0] eagerly.
    (let [[value gradient] (simple (double-array 0) 0)]
      (is (= 0.0 value))
      (is (= [] (vec gradient))))))

(deftest literal-read-admission-is-sequential-and-scope-checked
  (let [analyze (fn [body & options]
                  (rev/call-with-shared-ad-gensym
                   #(apply (var rev/analyze-par-map-body) body 'i options)))
        body '(+ acc (+ (aget b 0) (+ (aget b 1) (aget b 0))))]
    (is (empty? (:agets (analyze body))) "parallel map retains its original inline-read rule")
    (is (= [['b 0] ['b 1]] (mapv (juxt :arr :idx) (:agets (analyze body true)))))
    (doseq [scoped ['(+ acc (if (< i 1) (aget b 0) 0.0))
                    '(+ acc (aget b offset))
                    '(+ acc (let* [local 1.0] (+ local (aget b 0))))
                    '(+ acc (loop* [j 0] (if (< j 1) (recur (inc j)) (aget b 0))))]]
      (is (empty? (:agets (analyze scoped true))))
      (is (contains? (:free-syms (analyze scoped true)) 'b)))
    (let [tagged (with-meta body {:raster.type/tag 'double :line 42})]
      (is (= (meta tagged) (meta (:body-result (analyze tagged true))))))
    ;; The analyzer strips this outer let. Reads of its aliases/shadows must
    ;; remain inline rather than being replayed before the bindings exist.
    (doseq [scoped ['(let* [local b x (+ (aget local 0) 1.0)] (+ acc x))
                    '(let* [local b] (+ acc (aget local 0)))
                    '(let* [b other] (+ acc (aget b 0)))]]
      (is (empty? (:agets (analyze scoped true)))))
    ;; An initializer still sees the outer array before a later shadow binds it.
    (is (= [['b 0]]
           (mapv (juxt :arr :idx)
                 (:agets (analyze '(let* [x (+ (aget b 0) 1.0) b other]
                                    (+ acc x)) true)))))))

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

(deftm normal-double-prior-reduce
  [mu :- Double, ys :- (Array double), count :- Long,
   prior-sigma :- Double, observation-sigma :- Double] :- Double
  (par/reduce sum
    (n/+ (dist/logpdf (dist/->Normal 0.0 prior-sigma) mu)
         (dist/logpdf (dist/->Normal 1.0 prior-sigma) mu))
    i count
    (n/+ sum (dist/logpdf (dist/->Normal mu observation-sigma) (aget ys i)))))

(deftm normal-prior-then-loop [mu :- Double, ys :- (Array double), count :- Long,
                              prior-sigma :- Double, observation-sigma :- Double] :- Double
  (let [prior (dist/logpdf (dist/->Normal 0.0 prior-sigma) mu)]
    (loop [i 0 sum prior]
      (if (< i count)
        (recur (inc i)
               (n/+ sum (dist/logpdf (dist/->Normal mu observation-sigma)
                                     (aget ys i))))
        sum))))

(deftm normal-prior-then-loop-raster-aget
  [mu :- Double, ys :- (Array double), count :- Long,
   prior-sigma :- Double, observation-sigma :- Double] :- Double
  (let [prior (dist/logpdf (dist/->Normal 0.0 prior-sigma) mu)]
    (loop [i 0 sum prior]
      (if (< i count)
        (recur (inc i)
               (n/+ sum (dist/logpdf (dist/->Normal mu observation-sigma)
                                     (ra/aget ys i))))
        sum))))

(deftm normal-direct-prior-loop
  [mu :- Double, ys :- (Array double), count :- Long,
   prior-sigma :- Double, observation-sigma :- Double] :- Double
  (loop [i 0 sum (dist/logpdf (dist/->Normal 0.0 prior-sigma) mu)]
    (if (< i count)
      (recur (inc i)
             (n/+ sum (dist/logpdf (dist/->Normal mu observation-sigma)
                                   (aget ys i))))
      sum)))

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

(deftest constructed-prior-loop-read-spellings-and-direct-initializer
  (let [ys (double-array [0.1 0.7 -0.3])
        expected ((rev/value+grad #'normal-prior-then-loop :wrt [0 3 4])
                  0.2 ys 3 2.7 1.4)]
    (doseq [source [#'normal-prior-then-loop-raster-aget
                    #'normal-direct-prior-loop]]
      (let [actual ((rev/value+grad source :wrt [0 3 4]) 0.2 ys 3 2.7 1.4)]
        (is (< (Math/abs (- (first actual) (first expected))) 1e-10))
        (is (nil? (nth actual 2)))
        (is (nil? (nth actual 3)))
        (doseq [i [1 4 5]]
          (is (< (Math/abs (- (nth actual i) (nth expected i))) 1e-9)))))))

(deftest constructed-compound-reduction-initializer-has-gradient
  (let [ys (double-array [0.1 0.7 -0.3])
        mu 0.2 s0 2.7 s 1.4
        expected (+ (dist/logpdf (dist/->Normal 0.0 s0) mu)
                    (dist/logpdf (dist/->Normal 1.0 s0) mu)
                    (reduce + (map #(dist/logpdf (dist/->Normal mu s) %) ys)))
        [value dmu dys dcount ds0 ds]
        ((rev/value+grad #'normal-double-prior-reduce :wrt [0 3 4]) mu ys 3 s0 s)]
    (is (< (Math/abs (- value expected)) 1e-10))
    (is (nil? dys))
    (is (nil? dcount))
    (is (< (Math/abs (- dmu (+ (/ (- 1.0 (* 2.0 mu)) (* s0 s0))
                               (reduce + (map #(/ (- % mu) (* s s)) ys))))) 1e-9))
    (is (< (Math/abs (- ds0 (+ (- (/ 2.0 s0))
                                 (/ (+ (* mu mu) (* (- mu 1.0) (- mu 1.0)))
                                    (* s0 s0 s0))))) 1e-9))
    (is (< (Math/abs (- ds (reduce + (map #(let [delta (- % mu)]
                                             (+ (- (/ 1.0 s))
                                                (/ (* delta delta) (* s s s))))
                                          ys)))) 1e-9))))
