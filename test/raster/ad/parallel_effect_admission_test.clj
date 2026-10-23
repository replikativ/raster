(ns raster.ad.parallel-effect-admission-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.ad.templates :as tmpl]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm active-effect-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i (n/* x x)))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm core-spelled-active-effect-square [count :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i (n/* count count)))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm inactive-effect-square [h :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i 4.0))
    (n/+ h (par/reduce total 0.0 i size (n/+ total (ra/aget out i))))))

(deftm excluded-effect-square [h :- Double x :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i (n/* x x)))
    (n/+ h (par/reduce total 0.0 i size (n/+ total (ra/aget out i))))))

(deftm supported-map-reduce-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size double (n/* x x))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm supported-scan-sum [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        scanned (par/scan out acc 0.0 i size double (n/+ acc x))]
    (ra/aget scanned (dec size))))

(deftm supported-template-scaled-dot
  [alpha :- Double a :- (Array double) b :- (Array double)] :- Double
  (par/dot-product (par/scale alpha a) b))

(deftest proved-fresh-parallel-effects-preserve-captured-derivatives
  ;; Independent polynomial/central-difference witness. No generated function
  ;; may return the historically wrong [12, 0] for this source computation.
  (doseq [source [#'active-effect-square #'core-spelled-active-effect-square]]
    (let [x 2.0 size 3 h 0.125
          expected (* size x x)
          derivative (* 2.0 size x)]
      (is (= expected (source x size)))
      (is (= derivative (/ (- (source (+ x h) size) (source (- x h) size))
                           (* 2.0 h))))
      (is (= [expected derivative nil] ((rev/value+grad source :wrt [0]) x size)))
      (is (= [expected derivative] ((jvp/jvp source) x size 1.0))))))

(deftm unsupported-multiple-write-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size
      (do (ra/aset out i (n/* x x))
          (ra/aset out i (n/+ x 1.0))))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm unsupported-rounded-write [x :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i (float x)))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftest unproved-parallel-effects-still-refuse-in-both-modes
  (is (= 9.0 (unsupported-multiple-write-square 2.0 3)))
  (is (= 16777216.0 (unsupported-rounded-write 16777217.0 1)))
  (doseq [construct [rev/value+grad jvp/jvp]
          source [#'unsupported-multiple-write-square #'unsupported-rounded-write]]
    (is (= :ad-unsupported-active-parallel-region
           (try (construct source) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftm unsupported-direct-contract [x :- Double] :- Double
  (let [out (double-array 1)]
    ;; The linter has no hook for this primitive's axis binders.
    #_{:clj-kondo/ignore [:unresolved-symbol]}
    (par/contract out [[s 1]] [[i 2]] (n/* x x))
    (ra/aget out 0)))

(deftm unsupported-direct-product-reduce [x :- Double] :- Double
  (let [out (double-array 1)]
    ;; Component/combine binders are scoped by the compiler primitive.
    #_{:clj-kondo/ignore [:unresolved-symbol]}
    (par/product-reduce! [out] [[acc 0.0 :double]] [[s 1]] i 2
      [] [(n/* x x)] [[left right]] [] [(n/+ left right)] {})
    (ra/aget out 0)))

(deftm unsupported-direct-segmented-fold-map [x :- Double] :- Double
  (let [out (double-array 1)]
    (par/segmented-fold-map! [out] [[s 1]] i 1
      [[acc 0.0 :double 2 (n/+ acc (n/* x x))]] [acc])
    (ra/aget out 0)))

(deftm unsupported-direct-exclusive-scan [x :- Double] :- Double
  (let [out (double-array 3)]
    (par/scan-exclusive out acc 0.0 _i 2 double (n/+ acc (n/* x x)))
    (ra/aget out 2)))

(deftm unsupported-direct-scatter-add
  [x :- Double indices :- (Array long)] :- Double
  (let [src (par/map [i 2] (n/* x x))
        out (double-array 1)]
    (par/scatter! out src indices 2)
    (ra/aget out 0)))

(deftest direct-combinator-ad-capabilities-refuse-unsupported-active-regions
  ;; Primal compiler support does not establish an AD capability. Every case is
  ;; the nonconstant polynomial 2*x*x, so accepting a zero derivative is wrong.
  ;; When a scoped derivative is implemented, update that case's capability
  ;; expectation and exercise its analytic derivative (12) / seeded JVP (3),
  ;; rather than weakening this refusal into acceptance of an arbitrary result.
  (doseq [[combinator source extra-arguments]
          [[:contract #'unsupported-direct-contract []]
           [:product-reduce #'unsupported-direct-product-reduce []]
           [:segmented-fold-map #'unsupported-direct-segmented-fold-map []]
           [:exclusive-scan #'unsupported-direct-exclusive-scan []]
           [:scatter-add #'unsupported-direct-scatter-add [(long-array [0 0])]]]]
    (testing (name combinator)
      (let [primal #(apply source % extra-arguments)
            x 3.0 h 0.125]
        (is (= 18.0 (primal x)))
        (is (= 12.0 (/ (- (primal (+ x h)) (primal (- x h))) (* 2.0 h))))))
    (doseq [[mode construct] [[:reverse rev/value+grad] [:jvp jvp/jvp]]]
      (testing (str (name combinator) " " (name mode) " construction")
        ;; Refusal must occur while constructing the derivative: there is no
        ;; derivative invocation here whose failure could disguise admission.
        (is (= :ad-unsupported-active-parallel-region
               (try (construct source :wrt [0]) nil
                    (catch clojure.lang.ExceptionInfo error
                      (:reason (ex-data error))))))))))

(deftm fresh-gather-sum
  [src :- (Array double) indices :- (Array long) size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size
      (let [index (ra/aget indices i)]
        (ra/aset out i (ra/aget src index))))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm fresh-map-with-unused-source-read
  [x :- Double data :- (Array double) size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size
      (let [ignored (ra/aget data i)]
        (ra/aset out i (n/* x x))))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftest map-derivation-keeps-nonremovable-pure-source-evaluations
  (let [prepared (rev/prepare-value+grad #'fresh-map-with-unused-source-read [0])
        reverse (rev/value+grad #'fresh-map-with-unused-source-read :wrt [0])
        forward (jvp/jvp #'fresh-map-with-unused-source-read :wrt [0])
        data (double-array [7.0 8.0])]
    (is (some #(and (seq? %) (= 'clojure.core/aget (first %))
                    (:raster.effect/preserve-evaluation (meta %)))
              (tree-seq coll? seq (:walked-body prepared)))
        "source read obligations survive AD map canonicalization")
    (is (= [18.0 12.0 nil nil] (reverse 3.0 data 2)))
    (is (= [18.0 12.0] (forward 3.0 data 2 1.0)))
    (doseq [execute [fresh-map-with-unused-source-read reverse
                     (fn [x data size] (forward x data size 1.0))]]
      (is (thrown? ArrayIndexOutOfBoundsException (execute 3.0 (double-array 0) 1)))
      (is (thrown? NegativeArraySizeException (execute 3.0 data -1))))))

(deftest fresh-gather-write-transposes-repeated-indices
  (let [src (double-array [2.0 3.0 5.0])
        indices (long-array [2 0 2])
        tangent (double-array [0.25 -0.5 2.0])
        original (vec src)
        expected (reduce + 0.0 (map #(nth original %) indices))
        ;; Independent scatter-sum: the repeated index must accumulate twice.
        gradient (mapv #(double (count (filter #{%} indices))) (range (count src)))
        [v g _ _] ((rev/value+grad #'fresh-gather-sum :wrt [0]) src indices 3)
        [jv jg] ((jvp/jvp #'fresh-gather-sum) src indices 3 tangent)
        h 0.125]
    (is (= 12.0 expected (fresh-gather-sum src indices 3) v jv))
    (is (= gradient (vec g)))
    (is (= 4.25 jg (reduce + 0.0 (map * gradient tangent))))
    (dotimes [coordinate (count src)]
      (let [plus (double-array src) minus (double-array src)]
        (aset plus coordinate (+ (aget plus coordinate) h))
        (aset minus coordinate (- (aget minus coordinate) h))
        (is (= (nth gradient coordinate)
               (/ (- (fresh-gather-sum plus indices 3) (fresh-gather-sum minus indices 3))
                  (* 2.0 h))))))
    (is (= original (vec src)))
    (is (= [2 0 2] (vec indices)))
    (is (= [0.0 [0.0 0.0 0.0]]
           (let [[v g] ((rev/value+grad #'fresh-gather-sum :wrt [0])
                        src (long-array 0) 0)] [v (vec g)])))))

(deftm direct-map-local-gather-chain
  [src :- (Array double) indices :- (Array long) size :- Long index :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size nil
                        (let [index (ra/aget indices i)
                              alias index
                              value (ra/aget src alias)]
                          value))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm direct-map-mutated-local-gather-index
  [src :- (Array double) indices :- (Array long) size :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size nil
                        (let [index (ra/aget indices i)] (ra/aget src index)))
        overwritten (par/map! indices j size long 0)]
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm additive-local-index
  [src :- (Array double) weights :- (Array double) size :- Long] :- Double
  (par/reduce total 0.0 i size
    (let [offset (* 2 size) index (+ offset i) value (ra/aget src index)]
      (n/+ total (n/* (ra/aget weights i) value)))))

(deftm ordered-local-index
  [src :- (Array double) weights :- (Array double) size :- Long] :- Double
  (par/reduce total 0.0 i size
    (let [offset (* 2 size) index (+ offset i) value (ra/aget src index)]
      (n/+ (n/* 0.5 total) (n/* (ra/aget weights i) value)))))

(deftm scan-local-index
  [src :- (Array double) size :- Long] :- Double
  (let [out (double-array size)
        scanned (par/scan out acc 0.0 i size double
                         (let [offset (* 2 size) index (+ offset i)
                               value (ra/aget src index)]
                           (n/+ acc value)))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget scanned i)))))

(deftest reduction-and-scan-replay-local-index-dependencies
  (let [src (double-array [1 2 3 4 5 6]) weights (double-array [2 3])]
    (doseq [[source expected gradient] [[#'additive-local-index 28.0 [0.0 0.0 0.0 0.0 2.0 3.0]]
                                      [#'ordered-local-index 23.0 [0.0 0.0 0.0 0.0 1.0 3.0]]]]
      (let [vg (rev/value+grad source :wrt [0])
            [value derivative] (vg src weights 2)]
        (is (= expected (source src weights 2) value))
        (is (= gradient (vec derivative)))
        (doseq [j (range (alength src))]
          (let [left (aclone src) right (aclone src) h 1.0e-6]
            (aset-double left j (- (aget left j) h))
            (aset-double right j (+ (aget right j) h))
            (is (< (Math/abs (- (aget ^doubles derivative j)
                                (/ (- (source right weights 2) (source left weights 2)) (* 2 h))))
                   1.0e-7))))
        (is (= [0.0 []] (let [[v g] (vg (double-array 0) (double-array 0) 0)] [v (vec g)])))))
    (let [vg (rev/value+grad #'scan-local-index :wrt [0])
          [value derivative] (vg src 2)]
      (is (= 16.0 (scan-local-index src 2) value))
      (is (= [0.0 0.0 0.0 0.0 2.0 1.0] (vec derivative)))
      (is (= [0.0 []] (let [[v g] (vg (double-array 0) 0)] [v (vec g)]))))))

(deftest unproved-lifted-index-dependencies-decline-before-emission
  (doseq [body ['(let* [index (unknown-index i) value (clojure.core/aget src index)] value)
                '(let* [inc (unknown-index i) value (clojure.core/aget src inc)] value)
                '(let* [count (unknown-index i) value (clojure.core/aget src count)] value)
                '(let* [buf src value (clojure.core/aget buf i)] value)
                '(let* [buf src] (raster.numeric/* 2.0 (clojure.core/aget buf i)))
                '(let* [value (let* [buf src] (clojure.core/aget buf i))] value)
                '(raster.numeric/+ acc (let* [buf src] (clojure.core/aget buf i)))
                '(let* [index (clojure.core/+ acc i) value (clojure.core/aget src index)] value)
                '(let* [index (clojure.core/aget out i) value (clojure.core/aget src index)] value)]]
    (is (= :ad-unclosed-lifted-read-index
         (try (#'rev/analyze-par-map-body
               body 'i true #{'src 'acc 'out})
              nil
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        (str body))))

(deftest lifted-read-receivers-respect-sequential-shadowing
  (doseq [body ['(let* [value (clojure.core/aget src i) src other] value)
                '(let* [src (clojure.core/aget src i)] src)]]
    (let [analyzed (#'rev/analyze-par-map-body body 'i true)]
      (is (= [['src 'i]] (mapv (juxt :arr :idx) (:agets analyzed)))))))

(deftm local-index-check-before-read
  [src :- (Array double) size :- Long] :- Double
  (par/reduce total 0.0 i size
    (let [checked (quot 1 (- size 1)) index (+ (* 2 size) i)
          value (ra/aget src index)]
      (n/+ total (n/* (double checked) value)))))

(deftm local-index-check-after-read
  [src :- (Array double) size :- Long] :- Double
  (par/reduce total 0.0 i size
    (let [index (+ (* 2 size) i) value (ra/aget src index)
          checked (quot 1 (- size 1))]
      (n/+ total (n/* (double checked) value)))))

(deftest replay-closure-keeps-forward-exception-order-and-zero-trips
  (doseq [[source expected] [[#'local-index-check-before-read ArithmeticException]
                            [#'local-index-check-after-read ArrayIndexOutOfBoundsException]]]
    (let [vg (rev/value+grad source :wrt [0])
          failure (fn [f] (try (f (double-array 0) 1) nil (catch Throwable error (class error))))]
      (is (= expected (failure source) (failure vg)))
      (is (= [0.0 []] (let [[v g] (vg (double-array 0) 0)] [v (vec g)]))))))

(deftest unrecognized-inline-index-stays-in-its-original-scope
  (let [body '(let* [count (unknown-index i)] (clojure.core/aget src count))
        analyzed (#'rev/analyze-par-map-body body 'i true #{'src})]
    (is (empty? (:agets analyzed)))
    (is (= '([count (unknown-index i)]) (:scalar-bindings analyzed)))
    (is (= '(clojure.core/aget src count) (:body-result analyzed)))))

(deftest existing-functional-map-replays-pure-index-aliases-without-capture
  (let [src (double-array [2.0 3.0 5.0]) indices (long-array [2 0 2])
        tangent (double-array [0.25 -0.5 2.0])
        ;; The outer parameter is deliberately out of bounds: the inner lexical
        ;; index must win, and both local aliases must reach the scatter transpose.
        [value gradient] ((rev/value+grad #'direct-map-local-gather-chain :wrt [0])
                          src indices 3 99)
        [jvalue derivative] ((jvp/jvp #'direct-map-local-gather-chain)
                              src indices 3 99 tangent)]
    (is (= 12.0 value jvalue (direct-map-local-gather-chain src indices 3 99)))
    (is (= [1.0 0.0 2.0] (vec gradient)))
    (is (= 4.25 derivative))
    (is (= [2.0 3.0 5.0] (vec src)))
    (is (= [2 0 2] (vec indices)))
    (is (= [0.0 [0.0 0.0 0.0]]
           (let [[value gradient] ((rev/value+grad #'direct-map-local-gather-chain :wrt [0])
                                   src (long-array 0) 0 99)] [value (vec gradient)])))))

(deftest replayed-local-index-keeps-existing-source-mutation-protection
  (is (= :replayed-array-overwritten
         (try (rev/value+grad #'direct-map-mutated-local-gather-index :wrt [0]) nil
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

(deftest empty-fresh-square-write-remains-an-empty-primal-and-tangent
  (is (= 0.0 (active-effect-square 2.0 0)))
  (is (= [0.0 0.0 nil] ((rev/value+grad #'active-effect-square :wrt [0]) 2.0 0)))
  (is (= [0.0 0.0] ((jvp/jvp #'active-effect-square) 2.0 0 1.0))))

(deftest inactive-and-excluded-parallel-effects-remain-primal
  (is (= 14.0 (inactive-effect-square 2.0 3)))
  (is (= [14.0 1.0 nil] ((rev/value+grad #'inactive-effect-square :wrt [0]) 2.0 3)))
  (is (= [14.0 1.0] ((jvp/jvp #'inactive-effect-square) 2.0 3 1.0)))
  (is (= 17.0 (excluded-effect-square 5.0 2.0 3)))
  (is (= [17.0 1.0 nil nil]
         ((rev/value+grad #'excluded-effect-square :wrt [0]) 5.0 2.0 3))))

(deftest existing-scoped-map-reduce-and-scan-rules-remain-admitted
  (doseq [[source expected derivative] [[#'supported-map-reduce-square 12.0 12.0]
                                       [#'supported-scan-sum 6.0 3.0]]]
    (is (= expected (source 2.0 3)))
    (is (= [expected derivative nil] ((rev/value+grad source :wrt [0]) 2.0 3)))
    (is (= [expected derivative] ((jvp/jvp source) 2.0 3 1.0)))))

(deftest declared-parallel-call-templates-remain-differentiable
  (let [a (double-array [1.0 -2.0])
        b (double-array [3.0 4.0])
        seed (double-array [0.5 -1.0])
        dot -5.0
        gradient (rev/reified-grad #'supported-template-scaled-dot 'a)
        [value dalpha da db] ((rev/value+grad #'supported-template-scaled-dot) 2.0 a b)
        [tvalue tangent] ((jvp/jvp #'supported-template-scaled-dot)
                         2.0 a b 1.0 seed (double-array 2))
        [gvalue dc dga dgb dseed] ((rev/value+grad gradient) 2.0 a b seed)]
    ;; Independent coordinates, not an AD-generated comparison oracle.
    (is (= (* 2.0 dot) value tvalue))
    (is (= dot dalpha))
    (is (= [6.0 8.0] (vec da)))
    (is (= [2.0 -4.0] (vec db)))
    (is (= -10.0 tangent))
    ;; <gradient_a, seed> = alpha * sum_i b_i*seed_i.
    (is (= -5.0 gvalue))
    (is (= -2.5 dc))
    (is (or (nil? dga) (= [0.0 0.0] (vec dga))))
    (is (= [1.0 -2.0] (vec dgb)))
    (is (= [6.0 8.0] (vec dseed)))))

(deftest parallel-template-admission-respects-call-identity-and-mode
  (let [classify (ns-resolve 'raster.ad.reverse 'ad-form-kind)
        fold (ns-resolve 'raster.ad.jvp 'jvp-fold)
        expression '(raster.par/dot-product a b)]
    (is (= :call (classify expression true 'result)))
    (let [masked (with-meta expression {:raster.op/lexical-callee true})]
      (is (= :ad-unsupported-active-parallel-region
             (try (classify masked true 'result) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    ;; Direct calls derive identity from their actual head, not dispatch-only
    ;; advisory metadata. Withdrawal of the rule must restore the refusal.
    (with-redefs [tmpl/resolve-template (constantly nil)]
      (is (= :ad-unsupported-active-parallel-region
             (try (classify expression true 'result) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    ;; A reverse template alone must never authorize a forward zero tangent.
    (with-redefs [tmpl/op-jvp-fn (constantly nil)]
      (is (= :ad-unsupported-active-parallel-region
             (try (fold ['result expression] {'a 'da}) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
