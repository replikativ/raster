(ns raster.ad.map-store-callee-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.par :as par]))

(defn foreign-store [v] (n/* v v))

(deftm lexical-store-square [x :- Double size :- Long] :- Double
  (let [double (fn [v] (n/* v v))
        out (double-array size)
        mapped (par/map! out i size double x)]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm foreign-store-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size raster.ad.map-store-callee-test/foreign-store x)]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm captured-store-square [x :- Double size :- Long] :- Double
  (let [double (fn [_] (n/* x x))
        out (double-array size)
        mapped (par/map! out i size double 1.0)]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm core-store-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size double (n/* x x))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm qualified-core-store-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size clojure.core/double (n/* x x))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm nil-store-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size nil (n/* x x))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm float-store-square [x :- Double size :- Long] :- Double
  (let [out (float-array size)
        mapped (par/map! out i size float (n/* x x))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm repeated-float-reduce [xs :- (Array float) size :- Long] :- Double
  (par/reduce total (double 0.0) i size
    (n/+ total (double (ra/aget xs (rem i 2))))))

(deftm repeated-double-reduce [xs :- (Array double) size :- Long] :- Double
  (par/reduce total (double 0.0) i size
    (n/+ total (double (ra/aget xs (rem i 2))))))

(deftm float-read-dotimes [xs :- (Array float) size :- Long] :- Double
  (let [out (double-array size)]
    (dotimes [i size]
      (let [a (double (ra/aget xs i)) b (double (ra/aget xs i))]
        (ra/aset out i (n/* a b))))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm inactive-lexical-store [h :- Double size :- Long] :- Double
  (let [double (fn [v] (n/* v v))
        out (double-array size)
        mapped (par/map! out i size double 2.0)]
    (n/+ h (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i))))))

(deftm excluded-lexical-store [h :- Double x :- Double size :- Long] :- Double
  (let [double (fn [v] (n/* v v))
        out (double-array size)
        mapped (par/map! out i size double x)]
    (n/+ h (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i))))))

(defn- construction-reason [construct source]
  (try (construct source) nil
       (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))

(deftest nonlinear-store-callables-cannot-use-the-body-only-pullback
  (doseq [source [#'lexical-store-square #'foreign-store-square]]
    (let [x 2.0 size 3 h 0.125]
      (is (= 12.0 (source x size)))
      (is (= 12.0 (/ (- (source (+ x h) size) (source (- x h) size)) (* 2.0 h)))))
    (is (= 0.0 (source 2.0 0)))
    ;; Construction is independent of runtime extent: even zero extent cannot
    ;; turn unknown store-callee authority into a derivative capability.
    (doseq [construct [#(rev/value+grad % :wrt [0]) jvp/jvp]]
      (is (= :ad-map-store-callee (construction-reason construct source)))))
  (is (= 12.0 (captured-store-square 2.0 3)))
  (is (= 12.0 (/ (- (captured-store-square 2.125 3)
                     (captured-store-square 1.875 3)) 0.25)))
  ;; A callable whose body captures an active input must not become constant
  ;; merely because its map input is literal. Existing closure admission owns
  ;; this earlier refusal, rather than inventing a closure derivative here.
  (doseq [construct [#(rev/value+grad % :wrt [0]) jvp/jvp]]
    (is (= :ad-active-closure (construction-reason construct #'captured-store-square)))))

(deftest established-floating-store-conversions-preserve-existing-capability
  (doseq [source [#'core-store-square #'qualified-core-store-square
                  #'nil-store-square #'float-store-square]
          size [0 3]]
    (let [expected (* size 4.0) dx (* size 4.0)]
      (is (= expected (source 2.0 size)))
      (is (= dx (/ (- (source 2.125 size) (source 1.875 size)) 0.25)))
      (is (= [expected dx nil] ((rev/value+grad source :wrt [0]) 2.0 size)))
      (is (= [expected dx] ((jvp/jvp source) 2.0 size 1.0))))))

(deftest inactive-and-excluded-store-callables-retain-source-execution
  (doseq [size [0 3]]
    (let [expected (+ 2.0 (* size 4.0))]
      (is (= expected (inactive-lexical-store 2.0 size)))
      (is (= [expected 1.0 nil]
             ((rev/value+grad #'inactive-lexical-store :wrt [0]) 2.0 size)))
      (is (= [expected 1.0] ((jvp/jvp #'inactive-lexical-store) 2.0 size 1.0)))
      (is (= expected (excluded-lexical-store 2.0 2.0 size)))
      (is (= [expected 1.0 nil nil]
             ((rev/value+grad #'excluded-lexical-store :wrt [0]) 2.0 2.0 size))))))

(deftest source-store-identity-is-rederived-not-accepted-from-incoming-metadata
  (let [foreign 'raster.ad.map-store-callee-test/foreign-store
        source-ns (the-ns 'raster.ad.map-store-callee-test)]
    (doseq [[callee lexical expected]
            [['double false 'clojure.core/double]
             ['clojure.core/double false 'clojure.core/double]
             ['float false 'clojure.core/float]
             [foreign false foreign]
             ['double true nil]
             ['missing-store false nil]]]
      (let [callee (with-meta callee {:raster.op/resolved-callee 'clojure.core/double})
            env (cond-> {'out {:tag 'doubles} 'x {:tag 'double}}
                  lexical (assoc 'double {:tag nil}))
            expression (list 'raster.par/map! 'out 'i 3 callee 'x)
            walked (walker/walk-body expression {:source-ns source-ns :type-env env})
            retained (nth walked 4)]
        (is (= callee retained) "execution spelling remains unchanged")
        (is (= expected (:raster.op/resolved-callee (meta retained))))
        (is (string? (fingerprint/fingerprint walked)))
        (is (= (when-not (contains? '#{clojure.core/double clojure.core/float} expected)
                 :ad-map-store-callee)
               (try (rev/assert-map-store-callee! walked true) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
    (is (nil? (rev/assert-map-store-callee!
                '(raster.par/map! out i 3 nil x) true)))
    (is (nil? (rev/assert-map-store-callee!
                (list 'raster.par/map! 'out 'i 3 foreign 'x) false)))))

(deftest pure-map-materialization-can-retain-source-derived-store-authority
  (doseq [[cast expected] [['double 'clojure.core/double]
                          ['float 'clojure.core/float]]]
    (let [walked (walker/walk-body
                   (list 'raster.par/pmap 'i 3 cast 'x)
                   {:source-ns (the-ns 'raster.ad.map-store-callee-test)
                    :type-env {'x {:tag 'double}}})]
      (is (= cast (nth walked 3)))
      (is (= expected (:raster.op/resolved-callee (meta (nth walked 3))))))))

(deftest reduction-adjoint-scatter-uses-the-existing-destination-store-authority
  (doseq [[source make-array] [[#'repeated-float-reduce float-array]
                              [#'repeated-double-reduce double-array]]
          size [0 3 5]]
    (let [xs (make-array [2.0 5.0])
          direction (make-array [0.5 0.25])
          counts (mapv #(double (count (filter #{%} (map (fn [i] (rem i 2)) (range size)))))
                       [0 1])
          expected (reduce + 0.0 (map * counts [2.0 5.0]))
          expected-tangent (reduce + 0.0 (map * counts [0.5 0.25]))
          gradient-function (rev/value+grad source :wrt [0])
          [value gradient _] (gradient-function xs size)
          [forward-value forward-tangent] ((jvp/jvp source) xs size direction)]
      (is (= expected (source xs size) value forward-value))
      (is (= (class xs) (class gradient)))
      (is (= counts (vec gradient)) "repeated reads accumulate, not overwrite")
      (is (= expected-tangent forward-tangent
             (reduce + 0.0 (map * gradient direction))) "JVP/transpose pairing")
      (dotimes [coordinate 2]
        (let [plus (make-array xs) minus (make-array xs) h 0.125]
          (ra/aset plus coordinate (+ (double (nth xs coordinate)) h))
          (ra/aset minus coordinate (- (double (nth xs coordinate)) h))
          (is (= (nth counts coordinate)
                 (/ (- (source plus size) (source minus size)) (* 2.0 h))))))
      (is (= [2.0 5.0] (vec xs)))
      (is (= [0.5 0.25] (vec direction)))
      (when (= source #'repeated-float-reduce)
        (let [stores (filter #(and (seq? %) (= 'raster.arrays/aset (first %))
                                  (seq? (nth % 3 nil))
                                  (= 'raster.ad.reverse/grad-acc (first (nth % 3))))
                             (tree-seq coll? seq
                                       (:raster.core/deftm-walked-body (meta gradient-function))))]
          (is (seq stores) "generated numeric scatter selects the shared typed setter")
          (is (every? #(= 'floats (:raster.type/tag (meta (second %)))) stores)
              "the destination retains its original Float storage authority"))))))

(deftest reduction-hoisting-preserves-retained-operand-types
  (let [init (with-meta '(double 0.0) {:raster.type/tag 'double})
        bound (with-meta '(long 3) {:raster.type/tag 'long})
        form (list 'raster.par/reduce 'total init 'i bound 'total)
        [_ bindings reduction] (#'rev/hoist-nested-lets form)
        [bound-symbol _ init-symbol _] bindings]
    (is (= 'long (:raster.type/tag (meta bound-symbol))))
    (is (= 'double (:raster.type/tag (meta init-symbol))))
    (is (= init-symbol (nth reduction 2)))
    (is (= (meta init-symbol) (meta (nth reduction 2))))
    (is (= bound-symbol (nth reduction 4)))))

(deftest float-read-dotimes-keeps-numeric-shadow-storage
  (doseq [size [0 3]]
    (let [xs (float-array [1.0 -2.0 3.0])
          [value gradient _] ((rev/value+grad #'float-read-dotimes :wrt [0]) xs size)]
      (is (= (if (zero? size) 0.0 14.0) value (float-read-dotimes xs size)))
      (is (= (class xs) (class gradient)))
      (is (= (if (zero? size) [0.0 0.0 0.0] [2.0 -4.0 6.0]) (vec gradient)))
      (dotimes [coordinate 3]
        (let [plus (aclone xs) minus (aclone xs) h 0.125]
          (ra/aset plus coordinate (+ (double (nth xs coordinate)) h))
          (ra/aset minus coordinate (- (double (nth xs coordinate)) h))
          (is (= (nth gradient coordinate)
                 (/ (- (float-read-dotimes plus size) (float-read-dotimes minus size))
                    (* 2.0 h))))))
      (is (= [1.0 -2.0 3.0] (vec xs))))))
