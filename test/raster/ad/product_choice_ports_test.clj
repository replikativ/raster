(ns raster.ad.product-choice-ports-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]))

(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))

(defn- compile-program [params bindings result active]
  (let [policy {:unchecked-math false}]
    (#'reverse/make-runtime-value+grad-fn
     {:kind :generated-product-bindings :source-policy policy
      :bindings bindings :result result :active-inputs active}
     params {:generated-source-policy policy})))

(deftest an-existing-product-choice-routes-each-field-to-its-selected-source
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        square (typed 'square 'double) result (typed 'result 'double)
        f (compile-program [x flag]
                           [square '(raster.numeric/* x x)
                            'left ['x 'square nil false]
                            'right ['square 'x false nil]
                            'selected '(if flag left right)
                            result '(clojure.core/first selected)] result ['x])]
    (doseq [x [-2.0 0.0 3.0] flag [true false]]
      (let [[value pb] (f x flag)]
        (is (= (if flag x (* x x)) value))
        (is (= [(if flag 1.0 (* 2.0 x))] (pb 1.0)))))))

(deftest a-product-choice-preserves-dynamic-absence-versus-present-zero
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        singular (typed 'singular 'double) result (typed 'result 'double)
        f (compile-program [x flag]
                           [singular '(raster.numeric/sqrt x)
                            'left ['x] 'right ['singular]
                            'selected '(if flag left right)
                            result '(clojure.core/first selected)] result ['x])]
    (let [[_ pb] (f 0.0 true)] (is (= [1.0] (pb 1.0))))
    (let [[_ pb] (f 0.0 false)]
      (is (= [Double/POSITIVE_INFINITY] (pb 1.0)))
      (is (Double/isNaN (double (first (pb 0.0))))))))

(deftest joins-require-known-equal-shapes-and-leaf-declarations
  (let [x (typed 'x 'float) y (typed 'y 'double) flag (typed 'flag 'boolean)
        result (typed 'result 'double)
        reason (fn [bindings]
                 (try (compile-program [x y flag] bindings result ['x 'y]) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (doseq [bindings [['left ['x] 'right ['y] 'selected '(if flag left right)
                      result '(clojure.core/first selected)]
                     ['left ['y] 'right [['y]] 'selected '(if flag left right)
                      result '(clojure.core/first selected)]
                     ['left ['y] 'selected '(if flag left y)
                      result '(clojure.core/first selected)]]]
      (is (= :ad-product-choice-schema (reason bindings))))))

(deftest product-results-from-lexical-arms-use-the-selected-residual-once
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        left (typed 'left 'double) right (typed 'right 'double)
        result (typed 'result 'double)
        f (compile-program
           [x flag]
           ['selected (list 'if 'flag
                            (list 'let* [left '(raster.numeric/* x x)] [left nil false])
                            (list 'let* [right '(raster.numeric/+ x x)] [right false nil]))
            result '(clojure.core/first selected)] result ['x])]
    (doseq [x [-2.0 0.0 3.0] flag [true false]]
      (let [[value pb] (f x flag)]
        (is (= (if flag (* x x) (+ x x)) value))
        (is (= [(if flag (* 2.0 x) 2.0)] (pb 1.0)))))))

(deftest unselected-product-arms-never-evaluate-partial-primal-expressions
  (let [x (typed 'x 'double) xs (typed 'xs 'doubles) flag (typed 'flag 'boolean)
        left (typed 'left 'double) right (typed 'right 'double)
        result (typed 'result 'double)
        f (compile-program
           [x xs flag]
           ['selected (list 'if 'flag
                            (list 'let* [left '(raster.numeric/* x x)] [left])
                            (list 'let* [right '(raster.arrays/aget xs 99)] [right]))
            result '(clojure.core/first selected)] result ['x 'xs])
        input (double-array [2.0])
        [value pb] (f 3.0 input true)
        [dx dxs] (pb 1.0)]
    (is (= 9.0 value))
    (is (= 6.0 dx))
    (is (= [0.0] (vec dxs)))
    (is (= [2.0] (vec input)))
    (is (thrown? ArrayIndexOutOfBoundsException (f 3.0 input false)))))

(deftest nested-product-results-preserve-field-paths-and-finite-difference-gradients
  (let [x (typed 'x 'double) outer (typed 'outer 'boolean) inner (typed 'inner 'boolean)
        a (typed 'a 'double) b (typed 'b 'double) c (typed 'c 'double)
        result (typed 'result 'double)
        inner-choice (list 'if 'inner
                           (list 'let* [a '(raster.numeric/* x x)] [[a] nil])
                           (list 'let* [b '(raster.numeric/+ x x)] [[b] nil]))
        f (compile-program
           [x outer inner]
           ['selected (list 'if 'outer inner-choice
                            (list 'let* [c '(raster.numeric/- x)] [[c] nil]))
            'nested '(clojure.core/first selected)
            result '(clojure.core/first nested)] result ['x])
        h 1.0e-5]
    (doseq [x [-2.0 0.25 3.0] outer [true false] inner [true false]]
      (let [oracle (fn [x] (if outer (if inner (* x x) (+ x x)) (- x)))
            [value pb] (f x outer inner)
            fd (/ (- (oracle (+ x h)) (oracle (- x h))) (* 2.0 h))]
        (is (= (oracle x) value))
        (is (< (Math/abs (- fd (first (pb 1.0)))) 1.0e-7))))))

(deftest array-valued-fields-keep-aliases-borrowed-and-pullbacks-independent
  (let [xs (typed 'xs 'doubles) ys (typed 'ys 'doubles) flag (typed 'flag 'boolean)
        selected (typed 'selected 'doubles) result (typed 'result 'double)
        f (compile-program [xs ys flag]
                           ['p '(if flag [xs xs] [ys ys])
                            selected '(clojure.core/first p)
                            result '(raster.arrays/aget selected 0)] result ['xs 'ys])
        x (double-array [2.0 3.0]) y (double-array [5.0 6.0 7.0])]
    (doseq [flag [true false]]
      (let [[value pb] (f x y flag)
            [dx dy] (pb 1.0)
            [dx2 dy2] (pb 2.0)]
        (is (= (if flag 2.0 5.0) value))
        (is (= (if flag [1.0 0.0] [0.0 0.0]) (vec dx)))
        (is (= (if flag [0.0 0.0 0.0] [1.0 0.0 0.0]) (vec dy)))
        (is (= (if flag [2.0 0.0] [0.0 0.0]) (vec dx2)))
        (is (= (if flag [0.0 0.0 0.0] [2.0 0.0 0.0]) (vec dy2)))
        (is (not (identical? dx dx2)))
        (is (not (identical? dy dy2)))
        (is (= [2.0 3.0] (vec x)))
        (is (= [5.0 6.0 7.0] (vec y)))))))

(deftest numeric-literal-fields-have-their-source-declarations-without-call-inference
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean) result (typed 'result 'double)
        f (compile-program [x flag]
                           ['p '(if flag [x] [0.0])
                            result '(clojure.core/first p)] result ['x])]
    (doseq [flag [true false]]
      (let [[value pb] (f 2.0 flag)]
        (is (= (if flag 2.0 0.0) value))
        (is (= [(if flag 1.0 0.0)] (pb 1.0)))))))

(deftest lexical-product-arms-preserve-mixed-leaf-carriers-and-order
  (let [x (typed 'x 'float) y (typed 'y 'double) flag (typed 'flag 'boolean)
        a (typed 'a 'float) b (typed 'b 'double)
        c (typed 'c 'float) d (typed 'd 'double)
        result (typed 'result 'double)
        f (compile-program
           [x y flag]
           ['p (list 'if 'flag
                     (list 'let* [a '(raster.numeric/+ x x) b '(raster.numeric/* y y)] [a b])
                     (list 'let* [c '(raster.numeric/- x) d '(raster.numeric/+ y y)] [c d]))
            'first-value '(clojure.core/first p)
            'second-value '(clojure.core/second p)
            result '(raster.numeric/+ first-value second-value)] result ['x 'y])]
    (doseq [flag [true false] seed [1.0 Math/PI]]
      (let [[value pb] (f (float 2.0) 3.0 flag)
            [dx dy] (pb seed)]
        (is (= (if flag 13.0 4.0) value))
        (is (instance? Float dx))
        (is (instance? Double dy))
        (is (= (Float/floatToRawIntBits (float (* (if flag 2.0 -1.0) seed)))
               (Float/floatToRawIntBits dx)))
        (is (= (* (if flag 6.0 2.0) seed) dy))))))

(deftest inactive-product-choices-retain-their-structural-interface
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean) other (typed 'other 'boolean)
        result (typed 'result 'double)
        f (compile-program [x flag other]
                           ['zero [0.0] 'one [1.0]
                            'constant '(if flag zero one)
                            'live ['x]
                            'selected '(if other live constant)
                            result '(clojure.core/first selected)] result ['x])]
    (doseq [flag [true false] other [true false]]
      (let [[value pb] (f 2.0 flag other)]
        (is (= (if other 2.0 (if flag 0.0 1.0)) value))
        (is (= [(if other 1.0 0.0)] (pb 1.0)))))))
