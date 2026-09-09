(ns raster.compiler.passes.parallel.source-continuation-scope-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-carried-effect-frontend-test :as fixture]))

(defn- replace-form [form before after]
  (walk/postwalk #(if (= before %) after %) form))

(defn- execute-source [form]
  (let [result (fixture/attempt form)]
    (is (= :typed-soac (get-in result [:stats :route])))
    (let [scheduled (:form (segop-lower/segop-lower-pass
                            (:program result) {:target-device :ze:0 :dtype :float}))]
      (eval (list 'fn '[x packed scales sums rows width seed]
                  (:form (par-simd/simd-pass scheduled :min-elements 1)))))))

(deftest carry-body-and-continuation-reuse-source-local-names
  (let [form (replace-form fixture/source '(aset sums row (float sum))
                           '(let* [^float v (float sum)]
                              (loop* [j 0]
                                (if (< j width)
                                  (do (aset packed (+ (* row width) j) v)
                                      (recur (inc j)))
                                  nil))
                              (aset sums row v)))
        execute (execute-source form)
        packed (float-array (repeat 8 -77)) sums (float-array 2)]
    (execute (float-array [1 2 3 4 5 6]) packed (float-array 2) sums 2 3 (float 2.5))
    (is (= [6.25 15.25] (vec sums)))
    (is (= [6.25 6.25 6.25 15.25 15.25 15.25 -77.0 -77.0] (vec packed)))))

(deftest sibling-continuations-do-not-share-renamed-local-bindings
  ;; Both carried regions and their continuations reuse sum/acc/v. The second invocation
  ;; must capture its own value rather than the first continuation's generated local ID.
  (let [one (nth (nth fixture/source 1) 1)
        [_ index extent body] one
        second-body (replace-form body 0.25 1.25)
        continuation '(let* [^float v (float sum)] (aset sums row v))
        form (list 'raster.par/map-void! index extent
                   (list 'do
                         (replace-form body '(aset sums row (float sum)) continuation)
                         (replace-form second-body '(aset sums row (float sum)) continuation)))
        execute (execute-source (list 'let* ['effect form] 'effect))
        packed (float-array (repeat 4 -77)) sums (float-array 1)]
    (execute (float-array [1 2 3]) packed (float-array 1) sums 1 3 (float 2.5))
    (is (= [7.25] (vec sums)))
    (is (= [1.0 2.0 3.0 -77.0] (vec packed)))))

(deftest checked-continuation-initializer-fails-after-earlier-writes
  (let [form (replace-form fixture/source '(aset sums row (float sum))
                           '(let* [^long checked
                                   ^{:raster.type/tag long}
                                   (clojure.core/+ 9223372036854775807 width)]
                              (aset sums row (float checked))))
        execute (execute-source form)
        packed (float-array [-77 -77]) scales (float-array [-77]) sums (float-array [-77])]
    (is (thrown? ArithmeticException
                 (execute (float-array [9]) packed scales sums 1 1 (float 2.5))))
    (is (= [9.0 -77.0] (vec packed)))
    (is (= [2.5] (vec scales)))
    (is (= [-77.0] (vec sums)))))

(deftest computed-continuation-bound-scopes-over-its-store-loop
  (let [form (replace-form fixture/source '(aset sums row (float sum))
                           '(let* [^long stop ^{:raster.type/tag long} (clojure.core/+ width 1)]
                              (loop* [j 0]
                                (if (< j stop)
                                  (do (aset packed j (float sum)) (recur (inc j)))
                                  nil))
                              (aset sums row (float sum))))
        execute (execute-source form)
        packed (float-array (repeat 5 -77)) sums (float-array 1)]
    (execute (float-array [1 2 3]) packed (float-array 1) sums 1 3 (float 2.5))
    (is (= [6.25 6.25 6.25 6.25 -77.0] (vec packed)))
    (is (= [6.25] (vec sums)))))

(deftest continuation-bound-snapshots-after-stores-and-before-the-next-loop
  (let [form (replace-form fixture/source '(aset sums row (float sum))
                           '(let* [^long stop (long (aget packed 0))]
                              (loop* [j 0]
                                (if (< j stop)
                                  (do (aset packed j (float sum)) (recur (inc j))) nil))
                              (aset sums row (float sum))))
        execute (execute-source form)
        packed (float-array (repeat 8 -77)) sums (float-array 1)]
    (execute (float-array [3 2 1]) packed (float-array 1) sums 1 3 (float 2.5))
    (is (= [6.25 6.25 6.25 -77.0 -77.0 -77.0 -77.0 -77.0] (vec packed)))
    (is (= [6.25] (vec sums)))))
