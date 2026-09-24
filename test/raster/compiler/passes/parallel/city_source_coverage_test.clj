(ns raster.compiler.passes.parallel.city-source-coverage-test
  (:refer-clojure :exclude [aget aset])
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :refer [aget aset]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.gpu.core :as gpu]
            [raster.numeric]
            [raster.par :as par]))

(def ^:const scale 0.5)

(deftm city-like-scalar-helper [x :- Double] :- Double
  (* x 2.0))

(deftm city-like-helper-map!
  [xs :- (Array double), out :- (Array double), n :- Long] :- Void
  (par/map-void! i n (aset out i (city-like-scalar-helper (aget xs i)))))

(deftm city-like-constant-map!
  [xs :- (Array double), out :- (Array double), n :- Long] :- Void
  (par/map-void! i n (aset out i (* scale (aget xs i)))))

(deftm city-like-terminal-store!
  [weights :- (Array double), out :- (Array double), n :- Long, nc :- Long] :- Void
  (par/map-void! i n
    (loop [q 0 acc 0.0]
      (if (< q nc)
        (recur (inc q) (+ acc (aget weights q)))
        (aset out i acc)))))

(deftm city-like-binary-search!
  [starts :- (Array int), ends :- (Array int), cdf :- (Array double),
   targets :- (Array double), out :- (Array int), n :- Long] :- Void
  (par/map-void! i n
    (let [^int selected
          (loop [^int lo (int (aget starts i))
                 ^int hi (int (aget ends i))]
            (if (>= lo hi)
              lo
              (let [^int mid (int (quot (+ lo hi) 2))]
                (if (< (aget targets i) (aget cdf mid))
                  (recur lo mid)
                  (recur (inc mid) hi)))))]
      (aset out i selected))))

(deftm canonical-operator-dot
  [left :- (Array float), right :- (Array float), n :- Long] :- Double
  (par/reduce acc 0.0 i n
              (raster.numeric/+ acc
                                (raster.numeric/* (aget left i) (aget right i)))))

(defn- emitted-body
  [v array-types scalar-types]
  (let [source (first (gpu/get-walked-body v :double))
        options {:dtype :double :target-device :ocl:0
                 :array-types array-types :scalar-types scalar-types}
        scheduled (pipeline/schedule-parallel-form source options)
        operation (first (:operations (first (:equations (:form scheduled)))))
        artifact (when operation
                   (segop-opencl/generate-scheduled-segmap-kernel
                    operation :array-types array-types :scalar-types scalar-types))]
    {:scheduled scheduled :artifact artifact}))

(deftest walked-scalar-helpers-and-constants-use-the-shared-typed-route
  (doseq [v [#'city-like-helper-map! #'city-like-constant-map!]]
    (let [{:keys [scheduled artifact]}
          (emitted-body v {'xs :double 'out :double} {'n :long})]
      (is (= :typed-soac (get-in scheduled [:stats :source-dialect]))
          (str v " must not enter the compatibility fallback"))
      (is (= :kernel-body (get-in artifact [:attributes :emission-route]))
          (str v " must emit through the common typed scalar body")))))

(deftest walked-terminal-store-loop-is-a-typed-ordered-recurrence
  (let [{:keys [scheduled artifact]}
        (emitted-body #'city-like-terminal-store!
                      {'weights :double 'out :double}
                      {'n :long 'nc :long})]
    (is (= :typed-soac (get-in scheduled [:stats :source-dialect])))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))))

(deftest walked-city-like-binary-search-is-a-typed-while-loop
  (let [source (first (gpu/get-walked-body #'city-like-binary-search! :int))
        array-types {'starts :int 'ends :int 'cdf :double
                     'targets :double 'out :int}
        scalar-types {'n :long}
        scheduled (pipeline/schedule-parallel-form
                   source {:dtype :int :target-device :ocl:0
                           :array-types array-types :scalar-types scalar-types})
        operation (first (:operations (first (:equations (:form scheduled)))))
        artifact (when operation
                   (segop-opencl/generate-scheduled-segmap-kernel
                    operation :array-types array-types :scalar-types scalar-types))]
    (is (= :typed-soac (get-in scheduled [:stats :source-dialect])))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (re-find #"while \(1\)" (:source artifact)))))

(deftest helper-expansion-preserves-canonical-numeric-reductions
  (let [source (first (gpu/get-walked-body #'canonical-operator-dot :float))
        scheduled (pipeline/schedule-parallel-form
                   source {:dtype :float :target-device :ocl:0
                           :array-types {'left :float 'right :float}
                           :scalar-types {'n :long}})]
    (is (= :typed-soac (get-in scheduled [:stats :source-dialect])))
    (is (nil? (get-in scheduled [:stats :typed-soac-declined])))))
