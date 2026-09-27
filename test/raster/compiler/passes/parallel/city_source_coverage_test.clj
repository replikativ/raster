(ns raster.compiler.passes.parallel.city-source-coverage-test
  (:refer-clojure :exclude [aget aset])
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.arrays :refer [aget aset]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.backend.jvm.segop-simd :as segop-simd]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.parallel.city-workload-fixture :as city]
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

(deftm terminal-swap-and-read!
  [left :- (Array double), right :- (Array double), n :- Long, nc :- Long] :- Void
  (par/map-void! i n
    (loop [q 0 a 1.0 b 2.0]
      (if (< q nc)
        (recur (inc q) b a)
        (do (aset left i a)
            (aset right i (+ b (aget left i))))))))

(deftm terminal-read-before-write!
  [values :- (Array double), out :- (Array double), n :- Long, nc :- Long] :- Void
  (par/map-void! i n
    (loop [q 0 a 0.0 b 0.0]
      (if (< q nc)
        (let [v (aget values i)]
          (recur (inc q) (+ a v) (+ b (* 2.0 v))))
        (do (aset values i a)
            (aset out i b))))))

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
    {:scheduled scheduled :artifact artifact :operation operation}))

(deftest two-exit-search-retains-lazy-lexical-branches
  (let [{:keys [scheduled artifact operation]}
        (emitted-body #'city/two-exit-walk!
                      {'weights :double 'targets :double 'out :int} {'n :long 'nc :long})]
    (is (= :typed-soac (get-in scheduled [:stats :source-dialect])))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (re-find #"while \(1\)" (:source artifact)))
    (doseq [target [:cuda :hip]]
      (is (= :kernel-body
             (get-in (segop-opencl/generate-scheduled-segmap-kernel
                      operation :target-dialect target
                      :array-types {'weights :double 'targets :double 'out :int}
                      :scalar-types {'n :long 'nc :long}) [:attributes :emission-route]))))))

(deftest induction-only-branches-preserve-effects-and-lexical-locals
  (let [{:keys [operation]}
        (emitted-body #'city/branch-local-store-steps!
                      {'state :int 'out :int} {'n :long 'nc :long})
        execute (eval (list 'fn '[state out n nc] (segop-simd/compile-effect-segmap operation)))]
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (= :kernel-body
             (get-in (segop-opencl/generate-scheduled-segmap-kernel
                      operation :target-dialect target :array-types {'state :int 'out :int}
                      :scalar-types {'n :long 'nc :long}) [:attributes :emission-route]))))
    (doseq [nc [0 1 2 7]]
      (let [state (int-array 3) out (int-array 3)
            expected-state (int-array 3) expected-out (int-array 3)]
        (dotimes [_ 2]
          (city/branch-local-store-steps! expected-state expected-out 3 nc)
          (execute state out 3 nc)
          (is (= (vec expected-state) (vec state)))
          (is (= (vec expected-out) (vec out))))))))

(deftest carried-branches-yield-at-their-effect-position
  (doseq [kernel [#'city/branch-local-carried-steps! #'city/carried-steps-with-empty-arm!]]
    (let [{:keys [operation]}
          (emitted-body kernel {'state :double 'out :double} {'n :long 'nc :long})
          execute (eval (list 'fn '[state out n nc] (segop-simd/compile-effect-segmap operation)))
          effects (get-in operation [:scalar-region :effects])
          branches (filter :branch (tree-seq coll? seq effects))
          merge-results (into #{} (mapcat #(map :result (get-in % [:branch :results]))) branches)
          carries (mapcat #(get-in % [:loop :carries]) (filter :loop (tree-seq coll? seq effects)))]
      (is (seq branches) "source branch results remain first-class, not reconstructed conditional updates")
      (is (and (seq carries) (every? merge-results (map :update carries)))
          "loop updates consume the arm's merged tuple directly")
      (doseq [target [:opencl-portable :cuda :hip]]
        (is (= :kernel-body
               (get-in (segop-opencl/generate-scheduled-segmap-kernel
                        operation :target-dialect target :array-types {'state :double 'out :double}
                        :scalar-types {'n :long 'nc :long}) [:attributes :emission-route]))))
      (doseq [nc [0 1 2 7]]
        (let [state (double-array [-1 1 -1]) out (double-array [-77 -77 -77])
              expected-state (aclone state) expected-out (aclone out)]
          (dotimes [_ 2]
            (kernel expected-state expected-out 3 nc)
            (execute state out 3 nc)
            (is (= (vec expected-state) (vec state)))
            (is (= (vec expected-out) (vec out)))))))))

(deftest effectful-tuples-initialize-sequentially-and-update-simultaneously
  (let [{:keys [scheduled artifact operation]}
        (emitted-body #'city/three-carry-effects! {'out :int} {'n :long 'nc :long})
        execute (eval (list 'fn '[out n nc] (segop-simd/compile-effect-segmap operation)))
        body (get-in artifact [:attributes :kernel-body])
        operations (tree-seq #(and (map? %) (contains? % :operations)) :operations body)
        loop (first (filter #(seq (:iter-args %)) operations))]
    (is (= :typed-soac (get-in scheduled [:stats :source-dialect])))
    (is (= 3 (count (:iter-args loop))))
    (is (= 3 (count (:results loop))))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (= :kernel-body
             (get-in (segop-opencl/generate-scheduled-segmap-kernel
                      operation :target-dialect target :array-types {'out :int}
                      :scalar-types {'n :long 'nc :long}) [:attributes :emission-route]))))
    (doseq [nc [0 1 2 3 7]]
      (let [expected (int-array 9) actual (int-array 9)]
        (city/three-carry-effects! expected 3 nc)
        (execute actual 3 nc)
        (is (= (vec expected) (vec actual)))))))

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

(deftest walked-terminal-moments-share-one-ordered-recurrence
  (let [{:keys [scheduled artifact operation]}
        (emitted-body #'city/terminal-moments!
                      {'weights :double 'sums :double 'squares :double}
                      {'n :long 'nc :long})
        operations (tree-seq #(and (map? %) (contains? % :operations))
                             :operations (get-in artifact [:attributes :kernel-body]))]
    (is (= :typed-soac (get-in scheduled [:stats :source-dialect])))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (= 1 (count (filter #(= "ForLoop" (some-> % class .getSimpleName)) operations))))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (= :kernel-body
             (get-in (segop-opencl/generate-scheduled-segmap-kernel
                      operation :target-dialect target
                      :array-types {'weights :double 'sums :double 'squares :double}
                      :scalar-types {'n :long 'nc :long})
                     [:attributes :emission-route]))))
    (let [execute (eval (list 'fn '[weights sums squares n nc]
                              (segop-simd/compile-segmap operation (:out-sym operation) 'double)))]
      (doseq [nc [0 1 7]]
        (let [weights (double-array (range (max 1 (* 3 nc))))
              sums (double-array 3) squares (double-array 3)
              expected-sums (double-array 3) expected-squares (double-array 3)]
          (city/terminal-moments! weights expected-sums expected-squares 3 nc)
          (execute weights sums squares 3 nc)
          (is (= (vec expected-sums) (vec sums)))
          (is (= (vec expected-squares) (vec squares))))))))

(deftest terminal-effects-follow-simultaneous-carry-updates
  (let [{:keys [operation]} (emitted-body #'terminal-swap-and-read!
                                         {'left :double 'right :double}
                                         {'n :long 'nc :long})
        execute (eval (list 'fn '[left right n nc]
                            (segop-simd/compile-effect-segmap operation)))]
    (doseq [nc [0 1 2 7]]
      (let [left (double-array [-77 -77]) right (double-array [-77 -77])]
        (execute left right 2 nc)
        (is (= (vec (repeat 2 (if (odd? nc) 2.0 1.0))) (vec left)))
        (is (= [3.0 3.0] (vec right)) "second store observes the first terminal store")))))

(deftest terminal-write-cannot-change-another-fold-component
  (let [{:keys [operation]} (emitted-body #'terminal-read-before-write!
                                         {'values :double 'out :double}
                                         {'n :long 'nc :long})
        execute (eval (list 'fn '[values out n nc]
                            (segop-simd/compile-segmap operation (:out-sym operation) 'double)))]
    (doseq [nc [0 1 3]]
      (let [values (double-array [3 5]) out (double-array 2)]
        (execute values out 2 nc)
        (is (= (mapv #(* (double nc) %) [3.0 5.0]) (vec values)))
        (is (= (mapv #(* 2.0 nc %) [3.0 5.0]) (vec out)))))))

(deftest cross-row-terminal-reads-are-not-certified-independent
  (let [source (first (gpu/get-walked-body #'terminal-swap-and-read! :double))
        source (walk/postwalk
                #(if (= % '(clojure.core/aget left i))
                   '(clojure.core/aget left (mod (inc i) n)) %) source)
        result (route/attempt (list 'let* ['result source] 'result)
                              :double {'left :double 'right :double}
                              {:scalar-types {'n :long 'nc :long}})
        attrs (-> result :program :equations first :algorithm
                  dialect/equations first dialect/operation-parts :attributes)]
    (is (= :sequential (:iteration-order attrs)))))

(deftest terminal-branch-decision-precedes-its-writes
  (let [{:keys [operation]} (emitted-body #'city/terminal-branch-snapshot!
                                         {'left :double 'right :double}
                                         {'n :long 'nc :long})
        execute (eval (list 'fn '[left right n nc]
                            (segop-simd/compile-effect-segmap operation)))]
    (doseq [nc [0 1 2]]
      (let [left (double-array [1 -1]) right (double-array 2)
            expected-left (double-array [1 -1]) expected-right (double-array 2)]
        (city/terminal-branch-snapshot! expected-left expected-right 2 nc)
        (execute left right 2 nc)
        (is (= (vec expected-left) (vec left)))
        (is (= (vec expected-right) (vec right)))))))

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
