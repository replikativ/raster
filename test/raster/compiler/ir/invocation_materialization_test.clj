(ns raster.compiler.ir.invocation-materialization-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- scalar [dtype] (av/tensor {:dtype dtype :shape []}))
(defn- array-value [dtype extent]
  (av/tensor {:dtype dtype :shape [extent] :representation {:kind :plain}}))

(defn- fixture-plan []
  (let [public-array (array-value :double '(extent u0))
        internal-array (array-value :double 'n)]
    (invocation/from-prefix
     {:id :materialization-test
      :parameters '[u0 dt nsteps]
      :parameter-values {'u0 public-array 'dt (scalar :double) 'nsteps (scalar :long)}
      :bindings '[[n (int (alength u0))]
                  [u (aclone u0)]
                  [scratch (double-array n)]
                  [half-dt (* 0.5 dt)]
                  [nsteps (int nsteps)]]
      :binding-values {'n (scalar :int) 'u internal-array 'scratch internal-array
                       'half-dt (scalar :double) 'nsteps (scalar :int)}
      :program-values {'n (scalar :int) 'u internal-array 'scratch internal-array
                       'half-dt (scalar :double) 'nsteps (scalar :int)
                       'result internal-array}
      :program-inputs '[nsteps n half-dt u scratch]
      :program-outputs '[result]})))

(defn- evaluate-scalar [step operands]
  (let [{:keys [body-results]} (soac/lambda-parts (:region step))
        expression (first body-results)
        value
        (case (first expression)
          int (int (:value (get operands (second expression))))
          inc (inc (:value (get operands (second expression))))
          * (* (:value (get operands (nth expression 2))) (double (second expression))))]
    {:type (get-in step [:value :dtype]) :value value}))

(deftest ordered-public-values-materialize-the-exact-program-boundary
  (let [source (double-array [1.0 2.0 3.0])
        result (materialization/materialize (fixture-plan) [source 0.25 7] evaluate-scalar)
        buffers (:program-buffers result)
        scalars (:program-scalars result)]
    (is (materialization/materialized-invocation? result))
    (is (= '[nsteps n half-dt u scratch]
           (mapv :program-value (:bindings (:plan result)))))
    (is (= #{'u 'scratch} (set (keys buffers))))
    (is (= #{'nsteps 'n 'half-dt} (set (keys scalars))))
    (is (= {:type :int :value 7} (get scalars 'nsteps)))
    (is (= {:type :int :value 3} (get scalars 'n)))
    (is (= {:type :double :value 0.125} (get scalars 'half-dt)))
    (is (= [3] (:shape (get buffers 'u))))
    (is (= [3] (:shape (get buffers 'scratch))))
    (is (identical? source (:source (get buffers 'u))))
    (is (= :copy (:initialization (get buffers 'u))))
    (is (= :zero (:initialization (get buffers 'scratch))))
    (is (= {:source-inspected false :driver-allocations 0} (:attributes result)))))

(deftest public-buffer-dtype-and-arity-are-checked
  (let [plan (fixture-plan)]
    (testing "argument order is exact"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"parameter order"
                            (materialization/materialize plan [(double-array 3) 0.25]
                                                         evaluate-scalar))))
    (testing "raw storage cannot silently change dtype"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dtype differs"
                            (materialization/materialize plan [(float-array 3) 0.25 7]
                                                         evaluate-scalar))))))

(deftest caller-owned-write-only-storage-is-materialized-outside-logical-inputs
  (let [value (array-value :float 4)
        plan (invocation/from-prefix
              {:id :write-only-storage
               :parameters '[x y]
               :parameter-values {'x value 'y value}
               :bindings [] :binding-values {}
               :program-values {'x value 'y value}
               :program-inputs '[x]
               :program-storage '[y]
               :program-outputs []})
        x (float-array 4)
        y (float-array 4)
        materialized (materialization/materialize plan [x y] evaluate-scalar)]
    (is (= '[x] (mapv :program-value (:bindings plan))))
    (is (= '[y] (mapv :program-value (:storage-bindings plan))))
    (is (= #{'x 'y} (set (keys (:program-buffers materialized)))))
    (is (identical? y (get-in materialized [:program-buffers 'y :source])))
    (is (empty? (:program-scalars materialized)))))

(deftest concrete-public-and-program-shapes-are-checked
  (testing "a static public shape cannot silently accept a different array length"
    (let [value (array-value :double 3)
          plan (invocation/from-prefix
                {:id :static-shape
                 :parameters '[x]
                 :parameter-values {'x value}
                 :bindings [] :binding-values {}
                 :program-values {'x value}
                 :program-inputs '[x] :program-outputs '[x]})]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"length differs"
           (materialization/materialize plan [(double-array 4)] evaluate-scalar)))))
  (testing "a producer and program input cannot disagree after symbolic extents resolve"
    (let [source-value (array-value :double '(extent x0))
          produced-value (array-value :double 'n)
          program-value (array-value :double 'm)
          plan (invocation/from-prefix
                {:id :symbolic-shape
                 :parameters '[x0]
                 :parameter-values {'x0 source-value}
                 :bindings '[[n (int (alength x0))]
                             [m (inc n)]
                             [x (aclone x0)]]
                 :binding-values {'n (scalar :int) 'm (scalar :int) 'x produced-value}
                 :program-values {'n (scalar :int) 'm (scalar :int)
                                  'x program-value 'result program-value}
                 :program-inputs '[n m x]
                 :program-outputs '[result]})]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"producer shape differs"
           (materialization/materialize plan [(double-array 3)] evaluate-scalar))))))
