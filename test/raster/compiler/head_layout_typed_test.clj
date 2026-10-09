(ns raster.compiler.head-layout-typed-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.report :as report]
            [raster.dl.array-ops :as ops]))

(deftest head-layout-permutations-use-the-direct-typed-route
  (doseq [[operation scalar-equations]
          [[#'ops/pack-heads 2]
           [#'ops/unpack-heads 2]
           [#'ops/pack-heads-strided 2]
           [#'ops/unpack-heads-strided 2]]]
    (let [compiled (pipeline/show-pipeline operation :target-device :ze:0 :dtype :float)
          compiler-report (report/from-pipeline compiled)]
      (is (= {:backend :opencl :source-dialect :typed-soac
              :typed-validated true :declines []}
             (:route compiler-report))
          (str (:name (meta operation)) " route"))
      (is (= {:segops 1 :kernel-graphs 0 :structured-loops 0
              :typed-reused 1 :typed-scalar-equations scalar-equations
              :backend-reused 1 :backend-relowered 0 :fallback 0}
             (:lowering compiler-report))
          (str (:name (meta operation)) " lowering"))
      (is (= 1 (count (:kernels compiled))))
      (is (= #{['out :float] ['x :float]}
             (->> (:abi (first (:kernels compiled)))
                  (remove #(= :scalar (:kind %)))
                  (map (juxt :name :dtype))
                  set))
          "the typed schedule retains one input and one physical destination")
      (when (= operation #'ops/unpack-heads-strided)
        (let [bindings (into {} (map vec (partition 2 (second (get-in compiled [:soac-fused :source])))))
              allocation (get bindings 'out)
              map-body (last (second (get-in compiled [:soac-fused :source])))
              allocation-extent (last allocation)
              launch-extent (nth map-body 3)]
          (is (symbol? allocation-extent))
          (is (contains? bindings allocation-extent)
              "the checked extent is still evaluated at its dominating source binding")
          (is (= allocation-extent launch-extent)
              "allocation and launch reuse the same checked SSA extent, not a duplicate equation"))))))
