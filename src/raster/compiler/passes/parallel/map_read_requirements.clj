(ns raster.compiler.passes.parallel.map-read-requirements
  "Minimum flat storage derived from actual typed map loads, not from output size."
  (:require [clojure.set :as set]
            [raster.compiler.ir.scalar-range :as ranges]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.segmap-body :as map-body]))

(defn static-read-requirements
  "Optional all-load proof for a plain, positive static, one-dimensional result map.
   The map schedule establishes 0<=index<bound at each :map-active load. Indirect/local-SSA
   coordinates, effects and unknown domains decline. Returned counts are minimum capacities."
  [operation options]
  (when (and (instance? raster.compiler.ir.segop.SegMap operation)
             (:out-sym operation)
             (= 1 (count (get-in operation [:space :dims])))
             (empty? (set/intersection (set (:inputs operation)) (set (:outputs operation))))
             (not (seq (get-in operation [:scalar-region :effects]))))
    (let [{index :name bound :bound} (first (get-in operation [:space :dims]))]
      (when (and (integer? bound) (<= 1 bound Integer/MAX_VALUE))
        (try
          (let [lowered (map-body/lower operation options)
                loads (filter #(= "raster.compiler.ir.kernel_body.ScalarLoad"
                                  (some-> % class .getName))
                              (tree-seq coll? seq (get-in lowered [:kernel-body :operations])))
                requirements
                (mapv (fn [{:keys [buffer coordinates predicate]}]
                        (when (and (= :map-active predicate) (= 1 (count coordinates)))
                          (when-let [range (ranges/typed-index-range
                                           (first coordinates)
                                           (assoc (:scalar-types options) index :long)
                                           {index {:lower 0 :upper (dec bound)}})]
                            (when (<= 0 (:lower range))
                              [buffer (inc' (:upper range))]))))
                      loads)]
            (when (and (seq loads) (every? some? requirements))
              (reduce (fn [result [id extent]] (update result id (fnil max 0) extent))
                      {} requirements)))
          (catch clojure.lang.ExceptionInfo _ nil))))))
