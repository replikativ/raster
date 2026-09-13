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
                operations (tree-seq coll? seq (get-in lowered [:kernel-body :operations]))
                loads (filter #(= "raster.compiler.ir.kernel_body.ScalarLoad"
                                  (some-> % class .getName)) operations)
                ;; A lexical map local may be retained as typed SSA rather than beta-expanded
                ;; into the load coordinate.  Reuse only the range certificates carried by its
                ;; validated ScalarCompute; otherwise a harmless `(let* [k (+ i j)] (aget a k))`
                ;; loses the same capacity proof as its inlined spelling.
                computed-ranges
                (reduce
                 (fn [known node]
                   (if (= "raster.compiler.ir.kernel_body.ScalarCompute"
                          (some-> node class .getName))
                     (let [id (get-in node [:result :id])
                           type (get-in node [:result :type])
                           expression (:expression node)
                           proof (get-in expression [:options :proof])
                           argument-range
                           (fn [argument]
                             (cond
                               (symbol? argument) (get-in known [argument :range])
                               (and (integer? (:value argument)) (:type argument))
                               (ranges/literal (:value argument) (:type argument))
                               :else nil))
                           arguments (mapv argument-range (:arguments expression))
                           derived
                           (or (when (and (= :typed-scalar-range (:kind proof))
                                          (ranges/contained-in-dtype? proof type))
                                 (select-keys proof [:lower :upper]))
                               (when (every? some? arguments)
                                 (case (:op expression)
                                   (:+ :- :*)
                                   (let [range (ranges/arithmetic (:op expression) arguments)]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :quot
                                   (let [[numerator divisor] arguments
                                         range (when (and (<= 0 (:lower numerator))
                                                          (= (:lower divisor) (:upper divisor))
                                                          (pos? (:lower divisor)))
                                                 (ranges/quotient arguments))]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :rem
                                   (let [[numerator divisor] arguments
                                         d (:lower divisor)
                                         range (when (and (<= 0 (:lower numerator))
                                                          (= d (:upper divisor)) (pos? d))
                                                 {:lower 0
                                                  :upper (min (:upper numerator) (dec d))})]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :cast
                                   (let [range (first arguments)]
                                     (when (and (= :exact (get-in expression [:options :overflow]))
                                                (ranges/contained-in-dtype? range type))
                                       range))
                                   nil)))]
                       (if (and (symbol? id) derived)
                         (assoc known id {:type type :range derived})
                         known))
                     known))
                 {index {:type :long :range {:lower 0 :upper (dec bound)}}}
                 operations)
                leaf-types (merge (:scalar-types options)
                                  {index :long}
                                  (into {} (map (fn [[id fact]] [id (:type fact)]))
                                        computed-ranges))
                leaf-ranges (merge {index {:lower 0 :upper (dec bound)}}
                                   (into {} (map (fn [[id fact]] [id (:range fact)]))
                                         computed-ranges))
                requirements
                (mapv (fn [{:keys [buffer coordinates predicate]}]
                        (when (and (= :map-active predicate) (= 1 (count coordinates)))
                          (when-let [range (ranges/typed-index-range
                                           (first coordinates)
                                           leaf-types leaf-ranges)]
                            (when (<= 0 (:lower range))
                              [buffer (inc' (:upper range))]))))
                      loads)]
            (when (and (seq loads) (every? some? requirements))
              (reduce (fn [result [id extent]] (update result id (fnil max 0) extent))
                      {} requirements)))
          (catch clojure.lang.ExceptionInfo _ nil))))))
