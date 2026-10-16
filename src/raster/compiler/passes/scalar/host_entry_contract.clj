(ns raster.compiler.passes.scalar.host-entry-contract
  "Project bounded, unconditional entry assertions onto original host argument objects.
   This is not a general identity optimizer or a physical storage alias proof."
  (:require [raster.compiler.core.util :as util]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.passes.scalar.effects :as effects]))

(defn- closed-data?
  [value]
  (cond
    (or (nil? value) (boolean? value) (number? value) (string? value)
        (char? value) (keyword? value)) true
    (and (seq? value) (= 'quote (first value)) (= 2 (count value))) true
    (map? value) (every? closed-data? (mapcat identity value))
    (or (vector? value) (set? value)) (every? closed-data? value)
    :else false))

(defn- entry-assertion
  [expression environment public-arrays]
  (when (and (seq? expression) (= 'if (first expression)) (#{3 4} (count expression))
             (nil? (nth expression 3 nil)))
    (let [[_ predicate failure] expression
          left (when (seq? predicate) (second predicate))
          right (when (seq? predicate) (nth predicate 2 nil))]
      (when (and (seq? predicate) (= 3 (count predicate))
                 (= 'clojure.core/identical? (first predicate))
                 (not (:raster.op/lexical-callee (meta predicate)))
                 (not (contains? environment 'clojure.core/identical?))
                 (contains? public-arrays left) (contains? public-arrays right)
                 (not= left right)
                 (seq? failure) (= 'throw (first failure)) (= 2 (count failure)))
        (let [exception (second failure)]
          (when (and (seq? exception) (= 3 (count exception))
                     (= 'clojure.core/ex-info (first exception))
                     (not (:raster.op/lexical-callee (meta exception)))
                     (not (contains? environment 'clojure.core/ex-info))
                     (string? (second exception)) (map? (nth exception 2))
                     (closed-data? (nth exception 2)))
            (let [[left right] (sort-by str [left right])]
              {:op :host-object-distinct :scope :original-host-arguments
               :left left :right right})))))))

(defn project-entry-assertions
  "Return source plus checked preconditions. Only unused assertion bindings in a
   flat let entry prefix are removed. Cross only replay-safe values; reads,
   arithmetic, writes, calls and nested control close the prefix. Unmatched
   assertions remain source and acquire no invented assumptions. Parameter IDs
   must be canonical, distinct public array identities from the retained ABI."
  [source parameter-environment public-arrays]
  (when-not (and (map? parameter-environment) (vector? public-arrays)
                 (= (count public-arrays) (count (set public-arrays)))
                 (every? #(and (symbol? %) (contains? parameter-environment %)
                                (contains? types/primitive-array-tags
                                           (get parameter-environment %))) public-arrays))
    (throw (ex-info "Entry object conditions require distinct declared primitive-array parameters"
                    {:reason :invalid-host-entry-parameters})))
  (let [arrays (set public-arrays)]
    (if-not (and (seq? source) (contains? #{'let 'let*} (first source))
                 (vector? (second source)) (even? (count (second source)))
                 (every? symbol? (take-nth 2 (second source))))
      {:form source :preconditions []}
      (loop [remaining (seq (partition 2 (second source)))
             retained [] environment parameter-environment conditions [] open? true]
        (if-let [[id init] (first remaining)]
          (let [condition (when (and open? (not (contains? environment id)))
                            (entry-assertion init environment arrays))
                used? (when condition
                        (let [tail (list* 'let* (vec (mapcat identity (next remaining)))
                                          (nnext source))]
                          (binding [util/*shadowing-locals* (into (set (keys environment)) [id])]
                            (contains? (util/free-syms tail) id))))
                remove? (and condition (not used?))
                safe? (and (not (contains? environment id))
                           (effects/replay-safe-value? init environment))]
            (recur (next remaining)
                   (if remove? retained (conj retained id init))
                   (assoc environment id (:raster.type/tag (meta id)))
                   (cond-> conditions remove? (conj condition))
                   (and open? (or remove? safe?))))
          (let [conditions (vec (distinct conditions))]
            (precondition/validate! conditions #{} {:host-object-identities arrays})
            {:form (with-meta (list* (first source) (vec retained) (nnext source)) (meta source))
             :preconditions conditions}))))))
