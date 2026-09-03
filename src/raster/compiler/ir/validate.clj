(ns raster.compiler.ir.validate
  "Shared structural validation helpers for certified plan IRs.

   Distributed plans, adaptive-mesh plans and durable numerical-state manifests all fail loudly
   with one `ex-info` shape: a message, a keyword `:reason`, and structured data. This leaf holds
   that shape and the small predicates the three namespaces previously each redefined, so a
   certificate consumer sees one error contract and a fix to a predicate lands once."
  (:require [clojure.string :as string]))

(defn fail!
  "Throw the plan-IR error contract: `message`, keyword `reason` merged into `data` as `:reason`."
  [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

(defn finite-number?
  [value]
  (and (number? value) (Double/isFinite (double value))))

(defn non-negative-number?
  [value]
  (and (finite-number? value) (not (neg? value))))

(defn positive-number?
  [value]
  (and (finite-number? value) (pos? value)))

(defn non-blank-string?
  [value]
  (and (string? value) (not (string/blank? value))))

(defn unique-by!
  "Fail with `reason` unless `key-fn` is injective over `values`; returns `values`."
  [label reason key-fn values]
  (let [ids (mapv key-fn values)]
    (when-not (= (count ids) (count (distinct ids)))
      (fail! (str label " must have unique identities") reason {:ids ids})))
  values)

(defn exact-keys!
  "Fail with `reason` when `candidate` carries a key outside `allowed`; returns `candidate`."
  [label reason candidate allowed]
  (let [unexpected (vec (remove allowed (keys candidate)))]
    (when (seq unexpected)
      (fail! (str label " contains fields outside its versioned schema")
             reason {:unexpected unexpected :allowed allowed})))
  candidate)
