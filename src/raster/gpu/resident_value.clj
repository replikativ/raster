(ns raster.gpu.resident-value
  "Backend-neutral resident composite values used while flattening a logical KernelABI binding.

   Fields remain ordered because physical kernel arguments are positional. A field value may be a
   checked ResidentBufferView before materialization or a backend buffer afterwards."
  (:require [raster.compiler.core.dtype :as dtype]))

(defrecord ResidentComposite [id fields])

(defn resident-composite?
  [value]
  (and value (= "raster.gpu.resident_value.ResidentComposite" (.getName (class value)))))

(defn composite
  [id fields]
  (let [fields (vec fields)
        names (mapv :name fields)]
    (when-not (seq fields)
      (throw (ex-info "resident composite requires at least one physical field"
                      {:reason :resident-composite-empty :id id})))
    (when-not (every? #(and (map? %) (:name %) (contains? % :value)) fields)
      (throw (ex-info "resident composite fields require :name and :value"
                      {:reason :resident-composite-field :id id :fields fields})))
    (when-not (= (count names) (count (distinct names)))
      (throw (ex-info "resident composite field names must be unique and ordered"
                      {:reason :resident-composite-fields :id id :fields names})))
    (->ResidentComposite id fields)))

(defn map-values
  [f composite-value]
  (when-not (resident-composite? composite-value)
    (throw (ex-info "map-values requires a ResidentComposite"
                    {:actual (type composite-value)})))
  (composite (:id composite-value)
             (mapv #(update % :value f) (:fields composite-value))))

(defn select-fields
  "Select the ordered physical fields named by an ABI group.

   A kernel may consume only some leaves of a logical composite. Projected slots select those
   leaves by stable `:field` identity; a field-less ABI retains the exact positional contract."
  [{:keys [binding slots]} fields]
  (let [fields (vec fields)
        field-key #(if (instance? clojure.lang.Named %) (name %) %)]
    (cond
      (= (count slots) (count fields)) fields

      (every? :field slots)
      (let [by-name (into {} (map (juxt (comp field-key :name) identity)) fields)
            selected (mapv #(get by-name (field-key (:field %))) slots)]
        (when (some nil? selected)
          (throw (ex-info "resident composite omits an ABI-projected field"
                          {:binding binding :slots slots :fields (mapv :name fields)})))
        selected)

      :else
      (throw (ex-info "resident composite field count differs from its artifact binding"
                      {:binding binding :expected (count slots) :actual (count fields)
                       :slots slots :fields (mapv :name fields)})))))

(defn expand
  "Validate and flatten a ResidentComposite against one ordered logical ABI group. Backend
   buffers share the `:dtype` contract, so field identity and storage type are checked once here
   before either Level Zero or OpenCL sees physical pointer arguments."
  [{:keys [binding slots] :as group} composite-value]
  (when-not (resident-composite? composite-value)
    (throw (ex-info "expand requires a ResidentComposite"
                    {:binding binding :actual (type composite-value)})))
  (let [fields (select-fields group (:fields composite-value))]
    (doseq [[slot field] (map vector slots fields)]
      (when (and (:field slot) (not= (:field slot) (:name field)))
        (throw (ex-info "resident composite field order differs from its physical ABI slot"
                        {:binding binding :slot slot :field (:name field)})))
      (when-not (= (dtype/canon (:dtype slot))
                   (dtype/canon (:dtype (:value field))))
        (throw (ex-info "resident composite field dtype differs from its physical ABI slot"
                        {:binding binding :slot slot :field (:name field)
                         :expected (:dtype slot) :actual (:dtype (:value field))}))))
    (mapv :value fields)))
