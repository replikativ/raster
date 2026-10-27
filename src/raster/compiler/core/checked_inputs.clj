(ns raster.compiler.core.checked-inputs
  "Conditional successor facts from the existing global typed invocation CHECKCAST.
   No emitter is loaded, invocation executed, result inferred, or source form changed."
  (:require [raster.compiler.core.call-boundary :as boundary]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.support.mangled :as mangled]))

(defn typed-call-plan
  "Observe an exact global typed call's actual Method without loading a Class or emitter.
   This descriptor is not a whole-call expansion or operand-completion capability."
  ([expression available ineligible] (typed-call-plan expression available ineligible {}))
  ([expression available ineligible context]
  (let [{:keys [dispatch? implementation-op arguments lexical-callee semantic-op]}
        (descriptor/call-description expression)
        planner (some-> (find-ns 'raster.compiler.backend.jvm.bytecode)
                        (ns-resolve 'resolved-typed-invocation-plan))
        operation (when (symbol? implementation-op) (mangled/impl->op implementation-op))]
    (when (and entry/*emission-inputs* planner dispatch? (not lexical-callee)
               (qualified-symbol? implementation-op) operation
               (or (nil? semantic-op) (= operation semantic-op))
               (not (contains? available implementation-op))
               (not (contains? ineligible implementation-op)))
      (let [physical (entry/selected-parameter-carriers operation implementation-op (count arguments))
            interface (some-> ^java.lang.reflect.Method (:invocation-method physical) .getDeclaringClass)
            hint (or (:raster.type/tag (meta implementation-op)) (:tag (meta implementation-op)))
            plan (when (and interface (= (str hint) (.getName ^Class interface)))
                   (entry/observe-definition-input! planner (:ns (meta planner)) 'resolved-typed-invocation-plan)
                   (planner implementation-op (count arguments) {} context interface))
            signature (when (:global? plan)
                        (boundary/selected-typed-signature operation implementation-op
                                                           (count arguments) (:method plan)))
            signature? (= :known (:status signature))]
        (when (and physical signature? (= (:method plan) (:invocation-method physical))
                   (identical? (:interface plan) (.getDeclaringClass ^java.lang.reflect.Method (:method plan)))
                   (identical? (get-in signature [:definition :entry]) (get-in physical [:definition :entry])))
          {:plan plan :physical physical :arguments arguments}))))))

(defn array-refinements
  "New conditional successor bounds only; no presence, ownership, or earlier definition fact."
  [expression available ineligible values]
  (let [arguments (:arguments (descriptor/call-description expression))]
    (when (some #(and (symbol? %) (contains? available %)
                     (not (contains? ineligible %))
                     (or (nil? (get values %)) (= :opaque (:kind (get values %))))) arguments)
      (when-let [{:keys [physical]} (typed-call-plan expression available ineligible)]
          (let [by-argument (group-by first (map vector arguments (:parameter-carriers physical)))]
            (into {}
                  (keep (fn [[argument uses]]
                          (let [carriers (map second uses) carrier (first carriers)
                                storage-type (dtype/dtype-for-jvm-array-class carrier)]
                            (when (and storage-type (symbol? argument) (contains? available argument)
                                       (not (contains? ineligible argument))
                                       (every? #(identical? carrier %) carriers)
                                       ;; Existing tensor/layout contracts are never
                                       ;; overwritten by this conditional cast fact.
                                       (or (nil? (get values argument))
                                           (= :opaque (:kind (get values argument)))))
                              [argument (av/tensor {:dtype storage-type :shape [(list 'extent argument)]
                                                    :representation {:kind :plain}})]))))
                  by-argument))))))
