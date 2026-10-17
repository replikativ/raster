(ns raster.ad.accumulation
  "Checked scalar projection of the existing optional cotangent sum.

   Presence and physical width are independent premises supplied by a checked
   program or lexical producer analysis. This is not type inference, a new
   operation registry, or permission to reassociate contributions."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.dtype :as dtype]))

(defn- retain-approved-premise! []
  (let [v (find-var 'raster.ad.reverse/grad-acc)]
    (method-entry/observe-definition-input! v (:ns (meta v))
                                           'raster.ad.reverse/grad-acc)))

(defn approved-call?
  "An exact call to the unchanged library helper, not its semantic spelling.
   Resolve only an already loaded namespace; never execute a callee."
  [expression]
  (let [{:keys [dispatch? lexical-callee implementation-op arguments]}
        (when (seq? expression) (descriptor/call-description expression))
        loaded? (find-ns 'raster.ad.reverse)
        library (when loaded? (find-var 'raster.ad.reverse/grad-acc))
        guard (when loaded? (find-var 'raster.ad.reverse/approved-grad-acc-definition?))
        actual (when (and (qualified-symbol? implementation-op)
                          (find-ns (symbol (namespace implementation-op))))
                 (resolve implementation-op))]
    (boolean (and (not dispatch?) (not lexical-callee) (= 2 (count arguments))
                  (var? library) (identical? library actual) guard (bound? guard)
                  ((var-get guard))))))

(defn project-checked-addition
  "Return ordinary addition only for independently present homogeneous floating
   operands and a caller-validated replacement. Facts are compiler-local
   {:present? true :dtype ...}, not parameter annotations or public certificates.
   The validator must check ordinary addition at these actual operand widths.
   Unknown, nullable, mixed-width and non-scalar cases remain unprojected."
  [expression operand-facts validate-addition]
  (when (and (= 2 (count operand-facts))
             (every? :present? operand-facts)
             (apply = (map :dtype operand-facts))
             (contains? #{:float :double} (:dtype (first operand-facts)))
             (approved-call? expression))
    (let [dt (:dtype (first operand-facts))
          arguments (:arguments (descriptor/call-description expression))
          ;; Match grad-acc's real branches: Double bypasses extensible numeric
          ;; dispatch; Float uses it. Never replace the Double branch with a
          ;; potentially user-redefined raster.numeric/+ method.
          operation (if (= :double dt) 'clojure.core/+ 'raster.numeric/+)
          addition (with-meta (list* operation arguments)
                     (assoc (select-keys (meta expression) [:line :column :file])
                            :raster.type/tag (dtype/scalar-tag-for-dtype dt)))]
      (when (validate-addition addition dt)
        (retain-approved-premise!)
        addition))))

(defn nullable-result-tag
  "Project the existing result facet only for the approved helper's closed
   Double branch. This proves a present result's carrier, NOT presence: nil
   remains the optional identity. Float uses extensible dispatch, so this
   narrow proof makes no claim about it. Argument tags must be retained
   producer evidence, never the desired cotangent target type."
  [expression argument-tags]
  (when (and (approved-call? expression)
             (= :common-arg-tag (descriptor/result-type-rule 'raster.ad.reverse/grad-acc)))
    (let [tag (descriptor/result-tag 'raster.ad.reverse/grad-acc argument-tags)]
      (when (= 'double tag) tag))))

(defn project-checked-array-addition
  "The approved optional sum's two present homogeneous array inputs reach its
   existing numeric/+ call. The caller must prove that actual selected call's
   normal return and physical carrier. No residual-add substitute or result-tag
   assumption is permitted; unresolved dispatch remains optional."
  [expression operand-facts validate-addition]
  (when (and (approved-call? expression) (= 2 (count operand-facts))
             (every? #(= :present (:presence %)) operand-facts)
             (apply = (map :array-tag operand-facts))
             (contains? '#{floats doubles} (:array-tag (first operand-facts))))
    (let [tag (:array-tag (first operand-facts))
          addition (with-meta
                     (list* 'raster.numeric/+ (:arguments (descriptor/call-description expression)))
                     (assoc (select-keys (meta expression) [:line :column :file])
                            :raster.type/tag tag))]
      (when (validate-addition addition tag)
        (retain-approved-premise!)
        addition))))
