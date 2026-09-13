(ns raster.compiler.ir.extent-proof
  "Conservative product equality over retained integral scalar SSA.

   This is a proof, not an expression rewrite: scalar evaluation order and overflow checks
   stay in the program. Narrowing casts, floating arithmetic and unchecked products remain
   opaque. Only successful checked Long multiplication admits mathematical factorization."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.index-algebra :as algebra]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn- integral? [t] (contains? #{:byte :short :int :long} t))

(defn- expression [environment form]
  (cond
    (symbol? form) (get environment form)
    (and (integer? form) (<= 0 form Long/MAX_VALUE))
    {:dtype :long :product (algebra/monomial form)}
    (seq? form)
    (let [op (descriptor/semantic-op form)
          operands (mapv #(expression environment %) (descriptor/call-args form))]
      (cond
        (and (descriptor/cast-op? op) (= 1 (count operands)))
        (let [operand (first operands)
              target (some-> (descriptor/cast-result-tag op) dtype/dtype-for-scalar-tag)]
          (when (and (integral? (:dtype operand)) (integral? target)
                     (<= (dtype/bytes-of (:dtype operand)) (dtype/bytes-of target)))
            (assoc operand :dtype target)))
        ;; Raster's Integer multiplication wraps; its Long specialization is checked.
        (and (descriptor/multiplication-op? op) (seq operands)
             (every? #(and (= :long (:dtype %)) (:product %)) operands))
        (try {:dtype :long :product (apply algebra/product (map :product operands))}
             (catch ArithmeticException _ nil))))))

(defn environment
  "Build product witnesses from typed scalar equations. Unknown integral values are opaque
   factors; an unproved definition keeps its own SSA identity, never its guessed arithmetic."
  [program]
  (reduce
   (fn [environment equation]
     (let [{:keys [kind captures lambda]} (dialect/operation-parts equation)]
       (if (not= 'scalar kind)
         environment
         (let [{:keys [parameters locals body-results]} (dialect/lambda-parts lambda)
               local-env (merge environment (zipmap parameters (map environment captures)))
               local-env (reduce (fn [env {:keys [id dtype init]}]
                                   (assoc env id
                                          (when-let [proof (expression env init)]
                                            (when (= dtype (:dtype proof)) proof))))
                                 local-env locals)]
           (reduce (fn [env [id form]]
                     (let [proof (expression local-env form)]
                       (if (and proof (= (:dtype (get env id)) (:dtype proof)))
                         (assoc env id proof) env)))
                   environment (map vector (nth equation 2) body-results))))))
   (into {} (keep (fn [[id value]]
                    (when (and (= :tensor (:kind value)) (= [] (:shape value))
                               (integral? (:dtype value)))
                      [id {:dtype (:dtype value) :product (algebra/monomial id)}])))
         (:values (dialect/facts program)))
   (dialect/equations program)))

(defn same-volume?
  "Whether allocation extent equals the entire dense result shape. No capacity inequalities."
  [environment extent shape]
  (and (vector? shape)
   (or (= [extent] shape)
      (try
        (let [allocation (:product (expression environment extent))
              dimensions (map #(-> (expression environment %) :product) shape)]
          (boolean (and allocation (every? some? dimensions)
                        (= allocation (apply algebra/product dimensions)))))
        (catch ArithmeticException _ false)))))
