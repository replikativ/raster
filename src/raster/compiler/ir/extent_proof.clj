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

(defn- opaque-value [id value]
  (when (and (= :tensor (:kind value)) (= [] (:shape value))
             (integral? (:dtype value)))
    {:dtype (:dtype value) :product (algebra/monomial id)}))

(defn initial-environment
  "Only incoming scalar values are available before the first equation."
  [program]
  (let [facts (dialect/facts program)]
    (into {} (keep (fn [id]
                     (when-let [proof (opaque-value id (get-in facts [:values id]))]
                       [id proof]))) (:inputs facts))))

(defn advance
  "Extend witnesses after one equation executes. Results become available only here; unknown
   integral results retain opaque SSA identity, never guessed arithmetic."
  [environment facts equation]
  (let [results (into {} (keep (fn [id]
                                (when-let [proof (opaque-value id (get-in facts [:values id]))]
                                  [id proof]))) (nth equation 2))
        after (merge environment results)]
     (let [{:keys [kind captures lambda]} (dialect/operation-parts equation)]
       (if (not= 'scalar kind)
         after
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
                   after (map vector (nth equation 2) body-results)))))))

(defn environment
  "Witnesses available after the complete program. For a consumer, use initial-environment
   and advance only through its predecessors instead."
  [program]
  (reduce #(advance %1 (dialect/facts program) %2)
          (initial-environment program) (dialect/equations program)))

(defn same-volume?
  "Whether allocation extent equals the entire dense result shape. No capacity inequalities."
  [environment extent shape]
  (and (vector? shape)
      (try
        (let [allocation (:product (expression environment extent))
              dimensions (map #(-> (expression environment %) :product) shape)]
          (boolean (and allocation (every? some? dimensions)
                        (= allocation (apply algebra/product dimensions)))))
        (catch ArithmeticException _ false))))
