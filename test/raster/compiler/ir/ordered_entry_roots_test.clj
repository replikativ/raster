(ns raster.compiler.ir.ordered-entry-roots-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]))

(defn- buffer-value [id]
  (av/tensor {:dtype :double :shape [(list 'extent id)]}))

(defn- fixture
  ([] (fixture (double-array [1.0 2.0]) (double-array 2)))
  ([input output]
   (let [values (into {} (map (fn [id] [id (buffer-value id)])) '[input middle late output unused])
         plan (invocation/from-prefix
               {:id :late-entry-root :parameters '[input output unused]
                :parameter-values (select-keys values '[input output unused])
                :bindings '[[middle input] [late middle]]
                :binding-values (select-keys values '[middle late])
                :program-values (select-keys values '[late output])
                :program-inputs '[late output] :program-outputs '[output]})
         arguments [input output (double-array 1)]]
     {:arguments arguments
      :state {:materialization (materialization/begin-materialization plan arguments)
              ;; Original input has no numerical AV; captured layout must use its actual
              ;; validated physical parameter contract, not fabricate a new logical value.
              :program {:values (select-keys values '[late output])}
              :target :test}})))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- with-accesses [accesses f]
  ;; Isolate entry policy from numerical certificate construction. Device/public integration
  ;; exercises the ordinary checked emitted boundary; this unit controls only its access result.
  (with-redefs-fn {#'link/checked-ordered-validation (constantly {})
                  #'link/equation-boundaries (fn [& _] [{:buffer-accesses accesses}])}
    f))

(deftest later-aliases-keep-original-entry-roots-without-completion
  (let [{:keys [state]} (fixture)
        available (materialization/prefix-bindings (:materialization state))
        buffers (link/ordered-entry-buffers state)]
    (is (= #{'output} (set (keys (:program-buffers available)))))
    (is (= #{'input 'output} (set (keys buffers))))
    (is (= {'late 'input 'output 'output}
           (#'link/ordered-entry-program-roots state buffers)))
    (is (zero? (get-in state [:materialization :next-step])))
    (is (not (contains? buffers 'unused)))
    (with-accesses {'late :read 'output :write}
      #(do (is (= {'input :input 'output :state} (link/ordered-entry-roles state nil)))
           (is (= :invocation-ordered-entry-roles
                  (reason (fn [] (link/ordered-entry-roles state {'unused :constant})))))))))

(deftest later-alias-writes-and-captured-role-checks-use-the-same-root
  (let [{:keys [state]} (fixture)]
    (with-accesses {'late :write 'output :read}
      #(do (is (= {'input :state 'output :input} (link/ordered-entry-roles state nil)))
           (doseq [role [:input :constant]]
             (is (= :link-write-read-only
                    (reason (fn [] (link/ordered-entry-roles state {'input role}))))))))
    (let [roles {'input :constant 'output :input}
          buffers (link/ordered-entry-buffers state)
          layout (link/ordered-entry-layout state roles)
          input-token (:id (get buffers 'input))]
      (is (= {input-token :constant (:id (get buffers 'output)) :input} (:entry-refresh layout)))
      (is (= [input-token] (mapv :id (:nodes layout))))
      (is (identical? (:source (get buffers 'input)) (:source (first (:nodes layout)))))
      (is (= :invocation-public-role-alias
             (reason #(link/ordered-entry-layout state {'output :input})))))))

(deftest late-entry-input-replacement-retains-specialization-and-alias-checks
  (let [{:keys [state arguments]} (fixture)
        replacement (double-array [3.0 4.0])
        roles {'input :input 'output :state}
        updated (link/ordered-input-arguments state arguments roles {'input replacement})]
    (is (identical? replacement (first updated)))
    (is (identical? (second arguments) (second updated)))
    (is (= :invocation-ordered-input-role
           (reason #(link/ordered-input-arguments state arguments
                                                 (assoc roles 'input :constant) {'input replacement}))))
    (is (= :invocation-ordered-input-specialization
           (reason #(link/ordered-input-arguments state arguments roles {'input (double-array 1)}))))
    (is (= :invocation-ordered-source-alias
           (reason #(link/ordered-input-arguments state arguments roles {'input (second arguments)}))))))
