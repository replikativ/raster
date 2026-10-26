(ns raster.compiler.ir.ordered-entry-roots-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as soac]))

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
              :source-cursor 0 :acquired-storage #{}
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

(deftest resident-entry-preflight-returns-only-source-free-descriptors
  (let [{:keys [state arguments]} (fixture)
        roles {'input :input 'output :state}
        original (get (link/ordered-entry-buffers state) 'input)
        {:keys [state resident-inputs resident-realized]}
        (link/ordered-input-initial-state state arguments roles {} #{'input})
        current (get (link/ordered-entry-buffers state) 'input)
        descriptor (get resident-inputs (:id original))]
    (is (identical? (:source original) (first arguments)))
    (is (nil? (:source current)))
    (is (= :unspecified (:initialization current)))
    (is (= #{(:id original)} (set (keys resident-inputs))))
    (is (= ['input] (:values descriptor)))
    (is (= current (:buffer descriptor)))
    (is (= :input (get-in descriptor [:node :role])))
    (is (nil? (get-in descriptor [:node :source])))
    (is (= (:shape original) (get-in descriptor [:node :view :shape])))
    (is (not (contains? descriptor :initialized)))
    (is (not (contains? descriptor :ready?)))
    (is (= (:id original) (get-in resident-realized [:buffers 'input])))
    (is (= #{'input} (get-in resident-realized [:storage (:id original) :compiler-values])))
    (is (nil? (get-in resident-realized [:storage (:id original) :source])))
    (is (= :unspecified (get-in resident-realized [:storage (:id original) :initialization])))
    (is (empty? (:acquired-storage state)))
    (is (empty? (get-in state [:realized :storage])))
    (is (= :invocation-materialization-source-free-arguments
           (reason #(materialization/initial-parameter-arguments (:materialization state)))))))

(deftest resident-entry-preflight-refuses-host-content-consumers
  (let [{:keys [state arguments]} (fixture)
        roles {'input :input 'output :state}
        captures [(assoc state :program
                         (assoc (:program state) :equations
                                [{:id :host-read :operands ['late]
                                  :attributes {:host-only true}}]))
                  (assoc-in state [:program :attributes :host-read-values] ['late])]]
    (doseq [host-capture captures]
      (is (= :invocation-ordered-resident-host-use
             (reason #(link/ordered-input-initial-state
                       host-capture arguments roles {} #{'input}))))))
  (let [input (double-array 2)
        output (double-array 2)
        value (buffer-value 'input)
        clone-value value
        plan (invocation/from-prefix
              {:id :resident-clone-refusal
               :parameters '[input output]
               :parameter-values {'input value 'output (buffer-value 'output)}
               :bindings '[[clone (aclone input)]] :binding-values {'clone clone-value}
               :program-values {'input value 'clone clone-value 'output (buffer-value 'output)}
               :program-inputs '[input clone output] :program-outputs '[output]})
        state {:materialization (materialization/begin-materialization plan [input output])
               :program {:values {'input value 'clone clone-value 'output (buffer-value 'output)}}
               :source-cursor 0 :acquired-storage #{}
               :target :test}]
    (is (= :invocation-ordered-resident-host-use
           (reason #(link/ordered-input-initial-state
                     state [input output] {'input :input 'output :state} {} #{'input}))))))

(deftest resident-entry-preflight-allows-only-the-exact-plan-shape-query
  (let [input (double-array 3)
        output (double-array 3)
        input-value (buffer-value 'input)
        output-value (buffer-value 'output)
        length-value (av/tensor {:dtype :long :shape []})
        values {'input input-value 'alias input-value 'length length-value 'output output-value}
        expression '(clojure.core/long (clojure.core/alength alias))
        algorithm (soac/make
                   (soac/default-program-facts
                    {:values values :inputs '[alias]
                     :equations {:length (soac/default-equation-facts)}})
                   [(list '= :length '[length]
                          (list 'scalar {:dtypes [:long]} '[alias]
                                (soac/lambda-form '[alias] [expression])))]
                   '[length])
        equation (parallel-program/map->ProgramEquation
                  {:id 3 :site [:binding 'length] :source expression
                   :operands '[alias] :results '[length] :algorithm algorithm
                   :operations [] :effects #{} :provenance {:source-binding-id 0}
                   :attributes {:host-only true}})
        plan (invocation/from-prefix
              {:id :resident-shape-projection
               :parameters '[input output]
               :parameter-values (select-keys values '[input output])
               :bindings '[[alias input] [length (clojure.core/long
                                                  (clojure.core/alength alias))]]
               :binding-values (select-keys values '[alias length])
               :program-values values :program-inputs '[input alias length output]
               :program-outputs '[output]})
        state {:materialization (materialization/begin-materialization plan [input output])
               :program {:values values :equations [equation]
                         :attributes {:host-read-values '[alias]
                                      :host-read-sites [{:source-binding-id 0
                                                         :values #{'alias}}]}}
               :source-cursor 0 :acquired-storage #{} :target :test}]
    (is (= #{'input}
           (-> (link/ordered-input-initial-state
                state [input output] {'input :input 'output :state} {} #{'input})
               :resident-inputs vals first :values set)))
    (doseq [content-state
            [(update-in state [:program :attributes :host-read-sites]
                        conj {:source-binding-id 3 :values #{'alias}})
             (assoc-in state [:program :attributes :native-initialization-providers] ['alias])]]
      (is (= :invocation-ordered-resident-host-use
             (reason #(link/ordered-input-initial-state
                       content-state [input output] {'input :input 'output :state}
                       {} #{'input})))))
    (is (= :invocation-ordered-resident-position
           (reason #(link/ordered-input-initial-state
                     (assoc state :source-cursor 1) [input output]
                     {'input :input 'output :state} {} #{'input}))))))
