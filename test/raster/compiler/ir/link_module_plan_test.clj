(ns raster.compiler.ir.link-module-plan-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.parallel-program-opencl :as program-opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-module-plan :as module]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.structured-control-route :as route]))

(defn- reason [invoke]
  (try (invoke) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- node
  ([id role] (node id role nil))
  ([id role source]
   (link/node {:id id :dtype :float :shape [4] :device :ze:0
               :role role :source source})))

(def ^:private kernel
  (artifact/make
   {:kernel-name "module_legacy_map"
    :source "__kernel void module_legacy_map(float* x, float* y, long n) {}"
    :abi [(kabi/slot 'x :input :float) (kabi/slot 'y :output :float)
          (kabi/slot 'n :scalar :long)]
    :arguments '[x y n]
    :launch (launch/spec {:workgroup-size [4] :group-count [1]})
    :effects {:kind :map :reads '[x] :writes '[y]}}))

(defn- descriptor []
  {:dtype :float :all-params '[x n] :array-params '[x] :scalar-params '[n]
   :array-roles {'x :input}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [_] 4)}]
   :steps [{:phase :map :kernel-name "module_legacy_map" :convention :map
            :artifact kernel
            :argument-specs [{:kind :input :sym 'x} {:kind :output :sym 'y}
                             {:kind :scalar :type :long :value-fn (fn [_] 4)}]}]
   :result-sym 'y})

(defn- instance [id input output]
  (link/instance {:id id :descriptor (descriptor)
                  :bindings {'x input 'y output} :scalars {'n 4}}))

(defn- legacy-spine []
  (link/make {:id :module-spine :target :ze:0
              :nodes [(node :input :input (float-array 4))
                      (node :residual :internal) (node :result :internal)]
              :instances [(instance :primal :input :residual)
                          (instance :pullback :residual :result)]
              :outputs []}))

(defn- external-state-spine []
  (link/make {:id :external-state-spine :target :ze:0
              :nodes [(node :input :input (float-array 4))
                      (node :state :state (float-array 4))]
              :instances [(instance :advance :input :state)]
              :outputs []}))

(defn- ordered-external-spine [node-order]
  (let [parameters (mapv #(symbol (str "ordered_input_" %)) (range (count node-order)))
        output 'ordered_output
        artifact
        (artifact/make
         {:kernel-name "module_ordered_inputs"
          :source (str "__kernel void module_ordered_inputs("
                       (str/join ", " (map #(str "float* " %) (conj parameters output)))
                       ") {}")
          :abi (conj (mapv #(kabi/slot % :input :float) parameters)
                     (kabi/slot output :output :float))
          :arguments (conj parameters output)
          :launch (launch/spec {:workgroup-size [1] :group-count [1]})
          :effects {:kind :map :reads parameters :writes [output]}})
        descriptor
        {:dtype :float :all-params parameters :array-params parameters :scalar-params []
         :array-roles (zipmap parameters (repeat :input))
         :allocs [{:sym output :dtype :float :size-fn (fn [_] 1)}]
         :steps [{:phase :map :kernel-name "module_ordered_inputs" :convention :map
                  :artifact artifact
                  :argument-specs
                  (conj (mapv (fn [parameter] {:kind :input :sym parameter}) parameters)
                        {:kind :output :sym output})}]
         :result-sym output}
        bindings (assoc (zipmap parameters node-order) output :ordered-state)]
    (link/make
     {:id :ordered-external-spine :target :ze:0
      :nodes (conj (mapv #(link/node {:id % :dtype :float :shape [1] :device :ze:0
                                      :role :input :source (float-array 1)})
                         node-order)
                   (link/node {:id :ordered-state :dtype :float :shape [1]
                               :device :ze:0 :role :state :source (float-array 1)}))
      :instances [(link/instance {:id :ordered-step :descriptor descriptor
                                  :bindings bindings :scalars {}})]
      :outputs []})))

(defn- shared-node [id shape offset]
  (link/node {:id id :dtype :float :shape shape :device :ze:0 :role :internal
              :allocation-id :shared :byte-size 16 :byte-offset offset}))

(defn- semantic-map-instance [instance-id input-token output-token input-ownership]
  (let [input (av/tensor (cond-> {:dtype :float :shape [4]}
                           input-ownership (assoc :ownership input-ownership)))
        output (av/tensor {:dtype :float :shape [4]})
        equation (list '= 'map-0 '[y]
                       (list 'map {:index 'i :extent 4} '[x] []
                             (soac/lambda-form '[element] '[element])))
        typed (soac/make
               (soac/default-program-facts
                {:values {'x input 'y output}
                 :inputs '[x]
                 :equations {'map-0 (soac/default-equation-facts)}})
               [equation] '[y])
        promoted (route/promote-typed-algorithm typed)
        options {:target-device :ocl:0 :dtype :float :array-types {'x :float}}
        scheduled (route/schedule-program promoted options)
        emitted (:program (program-opencl/emit-program scheduled options))
        call (program-call/make emitted {'x input-token 'y output-token} {} {} nil)]
    (link/program-instance {:id instance-id :call call})))

(defn- semantic-module-spine []
  (let [abstract (av/tensor {:dtype :float :shape [4] :ownership :owned
                             :memory-space :device :placement {:device :ocl:0}})]
    (link/make
     {:id :semantic-module-spine :target :ocl:0
      :nodes [(link/node {:id :input :dtype :float :shape [4] :device :ocl:0
                          :role :input :source (float-array 4)})
              (link/node {:id :residual :dtype :float :shape [4] :device :ocl:0
                          :role :internal})
              (link/node {:id :result :dtype :float :shape [4] :device :ocl:0
                          :role :internal})]
      :values [(link/value {:id :input :abstract abstract
                            :leaves [{:name :value :node :input}]})
               (link/value {:id :residual :abstract abstract
                            :leaves [{:name :value :node :residual}]})
               (link/value {:id :result :abstract abstract
                            :leaves [{:name :value :node :result}]})]
      :instances [(semantic-map-instance :primal :input :residual nil)
                  (semantic-map-instance :pullback :residual :result :borrowed)]
      :outputs []})))

(deftest strict-relative-analysis-uses-only-certified-complete-views
  (let [nodes {:parent (shared-node :parent [4] 0)
               :left (shared-node :left [2] 0)
               :right (shared-node :right [2] 8)
               :gap-left (shared-node :gap-left [1] 0)
               :gap-right (shared-node :gap-right [2] 8)}
        analyze (fn [facts] (#'link/relative-entry-contract nodes facts))
        write (fn [node complete produced]
                {:facts [{:node node :access :write}]
                 :complete-writes complete :produced-views produced})
        read (fn [node] {:facts [{:node node :access :read}]
                         :complete-writes #{} :produced-views #{}})]
    (testing "one complete parent initializes a child"
      (is (= #{} (:requires (analyze [(write :parent #{:parent} #{})
                                      (read :left)])))))
    (testing "dense complete children jointly initialize their parent"
      (is (= #{} (:requires (analyze [(write :left #{:left} #{})
                                      (write :right #{:right} #{})
                                      (read :parent)])))))
    (testing "separated complete children do not initialize bytes in their gap"
      (is (= #{:parent}
             (:requires (analyze [(write :gap-left #{:gap-left} #{})
                                  (write :gap-right #{:gap-right} #{})
                                  (read :parent)])))))
    (testing "one complete child does not initialize the untouched parent"
      (is (= #{:parent} (:requires (analyze [(write :left #{:left} #{})
                                             (read :parent)])))))
    (testing "legacy and partial writes preserve a caller requirement"
      (is (= #{:parent} (:requires (analyze [(write :parent nil #{})]))))
      (is (= #{:parent} (:requires (analyze [(write :parent #{} #{})])))))
    (testing "a certified child result does not manufacture a parent requirement"
      (is (= {:requires #{} :full-produces #{:left}}
             (select-keys (analyze [(write :parent #{} #{:left})])
                          [:requires :full-produces]))))))

(deftest strict-relative-requirements-retain-first-fact-order
  (let [node-order [:node-11 :node-2 :node-9 :node-0 :node-7 :node-4
                    :node-10 :node-1 :node-8 :node-3 :node-6 :node-5]
        nodes (into {}
                    (map
                     (fn [node-id]
                       [node-id
                        (link/node {:id node-id :dtype :float :shape [1]
                                    :device :ze:0 :role :internal
                                    :allocation-id node-id :byte-size 4
                                    :byte-offset 0})])
                     node-order))
        facts (into (mapv (fn [node-id] {:node node-id :access :read}) node-order)
                    [{:node :node-9 :access :write}])
        analyze #(#'link/relative-entry-contract
                   nodes [{:facts facts :complete-writes #{} :produced-views #{}}])
        first-contract (analyze)
        second-contract (analyze)]
    (is (= node-order (:require-order first-contract)))
    (is (= first-contract second-contract))
    (is (= :read-write
           (second (nth (#'link/ordered-step-accesses facts) 2))))
    (is (= node-order (mapv first (#'link/ordered-step-accesses facts))))
    (let [built (module/make
                 {:id :ordered-module :spine (ordered-external-spine node-order)
                  :entries [{:name :ordered :instances [:ordered-step]}]})
          expected (conj node-order :ordered-state)]
      (is (= expected (get-in built [:entries 0 :contract :require-order])))
      (is (= built (module/validate! (module/validate! built)))))))

(deftest module-validation-freshly-enforces-partition-and-complete-producer-authority
  (let [spine (legacy-spine)
        request {:id :training-module :spine spine
                 :entries [{:name :primal :instances [:primal]}
                           {:name :pullback :instances [:pullback]}]}]
    (is (= :link-module-frontier-producer (reason #(module/make request)))
        "a legacy ABI writer cannot initialize a retained residual")
    (is (= :link-relative-entry-partition
           (reason #(module/make (assoc request :entries
                                        [{:name :pullback :instances [:pullback]}
                                         {:name :primal :instances [:primal]}])))))
    (is (= :link-relative-entry-partition
           (reason #(module/make (assoc request :entries
                                        [{:name :primal :instances [:primal]}])))))
    (is (= :link-module-entry-names
           (reason #(module/make (assoc request :entries
                                        [{:name :same :instances [:primal]}
                                         {:name :same :instances [:pullback]}])))))
    (is (= :link-module-entry
           (reason #(module/make
                     (assoc request :entries
                            [{:name (object-array 0) :instances [:primal]}
                             {:name :pullback :instances [:pullback]}])))))
    (let [{:keys [effect-evidence]}
          (link/validate-with-effect-evidence! spine nil nil)
          changed-math
          {:scalar-math {:overrides {[:tanh :float]
                                     :f64-target-library-rte-f32}}}]
      (is (= :link-relative-effect-evidence
             (reason #(link/relative-entry-contracts!
                       spine effect-evidence [[:primal :pullback]] changed-math)))))))

(deftest semantic-program-instances-derive-an-owned-residual-for-a-borrowed-consumer
  (let [spine (semantic-module-spine)
        built (module/make
               {:id :semantic-module :spine spine
                :entries [{:name :primal :instances [:primal]}
                          {:name :pullback :instances [:pullback]}]})]
    (is (= built (module/validate! (module/validate! built))))
    (is (= [{:value :residual :nodes [:residual]
             :producer :primal :consumer :pullback}]
           (:frontiers built)))
    (is (= [{:value :residual :nodes [:residual] :producer :primal
             :consumers #{:pullback}}]
           (:retained-groups built)))
    (is (= :owned (get-in built [:spine :values :residual :abstract :ownership])))
    (is (= :borrowed
           (get-in built [:spine :instances 1 :call :program :values 'x :ownership])))
    (is (= #{:residual} (get-in built [:entries 0 :contract :full-produces])))
    (is (= #{:residual} (get-in built [:entries 1 :contract :frontier-requires])))))

(deftest whole-logical-frontiers-and-unique-producers-are-derived-not-declared
  (let [spine (legacy-spine)
        primal (module/->LinkModuleEntry
                :primal [:primal]
                {:writes #{:residual} :full-produces #{:residual} :frontier-requires #{}})
        pullback (module/->LinkModuleEntry
                  :pullback [:pullback]
                  {:writes #{:result} :full-produces #{:result}
                   :frontier-requires #{:residual}
                   :frontier-require-order [:residual]})
        derived (#'module/derive-frontiers spine [primal pullback])]
    (is (= [{:value :residual :nodes [:residual]
             :producer :primal :consumer :pullback}]
           (:frontiers derived)))
    (is (= {:value :residual :nodes [:residual] :producer :primal
            :consumers #{:pullback}}
           (get-in derived [:groups [:primal :residual]])))
    (let [duplicate (assoc primal :name :other)]
      (is (= :link-module-frontier-producer
             (reason #(#'module/derive-frontiers spine
                        [primal duplicate pullback])))))
    (let [later-writer (module/->LinkModuleEntry
                        :later [:later]
                        {:writes #{:residual} :full-produces #{}
                         :frontier-requires #{}})]
      (is (= :link-module-frontier-writers
             (reason #(#'module/derive-frontiers spine
                        [primal pullback later-writer])))))
    (let [pair (link/value {:id :pair
                            :abstract (av/tensor {:dtype :float :shape [8]})
                            :physical-layout {:kind :soa :field-order [:left :right]}
                            :leaves [{:name :left :node :residual}
                                     {:name :right :node :result}]})
          paired-spine (assoc spine :values
                              {:input (get-in spine [:values :input]) :pair pair})]
      (is (= :link-module-frontier-whole-value
             (reason #(#'module/derive-frontiers paired-spine
                        [primal pullback])))
          "one leaf requirement cannot retain only part of a logical aggregate"))))

(deftest module-spine-has-no-eager-global-output-contract
  (let [spine (legacy-spine)]
    (is (empty? (:outputs spine)))
    (is (= :link-module-spine
           (reason #(module/make {:id :bad :spine (assoc spine :outputs [:result])
                                  :entries [{:name :all
                                             :instances [:primal :pullback]}]}))))))

(deftest constructed-modules-rederive-and-compare-every-stored-contract
  (let [built (module/make {:id :state-module :spine (external-state-spine)
                            :entries [{:name :advance :instances [:advance]}]})]
    (is (= built (module/validate! built)))
    (is (= #{:input :state} (get-in built [:entries 0 :contract :external-requires])))
    (is (empty? (:frontiers built)))
    (is (= :link-module-derived-target
           (reason #(module/validate! (assoc built :target :forged)))))
    (is (= :link-module-derived-contracts
           (reason #(module/validate! (assoc-in built [:entries 0 :contract :reads] #{})))))
    (is (= :link-module-derived-frontiers
           (reason #(module/validate! (assoc built :frontiers #{:forged})))))
    (is (= :link-module-derived-retained-groups
           (reason #(module/validate! (assoc built :retained-groups {:forged true})))))
    (let [source (get-in built [:spine :nodes :input :source])]
      (aset-float ^floats source 0 17.0)
      (is (= built (module/validate! built))
          "fresh validation depends on source presence and shape, not mutable contents"))))

(deftest ordinary-link-plan-initialization-behavior-is-unchanged
  (is (= {:requires #{} :initializers #{:input}
          :produces #{:residual :result} :complete-writes #{}
          :reads #{:input :residual} :writes #{:residual :result} :outputs #{}}
         (link/initialization-contract (legacy-spine)))))

(deftest retained-borrowed-program-values-remain-storage-compatible-with-owned-module-values
  (let [owned (av/tensor {:dtype :float :shape '[n] :ownership :owned
                          :memory-space :device :placement {:device :ze:0}})
        borrowed (assoc owned :ownership :borrowed)]
    (is (av/storage-contract-compatible? owned borrowed))
    (is (= :owned (:ownership owned)))
    (is (= :borrowed (:ownership borrowed)))))
