(ns raster.compiler.ir.distributed-math-request-test
  "Independent intent across actual selected equation/loop, planning and runtime ownership."
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.backend.gpu.segop-opencl :as opencl]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]
            [raster.compiler.passes.parallel.structured-control-lower-test :as loop-fixture]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]
            [raster.runtime.resident-state :as resident-state]))

(def request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}})

(defn selected-plan
  ([target] (selected-plan target {:execution {:scalar-dtype-support {:double :supported}}}))
  ([target descriptor]
  (let [scheduled (lower/schedule (#'loop-fixture/loop-program true true false)
                                  {:target-device target :dtype :float})
        certificates (into {} (map (fn [node]
                                    [(:id node) (segmap-body/schedule
                                                 (:operation node)
                                                 (assoc request :scalar-types {'alpha-in :float 'n-in :int}
                                                        :array-types {'u-in :float 'u-temporary :float 'u-next :float}))]))
                           (get-in scheduled [:graph :nodes]))
        graph (opencl/generate-kernel-graph (:graph scheduled)
                                           :scalar-types {'alpha-in :float 'n-in :int}
                                           :scalar-math (:scalar-math request)
                                           :target-descriptor descriptor
                                           :scheduled-bodies certificates)
        emission (emitted-loop/make scheduled graph {} request)
        call (program-call/make
              (#'loop-fixture/enclosing-loop-program emission)
              {'u0 :initial 'u-final :output}
              {'steps {:type :long :value 5} 'n {:type :int :value 64}
               'alpha {:type :float :value 0.25}}
              {'u-final :scratch} nil {} nil request)
        abstract (av/tensor {:dtype :float :shape [64] :representation {:kind :plain}})
        local (link/make
               {:id :selected-local :target target
                :nodes (mapv #(link/node {:id % :dtype :float :shape [64] :device target
                                         :role :state
                                         :source (when (= % :initial) (float-array 64))})
                             [:initial :output :scratch])
                :values (mapv #(link/value {:id % :abstract abstract
                                           :leaves [{:name :value :node %}]})
                              [:initial :output :scratch])
                :instances [(link/program-instance {:id :selected :call call} request)]
                :outputs [:output]} request)
        ids (set/union (set (keys (link/value-accesses local request)))
                       (set (link/output-value-ids local)))
        fields {:id :selected-distributed
                :mesh (distributed/mesh [{:name :worker :size 1}] [:worker])
                :topology (distributed/topology
                           [(distributed/device {:id :worker :memory-capacity-bytes 1048576})] [])
                :values (into {} (for [id ids]
                                   [id (assoc abstract :sharding {:kind :replicated :devices [:worker]}
                                                       :ownership :owned)]))
                :shards (into {} (for [id ids]
                                   [id [(distributed/shard {:id id :value id :device :worker
                                                            :offsets [0] :shape [64] :ownership :replica})]]))
                :device-plans {:worker {:target target :entries {:loop {:link-plan local}}
                                        :steps {:advance {:entry :loop
                                                          :bindings (into {} (for [id ids]
                                                                               [id {:value id :shard id}]))}}}}
                :steps [(distributed/compute-step {:id :advance :device :worker :duration-ns 10})]
                :outputs [:advance]}]
    (distributed/plan fields request))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest empty-caller-intent-preserves-default-options-identity
  (doseq [options [nil {} {:scalar-math nil} {:scalar-math {:overrides {}}}]]
    (is (= {} (distributed/caller-math-options options))))
  (is (= request (distributed/caller-math-options request))))

(deftest explicit-policy-closes-distributed-planning-and-certificate-boundaries
  (let [plan (selected-plan :ocl:0)
        certificate (distributed/certify plan request)
        options (assoc request :device-capacities {:ocl:0 1048576} :include-graph-temporaries? true)
        pool (distributed/resident-storage-plan plan options)]
    (is (identical? plan (distributed/validate! plan request)))
    (is (= #{:advance} (set (keys (:bindings (distributed/compute-bindings plan request))))))
    (is (= {} (distributed/transfer-bindings plan request)))
    (is (= [:advance] (mapv :id (:actions (distributed/check-readiness plan request)))))
    (is (= [:advance] (mapv :id (:actions (distributed/check-retained-output-readiness plan request)))))
    (is (pos? (get-in pool [:allocation-budgets :ocl:0 :graph-temporary-bytes])))
    (is (= (:allocation-budgets pool)
           (get-in (distributed/simulate plan options) [:resident-storage :allocation-budgets])))
    (is (identical? certificate (distributed/verify! certificate request)))
    (is (= (:scalar-math request) (get-in certificate [:certificate :scalar-math])))
    (doseq [query [#(distributed/validate! plan) #(distributed/compute-bindings plan)
                   #(distributed/transfer-bindings plan) #(distributed/check-readiness plan)
                   #(distributed/check-retained-output-readiness plan)
                   #(distributed/resident-storage-plan plan) #(distributed/simulate plan)
                   #(distributed/certify plan) #(distributed/verify! certificate)
                   #(distributed/validate! plan {:scalar-math {:overrides {}}})
                   #(distributed/validate! (assoc-in plan [:attributes :scalar-math] (:scalar-math request)))]]
      (is (some? (reason query)) "default/metadata cannot authorize selected leaves"))))

(deftest mismatched-policy-rejects-before-acquisition
  (let [plan (selected-plan :ocl:0) touched (atom [])]
    (with-redefs [gpu/execution-device-info (fn [& _] (swap! touched conj :capability)
                                             (throw (ex-info "unexpected capability query" {:reason :capability-boundary})))
                  gpu/make-session (fn [& _] (swap! touched conj :session)
                                   (throw (ex-info "resource boundary" {:reason :resource-boundary})))]
      (is (some? (reason #(runtime/instantiate! plan))))
      (is (some? (reason #(runtime/instantiate! plan {:scalar-math {:overrides {}}}))))
      (is (empty? @touched))
      (is (= :resource-boundary
             (reason #(runtime/instantiate! plan (assoc request :device-capacities {:ocl:0 1048576})))))
      (is (= [:session] @touched)))))

(deftest malformed-options-refuse-before-hardware-or-acquisition
  (let [plan (selected-plan :ocl:0) touched (atom [])]
    (with-redefs [gpu/make-session (fn [& _] (swap! touched conj :session))
                  gpu/execution-device-info (fn [& _] (swap! touched conj :capability))]
      (doseq [options [nil [] {:ignored-option true}]]
        (is (= :distributed-runtime-options (reason #(runtime/instantiate! plan options)))))
      (is (some? (reason #(runtime/instantiate! plan {:scalar-math {:not-a-policy true}}))))
      (is (= :distributed-math-options (reason #(distributed/validate! plan {:plan-consent request}))))
      (is (empty? @touched)))))

(deftest acquisition-does-not-inherit-ambient-compiler-consent
  (let [plan (selected-plan :ocl:0)
        caller-var (ns-resolve 'raster.compiler.ir.link-plan '*caller-options*)
        observed (atom ::not-called)]
    (with-redefs [gpu/make-session (fn [& _]
                                   (reset! observed (var-get caller-var))
                                   (throw (ex-info "resource boundary" {:reason :resource-boundary})))]
      (with-bindings {caller-var request}
        (is (some? (reason #(runtime/instantiate! plan))))
        (is (= ::not-called @observed))
        (is (= :resource-boundary
               (reason #(runtime/instantiate! plan (assoc request :device-capacities {:ocl:0 1048576})))))
        (is (nil? @observed))))))

(deftest copied-runtime-owner-cannot-inherit-intent
  ;; Synthetic owner exercises issuer identity only; no computation receipt is fabricated.
  (let [owner (#'runtime/seal-runtime-value
               (runtime/map->DistributedExecutable
                {:state (atom :ready) :caller-options request
                 :schedule {:operations []} :plan {:outputs []}}))
        copied (assoc owner :caller-options request)]
    (is (runtime/original-executable? owner))
    (is (not (runtime/original-executable? copied)))
    (is (= :distributed-runtime-owner (reason #(runtime/run! copied))))
    (is (= :distributed-runtime-owner (reason #(runtime/output-values copied))))
    (is (= :ready @(:state owner)))))

(deftest unused-intent-is-certificate-identity-not-certificate-permission
  (let [fields {:id :analytical :mesh (distributed/mesh [{:name :worker :size 1}] [:worker])
                :topology (distributed/topology
                           [(distributed/device {:id :worker :memory-capacity-bytes 1024})] [])
                :steps [(distributed/compute-step {:id :work :device :worker :duration-ns 10})]
                :outputs [:work]}
        plan (distributed/plan fields)
        default (distributed/certify plan)
        selected (distributed/certify plan request)]
    (is (= (distributed/simulate plan) (distributed/simulate plan request)))
    (is (not (contains? (:certificate default) :scalar-math)))
    (is (= default (distributed/certify plan {:scalar-math {:overrides {}}})))
    (is (not= (:certificate default) (:certificate selected)))
    (is (= :distributed-certificate (reason #(distributed/verify! selected))))
    (is (= :distributed-certificate (reason #(distributed/verify! default request))))
    (is (identical? selected (distributed/verify! selected request)))))

(deftest selected-double-realization-retains-target-capability-admission
  (doseq [descriptor [nil {:execution {:scalar-dtype-support {:double :unsupported}}}]]
    (is (= :kernel-body-target-math-capability
           (reason #(selected-plan :ocl:0 descriptor))))))

(deftest selected-math-durable-capture-is-still-an-independent-boundary
  ;; Pure producer description, not a fabricated execution/completion receipt. The current
  ;; durable adapter has no independent math-request input and must still refuse selected math.
  (let [plan (selected-plan :ocl:0)
        expected (reason #(distributed/check-retained-output-readiness plan))]
    (is (some? expected))
    (is (= expected (reason #(#'resident-state/distributed-producers plan {}))))))
