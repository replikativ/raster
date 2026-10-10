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
            [raster.gpu.distributed :as runtime]))

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

(deftest explicit-policy-closes-distributed-planning-and-certificate-boundaries
  (let [plan (selected-plan :ocl:0)
        certificate (distributed/certify plan request)
        options (assoc request :device-capacities {:ocl:0 1048576} :include-graph-temporaries? true)
        pool (distributed/resident-storage-plan plan options)]
    (is (identical? plan (distributed/validate! plan request)))
    (is (= #{:advance} (set (keys (:bindings (distributed/compute-bindings plan request))))))
    (is (= {} (distributed/transfer-bindings plan request)))
    (is (= [:advance] (mapv :id (:actions (distributed/check-readiness plan request)))))
    (is (pos? (get-in pool [:allocation-budgets :ocl:0 :graph-temporary-bytes])))
    (is (= (:allocation-budgets pool)
           (get-in (distributed/simulate plan options) [:resident-storage :allocation-budgets])))
    (is (identical? certificate (distributed/verify! certificate request)))
    (is (= (:scalar-math request) (get-in certificate [:certificate :scalar-math])))
    (doseq [query [#(distributed/validate! plan) #(distributed/compute-bindings plan)
                   #(distributed/transfer-bindings plan) #(distributed/check-readiness plan)
                   #(distributed/resident-storage-plan plan) #(distributed/simulate plan)
                   #(distributed/certify plan) #(distributed/verify! certificate)
                   #(distributed/validate! plan {:scalar-math {:overrides {}}})
                   #(distributed/validate! (assoc-in plan [:attributes :scalar-math] (:scalar-math request)))]]
      (is (some? (reason query)) "default/metadata cannot authorize selected leaves"))))

(deftest mismatched-policy-rejects-before-acquisition
  (let [plan (selected-plan :ocl:0) touched (atom [])]
    (with-redefs [gpu/make-session (fn [& _] (swap! touched conj :session)
                                   (throw (ex-info "resource boundary" {:reason :resource-boundary})))]
      (is (some? (reason #(runtime/instantiate! plan))))
      (is (some? (reason #(runtime/instantiate! plan {:scalar-math {:overrides {}}}))))
      (is (empty? @touched))
      (is (= :resource-boundary
             (reason #(runtime/instantiate! plan (assoc request :device-capacities {:ocl:0 1048576})))))
      (is (= [:session] @touched)))))

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
  (is (= {} (distributed/caller-math-options nil)
         (distributed/caller-math-options {:scalar-math {:overrides {}}})))
  (let [fields {:id :analytical :mesh (distributed/mesh [{:name :worker :size 1}] [:worker])
                :topology (distributed/topology
                           [(distributed/device {:id :worker :memory-capacity-bytes 1024})] [])
                :steps [(distributed/compute-step {:id :work :device :worker :duration-ns 10})]
                :outputs [:work]}
        plan (distributed/plan fields)
        default (distributed/certify plan)
        selected (distributed/certify plan request)]
    (is (= (distributed/simulate plan) (distributed/simulate plan request)))
    (is (= {:overrides {}} (get-in default [:certificate :scalar-math])))
    (is (not= (:certificate default) (:certificate selected)))
    (is (= :distributed-certificate (reason #(distributed/verify! selected))))
    (is (= :distributed-certificate (reason #(distributed/verify! default request))))
    (is (identical? selected (distributed/verify! selected request)))))

(deftest selected-double-realization-retains-target-capability-admission
  (doseq [descriptor [nil {:execution {:scalar-dtype-support {:double :unsupported}}}]]
    (is (= :kernel-body-target-math-capability
           (reason #(selected-plan :ocl:0 descriptor))))))

(deftest acquisition-and-output-callbacks-clear-compiler-proof-scopes
  (let [vars (mapv #(ns-resolve (first %) (second %))
                   '[[raster.compiler.ir.link-plan *validated-program-instances*]
                     [raster.compiler.ir.link-plan *retained-program-validations*]
                     [raster.compiler.ir.link-plan *caller-options*]
                     [raster.compiler.ir.emitted-parallel-program-call *validated-boundary-projections*]
                     [raster.compiler.ir.emitted-parallel-program-call *validated-projection-policy*]])
        observe #(mapv var-get vars)
        observed (atom [])
        foreign #(with-bindings (zipmap vars (repeat :foreign-proof)) (%))
        plan (selected-plan :ocl:0)]
    (with-redefs [gpu/make-session
                  (fn [_]
                    (swap! observed conj (observe) @(future (observe)))
                    (throw (ex-info "stop at resource boundary" {:reason :resource-boundary})))]
      (is (= :resource-boundary
             (foreign #(reason (fn [] (runtime/instantiate! plan
                                       (assoc request :device-capacities {:ocl:0 1048576}))))))))
    (let [owner (#'runtime/seal-runtime-value
                 (runtime/map->DistributedExecutable
                  {:state (atom :complete) :caller-options request :plan {:outputs []}
                   :sessions {:unit :fake-session}}))]
      (add-watch (:state owner) :scope-watch
                 (fn [& _] (swap! observed conj (observe) @(future (observe)))))
      (is (= :copied
             (foreign #(runtime/with-output-values!
                        owner (fn [_]
                                (swap! observed conj (observe) @(future (observe))) :copied)))))
      (is (= :complete @(:state owner)))
      (with-redefs [gpu/close-session! (fn [_] (swap! observed conj (observe) @(future (observe))))]
        (is (nil? (foreign #(runtime/close! owner)))))
      (is (= :closed @(:state owner))))
    (is (= (vec (repeat 14 (vec (repeat 5 nil)))) @observed))))
