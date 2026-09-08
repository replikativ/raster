(ns raster.compiler.passes.parallel.staged-scalar-body-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.equation-first :as equation-first]
            [raster.core :refer [deftm]]
            [raster.runtime.hardware :as hardware]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.contraction-closure :as closure]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.parallel-program :as parallel]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.backend.gpu.segop-opencl :as emitter]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.compiler.passes.parallel.staged-scalar-body :as staged]))

(defn three-stage-facts []
  (let [a-map (am/of-axes '[[i 2] [blk 2] [sub 3] [t 4]])
        b-map (am/of-axes '[[j 3] [blk 2] [sub 3] [t 4]])]
    (facts/from-components
     {:out 'out :free-axes '[[i 2] [j 3]] :contract-axes '[[blk 2] [sub 3] [t 4]]
      :dtype :float
      :body (list 'raster.numeric/* (list 'aget 'a (am/index-expr a-map))
                  (list 'aget 'b (am/index-expr b-map)))
      :opts {:operands [{:sym 'a :map a-map} {:sym 'b :map b-map}]
             :stages [{:axis 'blk :extent 2 :dtype :float :init 0.0
                       :lift '(* inner scale)}
                      {:axis 'sub :extent 3 :dtype :float :init 0.0
                       :lift '(* inner (aget weights _))
                       :operands [{:sym 'weights :dtype :float
                                   :map (am/of-axes '[[j 3] [blk 2] [sub 3]])}]}
                      {:axis 't :extent 4 :dtype :float :init 0.0}]}})))

(deftm public-three-stage!
  [a :- (Array float) b :- (Array float) weights :- (Array float)
   out :- (Array float) scale :- Float] :- Void
  (raster.par/contract out [[i 2] [j 3]] [[blk 2] [sub 3] [t 4]]
    (raster.numeric/* (raster.arrays/aget a (+ (* i 24) (* blk 12) (* sub 4) t))
                      (raster.arrays/aget b (+ (* j 24) (* blk 12) (* sub 4) t)))
    :stages [{:axis blk :extent 2 :dtype :float :init 0.0 :lift (* inner scale)}
             {:axis sub :extent 3 :dtype :float :init 0.0 :lift (* inner (aget weights _))
              :operands [{:sym weights :dtype :float :map {:groups [[[j 3] [blk 2] [sub 3]]]}}]}
             {:axis t :extent 4 :dtype :float :init 0.0}]))

(deftm public-long-stage!
  [a :- (Array float) b :- (Array float) out :- (Array float) gain :- Long] :- Void
  (raster.par/contract out [[i 1]] [[blk 2] [t 4]]
    (raster.numeric/* (raster.arrays/aget a (+ (* blk 4) t))
                      (raster.arrays/aget b (+ (* blk 4) t)))
    :stages [{:axis blk :extent 2 :dtype :float :init 0.0
              :lift (* inner (double (clojure.core/+ gain 1)))}
             {:axis t :extent 4 :dtype :float :init 0.0}]))

(deftest public-stages-retain-checked-integer-arithmetic
  ;; Compile only: deliberately overflowing a device trap would poison the shared device context.
  (hardware/register-target-device! :cuda:checked-stage-test
                                    {:type :cuda :capabilities {:compute-capability [8 0]
                                                               :warp-size 32 :subgroup-sizes [32]
                                                               :max-workgroup-size 1024}})
  (let [compilation (equation-first/compile #'public-long-stage!
                                          {:target :cuda:checked-stage-test :dtype :float})
        body (get-in compilation [:kernels 0 :attributes :kernel-body])]
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (some #(and (map? %) (= :trap (:overflow %))) (tree-seq coll? seq body)))))

(deftest public-floating-stages-enter-the-generated-vertical
  (hardware/register-target-device! :cuda:scalar-stage-test
                                    {:type :cuda :capabilities {:compute-capability [8 0]
                                                               :warp-size 32 :subgroup-sizes [32]
                                                               :max-workgroup-size 1024}})
  (let [compilation (equation-first/compile #'public-three-stage!
                                            {:target :cuda:scalar-stage-test :dtype :float})
        plan (equation-first/lower compilation [(float-array 48) (float-array 72)
                                                 (float-array 18) (float-array 6) (float 0.25)])]
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (= 0 (get-in plan [:attributes :driver-allocations])))
    (is (= :staged-scalar
           (get-in compilation [:kernels 0 :attributes :kernel-body :schedule :strategy])))))

(deftest public-floating-stages-execute-through-the-resident-link-plan
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "public floating staged contraction")
    (let [output (float-array (repeat 6 -555.0))
          compilation (equation-first/compile #'public-three-stage! {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compilation
                                     [(float-array (repeat 48 1.0)) (float-array (repeat 72 2.0))
                                      (float-array (repeat 18 0.5)) output (float 0.25)])
          output-id (some (fn [[id node]] (when (identical? output (:source node)) id)) (:nodes plan))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (some? output-id))
      (let [executable (link/instantiate! plan)]
        (try
          (link/run! executable)
          (is (= (vec (repeat 6 6.0)) (vec (link/download executable output-id))))
          (finally (link/close! executable)))))))

(deftest scalar-stages-emit-through-every-common-target
  (let [source (three-stage-facts)
        scheduled (with-redefs [facts/contraction-facts
                                (fn [& _] (throw (Exception. "reparsed source")))]
                    (staged/lower source :scalar-types {'scale :float}))]
    (is (identical? source (:source scheduled)))
    (is (= '[a b weights out scale] (:arguments scheduled)))
    (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (let [artifact (target/emit-artifact "three_stage_scalar" scheduled dialect)]
        (is (= [:float :float :float :float :float] (mapv :dtype (:abi artifact))))
        (is (not (re-find #"rstr_dp4a" (:source artifact))))))))

(deftest admission-constructs-typed-stages-without-materializing-a-kernel
  (let [plan (with-redefs [kernel-body/make (fn [& _] (throw (Exception. "constructed KernelBody")))
                           target/emit-artifact (fn [& _] (throw (Exception. "emitted target source")))]
               (staged/analyze! (three-stage-facts) :scalar-types {'scale :float}))]
    (is (= '[a b weights out scale] (:arguments plan)))
    (is (= :staged-scalar (get-in plan [:body-spec :schedule :strategy])))))

(deftest unsupported-stage-capabilities-decline-before-public-admission
  (doseq [[source rule]
          [[(assoc (three-stage-facts) :free-axes '[[i n] [j 3]]) :static-domain]
           [(assoc (three-stage-facts) :free-axes '[[i 0] [j 3]]) :static-domain]
           [(assoc (three-stage-facts) :epilogue {:acc 'value :expr 'value}) :numerical-contract]
           [(update (three-stage-facts) :body
                    #(list 'clojure.core/identity %)) :scalar-expression]]]
    (let [failure (try (staged/analyze! source :scalar-types {'scale :float}) nil
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= :staged-scalar-body-declined (:reason failure)))
      (is (= rule (:missing-rule failure))))))

(deftest scalar-conversions-require-and-preserve-source-arithmetic-types
  (let [source (fn [addition]
                 (assoc-in (three-stage-facts) [:stages 0 :lift]
                           (list '* 'inner (list 'double addition))))
        addition '(clojure.core/+ gain 1)
        failure (try (staged/analyze! (source addition) :scalar-types {'gain :long}) nil
                     (catch clojure.lang.ExceptionInfo e (ex-data e)))
        implicit-failure (try (staged/analyze!
                               (assoc-in (three-stage-facts) [:stages 0 :lift] (list '* 'inner addition))
                               :scalar-types {'gain :long}) nil
                              (catch clojure.lang.ExceptionInfo e (ex-data e)))
        typed-addition (with-meta addition {:raster.type/tag 'long})
        scheduled (staged/lower (source typed-addition) :scalar-types {'gain :long})
        implicit-conversion (staged/lower
                              (assoc-in (three-stage-facts) [:stages 0 :lift]
                                        (list '* 'inner typed-addition))
                              :scalar-types {'gain :long})]
    (is (= :staged-scalar-body-declined (:reason failure)))
    (is (= :scalar-source-type (:missing-rule failure)))
    (is (= :staged-scalar-body-declined (:reason implicit-failure)))
    (is (= :scalar-source-type (:missing-rule implicit-failure)))
    (is (some #(and (map? %) (= :trap (:overflow %)))
              (tree-seq coll? seq (:body scheduled))))
    (is (some #(and (map? %) (= :trap (:overflow %)))
              (tree-seq coll? seq (:body implicit-conversion))))
    (doseq [dialect [:opencl-intel :cuda :hip]]
      (is (re-find #"rstr_trap_add_i64" (:source (target/emit-artifact "checked_stage" scheduled dialect)))))
    (is (= :kernel-body-c-trap-unsupported
           (try (target/emit-artifact "checked_stage" scheduled :opencl-portable) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest malformed-stage-closures-remain-errors
  (let [failure (try (staged/analyze! (assoc (three-stage-facts) :out 'a)
                                     :scalar-types {'scale :float}) nil
                     (catch clojure.lang.ExceptionInfo e (ex-data e)))]
    (is (= :typed-soac-contraction (:reason failure)))
    (is (= :lexical-boundary (:missing-rule failure)))))

(deftest scalar-captures-require-authoritative-types
  (is (thrown? clojure.lang.ExceptionInfo (staged/lower (three-stage-facts))))
  (is (thrown? clojure.lang.ExceptionInfo
               (staged/lower (assoc (three-stage-facts) :out-dtype :double)
                             :scalar-types {'scale :float}))))

(defn production-graph []
  (let [source (three-stage-facts)
        attributes {:contraction source :array-parameters '[a b weights] :capture-parameters '[scale]}
        values (merge (into {} (map (fn [{:keys [parameter dtype elements]}]
                                     [parameter (av/tensor {:dtype dtype :shape [elements]})]))
                            (closure/storage-requirements attributes))
                      {'out (av/tensor {:dtype :float :shape [6]})
                       'result (av/tensor {:dtype :float :shape [2 3]})
                       'scale (av/tensor {:dtype :float :shape []})})
        facts (soac/default-program-facts
               {:values values :inputs '[a b weights scale] :effects #{:memory/read :memory/write}
                :equations {'contraction (assoc (soac/default-equation-facts)
                                               :effects #{:memory/read :memory/write}
                                               :aliases {'result 'out}
                                               :attributes {:result-storage [{:destination 'out :access :write
                                                                              :host-return :buffer}]})}})
        algorithm (soac/make facts [(list '= 'contraction '[result]
                                         (list 'contract attributes '[a b weights] '[scale]))]
                             '[result])
        operations (lower/lower-typed-contract algorithm :ocl:0)
        equation (parallel/->ProgramEquation 'contraction [:binding 'result] nil (:inputs facts)
                                             '[result] algorithm operations (:effects facts) {} {})
        body (parallel/make {:dialect :segop :values values :inputs (:inputs facts)
                             :equations [equation] :outputs '[result] :effects (:effects facts)
                             :operation? segop/segop-node?})]
    {:algorithm algorithm :body body :graph (equation-graph/make algorithm body)}))

(deftest generic-stages-use-the-existing-certified-graph-route
  (let [{:keys [algorithm body graph]} (production-graph)]
    (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (let [emitted (emitter/generate-kernel-graph
                     graph :target-dialect dialect :scheduled-equation-algorithm algorithm
                     :scheduled-equation-body body :scalar-types {'scale :float})]
        (is (= '[a b weights out scale] (:arguments emitted)))
        (is (= [:float :float :float :float :float] (mapv :dtype (:abi emitted))))
        (is (= :staged-scalar
               (get-in emitted [:nodes 0 :operation :attributes :kernel-body :schedule :strategy])))))))

(deftest generic-stages-bind-through-the-common-resident-graph
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "generic staged resident graph")
    (let [{:keys [algorithm body graph]} (production-graph)
          emitted (emitter/generate-kernel-graph
                   graph :target-dialect :opencl-portable :scheduled-equation-algorithm algorithm
                   :scheduled-equation-body body :scalar-types {'scale :float})]
      (gpu/with-gpu-session [session :ocl:0]
        (gpu/alloc! session {:a [:float 48 (float-array (repeat 48 1.0))]
                             :b [:float 72 (float-array (repeat 72 2.0))]
                             :weights [:float 18 (float-array (repeat 18 0.5))]
                             :out [:float 6 (float-array (repeat 6 -555.0))]})
        (let [handle (gpu/bind-kernel-graph! session :generic-stages emitted
                                            {'a :a 'b :b 'weights :weights 'out :out}
                                            {'scale {:type :float :value 0.25}})]
          (try
            (let [event (gpu/submit-kernel-graph! session handle)]
              (try (gpu/await-event! session event)
                   (is (= (vec (repeat 6 6.0)) (vec (gpu/download session :out))))
                   (finally (gpu/release-event! session event))))
            (finally (gpu/release-kernel-graph! session handle))))))))

(deftest conflicting-storage-declarations-cannot-retype-core-loads
  (let [source (facts/from-components
                (-> (select-keys (three-stage-facts) [:out :free-axes :contract-axes :body :opts])
                    (assoc :dtype :float)
                    (assoc-in [:opts :stages 1 :lift] '(* inner (aget a _)))
                    (assoc-in [:opts :stages 1 :operands 0 :sym] 'a)
                    (assoc-in [:opts :stages 1 :operands 0 :dtype] :double)))]
    (is (= :storage-types
           (try (staged/lower source :scalar-types {'scale :float}) nil
                (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e))))))))

(deftest omitted-stage-identities-default-to-positive-zero
  (doseq [initialize [#(dissoc % :init) #(assoc % :init nil)]]
    (let [source (facts/from-components
                  (-> (select-keys (three-stage-facts) [:out :free-axes :contract-axes :body :opts])
                      (assoc :dtype :float)
                      (update-in [:opts :stages] #(mapv initialize %))))
          scheduled (staged/lower source :scalar-types {'scale :float})
          initializers (keep #(when (and (map? %) (contains? % :iter-args))
                               (get-in % [:iter-args 0 :initial :value]))
                             (tree-seq coll? seq (:body scheduled)))]
      (is (= 3 (count initializers)))
      (is (every? #(= 0 (Double/doubleToRawLongBits (double %))) initializers)))))

(deftest declared-negative-zero-stage-identities-retain-their-sign
  (let [source (facts/from-components
                (-> (select-keys (three-stage-facts) [:out :free-axes :contract-axes :body :opts])
                    (assoc :dtype :float)
                    (update-in [:opts :stages] #(mapv (fn [stage] (assoc stage :init -0.0)) %))))
        scheduled (staged/lower source :scalar-types {'scale :float})
        initializers (keep #(when (and (map? %) (contains? % :iter-args))
                              (get-in % [:iter-args 0 :initial :value]))
                           (tree-seq coll? seq (:body scheduled)))]
    (is (= 3 (count initializers)))
    (is (every? #(= Long/MIN_VALUE (Double/doubleToRawLongBits (double %))) initializers))
    (if-not @probe/opencl-available?
      (probe/opencl-skip! "signed-zero staged identities")
      (let [ocl (find-ns 'raster.gpu.ocl-runtime)
            op #(ns-resolve ocl %)
            artifact (target/emit-artifact "negative_zero_stages" scheduled :opencl-portable)
            buffers (mapv #((op 'buffer-of-array) % :float)
                          [(float-array (repeat 48 -0.0)) (float-array (repeat 72 1.0))
                           (float-array (repeat 18 1.0)) (float-array (repeat 6 555.0))])]
        (try
          ((op 'register-kernel!) (:kernel-name artifact) artifact)
          (let [prepared ((op 'bind-kernel-call)
                          (call/make artifact (conj buffers {:type :float :value 1.0})))]
            (try ((op 'launch-registered-bound!) prepared)
                 (finally ((op 'destroy-prepared!) prepared))))
          (is (= (vec (repeat 6 Integer/MIN_VALUE))
                 (mapv #(Float/floatToRawIntBits (float %)) ((op 'buffer->array) (peek buffers)))))
          (finally (doseq [buffer (reverse buffers)] ((op 'free-buffer!) buffer))))))))

(defn- reference [a b weights scale]
  (let [sum-f32 (fn [xs] (reduce #(float (+ (double %1) (double %2))) (float 0) xs))]
    (vec (for [i (range 2) j (range 3)]
           (sum-f32
            (for [blk (range 2)]
              (float
               (* scale
                  (sum-f32
                   (for [sub (range 3)]
                     (float
                      (* (aget ^floats weights (+ (* j 6) (* blk 3) sub))
                         (sum-f32
                          (for [t (range 4)]
                            (float (* (aget ^floats a (+ (* i 24) (* blk 12) (* sub 4) t))
                                      (aget ^floats b (+ (* j 24) (* blk 12) (* sub 4) t))))))))))))))))))

(deftest generated-three-stage-loops-match-rounded-reference
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "generic three-stage floating KernelBody")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          op #(ns-resolve ocl %)
          artifact (target/emit-artifact
                    "three_stage_reference"
                    (staged/lower (three-stage-facts) :scalar-types {'scale :float}) :opencl-portable)]
      (doseq [[a b] [[(float-array (map #(/ (- (mod % 11) 5) 8.0) (range 48)))
                      (float-array (map #(/ (- (mod % 7) 3) 4.0) (range 72)))]
                     [(float-array (take 48 (cycle [1.0e8 1.0 -1.0e8 3.0])))
                      (float-array (repeat 72 1.0))]]]
        (let [weights (float-array (map #(/ (inc (mod % 5)) 8.0) (range 18)))
              scale (float 0.75)
              buffers (mapv #((op 'buffer-of-array) % :float)
                            [a b weights (float-array (repeat 6 -555.0))])]
          (try
            ((op 'register-kernel!) (:kernel-name artifact) artifact)
            (let [prepared ((op 'bind-kernel-call)
                            (call/make artifact (conj buffers {:type :float :value scale})))]
              (try ((op 'launch-registered-bound!) prepared)
                   (finally ((op 'destroy-prepared!) prepared))))
            (is (= (reference a b weights scale) (vec ((op 'buffer->array) (peek buffers)))))
            (finally (doseq [buffer (reverse buffers)] ((op 'free-buffer!) buffer)))))))))
