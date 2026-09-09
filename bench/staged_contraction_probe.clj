(ns staged-contraction-probe
  "Opt-in internal candidate comparison, NOT a public compiler or external baseline benchmark.
   Run with :bench on an OpenCL GPU. Returns EDN data for explicit persistence by the caller."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.segop-opencl :as retained]
            [raster.compiler.backend.gpu.staged-contraction-fixtures :as fixtures]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.staged-contraction-body :as staged]
            [raster.compiler.passes.parallel.staged-scalar-body :as recursive]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.ocl-runtime :as ocl]))

(defn- retained-artifact [facts packed?]
  (let [emitted (retained/generate-staged-contraction-kernel
                 (assoc (select-keys facts [:free-axes :contract-axes :stages :body :dtype])
                        :inputs '[a b] :operands (get-in facts [:opts :operands])
                        :out-dtype :float :tensorize-inner? packed?) 'out)
        n (:out-elems emitted)]
    (artifact/make
     {:kernel-name (:kernel-name emitted) :target :opencl-c
      :source (:source emitted) :abi (:abi emitted) :arguments ['a 'b 'da 'db 'out n]
      :launch (launch/spec {:workgroup-size [64] :group-count [(quot (+ n 63) 64)]})
      :effects {:kind :tensor-contraction}
      :attributes {:out-elems n}
      :provenance {:dialect :benchmark :route :retained-staged-source}})))

(defn run!
  "Measure one [rows outputs blocks block-width] with a small exact dyadic host oracle.
   revision/environment identify the invocation; executable hashes identify actual emitted code.
   All candidates share logical Byte inputs and one output. Compilation/binding and transfers are
   outside device samples. No autotuning cache or production route is changed.
   :comparison-mode :typed-native-dot compares one typed schedule with emulated/native dot;
   both use explicit CL3.0. :typed-recursive compares the specialized two-stage and generic
   recursive generated schedules on the same two-stage workload. The default :retained keeps
   the three historical candidates. None measures public compiler or external baseline throughput."
  [{:keys [shape revision environment rounds warmup-rounds comparison-mode]
    :or {rounds 12 warmup-rounds 3 comparison-mode :retained}}]
  (when-not (contains? #{:retained :typed-native-dot :typed-recursive} comparison-mode)
    (throw (ex-info "unknown probe comparison mode" {:comparison-mode comparison-mode})))
  (when-not (and (vector? shape) (= 4 (count shape))
                 (every? #(and (integer? %) (pos? %)) shape)
                 (string? revision) (seq revision) (string? environment) (seq environment))
    (throw (ex-info "probe requires positive shape, revision and environment identity" {})))
  (when-not (and (integer? rounds) (<= 1 rounds 120)
                 (integer? warmup-rounds) (<= 0 warmup-rounds 30))
    (throw (ex-info "probe permits 1–120 rounds and 0–30 warmup rounds" {})))
  (let [[m n blocks width] shape
        work (*' m n blocks width)
        bytes (+' (*' (+' m n) blocks width) (*' 4 (+' (*' (+' m n) blocks) (*' m n))))
        _ (when (or (> work 8000000) (> bytes (* 64 1024 1024)))
            (throw (ex-info "probe exceeds laptop host-work or resident-storage budget"
                            {:host-products work :resident-bytes bytes})))
        facts (fixtures/packed-facts m n blocks width)
        ;; Lower first: admission checks static index/capacity and DP4A prefix ranges.
        scheduled (staged/lower facts)
        typed (target/emit-static-dense-graph "probe_typed" scheduled :opencl-portable
                 (if (= :typed-native-dot comparison-mode)
                   {:attributes {:compilation {:language-standard "CL3.0"}}} {}))
        candidates (case comparison-mode
                     :typed-native-dot
                     [[:typed typed]
                      [:native (target/emit-static-dense-graph
                                "probe_native" scheduled :opencl-portable
                                {:target-features
                                 {:intrinsic-implementations {:dp4a :opencl-packed-dot}}})]]
                     :typed-recursive
                     [[:typed typed]
                      [:recursive (target/emit-static-dense-graph
                                   "probe_recursive" (recursive/lower facts) :opencl-portable)]]
                     :retained
                     [[:typed typed] [:scalar (retained-artifact facts false)]
                      [:packed (retained-artifact facts true)]])
        ;; Positive products ensure even complete periodic cycles cannot hide a zero-writing bug.
        a (byte-array (map #(inc (mod % 3)) (range (* m blocks width))))
        b (byte-array (map #(inc (mod % 5)) (range (* n blocks width))))
        da (float-array (repeat (* m blocks) 0.5))
        db (float-array (repeat (* n blocks) 0.25))
        expected (vec (for [i (range m) j (range n)]
                        (reduce (fn [acc blk]
                                  (float (+ acc
                                            (float (* 0.125
                                                      (reduce + (for [t (range width)]
                                                                  (* (aget a (+ (* i blocks width) (* blk width) t))
                                                                     (aget b (+ (* j blocks width) (* blk width) t))))))))))
                                (float 0) (range blocks))))]
    (gpu/with-gpu-session [sess :ocl:0]
      (gpu/alloc! sess {:a [:byte (alength a) a] :b [:byte (alength b) b]
                        :da [:float (alength da) da] :db [:float (alength db) db]
                        :out [:float (* m n) nil]})
      (let [bound (atom [])
            poison! #(gpu/upload! sess :out (float-array (repeat (* m n) Float/NaN)))]
        (try
          (doseq [[id executable] candidates]
            (let [start (System/nanoTime)
                  handle (gpu/bind-kernel-executable!
                          sess id executable
                          (cond-> [:a :b :da :db :out]
                            (= :kernel-artifact (executable/kind executable))
                            (conj {:type :int :value (* m n)}))
                          {:profile? true})]
              (swap! bound conj {:id id :handle handle
                                :bind-ms (/ (- (System/nanoTime) start) 1.0e6)
                                :signature (tuning/executable-signature executable)})
              (poison!)
              (gpu/run-kernel-graph! sess handle)
              (let [actual (vec (gpu/download sess :out))]
                (when-not (= expected actual)
                  (throw (ex-info "candidate failed exact dyadic oracle" {:candidate id}))))))
          {:kind :internal-staged-candidate-comparison :shape shape
           :comparison-mode comparison-mode
           :revision revision :environment environment :device (ocl/selected-device-info)
           :input-recipe {:a :index-mod3-plus1 :b :index-mod5-plus1 :da 0.5 :db 0.25}
           :validation {:passed? true :oracle :explicit-host-dot-dyadic-lift :comparison :exact}
           :scope {:cache :warm :timing :device-event :transfers-included? false
                   :public-compiler-path? false :promotion? false}
           :candidates (mapv #(dissoc % :handle) @bound)
           :comparison (update
                        (gpu/measure-bound-kernel-graphs-interleaved!
                         sess (mapv #(assoc (select-keys % [:id :handle]) :before-sample! poison!) @bound)
                         :rounds rounds :warmup-rounds warmup-rounds)
                        :measurements #(update-vals % (fn [m] (into {} m))))}
          (finally
            (doseq [{:keys [handle]} (reverse @bound)]
              (gpu/release-kernel-graph! sess handle))))))))
