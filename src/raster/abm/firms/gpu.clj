(ns raster.abm.firms.gpu
  "GPU-specific orchestration for the firms ABM.

   The kernels live in raster.abm.firms.phases and the generic session API lives
   in raster.gpu.core. This namespace keeps only the model-specific phase order,
   buffer layout, and public entry points."
  (:refer-clojure :exclude [alength])
  (:require [raster.arrays :refer [alength]]
            [raster.abm.firms :as firms]
            [raster.abm.firms.phases :as phases]
            [raster.gpu.core :as gpu])
  (:import [raster.abm.firms AgentSoA FirmSoA]))

(def ^:private phase-map
  {:agent-decide        #'phases/agent-decide-par!
   :produce-output      #'phases/produce-output-par!
   :distribute-income   #'phases/distribute-income-par!
   :zero-firm-sizes     #'phases/zero-firm-sizes!
   :histogram-firms     #'phases/histogram-firms!
   :update-alive        #'phases/update-alive!
   :fill-members        #'phases/fill-members-par!
   :csr-scan            #'phases/csr-scan-par!
   :startup-scan        #'phases/startup-scan-par!
   :generate-active-ids #'phases/generate-active-ids-par!
   :fill-rng-seeds      #'phases/fill-rng-seeds-par!
   :execute-stay-switch #'phases/execute-stay-switch-par!
   :count-startups      #'phases/count-startups-par!
   :build-not-alive     #'phases/build-not-alive-par!
   :scatter-free-slots  #'phases/scatter-free-slots-par!
   :execute-startups    #'phases/execute-startups-par!
   :update-alive-post   #'phases/update-alive-post-decisions-par!})

(def ^:private agent-buffer-aliases
  {:firm :current-firm
   :cache :s-cache})

(def ^:private firm-buffer-aliases
  {:a :param-a
   :b :param-b
   :beta :param-beta
   :te :total-effort
   :n-workers :firm-size
   :offsets :member-offsets})

(defn compile-abm-kernels!
  "Compile all firms ABM kernels into the session."
  [sess]
  (gpu/compile-phases! sess phase-map))

(defn- byte-count [dtype n]
  (* n (case dtype
         :float 4
         :int 4
         :long 8)))

(defn- layout-from-state
  [^AgentSoA agents ^FirmSoA firms]
  (let [n-agents (alength (.effort agents))
        n-friends (if (zero? n-agents)
                    0
                    (quot (alength (.friends agents)) n-agents))
        max-firms (alength (.a firms))
        members-len (alength (.members firms))
        offsets-len (alength (.offsets firms))]
    (when (not= (alength (.income agents)) n-agents)
      (throw (ex-info "Agent arrays must have consistent lengths"
                      {:field :income :expected n-agents :actual (alength (.income agents))})))
    (when (not= offsets-len (inc max-firms))
      (throw (ex-info "Firm offsets must have max-firms+1 elements"
                      {:max-firms max-firms :offsets offsets-len})))
    {:n-agents n-agents
     :n-friends n-friends
     :max-firms max-firms
     :members-len members-len
     :offsets-len offsets-len}))

(defn- buffer-specs
  [^AgentSoA agents ^FirmSoA firms n-active]
  (let [{:keys [n-agents max-firms offsets-len]} (layout-from-state agents firms)]
    (merge (gpu/array-bundle-buffer-specs agents {:aliases agent-buffer-aliases})
           (gpu/array-bundle-buffer-specs firms {:aliases firm-buffer-aliases})
           {;; Scratch buffers
            :agent-ids      [:int n-active nil]
            :decision-types [:int n-active nil]
            :target-firms   [:int n-active nil]
            :new-efforts    [:float n-active nil]
            :q-count        [:int 1 nil]
            :active-ids     [:int n-active nil]
            :pos            [:int max-firms nil]
            :not-alive      [:int max-firms nil]
            :free-offsets   [:int offsets-len nil]
            :free-slots     [:int max-firms nil]
            :startup-indices [:int n-active nil]
            :startup-count  [:int 1 nil]
            :rng-seeds      [:long n-agents nil]})))

(defn- require-layout
  [sess]
  (or (:abm-layout @sess)
      (throw (ex-info "ABM layout missing from session; call make-abm-buffers first"
                      {:session-keys (keys @sess)}))))

(defn make-abm-buffers
  "Allocate persistent GPU buffers for the firms ABM and store the derived layout in the session."
  [sess ^AgentSoA agents ^FirmSoA firms n-active]
  (let [layout (layout-from-state agents firms)
        specs (buffer-specs agents firms n-active)
        device-id (:device-id @sess)
        total-bytes (reduce-kv (fn [acc _ [dtype n _]] (+ acc (byte-count dtype n))) 0 specs)
        hw-topo (try ((requiring-resolve 'raster.runtime.hardware/memory-topology) device-id)
                     (catch Exception _ {:model :discrete :integrated? false}))
        unified? (= :unified (:model hw-topo))]
    (println (format "[gpu-planner] %s  topology: %s%s"
                     (name device-id)
                     (name (:model hw-topo))
                     (if (:integrated? hw-topo) " (integrated)" "")))
    (println (format "              init: %s  est. %.1f GB"
                     (if unified? "zero-copy (FloatBuffer view)" "upload (memcpy)")
                     (/ total-bytes 1e9)))
    (swap! sess assoc :abm-layout (assoc layout :n-active-capacity n-active))
    (gpu/alloc! sess specs)))

(defn- reset-counter!
  [sess key]
  (gpu/upload! sess key (int-array 1)))

(defn run-period-gpu-fused!
  "Execute one firms ABM period using persistent GPU buffers.
   Returns the queue count for stats collection."
  [sess rng n-active]
  (let [{:keys [n-agents n-friends max-firms n-active-capacity]} (require-layout sess)]
    (when (> n-active n-active-capacity)
      (throw (ex-info "Requested active count exceeds allocated queue capacity"
                      {:requested n-active :capacity n-active-capacity})))
    (gpu/invoke-active-ids! sess :generate-active-ids :active-ids n-active n-agents (.nextLong rng))
    (gpu/invoke-rng-fill! sess :fill-rng-seeds :rng-seeds n-agents (.nextLong rng))
    (reset-counter! sess :q-count)
    (gpu/invoke! sess :agent-decide {} [{:type :int :value (int n-friends)}] n-active)
    (let [cnt (long (aget ^ints (gpu/download sess :q-count) 0))]
      (when (> cnt 0)
        (gpu/invoke! sess :execute-stay-switch {} [] cnt)
        (reset-counter! sess :startup-count)
        (gpu/invoke! sess :count-startups {} [] cnt))
      (let [n-startups (long (aget ^ints (gpu/download sess :startup-count) 0))]
        (when (> n-startups 0)
          (gpu/invoke! sess :build-not-alive {} [] max-firms)
          (gpu/invoke-scan! sess :startup-scan [:not-alive] :free-offsets max-firms)
          (gpu/invoke! sess :scatter-free-slots {} [] max-firms)
          (gpu/invoke! sess :execute-startups {} [] n-startups))
        (gpu/invoke! sess :update-alive-post {} [] max-firms)
        (gpu/invoke! sess :zero-firm-sizes {} [] max-firms)
        (gpu/invoke! sess :histogram-firms {} [] n-agents)
        (gpu/invoke-scan! sess :csr-scan [:firm-size] :member-offsets max-firms)
        (gpu/invoke! sess :update-alive {} [] max-firms)
        (gpu/invoke! sess :fill-members {} [] max-firms {:index 0})
        (gpu/invoke! sess :fill-members {} [] n-agents {:index 1})
        (gpu/invoke! sess :produce-output {} [] max-firms)
        (gpu/invoke! sess :distribute-income {} [] max-firms)
        cnt))))

(defn autotune-abm-kernels!
  "Tune the core firms ABM kernels using the session's actual layout and buffer set."
  [sess & {:keys [warmup timed wg-sizes load-cache?]
           :or {warmup 3 timed 5 wg-sizes [64 128 256 512] load-cache? true}}]
  (let [autotune! (requiring-resolve 'raster.compiler.support.autotuner/autotune-kernel!)
        save-cache! (requiring-resolve 'raster.compiler.support.autotuner/save-tuning-cache!)
        load-cache-fn (requiring-resolve 'raster.compiler.support.autotuner/load-tuning-cache)
        apply-cached! (requiring-resolve 'raster.compiler.support.autotuner/apply-cached-tuning!)
        {:keys [kernels buffers device-id]} @sess
        {:keys [n-agents n-friends max-firms n-active-capacity]} (require-layout sess)
        specs [{:kname (:kernel-name (first (:agent-decide kernels)))
                :arrays (mapv #(get buffers %) [:active-ids :agent-ids :alive :current-firm
                                                :decision-types :effort :endowment :firm-size
                                                :friends :new-efforts :output :param-a :param-b
                                                :param-beta :q-count :s-cache :target-firms
                                                :theta :total-effort])
                :scalars [{:type :int :value (int n-friends)}]
                :n n-active-capacity}
               {:kname (:kernel-name (first (:produce-output kernels)))
                :arrays (mapv #(get buffers %) [:param-a :param-b :param-beta :total-effort :output :alive])
                :scalars []
                :n max-firms}
               {:kname (:kernel-name (first (:distribute-income kernels)))
                :arrays (mapv #(get buffers %) [:income :output :firm-size :alive :members :member-offsets])
                :scalars []
                :n max-firms}
               {:kname (:kernel-name (first (:zero-firm-sizes kernels)))
                :arrays [(:firm-size buffers)]
                :scalars []
                :n max-firms}
               {:kname (:kernel-name (first (:histogram-firms kernels)))
                :arrays [(:firm-size buffers) (:current-firm buffers)]
                :scalars []
                :n n-agents}
               {:kname (:kernel-name (first (:update-alive kernels)))
                :arrays [(:alive buffers) (:firm-size buffers)]
                :scalars []
                :n max-firms}]
        cache (when load-cache? (load-cache-fn device-id))
        tuning-results (atom {})]
    (doseq [{:keys [kname arrays scalars n]} specs]
      (if (and load-cache? (get cache kname))
        (do
          (apply-cached! kname device-id)
          (println (str "[autotune] " kname " loaded from cache")))
        (swap! tuning-results assoc kname
               (autotune! kname arrays scalars n device-id
                          :wg-sizes wg-sizes :warmup warmup :timed timed))))
    (when (seq @tuning-results)
      (save-cache! device-id @tuning-results))
    @tuning-results))

(defn- sync-bufs-to-soa!
  [sess ^AgentSoA agents ^FirmSoA firms]
  (gpu/sync-to-arrays! sess
                       [[(.effort agents) :effort]
                        [(.income agents) :income]
                        [(.firm agents) :current-firm]
                        [(.te firms) :total-effort]
                        [(.output firms) :output]
                        [(.n-workers firms) :firm-size]
                        [(.alive firms) :alive]
                        [(.a firms) :param-a]
                        [(.b firms) :param-b]
                        [(.beta firms) :param-beta]
                        [(.members firms) :members]
                        [(.offsets firms) :member-offsets]]))

(defn bench-gpu!
  "Benchmark one period of the GPU-fused firms ABM pipeline."
  ([n-agents] (bench-gpu! n-agents {}))
  ([n-agents opts]
   (let [{:keys [device-id warmup timed]
          :or {device-id :ze:0 warmup 3 timed 5}} opts
         config (firms/default-config (long n-agents))
         [_ agents firms rng] (firms/init-simulation config)
         n-active (long (max 1 (int (Math/round (* (.activation-rate config)
                                                   (.n-agents config))))))]
     (gpu/with-gpu-session* device-id
       (fn [sess]
         (compile-abm-kernels! sess)
         (make-abm-buffers sess agents firms n-active)
         (dotimes [_ warmup]
           (run-period-gpu-fused! sess rng n-active))
         (let [times (mapv (fn [_]
                             (let [t0 (System/nanoTime)
                                   _ (run-period-gpu-fused! sess rng n-active)
                                   t1 (System/nanoTime)]
                               (/ (- t1 t0) 1e6)))
                           (range timed))]
           {:n (.n-agents config)
            :mean-ms (/ (apply + times) (count times))
            :min-ms (apply min times)
            :max-ms (apply max times)
            :times times}))))))

(defn run-simulation-gpu
  "Run the full firms simulation on GPU and return per-period stats."
  ([config] (run-simulation-gpu config :ze:0))
  ([config device-id]
   (let [[config agents firms rng] (firms/init-simulation config)
         n-active (long (max 1 (int (Math/round (* (.activation-rate config)
                                                   (.n-agents config))))))
         n-periods (.n-periods config)]
     (gpu/with-gpu-session* device-id
       (fn [sess]
         (compile-abm-kernels! sess)
         (make-abm-buffers sess agents firms n-active)
         (let [{:keys [n-agents max-firms]} (require-layout sess)]
           (loop [period 0 stats (transient [])]
             (if (>= period n-periods)
               (persistent! stats)
               (let [cnt (run-period-gpu-fused! sess rng n-active)]
                 (sync-bufs-to-soa! sess agents firms)
                 (recur (inc period)
                        (conj! stats
                               (firms/collect-stats agents firms n-agents max-firms period cnt))))))))))))
