(ns raster.compiler.ir.ordered-scratch-entry-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as parallel-program]
            [raster.par :as par]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- entry-fixture
  ([] (entry-fixture (float-array [81 82]) (float-array 2)))
  ([scratch out]
   (let [values {'scratch (av/tensor {:dtype :float :shape [2]})
                 'out (av/tensor {:dtype :float :shape [2]})}
         invocation (invocation/from-prefix
                     {:id :scratch-entry :parameters '[scratch out]
                      :parameter-values values :bindings [] :binding-values {}
                      :program-values values :program-inputs '[scratch out] :program-outputs '[out]})]
     {:arguments [scratch out]
      :state {:materialization (materialization/begin-materialization invocation [scratch out])
              :program {:values values} :target :test}})))

(defn- with-accesses [accesses f]
  ;; Entry policy only; emitted numerical integration below uses real checked programs.
  (with-redefs-fn {#'invocation-link/checked-ordered-validation (constantly {})
                  #'invocation-link/equation-boundaries (fn [& _] [{:buffer-accesses accesses}])}
    f))

(deftest scratch-intent-keeps-exact-class-capacity-and-host-alias-evidence
  (is (= :invocation-materialization-buffer
         (reason #(entry-fixture (Object.) (float-array 2)))))
  (is (= :invocation-materialization-buffer-dtype
         (reason #(entry-fixture (double-array 2) (float-array 2)))))
  (is (= :invocation-materialization-buffer-shape
         (reason #(entry-fixture (float-array 1) (float-array 2)))))
  (let [shared (float-array 2)
        {:keys [state]} (entry-fixture shared shared)]
    (is (= :invocation-ordered-source-alias
           (reason #(invocation-link/ordered-entry-layout state {'scratch :scratch 'out :state}))))))

(deftest scratch-layout-erases-only-owned-content-not-parameter-evidence
  (let [{:keys [state arguments]} (entry-fixture)
        original (get (invocation-link/ordered-entry-buffers state) 'scratch)
        roles (with-accesses {'scratch :read-write 'out :write}
                #(invocation-link/ordered-entry-roles state {'scratch :scratch}))
        layout (invocation-link/ordered-entry-layout state roles)
        token (:id original)
        node (some #(when (= token (:id %)) %) (:nodes layout))]
    (is (= :scratch (get roles 'scratch)))
    (is (= :scratch (get (:entry-refresh layout) token)))
    (is (= :scratch (:role node)))
    (is (= :owned (get-in node [:view :allocation :ownership])))
    (is (nil? (:source node)))
    (is (= :copy (:initialization original)))
    (is (identical? (first arguments) (:source original)))
    (is (identical? original (get (invocation-link/ordered-entry-buffers state) 'scratch)))
    (is (= :invocation-ordered-input-role
           (reason #(invocation-link/ordered-input-arguments
                     state arguments roles {'scratch (float-array 2)}))))
    (is (= :invocation-ordered-source-alias
           (reason #(invocation-link/ordered-input-arguments
                     state arguments {'scratch :scratch 'out :input} {'out (first arguments)})))
        "scratch erasure must not erase original host-alias evidence on input replacement")
    (let [coherent (-> state
                       (assoc :entry-roles roles :entry-refresh (:entry-refresh layout))
                       (assoc-in [:realized :storage token]
                                 {:role :scratch :source nil :initialization :unspecified}))]
      (is (= (:entry-refresh layout) (#'invocation-link/checked-ordered-entry-root-roles coherent)))
      (doseq [change [#(assoc % :role :state) #(dissoc % :role)
                      #(assoc % :source (first arguments)) #(assoc % :initialization :copy)]]
        (is (= :invocation-ordered-entry-role-root
               (reason #(#'invocation-link/checked-ordered-entry-root-roles
                         (update-in coherent [:realized :storage token] change)))))))
    (doseq [changed [(assoc state :entry-roles roles)
                     (assoc state :entry-roles roles :entry-refresh {})
                     (assoc state :entry-roles roles
                            :entry-refresh (assoc (:entry-refresh layout) token :state))
                     (assoc-in state [:realized :storage token :role] :scratch)
                     (-> state
                         (assoc :entry-roles (assoc roles 'scratch :state)
                                :entry-refresh (assoc (:entry-refresh layout) token :state))
                         (assoc-in [:realized :storage token :role] :scratch))]]
      (is (= :invocation-ordered-entry-role-root
             (reason #(#'invocation-link/checked-ordered-entry-root-roles changed)))))))

(deftm scratch-written-then-read
  [scratch :- (Array float), out :- (Array float), n :- Long] :- (Array float)
  (let [written (par/map! scratch i 2 float (+ 10.0 (float i)))
        ^long required ^{:raster.effect/preserve-evaluation true} (* (long n) (long n))]
    (par/map! out i 2 float (+ (aget scratch i) 1.0))))

(deftm scratch-prefix-then-read
  [scratch :- (Array float), out :- (Array float), n :- Long] :- (Array float)
  (let [written (par/map! scratch i 1 float 10.0)
        ^long required ^{:raster.effect/preserve-evaluation true} (* (long n) (long n))]
    (par/map! out i 2 float (+ (aget scratch i) 1.0))))

(deftm scratch-read-first
  [scratch :- (Array float), out :- (Array float), n :- Long] :- (Array float)
  (par/map! out i 2 float (+ (aget scratch i) (float n))))

(deftm scratch-partial-update-first
  [scratch :- (Array float), out :- (Array float), n :- Long] :- (Array float)
  (let [written (par/map-void! i n
                  (when (> (aget out i) 0.0) (aset scratch i 10.0)))
        ^long required ^{:raster.effect/preserve-evaluation true} (* (long n) (long n))]
    (par/map! out i 2 float (+ (aget scratch i) 1.0))))

(defn- owned-state [compilation arguments]
  (let [state (invocation-link/begin-ordered-realization compilation arguments nil)
        roles (invocation-link/ordered-entry-roles state {'scratch :scratch})
        layout (invocation-link/ordered-entry-layout state roles)]
    (assoc state :entry-roles roles :entry-refresh (:entry-refresh layout))))

(deftest scratch-owner-and-conditional-stage-have-distinct-readiness-contracts
  (doseq [[function complete? required?] [[#'scratch-written-then-read true false]
                                        [#'scratch-read-first false true]
                                        [#'scratch-partial-update-first true true]]]
    (let [compilation (equation-first/compile function {:target :ocl:0 :dtype :float})
          state (owned-state compilation [(float-array [81 82]) (float-array 2) 2])
          token (first (keep (fn [[token role]] (when (= :scratch role) token)) (:entry-refresh state)))
          first-entry (first (sort (keys (:equation-spans state))))
          state (assoc state :source-cursor first-entry)
          equation (get-in state [:equation-spans first-entry :equation])
          ready (invocation-link/realize-ready-equation state equation nil)
          storage (get-in ready [:state :realized :storage token])
          node (get-in ready [:stage-plan :nodes token])
          contract (:entry-contract ready)]
      (is (= :scratch (:role storage)))
      (is (= :unspecified (:initialization storage)))
      (is (nil? (:source storage)))
      (is (= :state (:role node)) "stage state is a conditional assumption, not owner intent")
      (is (nil? (:source node)))
      (is (= complete? (contains? (:full-produces contract) token)))
      (is (= required? (contains? (:requires contract) token)))
      ;; A conditional update is reconstructed as a full store with the old value in its
      ;; untaken branch. It produces the full view ONLY after its entry requirement is met.
      (is (empty? (:acquired-storage state)))
      (is (nil? (:pending state)))
      (when required?
        (is (= :link-ordered-uninitialized
               (reason #(#'link/ordered-require-initialized!
                         (get-in ready [:stage-plan :nodes]) [] #{token}))))))))

(defn- scratch-ready? [runtime token]
  (some #(= token (get-in % [:allocation :id])) (:initialized runtime)))

(defn- check-scratch-runtime! [target]
  (let [compilation (equation-first/compile #'scratch-written-then-read {:target target :dtype :float})
        scratch (float-array [81 82])
        uploads (atom [])
        upload gpu/upload-range!]
    (with-redefs [gpu/upload-range! (fn [& args] (swap! uploads conj (nth args 2)) (apply upload args))]
      (let [executable (link/instantiate-ordered-invocation!
                        compilation [scratch (float-array 2) 1] {:entry-roles {'scratch :scratch}})
            token (first (keep (fn [[token role]] (when (= :scratch role) token))
                               (get-in @(:execution-state executable) [:ordered :entry-refresh])))]
        (try
          (is (not (scratch-ready? (:ordered @(:execution-state executable)) token)))
          (is (= :link-ordered-entry-mutation
                 (reason #(link/upload! executable token (float-array [1 2])))))
          (doseq [replay [1 2]]
            (let [output (first (keys (link/run! executable)))
                  runtime (:ordered @(:execution-state executable))
                  final-plan (get-in @(:execution-state executable) [:resident-boundary :plan])]
              (is (= [11.0 12.0] (vec (link/download executable output))))
              (is (scratch-ready? runtime token))
              (is (= :scratch (get-in runtime [:entry-refresh token])))
              (is (= :scratch (get-in final-plan [:nodes token :role])))
              (is (nil? (get-in final-plan [:nodes token :source])))
              (is (= replay @(:completed-replays executable)))
              (aset scratch 0 99.0)))
          (is (not-any? #(identical? scratch %) @uploads)
              "scratch evidence never becomes a capture-time or replay upload")
          (is (= [99.0 82.0] (vec scratch)))
          (finally (link/close! executable))))))
  (doseq [[function expected-submissions]
          [[#'scratch-read-first 0] [#'scratch-partial-update-first 0] [#'scratch-prefix-then-read 1]]]
    (let [compilation (equation-first/compile function {:target target :dtype :float})
          executable (link/instantiate-ordered-invocation!
                      compilation [(float-array [81 82]) (float-array 2) 1]
                      {:entry-roles {'scratch :scratch}})
          submitted (atom 0) run parallel-program/run-prepared!
          preparations (atom 0) prepare parallel-program/prepare-sequence-with-retained-programs!]
      (try
        (with-redefs [parallel-program/prepare-sequence-with-retained-programs!
                      (fn [& args] (swap! preparations inc) (apply prepare args))
                      parallel-program/run-prepared!
                      (fn [& args]
                        ;; A sequence dispatches each leaf through this same function. Count
                        ;; actual leaf executions, not both the sequence and its child.
                        (when-not (parallel-program/prepared-sequence? (first args))
                          (swap! submitted inc))
                        (apply run args))]
          (is (= :link-ordered-uninitialized (reason #(link/run! executable)))))
        (is (= expected-submissions @preparations)
            "missing readiness rejects the stage before any program preparation")
        (is (= expected-submissions @submitted))
        (let [runtime (:ordered @(:execution-state executable))
              token (first (keep (fn [[token role]] (when (= :scratch role) token)) (:entry-refresh runtime)))]
          (is (not (some #(and (= token (get-in % [:allocation :id])) (= [2] (:shape %)))
                         (:initialized runtime)))
              "a produced prefix cannot initialize its untouched whole parent"))
        (finally (link/close! executable)))))
  (let [compilation (equation-first/compile #'scratch-written-then-read {:target target :dtype :float})
        executable (link/instantiate-ordered-invocation!
                    compilation [(float-array 2) (float-array 2) 1] {:entry-roles {'scratch :scratch}})]
    (try
      (with-redefs [parallel-program/run-prepared!
                    (fn [& _] (throw (ex-info "injected full-write failure" {:reason :scratch-write-failed})))]
        (is (= :scratch-write-failed (reason #(link/run! executable)))))
      (let [runtime (:ordered @(:execution-state executable))
            token (first (keep (fn [[token role]] (when (= :scratch role) token)) (:entry-refresh runtime)))]
        (is (not (scratch-ready? runtime token))))
      (is (zero? @(:completed-replays executable)))
      (finally (link/close! executable)))))

(deftest ordered-scratch-entry-uses-common-runtime-readiness-on-both-backends
  (if @opencl/opencl-available?
    (check-scratch-runtime! :ocl:0)
    (opencl/opencl-skip! "ordered scratch entry"))
  (if @ze/gpu-available?
    (check-scratch-runtime! :ze:0)
    (ze/gpu-skip! "ordered scratch entry")))
