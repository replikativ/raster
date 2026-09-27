(ns raster.gpu.program-memory-order-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.nn :as nn]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.analysis.physical-liveness :as liveness]
            [raster.compiler.ir.link-plan :as plan]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.passes.local-storage-reuse :as reuse]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as program]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as device]))

(deftm four-layers
  [w :- (Array double) b :- (Array double) x :- (Array double)] :- (Array double)
  (let [a (nn/dense w x b)
        middle (nn/dense w a b)
        c (nn/dense w middle b)]
    (nn/dense w c b)))

(defn- arguments []
  [(double-array [1.0 0.25 -0.5 1.0])
   (double-array [0.25 -0.125])
   (double-array [1.0 -2.0])])

(defn- lowered [target args]
  (equation/lower (equation/compile #'four-layers {:target target :dtype :double}) args))

(defn- assert-candidate [linked order]
  (let [report (liveness/report linked order)
        candidates (:proposals report)]
    (is (seq candidates) "real typed writes and selected execution order must meet")
    (doseq [candidate candidates]
      (is (= :witnessed (get-in candidate [:runtime-order :status])))
      (is (= :witnessed (:cross-replay-initialization candidate)))
      (is (= #{:alias-realization :completion-and-escape} (:pending candidate))))
    (is (= :unproven (:reuse report)))
    (is (= :unproven (:completion order)))))

(defn- selected-order [linked]
  (let [instance (first (:instances linked))]
    (-> (program-call/execution-order (:call instance))
        (assoc :plan (:id linked) :target (:target linked))
        (update :per-replay
                #(mapv (fn [entry] (assoc-in entry [:source :instance] (:id instance))) %)))))

(deftest public-tap-prevents-private-storage-reuse
  (doseq [[target available? skip!] [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
                                    [:ze:0 device/gpu-available? device/gpu-skip!]]]
    (if-not @available?
      (skip! (str "retained intermediate on " target))
      (let [[w b x :as args] (arguments)
            prepared (compiled/lower #'four-layers args
                                     {:compiler :equation-first :target target :dtype :double
                                      :taps '[a]})
            linked (compiled/plan prepared)
            expected {:result (vec (apply four-layers args))
                      :a (vec (nn/dense w x b))}
            result (link/evaluate! linked)]
        (is (= 0 (get-in result [:memory :allocations-saved])))
        (is (= 2 (count (:outputs result))))
        (doseq [{:keys [key node]} (:out-tree prepared)]
          (is (= (get expected key) (vec (get-in result [:outputs node])))
              (str target " " key)))))))

(deftest typed-program-order-meets-complete-write-evidence
  (let [linked (lowered :ocl:0 (arguments))
        call (get-in linked [:instances 0 :call])
        visited (atom [])
        prepared (program/prepare-with!
                  call {:bind! (fn [key graph _ _] {:key key :graph graph})
                        :run! #(swap! visited conj (:key %))
                        :release! (fn [_])})
        executable (link/map->LinkedExecutable
                    {:plan linked :prepared-program prepared
                     :closed? (atom false) :pending-inputs (atom #{})})]
    (try
      (is (= :parallel-program-graph-order
             (try (program/execution-order prepared (constantly nil))
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (with-redefs [gpu/kernel-graph-execution-order
                    (fn [_ handle]
                      {:record-time-prologue []
                       :per-replay (mapv #(hash-map :kernel-phase (:id %))
                                         (get-in handle [:graph :nodes]))})]
        (let [order (link/execution-order executable)]
          (is (= (selected-order linked) order))
          (assert-candidate linked order)
          (let [report (liveness/report linked order)
                candidate (first (:proposals report))
                left (first (get-in report [:slots (:from candidate) :nodes]))
                right (first (get-in report [:slots (:to candidate) :nodes]))
                realized (reuse/realize-one linked order)
                rewritten (:plan realized)
                no-candidate-plans
                [(update linked :outputs conj left)
                 (assoc-in linked [:nodes left :source] (double-array 2))
                 (assoc-in linked [:nodes right :view :allocation :coherence] :host-coherent)
                 (-> linked
                     (assoc-in [:nodes left :view :allocation :byte-size] 32)
                     (assoc-in [:nodes right :view :allocation :byte-size] 32))]]
            (is (= 1 (:allocations-saved realized)))
            (is (= rewritten (plan/validate! rewritten)))
            (is (= #{#{left right}} (:aliases rewritten)))
            (is (empty? (:aliases linked)))
            (is (not= (get-in linked [:nodes left :view :allocation :id])
                      (get-in linked [:nodes right :view :allocation :id])))
            (doseq [declined no-candidate-plans]
              (is (= {:plan declined :allocations-saved 0 :bytes-saved 0}
                     (reuse/realize-one declined order)))))
          (program/run-prepared! prepared)
          (is (= (:binding-order prepared) @visited))
          (is (= (set (keys (get-in prepared [:plan :step-keys])))
                 (set (map #(get-in % [:source :step]) (:per-replay order)))))
          (is (every? :complete-write?
                      (filter #(= :write (:access %)) (:accesses (plan/memory-report linked)))))))
      (with-redefs [gpu/kernel-graph-execution-order
                    (fn [_ handle]
                      {:record-time-prologue [{:kernel-phase :hoisted}]
                       :per-replay (mapv #(hash-map :kernel-phase (:id %))
                                         (get-in handle [:graph :nodes]))})]
        (is (every? #(= :record-time-prologue (get-in % [:runtime-order :reason]))
                    (:proposals (liveness/report linked (link/execution-order executable))))))
      (finally (program/release-prepared! prepared)))
    (is (= :parallel-program-closed
           (try (program/execution-order prepared identity)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest selected-program-order-is-observation-only-on-local-devices
  (doseq [[target available? skip!] [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
                                    [:ze:0 device/gpu-available? device/gpu-skip!]]]
    (if-not @available?
      (skip! (str "equation-first memory witness on " target))
      (let [args (arguments)
            expected (vec (apply four-layers args))
            linked (lowered target args)]
        (with-open [executable (link/instantiate! linked)]
          (let [info (link/execution-info executable)]
            (is (= (count (get-in executable [:prepared-program :binding-order])) (count info)))
            (is (seq info))
            (is (every? #(= :fixed (get-in % [:executable :selection])) info))
            (is (every? #(seq (get-in % [:executable :entry-points])) info)))
          (is (= (selected-order linked) (link/execution-order executable)))
          (assert-candidate linked (link/execution-order executable))
          (dotimes [_ 2]
            (link/run! executable)
            (is (= expected (vec (link/download executable (first (:outputs linked))))))))))))

(deftest private-storage-realization-preserves-two-device-replays
  (doseq [[target available? skip!] [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
                                    [:ze:0 device/gpu-available? device/gpu-skip!]]]
    (if-not @available?
      (skip! (str "owned private storage reuse on " target))
      (let [args (arguments)
            expected (vec (apply four-layers args))
            linked (lowered target args)
            ;; The private helper exposes a handle only to this lifecycle test. evaluate! never
            ;; returns it, and the ordinary resident API still gives distinct temporary storage.
            {:keys [executable memory]} (#'link/prepare-private-reuse! linked)]
        (with-open [executable executable]
          (is (= 1 (:allocations-saved memory)))
          (is (= 16 (:bytes-saved memory)))
          (is (= (dec (:owned-allocations-before memory)) (:owned-allocations-after memory)))
          (dotimes [_ 2]
            (link/run! executable)
            (is (= expected (vec (link/download executable (first (:outputs linked))))))))
        (is @(:closed? executable))
        (let [result (link/evaluate! linked)]
          (is (= #{:outputs :memory} (set (keys result))))
          (is (= expected (vec (get (:outputs result) (first (:outputs linked))))))
          (is (= 1 (get-in result [:memory :allocations-saved]))))
        (let [preserved (update linked :nodes
                                (fn [nodes] (update-vals nodes #(if (= :internal (:role %))
                                                                  (assoc % :role :state) %))))
              result (link/evaluate! preserved)]
          (is (= 0 (get-in result [:memory :allocations-saved])))
          (is (= expected (vec (get (:outputs result) (first (:outputs preserved)))))))))))

(deftest reusable-private-scope-keeps-one-binding-and-detached-snapshots
  (doseq [[target available? skip!] [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
                                    [:ze:0 device/gpu-available? device/gpu-skip!]]]
    (if-not @available?
      (skip! (str "reusable private execution on " target))
      (let [args (arguments)
            linked (lowered target args)
            input-id (first (keep (fn [[id node]] (when (identical? (last args) (:source node)) id))
                                 (:nodes linked)))
            output-id (first (:outputs linked))
            replacement (double-array [2 3])
            instantiate link/instantiate!
            bindings (atom 0)
            snapshots
            (with-redefs [link/instantiate! (fn [plan] (swap! bindings inc) (instantiate plan {}))]
              (with-open [execute (link/private-executor! linked)]
                [(execute) (execute {input-id replacement})]))]
        (is (= 1 @bindings))
        (is (= (vec (apply four-layers args))
               (vec (get-in (first snapshots) [:outputs output-id]))))
        (is (= (vec (apply four-layers (conj (pop args) replacement)))
               (vec (get-in (second snapshots) [:outputs output-id]))))
        (is (= [1 1] (mapv #(get-in % [:memory :allocations-saved]) snapshots)))
        (is (= [16 16] (mapv #(get-in % [:memory :bytes-saved]) snapshots)))))))

(deftest private-realization-does-not-trust-an-internal-role-as-exclusivity
  (let [linked (lowered :ocl:0 (arguments))
        borrowed (:plan (plan/borrow-owned-storage linked))]
    (with-redefs [link/instantiate! (fn [& _] (throw (AssertionError. "must reject before allocation")))]
      (is (= :link-private-reuse-boundary
             (try (link/evaluate! borrowed)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest private-binding-verifies-proofs-and-closes-on-failure
  (let [linked (lowered :ocl:0 (arguments))
        order (selected-order linked)]
    (doseq [[failure expected] [[:order :link-private-reuse-order-changed]
                              [:count :link-private-reuse-allocation-count]
                              [:bind :injected] [:run :injected] [:download :injected]]]
      (let [created (atom []) closed (atom []) launched (atom false) bindings (atom 0)
            fail! #(throw (ex-info "injected lifecycle failure" {:reason :injected}))]
        (with-redefs [link/instantiate!
                  (fn [plan]
                    (swap! bindings inc)
                    (when (= failure :bind) (fail!))
                    (let [e (link/map->LinkedExecutable {:plan plan :ordinal (count @created)})]
                      (swap! created conj e)
                      e))
                  link/instantiation-report
                  (fn [_] {:owned-allocations (if (= failure :count) 7 6)})
                  link/execution-order
                  (fn [_] (if (= failure :order)
                            (update order :per-replay #(vec (reverse %))) order))
                  link/close! (fn [e] (swap! closed conj (:ordinal e)))
                  link/run! (fn [_] (reset! launched true) (when (= failure :run) (fail!)))
                  link/download (fn [& _] (fail!))]
      (is (= expected
             (try (link/evaluate! linked)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= 1 @bindings) "the proof must not instantiate a baseline")
      (is (= (if (= failure :bind) [] [0]) @closed))
      (is (= (contains? #{:run :download} failure) @launched)))))))
