(ns raster.gpu.link-module-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.link-module-plan :as module-plan]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.structured-control-route :as route]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.link-module :as link-module]
            [raster.gpu.parallel-program :as parallel-program]
            [raster.gpu.resource-cleanup :as cleanup]))

(defn- reason [invoke]
  (try (invoke) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- error-of [invoke]
  (try (invoke) nil (catch Throwable error error)))

(defn- emitted-map [target id inputs output expression borrowed-inputs]
  (let [tensor (av/tensor {:dtype :float :shape [4]})
        borrowed (av/tensor {:dtype :float :shape [4] :ownership :borrowed})
        equation-id (symbol (str (name id) "-equation"))
        parameters (mapv #(symbol (str (name %) "-element")) inputs)
        equation (list '= equation-id (vector output)
                       (list 'map {:index 'i :extent 4} (vec inputs) []
                             (soac/lambda-form parameters [expression])))
        facts (soac/default-program-facts
               {:values (assoc (into {} (map (fn [input]
                                               [input (if (contains? borrowed-inputs input)
                                                        borrowed tensor)]))
                                     inputs)
                               output tensor)
                :inputs (vec inputs)
                :equations {equation-id (soac/default-equation-facts)}})
        typed (soac/make facts [equation] [output])
        promoted (route/promote-typed-algorithm typed)
        options {:target-device target :dtype :float
                 :array-types (zipmap inputs (repeat :float))}
        scheduled (route/schedule-program promoted options)]
    (:program (c-family/emit-program scheduled
                                     {:target-dialect :opencl-portable :dtype :float}))))

(defn- program-instance [target id input-bindings output-binding expression]
  (let [inputs (vec (keys input-bindings))
        output (first (keys output-binding))
        emitted (emitted-map target id inputs output expression
                             (if (= id :b) #{'frontier} #{}))
        call (program-call/make emitted (merge input-bindings output-binding) {} {} nil)]
    (link-plan/program-instance {:id id :call call})))

(defn- runtime-module [target]
  (let [abstract (av/tensor {:dtype :float :shape [4] :ownership :owned
                             :memory-space :device :placement {:device target}})
        a (program-instance target :a {'x :x} {'frontier :frontier}
                            'x-element)
        b (program-instance target :b {'frontier :frontier 'seed :seed} {'result :result}
                            '(clojure.core/+ frontier-element seed-element))
        node (fn [id role source]
               (link-plan/node (cond-> {:id id :dtype :float :shape [4]
                                        :device target :role role}
                                 source (assoc :source source))))
        value (fn [id]
                (link-plan/value {:id id :abstract abstract
                                  :leaves [{:name :value :node id}]}))
        spine (link-plan/make
               {:id :entry-runtime-spine :target target
                :nodes [(node :x :input (float-array 4))
                        (node :frontier :output nil)
                        (node :seed :input (float-array 4))
                        (node :result :output nil)]
                :values (mapv value [:x :frontier :seed :result])
                :instances [a b] :outputs []})]
    (is (= :borrowed (get-in b [:call :program :values 'frontier :ownership])))
    (is (= :owned (get-in spine [:values :frontier :abstract :ownership])))
    (module-plan/make
     {:id :entry-runtime :spine spine
      :entries [{:name :a :instances [:a]
                 :interface {:inputs [:x] :state [] :outputs [:frontier]}}
                {:name :b :instances [:b]
                 :interface {:inputs [:seed] :state [] :outputs [:result]}}]})))

(defn- host-output [owner lease value-id]
  (let [payload (#'link-module/opaque-payload owner)
        resident (get-in @lease [:outputs value-id])
        output (float-array 4)]
    (gpu/download-range! (:session payload) resident output {:elements 4})
    output))

(defn- with-mocked-owner-runtime [order! run! test!]
  (let [events (atom [])]
    (with-redefs [gpu/make-session
                  (fn [target] (swap! events conj :session) (atom {:device-id target}))
                  gpu/close-session! (fn [_] (swap! events conj :close-session))
                  gpu-link/realize-plan-storage!
                  (fn [plan _ _ _ _ _ _]
                    {:key-by-allocation {} :owned-specs {}
                     :node-views (into {} (map (fn [[id node]] [id {:view (:view node)}]))
                                       (:nodes plan))})
                  gpu-link/resident-program-executor (fn [& _] {})
                  parallel-program/prepare-sequence-with!
                  (fn [& _] (let [prepared (Object.)]
                              (swap! events conj [:prepare prepared]) prepared))
                  parallel-program/execution-order (fn [& args] (apply order! args))
                  parallel-program/run-prepared! (fn [prepared] (run! prepared))
                  parallel-program/release-prepared!
                  (fn [prepared] (swap! events conj [:release prepared]))
                  gpu/upload-range! (fn [& _] (swap! events conj :upload))]
      (test! events))))

(defn- run-two-entry-case! [target]
  (let [owner (link-module/instantiate! (runtime-module target))]
    (try
      (is (= :link-module-runtime-input-source
             (reason #(link-module/run-and-lease! owner :a {:inputs {:x nil}}))))
      (is (some? (reason #(link-module/run-and-lease!
                           owner :a {:inputs {:x (double-array [1 2 3 4])}}))))
      (is (some? (reason #(link-module/run-and-lease!
                           owner :a {:inputs {:x (float-array [1 2])}}))))
      (let [x (float-array [1 2 3 4])
            x-before (vec x)
            reentrant-reason (atom nil)
            original-run parallel-program/run-prepared!
            a-lease
            (with-redefs [parallel-program/run-prepared!
                          (fn [prepared]
                            (reset! reentrant-reason (reason #(link-module/close! owner)))
                            (original-run prepared))]
              (link-module/run-and-lease! owner :a {:inputs {:x x}}))]
        (is (= x-before (vec x)))
        (is (= :link-module-runtime-active @reentrant-reason))
        (try
          (is (= :link-module-runtime-dependencies
                 (reason #(link-module/run-and-lease!
                           owner :b {:inputs {:seed (float-array 4)}}))))
          (doseq [[seed expected] [[(float-array [10 20 30 40]) [11.0 22.0 33.0 44.0]]
                                   [(float-array [1 1 1 1]) [2.0 3.0 4.0 5.0]]]]
            (let [b-lease (link-module/run-and-lease!
                           owner :b {:inputs {:seed seed}
                                     :dependencies {:frontier a-lease}})]
              (try
                (is (= expected (vec (host-output owner b-lease :result))))
                (is (= :link-module-runtime-lease-active
                       (reason #(link-module/run-and-lease!
                                 owner :b {:inputs {:seed seed}
                                           :dependencies {:frontier a-lease}}))))
                (finally (.close ^java.io.Closeable b-lease)))))
          (is (= :link-module-runtime-lease-active
                 (reason #(link-module/run-and-lease!
                           owner :a {:inputs {:x (float-array [5 6 7 8])}}))))
          (finally (.close ^java.io.Closeable a-lease))))
      (let [released (link-module/run-and-lease!
                      owner :a {:inputs {:x (float-array [5 6 7 8])}})]
        (.close ^java.io.Closeable released)
        (is (= :link-module-runtime-lease-released
               (reason #(link-module/run-and-lease!
                         owner :b {:inputs {:seed (float-array 4)}
                                   :dependencies {:frontier released}})))))
      (let [a-lease (link-module/run-and-lease!
                     owner :a {:inputs {:x (float-array [5 6 7 8])}})
            failure (ex-info "backend replay failure" {:fixture :primary})]
        (try
          (with-redefs [parallel-program/run-prepared! (fn [_] (throw failure))]
            (is (identical? failure
                            (error-of #(link-module/run-and-lease!
                                       owner :b {:inputs {:seed (float-array 4)}
                                                 :dependencies {:frontier a-lease}})))))
          (is (= :link-module-runtime-poisoned
                 (reason #(link-module/run-and-lease!
                           owner :b {:inputs {:seed (float-array 4)}
                                     :dependencies {:frontier a-lease}}))))
          (finally (.close ^java.io.Closeable a-lease))))
      (finally (.close ^java.io.Closeable owner)))))

(deftest exact-opaque-owner-and-prebackend-admission
  (let [candidate (runtime-module :ocl:0)
        contacts (atom 0)]
    (with-redefs [gpu/make-session (fn [_] (swap! contacts inc) (throw (Exception. "contact")))]
      (is (= :link-module-runtime-options
             (reason #(link-module/instantiate! candidate {:session :forged}))))
      (is (zero? @contacts)))
    (with-redefs [program-call/execution-order
                  (fn [& _] {:record-time-prologue [{:kernel-phase :forged}]
                             :per-replay [] :completion :unproven})
                  gpu/make-session (fn [_] (swap! contacts inc) (throw (Exception. "contact")))]
      (is (= :link-module-runtime-prologue
             (reason #(link-module/instantiate! candidate))))
      (is (zero? @contacts) "raw prologue refusal precedes all backend contact"))
    (is (= :link-module-runtime-owner
           (reason #(link-module/close!
                     (raster.gpu.link_module.NamedEntryOwner. (atom {}))))))
    (let [ids (mapv #(keyword (str "output-" %)) (range 12))
          outputs (#'link-module/ordered-output-map ids identity)]
      (is (= ids (vec (keys outputs)))))))

(deftest prepared-correspondence-failure-releases-the-prepared-before-the-session
  (let [failure (ex-info "observer failure" {:fixture :observer})]
    (with-mocked-owner-runtime
      (fn [& _] (throw failure))
      (fn [_] nil)
      (fn [events]
        (is (identical? failure (error-of #(link-module/instantiate!
                                           (runtime-module :ocl:0)))))
        (let [names (mapv #(if (vector? %) (first %) %) @events)]
          (is (= [:session :prepare :release :close-session] names)))))))

(deftest mocked-owner-enforces-dependencies-reentrancy-poison-and-cleanup
  (let [run-hook (atom (fn [] nil))]
    (with-mocked-owner-runtime
     (fn [& _] {:record-time-prologue [] :per-replay [] :completion :unproven})
     (fn [_] (@run-hook))
     (fn [events]
       (let [owner (link-module/instantiate! (runtime-module :ocl:0))]
         (try
           (is (= :link-module-runtime-owner
                  (reason #(link-module/close!
                            (raster.gpu.link_module.NamedEntryOwner.
                             (.-payload ^raster.gpu.link_module.NamedEntryOwner owner)))))
               "a second instance sharing the private payload is not the sealed owner")
           (is (= :link-module-runtime-input-source
                  (reason #(link-module/run-and-lease! owner :a {:inputs {:x nil}}))))
           (is (some? (reason #(link-module/run-and-lease!
                                owner :a {:inputs {:x (double-array 4)}}))))
           (is (some? (reason #(link-module/run-and-lease!
                                owner :a {:inputs {:x (float-array 2)}}))))
           (let [nested (atom nil)]
             (reset! run-hook #(reset! nested
                                       (reason (fn [] (link-module/run-and-lease!
                                                       owner :a {:inputs {:x (float-array 4)}})))))
             (let [a-lease (link-module/run-and-lease!
                            owner :a {:inputs {:x (float-array [1 2 3 4])}})]
               (is (= :link-module-runtime-active @nested))
               (is (= :link-module-runtime-lease-active
                      (reason #(link-module/close! owner))))
               (is (= :link-module-runtime-dependencies
                      (reason #(link-module/run-and-lease!
                                owner :b {:inputs {:seed (float-array 4)}}))))
               (is (= :link-module-runtime-dependencies
                      (reason #(link-module/run-and-lease!
                                owner :b {:inputs {:seed (float-array 4)}
                                          :dependencies {:frontier a-lease
                                                         :extra a-lease}}))))
               (is (= :link-module-runtime-lease
                      (reason #(link-module/run-and-lease!
                                owner :b {:inputs {:seed (float-array 4)}
                                          :dependencies
                                          {:frontier
                                           (raster.gpu.link_module.EntryLease. (atom {}))}}))))
               (let [owner-state (:state (#'link-module/opaque-payload owner))
                     generation (get-in @owner-state [:generations :frontier])]
                 (swap! owner-state update-in [:generations :frontier] inc)
                 (try
                   (is (= :link-module-runtime-dependency-generation
                          (reason #(link-module/run-and-lease!
                                    owner :b {:inputs {:seed (float-array 4)}
                                              :dependencies {:frontier a-lease}}))))
                   (finally
                     (swap! owner-state assoc-in [:generations :frontier] generation))))
               (let [foreign-owner (link-module/instantiate! (runtime-module :ocl:0))]
                 (try
                   (reset! run-hook (fn [] nil))
                   (let [foreign-lease
                         (link-module/run-and-lease!
                          foreign-owner :a {:inputs {:x (float-array 4)}})]
                     (try
                       (is (= :link-module-runtime-lease-owner
                              (reason #(link-module/run-and-lease!
                                        owner :b {:inputs {:seed (float-array 4)}
                                                  :dependencies {:frontier foreign-lease}}))))
                       (finally (.close ^java.io.Closeable foreign-lease))))
                   (finally (link-module/close! foreign-owner))))
               (let [failure (ex-info "mock replay failure" {:fixture :primary})]
                 (reset! run-hook #(throw failure))
                 (is (identical? failure
                                 (error-of #(link-module/run-and-lease!
                                            owner :b {:inputs {:seed (float-array 4)}
                                                      :dependencies {:frontier a-lease}}))))
                 (is (= :link-module-runtime-poisoned
                        (reason #(link-module/run-and-lease!
                                  owner :b {:inputs {:seed (float-array 4)}
                                            :dependencies {:frontier a-lease}})))))
               ;; Global poison does not strand an already published dependency lease.
               (.close ^java.io.Closeable a-lease)))
           (finally (link-module/close! owner)))
         (is (= [:release :release :close-session
                 :release :release :close-session]
                (->> @events
                     (filter #(or (= :close-session %)
                                  (and (vector? %) (= :release (first %)))))
                     (mapv #(if (vector? %) (first %) %))))))))))

(deftest upload-failure-poisons-without-publishing-a-lease
  (let [failure (ex-info "mock upload failure" {:fixture :upload})]
    (with-mocked-owner-runtime
      (fn [& _] {:record-time-prologue [] :per-replay [] :completion :unproven})
      (fn [_] nil)
      (fn [_]
        (let [owner (link-module/instantiate! (runtime-module :ocl:0))]
          (try
            (with-redefs [gpu/upload-range! (fn [& _] (throw failure))]
              (is (identical? failure
                              (error-of #(link-module/run-and-lease!
                                         owner :a {:inputs {:x (float-array 4)}})))))
            (is (= :link-module-runtime-poisoned
                   (reason #(link-module/run-and-lease!
                             owner :a {:inputs {:x (float-array 4)}}))))
            (finally (link-module/close! owner))))))))

(deftest lease-cleanup-debt-remains-registered-and-retryable
  (with-mocked-owner-runtime
    (fn [& _] {:record-time-prologue [] :per-replay [] :completion :unproven})
    (fn [_] nil)
    (fn [_]
      (let [owner (link-module/instantiate! (runtime-module :ocl:0))
            lease (link-module/run-and-lease!
                   owner :a {:inputs {:x (float-array 4)}})
            lease-cleanup (:cleanup (#'link-module/opaque-payload lease))
            cleanup-state (:state lease-cleanup)
            original-release (get-in @cleanup-state [:remaining 0 :release])
            attempts (atom 0)]
        (try
          (swap! cleanup-state assoc-in [:remaining 0 :release]
                 #(if (= 1 (swap! attempts inc))
                    (throw (ex-info "retryable cleanup debt"
                                    {:reason :fixture-cleanup :cleanup-retry-safe? true}))
                    (original-release)))
          (is (= :fixture-cleanup (reason #(.close ^java.io.Closeable lease))))
          (is (= :link-module-runtime-lease-active
                 (reason #(link-module/close! owner)))
              "failed root-use release keeps the exact lease registered")
          (.close ^java.io.Closeable lease)
          (is (= 2 @attempts))
          (finally
            (when-not @(get (#'link-module/opaque-payload lease) :released?)
              (try (.close ^java.io.Closeable lease) (catch Throwable _)))
            (link-module/close! owner)))))))

(deftest failed-lease-issuance-retains-cleanup-debt-for-owner-close
  (with-mocked-owner-runtime
    (fn [& _] {:record-time-prologue [] :per-replay [] :completion :unproven})
    (fn [_] nil)
    (fn [events]
      (let [owner (link-module/instantiate! (runtime-module :ocl:0))
            issuance (ex-info "lease issuance failed" {:fixture :issuance})
            release-debt (ex-info "root use release failed"
                                  {:fixture :debt :cleanup-retry-safe? true})
            original-sealed @#'link-module/sealed
            original-release cleanup/release!
            release-attempts (atom 0)]
        (with-redefs-fn
          {#'link-module/sealed
           (fn [kind fields]
             (if (= :lease kind) (throw issuance) (original-sealed kind fields)))
           #'cleanup/release!
           (fn [resource]
             (if (= 1 (swap! release-attempts inc))
               (throw release-debt)
               (original-release resource)))}
          #(let [failure (error-of (fn [] (link-module/run-and-lease!
                                           owner :a {:inputs {:x (float-array 4)}})))]
             (is (identical? issuance failure))
             (is (some (fn [error] (identical? release-debt error))
                       (.getSuppressed ^Throwable failure)))
             (is (= :link-module-runtime-poisoned
                    (reason (fn [] (link-module/run-and-lease!
                                    owner :a {:inputs {:x (float-array 4)}})))))
             (link-module/close! owner)))
        (is (= 3 @release-attempts)
            "issuance rollback, owner debt retry, and root cleanup each execute once")
        (is (= [:release :release :close-session]
               (->> @events
                    (filter #(or (= :close-session %)
                                 (and (vector? %) (= :release (first %)))))
                    (mapv #(if (vector? %) (first %) %)))))))))

(deftest unresolved-retain-use-receipt-is-adopted-by-owner-close
  (with-mocked-owner-runtime
    (fn [& _] {:record-time-prologue [] :per-replay [] :completion :unproven})
    (fn [_] nil)
    (fn [events]
      (let [owner (link-module/instantiate! (runtime-module :ocl:0))
            original-retain cleanup/retain-use!
            failure
            (with-redefs [cleanup/retain-use!
                          (fn [root]
                            (let [use (original-retain root)]
                              (throw (ex-info "retain-use returned unresolved cleanup"
                                              {::cleanup/unresolved (::cleanup/owner use)
                                               :reason :fixture-retain-use}))))]
              (error-of #(link-module/run-and-lease!
                          owner :a {:inputs {:x (float-array 4)}})))]
        (is (= :fixture-retain-use (:reason (ex-data failure))))
        (is (= :link-module-runtime-poisoned
               (reason #(link-module/run-and-lease!
                         owner :a {:inputs {:x (float-array 4)}}))))
        (link-module/close! owner)
        (is (= [:release :release :close-session]
               (->> @events
                    (filter #(or (= :close-session %)
                                 (and (vector? %) (= :release (first %)))))
                    (mapv #(if (vector? %) (first %) %)))))))))

(deftest opencl-named-entries-share-one-owner-and-frontier-generation
  (if @ocl-probe/opencl-available?
    (run-two-entry-case! :ocl:0)
    (ocl-probe/opencl-skip! "named entry shared owner")))

(deftest level-zero-named-entries-share-one-owner-and-frontier-generation
  (if @ze-probe/gpu-available?
    (run-two-entry-case! :ze:0)
    (ze-probe/gpu-skip! "named entry shared owner")))
