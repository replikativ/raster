(ns raster.ad.canonical-resident-device-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.link-module-plan :as module-plan]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.link-module :as link-module]
            [raster.numeric :as numeric]))

(defn- reason [invoke]
  (try (invoke) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- square-map []
  (let [tensor (av/tensor {:dtype :float :shape '[n]})
        scalar (av/tensor {:dtype :long :shape []})
        equation
        (list '= 'square-map '[y]
              (list 'map {:index 'i :extent 'n} '[x] []
                    (soac/lambda-form
                     '[element] [] '[(raster.numeric/* element element)])))]
    (soac/make
     (soac/default-program-facts
      {:values {'n scalar 'x tensor 'y tensor}
       :inputs '[n x]
       :equations {'square-map (soac/default-equation-facts)}})
     [equation] '[y])))

(defn- authority []
  (let [checked (normalized/linearize-map-program
                 (square-map) (the-ns 'raster.ad.canonical-resident-device-test))
        owner (canonical/adapt checked)
        bundle (get-in owner [:derivation :bundle])
        residual-plan (canonical/execution-residual-plan owner)
        interface (:interface owner)
        inputs (into {} (map (juxt :id identity)) (:inputs interface))
        residuals (:owned-arrays residual-plan)
        residual (when (= 1 (count residuals)) (val (first residuals)))
        projections (normalized/execution-sources bundle)
        expected (normalized/revisions)]
    {:owner owner
     :guard! (normalized/execution-guard (:source-ns owner) expected projections)
     :residual-plan residual-plan
     :ids {:x (get-in inputs ['x :id])
           :n (get-in inputs ['n :id])
           :primal (get-in interface [:primal-result :id])
           :seed (get-in interface [:cotangent-input :id])
           :gradient (get-in interface [:cotangent-result :id])
           :residual (:id residual)}}))

(defn- check-authority! [{:keys [owner guard!]}]
  (guard!)
  (canonical/validate! owner)
  true)

(defn- compile-phases!
  [{:keys [owner] :as authority} target observe]
  (check-authority! authority)
  (let [options {:target target :schedule {:precision :f32-scalar}}
        compile-one
        (fn [phase]
          (check-authority! authority)
          (observe phase)
          (let [compiled (equation-first/compile-typed (get owner phase) options)]
            (check-authority! authority)
            compiled))]
    {:primal (compile-one :primal)
     :transpose (compile-one :transpose)}))

(defn- resident-module
  [authority target observe]
  (let [{:keys [ids residual-plan]} authority
        {:keys [x primal seed gradient residual]} ids
        compilations (compile-phases! authority target observe)
        logical-ids [x primal residual seed gradient]
        node-id (fn [id] [::storage id])
        tensor (fn [ownership]
                 (av/tensor {:dtype :float :shape [3] :ownership ownership
                             :memory-space :device :placement {:device target}}))
        node (fn [id role]
               (link-plan/node {:id (node-id id) :dtype :float :shape [3]
                                :device target :role role}))
        nodes (into {}
                    (map (fn [id]
                           (let [input? (contains? #{x seed} id)
                                 n (node id (if input? :input :output))]
                             [(:id n) n])))
                    logical-ids)
        values (into {}
                     (map (fn [id]
                            (let [value (link-plan/value
                                         {:id id :abstract (tensor :owned)
                                          :leaves [{:name :value :node (node-id id)}]})]
                              [id value])))
                     logical-ids)
        scalar-values {'n {:type :long :value (long 3)}}
        bind (fn [entry phase bindings]
               (invocation-link/bind-resident-program
                (get-in compilations [phase :emitted])
                {:id entry :target target :nodes nodes :values values
                 :buffer-bindings bindings :scalar-values scalar-values}))
        primal-instance (bind :canonical-primal :primal
                              {x x primal primal residual residual})
        transpose-instance (bind :canonical-transpose :transpose
                                 {seed seed residual residual gradient gradient})
        spine (link-plan/make
               {:id :canonical-resident-spine :target target
                :nodes (vec (vals nodes)) :values (vec (vals values))
                :instances [primal-instance transpose-instance] :outputs []})
        module (module-plan/make
                {:id :canonical-resident-ad :spine spine
                 :entries [{:name :primal :instances [:canonical-primal]
                            :interface {:inputs [x] :state [] :outputs [primal]}}
                           {:name :transpose :instances [:canonical-transpose]
                            :interface {:inputs [seed] :state [] :outputs [gradient]}}]})]
    (check-authority! authority)
    {:authority authority :compilations compilations :module module
     :nodes nodes :values values :node-id node-id :residual-plan residual-plan
     :instances {:primal primal-instance :transpose transpose-instance}}))

(defn- guarded-call!
  [authority invoke]
  (check-authority! authority)
  (let [lease (invoke)]
    (try
      (check-authority! authority)
      lease
      (catch Throwable error
        (try
          (.close ^java.io.Closeable lease)
          (catch Throwable cleanup-error
            (.addSuppressed error cleanup-error)))
        (throw error)))))

(defn- guarded-reason!
  [authority invoke]
  (try
    (let [lease (guarded-call! authority invoke)]
      (try nil (finally (.close ^java.io.Closeable lease))))
    (catch clojure.lang.ExceptionInfo error
      (check-authority! authority)
      (:reason (ex-data error)))))

(defn- lease-output
  [owner lease value-id]
  (let [payload-fn (ns-resolve 'raster.gpu.link-module 'opaque-payload)
        payload (payload-fn owner)
        resident (get-in @lease [:outputs value-id])
        output (float-array 3)]
    ;; The named-entry landing deliberately exposes resident views, not DeviceArrays. This
    ;; acceptance-only projection uses the owner's private session; it is not a public AD API.
    (gpu/download-range! (:session payload) resident output {:elements 3})
    output))

(defn- fmul [left right] (float (* (float left) (float right))))
(defn- fadd [left right] (float (+ (float left) (float right))))
(defn- primal-oracle [xs] (mapv #(fmul % %) xs))
(defn- transpose-oracle [xs seeds]
  (mapv (fn [x seed] (fadd (fmul seed x) (fmul seed x))) xs seeds))
(defn- fbits [value] (Float/floatToRawIntBits (float value)))
(defn- raw-bits [values] (mapv fbits values))

(defn- assert-structure!
  [{:keys [authority compilations module nodes values node-id instances residual-plan]}]
  (let [{:keys [ids owner]} authority
        {:keys [x primal seed gradient residual]} ids
        residual-binding (val (first (:owned-arrays residual-plan)))
        frontiers (:frontiers module)
        interfaces (mapv #(select-keys % [:name :instances :interface]) (:entries module))]
    (is (= 1 (count (:owned-arrays residual-plan))))
    (is (empty? (:immutable-scalars residual-plan)))
    (is (empty? (:physical-bindings residual-plan)))
    (is (identical? residual-binding (val (first (:residual-bindings owner))))
        "the execution plan retains the canonical residual descriptor, including its origin")
    (is (= residual (:id residual-binding)
           (get-in residual-binding [:producer :result])))
    (is (= (:value residual-binding)
           (get-in (soac/facts (:primal owner)) [:values residual])))
    (is (every? #(not= % (node-id %)) [x primal residual seed gradient]))
    (is (= :owned (get-in values [residual :abstract :ownership])))
    (is (= :owned
           (get-in compilations [:transpose :emitted :values residual :ownership]))
        "the canonical tape snapshot remains owned in the derivative program")
    (is (= :owned
           (get-in instances [:transpose :call :program :values residual :ownership])))
    (is (= :borrowed
           (get-in compilations [:transpose :emitted :values seed :ownership])))
    (is (= :borrowed
           (get-in instances [:transpose :call :program :values seed :ownership])))
    (is (= :owned (get-in values [seed :abstract :ownership]))
        "the module owns the seed allocation while the emitted consumer borrows it")
    (is (= [{:name :primal :instances [:canonical-primal]
             :interface {:inputs [x] :state [] :outputs [primal]}}
            {:name :transpose :instances [:canonical-transpose]
             :interface {:inputs [seed] :state [] :outputs [gradient]}}]
           interfaces))
    (is (= [{:value residual :nodes [(node-id residual)]
             :producer :primal :consumer :transpose}]
           frontiers))
    (is (= :owned (get-in nodes [(node-id residual) :view :allocation :ownership])))))

(defn- run-device-case! [target]
  (let [authority (authority)
        compiled (resident-module authority target (fn [_] nil))
        {:keys [x primal seed gradient residual]} (:ids authority)
        initial-values [-2.0 0.5 4.0]
        initial (float-array (map float initial-values))
        initial-before (vec initial)
        seeds [[0.5 -0.25 2.0] [-1.0 1.0 0.125]]]
    (assert-structure! compiled)
    (let [owner (link-module/instantiate! (:module compiled))]
      (try
        (let [primal-lease
            (guarded-call!
             authority #(link-module/run-and-lease! owner :primal {:inputs {x initial}}))]
          (try
            (is (= (raw-bits (primal-oracle initial-values))
                   (raw-bits (lease-output owner primal-lease primal))))
            (is (= initial-before (vec initial)))
            (dotimes [index 3] (aset initial index (float 99.0)))
            (doseq [seed-values seeds]
              (let [seed-array (float-array (map float seed-values))
                    seed-before (vec seed-array)
                    transpose-lease
                    (guarded-call!
                     authority
                     #(link-module/run-and-lease!
                       owner :transpose
                       {:inputs {seed seed-array} :dependencies {residual primal-lease}}))]
                (try
                  (is (= (raw-bits (transpose-oracle initial-values seed-values))
                         (raw-bits (lease-output owner transpose-lease gradient))))
                  (is (= seed-before (vec seed-array)))
                  (finally (.close ^java.io.Closeable transpose-lease)))))
            (is (= :link-module-runtime-lease-active
                   (guarded-reason!
                    authority
                    #(link-module/run-and-lease!
                      owner :primal {:inputs {x (float-array [1.0 2.0 -4.0])}}))))
            (finally (.close ^java.io.Closeable primal-lease)))
          (is (= :link-module-runtime-lease-released
                 (guarded-reason!
                  authority
                  #(link-module/run-and-lease!
                    owner :transpose
                    {:inputs {seed (float-array [1.0 1.0 1.0])}
                     :dependencies {residual primal-lease}})))))
        (let [changed-values [1.0 2.0 -4.0]
              changed (float-array (map float changed-values))
              changed-before (vec changed)
              primal-lease
              (guarded-call!
               authority #(link-module/run-and-lease! owner :primal {:inputs {x changed}}))]
          (try
            (is (= (raw-bits (primal-oracle changed-values))
                   (raw-bits (lease-output owner primal-lease primal))))
            (let [seed-values [0.25 -2.0 0.5]
                  transpose-lease
                  (guarded-call!
                   authority
                   #(link-module/run-and-lease!
                     owner :transpose
                     {:inputs {seed (float-array (map float seed-values))}
                      :dependencies {residual primal-lease}}))]
              (try
                (is (= (raw-bits (transpose-oracle changed-values seed-values))
                       (raw-bits (lease-output owner transpose-lease gradient))))
                (finally (.close ^java.io.Closeable transpose-lease))))
            (is (= changed-before (vec changed)))
            (finally (.close ^java.io.Closeable primal-lease))))
        (finally (.close ^java.io.Closeable owner))))))

(deftest execution-guards-stop-stale-rules-and-callee-replacement-before-observation
  (let [checked (authority)
        revision (normalized/revisions)]
    (doseq [[label replacements expected]
            [[:stale {#'normalized/revisions
                      (fn [] (update revision :compiler-revision inc))}
              :ad-normalized-stale-rules]
             [:callee {#'numeric/* (fn [& _] (float 999.0))}
              :ad-normalized-execution-callee]]]
      (testing (name label)
        (let [compiles (atom 0) runs (atom 0)]
          (with-redefs-fn
            (merge replacements
                   {#'equation-first/compile-typed
                    (fn [& _] (swap! compiles inc) (throw (Exception. "compile observed")))
                    #'link-module/run-and-lease!
                    (fn [& _] (swap! runs inc) (throw (Exception. "run observed")))})
            #(do
               (is (= expected
                      (reason (fn [] (resident-module checked :ocl:0
                                                      (fn [_] nil))))))
               (is (= expected
                      (reason
                       (fn []
                         (guarded-call!
                          checked
                          (fn []
                            (swap! runs inc)
                            (reify java.io.Closeable (close [_]))))))))))
          (is (zero? @compiles))
          (is (zero? @runs)))))))

(deftest a-post-run-guard-failure-closes-the-new-lease
  (let [checked (authority)]
    (doseq [cleanup-failure [nil (ex-info "lease cleanup" {:reason :fixture-cleanup})]]
      (let [calls (atom 0)
            closed (atom 0)
            lease (reify java.io.Closeable
                    (close [_]
                      (swap! closed inc)
                      (when cleanup-failure (throw cleanup-failure))))
            failure (ex-info "post-run guard" {:reason :fixture-post-run})
            guarded (assoc checked :guard!
                           (fn [] (when (= 2 (swap! calls inc)) (throw failure))))
            actual (try (guarded-call! guarded (fn [] lease))
                        nil (catch Throwable error error))]
        (is (identical? failure actual))
        (is (= 1 @closed))
        (is (= (if cleanup-failure [cleanup-failure] [])
               (vec (.getSuppressed ^Throwable actual))))))))

(deftest canonical-resident-primal-and-transpose-run-on-opencl
  (if @ocl-probe/opencl-available?
    (run-device-case! :ocl:0)
    (ocl-probe/opencl-skip! "canonical resident primal/transpose acceptance")))

(deftest canonical-resident-primal-and-transpose-run-on-level-zero
  (if @ze-probe/gpu-available?
    (run-device-case! :ze:0)
    (ze-probe/gpu-skip! "canonical resident primal/transpose acceptance")))
