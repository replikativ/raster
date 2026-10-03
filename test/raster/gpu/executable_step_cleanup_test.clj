(ns raster.gpu.executable-step-cleanup-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.gpu.core :as gpu]
            [raster.gpu.resource-cleanup :as cleanup]))

(def ^:private scan-graph
  (delay
    (let [node (soac/par-form->soac
                'scan-result '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i))) 92)]
      (emit/generate-scan-kernel-graph
       (lower/scan-kernel-graph node (lower/lower-scan node nil :dtype :float)
                                {:array-types {'values :float 'out :float}})))))

(defn- session []
  (atom {:device-id :ocl:0 :closed? false :graphs {} :prepared {} :kernel-graphs {}
         :buffers {:values {:id :values :dtype :float :n-elements 1025 :byte-size 4100}
                   :out {:id :out :dtype :float :n-elements 1025 :byte-size 4100}}
         :allocations
         (into {} (for [key [:values :out]]
                    [key (bview/allocation {:id key :byte-size 4100 :memory-space :device
                                            :device :ocl:0 :coherence :explicit-transfer
                                            :ownership :owned})]))}))

(defn- step [n]
  ;; A second legal graph-private allocation exercises successful-prefix rollback. Its
  ;; liveness optimization is irrelevant here: this test begins at executable binding.
  {:phase :scan :convention :executable
   :artifact (update @scan-graph :temporaries conj
                     (assoc (first (:temporaries @scan-graph)) :id 'extra-private))
   :argument-specs [{:kind :array :sym 'values} {:kind :array :sym 'out}
                    {:kind :scalar :type :int :value-fn (constantly n)}]})

(deftest private-allocation-failures-retain-indeterminate-debt
  (doseq [kind [:temporary :owned-view] ordinal [1 2]]
    (let [sess (session) failure (ex-info "Native create outcome unknown" {:kind kind})
          created (atom 0) freed (atom [])
          acquire (fn [n dtype]
                    (let [i (swap! created inc)]
                      (when (= i ordinal) (throw failure))
                      {:id [:private i] :dtype dtype :n-elements n :byte-size (* 4 n)}))
          resolver (fn [_ name]
                     (case name
                       "register-kernel!" (fn [& _])
                       "bind-kernel-call" (fn [& _] (throw (ex-info "Binding reached after expected allocation failure" {})))
                       "make-buffer" acquire
                       "slice-buffer" (fn [_ _ bytes dtype] (acquire (quot bytes 4) dtype))
                       "free-buffer!" #(swap! freed conj %)
                       (throw (ex-info "Unexpected native contact" {:name name}))))
          bindings (if (= kind :owned-view)
                     {'values (gpu/buffer-view sess :values {:dtype :float :shape [512]})
                      'out (gpu/buffer-view sess :out {:dtype :float :shape [512]})}
                     '{values :values out :out})]
      (with-redefs-fn
        {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
        (fn []
          (is (identical? failure
                          (try (gpu/bind-step! sess (step (if (= kind :owned-view) 512 1025))
                                              [] bindings)
                               (catch Throwable e e))) (str kind " " ordinal))
          (is (= ordinal @created))
          (is (nil? (get-in @sess [:prepared :scan])))
          (is (= 1 (count (:prepared @sess))))
          (let [[key entry] (first (:prepared @sess))]
            (is (:failed-construction? entry))
            (is (seq (cleanup/pending (::cleanup/owner entry))))
            (is (identical? failure
                            (try (gpu/release-prepared! sess key) (catch Throwable e e)))))
          (is (= :prepared-buffer-retained
                 (try (gpu/free-buffer! sess :values)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (empty? @freed))
          (is (identical? failure
                          (try (gpu/close-session! sess) (catch Throwable e e))))
          (is (= #{:values :out} (set (keys (:buffers @sess)))))
          (is (empty? @freed))
          (is (= ordinal @created)))))))

(deftest successful-multinode-step-releases-kernels-before-private-storage
  (let [sess (session) releases (atom []) created (atom 0) bound (atom 0)
        resolver (fn [_ name]
                   (case name
                     "register-kernel!" (fn [& _])
                     "make-buffer" (fn [n dtype]
                                     {:id (swap! created inc) :dtype dtype
                                      :n-elements n :byte-size (* 4 n)})
                     "bind-kernel-call" (fn [call _]
                                          (let [i (swap! bound inc)]
                                            {:kernel-call call ::cleanup/owner
                                             (cleanup/owner [{:id :kernel
                                                              :release #(swap! releases conj [:kernel i])}])}))
                     "destroy-prepared!" #(cleanup/release! (::cleanup/owner %))
                     "free-buffer!" #(swap! releases conj [:buffer (:id %)])
                     (throw (ex-info "Unexpected native contact" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) resolver}
      (fn []
        (gpu/bind-step! sess (step 1025) [] '{values :values out :out})
        (is (> @bound 1))
        (is (pos? @created))
        (gpu/release-prepared! sess :scan)
        (gpu/release-prepared! sess :scan)
        (is (= @bound (count (filter #(= :kernel (first %)) @releases))))
        (is (= @created (count (filter #(= :buffer (first %)) @releases))))
        (is (= (concat (repeat @bound :kernel) (repeat @created :buffer))
               (map first @releases)))
        (is (= #{:values :out} (set (keys (:buffers @sess)))))))))

(deftest partial-child-bind-failure-retains-storage-until-kernel-debt-clears
  (let [sess (session) primary (ex-info "Second kernel bind failed" {})
        secondary (ex-info "First kernel destroy unknown" {})
        binds (atom 0) releases (atom 0) freed (atom [])
        resolver (fn [_ name]
                   (case name
                     "register-kernel!" (fn [& _])
                     "make-buffer" (fn [n dtype]
                                     {:dtype dtype :n-elements n :byte-size (* 4 n)})
                     "bind-kernel-call"
                     (fn [call _]
                       (when (= 2 (swap! binds inc)) (throw primary))
                       {:kernel-call call ::cleanup/owner
                        (cleanup/owner [{:id :kernel
                                         :release #(do (swap! releases inc) (throw secondary))}])})
                     "destroy-prepared!" #(cleanup/release! (::cleanup/owner %))
                     "free-buffer!" #(swap! freed conj %)
                     (throw (ex-info "Unexpected native contact" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) resolver}
      (fn []
        (is (identical? primary
                        (try (gpu/bind-step! sess (step 1025) [] '{values :values out :out})
                             (catch Throwable e e))))
        (is (some #(identical? secondary %) (.getSuppressed primary)))
        (is (= 1 (count (:prepared @sess))))
        (let [[key entry] (first (:prepared @sess))]
          (is (:failed-construction? entry))
          (is (identical? secondary
                          (try (gpu/release-prepared! sess key) (catch Throwable e e)))))
        (is (= 1 @releases))
        (is (empty? @freed))
        (is (identical? secondary
                        (try (gpu/close-session! sess) (catch Throwable e e))))
        (is (= 1 @releases))
        (is (= #{:values :out} (set (keys (:buffers @sess)))))
        (is (empty? @freed))))))
