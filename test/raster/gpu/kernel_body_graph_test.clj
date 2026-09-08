(ns raster.gpu.kernel-body-graph-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.staged-contraction-fixtures :as fixtures]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.passes.parallel.staged-contraction-body :as staged]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as probe]))

(defn- candidate []
  (target/emit-static-dense-graph
   "checked_packed" (staged/lower (fixtures/packed-facts 3 5 3 32)) :opencl-portable))

(defn- session [graph short-id]
  (let [specs (concat (:inputs graph) (:outputs graph))
        buffers (into {} (for [{:keys [id dtype elements]} specs
                               :let [n (if (= id short-id) (dec elements) elements)]]
                           [id {:id id :dtype dtype :n-elements n
                                :byte-size (* n (dtype/bytes-of dtype))}]))]
    (atom {:device-id :ocl:0 :session-id :body-graph-test
           :buffers buffers
           :allocations (into {} (for [[id buffer] buffers]
                                   [id (bview/allocation
                                        {:id id :byte-size (:byte-size buffer)
                                         :memory-space :device :device :ocl:0
                                         :coherence :explicit-transfer :ownership :owned})]))
           :kernel-graphs {} :events {} :closed? false})))

(defn- bind! [boundary sess graph bindings]
  (case boundary
    :session (gpu/bind-kernel-graph! sess :checked graph bindings {})
    :descriptor
    (gpu/bind-step!
     sess {:convention :executable :phase :checked :kernel-name "checked_packed"
           :artifact graph
           :argument-specs (mapv (fn [slot sym] {:kind (:kind slot) :sym sym})
                                 (:abi graph) (:arguments graph))}
     [] bindings)))

(defn- mocked-runtime [calls]
  (fn [_ name]
    (case name
      "register-kernel!" (fn [& _] (swap! calls conj :register))
      "bind-kernel-call" (fn [call] (swap! calls conj call) {:call call})
      "record-graph!" (fn [& _] (swap! calls conj :record) {})
      "make-buffer" (fn [& _] (throw (ex-info "unexpected allocation" {})))
      (throw (ex-info "unexpected runtime request" {:name name})))))

(deftest generated-body-capacities-are-enforced-before-driver-work
  (let [graph (candidate)
        bindings (zipmap (:arguments graph) (:arguments graph))
        calls (atom [])]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) (mocked-runtime calls)}
      (fn []
        (doseq [boundary [:session :descriptor]
                id (:arguments graph)]
          (testing (str boundary " undersized " id)
            (let [failure (try (bind! boundary (session graph id) graph bindings) nil
                               (catch clojure.lang.ExceptionInfo e (ex-data e)))]
              (is (= id (:graph-buffer failure)))
              (is (and (integer? (:elements failure))
                       (= (dec (:elements failure)) (:view-elements failure)))))
            (is (empty? @calls) "no register, bind, allocation or recording")))))))

(deftest generated-body-alias-contracts-survive-both-binders
  (let [graph (candidate)
        bindings (zipmap (:arguments graph) (:arguments graph))
        calls (atom [])]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) (mocked-runtime calls)}
      (fn []
        (doseq [boundary [:session :descriptor]]
          (testing (str boundary)
            ;; db and out both contain 15 Floats. A capacity-only check cannot catch this.
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"overlapping writable|overlaps a writable"
                         (bind! boundary (session graph nil) graph (assoc bindings 'out 'db))))
            (is (empty? @calls))
            (bind! boundary (session graph nil) graph bindings)
            (is (some map? @calls) "valid storage reaches a bound leaf")
            (is (= 15 (->> @calls (filter map?) first :arguments last :value))
                "the private launch scalar is supplied by the graph")
            (reset! calls [])))))))

(deftest generated-body-graph-replays-on-device
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "generated dense body graph replay")
    (let [graph (candidate)]
      (gpu/with-gpu-session [sess :ocl:0]
        (gpu/alloc! sess {:a [:byte 288 (byte-array (repeat 288 2))]
                          :b [:byte 480 (byte-array (repeat 480 -3))]
                          :da [:float 9 (float-array (repeat 9 0.5))]
                          :db [:float 15 (float-array (repeat 15 0.25))]
                          :out [:float 15 nil]})
        (let [handle (gpu/bind-kernel-graph!
                      sess :checked graph {'a :a 'b :b 'da :da 'db :db 'out :out} {})]
          (try
            (doseq [_ (range 2)]
              (gpu/upload! sess :out (float-array (repeat 15 -555.0)))
              (let [event (gpu/submit-kernel-graph! sess handle)]
                (try (gpu/await-event! sess event)
                     (is (= (vec (repeat 15 -72.0)) (vec (gpu/download sess :out))))
                     (finally (gpu/release-event! sess event)))))
            (finally (gpu/release-kernel-graph! sess handle))))))))
