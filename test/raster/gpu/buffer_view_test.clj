(ns raster.gpu.buffer-view-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.gpu.core :as gpu]))

(defn- allocation [id bytes ownership]
  (bview/allocation {:id id :byte-size bytes :memory-space :shared :device :ze:0
                     :coherence :host-coherent :ownership ownership}))

(defn- mock-session [buffer allocation]
  (atom {:device-id :ze:0 :session-id :session :buffers {:cache buffer}
         :allocations {:cache allocation} :kernel-graphs {} :events {} :closed? false}))

(deftest resident-views-translate-ranges-and-expire-with-the-allocation
  (let [buffer {:dtype :float :n-elements 16 :byte-size 64}
        sess (mock-session buffer (allocation :cache-v1 64 :owned))
        uploaded (atom nil)
        resolver (fn [_ name]
                   (case name
                     "upload-range!" (fn [actual _ spec]
                                       (reset! uploaded [actual spec])
                                       actual)
                     "free-buffer!" (fn [_])
                     (throw (ex-info "unexpected runtime function" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
      (fn []
        (let [middle (gpu/buffer-view sess :cache {:byte-offset 16 :shape [8]})]
          (is (gpu/resident-buffer-view? middle))
          (gpu/upload-range! sess middle (float-array 2)
                             {:src-element 0 :dst-element 1 :elements 2})
          (is (= [buffer {:src-element 0 :dst-element 5 :elements 2}] @uploaded))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds the buffer view"
                                (gpu/upload-range! sess middle (float-array 2)
                                                   {:dst-element 7 :elements 2})))
          (gpu/free-buffer! sess :cache)
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no longer names"
                                (gpu/sub-buffer-view sess middle
                                                     {:dtype :float :shape [1]}))))))))

(deftest external-registrations-are-never-freed-by-raster
  (let [buffer {:dtype :float :n-elements 4 :byte-size 16}
        sess (atom {:device-id :ze:0 :session-id :session :buffers {} :allocations {}
                    :closed? false})
        freed (atom [])
        resolver (fn [_ name]
                   (case name
                     "device-buffer?" map?
                     "free-buffer!" #(swap! freed conj %)
                     (throw (ex-info "unexpected runtime function" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
      (fn []
        (gpu/register-buffer! sess :foreign buffer {:allocation-id :foreign-allocation})
        (is (= :external (get-in @sess [:allocations :foreign :ownership])))
        (gpu/free-buffer! sess :foreign)
        (is (empty? @freed))
        (is (empty? (:buffers @sess)))))))

(deftest physical-aliases-use-ranges-and-scheduled-accesses
  (let [allocation (allocation :shared 64 :owned)
        left (bview/view allocation {:id :left :dtype :float :shape [8]})
        right (bview/view allocation {:id :right :byte-offset 32 :dtype :float :shape [8]})
        overlap (bview/view allocation {:id :overlap :byte-offset 16 :dtype :float :shape [8]})
        use (fn [id access] (kgraph/->ValueUse id access))
        node (fn [id uses deps] (kgraph/->ScheduledKernel id :mock uses deps))
        validate! (ns-resolve 'raster.gpu.core 'validate-physical-aliases!)]
    (testing "disjoint views of one allocation are legal"
      (is (map? (validate! {:nodes [(node :one [(use :a :read) (use :b :write)] [])]}
                           {:a {:view left} :b {:view right}}))))
    (testing "one kernel cannot receive contradictory overlapping writable identities"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"overlapping writable"
                            (validate! {:nodes [(node :one [(use :a :read) (use :b :write)] [])]}
                                       {:a {:view left} :b {:view overlap}}))))
    (testing "an alias-induced cross-kernel hazard needs an edge"
      (let [first-node (node :first [(use :a :write)] [])
            unsafe (node :second [(use :b :read)] [])
            safe (assoc unsafe :dependencies [:first])
            bindings {:a {:view left} :b {:view overlap}}]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"omits a dependency"
                              (validate! {:nodes [first-node unsafe]} bindings)))
        (is (map? (validate! {:nodes [first-node safe]} bindings)))))))
