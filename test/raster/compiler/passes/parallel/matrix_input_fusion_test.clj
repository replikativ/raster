(ns raster.compiler.passes.parallel.matrix-input-fusion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.layout-stage :as layout]
            [raster.compiler.ir.matrix-stage :as matrix]
            [raster.compiler.passes.parallel.matrix-input-fusion :as fusion]))

(defn- stage-graph []
  (let [cast (layout/make {:id [:test :cast] :operation :cast :input 'A :output 'A16
                           :input-shape [416] :output-shape [416]
                           :input-dtype :float :output-dtype :half
                           :policy {:rounding :nearest-even :overflow :ieee}})
        matrix (matrix/make
                {:id [:test :matrix] :lhs 'A16 :rhs 'B :result 'C
                 :dimensions [13 32 32] :result-shape [13 32]
                 :reduction {:kind :full :range [0 32]}
                 :schedule {:kind :matrix-instruction-tiling
                            :tile {:block-m 16 :block-n 32 :sg-m 8 :sg-n 16 :block-k 32
                                   :num-stages 1
                                   :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}}})]
    (graph/make
     {:inputs [(graph/buffer 'A :float 416 :device :input)
               (graph/buffer 'B :half 1024 :device :input)]
      :outputs [(graph/buffer 'C :float 416 :device :output)]
      :temporaries [(graph/buffer 'A16 :half 416 :device :temporary)] :scalars []
      :nodes [(graph/->ScheduledKernel [:test :cast] cast
                                      [(graph/->ValueUse 'A :read) (graph/->ValueUse 'A16 :write)] #{} [])
              (graph/->ScheduledKernel [:test :matrix] matrix
                                      [(graph/->ValueUse 'A16 :read) (graph/->ValueUse 'B :read)
                                       (graph/->ValueUse 'C :write)] #{} [[:test :cast]])]
      :abi [(abi/slot 'A :input :float) (abi/slot 'B :input :half) (abi/slot 'C :output :float)]
      :arguments '[A B C] :effects {:reads ['A 'B] :writes ['C]}})))

(defn- fuse [g] (fusion/fuse-lhs-cast g [:test :cast] [:test :matrix]))

(deftest private-cast-becomes-a-typed-load-region
  (let [original (stage-graph) candidate (fuse original)
        stage (get-in candidate [:nodes 0 :operation])
        emitted (gemm/emit-scheduled-stage-graph candidate {:target-dialect :opencl-intel})
        artifact (get-in emitted [:nodes 0 :operation])]
    (is (some? candidate))
    (is (= (graph/boundary-contract original) (graph/boundary-contract candidate)))
    (is (= [] (:temporaries candidate)))
    (is (= 1 (count (:nodes candidate))))
    (is (= [] (get-in candidate [:nodes 0 :dependencies])))
    (is (= 'A (:lhs stage)))
    (is (= :float (get-in stage [:input-value-regions 'A :accumulator-dtype])))
    (is (= :binding-admission-required (get-in candidate [:attributes :input-fusion :selection])))
    (is (identical? (get-in original [:nodes 1 :operation])
                    (get-in candidate [:attributes :input-fusion :consumer])))
    (is (= :no-write-alias (get-in artifact [:abi 0 :aliasing])))
    (is (re-find #"convert_half_rte" (:source artifact)))
    (is (identical? stage (get-in artifact [:attributes :scheduled-kernel-body :source])))))

(deftest changed-policy-shape-storage-or-consumer-declines
  (let [g (stage-graph)]
    (doseq [bad [(assoc-in g [:nodes 0 :operation :policy :rounding] :toward-zero)
                 (assoc-in g [:nodes 0 :operation :policy :nan-policy] :canonicalize)
                 (-> g (assoc-in [:nodes 0 :operation :input-shape] [448])
                     (assoc-in [:nodes 0 :operation :output-shape] [448]))
                 (assoc-in g [:temporaries 0 :elements] 448)
                 (assoc-in g [:inputs 0 :elements] 448)
                 (assoc-in g [:inputs 0 :elements] nil)
                 (assoc-in g [:nodes 1 :operation :rhs] 'A16)
                 (assoc-in g [:nodes 1 :operation :epilogue] {:operands [{:sym 'A16}]})
                 (assoc-in g [:nodes 1 :operation :batching] {:extent 2 :lhs true :rhs true})
                 (update g :nodes conj
                         (graph/->ScheduledKernel [:test :observer] :observer
                                                  [(graph/->ValueUse 'A16 :read)] #{} [[:test :cast]]))]]
      (is (nil? (fuse bad))))))

(deftest explicit-production-candidate-retains-weight-conversion
  (let [spec {:id :fused-test :a 'A :b 'B :c 'C :m 16 :n 32 :k 32 :variant :nn
              :fill-workgroups 16
              :tile (get-in (stage-graph) [:nodes 1 :operation :schedule :tile])}
        candidate (gemm/emit-matrix-input-fusion-alternative spec)
        ordinary (gemm/emit-matrix-alternatives spec)]
    (is (= :xmx-direct-lhs-tile-cast (get-in candidate [:attributes :strategy])))
    (is (= [:convert-b :contract] (mapv (comp last :id) (:nodes candidate))))
    (is (= [:b16] (mapv (comp last :id) (:temporaries candidate))))
    (is (= :xmx-direct (get-in ordinary [:selector :default])))
    (is (contains? (set (map executable/strategy (:alternatives ordinary)))
                   :xmx-direct-lhs-tile-cast))
    (doseq [override [{:variant :tn} {:variant :tt} {:target-dialect :cuda}
                     {:target-dialect :hip}]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (gemm/emit-matrix-input-fusion-alternative (merge spec override)))))))

(deftest normal-enumeration-retains-checked-input-fusion-without-changing-selection
  (let [spec {:id :enumeration-test :a 'A :b 'B :c 'C :m 16 :n 32 :k 32
              :fill-workgroups 16
              :tile (get-in (stage-graph) [:nodes 1 :operation :schedule :tile])}]
    (doseq [variant [:nn :nt :tn :tt]]
      (let [result (gemm/emit-matrix-alternatives (assoc spec :variant variant))
            strategies (set (map executable/strategy (:alternatives result)))]
        (is (= (contains? #{:nn :nt} variant)
               (contains? strategies :xmx-direct-lhs-tile-cast)))
        (is (= :xmx-direct (get-in result [:selector :default])))))
    (with-redefs [fusion/fuse-lhs-cast (constantly nil)]
      (is (not (some #(= :xmx-direct-lhs-tile-cast (executable/strategy %))
                     (:alternatives (gemm/emit-matrix-alternatives (assoc spec :variant :nn))))))
      (is (= :matrix-input-fusion-ineligible
             (try (gemm/emit-matrix-input-fusion-alternative (assoc spec :variant :nn))
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (let [failure (ex-info "broken fusion pass" {})]
      (with-redefs [fusion/fuse-lhs-cast (fn [& _] (throw failure))]
        (is (identical? failure
                        (try (gemm/emit-matrix-alternatives (assoc spec :variant :nn))
                             (catch clojure.lang.ExceptionInfo e e))))))))

(deftest fused-leaf-requires-disjoint-physical-input-and-output
  (let [spec {:id :alias-test :a 'A :b 'B :c 'C :m 16 :n 32 :k 32 :variant :nn
              :fill-workgroups 16
              :tile (get-in (stage-graph) [:nodes 1 :operation :schedule :tile])}
        fused (gemm/emit-matrix-input-fusion-alternative spec)
        ordinary (first (:alternatives (gemm/emit-matrix-alternatives spec)))
        calls (fn [g overlap?]
                (let [buffers (into {} (for [b (concat (:inputs g) (:outputs g) (:temporaries g))]
                                         [(:id b) {:id (:id b) :alignment 64 :dtype (:dtype b)}]))
                      buffers (cond-> buffers overlap? (assoc 'C (get buffers 'A)))]
                  (mapv (fn [{a :operation}]
                          (call/make a (mapv (fn [slot value]
                                               (if (= :scalar (:kind slot))
                                                 {:type (:dtype slot) :value (launch/resolve-expression {} value)}
                                                 (get buffers value)))
                                             (:abi a) (:arguments a)))) (:nodes g))))]
    (is (= [] (graph-call/external-alias-violations
               ordinary {'A :shared-ac 'B :weights 'C :shared-ac} =)))
    (is (= [:kernel-graph-writable-alias]
           (mapv :reason (graph-call/external-alias-violations
                          fused {'A :shared-ac 'B :weights 'C :shared-ac} =))))
    (is (= [] (graph-call/external-alias-violations
               fused {'A :activation 'B :weights 'C :result} =)))
    (let [choice (dispatch/make
                  {:id "matrix-storage-admission" :alternatives [ordinary fused]
                   :default-strategy (executable/strategy ordinary)
                   :selector {:kind :fixed-strategy :strategy (executable/strategy fused)}})
          admit (fn [buffers override]
                  (dispatch/admit-alternative
                   choice (mapv buffers (:arguments ordinary)) override
                   #(graph-call/binding-alias-violations % buffers =)))]
      (is (= ordinary (:executable (admit {'A :same 'B :weights 'C :same} :auto))))
      (is (= fused (:executable (admit {'A :input 'B :weights 'C :output} :auto))))
      (is (= :kernel-dispatch-inapplicable
             (try (admit {'A :same 'B :weights 'C :same} (executable/strategy fused))
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (= 3 (count (calls ordinary true))) "global conversion snapshots A before C is written")
    (is (= 2 (count (calls fused false))))
    (is (= :kernel-abi-no-write-alias
           (try (calls fused true) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
