(ns raster.ode.amr-subcycle-device-test
  "Ordinary typed coarse/fine cycles and synchronized restart; no mid-cycle restart claim."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-state :as state]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.completed-evidence-device-test :as producer]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.ode.amr-subcycle :as subcycle]
            [raster.ode.amr-subcycle-test :as oracle]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state :as resident]
            [raster.runtime.resident-state-test :as capture-fixture]
            [raster.test-support.numerical-checkpoint :as checkpoint])
  (:import [java.nio.file Files OpenOption]))

(defn- output-node [prepared key]
  (:node (first (filter #(= key (:key %)) (:out-tree prepared)))))

(defn- synchronized-semantics [projection]
  ;; Independent application semantics, not a claim that a manifest proves the PDE.
  (state/restore-semantics
   {:logical-coordinate {:phase :synchronized :completed-coarse-cycles 1 :time 0.001}
    :fields (mapv (fn [[id level n]]
                    {:id id :value (av/tensor {:dtype :double :shape [n]})
                     :coordinate-space {:layout-fingerprint (fingerprint/fingerprint projection)
                                        :level level :centering :cell}})
                  [[:coarse 0 (count (get-in projection [:coarse :cells]))]
                   [:fine 1 (:fine-cell-count projection)]])
    :numerical-contract {:mode :ieee-fp64 :determinism :reproducible-order
                         :compatibility-id "ratio2-subcycled-diffusion-reflux-f64-v1"}}))

(defn- capture-synchronized! [prepared projection paths]
  (let [{:keys [provider blobs events]} (#'capture-fixture/provider (fn [& _]))
        semantics (synchronized-semantics projection)
        c (compiled/instantiate! prepared)]
    (try
      (let [representations (into {} (map (fn [dtype]
                                           [dtype (compiled/measure-storage-representation! c dtype)]))
                                  [:int :double])]
        (with-open [receipt (compiled/invoke-with-evidence c {})]
          (let [certificate
                (:state (resident/capture!
                         receipt representations provider :local
                         (assoc semantics :id [:subcycle :synchronized (random-uuid)] :parents []
                                :fields (mapv (fn [field key]
                                                (assoc field :source :outputs
                                                       :node (output-node prepared key)))
                                              (:fields semantics) [:coarse-out :fine-out]))))]
            (is (identical? certificate
                            (resident/verify-restore!
                             certificate semantics prepared
                             [{:id :coarse :source :outputs :key :coarse-out}
                              {:id :fine :source :outputs :key :fine-out}])))
            (is (= :published
                   (:publication (content/finalize-state-availability!
                                  provider certificate :durable (fn [_] :published)))))
            (doseq [field (get-in certificate [:manifest :fields])]
              (Files/write (paths (:id field))
                           ^bytes (get @blobs (get-in field [:chunks 0 :content]))
                           (make-array OpenOption 0)))
            (is (empty? @events))
            certificate)))
      (finally (compiled/close! c)))))

(defn- run-synchronized-restart! [target]
  (with-redefs [build/current-identity #'producer/test-build]
    (let [projection (subcycle/project (#'oracle/hierarchy)
                                       {:domain-lengths [1.0 1.0] :diffusivity 0.2})
          {:keys [coarse fine]} (#'oracle/initial-state)
          midpoint (#'oracle/reference-cycle coarse fine 0.001 false)
          expected (#'oracle/reference-cycle (:coarse midpoint) (:fine midpoint) 0.001 false)
          prepare (fn [coarse fine]
                    (compiled/lower
                     #'subcycle/diffusion-cycle!
                     (:arguments (subcycle/cycle-inputs projection coarse fine 0.001))
                     {:target target :compiler :equation-first :dtype :double
                      :inline? true :outputs '[coarse-out fine-out]}))
          prepared (prepare (double-array coarse) (double-array fine))
          paths (checkpoint/temp-files [:coarse :fine])
          raw-bits #(mapv (fn [x] (Double/doubleToRawLongBits (double x))) %)]
      (try
        (let [baseline
              (let [c (compiled/instantiate! prepared)]
                (try
                  (let [first-cycle (compiled/invoke-compiled c {})]
                    ;; The second cycle consumes resident outputs directly; the
                    ;; public invocation handoff owns any D2D copies and alias retirement.
                    (compiled/invoke-compiled c {:coarse (:coarse-out first-cycle)
                                                 :fine (:fine-out first-cycle)})
                    (into {} (map (fn [key]
                                    [key (vec (link/download (:executable c)
                                                            (output-node prepared key)))]))
                          [:coarse-out :fine-out]))
                  (finally (compiled/close! c))))
              certificate (capture-synchronized! prepared projection paths)
              chunks (into {} (map (fn [field] [(:id field) (first (:chunks field))]))
                           (get-in certificate [:manifest :fields]))
              fresh (prepare (double-array (repeat 16 -997.0))
                             (double-array (repeat 16 -991.0)))]
          ;; A different phase must fail even when geometry and bytes match.
          (is (thrown? clojure.lang.ExceptionInfo
                       (resident/verify-restore!
                        certificate (assoc-in (synchronized-semantics projection)
                                              [:logical-coordinate :phase] :fine-substep)
                        prepared [{:id :coarse :source :outputs :key :coarse-out}
                                  {:id :fine :source :outputs :key :fine-out}])))
          ;; The producing sessions are closed before opening read-only mappings.
          (with-open [coarse-lease (checkpoint/open-chunk-lease (paths :coarse) (chunks :coarse))
                      fine-lease (checkpoint/open-chunk-lease (paths :fine) (chunks :fine))]
            (let [inputs (into {} (map (juxt :key :node)) (:in-tree fresh))
                  restored (-> (compiled/plan fresh)
                               (assoc-in [:nodes (inputs :coarse) :source]
                                         (content/lease-segment coarse-lease))
                               (assoc-in [:nodes (inputs :fine) :source]
                                         (content/lease-segment fine-lease)))]
              (with-open [executable (link/instantiate! restored)]
                (.close coarse-lease)
                (.close fine-lease)
                (is (and (content/lease-closed? coarse-lease) (content/lease-closed? fine-lease)))
                (link/run! executable)
                (let [actual-coarse (vec (link/download executable (output-node fresh :coarse-out)))
                      actual-fine (vec (link/download executable (output-node fresh :fine-out)))]
                  (is (= (raw-bits (:coarse-out baseline)) (raw-bits actual-coarse)))
                  (is (= (raw-bits (:fine-out baseline)) (raw-bits actual-fine)))
                  (is (every? #(< (Math/abs (double %)) 1.0e-11)
                              (map - (:coarse expected) actual-coarse)))
                  (is (every? #(< (Math/abs (double %)) 1.0e-11)
                              (map - (:fine expected) actual-fine)))
                  (is (< (Math/abs (- (#'oracle/composite-mass coarse fine)
                                      (#'oracle/composite-mass actual-coarse actual-fine))) 1.0e-11))
                  (is (= actual-coarse (#'oracle/average-down actual-coarse actual-fine))))))))
        (finally (doseq [path (vals paths)] (Files/deleteIfExists path)))))))

(defn- run-cycle! [target]
  (with-redefs [build/current-identity #'producer/test-build]
    (let [p (subcycle/project (#'oracle/hierarchy) {:domain-lengths [1.0 1.0] :diffusivity 0.2})
          {:keys [coarse fine]} (#'oracle/initial-state)
          expected (#'oracle/reference-cycle coarse fine 0.001 false)
          inputs (subcycle/cycle-inputs p (double-array coarse) (double-array fine) 0.001)
          prepared (compiled/lower #'subcycle/diffusion-cycle! (:arguments inputs)
                                   {:target target :compiler :equation-first :dtype :double
                                    :inline? true :outputs '[coarse-out fine-out register]})
          c (compiled/instantiate! prepared)]
      (try
        (with-open [receipt (compiled/invoke-with-evidence c {})]
          (let [values (into {} (map (fn [{:keys [key node]}]
                                       [key (vec (link/download (:executable c) node))]))
                             (:out-tree prepared))
                actual-coarse (:coarse-out values) actual-fine (:fine-out values)]
            (is (= [16 16 32] (mapv #(count (get values %)) [:coarse-out :fine-out :register])))
            (is (every? #(< (Math/abs (double %)) 1.0e-11) (map - (:coarse expected) actual-coarse)))
            (is (every? #(< (Math/abs (double %)) 1.0e-11) (map - (:fine expected) actual-fine)))
            (is (< (Math/abs (- (#'oracle/composite-mass coarse fine)
                                (#'oracle/composite-mass actual-coarse actual-fine))) 1.0e-11))
            (is (= actual-coarse (#'oracle/average-down actual-coarse actual-fine)))
            (doseq [[i {:keys [axis plane start]}] (map-indexed vector (get-in p [:coarse :faces]))]
              (is (< (Math/abs (- ((:register expected) axis plane start)
                                  (nth (:register values) i))) 1.0e-11)))
            (is (some #(> (Math/abs (double %)) 1.0e-8) (:register values)))
            (is (= 3 (count (:outputs @receipt))))))
        ;; Reuse the same prepared program and scratch after a nonzero-register cycle.
        ;; An equilibrium input must overwrite all evolving state, not inherit old fluxes.
        (with-open [receipt (compiled/invoke-with-evidence
                             c {:coarse (double-array (repeat 16 3.25))
                                :fine (double-array (repeat 16 3.25))})]
          (let [values (into {} (map (fn [{:keys [key node]}]
                                       [key (vec (link/download (:executable c) node))]))
                             (:out-tree prepared))]
            (is (= (vec (repeat 16 3.25)) (:coarse-out values)))
            (is (= (vec (repeat 16 3.25)) (:fine-out values)))
            (is (every? zero? (:register values)))
            (is (= 3 (count (:outputs @receipt))))))
        (finally (compiled/close! c))))))

(deftest subcycled-diffusion-cycle-on-opencl
  (if @opencl/opencl-fp64-available?
    (run-cycle! :ocl:0)
    (opencl/opencl-skip! "FP64 subcycled diffusion and reflux" :fp64)))

(deftest subcycled-diffusion-cycle-on-level-zero
  (if @ze/gpu-available?
    (run-cycle! :ze:0)
    (ze/gpu-skip! "FP64 subcycled diffusion and reflux")))

(deftest synchronized-subcycle-restart-on-opencl
  (if @opencl/opencl-fp64-available?
    (run-synchronized-restart! :ocl:0)
    (opencl/opencl-skip! "FP64 synchronized subcycle restart" :fp64)))

(deftest synchronized-subcycle-restart-on-level-zero
  (if @ze/gpu-available?
    (run-synchronized-restart! :ze:0)
    (ze/gpu-skip! "FP64 synchronized subcycle restart")))
