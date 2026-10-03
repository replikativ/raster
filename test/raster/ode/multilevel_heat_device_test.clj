(ns raster.ode.multilevel-heat-device-test
  "Resident full-domain refinement and actual mapped-byte continuation. This is not partial-patch
   AMR, a network benchmark, or a production persistence-provider implementation."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-state :as state]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.completed-evidence-device-test :as producer]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.ode.multilevel :as multilevel]
            [raster.ode.multilevel-heat-test :as host]
            [raster.ode.pde :as pde]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state :as resident]
            [raster.runtime.resident-state-test :as capture-fixture]
            [raster.test-support.numerical-checkpoint :as checkpoint])
  (:import [java.nio.file Files OpenOption]))

(deftm refined-heat-checkpoint-step!
  [coarse :- (Array double), fine :- (Array double), scratch :- (Array double),
   nx :- Long, ny :- Long, alpha :- Double, dt :- Double] :- Void
  (let [fx (* 2 nx) fy (* 2 ny)
        sub-dt (/ dt 4.0) dx2 (double (* fx fx)) dy2 (double (* fy fy))]
    ;; Four stable fine steps leave the state in the same caller-owned buffer.
    ;; Restriction observes that evolving state; it must not re-prolong the old coarse field.
    (pde/periodic-heat-step-2d! scratch fine fx fy alpha sub-dt dx2 dy2)
    (pde/periodic-heat-step-2d! fine scratch fx fy alpha sub-dt dx2 dy2)
    (pde/periodic-heat-step-2d! scratch fine fx fy alpha sub-dt dx2 dy2)
    (pde/periodic-heat-step-2d! fine scratch fx fy alpha sub-dt dx2 dy2)
    (multilevel/restrict-average-2d! coarse fine nx ny)))

(defn- initial [nx ny]
  (double-array
   (for [i (range nx) j (range ny)]
     (+ 2.0 (* 0.4 (Math/sin (* 2.0 Math/PI (/ (+ i 0.5) nx))))
        (* 0.2 (Math/cos (* 2.0 Math/PI (/ (+ j 0.5) ny))))))))

(defn- prepare [target coarse fine nx ny alpha dt]
  (let [scratch (double-array (repeat (alength ^doubles fine) Double/NaN))
        prepared (with-redefs [build/current-identity #'producer/test-build]
                   (compiled/lower #'refined-heat-checkpoint-step!
                                   [coarse fine scratch nx ny alpha dt]
                                   {:compiler :equation-first :target target :dtype :double
                                    :inline? true :donate '[fine] :outputs '[coarse]}))
        plan (compiled/plan prepared)
        ids (into {} (for [[id node] (:nodes plan)
                           [field array] [[:coarse coarse] [:fine fine]]
                           :when (identical? array (:source node))]
                       [field id]))]
    (is (= #{:coarse :fine} (set (keys ids))))
    (is (= 2 (count (:outputs plan))))
    (is (= 0 (get-in plan [:attributes :driver-allocations])))
    {:prepared prepared :plan plan :ids ids}))

(defn- capture-midpoint! [midpoint paths nx ny dt]
  ;; The existing fixture provides explicitly synthetic packaged build evidence; this is
  ;; real compiler-owned execution provenance, not release-build authentication.
  (with-redefs [build/current-identity #'producer/test-build]
    (let [{:keys [provider blobs events]} (#'capture-fixture/provider (fn [& _]))
          published (atom {})
          c (compiled/instantiate! (:prepared midpoint))]
      (try
        (let [fact (compiled/measure-storage-representation! c :double)
              fields (mapv (fn [[field source grid-shape level]]
                             {:id field :source source :node (get-in midpoint [:ids field])
                              :value (av/tensor {:dtype :double :shape [(reduce * grid-shape)]})
                              :coordinate-space {:hierarchy :heat/full-domain :level level :patch field
                                                 :grid-shape grid-shape :centering :cell}})
                           [[:coarse :outputs [nx ny] 0] [:fine :post-state [(* 2 nx) (* 2 ny)] 1]])
              opts {:fields fields
                    :numerical-contract {:mode :ieee-fp64 :determinism :reproducible-order
                                         :compatibility-id "periodic-heat-full-refinement-f64-v1"}}
              captures
              (loop [step 1 previous nil captured []]
                (if (> step 3)
                  captured
                  (let [result
                        (with-open [receipt (compiled/invoke-with-evidence c {})]
                          (let [result (resident/capture!
                                        receipt {:double fact} provider :local
                                        (assoc opts :id (keyword "heat" (str "refined-step-" step))
                                               :parents (if previous [(get-in previous [:state :manifest :id])] [])
                                               :logical-coordinate {:step step :time (* step dt)}))]
                            (when previous
                              (is (= (get-in previous [:state :manifest :provenance :completed-fingerprint])
                                     (get-in result [:state :manifest :provenance :parent-replay]))))
                            (is (= :published (:publication (content/finalize-state-availability!
                                                             provider (:state result) :durable
                                                             (fn [manifest]
                                                               (is (every? #(contains? @published %) (:parents manifest)))
                                                               (swap! published assoc (:id manifest) manifest)
                                                               :published)))))
                            result))]
                    ;; Close before the next replay; captures contain no output/session ownership.
                    (recur (inc step) result (conj captured result)))))
              certificate (:state (peek captures))]
          (is (= [:heat/refined-step-2] (get-in certificate [:manifest :parents])))
          (is (= #{:heat/refined-step-1 :heat/refined-step-2 :heat/refined-step-3}
                 (set (keys @published))))
          (is (= [:outputs :post-state]
                 (mapv :source (get-in certificate [:manifest :provenance :field-producers]))))
          (is (seq (get-in certificate [:manifest :provenance :bound-schedules])))
          (is (empty? @events))
          ;; Realize the verified provider bytes in actual files. The existing mapped leases
          ;; independently recheck each chunk before upload into the fresh continuation owner.
          (doseq [field (get-in certificate [:manifest :fields])]
            (Files/write (paths (:id field))
                         ^bytes (get @blobs (:content (first (:chunks field))))
                         (make-array OpenOption 0)))
          certificate)
        (finally (compiled/close! c))))))

(defn- near? [expected actual]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) 1.0e-11) expected actual))))

(deftest refined-heat-step-keeps-the-evolving-fine-state
  (let [nx 4 ny 5 coarse (initial nx ny) fine (double-array (* 4 nx ny))
        scratch (double-array (repeat (* 4 nx ny) Double/NaN))
        _ (multilevel/prolong-constant-2d! fine coarse nx ny)
        expected (#'host/evolve-refined (aclone fine) nx ny 0.2 0.001 3)]
    (dotimes [_ 3] (refined-heat-checkpoint-step! coarse fine scratch nx ny 0.2 0.001))
    (is (= (vec (:fine expected)) (vec fine)))
    (is (= (vec (:coarse expected)) (vec coarse)))))

(defn- run-continuation [target]
  (let [nx 4 ny 5 alpha 0.2 dt 0.001
        coarse (initial nx ny) fine (double-array (* 4 nx ny))
        _ (multilevel/prolong-constant-2d! fine coarse nx ny)
        expected (#'host/evolve-refined (aclone fine) nx ny alpha dt 6)
        uninterrupted (prepare target (aclone coarse) (aclone fine) nx ny alpha dt)
        paths (checkpoint/temp-files [:coarse :fine])]
    (try
      (let [baseline
            (with-open [executable (link/instantiate! (:plan uninterrupted))]
              (dotimes [_ 6] (link/run! executable))
              (into {} (for [[field id] (:ids uninterrupted)]
                         [field (vec (link/download executable id))])))
            midpoint (prepare target (aclone coarse) (aclone fine) nx ny alpha dt)
            certificate (capture-midpoint! midpoint paths nx ny dt)
            chunks (into {} (for [field (get-in certificate [:manifest :fields])]
                              [(:id field) (first (:chunks field))]))
            fresh (prepare target (double-array (* nx ny)) (double-array (* 4 nx ny))
                           nx ny alpha dt)]
        (is (= certificate (state/verify! certificate)))
        (is (near? (vec (:fine expected)) (:fine baseline)))
        (is (near? (vec (:coarse expected)) (:coarse baseline)))
        ;; The producer session and writable mappings are closed before either read lease opens.
        (with-open [coarse-lease (checkpoint/open-chunk-lease (paths :coarse) (chunks :coarse))
                    fine-lease (checkpoint/open-chunk-lease (paths :fine) (chunks :fine))]
          (let [restored (reduce (fn [plan [field lease]]
                                   (assoc-in plan [:nodes (get-in fresh [:ids field]) :source]
                                             (content/lease-segment lease)))
                                 (:plan fresh) [[:coarse coarse-lease] [:fine fine-lease]])]
            (with-open [executable (link/instantiate! restored)]
              ;; Synchronous initialization completes before the mmap arenas are released.
              (.close coarse-lease)
              (.close fine-lease)
              (is (and (content/lease-closed? coarse-lease) (content/lease-closed? fine-lease)))
              (dotimes [_ 3] (link/run! executable))
              (doseq [[field id] (:ids fresh)]
                (is (= (baseline field) (vec (link/download executable id)))
                    "the same backend must resume bit-for-bit from the actual checkpoint bytes")))))
        (let [mass (fn [cells] (/ (reduce + 0.0 cells) (count cells)))]
          (is (< (Math/abs (- (mass (vec coarse)) (mass (:fine baseline)))) 1.0e-12))
          (is (< (Math/abs (- (mass (:coarse baseline)) (mass (:fine baseline)))) 1.0e-12))
          (is (> (Math/abs (- (aget coarse 0) (first (:coarse baseline)))) 1.0e-4))))
      (finally (doseq [path (vals paths)] (Files/deleteIfExists path))))))

(deftest level-zero-refined-heat-resumes-from-addressed-bytes
  (if @gp/gpu-available?
    (run-continuation :ze:0)
    (gp/gpu-skip! "resident refined heat mapped-byte evolution")))

(deftest opencl-refined-heat-resumes-from-addressed-bytes
  (if @opencl/opencl-available?
    (run-continuation :ocl:0)
    (opencl/opencl-skip! "resident refined heat mapped-byte evolution")))
