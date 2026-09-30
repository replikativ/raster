(ns raster.ode.multilevel-heat-device-test
  "Resident full-domain refinement and actual mapped-byte continuation. This is not partial-patch
   AMR, a network benchmark, or a production persistence-provider implementation."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-state :as state]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.ode.multilevel :as multilevel]
            [raster.ode.multilevel-heat-test :as host]
            [raster.ode.pde :as pde]
            [raster.runtime.numerical-content :as content]
            [raster.test-support.numerical-checkpoint :as checkpoint])
  (:import [java.nio.file Files]))

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
        prepared (compiled/lower #'refined-heat-checkpoint-step!
                                 [coarse fine scratch nx ny alpha dt]
                                 {:compiler :equation-first :target target :dtype :double
                                  :inline? true :donate '[fine] :outputs '[coarse]})
        plan (compiled/plan prepared)
        ids (into {} (for [[id node] (:nodes plan)
                          [field array] [[:coarse coarse] [:fine fine]]
                          :when (identical? array (:source node))]
                      [field id]))]
    (is (= #{:coarse :fine} (set (keys ids))))
    (is (= 2 (count (:outputs plan))))
    (is (= 0 (get-in plan [:attributes :driver-allocations])))
    {:plan plan :ids ids}))

(defn- snapshot [captured nx ny]
  (state/certify
   (state/manifest
    {:id :heat/refined-step-3 :parents [] :logical-coordinate {:step 3 :time 0.003}
     :fields
     (mapv (fn [[field shape level]]
             (state/field
              {:id field :value (av/tensor {:dtype :double :shape shape}) :chunk-shape shape
               :coordinate-space {:hierarchy :heat/full-domain :level level :patch field
                                  :axes [{:name :x :centering :cell} {:name :y :centering :cell}]}
               :chunks [(state/chunk
                         {:id [field 0] :offsets [0 0] :shape shape
                          :logical-byte-length (get-in captured [field :bytes])
                          :stored-byte-length (get-in captured [field :bytes])
                          :content (get-in captured [field :content])
                          :storage {:format :raw-array :byte-order :little-endian}})]}))
           [[:coarse [nx ny] 0] [:fine [(* 2 nx) (* 2 ny)] 1]])
     :numerical-contract {:mode :ieee-fp64 :determinism :reproducible-order
                          :compatibility-id "periodic-heat-full-refinement-f64-v1"}
     :provenance {:program-fingerprint "refined-heat-checkpoint-step-v1"}})))

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
            captured
            (with-open [executable (link/instantiate! (:plan midpoint))]
              (dotimes [_ 3] (link/run! executable))
              (into {} (for [[field id] (:ids midpoint)]
                         [field (checkpoint/capture-f64!
                                 (:session executable) (link/node-view executable id)
                                 (paths field) (if (= field :fine) (* 4 nx ny) (* nx ny)))])))
            certificate (snapshot captured nx ny)
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
