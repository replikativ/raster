(ns raster.ode.multilevel-heat-test
  "A whole-domain refinement oracle for numerical multilevel evolution. This does not model a
   partial fine patch or claim coarse/fine interface flux correction."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.equation-first :as equation]
            [raster.gpu.device-probe :as device-probe]
            [raster.gpu.link :as link]
            [raster.ode.multilevel :as multilevel]
            [raster.ode.pde :as pde]))

(defn- total [^doubles cells]
  (reduce + 0.0 (seq cells)))

(defn- initial-cells [nx ny]
  (double-array
   (for [i (range nx) j (range ny)]
     (+ 2.0
        (* 0.4 (Math/sin (* 2.0 Math/PI (/ (+ i 0.5) nx))))
        (* 0.2 (Math/cos (* 2.0 Math/PI (/ (+ j 0.5) ny))))))))

(defn- evolve-refined
  "Advance the full-domain fine representation four stable diffusion substeps per coarse dt.
   Restriction is a numerical observation at each coarse checkpoint, not a solver-specific IR op."
  [^doubles fine nx ny alpha dt steps]
  (let [fine-nx (* 2 nx) fine-ny (* 2 ny)
        scratch (double-array (alength fine))
        coarse (double-array (* nx ny))]
    (loop [step 0 current fine next scratch]
      (if (= step steps)
        (do (multilevel/restrict-average-2d! coarse current nx ny)
            {:coarse coarse :fine current})
        (let [[current next]
              (loop [substep 0 current current next next]
                (if (= substep 4)
                  [current next]
                  (do (pde/periodic-heat-step-2d!
                       next current fine-nx fine-ny alpha (/ dt 4.0)
                       (double (* fine-nx fine-nx)) (double (* fine-ny fine-ny)))
                      (recur (inc substep) next current))))]
          (multilevel/restrict-average-2d! coarse current nx ny)
          (recur (inc step) current next))))))

(deftest periodic-heat-refinement-conserves-and-restores
  (let [nx 8 ny 8 alpha 0.2 dt 0.001
        coarse0 (initial-cells nx ny)
        fine0 (double-array (* 4 nx ny))
        _ (multilevel/prolong-constant-2d! fine0 coarse0 nx ny)
        uninterrupted (evolve-refined (aclone fine0) nx ny alpha dt 6)
        midpoint (evolve-refined (aclone fine0) nx ny alpha dt 3)
        ;; A durable restore is represented here by detached bytes in fresh host arrays. The
        ;; numerical path must not rely on process-local scratch identity or previous calls.
        restored (evolve-refined (aclone ^doubles (:fine midpoint)) nx ny alpha dt 3)
        initial-mass (/ (total coarse0) (* nx ny))
        coarse-mass (/ (total ^doubles (:coarse uninterrupted)) (* nx ny))
        fine-mass (/ (total ^doubles (:fine uninterrupted)) (* 4 nx ny))]
    (is (= (vec (:fine uninterrupted)) (vec (:fine restored))))
    (is (= (vec (:coarse uninterrupted)) (vec (:coarse restored))))
    (is (< (Math/abs (- initial-mass fine-mass)) 1.0e-12))
    (is (< (Math/abs (- coarse-mass fine-mass)) 1.0e-12))
    (is (> (Math/abs (- (aget coarse0 0)
                        (aget ^doubles (:coarse uninterrupted) 0))) 1.0e-4))))

(deftest periodic-heat-eigenmode-converges-under-refinement
  (let [alpha 0.2 end-time 0.02
        error (fn [n]
                (let [initial (double-array
                               (for [i (range n) j (range n)]
                                 (* (Math/sin (* 2.0 Math/PI (/ (+ i 0.5) n)))
                                    (Math/sin (* 2.0 Math/PI (/ (+ j 0.5) n))))))
                      scratch (double-array (* n n))
                      reference (aclone initial)
                      steps (long (Math/ceil (/ end-time (/ (* 0.05 (/ 1.0 (* n n))) alpha))))
                      dt (/ end-time steps)
                      decay (Math/exp (* -8.0 Math/PI Math/PI alpha end-time))]
                  (loop [step 0 current initial next scratch]
                    (if (= step steps)
                      (Math/sqrt
                       (/ (reduce + 0.0
                                  (map (fn [actual starting]
                                         (let [difference (- actual (* decay starting))]
                                           (* difference difference)))
                                       current reference))
                          (* n n)))
                      (do (pde/periodic-heat-step-2d!
                           next current n n alpha dt (double (* n n)) (double (* n n)))
                          (recur (inc step) next current))))))
        errors (mapv error [8 16 32])]
    (is (every? pos? errors))
    (is (every? #(< 2.0 %) (map / (butlast errors) (rest errors)))
        (str "expected spatial-temporal refinement to converge, errors=" errors))))

(deftest periodic-heat-has-a-direct-typed-link-plan
  (let [nx 4 ny 5 input (initial-cells nx ny) output (double-array (* nx ny))
        compiled (equation/compile #'pde/periodic-heat-step-2d!
                                   {:target :ocl:0 :dtype :double})
        plan (equation/lower compiled
                             [output input nx ny 0.2 0.001 (double (* nx nx)) (double (* ny ny))])]
    (is (pos? (count (get-in plan [:instances 0 :call :steps]))))
    (is (= :none (get-in compiled [:stats :fallback])))
    (when @device-probe/opencl-available?
      (testing "generated OpenCL execution agrees with the JVM numerical operator"
        (let [expected (double-array (* nx ny))
              _ (pde/periodic-heat-step-2d!
                 expected input nx ny 0.2 0.001 (double (* nx nx)) (double (* ny ny)))
              result (link/evaluate! plan)
              actual (first (vals (:outputs result)))]
          (is (every? true?
                      (map (fn [expected-cell actual-cell]
                             (< (Math/abs (- expected-cell actual-cell)) 1.0e-11))
                           expected actual))))))))
