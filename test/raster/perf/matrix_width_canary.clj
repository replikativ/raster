(ns raster.perf.matrix-width-canary
  "Opt-in, bounded DPAS emission regression probe; never part of the timed unit-test loop.
  This compares a generated matrix leaf with the historical test-only source, not public
  compilation, precision conversion, transfer throughput, or external SOTA libraries."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.backend.gpu.typed-matrix-device-support :as support]
            [raster.compiler.reference.gemm-opencl :as reference]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.gpu.core :as gpu]
            [raster.gpu.dispatch-benchmark :as benchmark]
            [raster.runtime.hardware :as hardware]))

(defn candidate-case
  "Resident arguments and an independent poisoned-output oracle; resets are outside device timing."
  [session [m n k :as shape] expected]
  (let [poison (float-array (* m n) Float/NaN)]
    {:arguments [:a :b :c {:type :int :value m}
                 {:type :int :value n} {:type :int :value k}]
     :before-run! (fn [_] (gpu/upload! session :c poison))
     :validate! (fn [_]
                  (let [actual (gpu/download session :c)
                        error (if (= (alength ^floats actual) (alength ^floats expected))
                                (support/relative-l1 actual expected)
                                Double/POSITIVE_INFINITY)]
                    {:passed? (< error 1.0e-3) :relative-l1 error
                     :oracle-hash (str "f16-cpu-reference-v1-seeds100-101-" shape)}))
     :measurement {:warmup-iterations 5 :budget-ms 20
                   :min-samples 20 :max-samples 80 :cv-threshold 0.10}}))

(defn run!
  "Return raw validated device-event samples in both candidate orders. No tuning cache is written.
  Supply revision and driver identity when recording a baseline; unknown identities stay explicit."
  [device-id shapes & {:keys [revision driver-identity]
                      :or {revision :unrecorded driver-identity :unreported}}]
  (when-not (and (vector? shapes) (seq shapes) (<= (count shapes) 8)
                 (every? (fn [shape]
                           (and (vector? shape) (= 3 (count shape))
                                (every? #(and (integer? %) (<= 1 % 1024)) shape)
                                (zero? (mod (second shape) 16))
                                (zero? (mod (nth shape 2) 16))
                                (<= (apply * shape) 33554432))) shapes))
    (throw (ex-info "matrix width canary requires bounded positive aligned shapes"
                    {:shapes shapes})))
  (let [scheduled (support/dense-dispatch device-id)
        generated (assoc-in (support/matrix-artifact scheduled :xmx-direct)
                            [:attributes :strategy] :typed-body)
        tile (get-in generated [:attributes :tile])
        baseline (artifact/make
                  (merge (select-keys generated [:abi :arguments :launch :effects])
                         {:kernel-name "matrix_width_reference"
                          :source (apply reference/emit-gemm-tiled "matrix_width_reference"
                                         :c-dtype :float :prefetch (:num-stages tile)
                                         (mapcat identity (select-keys tile
                                                                      [:block-m :block-n :sg-m :sg-n :block-k :matrix])))
                          :attributes {:strategy :legacy-reference}
                          :provenance {:kind :test-only-reference}}))
        candidates [generated baseline]
        choice (dispatch/make {:id "matrix-width-probe" :alternatives candidates
                               :default-strategy :typed-body
                               :selector {:kind :fixed-strategy :strategy :typed-body}})]
    {:revision revision :driver-identity driver-identity
     :java-version (System/getProperty "java.version")
     :device (hardware/device device-id)
     :comparison :matrix-leaf-regression-not-sota
     :cases (mapv
     (fn [[m n k :as shape]]
       (let [a (support/input-array (* m k) 100)
             b (support/input-array (* k n) 101)
             expected (support/reference a b m n k)]
         (gpu/with-gpu-session [session device-id]
           (gpu/alloc! session {:a [:half (* m k) (support/half-array a)]
                                :b [:half (* k n) (support/half-array b)]
                                :c [:float (* m n) nil]})
           (let [case (candidate-case session shape expected)
                 results (vec (for [round (range 2)
                                    candidate (if (even? round) candidates (reverse candidates))]
                                (let [result (benchmark/benchmark-candidate!
                                              session choice candidate :fixed (constantly case))]
                                  {:round round :strategy (get-in candidate [:attributes :strategy])
                                   :validation (:validation result)
                                   :measurement (into {} (:measurement result))})))]
             {:shape shape :device device-id :tile tile :results results}))))
     shapes)}))
