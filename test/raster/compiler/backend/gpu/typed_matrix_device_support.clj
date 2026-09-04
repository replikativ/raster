(ns raster.compiler.backend.gpu.typed-matrix-device-support
  "Test support for exercising matrix schedules from ordinary TypedSOAC contractions.

   Performance tests may isolate the target matrix artifact after the compiler has proved and
   scheduled the contraction.  They still bind it through the backend-neutral KernelExecutable
   API; no backend runtime is allowed to reconstruct a GEMM ABI, launch, or kernel cache."
  (:require [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.passes.parallel.contract-route :as contract-route]
            [raster.compiler.pipeline :as pipeline]))

(def dense-source
  '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                    (* (clojure.core/aget A (+ (* i k) l))
                                       (clojure.core/aget B (+ (* l n) j))))]
         step))

(def batched-source
  '(let* [step (raster.par/contract
                C [[b batch] [i m] [j n]] [[l k]]
                (* (clojure.core/aget A (+ (* (+ (* b m) i) k) l))
                   (clojure.core/aget B (+ (* (+ (* b k) l) n) j))))]
         step))

(def shared-column-batched-source
  '(let* [step (raster.par/contract
                C [[b batch] [i m] [j n]] [[l k]]
                (* (clojure.core/aget A (+ (* (+ (* b m) i) k) l))
                   (clojure.core/aget B (+ (* l n) j))))]
         step))

(defn route
  "Compile `source` through TypedSOAC and return its target-aware contraction dispatch."
  [device-id source array-types scalar-types & {:keys [tile precision split-factors]
                                                :or {precision :mixed-f16-f32}}]
  (let [{:keys [form]}
        (pipeline/schedule-parallel-form
         source {:target-device device-id
                 :dtype :float
                 :array-types array-types
                 :scalar-types scalar-types})
        equation (first (:equations form))]
    (apply contract-route/route-typed-contraction-dispatch
           (:algorithm equation) (first (:operations equation))
           (cond-> [:dtype :float
                    :desc (hardware/descriptor-for device-id)
                    :precision precision]
             tile (conj :tile tile)
             split-factors (conj :split-factors split-factors)))))

(defn dense-dispatch
  [device-id & {:keys [tile split-factors]}]
  (route device-id dense-source
         {'A :float 'B :float 'C :float}
         {'m :int 'n :int 'k :int}
         :tile tile :split-factors split-factors))

(defn batched-dispatch
  [device-id & {:keys [shared-column?]
                :or {shared-column? false}}]
  (route device-id
         (if shared-column? shared-column-batched-source batched-source)
         {'A :float 'B :float 'C :float}
         {'batch :int 'm :int 'n :int 'k :int}))

(defn epilogue-dispatch
  "Compile C=(A·B+bias)*scale as a contraction with a typed result ScalarRegion."
  [device-id & {:keys [tile]}]
  (let [transform {:acc 'acc
                   :expr '(raster.numeric/*
                           (raster.numeric/+ acc (clojure.core/aget bias j)) scale)
                   :operands [{:sym 'bias :dtype :float
                               :map {:groups [[['j 'n]]]}}]
                   :scalars [{:sym 'scale :dtype :float}]
                   :dtype :float}
        contract (apply list
                        (concat
                         '(raster.par/contract C [[i m] [j n]] [[l k]]
                                               (* (clojure.core/aget A (+ (* i k) l))
                                                  (clojure.core/aget B (+ (* l n) j))))
                         [:epilogue transform]))]
    (route device-id (list 'let* ['step contract] 'step)
           {'A :float 'B :float 'C :float 'bias :float}
           {'m :int 'n :int 'k :int 'scale :float}
           :tile tile)))

(defn matrix-artifact
  "Return the one matrix-instruction artifact contributed by `strategy`.

   Conversion and transpose artifacts have no :tile.  A split graph also has a portable final
   combine, so selecting by the certified matrix tile is stable across all graph topologies."
  [scheduled strategy]
  (let [artifacts (->> (dispatch/alternative scheduled strategy)
                       executable/artifacts
                       (filter #(get-in % [:attributes :tile]))
                       vec)]
    (when-not (= 1 (count artifacts))
      (throw (ex-info "matrix schedule must contain exactly one tiled matrix artifact"
                      {:strategy strategy
                       :artifacts (mapv #(select-keys % [:kernel-name :attributes]) artifacts)})))
    (first artifacts)))

(defn input-array
  [n seed]
  (let [result (float-array n)
        random (java.util.Random. (long seed))]
    (dotimes [index n]
      (aset result index (float (* 0.05 (.nextGaussian random)))))
    result))

(defn half-array
  ^shorts [^floats values]
  (let [result (short-array (alength values))]
    (dotimes [index (alength values)]
      (aset result index (Float/floatToFloat16 (aget values index))))
    result))

(defn f16
  ^double [value]
  (double (Float/float16ToFloat (Float/floatToFloat16 (float value)))))

(defn reference
  [^floats a ^floats b m n k]
  (let [result (float-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset result (+ (* i n) j)
              (float
               (loop [l 0 sum 0.0]
                 (if (< l k)
                   (recur (inc l)
                          (+ sum (* (f16 (aget a (+ (* i k) l)))
                                    (f16 (aget b (+ (* l n) j))))))
                   sum))))))
    result))

(defn relative-l1
  [^floats actual ^floats expected]
  (loop [index 0 difference 0.0 scale 0.0]
    (if (< index (alength actual))
      (recur (inc index)
             (+ difference
                (Math/abs (- (double (aget actual index))
                             (double (aget expected index)))))
             (+ scale (Math/abs (double (aget expected index)))))
      (/ difference (max scale 1.0e-30)))))
