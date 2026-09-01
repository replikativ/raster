(ns raster.compiler.ir.emitted-parallel-program
  "Validation for an equation-first emitted ParallelProgram.

   The target backend may replace scheduled operations, but the retained typed algorithms and
   ordered program dataflow remain authoritative. Host-only scalar equations are the only
   equations with an empty emitted operation sequence."
  (:require [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]))

(defn emitted-operation?
  [operation]
  (or (emitted-loop/emitted-loop? operation)
      (emitted-equation/emitted-equation? operation)))

(defn emitted-boundary?
  [equation algorithm]
  (cond
    (control/loop-program? algorithm)
    (and (= 1 (count (:operations equation)))
         (let [operation (first (:operations equation))]
           (and (emitted-loop/emitted-loop? operation)
                (= algorithm (:algorithm (:schedule (emitted-loop/validate! operation)))))))

    (soac/program-form? algorithm)
    (if (true? (get-in equation [:attributes :host-only]))
      (and (empty? (:operations equation))
           (every? #(= 'scalar (soac/operation-kind %)) (soac/equations algorithm)))
      (and (= 1 (count (:operations equation)))
           (let [operation (first (:operations equation))]
             (and (emitted-equation/emitted-equation? operation)
                  (= algorithm (:algorithm (emitted-equation/validate! operation)))))))

    :else false))

(defn validate!
  "Validate a fully emitted equation-first program without depending on a target backend."
  [parallel-program]
  (when-not (= :opencl-parallel (:dialect parallel-program))
    (throw (ex-info "emitted parallel program requires :opencl-parallel"
                    {:reason :emitted-parallel-program-dialect
                     :dialect (:dialect parallel-program)
                     :ir :emitted-parallel-program})))
  (program/validate! parallel-program emitted-operation? emitted-boundary?))
