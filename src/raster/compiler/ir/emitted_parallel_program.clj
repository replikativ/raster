(ns raster.compiler.ir.emitted-parallel-program
  "Validation for an equation-first emitted ParallelProgram.

   The target backend may replace scheduled operations, but the retained typed algorithms and
   ordered program dataflow remain authoritative. Host-only scalar equations are the only
   equations with an empty emitted operation sequence."
  (:require [clojure.set :as set]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]))

(def ^:private dialect-targets
  {:opencl-parallel :opencl-c
   :cuda-parallel :cuda-c
   :hip-parallel :hip-cpp})

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

(defn- scheduled-equation-view
  "The outer equation contract before target emission replaces its operation sequence."
  [equation]
  (dissoc equation :operations))

(defn- validate-host-prefix-slices!
  "Bind every narrowed emitted body to its exact enclosing host-scalar execution prefix.

   An EmittedParallelEquation can validate a self-contained numerical graph, but a graph-storage
   extent may also use a preceding host scalar.  This outer check prevents an isolated body from
   substituting a different (though locally valid) prefix: its host equations, terminal numerical
   equation, inferred inputs, outputs, and effects must be the exact slice of this program."
  [parallel-program]
  (loop [host-prefix [] remaining (:equations parallel-program)]
    (when-let [equation (first remaining)]
      (if (true? (get-in equation [:attributes :host-only]))
        (recur (conj host-prefix equation) (next remaining))
        (let [operation (first (:operations equation))]
          (when (emitted-equation/emitted-equation? operation)
            (let [body (:body operation)
                  body-equations (:equations body)
                  actual-prefix (vec (butlast body-equations))
                  terminal-body (last body-equations)
                  expected-inputs (program/infer-inputs body-equations)
                  expected-effects (reduce set/union #{} (map :effects body-equations))]
              (when-not (and (= host-prefix actual-prefix)
                             (= (scheduled-equation-view equation)
                                (scheduled-equation-view terminal-body))
                             (= expected-inputs (:inputs body))
                             (= (:results equation) (:outputs body))
                             (= expected-effects (:effects body)))
                (throw (ex-info "emitted numerical body differs from its enclosing host-scalar prefix slice"
                                {:reason :emitted-parallel-program-host-prefix
                                 :outer-equation (:id equation)
                                 :expected-prefix (mapv :id host-prefix)
                                 :actual-prefix (mapv :id actual-prefix)
                                 :expected-inputs expected-inputs :actual-inputs (:inputs body)
                                 :ir :emitted-parallel-program})))))
          (recur host-prefix (next remaining)))))))

(defn validate!
  "Validate a fully emitted equation-first program without depending on a target backend."
  [parallel-program]
  (when-not (contains? dialect-targets (:dialect parallel-program))
    (throw (ex-info "emitted parallel program requires a supported C-family dialect"
                    {:reason :emitted-parallel-program-dialect
                     :dialect (:dialect parallel-program)
                     :supported (set (keys dialect-targets))
                     :ir :emitted-parallel-program})))
  (let [parallel-program
        (program/validate! parallel-program emitted-operation? emitted-boundary?)
        expected-target (get dialect-targets (:dialect parallel-program))
        _ (validate-host-prefix-slices! parallel-program)
        artifacts
        (vec
         (for [equation (:equations parallel-program)
               operation (:operations equation)
               node (:nodes (:graph operation))]
           (artifact/validate! (:operation node))))
        mismatches (filterv #(not= expected-target (:target %)) artifacts)]
    (when (seq mismatches)
      (throw (ex-info "emitted program dialect disagrees with a contained kernel target"
                      {:reason :emitted-parallel-program-target
                       :dialect (:dialect parallel-program)
                       :expected-target expected-target
                       :artifact-targets (mapv :target artifacts)
                       :ir :emitted-parallel-program})))
    parallel-program))
