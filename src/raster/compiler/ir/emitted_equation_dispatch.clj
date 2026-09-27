(ns raster.compiler.ir.emitted-equation-dispatch
  "A numerical-policy-checked choice among independently certified equation emissions.

   This first vertical is deliberately limited to SegmentedWeightedReduction. Each candidate
   retains its own scheduled body and artifact projection; a shared ABI is only a binding
   contract, never a proof that different floating-point schedules are interchangeable."
  (:require [raster.compiler.ir.emitted-parallel-equation :as equation]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defrecord EmittedEquationDispatch [alternatives dispatch numerical-policy])

(defn emitted-equation-dispatch?
  [value]
  (and value
       (= "raster.compiler.ir.emitted_equation_dispatch.EmittedEquationDispatch"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :emitted-equation-dispatch))))

(defn- numerical-mode
  [candidate]
  (get-in (last (get-in candidate [:body :equations]))
          [:operations 0 :numerics :mode]))

(defn validate!
  "Verify every schedule independently and require explicit permission for its numerical mode.

   The reference/default must retain exact evaluation order. A caller may admit reassociation
   only by naming it in :permitted-modes; the runtime selector cannot enlarge that permission."
  [value]
  (when-not (emitted-equation-dispatch? value)
    (fail! :equation-dispatch-type "expected an EmittedEquationDispatch"
           {:actual (type value)}))
  (let [{:keys [alternatives numerical-policy] selection :dispatch} value]
    (when-not (and (vector? alternatives) (seq alternatives)
                   (every? equation/emitted-equation? alternatives))
      (fail! :equation-dispatch-alternatives
             "equation dispatch requires certified emitted equations" {}))
    (doseq [candidate alternatives] (equation/validate! candidate))
    (when-not (every? #(swr/plan? (:algorithm %)) alternatives)
      (fail! :equation-dispatch-algorithm
             "equation dispatch currently supports segmented weighted reductions" {}))
    (let [first-candidate (first alternatives)
          allowed (:permitted-modes numerical-policy)
          modes (mapv numerical-mode alternatives)
          selection (dispatch/validate! selection)]
      (when-not (and (map? numerical-policy)
                     (set? allowed)
                     (contains? allowed :exact)
                     (every? #{:exact :reassociated} allowed))
        (fail! :equation-dispatch-numerical-policy
               "dispatch requires an explicit exact/reassociated numerical permission set"
               {:policy numerical-policy}))
      (when-not (= (mapv :graph alternatives) (:alternatives selection))
        (fail! :equation-dispatch-executables
               "dispatch alternatives differ from independently certified emitted graphs" {}))
      (when-not (every? #(= (:algorithm first-candidate) (:algorithm %))
                        (rest alternatives))
        (fail! :equation-dispatch-semantics
               "dispatch alternatives must refine the same semantic algorithm" {}))
      (when-not (apply = (map equation/physical-results alternatives))
        (fail! :equation-dispatch-storage
               "dispatch alternatives must retain the same physical result mapping" {}))
      (when-not (apply = (map equation/complete-write-domains alternatives))
        (fail! :equation-dispatch-complete-write
               "every dispatch alternative must prove the same complete-write domain" {}))
      (when-not (every? allowed modes)
        (fail! :equation-dispatch-numerics
               "candidate numerical mode is not authorized by the equation policy"
               {:modes modes :permitted-modes allowed}))
      (let [default-index (.indexOf (mapv :graph alternatives)
                                   (dispatch/default-alternative selection))]
        (when-not (= :exact (nth modes default-index))
          (fail! :equation-dispatch-default-numerics
                 "the safe fallback must retain exact evaluation order"
                 {:default-strategy (:default-strategy selection)}))))
    value))

(defn make
  [alternatives selection numerical-policy]
  (validate! (->EmittedEquationDispatch alternatives selection numerical-policy)))
