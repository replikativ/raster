(ns raster.compiler.passes.parallel.staged-scalar-body
  "Ordered floating staged reductions expressed as reusable typed loops, not source templates."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.ir.contract-stages :as stages]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]
            [raster.compiler.passes.parallel.staged-scalar-admission :as admission]))

(def decline! admission/decline!)

(defn analyze!
  "Construct the typed stage plan using the shared scalar lowerer. No target emission,
   KernelBody construction or driver work occurs during frontend capability checking."
  [source & {:keys [scalar-types workgroup-size] :or {scalar-types {} workgroup-size 64}}]
  (let [{:keys [reads attributes stage-list axes n out-type sizes array-types]} (admission/analyze! source :scalar-types scalar-types
                                                    :workgroup-size workgroup-size)
        reserved (atom (set (filter symbol? (tree-seq coll? seq source))))
        fresh (fn [prefix]
                (loop [id (gensym prefix)]
                  (if (contains? @reserved id) (recur (gensym prefix))
                      (do (swap! reserved conj id) id))))
        group (fresh "stage_group") lane (fresh "stage_lane") segment (fresh "stage_segment")
        mask (fresh "stage_mask")
        scope (set (concat (map first axes) [group lane segment]))
        builder (scalar/make-lowerer
                 {:arrays reads :array-types array-types :scalar-types scalar-types
                  :require-source-types? true
                  :index-scope scope :lower-index #(index/lower %1 (into scope %2) decline!)
                  :predicate mask :source-region source :id-prefix (str (fresh "stage_scalar"))
                  :decline! decline!})
        build-stage
        (fn build-stage [depth]
          (let [{:keys [axis extent lift] :as stage} (nth stage-list depth)
                dt (dtype/canon (:dtype stage))
                carry (fresh "stage_carry") result (fresh "stage_result")
                child (when (< depth (dec (count stage-list))) (build-stage (inc depth)))
                expression (if child
                             (walk/postwalk-replace
                              {'inner (:result child)}
                              (stages/substitute-operand-indices lift (stages/stage-index-exprs stage-list)))
                             (:body source))
                term ((:lower builder) expression dt (if child {(:result child) (:dtype child)} {}))
                sum ((:compute builder) :+ dt [carry (:result term)] {})]
            {:result result :dtype dt
             :operations [(body/->ForLoop
                           (body/value axis :int) 0 extent 1
                           [(body/->LoopArg (body/value carry dt)
                                           (body/literal (constant/literal-or-original
                                                          (if (nil? (:init stage)) 0.0 (:init stage))) dt))]
                           (vec (concat (:operations child) (:operations term) (:operations sum)
                                        [(body/->Yield [(:result sum)])]))
                           [(body/value result dt)] {})]}))
        computation (build-stage 0)
        arrays (:array-parameters attributes)
        captures (:capture-parameters attributes)
        parameter (fn [id kind dt elements]
                    (body/->KernelParameter id kind dt [elements] :global
                                            (layout/row-major [elements] dt) kind))
        parameters (vec (concat (map #(parameter % :input (array-types %) (sizes %)) arrays)
                                 [(parameter (:out source) :output out-type n)]
                                 (map #(body/->KernelParameter % :scalar (scalar-types %) [] nil nil :capture)
                                      captures)))
        arguments (vec (concat arrays [(:out source)] captures))
        body-spec {:id [:staged-scalar (:out source)] :parameters parameters
                 :stable-reads (mapv body/stable-read arrays)
                 :indices (vec (concat
                                [(body/->IndexBinding group :group 0) (body/->IndexBinding lane :local 0)
                                 (body/->IndexCompute segment
                                                     (body/expression :add (body/expression :mul group workgroup-size) lane))]
                                (map-indexed
                                 (fn [i [axis extent]]
                                   (body/->IndexCompute axis
                                    (body/expression :mod
                                     (body/expression :floor-div segment
                                      (reduce *' 1 (map second (drop (inc i) (:free-axes source))))) extent)))
                                 (:free-axes source))))
                 :masks [(body/->Mask mask [(body/predicate :lt segment n)])]
                 :operations (conj (:operations computation)
                                   (body/->ScalarStore (:out source) [segment] (:result computation) mask))
                 :launch (launch/spec {:workgroup-size [workgroup-size]
                                       :group-count [(quot (+ n (dec workgroup-size)) workgroup-size)]})
                 :schedule {:strategy :staged-scalar :workgroup-size workgroup-size}}]
    {:source source :body-spec body-spec :arguments arguments
     :sizes sizes :output-elements n :out-type out-type}))

(defn lower
  "Materialize a scheduled KernelBody from the same typed plan used by admission."
  [source & options]
  (let [{:keys [body-spec arguments sizes output-elements out-type]}
        (apply analyze! source options)
        kernel (body/make body-spec)]
    (scheduled/make
     {:source source :body kernel :arguments arguments
      :effects {:kind :staged-contraction :uses (scheduled/derive-uses kernel arguments)}
      :legality {:kind :staged-scalar :storage-elements sizes :output-elements output-elements}
      :numerics {:mode :reassociated :policy :explicit-stage-accumulators
                 :accumulator-dtype out-type :rounding :nearest-even}})))
