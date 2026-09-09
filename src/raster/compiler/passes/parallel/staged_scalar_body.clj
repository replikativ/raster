(ns raster.compiler.passes.parallel.staged-scalar-body
  "Ordered floating staged reductions expressed as reusable typed loops, not source templates."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.ir.contract-stages :as stages]
            [raster.compiler.ir.contraction-closure :as closure]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]
            [raster.compiler.passes.parallel.packed-stage-fragment :as packed-stage]
            [raster.compiler.passes.parallel.staged-scalar-admission :as admission]))

(def decline! admission/decline!)

(defn analyze!
  "Construct the typed stage plan using the shared scalar lowerer. No target emission,
   KernelBody construction or driver work occurs during frontend capability checking."
  [source & {:keys [scalar-types workgroup-size] :or {scalar-types {} workgroup-size 64}}]
  (let [{:keys [reads attributes stage-list axes n out-type sizes array-types packed-plan packed-operands]} (admission/analyze! source :scalar-types scalar-types
                                                    :workgroup-size workgroup-size)
        reserved (atom (set (filter symbol? (tree-seq coll? seq source))))
        fresh (fn [prefix]
                (loop [id (gensym prefix)]
                  (if (contains? @reserved id) (recur (gensym prefix))
                      (do (swap! reserved conj id) id))))
        group (fresh "stage_group") lane (fresh "stage_lane") segment (fresh "stage_segment")
        mask (fresh "stage_mask") packed-index (fresh "stage_word")
        scope (set (concat (map first axes) [group lane segment packed-index]))
        builder (scalar/make-lowerer
                 {:arrays reads :array-types array-types :scalar-types scalar-types
                  :require-source-types? true
                  :index-scope scope :lower-index #(index/lower %1 (into scope %2) decline!)
                  :predicate mask :source-region source :id-prefix (str (fresh "stage_scalar"))
                  :decline! decline!})
        build-stage
        (fn build-stage [depth]
          (if (and packed-plan (= depth (dec (count stage-list))))
            (packed-stage/lower
             {:builder builder :lower-index #(index/lower %1 (into scope %2) decline!)
              :scope scope :operands packed-operands :inner (nth stage-list depth)
              :plan packed-plan :packed-index packed-index
              :carry (fresh "stage_dot") :result (fresh "stage_dot_result")})
            (let [{:keys [axis extent lift] :as stage} (nth stage-list depth)
                dt (dtype/canon (:dtype stage))
                carry (fresh "stage_carry") result (fresh "stage_result")
                child (when (< depth (dec (count stage-list))) (build-stage (inc depth)))
                expression (if child
                             (walk/postwalk-replace
                              {'inner (:result child)}
                              (stages/substitute-operand-indices lift (stages/stage-index-exprs stage-list)))
                             (:body source))
                ;; The stage contract declares its root result type; nested expressions must
                ;; retain their own checked types, never inherit a consumer's conversion.
                ;; Integral storage is not an arithmetic-width declaration for a floating
                ;; fold. Leave untyped roots to strict scalar admission in that case; typed
                ;; loads/casts remain understood, but compound arithmetic must retain its type.
                expression (if (and (seq? expression)
                                    (not (and (nil? child) (dtype/fp-dtype? dt)
                                              (not (dtype/fp-dtype? (:dtype source))))))
                             (vary-meta expression
                                        #(if (or (:raster.type/tag %) (:tag %)) %
                                           (assoc % :raster.type/tag
                                                  (dtype/scalar-tag-for-dtype
                                                   (if (or child (contains? #{:int :long} dt))
                                                     dt (:dtype source))))))
                             expression)
                term ((:cast builder)
                      ((:lower builder) expression dt (if child {(:result child) (:dtype child)} {}))
                      dt expression)
                sum ((:compute builder) :+ dt [carry (:result term)]
                     (if (contains? #{:int :long} dt) {:overflow :no-overflow} {}))]
            {:result result :dtype dt
             :operations [(body/->ForLoop
                           (body/value axis :int) 0 extent 1
                           [(body/->LoopArg (body/value carry dt)
                                           (body/literal (constant/literal-or-original
                                                          (if (nil? (:init stage))
                                                            (if (contains? #{:int :long} dt) 0 0.0)
                                                            (:init stage))) dt))]
                           (vec (concat (:operations child) (:operations term) (:operations sum)
                                        [(body/->Yield [(:result sum)])]))
                           [(body/value result dt)] {})]})))
        computation (build-stage 0)
        store-cast (fn [lowered expression]
                     (when-not (conversion/policy (:type lowered) out-type :reject)
                       (decline! :result-conversion "result storage conversion needs an explicit supported policy"
                                 {:source (:type lowered) :target out-type}))
                     ((:cast builder) lowered out-type expression))
        result-transform
        (when-let [epilogue (:epilogue source)]
          (let [legality (body/scalar-region-legal? epilogue)
                _ (when-not (:ok legality)
                    (decline! :result-transform "staged result transform is not a scalar store region"
                              legality))
                dt (dtype/canon (or (:dtype epilogue) out-type))
                expression (walk/postwalk-replace
                            {(:acc epilogue) (:result computation)}
                            (closure/result-expression source))
                expression (if (and (seq? expression)
                                    (not (or (:tag (meta expression)) (:raster.type/tag (meta expression)))))
                             (vary-meta expression assoc :raster.type/tag (dtype/scalar-tag-for-dtype dt))
                             expression)
                lowered ((:cast builder)
                         ((:lower builder) expression dt {(:result computation) (:dtype computation)})
                         dt expression)]
            (store-cast lowered expression)))
        ;; Storage conversion follows the complete staged computation (and optional epilogue).
        ;; Changing the accumulator dtype here would change intermediate rounding.
        stored-result (or result-transform
                          (store-cast
                           {:result (:result computation) :type (:dtype computation) :operations []}
                           (:body source)))
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
                 :operations (conj (into (:operations computation) (:operations stored-result))
                                   (body/->ScalarStore (:out source) [segment]
                                                        (:result stored-result) mask))
                 :launch (launch/spec {:workgroup-size [workgroup-size]
                                       :group-count [(quot (+ n (dec workgroup-size)) workgroup-size)]})
                 :schedule {:strategy :staged-scalar :workgroup-size workgroup-size}}]
    (try (body/validate-spec! body-spec)
         (catch clojure.lang.ExceptionInfo e
           (decline! :kernel-body-proof "staged body does not satisfy the shared verifier"
                     {:cause (ex-data e) :message (.getMessage e)})))
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
