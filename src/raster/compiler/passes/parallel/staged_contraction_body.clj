(ns raster.compiler.passes.parallel.staged-contraction-body
  "A bounded packed staged-contraction schedule, expressed entirely as typed KernelBody.

   Packing is four byte loads plus word operations, not a pointer reinterpretation. This is a
   correctness-first schedule; aligned vector loads and throughput tuning are separate work."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contract-stages :as stages]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.staged-contraction-admission :as admission]))

(def decline! admission/decline!)
(def declined? admission/declined?)

(defn- require! [condition rule data]
  (when-not condition
    (decline! rule "typed staged schedule is not proved" data)))

(defn lower
  "Schedule verified staged contraction facts through shared non-emitting admission."
  [source & {:keys [workgroup-size] :or {workgroup-size 64}}]
  (let [{:keys [free-axes out stage-list outer inner axis-ids operands lifts array-ids legality plan n sizes]} (admission/analyze! source :workgroup-size workgroup-size)]
    (let [reserved (atom (set (filter symbol? (tree-seq coll? seq source))))
          fresh (fn [prefix]
                  (loop [id (gensym prefix)]
                    (if (contains? @reserved id) (recur (gensym prefix))
                        (do (swap! reserved conj id) id))))
          group (fresh "stage_group") lane (fresh "stage_lane") segment (fresh "stage_segment")
          mask (fresh "stage_active") packed-index (fresh "stage_word")
          inner-carry (fresh "stage_dot") inner-result (fresh "stage_dot_result")
          outer-carry (fresh "stage_sum") outer-result (fresh "stage_sum_result")
          scope (set (concat axis-ids [group lane segment packed-index '_nseg]))
          lower-index #(index/lower %1 (into scope %2) decline!)
          builder (scalar/make-lowerer
                   {:arrays (set array-ids)
                    :array-types (merge (zipmap (map :sym operands) (repeat :byte))
                                        (zipmap (map :sym lifts) (repeat :float)))
                    :scalar-types {} :index-scope scope :lower-index lower-index :predicate mask
                    :source-region [source @reserved inner-carry inner-result outer-carry]
                    :id-prefix (str (fresh "stage_scalar")) :decline! decline!
                    :conversion-policy (fn [from to]
                                         (when (contains? #{[:byte :int] [:int :float]} [from to])
                                           (conversion/policy from to :reject)))})
          pack (fn [{:keys [sym map]}]
                 (reduce
                  (fn [word offset]
                    (let [coordinate (walk/postwalk-replace
                                      {(:axis inner) (list 'clojure.core/+ (list 'clojure.core/* packed-index 4) offset)}
                                      (am/index-expr map))
                          loaded ((:cast builder)
                                  ((:load builder) sym [(lower-index coordinate scope)]) :int coordinate)
                          byte-word ((:compute builder) :bit-and :int
                                     [(:result loaded) (body/literal 255 :int)] {})
                          shifted ((:compute builder) :shl :int
                                   [(:result byte-word) (body/literal (* offset 8) :int)] {})
                          joined ((:compute builder) :bit-or :int [(:result word) (:result shifted)] {})]
                      {:operations (vec (concat (:operations word) (:operations loaded)
                                                (:operations byte-word) (:operations shifted) (:operations joined)))
                       :result (:result joined)}))
                  {:operations [] :result (body/literal 0 :int)} (range 4)))
          [a b] (mapv pack operands)
          dot ((:compute builder) :dp4a :int [(:result a) (:result b) inner-carry] {})
          lift-expr (walk/postwalk-replace
                     {'inner inner-result}
                     (stages/substitute-operand-indices (:lift outer) (stages/stage-index-exprs stage-list)))
          lift ((:lower builder) lift-expr :float {inner-result :int})
          add ((:compute builder) :+ :float [outer-carry (:result lift)] {})
          parameter (fn [id kind type count role]
                      (body/->KernelParameter id kind type [count] :global
                                              (layout/row-major [count] type) role))
          parameters (vec (concat (map #(parameter (:sym %) :input :byte (sizes (:sym %)) :operand) operands)
                                  (map #(parameter (:sym %) :input :float (sizes (:sym %)) :lift) lifts)
                                  [(parameter out :output :float (long n) :result)
                                   (body/->KernelParameter '_nseg :scalar :int [] nil nil :bound)]))
          arguments (vec (concat array-ids [out (long n)]))
          kernel (body/make
                  {:id [:staged-packed out]
                   :parameters parameters :stable-reads (mapv body/stable-read array-ids)
                   :indices (vec (concat
                                  [(body/->IndexBinding group :group 0) (body/->IndexBinding lane :local 0)
                                   (body/->IndexCompute segment (body/expression :add (body/expression :mul group workgroup-size) lane))]
                                  (map-indexed
                                   (fn [i [axis extent]]
                                     (body/->IndexCompute axis
                                      (body/expression :mod
                                       (body/expression :floor-div segment (reduce *' 1 (map second (drop (inc i) free-axes)))) extent)))
                                   free-axes)))
                   :masks [(body/->Mask mask [(body/predicate :lt segment (long n))
                                             (body/predicate :lt segment '_nseg)])]
                   :operations
                   [(body/->ForLoop
                     (body/value (:axis outer) :int) 0 (:extent outer) 1
                     [(body/->LoopArg (body/value outer-carry :float) (body/literal 0.0 :float))]
                     (vec (concat
                           [(body/->ForLoop
                             (body/value packed-index :int) 0 (:packed-extent plan) 1
                             [(body/->LoopArg (body/value inner-carry :int) (body/literal 0 :int))]
                             (vec (concat (:operations a) (:operations b) (:operations dot)
                                          [(body/->Yield [(:result dot)])]))
                             [(body/value inner-result :int)] {})]
                           (:operations lift) (:operations add) [(body/->Yield [(:result add)])]))
                     [(body/value outer-result :float)] {})
                    (body/->ScalarStore out [segment] outer-result mask)]
                   :launch (launch/spec {:workgroup-size [workgroup-size]
                                         :group-count [(quot (+ (long n) (dec workgroup-size)) workgroup-size)]})
                   :schedule {:strategy :staged-packed :workgroup-size workgroup-size}
                   :attributes {:kind :staged-packed :packing :byte-loads}})]
      (scheduled/make
       {:source source :body kernel :arguments arguments
        :effects {:kind :staged-contraction :uses (scheduled/derive-uses kernel arguments)}
        :legality {:kind :staged-packed :stage-legality legality :packed-plan plan
                   :storage-elements sizes :output-elements (long n)}
        :numerics {:mode :reassociated :policy :staged-int32-float
                   :accumulator-dtype :float :rounding :nearest-even}}))))

(defn schedule-for-node
  "Refine an exact retained typed contraction graph node through the existing staged body.
   Graph/algorithm agreement is checked before lexical bindings reach the emitted ABI."
  [node graph algorithm scheduled-program]
  (equation-graph/validate-projection! graph algorithm scheduled-program)
  (let [equations (soac/equations algorithm)
        operation (:operation node)]
    (require! (and (= 1 (count equations))
                   (some #{node} (:nodes graph))
                   (instance? raster.compiler.ir.segop.SegContract operation))
              :typed-node {:node node})
    (let [equation (first equations)
          {:keys [facts bindings]} (projection/contraction-binding algorithm equation)
          expected (assoc (segop/->SegContract (second equation) facts (:dtype facts)
                                              (:device-id operation))
                          :bindings bindings)]
      (require! (= expected operation) :typed-operation
                {:expected expected :operation operation})
      (let [lowered (lower facts)
            arguments (mapv #(get bindings % %) (:arguments lowered))
            rebound (scheduled/make
                     (-> (into {} lowered)
                         (assoc :source operation :arguments arguments)
                         (assoc-in [:effects :uses]
                                   (scheduled/derive-uses (:body lowered) arguments))))]
        (scheduled/validate-against-node! rebound node graph)))))
