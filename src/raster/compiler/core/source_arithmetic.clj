(ns raster.compiler.core.source-arithmetic
  "Source core-inline arithmetic decisions over actual JVM operand carriers.
   These are emission plans, not type inference or proofs of operand completion.
   In particular :ref does not imply a present, numeric, or nonthrowing value.
   Do not apply these recipes to arbitrary generic calls or typed implementations.")

(defn comparison-carrier
  "Select the common comparison carrier from independently known operand carriers.
   Unlike arithmetic promotion, mixed Float/integer comparisons widen to Double.
   Unknown/Object carriers retain the JVM emitter's Double conversion path; this
   decision is not evidence that those conversions complete normally."
  [left right]
  (cond
    (= [:int :int] [left right]) :int
    (and (#{:int :long} left) (#{:int :long} right)) :long
    (= [:float :float] [left right]) :float
    :else :double))

(defn binary-operator?
  "The core-inline arithmetic vocabulary owned by this source planner."
  [operator]
  (contains? #{"+" "-" "*" "/"} operator))

(defn loop-carrier-join
  "The JVM loop-slot join over actual initializer and recurrence carriers.
   Unknown/reference carries do not become primitive from a declaration. This
   plans representation, not operand completion or source numeric equivalence."
  [initializer recurrence]
  (let [numeric? #{:int :long :float :double}
        rank {:int 0 :long 1 :float 2 :double 3}]
    (cond
      (or (nil? recurrence) (= initializer recurrence)) initializer
      (not (and (numeric? initializer) (numeric? recurrence))) initializer
      (or (= initializer :double) (= recurrence :double)) :double
      (or (= [initializer recurrence] [:long :float])
          (= [initializer recurrence] [:float :long])) :double
      (> (rank recurrence) (rank initializer)) recurrence
      :else initializer)))

(defn loop-initializer-plan
  "Plan the existing contextual Float-literal checkpoint before the slot join.
   `input` is the actually emitted initializer carrier; `recurrence` must come
   from the loop's recurrence plan. A retained Float binder can contextualize a
   bare Double literal only when the recurrence is Float. Explicit expressions,
   reference initializers and wider recurrences are never narrowed from a tag.
   This does not prove a loop's normal-return carrier or reassociation legality."
  [binder initializer input recurrence]
  (let [contextual? (and (= input :double) (= recurrence :float)
                         (= 'float (:raster.type/tag (meta binder)))
                         (instance? Double initializer))
        completed (if contextual? :float input)]
    {:input-carrier input :completed-carrier completed
     :slot-carrier (loop-carrier-join completed recurrence)
     :contextual-float-literal? (boolean contextual?)}))

(defn numbers-method-name
  "Numbers overload spelling for an already selected core arithmetic operation.
   This does not select an overload, operand carrier or source definition."
  [operator]
  (case operator "+" "add" "-" "minus" "*" "multiply" "/" "divide"
        (throw (ex-info "Unsupported source arithmetic invocation"
                        {:reason :unsupported-source-arithmetic-invocation :operator operator}))))

(defn binary-plan
  "Plan one step of the source inline left fold for +, -, *, or /.
   Inputs are actual stack carriers, never logical dtype annotations. Callers
   retain operand evaluation order and handle unary/zero-argument source calls
   separately. A :numbers plan can still throw during numeric conversion, even
   when its selected overload uses IEEE division."
  [operator left right]
  (when-not (and (binary-operator? operator)
                 (every? #{:int :long :float :double :ref :bool} [left right]))
    (throw (ex-info "Unsupported source arithmetic invocation"
                    {:reason :unsupported-source-arithmetic-invocation
                     :operator operator :operand-carriers [left right]})))
  (if (or (contains? #{:ref :bool} left) (contains? #{:ref :bool} right))
    (let [normalize #(case % :int :long :float :double :bool :ref %)
          inputs [(normalize left) (normalize right)]
          result (if (some #{:double} inputs) :double :ref)]
      {:kind :numbers :input-carriers inputs :result-carrier result
       :method-name (numbers-method-name operator)})
    (let [target (if (= operator "/")
                   (if (= [:float :float] [left right]) :float :double)
                   (case left
                     :int (case right :int :int :long :long :float :float :double)
                     :long (case right (:int :long) :long :double)
                     :float (if (= right :float) :float :double)
                     :double :double))]
      {:kind :primitive :input-carriers [target target] :result-carrier target})))
