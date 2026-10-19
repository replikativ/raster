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

(defn binary-plan
  "Plan one step of the source inline left fold for +, -, *, or /.
   Inputs are actual stack carriers, never logical dtype annotations. Callers
   retain operand evaluation order and handle unary/zero-argument source calls
   separately. A :numbers plan can still throw during numeric conversion, even
   when its selected overload uses IEEE division."
  [operator left right]
  (when-not (and (contains? #{"+" "-" "*" "/"} operator)
                 (every? #{:int :long :float :double :ref :bool} [left right]))
    (throw (ex-info "Unsupported source arithmetic invocation"
                    {:reason :unsupported-source-arithmetic-invocation
                     :operator operator :operand-carriers [left right]})))
  (if (or (contains? #{:ref :bool} left) (contains? #{:ref :bool} right))
    (let [normalize #(case % :int :long :float :double :bool :ref %)
          inputs [(normalize left) (normalize right)]
          result (if (some #{:double} inputs) :double :ref)]
      {:kind :numbers :input-carriers inputs :result-carrier result
       :method-name (case operator "+" "add" "-" "minus" "*" "multiply" "/" "divide")})
    (let [target (if (= operator "/")
                   (if (= [:float :float] [left right]) :float :double)
                   (case left
                     :int (case right :int :int :long :long :float :float :double)
                     :long (case right (:int :long) :long :double)
                     :float (if (= right :float) :float :double)
                     :double :double))]
      {:kind :primitive :input-carriers [target target] :result-carrier target})))
