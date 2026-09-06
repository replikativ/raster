(ns raster.compiler.passes.parallel.scalar-region-body-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.patterns :as patterns]
            [raster.compiler.passes.parallel.segred-body :as segred]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- lowerer []
  (let [decline! (fn [rule message data] (throw (ex-info message (assoc data :rule rule))))]
    (scalar/make-lowerer
     {:array-types {'x :float} :arrays #{'x} :scalar-types {'i :int}
      :lower-index (fn [expression scope] (index/lower expression (conj scope 'i) decline!))
      :decline! decline!})))

(defn- mixed-region [lowerer]
  ((:lower-region lowerer)
   '{:bindings [candidate (aget x i) better (> candidate old-value)]
     :results [(if better candidate old-value) (if better i old-index)]}
   [:float :int] {'candidate :float 'better :predicate}
   {'old-value :float 'old-index :int}))

(deftest nested-arithmetic-retains-precision-before-consumer-promotion
  (doseq [tag-key [:raster.type/tag :tag]
          expression ['(+ a b) '(inc a) '(dec a) '(- a) '(+ a b a)]]
    (let [inner (with-meta expression {tag-key 'float})
          result ((:lower (lowerer)) (list '+ inner 'wide) :double
                  {'a :float 'b :float 'wide :double})
          operations (:operations result)
          arithmetic (remove #(= :cast (get-in % [:expression :op])) operations)
          casts (filter #(= :cast (get-in % [:expression :op])) operations)]
      (is (= :double (:type result)))
      (is (every? #(= :float (get-in % [:result :type])) (butlast arithmetic))
          (str "retained inner arithmetic: " expression))
      (is (= :double (get-in (last arithmetic) [:result :type])))
      (is (= 1 (count casts)))
      (is (= [(get-in (last (butlast arithmetic)) [:result :id])]
             (get-in (first casts) [:expression :arguments])))
      (is (= {:rounding :exact :overflow :exact}
             (get-in (first casts) [:expression :options])))))
  ;; A retained region result cannot silently change its declared binding type either.
  (let [expression (with-meta '(+ a b) {:raster.type/tag 'float})]
    (is (= :scalar-region-dtype
           (try ((:lower-region (lowerer)) {:bindings [] :results [expression]}
                 [:double] {} {'a :float 'b :float})
                nil
                (catch clojure.lang.ExceptionInfo e (:rule (ex-data e))))))))

(deftest retained-precision-composes-with-branches-and-predicates
  (let [inner (with-meta '(if (> a b) (+ a b) (- a b))
                {:raster.type/tag 'float :tag 'double})
        result ((:lower (lowerer)) (list '+ inner 'wide) :double
                {'a :float 'b :float 'wide :double})
        [comparison branch conversion outer] (:operations result)]
    (is (= :predicate (get-in comparison [:result :type])))
    (is (= :float (get-in branch [:results 0 :type])))
    (is (= :cast (get-in conversion [:expression :op])))
    (is (= :double (get-in conversion [:result :type])))
    (is (= :double (get-in outer [:result :type]))))
  (let [result ((:lower (lowerer)) '(+ (+ a b) wide) :double
                {'a :float 'b :float 'wide :double})]
    (is (every? #(= :double (get-in % [:result :type])) (:operations result))
        "Absent retained metadata still uses the owner's contextual dtype")))

(deftest floating-precision-change-does-not-retarget-integral-loop-carries
  ;; Reduced from the public Q4 projection fixture. This freezes existing behavior, not a
  ;; proof that widened source carries and the narrower target intrinsic are equivalent.
  (let [update (with-meta '(raster.par/dp4a a b acc) {:raster.type/tag 'int})
        expression (list 'loop '[j 0 acc 0]
                         (list 'if '(< j n) (list 'recur '(inc j) update) 'acc))
        result ((:lower (lowerer)) expression :long {'a :long 'b :long 'n :long})
        loop (last (:operations result))]
    (is (= :long (:type result)))
    (is (= :long (get-in loop [:results 0 :type])))
    (is (= :long (get-in loop [:operations 0 :result :type])))))

(deftest shared-conversion-policy-keeps-narrowing-an-explicit-owner-decision
  (let [types [:byte :int :long :half :float :double]
        exact [:exact :exact]
        rounded [:nearest-even :exact]
        ieee [:nearest-even :ieee]
        wrap [:exact :wrap]
        ;; Independent complete matrix: rows are source types, columns target types.
        expected [[exact exact exact ieee rounded exact]
                  [wrap exact exact ieee rounded exact]
                  [wrap wrap exact ieee rounded rounded]
                  [nil nil nil exact exact exact]
                  [nil nil nil ieee exact exact]
                  [nil nil nil ieee ieee exact]]]
    (is (= (set types) (set (keys dtype/dtype-info))))
    (doseq [[row source] (map-indexed vector types)
            [column target] (map-indexed vector types)]
      (let [policy (get-in expected [row column])]
        (is (= policy (conversion/policy source target :wrap)) (str source " → " target))
        (is (= (when-not (= wrap policy) policy) (conversion/policy source target))
            (str "checked owner: " source " → " target))))
    (is (= exact (conversion/policy :i32 :f64)))
    (is (= ieee (conversion/policy :f64 :f16)))
    (is (thrown? clojure.lang.ExceptionInfo (conversion/policy :missing :float)))
    (is (thrown? clojure.lang.ExceptionInfo (conversion/policy :int :float :unchecked)))))

(deftest reduction-element-conversions-retain-their-stricter-admission
  (let [expression (with-meta '(+ (aget x i) scale) {:raster.type/tag 'double})
        options {:index 'i :coordinate 'i :dtype :double
                 :arrays #{'x} :array-types {'x :float}
                 :scalars #{'scale} :scalar-types {'scale :int}}
        result (segred/lower-element-operations expression options)
        casts (filter #(= :cast (get-in % [:expression :op])) (:operations result))]
    (is (= [:double :double] (mapv #(get-in % [:result :type]) casts)))
    (is (every? #(= {:rounding :exact :overflow :exact}
                   (get-in % [:expression :options])) casts))
    ;; The source-level checked-cast gate remains independent from implicit promotion policy.
    (doseq [expression ['(int scale) '(long scale) '(float (int scale))]]
      (is (= :checked-scalar-cast
             (try (segred/lower-element-operations expression options) nil
                  (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e)))))))))

(deftest scalar-casts-use-the-shared-descriptor-vocabulary
  (doseq [[head target source overflow]
          [['byte :byte :long :wrap] ['int :int :long :wrap]
           ['long :long :int :exact] ['float :float :double :ieee]
           ['double :double :float :exact]]
          qualified? [false true]]
    (let [head (if qualified? (symbol "clojure.core" (name head)) head)
          result ((:lower (lowerer)) (list head 'value) target {'value source})
          expression (:expression (last (:operations result)))]
      (is (= target (:type result)))
      (is (= :cast (:op expression)))
      (is (= overflow (get-in expression [:options :overflow])))))
  (let [result ((:lower (lowerer)) '(double (int value)) :double {'value :long})]
    (is (= [:int :double] (mapv #(get-in % [:result :type]) (:operations result))))
    (is (= [:wrap :exact] (mapv #(get-in % [:expression :options :overflow])
                              (:operations result))))))

(deftest unary-subtraction-retains-floating-sign-and-integral-overflow
  (doseq [type [:float :double :int :long]]
    (let [lower (:lower-region (lowerer))
          lowered (lower '{:bindings [] :results [(- a)]} [type] {} {'a type})
          expression (get-in lowered [:operations 0 :expression])
          floating? (contains? #{:float :double} type)]
      (is (= (if floating? :neg :-) (:op expression)))
      (is (= (if floating? 1 2) (count (:arguments expression))))
      (is (= (when-not floating? :trap) (get-in expression [:options :overflow])))
      (let [kernel (body/make
                    {:id :unary-minus
                     :parameters [(body/->KernelParameter 'a :scalar type [] nil nil :parameter)
                                  (body/->KernelParameter 'out :output type [1] :global
                                                         (layout/row-major [1] type) :result)]
                     :operations (conj (:operations lowered)
                                       (body/->ScalarStore 'out [0] (first (:results lowered)) nil))
                     :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                     :provenance {:dialect :test} :attributes {}})]
        (doseq [target [:opencl-portable :cuda :hip]]
          (if (and (not floating?) (= :opencl-portable target))
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no portable trapping"
                                 (emit/emit-scalar-kernel "unary_minus" kernel
                                                          {:target-dialect target})))
            (is (string? (emit/emit-scalar-kernel "unary_minus" kernel
                                                {:target-dialect target})))))))))

(deftest integer-prefix-negation-cannot-bypass-the-overflow-contract
  (doseq [type [:byte :int :long]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (body/make
                  {:id :invalid-integer-prefix
                   :parameters [(body/->KernelParameter 'a :scalar type [] nil nil :parameter)]
                   :operations [(body/->ScalarCompute (body/value 'negative type)
                                                      (body/scalar-expression :neg type ['a]))]
                   :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                   :provenance {:dialect :test} :attributes {}})))))

(deftest ordered-loop-origin-is-retained-without-widening-soac-recognition
  (doseq [origin [0 1 3 Long/MAX_VALUE]]
    (let [form (list 'loop ['r origin 'acc (body/literal 7.0 :float)]
                     '(if (< r n) (recur (inc r) (+ acc (float r))) acc))
          result ((:lower-region (lowerer)) {:bindings [] :results [form]}
                  [:float] {} {'n :long})
          loop (last (:operations result))]
      (is (= origin (:index-init (patterns/match-ordered-reduce-loop form))))
      (is (= (zero? origin) (some? (patterns/match-reduce-loop form))))
      (is (= (body/index-cast origin :long :exact) (:lower loop)))
      (is (= :ordered (get-in loop [:attributes :association])))))
  (doseq [origin [-1 0.5 'start 9223372036854775808N]]
    (let [form (list 'loop ['r origin 'acc 0.0]
                     '(if (< r n) (recur (inc r) (+ acc r)) acc))]
      (is (nil? (patterns/match-ordered-reduce-loop form))))))

(deftest ordered-loop-admission-does-not-drop-effects-or-swap-recur-slots
  (doseq [form ['(loop [r 1 acc 0.0]
                  (aset out 0 7.0)
                  (if (< r n) (recur (inc r) (+ acc 1.0)) acc))
                '(loop [r 1 acc 0]
                   (if (< r n) (recur (+ acc 1) (inc r)) acc))
                '(loop [r 1 acc 0.0]
                   (if (< r n) (do (aset out 0 7.0) (recur (inc r) (+ acc 1.0))) acc))
                '(loop [r 2147483648 acc 0.0]
                   (if (< (int r) n) (recur (inc r) (+ acc 1.0)) acc))
                '(loop [r 2147483648 acc 0.0]
                   (if (< r n) (recur (inc (int r)) (+ acc 1.0)) acc))
                '(loop [r 1 r 0.0] (if (< r n) (recur (inc r) r) r))]]
    (is (nil? (patterns/match-ordered-reduce-loop form))))
  (is (= 1 (:index-init
            (patterns/match-ordered-reduce-loop
             '(loop [acc 0.0 r 1] (if (< (long r) n) (recur (+ acc 1.0) (inc r)) acc)))))))

(deftest coupled-results-share-bindings-without-sharing-component-types
  (let [{:keys [operations results types] :as lowered} (mixed-region (lowerer))
        kernel
        (body/make
         {:id :mixed-region
          :parameters [(body/->KernelParameter 'x :input :float [16] :global
                                               (layout/row-major [16] :float) :operand)
                       (body/->KernelParameter 'i :scalar :int [] nil nil :parameter)
                       (body/->KernelParameter 'old-value :scalar :float [] nil nil :parameter)
                       (body/->KernelParameter 'old-index :scalar :int [] nil nil :parameter)
                       (body/->KernelParameter 'values :output :float [1] :global
                                               (layout/row-major [1] :float) :result)
                       (body/->KernelParameter 'indices :output :int [1] :global
                                               (layout/row-major [1] :int) :result)]
          :stable-reads [(body/stable-read 'x)]
          :operations (into operations [(body/->ScalarStore 'values [0] (first results) nil)
                                        (body/->ScalarStore 'indices [0] (second results) nil)])
          :launch (launch/spec {:workgroup-size [1] :group-count [1]})
          :provenance {:dialect :test} :attributes {}})]
    (is (= [:float :int] types))
    (is (= ["ScalarLoad" "ScalarCompute" "IfRegion" "IfRegion"]
           (mapv #(.getSimpleName (class %)) operations)))
    (is (= (:condition (nth operations 2)) (:condition (nth operations 3)))
        "both components consume the same predicate SSA value")
    (is (= results (mapv (comp :id first :results) (drop 2 operations))))
    (is (= [:float :int] (mapv (comp :type first :results) (drop 2 operations))))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (string? (emit/emit-scalar-kernel "mixed_region" kernel {:target-dialect target}))))))

(deftest ordered-bindings-and-fresh-ssa-across-region-instances
  (let [lower (lowerer)
        a (mixed-region lower)
        b (mixed-region lower)
        constants ((:lower-region lower)
                   '{:bindings [a 3 b a] :results [b a]}
                   [:int :int] {'a :int 'b :int} {})]
    (is (empty? (:operations constants)))
    (is (= [(body/literal 3 :int) (body/literal 3 :int)] (:results constants)))
    (is (not-any? (set (:results a)) (:results b)))
    (is (= {:operations [] :results [] :types []}
           ((:lower-region lower) {:bindings [] :results []} [] {} {})))))

(deftest missing-and-inconsistent-type-evidence-is-not-inferred-from-consumers
  (doseq [[region result-types binding-types env rule]
          [['{:bindings [a (aget x i)] :results [a]} [:float] {} {}
            :scalar-region-binding-dtype]
           ['{:bindings [a (aget x i)] :results [a]} [:int] {'a :int} {}
            :scalar-region-dtype]
           ['{:bindings [] :results [a]} [:int] {} {'a :float}
            :scalar-region-dtype]
           ['{:bindings [a 1 a 2] :results [a]} [:int] {'a :int} {}
            :scalar-region-shape]
           ['{:bindings [] :results [1 2]} [:int] {} {} :scalar-region-shape]
           ['{:bindings [a b b 1] :results [a]} [:int] {'a :int 'b :int} {}
            :unbound-scalar]]]
    (try
      ((:lower-region (lowerer)) region result-types binding-types env)
      (is false (str "expected " rule))
      (catch clojure.lang.ExceptionInfo exception
        (is (= rule (:rule (ex-data exception))))))))

(deftest typed-constant-bindings-retain-comparison-and-index-types
  (let [lower (:lower-region (lowerer))
        comparison (lower '{:bindings [threshold 0.5] :results [(> threshold old-value)]}
                          [:predicate] {'threshold :float} {'old-value :float})
        indexed (lower '{:bindings [j 0] :results [(aget x j)]}
                       [:float] {'j :long} {})]
    (is (= (body/literal 0.5 :float)
           (get-in comparison [:operations 0 :expression :arguments 0])))
    (is (= [(body/index-cast 0 :long :exact)]
           (get-in indexed [:operations 0 :coordinates])))
    (doseq [[value type] [[0.5 :float] [256 :byte] [2147483648 :int]]]
      (try
        (lower {:bindings ['j value] :results '[(aget x j)]} [:float] {'j type} {})
        (is false "nonintegral or out-of-range index must decline")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :index-literal (:rule (ex-data exception)))))))))

(deftest region-substitution-respects-nested-lexical-bindings
  (let [lower (:lower-region (lowerer))
        result (lower '{:bindings [a 3]
                        :results [(let* [a 9] a) a]}
                      [:int :int] {'a :int} {})]
    (is (= [(body/literal 9 :int) (body/literal 3 :int)] (:results result)))))

(deftest typed-index-lowering-consumes-local-ssa-facts
  (let [decline! (fn [rule message data] (throw (ex-info message (assoc data :rule rule))))
        lower (:lower-region
               (scalar/make-lowerer
                {:arrays #{'x} :array-types {'x :float} :scalar-types {'i :int}
                 :lower-index (fn [expression scope]
                                (index/lower-typed expression scope {'i :int} :long decline!))
                 :decline! decline!}))]
    (doseq [[init expected-kind] [['(long i) "ScalarCompute"] [0 nil]]]
      (let [result (lower {:bindings ['offset init] :results '[(aget x offset)]}
                          [:float] {'offset :long} {})
            load (peek (:operations result))
            coordinate (first (:coordinates load))]
        (is (= :float (get-in load [:result :type])))
        (if expected-kind
          (do
            (is (= expected-kind (.getSimpleName (class (first (:operations result))))))
            (is (= 'long (:raster.type/tag (meta coordinate)))))
          (is (= (body/index-cast 0 :long :exact) coordinate)))))
    (let [result (lower '{:bindings [offset 0] :results [(aget x (long offset))]}
                        [:float] {'offset :int} {})]
      (is (= (body/index-cast (body/index-cast 0 :int :exact) :long :exact)
             (first (:coordinates (peek (:operations result)))))))
    (doseq [[value type] [[2147483648 :int] [0.5 :float]]]
      (try
        (lower {:bindings ['offset value] :results '[(aget x offset)]}
               [:float] {'offset type} {})
        (is false "malformed typed index constants must decline")
        (catch clojure.lang.ExceptionInfo e
          (is (= :index-literal (:rule (ex-data e)))))))))

(deftest kernel-only-dtypes-do-not-require-a-jvm-tag
  (let [lower (:lower-region
               (scalar/make-lowerer
                {:arrays #{'x} :array-types {'x :half} :scalar-types {}
                 :lower-index (fn [coordinate _] coordinate)
                 :decline! (fn [rule message data]
                             (throw (ex-info message (assoc data :rule rule))))}))
        result (lower '{:bindings [h (aget x 0)] :results [h h]} [:half :half] {'h :half} {})]
    (is (= [:half :half] (:types result)))
    (is (= 1 (count (:operations result))))
    (is (apply = (:results result)))
    (is (= :half (get-in result [:operations 0 :result :type])))))
