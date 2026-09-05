(ns raster.compiler.ir.kernel-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]))

(def ^:private matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16})
(def ^:private acc-layout (layout/mma-frag matrix :float))
(def ^:private lhs-layout (layout/dot-operand 0 acc-layout 2 :half))
(def ^:private rhs-layout (layout/dot-operand 1 acc-layout 2 :half))

(defn- minimal-body []
  (body/make
   {:id :matrix
    :parameters [(body/->KernelParameter 'A :input :half [16 16] :global
                                         (layout/row-major [16 16] :half) :lhs)
                 (body/->KernelParameter 'B :input :half [16 16] :global
                                         (layout/row-major [16 16] :half) :rhs)
                 (body/->KernelParameter 'C :output :half [16 16] :global
                                         (layout/row-major [16 16] :half) :result)]
    :indices [(body/->IndexBinding 'group-x :group 0)]
    :masks [(body/->Mask :in-bounds [(body/predicate :lt 'group-x 16)])]
    :fragments [(body/->Fragment :lhs :half [8 16] lhs-layout)
                (body/->Fragment :rhs :half [16 16] rhs-layout)
                (body/->Fragment :acc :float [8 16] acc-layout)]
    :operations [(body/->Guard
                  :in-bounds
                  [(body/->FragmentInit :acc 0.0)
                   (body/->Loop
                    'k 0 16 16
                    [(body/->TileLoad :lhs 'A ['group-x 'k] :in-bounds :cached)
                     (body/->TileLoad :rhs 'B ['k 'group-x] :in-bounds :cached)
                     (body/->MatrixMad :acc :lhs :rhs matrix)]
                    {:unroll true})
                   (body/->TileStore 'C :acc ['group-x 0] :in-bounds nil)])]
    :schedule {:matrix matrix}
    :launch (launch/spec {:workgroup-size [16] :group-count [1]})
    :provenance {:dialect :test}
    :attributes {:kind :matrix-contraction}}))

(deftest a-kernel-body-makes-schedule-decisions-explicit
  (let [kernel (minimal-body)
        guard (first (:operations kernel))
        loop-op (second (:operations guard))]
    (is (body/kernel-body? kernel))
    (is (= :dot-operand (get-in kernel [:fragments 0 :layout :kind])))
    (is (= :mma-frag (get-in kernel [:fragments 2 :layout :kind])))
    (is (= 16 (:step loop-op)))
    (is (= :in-bounds (:mask guard)))
    (is (= matrix (:instruction (nth (:operations loop-op) 2))))))

(deftest kernel-buffer-shapes-retain-symbolic-launch-arithmetic
  (let [extent (launch/ceil-div 'n 256)
        kernel (-> (minimal-body)
                   (assoc-in [:parameters 0 :shape] [extent 16])
                   (assoc-in [:parameters 0 :layout]
                             (layout/row-major [extent 16] :half)))]
    (is (= kernel (body/validate! kernel)))
    (is (= extent (get-in kernel [:parameters 0 :shape 0])))))

(deftest the-verifier-refuses-implicit-or-incompatible-kernel-semantics
  (testing "opaque list arithmetic cannot hide in coordinates"
    (let [kernel (minimal-body)
          bad (assoc-in kernel [:operations 0 :operations 1 :operations 0 :coordinates 0]
                        '(+ group-x 1))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicit index expressions"
                            (body/validate! bad)))))
  (testing "matrix operands must inherit the accumulator distribution"
    (let [kernel (minimal-body)
          wrong-parent (layout/mma-frag matrix :half)
          bad (assoc-in kernel [:fragments 0 :layout]
                        (layout/dot-operand 0 wrong-parent 2 :half))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fragment layouts do not agree"
                            (body/validate! bad)))))
  (testing "all memory guards name declared masks"
    (let [kernel (minimal-body)
          bad (assoc-in kernel [:operations 0 :operations 2 :mask] :missing)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"undeclared mask"
                            (body/validate! bad)))))
  (testing "matrix fragment shapes are fixed by the instruction"
    (let [bad (assoc-in (minimal-body) [:fragments 0 :shape] [8 8])]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fragment shapes"
                            (body/validate! bad)))))
  (testing "tile coordinates match storage rank"
    (let [bad (update-in (minimal-body)
                         [:operations 0 :operations 1 :operations 0 :coordinates]
                         pop)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"buffer rank"
                            (body/validate! bad)))))
  (testing "tile-load dtypes follow storage rather than target inference"
    (let [bad (-> (minimal-body)
                  (assoc-in [:fragments 0 :dtype] :float)
                  (assoc-in [:fragments 0 :layout :dtype] :float))]
      (try
        (body/validate! bad)
        (is false "a tile load accepted a fragment dtype different from its source buffer")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :kernel-body-tile-load-dtype (:reason (ex-data exception))))))))
  (testing "storage shape and dtype agree with their named layout"
    (let [bad-shape (assoc-in (minimal-body) [:parameters 0 :shape] [8 16])
          bad-dtype (assoc-in (minimal-body) [:parameters 0 :dtype] :float)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape disagrees"
                            (body/validate! bad-shape)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dtype disagrees"
                            (body/validate! bad-dtype))))))

(deftest buffer-views-retain-parent-ownership-and-explicit-offsets
  (let [kernel (minimal-body)
        kernel (-> kernel
                   (assoc-in [:parameters 0 :shape] [16 16 16])
                   (assoc-in [:parameters 0 :layout]
                             (layout/row-major [16 16 16] :half))
                   (assoc-in [:launch :group-count] [16]))
        view (body/->BufferView 'A-slab 'A
                                (body/expression :mul 'group-x 16 16)
                                [16 16] (layout/row-major [16 16] :half))
        viewed (-> kernel
                   (assoc :views [view])
                   (assoc-in [:operations 0 :operations 1 :operations 0 :buffer] 'A-slab))]
    (is (= viewed (body/validate! viewed)))
    (testing "a view may only reference declared storage"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"undeclared storage"
                            (body/validate! (assoc viewed :views [(assoc view :buffer 'missing)])))))
    (testing "its offset is checked in the scalar/index scope"
      (let [unknown-offset (assoc view :element-offset
                                  (body/expression :mul 'unknown 256))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"outside its scope"
                              (body/validate!
                               (assoc viewed :views [unknown-offset]))))))
    (testing "the parent extent is tied to the hardware axis launch bound"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"launch-bounded"
                            (body/validate! (assoc-in viewed [:launch :group-count 0] 15)))))))

(deftest scalar-regions-are-checked-against-the-ordered-kernel-abi
  (let [region (body/->ScalarRegion
                ['value 'bias 'scale]
                '(raster.numeric/* (raster.numeric/+ value (aget bias group-x)) scale)
                [{:sym 'bias :map (axis-map/of-axes [['group-x 16]]) :dtype :half}]
                :float)
        kernel (-> (minimal-body)
                   (update :parameters into
                           [(body/->KernelParameter
                             'bias :input :half [16] :global
                             (layout/row-major [16] :half) :epilogue)
                            (body/->KernelParameter 'scale :scalar :float [] nil nil :epilogue)])
                   (assoc-in [:operations 0 :operations 2 :value-region] region))]
    (is (= kernel (body/validate! kernel)))
    (testing "operand identities occupy the ordered prefix after the accumulator"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"ordered prefix"
                            (body/validate!
                             (assoc-in kernel [:operations 0 :operations 2 :value-region
                                               :parameters]
                                       ['value 'scale 'bias])))))
    (testing "every region parameter has exactly one epilogue ABI slot"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"epilogue ABI disagree"
                            (body/validate! (update kernel :parameters pop)))))
    (testing "an operand map and physical ABI shape cannot drift"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"operand and its kernel ABI slot disagree"
                            (body/validate!
                             (assoc-in kernel [:operations 0 :operations 2 :value-region
                                               :operands 0 :map]
                                       (axis-map/of-axes [['group-x 8]]))))))))

(deftest typed-scalar-store-regions-have-closed-ssa-and-a-shared-buffer-abi
  (let [region
        (body/->ScalarSSARegion
         ['value 'A] [{:sym 'A :map (axis-map/of-axes [['i 16] ['j 16]]) :dtype :half}]
         ['i 'j] :half
         [(body/->ScalarLoad (body/value 'loaded :half) 'A ['i 'j] nil nil :cached)
          (body/->ScalarCompute
           (body/value 'sum :half)
           (body/scalar-expression :+ :half ['value 'loaded]))]
         'sum :half)
        kernel (-> (minimal-body)
                   (assoc :stable-reads [(body/stable-read 'A)])
                   (assoc-in [:operations 0 :operations 2 :value-region] region))]
    (is (= kernel (body/validate! kernel)))
    (testing "a shared contraction input does not need a second epilogue ABI slot"
      (is (= ['A 'B 'C] (mapv :id (:parameters kernel)))))
    (testing "uses must follow definitions inside the closed region"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before it is defined"
           (body/validate!
            (assoc-in kernel [:operations 0 :operations 2 :value-region :operations]
                      [(body/->ScalarCompute
                        (body/value 'sum :half)
                        (body/scalar-expression :+ :half ['value 'loaded]))])))))
    (testing "the declared region result dtype is checked against typed SSA"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"wrong dtype"
           (body/validate!
            (assoc-in kernel [:operations 0 :operations 2 :value-region :result-dtype]
                      :float)))))
    (testing "the nested region admits only its small scalar/load vocabulary"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"incomplete"
           (body/validate!
            (update-in kernel [:operations 0 :operations 2 :value-region :operations]
                       conj (body/->ScalarStore 'C ['i 'j] 'sum nil))))))
    (testing "unclaimed epilogue slots cannot drift outside the region boundary"
      (let [bias (body/->KernelParameter 'bias :scalar :float [] nil nil :epilogue)]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"epilogue ABI disagree"
             (body/validate! (update kernel :parameters conj bias))))))))

(defn- scalar-body
  [operations & {:keys [masks stable-reads allocations schedule shared-memory-bytes]
                 :or {masks [] stable-reads [] allocations [] schedule {:subgroup-size 16}
                      shared-memory-bytes 0}}]
  (body/make
   {:id :scalar
    :parameters [(body/->KernelParameter
                  'x :input :float [16] :global
                  (layout/row-major [16] :float) :input)
                 (body/->KernelParameter
                  'y :output :float [1] :global
                  (layout/row-major [1] :float) :result)]
    :indices [(body/->IndexBinding 'group :group 0)
              (body/->IndexBinding 'lane :lane 0)]
    :masks masks
    :stable-reads stable-reads
    :allocations allocations
    :operations operations
    :schedule schedule
    :launch (launch/spec {:workgroup-size [16] :group-count [1]
                          :shared-memory-bytes shared-memory-bytes})
    :provenance {:dialect :test}
    :attributes {:kind :scalar}}))

(defn- load-x
  ([] (load-x (body/value 'x-value :float)))
  ([result]
   (body/->ScalarLoad result 'x ['lane] nil nil :cached)))

(defn- reduce-x
  [input]
  (body/->Collective
   (body/value 'sum :float) :reduce :subgroup 16 input :+ nil
   (body/full-participation) :implementation-defined))

(deftest typed-scalar-memory-control-and-collectives-share-one-body
  (let [predicate (body/->ScalarCompute
                   (body/value 'negative? :predicate)
                   (body/scalar-expression :lt :predicate
                                           ['x-value (body/literal 0.0 :float)]))
        choose (body/->IfRegion
                'negative?
                [(body/->ScalarCompute
                  (body/value 'negated :float)
                  (body/scalar-expression :- :float
                                          [(body/literal 0.0 :float) 'x-value]))
                 (body/->Yield ['negated])]
                [(body/->Yield ['x-value])]
                [(body/value 'magnitude :float)])
        loop-op (body/->ForLoop
                 (body/value 'iteration :int) 0 4 1
                 [(body/->LoopArg (body/value 'acc :float) 'magnitude)]
                 [(body/->ScalarCompute
                   (body/value 'next-acc :float)
                   (body/scalar-expression :+ :float
                                           ['acc (body/literal 1.0 :float)]))
                  (body/->Yield ['next-acc])]
                 [(body/value 'loop-result :float)]
                 {:unroll true})
        kernel (scalar-body
                [(load-x) predicate choose loop-op
                 (reduce-x 'loop-result)
                 (body/->ScalarStore 'y ['group] 'sum nil)])]
    (is (body/kernel-body? kernel))
    (is (= :predicate (get-in kernel [:operations 1 :result :type])))
    (is (= :full (get-in kernel [:operations 4 :participation :kind])))
    (is (launch/launch-spec? (:launch kernel)))))

(deftest scalar-ssa-and-conversion-policies-fail-loudly
  (testing "SSA values cannot be used before definition"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"before it is defined"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'result :float)
            (body/scalar-expression :+ :float
                                    ['missing (body/literal 1.0 :float)]))]))))
  (testing "result identities are globally unique"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not globally unique"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'x-value :float)
            (body/scalar-expression :+ :float
                                    ['x-value (body/literal 1.0 :float)]))]))))
  (testing "comparisons produce predicates"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"comparison intrinsic must produce a predicate"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/scalar-expression :lt :float
                                    ['x-value (body/literal 0.0 :float)]))]))))
  (testing "casts cannot leave rounding and overflow behavior implicit"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"rounding and overflow"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/scalar-expression :cast :int ['x-value]))]))))
  (testing "a fully specified cast is legal"
    (is (body/kernel-body?
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'converted :int)
            (body/cast-expression 'x-value :int :toward-zero :saturate))]))))
  (testing "integral arithmetic may carry an explicit wrapping contract"
    (is (body/kernel-body?
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'wrapped :long)
            (body/scalar-expression
             :* :long [(body/literal Long/MAX_VALUE :long)
                       (body/literal 2 :long)]
             {:overflow :wrap}))]))))
  (testing "integral arithmetic cannot omit its overflow semantics"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"requires exactly one overflow contract"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'ambiguous :long)
            (body/scalar-expression
             :+ :long [(body/literal 1 :long) (body/literal 2 :long)]))]))))
  (testing "trapping and compiler-proved arithmetic are distinct legal contracts"
    (doseq [policy [:trap :no-overflow]]
      (is (body/kernel-body?
           (scalar-body
            [(body/->ScalarCompute
              (body/value (symbol (name policy)) :long)
              (body/scalar-expression
               :- :long [(body/literal 4 :long) (body/literal 2 :long)]
               {:overflow policy}))])))))
  (testing "wrapping overflow is not a floating-point policy"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"overflow contracts are only defined"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/scalar-expression :+ :float
                                    ['x-value (body/literal 1.0 :float)]
                                    {:overflow :wrap}))]))))
  (testing "wrapping overflow cannot annotate unrelated integral intrinsics"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"overflow contracts are only defined"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'bad :int)
            (body/scalar-expression :bit-xor :int
                                    [(body/literal 1 :int) (body/literal 2 :int)]
                                    {:overflow :wrap}))]))))
  (testing "integral sources do not accept a fictitious rounding direction"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'bad :long)
            (body/cast-expression (body/literal 1 :int) :long :up :exact))]))))
  (testing "integral-to-floating conversion states its precision loss"
    (is (body/kernel-body?
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'converted :float)
            (body/cast-expression (body/literal 1 :long)
                                  :float :nearest-even :exact))]))))
  (testing "floating conversions cannot silently request integer wrapping"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/cast-expression 'x-value :int :toward-zero :wrap))]))))
  (testing "FP narrowing states IEEE overflow and nearest-even rounding"
    (is (body/kernel-body?
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'narrowed :half)
            (body/cast-expression 'x-value :half :nearest-even :ieee))]))))
  (testing "IEEE overflow is not an integer conversion policy"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/cast-expression 'x-value :int :toward-zero :ieee))]))))
  (testing "same-width floating casts are exact rather than fictitious narrowing"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/cast-expression 'x-value :float :nearest-even :ieee))]))))
  (testing "floating widening is exact"
    (is (body/kernel-body?
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'widened :float)
            (body/cast-expression (body/literal 1.0 :half) :float :exact :exact))]))))
  (testing "floating narrowing cannot borrow integer saturation semantics"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :half)
            (body/cast-expression 'x-value :half :nearest-even :saturate))])))))

(deftest scalar-memory-is-ranked-masked-and-typed
  (testing "a load cannot silently reinterpret or widen storage"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"buffer element type"
         (scalar-body [(load-x (body/value 'x-value :int))]))))
  (testing "coordinates are element coordinates with the buffer rank"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"match the buffer rank"
         (scalar-body
          [(body/->ScalarLoad (body/value 'x-value :float) 'x ['group 'lane]
                              nil nil :cached)]))))
  (testing "a false load predicate always has a defined SSA value"
    (let [active (body/->Mask :active [(body/predicate :lt 'lane 8)])]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"explicit other value"
           (scalar-body
            [(body/->ScalarLoad (body/value 'x-value :float) 'x ['lane]
                                :active nil :cached)]
            :masks [active])))))
  (testing "an unmasked load cannot retain a meaningless fallback value"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"ignored other"
         (scalar-body
          [(body/->ScalarLoad (body/value 'x-value :float) 'x ['lane]
                              nil (body/literal 0.0 :float) :cached)]))))
  (testing "input-only storage cannot be written"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"writable output storage"
         (scalar-body
          [(load-x) (body/->ScalarStore 'x ['lane] 'x-value nil)])))))

(deftest atomic-updates-require-an-explicit-read-write-contract
  (let [atomic (body/->AtomicRMW 'y [0] (body/literal 1.0 :float) :+ nil)
        inout-body (fn [operation]
                     (-> (scalar-body [])
                         (assoc-in [:parameters 1 :kind] :inout)
                         (assoc :operations [operation])
                         body/validate!))]
    (is (body/kernel-body? (inout-body atomic)))
    (testing "write-only output storage cannot hide a read-modify-write"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"read-write kernel storage"
           (body/validate! (assoc (scalar-body []) :operations [atomic])))))
    (testing "the target-neutral body only accepts a proved update algebra"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"unsupported operator"
           (inout-body (assoc atomic :operator :max)))))
    (testing "the contribution must agree with the resident element dtype"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"contribution type"
           (inout-body (assoc atomic :value (body/literal 1 :int))))))))

(deftest structured-regions-prove-yields-and-collective-convergence
  (testing "both if branches terminate with typed yields"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"must terminate in Yield"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'condition :predicate)
            (body/scalar-expression :lt :predicate
                                    ['x-value (body/literal 0.0 :float)]))
           (body/->IfRegion 'condition [] [(body/->Yield [])] [])]))))
  (testing "loop-carried yields match the declared result product"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"yield arity or types"
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'acc :float) 'x-value)]
            [(body/->Yield [(body/literal 1 :int)])]
            [(body/value 'result :float)] {})]))))
  (testing "full-subgroup collectives cannot execute under lane-varying control"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lane-divergent"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'condition :predicate)
            (body/scalar-expression :lt :predicate
                                    ['x-value (body/literal 0.0 :float)]))
           (body/->IfRegion
            'condition
            [(reduce-x 'x-value) (body/->Yield [])]
            [(body/->Yield [])]
            [])]))))
  (testing "collective width agrees with the static launch and schedule"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"subgroup geometry"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'sum :float) :reduce :subgroup 8 'x-value :+ nil
            (body/full-participation) :implementation-defined)]))))
  (testing "broadcast source lanes are statically in range"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"statically in range"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'shared :float) :broadcast :subgroup 16 'x-value nil 16
            (body/full-participation) nil)]))))
  (testing "broadcasts do not claim a reduction association"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"no reduction association"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'shared :float) :broadcast :subgroup 16 'x-value nil 0
            (body/full-participation) :implementation-defined)])))))

(deftest memory-loads-cannot-prove-convergence-without-alias-facts
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"lane-divergent"
       (scalar-body
        [(body/->ScalarLoad (body/value 'shared-load :float) 'x ['group]
                            nil nil :cached)
         (body/->ScalarCompute
          (body/value 'condition :predicate)
          (body/scalar-expression :lt :predicate
                                  ['shared-load (body/literal 0.0 :float)]))
         (body/->IfRegion
          'condition
          [(reduce-x 'shared-load) (body/->Yield [])]
          [(body/->Yield [])]
          [])]))))

(deftest stable-input-loads-prove-only-their-coordinate-uniformity
  (let [stable [(body/stable-read 'x)]
        uniform-branch
        [(body/->ScalarLoad (body/value 'shared-load :float) 'x ['group]
                            nil nil :cached)
         (body/->ScalarCompute
          (body/value 'condition :predicate)
          (body/scalar-expression :lt :predicate
                                  ['shared-load (body/literal 0.0 :float)]))
         (body/->IfRegion
          'condition
          [(reduce-x 'shared-load) (body/->Yield [])]
          [(body/->Yield [])]
          [])]]
    (is (= stable (:stable-reads (scalar-body uniform-branch :stable-reads stable))))
    (testing "a lane-varying address remains lane-varying"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"lane-divergent"
           (scalar-body
            [(load-x)
             (body/->ScalarCompute
              (body/value 'condition :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->IfRegion
              'condition
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])]
            :stable-reads stable))))
    (testing "the contract names unique input parameters"
      (doseq [requirements [[(body/stable-read 'y)]
                            [(body/stable-read 'missing)]
                            [(body/stable-read 'x) (body/stable-read 'x)]]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (scalar-body [] :stable-reads requirements)))))))

(deftest kernel-storage-and-fragments-require-canonical-dtype-facts
  (let [kernel (minimal-body)]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"known dtype"
         (body/validate! (assoc-in kernel [:parameters 0 :dtype] :mystery))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"known dtype"
         (body/validate! (assoc-in kernel [:fragments 0 :dtype] :mystery))))))

(deftest loop-uniformity-is-safe-across-backedges-and-zero-trips
  (testing "a uniform initial carry cannot prove later iterations uniform"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lane-divergent"
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'continue? :predicate)
                             (body/literal true :predicate))]
            [(body/->IfRegion
              'continue?
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])
             (body/->ScalarCompute
              (body/value 'next? :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->Yield ['next?])]
            [(body/value 'finished? :predicate)] {})]))))
  (testing "an explicit inductive invariant admits a genuinely uniform carry"
    (is (body/kernel-body?
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'continue? :predicate)
                             (body/literal true :predicate))]
            [(body/->IfRegion
              'continue?
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])
             (body/->Yield [(body/literal true :predicate)])]
            [(body/value 'finished? :predicate)]
            {:uniform-iter-args #{'continue?}})]))))
  (testing "the verifier checks rather than trusts the claimed backedge invariant"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not preserved by its backedge"
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'continue? :predicate)
                             (body/literal true :predicate))]
            [(body/->IfRegion
              'continue?
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])
             (body/->ScalarCompute
              (body/value 'next? :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->Yield ['next?])]
            [(body/value 'finished? :predicate)]
            {:uniform-iter-args #{'continue?}})]))))
  (testing "uniformity annotations can name only carried bindings"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"must name carried bindings"
         (scalar-body
          [(body/->ForLoop
            (body/value 'i :int) 0 1 1
            [(body/->LoopArg (body/value 'carry :predicate)
                             (body/literal true :predicate))]
            [(body/->Yield [(body/literal true :predicate)])]
            [(body/value 'result :predicate)]
            {:uniform-iter-args #{'missing}})]))))
  (testing "a zero-trip loop result retains its initial value's variation"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lane-divergent"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'initial? :predicate)
            (body/scalar-expression :lt :predicate
                                    ['x-value (body/literal 0.0 :float)]))
           (body/->ForLoop
            (body/value 'i :int) 0 0 1
            [(body/->LoopArg (body/value 'carry? :predicate) 'initial?)]
            [(body/->Yield [(body/literal true :predicate)])]
            [(body/value 'result? :predicate)] {})
           (body/->IfRegion
            'result?
            [(reduce-x 'x-value) (body/->Yield [])]
            [(body/->Yield [])]
            [])])))))

(deftest scalar-and-collective-operators-have-semantic-dtype-domains
  (testing "floating intrinsics do not accept integers"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not defined for its operand dtype"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'integer :int)
            (body/cast-expression 'x-value :int :toward-zero :saturate))
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/scalar-expression :sqrt :int ['integer]))]))))
  (testing "bitwise intrinsics do not accept floats"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not defined for its operand dtype"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/scalar-expression :bit-and :float
                                    ['x-value (body/literal 1.0 :float)]))]))))
  (testing "collective operator legality includes its input dtype"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not associative for its input dtype"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'bad :float) :reduce :subgroup 16 'x-value :bit-and nil
            (body/full-participation) :implementation-defined)])))))

(deftest scalar-ssa-cannot-shadow-legacy-loop-indices
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"not globally unique"
       (scalar-body
        [(body/->Loop
          'legacy-index 0 1 1
          [(body/->ScalarCompute
            (body/value 'legacy-index :float)
            (body/scalar-expression :+ :float
                                    [(body/literal 1.0 :float)
                                     (body/literal 2.0 :float)]))]
          {})]))))

(deftest hardware-index-axes-must-exist-in-the-launch-contract
  (let [kernel (scalar-body [])]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"invalid hardware source"
         (body/validate! (assoc-in kernel [:indices 0 :axis] 1))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"invalid hardware source"
         (body/validate! (assoc-in kernel [:indices 1 :axis] 1))))))

(defn- scratch-allocation
  ([] (scratch-allocation 'scratch :float [16] 16))
  ([id dtype shape alignment]
   (body/->WorkgroupAllocation id dtype shape (layout/row-major shape dtype) alignment)))

(defn- workgroup-barrier []
  (body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                           (body/full-participation)))

(deftest workgroup-storage-has-exact-static-resource-accounting
  (let [allocations [(scratch-allocation 'bytes :byte [3] 1)
                     (scratch-allocation 'values :float [4] 16)]
        plan (body/workgroup-memory-plan allocations)
        kernel (scalar-body
                [(body/->ScalarStore 'values [0] (body/literal 1.0 :float) nil)
                 (workgroup-barrier)
                 (body/->ScalarLoad (body/value 'loaded :float) 'values [0] nil nil :cached)]
                :allocations allocations :shared-memory-bytes 32)]
    (is (= [{:allocation (first allocations) :byte-offset 0 :byte-size 3}
            {:allocation (second allocations) :byte-offset 16 :byte-size 16}]
           (:allocations plan)))
    (is (= 32 (:bytes plan)))
    (is (= 16 (:alignment plan)))
    (is (body/kernel-body? kernel))
    (testing "the launch charge cannot under- or over-state the packed arena"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"shared-memory charge disagree"
           (scalar-body [] :allocations allocations :shared-memory-bytes 31))))
    (testing "dynamic allocation shapes are rejected before target lowering"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"static shape"
           (scalar-body []
                        :allocations [(scratch-allocation 'dynamic :float ['n] 16)]
                        :shared-memory-bytes 0))))
    (testing "the allocation layout cannot drift from its storage contract"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"agree with its layout"
           (scalar-body []
                        :allocations [(assoc (scratch-allocation) :layout
                                             (layout/row-major [8] :float))]
                        :shared-memory-bytes 64))))))

(deftest shared-memory-layouts-have-an-explicit-storage-and-copy-boundary
  (is (body/kernel-body? (fixtures/swizzled-workgroup-memory-body 32)))
  (testing "a shared layout cannot leak onto a global ABI parameter"
    (let [kernel (scalar-body [])]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"restricted to workgroup allocations"
           (body/validate!
            (assoc-in kernel [:parameters 0 :layout]
                      (layout/shared-memory [16] :float :identity)))))))
  (testing "a swizzled logical row copy is explicit and cannot promise native overlap"
    (let [kernel (-> (fixtures/async-staging-body 32 :preferred)
                     (assoc-in [:allocations 0 :shape] [32 32])
                     (assoc-in [:allocations 0 :layout]
                               (layout/shared-memory [32 32] :float :xor-32))
                     (assoc-in [:operations 0 :destination-coordinates] [0 0])
                     (assoc-in [:operations 4 :coordinates] [0 'lane])
                     (assoc-in [:launch :shared-memory-bytes] 4096))]
      (is (body/kernel-body? (body/validate! kernel)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"contiguous global-to-workgroup transfer"
                            (body/validate! (assoc-in kernel [:operations 0 :overlap]
                                                      :required)))))))

(deftest workgroup-barriers-require-full-convergent-participation
  (let [barrier (workgroup-barrier)]
    (is (body/kernel-body? (scalar-body [barrier])))
    (testing "a lane-varying branch cannot contain a workgroup barrier"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"divergent control flow"
           (scalar-body
            [(load-x)
             (body/->ScalarCompute
              (body/value 'negative? :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->IfRegion 'negative?
                              [barrier (body/->Yield [])]
                              [(body/->Yield [])]
                              [])]))))
    (testing "barrier scope, memory semantics, and participation are explicit"
      (doseq [invalid [(assoc barrier :scope :subgroup)
                       (assoc barrier :memory-spaces #{:global})
                       (assoc barrier :semantics :relaxed)
                       (assoc barrier :participation (body/->Participation :masked))]]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"unsupported synchronization contract"
             (scalar-body [invalid])))))))

(defn- async-stage-copy
  ([] (async-stage-copy [0]))
  ([source-coordinates]
   (body/->AsyncWorkgroupCopy
    'copy-x 'x source-coordinates 'scratch [0] 16 16 :cached :preferred
    (body/full-participation))))

(defn- async-stage-operations
  []
  [(async-stage-copy)
   (body/->AsyncCommit 'copy-group ['copy-x])
   (body/->AsyncWait ['copy-group] 0 :acquire (body/full-participation))
   (workgroup-barrier)
   (body/->ScalarLoad (body/value 'staged :float) 'scratch ['lane] nil nil :cached)])

(deftest async-staging-has-verified-issue-commit-wait-lifetimes
  (let [options [:stable-reads [(body/stable-read 'x)]
                 :allocations [(scratch-allocation)]
                 :shared-memory-bytes 64]]
    (is (body/kernel-body? (apply scalar-body (async-stage-operations) options)))
    (testing "the source must stay stable for the entire asynchronous lifetime"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"stable-read contract"
           (scalar-body (async-stage-operations)
                        :allocations [(scratch-allocation)]
                        :shared-memory-bytes 64))))
    (testing "a commit closes exactly the copies issued since the previous commit"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"close every copy"
           (apply scalar-body
                  (assoc (async-stage-operations) 1
                         (body/->AsyncCommit 'copy-group ['different-copy]))
                  options))))
    (testing "waits consume the oldest group prefix and state remaining depth"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"consume an oldest prefix"
           (apply scalar-body
                  (assoc (async-stage-operations) 2
                         (body/->AsyncWait ['copy-group] 1 :acquire
                                           (body/full-participation)))
                  options))))
    (testing "wait completion does not silently imply cross-thread visibility"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before a workgroup barrier"
           (apply scalar-body
                  (vec (concat (subvec (async-stage-operations) 0 3)
                               (subvec (async-stage-operations) 4)))
                  options))))
    (testing "staged storage cannot be consumed while its copy group is incomplete"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before its async wait"
           (apply scalar-body
                  [(async-stage-copy)
                   (body/->ScalarLoad (body/value 'too-early :float)
                                      'scratch ['lane] nil nil :cached)]
                  options))))
    (testing "the cooperative base address must be workgroup-uniform"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"issued uniformly"
           (apply scalar-body
                  (assoc (async-stage-operations) 0 (async-stage-copy ['lane]))
                  options))))
    (testing "async coordinates may use prior uniform scalar SSA values"
      (is (body/kernel-body?
           (apply scalar-body
                  (into [(body/->ScalarCompute
                          (body/value 'source-base :int)
                          (body/scalar-expression
                           :+ :int ['group (body/literal 0 :int)]
                           {:overflow :no-overflow}))]
                        (assoc (async-stage-operations) 0
                               (async-stage-copy ['source-base])))
                  options))))
    (testing "undefined async coordinate SSA values still fail in the typed verifier"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before it is defined"
           (apply scalar-body
                  (assoc (async-stage-operations) 0 (async-stage-copy ['missing-base]))
                  options))))))

(deftest pipelined-for-carries-a-verified-rotating-async-queue
  (let [kernel (fixtures/pipelined-staging-body 16 :preferred)
        pipeline-index 4]
    (is (body/kernel-body? kernel))
    (is (= :exact (get-in kernel [:operations pipeline-index :attributes :tail-policy])))
    (is (= ['pipeline-group-a 'pipeline-group-b]
           (get-in kernel [:operations pipeline-index :async-results])))
    (testing "the tail policy is part of the scheduled loop contract"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"explicit tail policy"
           (body/validate! (assoc-in kernel
                                     [:operations pipeline-index :attributes]
                                     {})))))
    (testing "every carried group is the complete ordered queue at loop entry"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"complete ordered async queue"
           (body/validate!
            (-> kernel
                (assoc-in [:operations pipeline-index :async-iter-args]
                          [(body/->AsyncLoopArg 'carry-a 'warm-group-a)])
                (assoc-in [:operations pipeline-index :async-results]
                          ['pipeline-group-a]))))))
    (testing "a stage cannot be refilled before all workgroup readers finish"
      (let [pipeline-operations (get-in kernel [:operations pipeline-index :operations])]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"destination is still live"
             (body/validate!
              (assoc-in kernel [:operations pipeline-index :operations]
                        (vec (concat (subvec pipeline-operations 0 3)
                                     (subvec pipeline-operations 4)))))))))
    (testing "the backedge preserves each rotating stage position"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"preserve its rotating async stages"
           (body/validate!
            (assoc-in kernel [:operations pipeline-index :operations 4 :elements] 8)))))
    (testing "pending pipeline results must be drained outside the loop"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"lifetime crosses a structured region boundary"
           (body/validate! (assoc kernel :operations
                                  (subvec (:operations kernel) 0 5))))))))
