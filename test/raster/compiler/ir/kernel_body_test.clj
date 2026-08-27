(ns raster.compiler.ir.kernel-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]))

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
    :launch {:workgroup-size [16] :group-count [1]}
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
                            (body/validate! bad))))))

(deftest buffer-views-retain-parent-ownership-and-explicit-offsets
  (let [kernel (minimal-body)
        kernel (-> kernel
                   (assoc-in [:parameters 0 :shape] [16 16 16])
                   (assoc-in [:launch :group-count] [16]))
        view (body/->BufferView 'A-slab 'A
                                (body/expression :mul 'group-x 16 16)
                                [16 16] (:layout (first (:parameters kernel))))
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
