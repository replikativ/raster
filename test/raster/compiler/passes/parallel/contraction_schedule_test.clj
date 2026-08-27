(ns raster.compiler.passes.parallel.contraction-schedule-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.backend.gpu.opencl-codegen :as opencl-codegen]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.passes.parallel.contract-lower :as contract-lower]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]))

(defn- matrix-form [m n k & [init]]
  (concat
   (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
         (list 'raster.numeric/*
               (list 'clojure.core/aget 'A
                     (list 'clojure.core/+ (list 'clojure.core/* 'i k) 'l))
               (list 'clojure.core/aget 'B
                     (list 'clojure.core/+ (list 'clojure.core/* 'l n) 'j))))
   (when (some? init) [:init init])))

(deftest applying-a-matrix-schedule-produces-a-target-neutral-kernel-body
  (let [contract-facts (facts/contraction-facts (matrix-form 128 128 128) :dtype :half)
        planned (schedule/plan-matrix-body contract-facts nil nil {:operation-id 41})
        kernel (:body planned)
        guard (first (:operations kernel))
        outer-loop (first (filter #(contains? % :step) (:operations guard)))
        inner-loop (first (:operations outer-loop))]
    (is (:ok planned))
    (is (body/kernel-body? kernel))
    (is (= [:contraction 41] (:id kernel)))
    (is (= 41 (get-in kernel [:provenance :operation-id])))
    (is (= {:row 'A :col 'B} (:bindings planned)))
    (is (= :dpas (get-in kernel [:attributes :instruction-family])))
    (is (= {:m :masked :n :masked :k :exact-fragments}
           (get-in kernel [:attributes :boundary-policy])))
    (is (= 32 (:step outer-loop)) "the scheduled K block is IR")
    (is (= 16 (:step inner-loop)) "the hardware fragment step is independently IR")
    (is (= 3 (get-in inner-loop [:operations 0 :distance]))
        "pipeline depth reaches the explicit prefetch op")
    (is (= :dot-operand (get-in kernel [:fragments 8 :layout :kind])))
    (is (= :mma-frag (get-in kernel [:fragments 0 :layout :kind])))))

(deftest matrix-fragment-boundaries-are-a-legality-policy
  (testing "K=24 meets the old 16-byte pitch test but cannot form a final K16 instruction"
    (let [planned (schedule/plan-matrix-body
                   (facts/contraction-facts (matrix-form 128 128 24) :dtype :half) nil nil)]
      (is (false? (:ok planned)))
      (is (= :partial-matrix-k-fragment (:reason planned)))
      (is (= 16 (:matrix-k planned)))))
  (testing "a non-zero reduction initializer cannot be dropped by a zero-initialized matrix body"
    (is (= :non-zero-matrix-init
           (:reason (schedule/plan-matrix-body
                     (facts/contraction-facts (matrix-form 128 128 128 1.0) :dtype :half)
                     nil nil)))))
  (testing "matrix N and subgroup width must agree with an implemented DPAS spelling"
    (let [desc {:matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 8}}
          planned (schedule/plan-matrix-body
                   (facts/contraction-facts (matrix-form 128 128 128) :dtype :half)
                   desc nil)
          routed (route/route-contraction (matrix-form 128 128 128)
                                          :dtype :half :desc desc)]
      (is (false? (:ok planned)))
      (is (= :matrix-instruction-not-lowered (:reason planned)))
      (is (= :regtiled (:strategy routed)))
      (is (= :matrix-instruction-not-lowered (:fallback-reason routed))))))

(deftest authoritative-targets-without-matrix-capabilities-do-not-inherit-dpas
  (let [desc {:device-type :gpu :backend :ocl
              :execution {:subgroup-sizes #{}
                          :preferred-subgroup-size nil
                          :max-workgroup-size 256}}
        planned (schedule/plan-matrix-body
                 (facts/contraction-facts (matrix-form 128 128 128) :dtype :half)
                 desc nil)]
    (is (false? (:ok planned)))
    (is (= :matrix-capability-unavailable (:reason planned)))))

(deftest the-production-route-carries-the-body-into-the-executable-artifact
  (let [routed (route/route-contraction (matrix-form 128 128 128) :dtype :half)
        kernel (:kernel-body routed)]
    (is (= :dpas (:strategy routed)))
    (is (body/kernel-body? kernel))
    (is (= kernel (artifact/attribute (:artifact routed) :kernel-body)))
    (is (= (:tile routed) (:schedule kernel)))
    (is (= '[A B] (:array-params routed)))))

(deftest production-lowering-preserves-the-semantic-operation-identity
  (let [contract (matrix-form 128 128 128)
        source (list 'let* ['c contract] 'c)
        lowered (:form (segop-lower/segop-lower-pass
                        source {:target-device :ze:0 :dtype :half}))
        semantic-operation (first (program/operations-for-binding lowered 'c contract))
        compiled (opencl/opencl-pass lowered :dtype :half :min-elements 0)
        kernel (artifact/attribute (first (:kernels compiled)) :kernel-body)]
    (is (some? semantic-operation))
    (is (= [:contraction (:id semantic-operation)] (:id kernel)))
    (is (= (:id semantic-operation) (get-in kernel [:provenance :operation-id])))))

(deftest kernel-body-lowering-shadows-the-proven-dpas-oracle
  (let [form (matrix-form 128 128 128)
        routed (route/route-contraction form :dtype :half)
        oracle (segop-opencl/generate-dpas-contraction-kernel
                (contract-lower/contract-form->segred form :dtype :half) 'C)
        normalize #(str/replace % #"dpas_contract_[0-9]+" "dpas_contract_N")]
    (is (= (normalize (:source oracle)) (normalize (:source routed))))
    (is (= (:tile oracle) (:tile routed)))
    (is (= (:workgroup oracle) (:wg routed)))
    (is (body/kernel-body? (:kernel-body routed)))))

(deftest typed-store-regions-lower-identically-to-the-epilogue-oracle
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/+ acc (clojure.core/aget bias j))
                  :operands [{:sym 'bias
                              :map (axis-map/of-axes [['j 128]])
                              :dtype :float}]}
        form (concat (matrix-form 128 128 128) [:epilogue epilogue])
        routed (route/route-contraction form :dtype :half)
        oracle (segop-opencl/generate-dpas-contraction-kernel
                (contract-lower/contract-form->segred form :dtype :half) 'C
                :epilogue epilogue)
        normalize #(str/replace % #"dpas_contract_[0-9]+" "dpas_contract_N")]
    (is (= (normalize (:source oracle)) (normalize (:source routed))))
    (is (= '[bias] (:epilogue-operands routed)))
    (is (re-find #"bias\[col\]" (:source routed)))
    (is (some :value-region (get-in routed [:kernel-body :operations 0 :operations])))))

(deftest production-kernel-body-lowering-does-not-call-the-legacy-template
  (with-redefs [opencl-codegen/emit-gemm-tiled
                (fn [& _]
                  (throw (ex-info "legacy template was called" {:reason :test/failure})))]
    (let [routed (route/route-contraction (matrix-form 128 128 128) :dtype :half)]
      (is (= :dpas (:strategy routed)))
      (is (re-find #"intel_sub_group_f16_f16_matrix_mad_k16" (:source routed)))
      (is (body/kernel-body? (:kernel-body routed))))))

(deftest raw-epilogue-helper-source-never-enters-production-lowering
  (let [epilogue {:acc 'acc
                  :expr '(silu_f acc)
                  :helpers "inline float silu_f(float x) { return x; }"}
        form (concat (matrix-form 128 128 128) [:epilogue epilogue])
        scheduled (schedule/plan-matrix-body
                   (facts/contraction-facts form :dtype :half) nil nil)]
    (is (= {:ok false :reason :opaque-epilogue-helper} scheduled))
    (with-redefs [opencl-codegen/emit-gemm-tiled
                  (fn [& _]
                    (throw (ex-info "legacy template was called" {:reason :test/failure})))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"has no store splice"
                            (route/route-contraction form :dtype :half))))))

(deftest direct-lowering-refuses-an-operation-coordinate-that-disagrees-with-the-schedule
  (let [kernel (:body (schedule/plan-matrix-body
                       (facts/contraction-facts (matrix-form 128 128 128) :dtype :half)
                       nil nil))
        changed? (atom false)
        corrupt (walk/postwalk
                 (fn [value]
                   (if (and (not @changed?)
                            (instance? raster.compiler.ir.kernel_body.TileLoad value))
                     (do (reset! changed? true)
                         (update value :coordinates
                                 #(assoc % 0 (body/expression :add (first %) 1))))
                     value))
                 kernel)]
    (is @changed?)
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"rhs tile coordinates"
                          (segop-opencl/generate-dpas-kernel-body corrupt 'C)))))

(deftest a-partial-matrix-fragment-falls-back-with-an-explanation
  (let [routed (route/route-contraction (matrix-form 128 128 24) :dtype :half)]
    (is (= :regtiled (:strategy routed)))
    (is (= :partial-matrix-k-fragment (:fallback-reason routed)))
    (is (= :partial-matrix-k-fragment (get-in routed [:declines 0 :reason])))))

(deftest opencl-consumes-the-resolved-contraction-tile
  (let [tile (assoc (hardware/derive-gemm-tile {})
                    :block-m 64 :block-n 64 :num-stages 2)
        compiled (opencl/opencl-pass (matrix-form 128 128 128)
                                     :dtype :half :min-elements 0
                                     :schedule {:tile tile})
        artifact (first (:kernels compiled))
        kernel (artifact/attribute artifact :kernel-body)]
    (is (= tile (:schedule kernel)))
    (is (= [64 64] ((juxt :block-m :block-n) (:schedule kernel))))
    (is (= 2 (get-in kernel [:schedule :num-stages])))
    (is (re-find #"WG 64x64" (:source artifact)))
    (is (re-find #"\+ 32;" (:source artifact))
        "num-stages=2 changes the K16 prefetch distance to 32")))
