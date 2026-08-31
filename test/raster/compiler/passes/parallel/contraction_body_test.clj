(ns raster.compiler.passes.parallel.contraction-body-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.passes.parallel.contract-lower :as lower]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]))

(def ^:private matvec
  '(raster.par/contract y [[i m]] [[l k]]
     (* (aget A (+ (* i k) l)) (aget x l))))

(defn- portable-plan []
  (let [verified (facts/contraction-facts matvec :dtype :float)
        segred (lower/contract-form->segred matvec :dtype :float :facts verified)]
    (schedule/plan-portable-body verified segred nil)))

(deftest portable-contraction-is-a-complete-scheduled-kernel-body
  (let [plan (portable-plan)
        kernel (:body plan)
        loop (first (:operations kernel))]
    (is (:ok plan))
    (is (body/kernel-body? kernel))
    (is (= :sequential-segments (get-in kernel [:schedule :strategy])))
    (is (= [256] (get-in kernel [:launch :workgroup-size])))
    (is (= '[A x y k m _nseg] (mapv :id (:parameters kernel))))
    (is (= [:input :input :output :scalar :scalar :scalar]
           (mapv :kind (:parameters kernel))))
    (is (= 'i (:id (last (:indices kernel))))
        "the free output coordinate is decomposed from the flat segment index")
    (is (= 'l (get-in loop [:index :id])))
    (is (= 'k (:upper loop)))
    (is (= 1 (:step loop)))
    (is (= :active-segment (:predicate (last (:operations kernel)))))))

(deftest one-portable-body-emits-through-every-c-family-dialect
  (let [kernel (:body (portable-plan))]
    (doseq [[dialect target entry]
            [[:opencl-portable :opencl-c "__kernel void"]
             [:cuda :cuda-c "extern \"C\" __global__ void"]
             [:hip :hip-cpp "extern \"C\" __global__ void"]]]
      (testing (name dialect)
        (let [emitted (emit/generate-contraction-kernel-body
                       kernel :target-dialect dialect)]
          (is (= target (:target emitted)))
          (is (str/includes? (:source emitted) entry))
          (is (str/includes? (:source emitted) "for (int rstr_l = 0"))
          (is (= '[A x] (:array-params emitted)))
          (is (= '[k m] (:scalar-params emitted))))))))

(deftest production-general-contraction-route-carries-the-scheduled-body
  (let [routed (route/route-contraction matvec :dtype :float :desc {})]
    (is (= :portable-segred (:strategy routed)))
    (is (body/kernel-body? (:kernel-body routed)))
    (is (= (:kernel-body routed)
           (artifact/attribute (:artifact routed) :kernel-body)))
    (is (= '[A x y k m _nseg] (mapv :name (:abi routed))))
    (is (= [256] (:wg routed)))
    (is (= (:launch (:kernel-body routed))
           (:launch (:artifact routed)))
        "routing preserves the schedule's actual 1-D launch contract")
    (is (empty? (:declines routed)))))

(deftest eligible-affine-contractions-never-reenter-the-source-template
  (with-redefs [emit/generate-segmented-reduce-kernel
                (fn [& _]
                  (throw (ex-info "legacy source template was called" {})))]
    (is (= :portable-segred
           (:strategy (route/route-contraction matvec :dtype :float :desc {}))))))

(deftest flattened-symbolic-reduction-extents-remain-in-the-typed-abi
  (let [form '(raster.par/contract y [[i m]] [[l1 k1] [l2 k2]]
                (* (aget A (+ (* i (* k1 k2)) (* l1 k2) l2))
                   (aget x (+ (* l1 k2) l2))))
        routed (route/route-contraction form :dtype :float :desc {})]
    (is (= :portable-segred (:strategy routed)))
    (is (= '[k1 k2 m] (mapv :value (butlast (:scalar-args routed)))))
    (is (= '[A x y k1 k2 m _nseg] (mapv :name (:abi routed))))
    (is (= #{'k1 'k2 'm}
           (set (map :id (filter #(= :scalar (:kind %))
                                (butlast (:parameters (:kernel-body routed))))))))))

(deftest non-affine-gather-declines-without-changing-semantics
  (let [form '(raster.par/contract y [[i m]] [[l k]]
                (* (aget A (aget indices i)) (aget x l)))
        routed (route/route-contraction form :dtype :float :desc {})]
    (is (= :naive-segred (:strategy routed)))
    (is (= :operand-layout (:fallback-reason routed)))
    (is (= :portable-kernel-body (-> routed :declines last :leaf)))
    (is (nil? (:kernel-body routed)))))
