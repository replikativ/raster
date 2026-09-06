(ns raster.compiler.passes.parallel.contraction-body-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
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

(defn- portable-result-transform-form []
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/*
                          (raster.numeric/+ acc (clojure.core/aget bias j)) scale)
                  :operands [{:sym 'bias :dtype :float
                              :map (axis-map/of-axes [['j 8]])}]
                  :scalars [{:sym 'scale :dtype :float}]
                  :dtype :float}
        form '(raster.par/contract C [[i 4] [j 8]] [[l 16]]
                                    (raster.numeric/*
                                     (clojure.core/aget A (clojure.core/+ (clojure.core/* i 16) l))
                                     (clojure.core/aget B (clojure.core/+ (clojure.core/* l 8) j))))]
    (concat form [:epilogue epilogue])))

(defn- portable-result-transform-plan []
  (let [form (portable-result-transform-form)
        verified (facts/contraction-facts form :dtype :half)
        segred (lower/contract-form->segred form :dtype :half :facts verified)]
    (schedule/plan-portable-body verified segred nil)))

(deftest portable-contraction-retains-load-scalar-and-index-widths
  (let [form '(raster.par/contract y [[i m]] [[l k]] (* scale (aget x l)))
        verified (facts/contraction-facts form :dtype :double)
        segred (update (lower/contract-form->segred form :dtype :double :facts verified)
                       :scalars conj 'scale)
        plan (schedule/plan-portable-body
              verified segred nil
              {:array-types {'x :float}
               :scalar-types {'m :long 'k :int 'scale :float}})
        kernel (:body plan)
        parameters (into {} (map (juxt :id :dtype)) (:parameters kernel))
        loop (first (:operations kernel))]
    (is (:ok plan))
    (is (= kernel (body/validate! kernel)))
    (is (= :float (get parameters 'x)))
    (is (= :float (get parameters 'scale)))
    (is (= :double (get parameters 'y)))
    (is (= :long (get-in loop [:index :type])))
    (is (some #(and (= :cast (get-in % [:expression :op]))
                    (= :double (get-in % [:expression :result-type])))
              (:operations loop)))
    (doseq [target [:opencl-intel :cuda :hip]]
      (is (string? (:source (emit/generate-contraction-kernel-artifact
                            kernel :target-dialect target)))))))

(deftest mixed-storage-cannot-select-a-uniform-pointer-leaf
  (let [form '(raster.par/contract C [[i 4] [j 8]] [[l 16]]
                                 (* (aget A (+ (* i 16) l)) (aget B (+ (* l 8) j))))
        routed (route/route-contraction form :dtype :double
                                        :array-types {'A :float 'B :float})
        kernel (artifact/attribute (:artifact routed) :kernel-body)]
    (is (= :portable-segred (:strategy routed)))
    (is (body/kernel-body? kernel))
    (is (= [:float :float] (mapv :dtype (take 2 (:parameters kernel)))))
    (with-redefs [schedule/plan-portable-body (fn [& _] {:ok false :reason :test-decline})]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"mixed storage or floating captures require a typed portable"
           (route/route-contraction form :dtype :double
                                    :array-types {'A :float 'B :float}))))
    (doseq [family [:matrix :register-tiled]]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"no enabled contraction schedule family"
           (route/route-contraction form :dtype :double
                                    :array-types {'A :float 'B :float}
                                    :candidate-families #{family}))))))

(deftest portable-contraction-is-a-complete-scheduled-kernel-body
  (let [plan (portable-plan)
        kernel (:body plan)
        loop (first (:operations kernel))]
    (is (:ok plan))
    (is (body/kernel-body? kernel))
    (is (= :sequential-segments (get-in kernel [:schedule :strategy])))
    (is (= [256] (get-in kernel [:launch :workgroup-size])))
    (is (= [1]
           (mapv #(launch/resolve-expression {'m 7} %)
                 (get-in kernel [:launch :group-count]))))
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

(deftest portable-result-transform-is-typed-scalar-ssa-on-every-c-family-target
  (let [plan (portable-result-transform-plan)
        kernel (:body plan)
        operations (:operations kernel)
        epilogue-operations (subvec operations 1 (dec (count operations)))
        classes (mapv #(some-> % class .getSimpleName) epilogue-operations)]
    (is (:ok plan))
    (is (= '[A B C bias scale _nseg] (mapv :id (:parameters kernel))))
    (is (= [:operand :operand :result :epilogue :epilogue :bound]
           (mapv :role (:parameters kernel))))
    (is (= ["ScalarCompute" "ScalarLoad" "ScalarCompute" "ScalarCompute" "ScalarCompute"]
           classes)
        "widen accumulator, load bias, add, scale and narrow are explicit SSA")
    (is (= [:float :float :float :float :half]
           (mapv #(get-in % [:result :type]) epilogue-operations)))
    (is (= 'C (:buffer (last operations))))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (testing (name dialect)
        (let [emitted (emit/generate-contraction-kernel-body
                       kernel :target-dialect dialect)]
          (is (= '[A B] (:array-params emitted)))
          (is (= '[bias] (:epilogue-operands emitted)))
          (is (= '[scale] (:epilogue-scalars emitted)))
          (is (= '[A B C bias scale _nseg] (mapv :name (:abi emitted))))
          (is (str/includes? (:source emitted) "bias["))
          (is (str/includes? (:source emitted) "scale")))))))

(defn- let-epilogue-plan [expr]
  (let [epilogue {:acc 'acc
                  :expr expr
                  :operands [{:sym 'bias :dtype :float
                              :map (axis-map/of-axes [['j 8]])}]
                  :dtype :float}
        form '(raster.par/contract C [[i 4] [j 8]] [[l 16]]
                                    (raster.numeric/*
                                     (clojure.core/aget A (clojure.core/+ (clojure.core/* i 16) l))
                                     (clojure.core/aget B (clojure.core/+ (clojure.core/* l 8) j))))
        form (concat form [:epilogue epilogue])
        verified (facts/contraction-facts form :dtype :half)
        segred (lower/contract-form->segred form :dtype :half :facts verified)]
    (schedule/plan-portable-body verified segred nil)))

(deftest a-let-bound-result-transform-lowers-to-shared-typed-ssa
  (testing "a let binding is one SSA value reused by every later reference"
    (let [plan (let-epilogue-plan
                '(let [x (raster.numeric/+ acc (clojure.core/aget bias j))]
                   (raster.numeric// x (raster.numeric/+ 1.0
                                                         (raster.math/exp
                                                          (raster.numeric/* -1.0 x))))))
          kernel (:body plan)
          operations (:operations kernel)
          epilogue-operations (subvec operations 1 (dec (count operations)))
          operators (keep #(get-in % [:expression :op]) epilogue-operations)]
      (is (:ok plan))
      ;; widen acc, load bias, x = acc + bias, -1*x, exp, 1+exp, x/(1+exp), narrow to half
      (is (= 8 (count epilogue-operations)))
      (is (= [:cast :+ :* :exp :+ :div :cast] (vec operators))
          "the addition bound to x is emitted once and reused, not once per use")
      (doseq [dialect [:opencl-portable :cuda :hip]]
        (let [emitted (emit/generate-contraction-kernel-body kernel :target-dialect dialect)]
          (is (str/includes? (:source emitted) "exp("))
          (is (= '[bias] (:epilogue-operands emitted)))))))
  (testing "a let binder shadows the accumulator lexically"
    (let [plan (let-epilogue-plan '(let [acc (raster.numeric/* acc 2.0)] acc))
          operations (:operations (:body plan))
          epilogue-operations (subvec operations 1 (dec (count operations)))]
      (is (:ok plan))
      ;; widen acc, multiply, narrow to half
      (is (= 3 (count epilogue-operations))))))

(deftest a-result-transform-can-reuse-one-contraction-operand-without-a-second-abi-slot
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/+ acc (clojure.core/aget A
                                                                  (clojure.core/+
                                                                   (clojure.core/* i 8) j)))
                  :operands [{:sym 'A :dtype :half
                              :map (axis-map/of-axes [['i 4] ['j 8]])}]
                  :dtype :half}
        form (concat
              '(raster.par/contract C [[i 4] [j 8]] [[l 16]]
                                    (raster.numeric/*
                                     (clojure.core/aget A (clojure.core/+ (clojure.core/* i 16) l))
                                     (clojure.core/aget B (clojure.core/+ (clojure.core/* l 8) j))))
              [:epilogue epilogue])
        routed (route/route-contraction form :dtype :half :desc {}
                                        :candidate-families #{:portable})]
    (is (= :portable-segred (:strategy routed)))
    (is (= '[A B C _nseg] (mapv :name (:abi routed))))
    (is (= '[A B] (:array-params routed)))
    (is (empty? (:epilogue-operands routed)))
    (is (= '[A B] (mapv :buffer (get-in routed [:kernel-body :stable-reads]))))))

(deftest result-transforms-use-the-register-tiled-body-when-matrix-is-ineligible
  (let [descriptor {:device-type :gpu :backend :ocl
                    :execution {:subgroup-sizes #{}
                                :preferred-subgroup-size nil
                                :max-workgroup-size 256}}
        routed (route/route-contraction (portable-result-transform-form)
                                        :dtype :half :desc descriptor)]
    (is (= :regtiled (:strategy routed)))
    (is (true? (:fused-epilogue routed)))
    (is (body/kernel-body? (:kernel-body routed)))
    (is (= [:matrix-capability-unavailable]
           (mapv :reason (:declines routed))))))

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
    (is (= :kernel-body (artifact/emission-route (:artifact routed)))
        "the descriptor must preserve the emitter route independently of semantic provenance")
    (is (empty? (:declines routed)))))

(deftest eligible-affine-contractions-never-reenter-the-source-template
  (with-redefs [emit/generate-segmented-reduce-kernel
                (fn [& _]
                  (throw (ex-info "legacy source template was called" {})))]
    (is (= :portable-segred
           (:strategy (route/route-contraction matvec :dtype :float :desc {}))))))

(deftest full-reduction-emitter-refuses-a-segmented-space
  (let [verified (facts/contraction-facts matvec :dtype :float)
        segmented (lower/contract-form->segred matvec :dtype :float :facts verified)
        failure (try
                  (emit/generate-segred-kernel segmented 'y :dtype :float)
                  nil
                  (catch clojure.lang.ExceptionInfo exception
                    (ex-data exception)))]
    (is (= :segmented-reduction-requires-contraction-schedule (:reason failure)))
    (is (= :none (:fallback failure)))))

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
    (is (= :verified-segmented-opencl (artifact/emission-route (:artifact routed))))
    (is (= :operand-layout (:fallback-reason routed)))
    (is (= :portable-kernel-body (-> routed :declines last :leaf)))
    (is (nil? (:kernel-body routed)))))

(defn- destination-reading-form
  [dtype]
  (concat
   '(raster.par/contract C [[i 4] [j 8]] [[l 16]]
                         (raster.numeric/*
                          (clojure.core/aget A (clojure.core/+ (clojure.core/* i 16) l))
                          (clojure.core/aget B (clojure.core/+ (clojure.core/* l 8) j))))
   [:epilogue {:acc 'acc
               :expr '(raster.numeric/+ acc (raster.numeric/* beta (clojure.core/aget C (clojure.core/+ (clojure.core/* i 8) j))))
               :operands [{:sym 'C :dtype dtype :map (axis-map/of-axes [['i 4] ['j 8]])}]
               :scalars [{:sym 'beta :dtype dtype}]
               :dtype dtype}]))

(deftest a-result-transform-reading-the-destination-makes-it-one-read-write-parameter
  ;; `C := acc + beta·C` (an accumulating GEMM) reads the element this work item stores. The
  ;; destination is a single `:inout` parameter: no second read-only view of the same storage,
  ;; no stable-read claim on a buffer the kernel writes, and no `const` on its pointer.
  (let [routed (route/route-contraction (destination-reading-form :float) :dtype :float :desc {}
                                        :candidate-families #{:portable})]
    (is (= :portable-segred (:strategy routed)))
    (is (= '[[A :input] [B :input] [C :inout] [beta :scalar] [_nseg :scalar]]
           (mapv (juxt :name :kind) (:abi routed))))
    (is (= '[A B] (:array-params routed)))
    (is (empty? (:epilogue-operands routed)))
    (is (= '[beta] (:epilogue-scalars routed)))
    (is (= '[A B] (mapv :buffer (get-in routed [:kernel-body :stable-reads]))))
    (is (re-find #"__global float\* C" (:source routed)))
    (is (not (re-find #"const float\* C" (:source routed))))))

(deftest the-matrix-leaf-declines-a-destination-reading-transform-by-name
  ;; The tile store writes whole fragments; reading the destination element inside that store
  ;; is not expressed yet, so the FP16 product routes to the register-tiled body and says why.
  (let [routed (route/route-contraction (destination-reading-form :half) :dtype :half)]
    (is (= :regtiled (:strategy routed)))
    (is (= [[:dpas :epilogue-reads-destination]]
           (mapv (juxt :leaf :reason) (:declines routed))))
    (is (= :inout (:kind (first (filter #(= 'C (:name %)) (:abi routed))))))))
