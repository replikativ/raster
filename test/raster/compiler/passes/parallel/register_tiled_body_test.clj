(ns raster.compiler.passes.parallel.register-tiled-body-test
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as body-emit]
            [raster.compiler.backend.gpu.segop-opencl :as segop-emit]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.register-tiled-body :as register-tiled]))

(def ^:private small-tile
  {:block-m 4 :block-n 4 :block-k 2 :thread-m 2 :thread-n 2})

(defn- contraction
  [& [epilogue init]]
  (facts/from-components
   {:out 'C
    :free-axes [['i 8] ['j 8]]
    :contract-axes [['k 8]]
    :body '(raster.numeric/*
            (clojure.core/aget A (clojure.core/+ (clojure.core/* i 8) k))
            (clojure.core/aget B (clojure.core/+ (clojure.core/* k 8) j)))
    :opts (cond-> {} epilogue (assoc :epilogue epilogue)
                  (some? init) (assoc :init init))
    :dtype :float}))

(defn- operation-kinds
  [operations]
  (mapcat
   (fn [operation]
     (concat [(some-> operation class .getSimpleName)]
             (when (instance? raster.compiler.ir.kernel_body.ForLoop operation)
               (operation-kinds (:operations operation)))))
   operations))

(deftest checked-zero-identities-reach-the-register-tiled-body
  (doseq [init '[0.0 (float 0.0) (double (float 0))]]
    (let [proof (contraction nil init)
          emitted (segop-emit/generate-register-tiled-kernel-body proof 'C :tile small-tile)]
      (is (body/kernel-body? (:kernel-body emitted)))
      (is (= init (:neutral (facts/scalar-reduction-view proof)))))))

(deftest cooperative-register-tile-is-a-verified-scalar-kernel-body
  (let [kernel-body (:kernel-body
                     (register-tiled/lower (contraction) {:tile small-tile}))
        kinds (frequencies (operation-kinds (:operations kernel-body)))
        opencl (body-emit/emit-scalar-kernel "register_tile" kernel-body)
        cuda (body-emit/emit-scalar-kernel
              "register_tile" kernel-body {:target-dialect :cuda})
        hip (body-emit/emit-scalar-kernel
             "register_tile" kernel-body {:target-dialect :hip})]
    (is (body/kernel-body? kernel-body))
    (is (= [[4 2] [2 4]] (mapv :shape (:allocations kernel-body))))
    (is (= [2 2] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [2 2] (mapv #(launch/resolve-expression {} %)
                       (get-in kernel-body [:launch :group-count]))))
    (is (= 4 (get kinds "ForLoop")) "outer K, inner K and two cooperative staging loops are explicit")
    (is (= 2 (get kinds "WorkgroupBarrier")))
    (is (str/includes? opencl "__local"))
    (is (str/includes? opencl "barrier(CLK_LOCAL_MEM_FENCE)"))
    (is (str/includes? cuda "__shared__"))
    (is (str/includes? cuda "__syncthreads()"))
    (is (str/includes? hip "__shared__"))
    (is (str/includes? hip "__syncthreads()"))
    (testing "the nested cooperative schedule is valid OpenCL C"
      (if-not (zero? (:exit (shell/sh "sh" "-c" "command -v clang")))
        (is true "clang unavailable")
        (let [compiled (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                 "-fsyntax-only" "-" :in opencl)]
          (is (zero? (:exit compiled)) (:err compiled)))))))

(deftest result-transform-is-alpha-renamed-per-microtile-store
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/*
                          (raster.numeric/+ acc (clojure.core/aget bias j)) scale)
                  :operands [{:sym 'bias :dtype :float
                              :map (axis-map/of-axes [['j 8]])}]
                  :scalars [{:sym 'scale :dtype :float}]
                  :dtype :float}
        emitted (segop-emit/generate-register-tiled-kernel-body
                 (contraction epilogue) 'C :tile small-tile)
        kernel-body (:kernel-body emitted)]
    (is (body/kernel-body? kernel-body))
    (is (= '[A B C bias scale] (mapv :name (:abi emitted))))
    (is (= '[bias] (:epilogue-operands emitted)))
    (is (= '[scale] (:epilogue-scalars emitted)))
    (is (= '[A B bias] (mapv :buffer (:stable-reads kernel-body))))
    (is (str/includes? (:source emitted) "bias["))
    (is (str/includes? (:source emitted) "* scale"))))

(deftest target-resource-limits-select-or-refuse-a-finite-tile
  (let [lowered (register-tiled/lower
                 (contraction)
                 {:descriptor {:execution {:max-workgroup-size 64}
                               :shared-local-memory 4096}})]
    (is (= {:block-m 32 :block-n 32 :block-k 16 :thread-m 4 :thread-n 4}
           (:tile lowered)))
    (is (= [8 8] (get-in lowered [:kernel-body :launch :workgroup-size]))))
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"cannot host any register-tiled"
       (register-tiled/lower
        (contraction)
        {:descriptor {:execution {:max-workgroup-size 8}
                      :shared-local-memory 128}}))))

(deftest a-destination-reading-result-transform-stores-through-one-read-write-parameter
  (let [epilogue {:acc 'acc
                  :expr '(raster.numeric/+ acc (raster.numeric/* beta (clojure.core/aget C (clojure.core/+ (clojure.core/* i 8) j))))
                  :operands [{:sym 'C :dtype :float
                              :map (axis-map/of-axes [['i 8] ['j 8]])}]
                  :scalars [{:sym 'beta :dtype :float}]
                  :dtype :float}
        emitted (segop-emit/generate-register-tiled-kernel-body
                 (contraction epilogue) 'C :tile small-tile)
        kernel-body (:kernel-body emitted)]
    (is (body/kernel-body? kernel-body))
    (is (= '[[A :input] [B :input] [C :inout] [beta :scalar]]
           (mapv (juxt :name :kind) (:abi emitted))))
    (is (empty? (:epilogue-operands emitted)))
    (is (= '[A B] (mapv :buffer (:stable-reads kernel-body)))
        "the destination is written, so it carries no stable-read contract")
    (is (re-find #"__global float\* out" (:source emitted))
        "the destination pointer is writable: no const qualifier")
    (is (re-find #"\? out\[" (:source emitted))
        "the destination element is loaded under the same store mask")
    (is (str/includes? (:source emitted) "(beta * "))))
