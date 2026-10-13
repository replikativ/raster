(ns raster.compiler.public-offset-fold-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

;; Public counterpart of nanparth's bounded Int32-offset Fold control, extracted
;; from #1338 at 063e7fc0. No runtime offset value is assumed during compilation.
(deftm ordered-offset-fold!
  [offsets :- (Array int), out :- (Array double)] :- (Array double)
  (par/map! out row 1 double
    (let [start (long (aget offsets 0))
          end (long (aget offsets 1))
          degree (- end start)]
      (loop [i (long 0) sum 0.0]
        (if (< i degree)
          (recur (inc i) (+ sum (double (+ start i))))
          sum))))
  out)

(deftm ordered-long-offset-fold!
  [offsets :- (Array long), out :- (Array double)] :- (Array double)
  (par/map! out row 1 double
    (let [start (aget offsets 0)
          end (aget offsets 1)
          degree (- end start)]
      (loop [i (long 0) sum 0.0]
        (if (< i degree)
          (recur (inc i) (+ sum (double (+ start i))))
          sum))))
  out)

(defn- source-target [backend]
  ;; Capture scheduling fields from an existing compiler descriptor.
  ;; Identity/provenance below explicitly describe a source fixture,
  ;; not an observed physical device or a measured CUDA/HIP execution receipt.
  (let [target (keyword (str (name backend) ":public-offset-fold-source"))
        descriptor (select-keys (hardware/descriptor-for :cuda:0)
                                [:vector-bits :num-vector-registers :machine-lanes
                                 :grf-bytes-per-lane :subgroup-size :subgroup-sizes
                                 :max-workgroup-size :execution :cache :llc-bytes
                                 :bandwidth-bytes-s :peak-flops :launch-overhead-ns])]
    [target (-> descriptor
                (assoc :device-id target :device-type :gpu :backend backend
                       :device-name "Declared offset-Fold source fixture"
                       :subgroup-dialect (case backend :ocl :opencl-portable :cuda :cuda :hip :hip)
                       :provenance {:kind :source-fixture})
                (assoc-in [:execution :provenance] {:kind :source-fixture})
                (assoc-in [:execution :scalar-dtype-support :double] :supported))]))

(defn- long-additions [kernel]
  (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                (= :+ (get-in % [:expression :op]))
                (= :long (get-in % [:expression :result-type])))
          (tree-seq coll? seq (:operations kernel))))

(deftest public-offset-fold-has-one-portable-proved-body
  (doseq [backend [:ocl :cuda :hip]
          :let [[target descriptor] (source-target backend)
                compilation (equation/compile #'ordered-offset-fold!
                                              {:target target :dtype :double} descriptor)
                artifact (first (:kernels compilation))
                kernel (get-in artifact [:attributes :kernel-body])
                additions (long-additions kernel)]]
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (= 1 (count (:kernels compilation))))
    (is (body/kernel-body? kernel))
    (is (seq additions))
    (is (every? #(= :no-overflow (get-in % [:expression :options :overflow])) additions))
    (is (not-any? #(and (map? %) (= :trap (:overflow %))) (tree-seq coll? seq kernel)))
    (is (map? (equation/lower compilation [(int-array [3 7]) (double-array 1)])))))

(deftest long-storage-cannot-borrow-the-int32-fold-proof
  (let [[target descriptor] (source-target :cuda)
        compilation (equation/compile #'ordered-long-offset-fold!
                                      {:target target :dtype :double} descriptor)
        kernel (get-in compilation [:kernels 0 :attributes :kernel-body])]
    (is (some #(= :trap (get-in % [:expression :options :overflow])) (long-additions kernel)))
    (is (= :kernel-body-c-trap-unsupported
           (try (emitter/emit-scalar-module "unsafe_long_offsets" kernel
                                           {:target-dialect :opencl-portable})
                :accepted
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(defn- reference [[start end]]
  ;; Independent unbounded integer oracle. Device fixtures keep domains tiny;
  ;; machine-extreme intervals are verifier tests, never giant device loops.
  (double (reduce +' 0 (range start end))))

(defn- run-device [target]
  (let [prepared (compiled/lower #'ordered-offset-fold!
                                 [(int-array [3 7]) (double-array 1)]
                                 {:compiler :equation-first :target target :dtype :double})
        program (compiled/instantiate! prepared)]
    (try
      (is (= 0 (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
      (doseq [pair [[3 7] [-7 -2] [4 4] [7 3]]]
        (let [offsets (int-array pair)
              expected (reference pair)
              cpu (ordered-offset-fold! offsets (double-array 1))
              result (program {:offsets offsets})]
          (is (= [expected] (vec cpu)))
          (is (= [expected] (vec (value/->host (:result result)))))))
      (finally (compiled/close! program)))))

(deftest public-offset-fold-opencl-replay-matches-independent-oracle
  (if @opencl/opencl-fp64-available?
    (run-device :ocl:0)
    (opencl/opencl-skip! "public offset Fold" :fp64)))

(deftest public-offset-fold-level-zero-replay-matches-independent-oracle
  (if @ze/gpu-available?
    (let [capabilities ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
      (if (:fp64? capabilities)
        (run-device :ze:0)
        (ze/gpu-capability-skip! "public offset Fold" :fp64? capabilities)))
    (ze/gpu-skip! "public offset Fold")))
