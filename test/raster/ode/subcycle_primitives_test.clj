(ns raster.ode.subcycle-primitives-test
  "Numerical building blocks for the pending complete subcycled PDE vertical, not a
   subcycle schedule, reflux certificate or checkpoint acceptance by themselves."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.build-manifest :as build]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.completed-evidence-device-test :as producer]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.ode.finite-volume :as fv]))

(deftm temporal-and-transport!
  [at-old :- (Array double), at-mid :- (Array double), at-new :- (Array double),
   old :- (Array double), predicted :- (Array double), current :- (Array double),
   coarse-owner :- (Array int), cells :- Long,
   register :- (Array double), coarse-flux :- (Array double),
   fine-flux-0 :- (Array double), fine-flux-1 :- (Array double),
   interface :- (Array int), faces :- Long, dt :- Double] :- Void
  (fv/temporal-boundary! at-old old predicted current coarse-owner cells 0.0)
  (fv/temporal-boundary! at-mid old predicted current coarse-owner cells 0.5)
  (fv/temporal-boundary! at-new old predicted current coarse-owner cells 1.0)
  (fv/accumulate-interface-transport! register coarse-flux interface faces (- dt))
  (fv/accumulate-interface-transport! register fine-flux-0 interface faces (* 0.5 dt))
  (fv/accumulate-interface-transport! register fine-flux-1 interface faces (* 0.5 dt)))

(defn- arguments []
  [(double-array 4) (double-array 4) (double-array 4)
   (double-array [-0.0 4.0 10.0 8.0]) (double-array [2.0 -0.0 14.0 24.0])
   (double-array [91.0 92.0 93.0 -0.0]) (int-array [1 1 1 0]) 4
   ;; The excluded face deliberately contains a sentinel: skipped accumulation must not
   ;; read its flux or alter the existing register value.
   (double-array [0.0 0.0 0.0 117.0])
   (double-array [3.0 -4.0 5.0 Double/NaN])
   (double-array [1.0 -2.0 8.0 Double/NaN])
   (double-array [2.0 -1.0 9.0 Double/NaN])
   (int-array [1 1 1 0]) 4 0.125])

(defn- bits [values] (mapv #(Double/doubleToRawLongBits (double %)) values))

(defn- check-values! [values]
  (is (= (bits [-0.0 4.0 10.0 -0.0]) (bits (:at-old values))))
  (is (= (bits [1.0 2.0 12.0 -0.0]) (bits (:at-mid values))))
  (is (= (bits [2.0 -0.0 14.0 -0.0]) (bits (:at-new values))))
  ;; Face-integrated rates are weighted by time once, never multiplied by area again.
  (is (= [-0.1875 0.3125 0.4375 117.0] (vec (:register values)))))

(deftest temporal-endpoints-and-signed-transport-on-jvm
  (let [args (arguments)]
    (apply temporal-and-transport! args)
    (check-values! (zipmap [:at-old :at-mid :at-new :register]
                           (map #(nth args %) [0 1 2 8])))))

(defn- run-device! [target]
  (with-redefs [build/current-identity #'producer/test-build]
    (let [prepared (compiled/lower #'temporal-and-transport! (arguments)
                                   {:target target :compiler :equation-first :dtype :double
                                    :inline? true :outputs '[at-old at-mid at-new register]})
          c (compiled/instantiate! prepared)]
      (try
        (with-open [receipt (compiled/invoke-with-evidence c {})]
          (check-values! (into {} (map (fn [{:keys [key node]}]
                                         [key (link/download (:executable c) node)]))
                               (:out-tree prepared)))
          (is (= 4 (count (:outputs @receipt)))))
        (finally (compiled/close! c))))))

(deftest temporal-and-transport-on-opencl
  (if @opencl/opencl-fp64-available?
    (run-device! :ocl:0)
    (opencl/opencl-skip! "FP64 temporal interpolation and interface transport" :fp64)))

(deftest temporal-and-transport-on-level-zero
  (if @ze/gpu-available?
    (run-device! :ze:0)
    (ze/gpu-skip! "FP64 temporal interpolation and interface transport")))
