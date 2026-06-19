(ns raster.compiler.backend.gpu.wgsl-test
  "Track C — WGSL compute emitter (WebGPU, f32). Structural validation of the
   generated shader (bindings, entry point, registry-lowered body). Execution on
   a real WebGPU adapter is validated out-of-suite via wgpu-py over Vulkan
   (cljs-sandbox/webgpu/wgsl_validate.py) — vadd + saxpy bit-exact vs an f64 ref."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.core :refer [deftm]]
            [raster.numeric]
            [raster.arrays]
            [raster.compiler.pipeline :as pl]))

(deftm wgsl-vadd! [a :- (Array float), b :- (Array float), out :- (Array float), n :- Long] :- nil
  (dotimes [i n]
    (raster.arrays/aset out i (raster.numeric/+ (raster.arrays/aget a i) (raster.arrays/aget b i)))))

(deftm wgsl-saxpy! [s :- Float, x :- (Array float), y :- (Array float), out :- (Array float), n :- Long] :- nil
  (dotimes [i n]
    (raster.arrays/aset out i (raster.numeric/+ (raster.numeric/* s (raster.arrays/aget x i))
                                                (raster.arrays/aget y i)))))

(deftest vadd-wgsl-structure
  (testing "elementwise f32 map → WGSL compute shader"
    (let [{:keys [wgsl array-params scalar-params n-sym]} (pl/compile-wgsl #'wgsl-vadd!)]
      (is (= '[a b out] array-params))
      (is (= [] scalar-params))
      (is (= 'n n-sym))
      ;; storage buffer bindings, read vs read_write
      (is (str/includes? wgsl "@group(0) @binding(0) var<storage, read> a: array<f32>;"))
      (is (str/includes? wgsl "@group(0) @binding(2) var<storage, read_write> out: array<f32>;"))
      ;; uniform with the element count
      (is (str/includes? wgsl "_n: u32,"))
      ;; one-element-per-invocation guard + registry-lowered body
      (is (str/includes? wgsl "@compute @workgroup_size(64)"))
      (is (str/includes? wgsl "let gi: u32 = g.x;"))
      (is (str/includes? wgsl "if (gi < U._n)"))
      (is (str/includes? wgsl "out[gi] = "))
      (is (str/includes? wgsl "a[gi] + b[gi]")))))

(deftest saxpy-wgsl-scalar-broadcast
  (testing "scalar param becomes a uniform field referenced as U.<name>"
    (let [{:keys [wgsl scalar-params n-sym]} (pl/compile-wgsl #'wgsl-saxpy!)]
      (is (= '[s] scalar-params))
      (is (= 'n n-sym))
      (is (str/includes? wgsl "s: f32,"))
      (is (str/includes? wgsl "(U.s * x[gi])"))
      (is (str/includes? wgsl "out[gi] = ")))))
