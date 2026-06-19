(ns sandbox.webgpu-compute
  "Experiment 2 — WebGPU compute, the f32-on-GPU path raster's GPU backend must
   target (WGSL has no f64). A saxpy kernel  y[i] = a*x[i] + y[i]  emitted as
   WGSL, dispatched as a compute shader, read back, and checked against a CPU
   reference. This is the shape every raster par/map! kernel takes on WebGPU.

   It also exercises the binding/workgroup conventions that a future
   `wgsl-config` in raster's c_emit would generate."
  (:require [clojure.string :as str]))

(def saxpy-wgsl
  "
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read_write> y : array<f32>;
@group(0) @binding(2) var<uniform>             params : vec4<f32>; // .x = a, .y = n

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(params.y)) { return; }
  y[i] = params.x * x[i] + y[i];
}")

(defn log! [s]
  (when-let [el (.getElementById js/document "log")]
    (set! (.-textContent el) (str (.-textContent el) "\n" s)))
  (js/console.log s))

(defn ^:async run! []
  (if-not (.-gpu js/navigator)
    (log! "WebGPU not available (navigator.gpu is undefined). Use Chrome/Edge/Safari 26, or Firefox Nightly on Linux.")
    (-> (js/Promise.resolve)
        (.then
         (fn []
           (let [n      1024
                 a      2.0
                 x      (js/Float32Array. n)
                 y      (js/Float32Array. n)]
             (dotimes [i n] (aset x i (js/Math.fround i))
                            (aset y i (js/Math.fround (* 0.5 i))))
             ;; CPU reference
             (let [ref (js/Float32Array. n)]
               (dotimes [i n] (aset ref i (js/Math.fround (+ (* a (aget x i)) (aget y i)))))
               (-> (.requestAdapter (.-gpu js/navigator))
                   (.then
                    (fn [adapter]
                      (if-not adapter
                        (log! "requestAdapter returned null — no compatible GPU adapter.")
                        (-> (.requestDevice adapter)
                            (.then
                             (fn [device]
                               (let [usage   (.-GPUBufferUsage js/window)
                                     STORAGE (bit-or js/GPUBufferUsage.STORAGE js/GPUBufferUsage.COPY_SRC js/GPUBufferUsage.COPY_DST)
                                     xbuf (.createBuffer device #js {:size (* 4 n) :usage STORAGE})
                                     ybuf (.createBuffer device #js {:size (* 4 n) :usage STORAGE})
                                     ubuf (.createBuffer device #js {:size 16 :usage (bit-or js/GPUBufferUsage.UNIFORM js/GPUBufferUsage.COPY_DST)})
                                     rbuf (.createBuffer device #js {:size (* 4 n) :usage (bit-or js/GPUBufferUsage.COPY_DST js/GPUBufferUsage.MAP_READ)})]
                                 (.writeBuffer (.-queue device) xbuf 0 x)
                                 (.writeBuffer (.-queue device) ybuf 0 y)
                                 (.writeBuffer (.-queue device) ubuf 0 (js/Float32Array. #js [a n 0 0]))
                                 (let [module (.createShaderModule device #js {:code saxpy-wgsl})
                                       pipeline (.createComputePipeline device
                                                  #js {:layout "auto"
                                                       :compute #js {:module module :entryPoint "main"}})
                                       bind (.createBindGroup device
                                              #js {:layout (.getBindGroupLayout pipeline 0)
                                                   :entries #js [#js {:binding 0 :resource #js {:buffer xbuf}}
                                                                 #js {:binding 1 :resource #js {:buffer ybuf}}
                                                                 #js {:binding 2 :resource #js {:buffer ubuf}}]})
                                       enc  (.createCommandEncoder device)
                                       pass (.beginComputePass enc)]
                                   (.setPipeline pass pipeline)
                                   (.setBindGroup pass 0 bind)
                                   (.dispatchWorkgroups pass (js/Math.ceil (/ n 64)))
                                   (.end pass)
                                   (.copyBufferToBuffer enc ybuf 0 rbuf 0 (* 4 n))
                                   (.submit (.-queue device) #js [(.finish enc)])
                                   (-> (.mapAsync rbuf js/GPUMapMode.READ)
                                       (.then
                                        (fn [_]
                                          (let [got (js/Float32Array. (.getMappedRange rbuf))
                                                max-err (reduce (fn [m i] (max m (js/Math.abs (- (aget got i) (aget ref i)))))
                                                                0 (range n))]
                                            (log! (str "WebGPU saxpy done. n=" n
                                                       "  max|gpu-cpu|=" max-err
                                                       "  => " (if (< max-err 1e-4) "PASS" "FAIL")))
                                            (log! (str "  sample: y[10]=" (aget got 10) " (ref " (aget ref 10) ")"))
                                            (.unmap rbuf))))))))))))))))))
        (.catch (fn [e] (log! (str "WebGPU error: " e)))))))
