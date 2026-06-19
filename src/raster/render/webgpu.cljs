(ns raster.render.webgpu
  "WebGPU/browser backend for the portable renderer protocol (raster.render).

   The browser counterpart of raster.render.vulkan: the SAME neutral pipeline
   spec, mesh and run-loop, implemented on WebGPU. Game render code written against
   raster.render runs unchanged here. WebGPU maps closely to the engine's Vulkan
   1.3 model (single queue, render pass = load/store, implicit resource state) —
   the two differences the abstraction has to absorb are handled here:
     - no push-constants → uniforms become a uniform buffer + bind group
     - NDC Y is up (vs Vulkan Y-down) → handled in the WGSL shader source, not here

   All host interop goes through applied-science.js-interop (string-keyed, so it
   survives :advanced renaming); descriptors are #js literals (string keys too)."
  (:require [raster.render :as r]
            [applied-science.js-interop :as j]))

(declare run-loop! make-pipeline* make-mesh* upload-mesh*)

(def ^:private vfmt {:float2 "float32x2" :float3 "float32x3" :float4 "float32x4"})
(def ^:private vtopo {:lines "line-list" :triangles "triangle-list"})

(defn- buf-usage [& ks]
  (reduce (fn [acc k] (bit-or acc (j/get js/GPUBufferUsage k))) 0 ks))

;; --- per-frame command recorder ----------------------------------------------
(defrecord GpuFrame [pass device]
  r/Frame
  (-bind-pipeline! [_ pipeline]
    (j/call pass :setPipeline (:pipeline pipeline))
    (j/call pass :setBindGroup 0 (:bind-group pipeline)))
  (-set-uniform! [_ pipeline floats]
    (j/call (j/get device :queue) :writeBuffer
            (:uniform-buf pipeline) 0 (js/Float32Array. (clj->js (vec floats)))))
  (-bind-mesh! [_ mesh]
    (j/call pass :setVertexBuffer 0 (:buf mesh)))
  (-draw! [_ vertex-count first-vertex]
    (j/call pass :draw vertex-count 1 first-vertex 0)))

;; --- the renderer ------------------------------------------------------------
(defrecord WebGPURenderer [device context format]
  r/Renderer
  (-make-pipeline [_ spec] (make-pipeline* device format spec))
  (-make-mesh [_ max-verts stride] (make-mesh* device max-verts stride))
  (-upload-mesh! [_ mesh verts] (upload-mesh* device mesh verts))
  (-run! [this opts] (run-loop! this opts)))

(defn- make-pipeline* [device fmt spec]
  (let [{:keys [shaders topology vertex uniform-size]} spec
        wgsl (:wgsl shaders)
        vs (j/call device :createShaderModule #js {:code (:vert wgsl)})
        fs (j/call device :createShaderModule #js {:code (:frag wgsl)})
        attrs (apply array (map (fn [{:keys [location format offset]}]
                                  #js {:shaderLocation location :format (vfmt format) :offset offset})
                                (:attributes vertex)))
        pipeline (j/call device :createRenderPipeline
                         #js {:layout "auto"
                              :vertex #js {:module vs :entryPoint "main"
                                           :buffers #js [#js {:arrayStride (:stride vertex)
                                                              :attributes attrs}]}
                              :fragment #js {:module fs :entryPoint "main"
                                             :targets #js [#js {:format fmt}]}
                              :primitive #js {:topology (vtopo topology)}})
        ubuf (j/call device :createBuffer
                     #js {:size (max 16 (or uniform-size 16))
                          :usage (buf-usage :UNIFORM :COPY_DST)})
        bg (j/call device :createBindGroup
                   #js {:layout (j/call pipeline :getBindGroupLayout 0)
                        :entries #js [#js {:binding 0 :resource #js {:buffer ubuf}}]})]
    {:pipeline pipeline :uniform-buf ubuf :bind-group bg}))

(defn- make-mesh* [device max-verts stride]
  {:buf (j/call device :createBuffer
                #js {:size (* max-verts stride) :usage (buf-usage :VERTEX :COPY_DST)})
   :stride stride :vertex-count 0})

(defn- upload-mesh* [device mesh verts]
  (let [arr (js/Float32Array. (clj->js (vec verts)))
        fpv (quot (:stride mesh) 4)]
    (j/call (j/get device :queue) :writeBuffer (:buf mesh) 0 arr)
    (assoc mesh :vertex-count (quot (.-length arr) fpv))))

(def ^:private key->action {"ArrowLeft" :left "ArrowRight" :right "ArrowUp" :up "Space" :fire})

(defn- run-loop! [rnd {:keys [init-state update render clear-color]
                       :or {clear-color [0.0 0.0 0.0 1.0]}}]
  (let [device (:device rnd)
        context (:context rnd)
        state (atom init-state)
        input (atom #{})
        last  (atom (js/performance.now))
        acc   (atom 0.0)
        fixed (/ 1.0 60.0)
        [cr cg cb ca] clear-color]
    (.addEventListener js/document "keydown"
                       (fn [e] (when-let [a (key->action (j/get e :code))]
                                 (swap! input conj a)
                                 (when (= a :fire) (j/call e :preventDefault)))))
    (.addEventListener js/document "keyup"
                       (fn [e] (when-let [a (key->action (j/get e :code))] (swap! input disj a))))
    (letfn [(one-frame [now]
              (let [dt (min 0.1 (/ (- now @last) 1000.0))]
                (reset! last now)
                (swap! acc + dt)
                (loop [] (when (>= @acc fixed)
                           (swap! state #(update % @input fixed))
                           (swap! acc - fixed)
                           (recur)))
                (let [enc  (j/call device :createCommandEncoder)
                      view (j/call (j/call context :getCurrentTexture) :createView)
                      pass (j/call enc :beginRenderPass
                                   #js {:colorAttachments
                                        #js [#js {:view view
                                                  :clearValue #js {:r cr :g cg :b cb :a ca}
                                                  :loadOp "clear" :storeOp "store"}]})]
                  (render @state (->GpuFrame pass device))
                  (j/call pass :end)
                  (j/call (j/get device :queue) :submit #js [(j/call enc :finish)]))
                (js/requestAnimationFrame one-frame)))]
      (js/requestAnimationFrame one-frame))))

(defn make-renderer
  "Async: request adapter/device, configure the canvas' webgpu context.
   Returns a promise of a WebGPURenderer."
  [canvas]
  (let [gpu (j/get js/navigator :gpu)]
    (when-not gpu (throw (js/Error. "WebGPU not available (navigator.gpu missing)")))
    (-> (j/call gpu :requestAdapter)
        (.then (fn [adapter]
                 (when-not adapter (throw (js/Error. "No WebGPU adapter")))
                 (j/call adapter :requestDevice)))
        (.then (fn [device]
                 (let [ctx (j/call canvas :getContext "webgpu")
                       fmt (j/call gpu :getPreferredCanvasFormat)]
                   (j/call ctx :configure #js {:device device :format fmt :alphaMode "opaque"})
                   (->WebGPURenderer device ctx fmt)))))))
