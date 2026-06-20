(ns raster.render.webgpu
  "WebGPU/browser backend for the portable renderer protocol (raster.render).

   The browser counterpart of raster.render.vulkan: the SAME neutral pipeline
   spec, meshes, textures and run-loop, implemented on WebGPU. Game render code
   written against raster.render runs unchanged here. WebGPU maps closely to the
   engine's Vulkan 1.3 model (single queue, render pass = load/store, implicit
   resource state); the differences the abstraction absorbs here:
     - no push-constants → uniforms become a uniform buffer + bind group (group 0)
     - NDC Y is up (vs Vulkan Y-down) → handled in the WGSL shader source
     - textures → group 1 (texture + sampler), derived from the WGSL via auto layout
     - depth → a depth texture + depthStencilAttachment, opt-in via :depth?

   All host interop goes through applied-science.js-interop (string-keyed, so it
   survives :advanced renaming); descriptors are #js literals (string keys too)."
  (:require [raster.render :as r]
            [raster.render.shader :as shader]
            [applied-science.js-interop :as j]))

(declare run-loop! make-pipeline* make-mesh* upload-mesh* make-static-mesh*
         make-texture* make-texture-array* ensure-depth!)

(def ^:private vfmt {:float "float32" :float2 "float32x2" :float3 "float32x3" :float4 "float32x4"})
(def ^:private vtopo {:lines "line-list" :triangles "triangle-list"})
(def ^:private cullmode {:back "back" :front "front" :none "none"})

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
    (j/call pass :setVertexBuffer 0 (or (:vbuf mesh) (:buf mesh)))
    (when (:ibuf mesh) (j/call pass :setIndexBuffer (:ibuf mesh) "uint32")))
  (-bind-texture! [_ pipeline texture]
    ;; build (once, cached) a group-1 bind group against this pipeline's layout
    (let [bg (or @(:bg texture)
                 (reset! (:bg texture)
                         (j/call device :createBindGroup
                                 #js {:layout (j/call (:pipeline pipeline) :getBindGroupLayout 1)
                                      :entries #js [#js {:binding 0 :resource (:view texture)}
                                                    #js {:binding 1 :resource (:sampler texture)}]})))]
      (j/call pass :setBindGroup 1 bg)))
  (-draw! [_ vertex-count first-vertex]
    (j/call pass :draw vertex-count 1 first-vertex 0))
  (-draw-indexed! [_ index-count first-index]
    (j/call pass :drawIndexed index-count 1 first-index 0 0)))

;; --- the renderer ------------------------------------------------------------
(defrecord WebGPURenderer [device context format canvas state]
  r/Renderer
  (-make-pipeline [this spec] (make-pipeline* this spec))
  (-make-mesh [_ max-verts stride] (make-mesh* device max-verts stride))
  (-upload-mesh! [_ mesh verts] (upload-mesh* device mesh verts))
  (-make-static-mesh [_ verts indices stride] (make-static-mesh* device verts indices stride))
  (-make-texture [_ opts] (make-texture* device opts))
  (-make-texture-array [_ opts] (make-texture-array* device opts))
  (-run! [this opts] (run-loop! this opts)))

(defn- ensure-depth!
  "Create (once) a canvas-sized depth texture; record its view + the depth flag."
  [rnd]
  (let [st (:state rnd)]
    (or (:depth-view @st)
        (let [w (j/get (:canvas rnd) :width) h (j/get (:canvas rnd) :height)
              tex (j/call (:device rnd) :createTexture
                          #js {:size #js [w h] :format "depth24plus"
                               :usage (j/get js/GPUTextureUsage :RENDER_ATTACHMENT)})
              v (j/call tex :createView)]
          (swap! st assoc :depth? true :depth-view v)
          v))))

(defn- make-pipeline* [rnd spec]
  (let [device (:device rnd) fmt (:format rnd)
        {:keys [shader topology vertex depth? cull blend?]} spec
        wgsl (shader/->wgsl shader)
        uniform-size (shader/uniform-bytes shader)
        _  (ensure-depth! rnd)   ; always: a depth-attached pass needs every pipeline to declare depthStencil
        vs (j/call device :createShaderModule #js {:code (:vert wgsl)})
        fs (j/call device :createShaderModule #js {:code (:frag wgsl)})
        attrs (apply array (map (fn [{:keys [location format offset]}]
                                  #js {:shaderLocation location :format (vfmt format) :offset offset})
                                (:attributes vertex)))
        target #js {:format fmt}
        _ (when blend?
            (j/assoc! target :blend
                      #js {:color #js {:srcFactor "src-alpha" :dstFactor "one-minus-src-alpha" :operation "add"}
                           :alpha #js {:srcFactor "one" :dstFactor "zero" :operation "add"}}))
        desc #js {:layout "auto"
                  :vertex #js {:module vs :entryPoint "main"
                               :buffers #js [#js {:arrayStride (:stride vertex) :attributes attrs}]}
                  :fragment #js {:module fs :entryPoint "main" :targets #js [target]}
                  :primitive #js {:topology (vtopo topology) :cullMode (cullmode (or cull :none))}}
        ;; always declare depthStencil (the pass always has a depth attachment);
        ;; :depth? false → no write + always-pass (background/overlay like the sky)
        _ (j/assoc! desc :depthStencil
                    #js {:format "depth24plus" :depthWriteEnabled (boolean depth?)
                         :depthCompare (if depth? "less" "always")})
        pipeline (j/call device :createRenderPipeline desc)
        ubuf (j/call device :createBuffer
                     #js {:size (max 16 (or uniform-size 16)) :usage (buf-usage :UNIFORM :COPY_DST)})
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

(defn- make-static-mesh* [device verts indices stride]
  (let [vb (js/Float32Array. (clj->js (vec verts)))
        ib (js/Uint32Array. (clj->js (vec indices)))
        vbuf (j/call device :createBuffer #js {:size (.-byteLength vb) :usage (buf-usage :VERTEX :COPY_DST)})
        ibuf (j/call device :createBuffer #js {:size (.-byteLength ib) :usage (buf-usage :INDEX :COPY_DST)})]
    (j/call (j/get device :queue) :writeBuffer vbuf 0 vb)
    (j/call (j/get device :queue) :writeBuffer ibuf 0 ib)
    {:vbuf vbuf :ibuf ibuf :index-count (count indices) :stride stride}))

(defn- make-texture* [device {:keys [width height pixels filter]}]
  (let [data (js/Uint8Array. (clj->js (vec pixels)))
        tex (j/call device :createTexture
                    #js {:size #js [width height 1] :format "rgba8unorm"
                         :usage (bit-or (j/get js/GPUTextureUsage :TEXTURE_BINDING)
                                        (j/get js/GPUTextureUsage :COPY_DST))})
        f (name (or filter :linear))]
    (j/call (j/get device :queue) :writeTexture
            #js {:texture tex} data
            #js {:bytesPerRow (* width 4) :rowsPerImage height}
            #js [width height 1])
    {:texture tex
     :view (j/call tex :createView)
     :sampler (j/call device :createSampler #js {:magFilter f :minFilter f})
     :bg (atom nil)}))

(defn- make-texture-array* [device {:keys [width height layers pixels filter]}]
  (let [data (js/Uint8Array. (clj->js (vec pixels)))
        tex (j/call device :createTexture
                    #js {:size #js [width height layers] :format "rgba8unorm"
                         :usage (bit-or (j/get js/GPUTextureUsage :TEXTURE_BINDING)
                                        (j/get js/GPUTextureUsage :COPY_DST))})
        f (name (or filter :nearest))]
    ;; layer-major data → one writeTexture covering all array layers
    (j/call (j/get device :queue) :writeTexture
            #js {:texture tex} data
            #js {:bytesPerRow (* width 4) :rowsPerImage height}
            #js [width height layers])
    {:texture tex
     :view (j/call tex :createView #js {:dimension "2d-array"})
     :sampler (j/call device :createSampler #js {:magFilter f :minFilter f})
     :bg (atom nil)}))

(def ^:private key->action
  {"ArrowLeft" :left "ArrowRight" :right "ArrowUp" :up "ArrowDown" :down "Space" :fire
   "KeyW" :w "KeyA" :a "KeyS" :s "KeyD" :d "KeyQ" :q "KeyE" :e})

(defn- run-loop! [rnd {:keys [init-state update render clear-color]
                       :or {clear-color [0.0 0.0 0.0 1.0]}}]
  (let [device (:device rnd)
        context (:context rnd)
        depth-view (:depth-view @(:state rnd))
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
              ;; dt clamped to [0, 0.1]: the lower clamp matters — the first frame's
              ;; rAF timestamp can predate `last` (set after a slow init), giving a
              ;; negative dt that would drive the fixed-step accumulator negative and
              ;; stall updates for seconds.
              (let [dt (max 0.0 (min 0.1 (/ (- now @last) 1000.0)))]
                (reset! last now)
                (swap! acc + dt)
                (loop [] (when (>= @acc fixed)
                           (swap! state #(update % @input fixed))
                           (swap! acc - fixed)
                           (recur)))
                (let [enc  (j/call device :createCommandEncoder)
                      view (j/call (j/call context :getCurrentTexture) :createView)
                      pdesc #js {:colorAttachments
                                 #js [#js {:view view
                                           :clearValue #js {:r cr :g cg :b cb :a ca}
                                           :loadOp "clear" :storeOp "store"}]}
                      _ (when depth-view
                          (j/assoc! pdesc :depthStencilAttachment
                                    #js {:view depth-view :depthClearValue 1.0
                                         :depthLoadOp "clear" :depthStoreOp "store"}))
                      pass (j/call enc :beginRenderPass pdesc)]
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
                   (->WebGPURenderer device ctx fmt canvas (atom {}))))))))
