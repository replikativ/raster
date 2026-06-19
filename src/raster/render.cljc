(ns raster.render
  "Portable renderer abstraction — the neutral surface that makes raster's Vulkan
   engine browser-portable.

   Game render code is written ONCE against these protocols and runs on both the
   JVM (raster.render.vulkan, wrapping raster.vk.* / LWJGL Vulkan 1.3) and the
   browser (raster.render.webgpu, WebGPU). The two backends are the only
   per-platform code; pipelines, meshes, uniforms and draws are described in a
   backend-neutral vocabulary here.

   A pipeline spec carries BOTH shader sources (GLSL for Vulkan, WGSL for WebGPU);
   each backend picks its own. Vertex attribute formats are neutral keywords
   (:float2 :float3 :float4). Uniforms are a flat block of floats (the Vulkan
   backend pushes them as push-constants; WebGPU emulates with a uniform buffer +
   bind group — WebGPU has no push-constants).

   Conventions a backend must honor:
     - screen-space input: the game feeds pixel coordinates; the shader maps them
       to NDC using the `viewport` uniform. (Y handedness differs between Vulkan
       and WebGPU — handled in each backend's shader source, not here.)
     - a dynamic mesh is a reusable GPU vertex buffer rewritten each frame."
  (:refer-clojure :exclude [run!]))

;; --- vocabulary ---------------------------------------------------------------
(def format-floats {:float2 2 :float3 3 :float4 4})
(defn format-bytes [fmt] (* 4 (get format-floats fmt)))

;; --- protocols ----------------------------------------------------------------
(defprotocol Renderer
  "A GPU device + surface. Creates resources and drives the frame loop."
  (-make-pipeline [this spec]
    "spec = {:shader <raster.render.shader description>   ; → GLSL (vk) / WGSL (webgpu)
             :topology :lines|:triangles
             :vertex  {:stride bytes :attributes [{:location n :format kw :offset bytes}]}
             :depth? bool        ; enable depth test/write (3D)
             :cull   :none|:back|:front
             :blend? bool}")
  (-make-mesh [this max-verts stride]
    "A dynamic vertex-only mesh: a GPU buffer of max-verts * stride bytes,
     rewritten each frame via -upload-mesh!.")
  (-upload-mesh! [this mesh verts]
    "Rewrite the mesh's vertices. `verts` is a flat sequence of floats. Returns
     the mesh with :vertex-count set.")
  (-make-static-mesh [this verts indices stride]
    "An immutable indexed mesh: `verts` a flat float seq, `indices` a flat int seq.
     Returns a mesh with :index-count, drawn via -draw-indexed!.")
  (-make-texture [this opts]
    "A 2D texture + sampler. opts = {:width :height :pixels <flat RGBA8 byte/int
     seq, row-major> :filter :nearest|:linear}. Returns an opaque texture handle.")
  (-run! [this opts]
    "Drive the game loop. opts = {:init-state s :update (fn [state input dt] s')
     :render (fn [state frame]) :clear-color [r g b a] :title str}. `input` is a
     set of active key actions (#{:left :right :up :fire}); `frame` is a Frame."))

(defprotocol Frame
  "A per-frame command recorder handed to the :render callback."
  (-bind-pipeline! [frame pipeline])
  (-set-uniform! [frame pipeline floats]
    "Set the pipeline's uniform/push block from a flat seq of floats.")
  (-bind-mesh! [frame mesh])
  (-bind-texture! [frame pipeline texture]
    "Bind a texture (+ sampler) for the pipeline's sampler binding.")
  (-draw! [frame vertex-count first-vertex])
  (-draw-indexed! [frame index-count first-index]))

;; --- public API (so call sites avoid the dashed protocol fns) -----------------
(defn make-pipeline    [r spec]              (-make-pipeline r spec))
(defn make-mesh        [r max-verts stride]  (-make-mesh r max-verts stride))
(defn upload-mesh!     [r mesh verts]        (-upload-mesh! r mesh verts))
(defn make-static-mesh [r verts indices stride] (-make-static-mesh r verts indices stride))
(defn make-texture     [r opts]              (-make-texture r opts))
(defn run!             [r opts]              (-run! r opts))

(defn bind-pipeline! [f p]            (-bind-pipeline! f p))
(defn set-uniform!   [f p floats]     (-set-uniform! f p floats))
(defn bind-mesh!     [f m]            (-bind-mesh! f m))
(defn bind-texture!  [f p t]          (-bind-texture! f p t))
(defn draw!          [f n first-vert] (-draw! f n first-vert))
(defn draw-indexed!  [f n first-idx]  (-draw-indexed! f n first-idx))
