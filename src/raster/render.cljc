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
  #?(:clj (:refer-clojure :exclude [])))

;; --- vocabulary ---------------------------------------------------------------
(def format-floats {:float2 2 :float3 3 :float4 4})
(defn format-bytes [fmt] (* 4 (get format-floats fmt)))

;; --- protocols ----------------------------------------------------------------
(defprotocol Renderer
  "A GPU device + surface. Creates resources and drives the frame loop."
  (-make-pipeline [this spec]
    "spec = {:shaders {:glsl {:vert s :frag s} :wgsl {:vert s :frag s}}
             :topology :lines|:triangles
             :vertex  {:stride bytes :attributes [{:location n :format kw :offset bytes}]}
             :uniform-size bytes        ; size of the float uniform/push block
             :blend? bool}")
  (-make-mesh [this max-verts stride]
    "A dynamic vertex-only mesh: a GPU buffer of max-verts * stride bytes,
     rewritten each frame via -upload-mesh!.")
  (-upload-mesh! [this mesh verts]
    "Rewrite the mesh's vertices. `verts` is a flat sequence of floats. Returns
     the mesh with :vertex-count set.")
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
  (-draw! [frame vertex-count first-vertex]))

;; --- public API (so call sites avoid the dashed protocol fns) -----------------
(defn make-pipeline [r spec]          (-make-pipeline r spec))
(defn make-mesh     [r max-verts stride] (-make-mesh r max-verts stride))
(defn upload-mesh!  [r mesh verts]    (-upload-mesh! r mesh verts))
(defn run!          [r opts]          (-run! r opts))

(defn bind-pipeline! [f p]            (-bind-pipeline! f p))
(defn set-uniform!   [f p floats]     (-set-uniform! f p floats))
(defn bind-mesh!     [f m]            (-bind-mesh! f m))
(defn draw!          [f n first-vert] (-draw! f n first-vert))
