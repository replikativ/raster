(ns raster.render.shader-test
  "The one-source shader emitter (raster.render.shader): GLSL for Vulkan, WGSL for
   WebGPU, from a single description. Guards the surface the renderer backends rely
   on — uniform layout/stages, GLSL push-constant + gl_Position, WGSL uniform buffer
   + the WebGPU Y-flip."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.render.shader :as s]))

(def spec
  '{:uniform    {:name "U" :fields [[:viewport :vec4]]}
    :attributes [[inPos vec2 0] [inColor vec3 1]]
    :varyings   [[vColor vec3 0]]
    :vertex     [(let ndc vec2 (- (* (/ inPos (swizzle viewport xy)) 2.0) 1.0))
                 (set-position (vec4 (swizzle ndc x) (swizzle ndc y) 0.0 1.0))
                 (out vColor inColor)]
    :fragment   [(color (vec4 vColor 1.0))]})

(deftest uniform-layout
  (is (= 16 (s/uniform-bytes spec)))
  (is (= #{:vertex} (s/uniform-stages spec)))
  (is (= 0 (s/uniform-bytes (dissoc spec :uniform)))))

(deftest glsl-emit
  (let [{:keys [vert frag]} (s/->glsl spec)]
    (is (str/includes? vert "#version 450"))
    (is (str/includes? vert "layout(push_constant) uniform U"))
    (is (str/includes? vert "layout(location=0) in vec2 inPos;"))
    (is (str/includes? vert "gl_Position = vec4(ndc.x, ndc.y, 0.0, 1.0);"))
    (is (str/includes? vert "u.viewport.xy"))
    (is (str/includes? frag "outColor = vec4(vColor, 1.0);"))))

(deftest wgsl-emit
  (let [{:keys [vert frag]} (s/->wgsl spec)]
    (is (str/includes? vert "var<uniform> u: U"))
    (is (str/includes? vert "@location(0) inPos: vec2<f32>"))
    (is (str/includes? vert "vec4<f32>(ndc.x, ndc.y, 0.0, 1.0)"))
    ;; WebGPU NDC is Y-up: the emitter flips Y, the author does not
    (is (str/includes? vert "out.position = vec4<f32>(_p.x, -_p.y, _p.z, _p.w);"))
    (is (str/includes? frag "return vec4<f32>(vColor, 1.0);"))))

(def tex-spec
  '{:uniform    {:name "U" :fields [[:mvp :mat4]]}
    :attributes [[inPos vec3 0] [inUv vec2 1]]
    :varyings   [[vUv vec2 0]]
    :textures   [[tex 0]]
    :vertex     [(set-position (* mvp (vec4 inPos 1.0))) (out vUv inUv)]
    :fragment   [(color (sample tex vUv))]})

(deftest textured-shader-emit
  (testing "R4: texture decls + sample op + mat4 uniform, GLSL and WGSL"
    (is (= 64 (s/uniform-bytes tex-spec)))           ; mat4
    (let [{gv :vert gf :frag} (s/->glsl tex-spec)]
      (is (str/includes? gv "uniform U {\n  mat4 mvp;"))
      (is (str/includes? gv "(u.mvp * vec4(inPos, 1.0))"))
      (is (str/includes? gf "layout(set=0, binding=0) uniform sampler2D tex;"))
      (is (str/includes? gf "texture(tex, vUv)")))
    (let [{wv :vert wf :frag} (s/->wgsl tex-spec)]
      (is (str/includes? wv "mvp: mat4x4<f32>"))
      ;; texture decls only in the stage that samples (fragment), not vertex
      (is (not (str/includes? wv "texture_2d")))
      (is (str/includes? wf "@group(1) @binding(0) var tex: texture_2d<f32>;"))
      (is (str/includes? wf "@group(1) @binding(1) var tex_s: sampler;"))
      (is (str/includes? wf "textureSample(tex, tex_s, vUv)")))))

(def arr-spec
  '{:uniform        {:name "U" :fields [[:mvp :mat4]]}
    :attributes     [[inPos vec3 0] [inUv vec2 1] [inLayer float 2]]
    :varyings       [[vUv vec2 0] [vLayer float 1]]
    :texture-arrays [[atlas 0]]
    :vertex         [(set-position (* mvp (vec4 inPos 1.0))) (out vUv inUv) (out vLayer inLayer)]
    :fragment       [(color (sample-layer atlas vUv vLayer))]})

(deftest texture-array-shader-emit
  (testing "R5a: sampler2DArray / texture_2d_array + sample-layer, GLSL and WGSL"
    (let [{_ :vert gf :frag} (s/->glsl arr-spec)]
      (is (str/includes? gf "uniform sampler2DArray atlas;"))
      (is (str/includes? gf "texture(atlas, vec3(vUv, vLayer))")))
    (let [{_ :vert wf :frag} (s/->wgsl arr-spec)]
      (is (str/includes? wf "var atlas: texture_2d_array<f32>;"))
      (is (str/includes? wf "var atlas_s: sampler;"))
      (is (str/includes? wf "textureSample(atlas, atlas_s, vUv, i32(vLayer))")))))
