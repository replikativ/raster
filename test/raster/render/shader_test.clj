(ns raster.render.shader-test
  "The one-source shader emitter (raster.render.shader): GLSL for Vulkan, WGSL for
   WebGPU, from a single description. Guards the surface the renderer backends rely
   on — uniform layout/stages, GLSL push-constant + gl_Position, WGSL uniform buffer
   + the WebGPU Y-flip."
  (:require [clojure.test :refer [deftest is]]
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
