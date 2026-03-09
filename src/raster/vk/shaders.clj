(ns raster.vk.shaders
  "Reusable GLSL shader library for the Raster Vulkan engine.

  Shader construction follows a functional composition pattern:
  plain Clojure functions return GLSL source strings, parameterized
  by their arguments. This is simpler than a template engine and
  gives full Clojure power for conditional compilation.

  Modules:
    - Math utilities (mat4 helpers, noise, etc.)
    - Vertex shaders (MVP transform, point sprites, etc.)
    - Fragment shaders (flat color, PBR, etc.)
    - GLSL snippets for inclusion in compute shaders

  All shaders target GLSL 450 / Vulkan 1.3."
  (:require [clojure.string :as str]))

;; ================================================================
;; GLSL snippet library (composable fragments)
;; ================================================================

(def glsl-math
  "Common math functions for GLSL shaders."
  "
// --- Raster math library ---
#define PI 3.14159265358979323846
#define TAU 6.28318530717958647692

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3 saturate3(vec3 v) { return clamp(v, vec3(0.0), vec3(1.0)); }

float remap(float val, float lo, float hi, float new_lo, float new_hi) {
    return new_lo + (val - lo) * (new_hi - new_lo) / (hi - lo);
}
")

(def glsl-hash
  "Hash functions for noise/random in shaders."
  "
// --- Hash/noise ---
float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
")

(def glsl-noise
  "Simplex-style value noise (2D)."
  (str glsl-hash "
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
"))

;; ================================================================
;; Vertex shaders
;; ================================================================

(defn vert-mvp
  "Vertex shader: MVP transform with per-vertex color.
  Options:
    :point-size - if set, enables gl_PointSize (for particle rendering)"
  [& {:keys [point-size]}]
  (str "#version 450\n"
       "\n"
       "layout(push_constant) uniform PushConstants {\n"
       "    mat4 mvp;\n"
       "};\n"
       "\n"
       "layout(location = 0) in vec3 inPos;\n"
       "layout(location = 1) in vec3 inColor;\n"
       "\n"
       "layout(location = 0) out vec3 fragColor;\n"
       "\n"
       "void main() {\n"
       "    gl_Position = mvp * vec4(inPos, 1.0);\n"
       "    fragColor = inColor;\n"
       (when point-size
         (str "    gl_PointSize = " (float point-size) ";\n"))
       "}\n"))

(defn vert-mvp-uv
  "Vertex shader: MVP transform with UV coordinates."
  []
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
};

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 fragUV;

void main() {
    gl_Position = mvp * vec4(inPos, 1.0);
    fragUV = inUV;
}
")

(defn vert-particle
  "Vertex shader for point-sprite particles.
  Particle data comes from SSBO (compute-written), not vertex attributes.
  Push constants: mat4 mvp + float point_size."
  []
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    float point_size;
};

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 fragColor;

void main() {
    gl_Position = mvp * vec4(inPos, 1.0);
    gl_PointSize = point_size;
    fragColor = inColor;
}
")

;; ================================================================
;; Fragment shaders
;; ================================================================

(defn frag-color
  "Fragment shader: pass-through vertex color."
  []
  "#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
")

(defn frag-color-alpha
  "Fragment shader: pass-through with alpha from vertex color w component."
  []
  "#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = fragColor;
}
")

(defn frag-point-sprite
  "Fragment shader for round point sprites with alpha falloff."
  []
  "#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r2 = dot(coord, coord);
    if (r2 > 0.25) discard;
    float alpha = fragColor.a * smoothstep(0.25, 0.1, r2);
    outColor = vec4(fragColor.rgb, alpha);
}
")

;; ================================================================
;; Shader composition helpers
;; ================================================================

(defn compose
  "Compose multiple GLSL snippets into a single source string.
  Deduplicates the #version directive."
  [& parts]
  (let [parts (remove nil? parts)
        version-line (some #(re-find #"#version \d+" %) parts)
        stripped (map #(str/replace % #"#version \d+\n?" "") parts)]
    (str (or version-line "#version 450") "\n"
         (str/join "\n" stripped))))
