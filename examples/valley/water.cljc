(ns valley.water
  "Faithful animated water (port of the JVM valley.game water shader to the cross-platform
   shader DSL → GLSL + WGSL). Water faces are meshed into their own per-column sub-mesh
   (pos3 only) and drawn in a transparent pass AFTER the opaque terrain. The fragment shader
   builds an animated normal from a sum of sines, then Fresnel-mixes a deep-water body colour
   with a reflection of the sky gradient; alpha rises with the Fresnel term."
  (:require [clojure.string]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(def ^:const STRIDE 12)   ; pos3 (world position; the shader is fully procedural)

(defn uniform
  "Flat float vector: mvp(16) cam(4: x y z time) topCol(4) botCol(4) — 28 floats."
  [mvp cam-pos t [top bot]]
  (let [[cx cy cz] cam-pos [tr tg tb] top [br bg bb] bot]
    (vec (concat mvp [cx cy cz t] [tr tg tb 0.0] [br bg bb 0.0]))))

(def shader
  '{:uniform   {:name "W" :fields [[:mvp :mat4] [:cam :vec4] [:topCol :vec4] [:botCol :vec4]]}
    :attributes [[inPos vec3 0]]
    :varyings   [[vWorld vec3 0]]
    :vertex     [(set-position (* mvp (vec4 inPos 1.0))) (out vWorld inPos)]
    :fragment   [(let p vec2 (swizzle vWorld xz))
                 (let t float (swizzle cam w))
                 ;; two directional sine waves; d* hold freq·direction so the cos terms are the
                 ;; analytic height gradient → surface normal N
                 (let d1 vec2 (vec2 0.30 0.22))
                 (let d2 vec2 (vec2 (- 0.18) 0.34))
                 (let c1 float (cos (+ (dot p d1) (* t 1.3))))
                 (let c2 float (cos (+ (dot p d2) (* t 0.9))))
                 (let dhx float (+ (* (swizzle d1 x) c1 0.32) (* (swizzle d2 x) c2 0.32)))
                 (let dhz float (+ (* (swizzle d1 y) c1 0.32) (* (swizzle d2 y) c2 0.32)))
                 (let N vec3 (normalize (vec3 (- dhx) 1.0 (- dhz))))
                 (let V vec3 (normalize (- (swizzle cam xyz) vWorld)))
                 (let fres float (clamp (+ 0.04 (* 0.96 (pow (- 1.0 (max (dot N V) 0.0)) 5.0))) 0.0 1.0))
                 (let up float (clamp (+ (* (swizzle N y) 0.5) 0.5) 0.0 1.0))
                 (let refl vec3 (mix (swizzle botCol xyz) (swizzle topCol xyz) up))
                 (let body vec3 (vec3 0.04 0.12 0.19))
                 ;; distance fog → blend toward the horizon colour (matches terrain; 50..76 blocks)
                 (let dist float (length (- (swizzle cam xyz) vWorld)))
                 (let fogf float (clamp (smoothstep 50.0 76.0 dist) 0.0 1.0))
                 (color (vec4 (mix (mix body refl fres) (swizzle botCol xyz) fogf)
                              (mix (mix 0.78 0.95 fres) 1.0 fogf)))]})

(def pipeline-spec
  ;; depth test on (terrain in front occludes water) + write on; blended. Drawn after opaque,
  ;; so terrain under the surface is already in the depth/colour buffer and shows through alpha.
  {:shader shader :topology :triangles :depth? true :cull :none :blend? true
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float3 :offset 0}]}})
