(ns valley.game
  "Living Valley — voxel world with procedural terrain.
  Uses Raster Vulkan engine for rendering."
  (:require
   [valley.world :as w]
   [valley.terrain :as terrain]
   [valley.mesher :as mesher]
   [valley.textures :as textures]
   [valley.lighting :as lighting]
   [valley.entity :as entity]
   [valley.hud :as hud]
   [valley.items :as items]
   [valley.inventory :as inventory]
   [valley.mobs :as mobs]
   [valley.hostile :as hostile]
   [valley.core :as tc]
   [valley.state :as vs]
   [valley.mob-models :as mob-models]
   [valley.hunger :as hunger]
   [valley.farming :as farming]
   [valley.persistence :as persist]
   [raster.vk.gpu :as gpu]
   [raster.vk.input :as input]
   [raster.vk.frame :as frame]
   [raster.vk.math :as math]
   [raster.vk.mesh :as mesh]
   [raster.vk.pipeline :as pipeline]
   [raster.vk.texture :as texture]
   [raster.vk.camera :as camera]
   [raster.vk.shader :as shader])
  (:import
   [org.lwjgl.vulkan VK13 VkCommandBuffer]
   [org.lwjgl.system MemoryStack]
   [org.lwjgl.glfw GLFW]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Frame profiler — REPL-queryable
;; ================================================================

(defonce ^:private frame-profile (atom {}))
(defonce ^:private slow-frames (atom []))

(defn prof-reset!
  "Reset profiler state."
  []
  (reset! frame-profile {})
  (reset! slow-frames []))

(defn prof-report
  "Return profiler summary: {:phase {:count N :total-ms X :max-ms Y} ...}
  Also shows recent slow frames via (prof-slow-frames)."
  []
  @frame-profile)

(defn prof-slow-frames
  "Return last N slow frames (>16ms). Each is {:frame N :total-ms X :phases {...}}."
  ([] (prof-slow-frames 10))
  ([n] (take-last n @slow-frames)))

(defn- prof-record!
  "Record elapsed time for a phase."
  [phase-key ^long elapsed-ns]
  (let [ms (/ (double elapsed-ns) 1e6)]
    (swap! frame-profile update phase-key
      (fn [old]
        (let [{:keys [count total-ms max-ms] :or {count 0 total-ms 0.0 max-ms 0.0}} old]
          {:count (inc count)
           :total-ms (+ total-ms ms)
           :max-ms (max max-ms ms)})))))

(defmacro with-prof
  "Profile a phase: (with-prof :physics body...) → result of body."
  [phase-key & body]
  `(let [t0# (System/nanoTime)
         result# (do ~@body)
         elapsed# (- (System/nanoTime) t0#)]
     (prof-record! ~phase-key elapsed#)
     result#))

;; ================================================================
;; Constants
;; ================================================================

(def ^:const W 1280)
(def ^:const H 720)
(def ^:const MOVE-SPEED 30.0)    ;; blocks/sec
(def ^:const MOUSE-SENSITIVITY 0.15)
(def ^:const GRAVITY 32.0)       ;; blocks/sec^2
(def ^:const JUMP-VELOCITY 9.0)  ;; blocks/sec
(def ^:const PLAYER-HEIGHT 1.7)  ;; eye height in blocks
(def ^:const PLAYER-RADIUS 0.3)
(def ^:const LOAD-RADIUS 5)     ;; chunk radius to keep loaded
(def ^:const UNLOAD-RADIUS 9)   ;; chunk radius beyond which to unload (> sqrt(2)*LOAD-RADIUS)
(def ^:const HEIGHT-CHUNKS 6)   ;; vertical chunk layers
(def ^:const STREAM-INTERVAL 5)  ;; frames between streaming checks (cheap — gen is background)

;; Day/night cycle
(def ^:const TIME-SPEED 72.0)    ;; game ticks per real second (20 min real = 1 game day)
(def ^:const DAY-LENGTH 24000.0) ;; ticks per game day

;; Day/night ratio keyframes: [time ratio]
;; 0=midnight, 6000=dawn, 12000=noon, 18000=dusk
(def day-night-keyframes
  [[0     0.05]
   [4500  0.05]
   [5000  0.15]
   [5500  0.35]
   [6000  0.70]
   [6500  1.00]
   [17500 1.00]
   [18000 0.70]
   [18500 0.35]
   [19000 0.15]
   [19500 0.05]
   [24000 0.05]])

;; Sky top color keyframes: [time [r g b]]
(def sky-top-keyframes
  [[0     [0.02 0.02 0.08]]
   [4500  [0.02 0.02 0.08]]
   [5000  [0.15 0.10 0.25]]
   [5500  [0.20 0.15 0.40]]
   [6000  [0.25 0.35 0.70]]
   [6500  [0.30 0.50 0.90]]
   [17500 [0.30 0.50 0.90]]
   [18000 [0.25 0.30 0.60]]
   [18500 [0.20 0.15 0.40]]
   [19000 [0.10 0.08 0.25]]
   [19500 [0.02 0.02 0.08]]
   [24000 [0.02 0.02 0.08]]])

;; Sky bottom/horizon color keyframes: [time [r g b]]
(def sky-bot-keyframes
  [[0     [0.05 0.05 0.15]]
   [4500  [0.05 0.05 0.15]]
   [5000  [0.30 0.20 0.35]]
   [5500  [0.70 0.40 0.30]]
   [6000  [0.90 0.60 0.40]]
   [6500  [0.70 0.80 1.00]]
   [17500 [0.70 0.80 1.00]]
   [18000 [0.90 0.50 0.30]]
   [18500 [0.80 0.35 0.20]]
   [19000 [0.30 0.15 0.20]]
   [19500 [0.05 0.05 0.15]]
   [24000 [0.05 0.05 0.15]]])

(defn lerp-keyframes
  "Linearly interpolate a value from keyframes [[time val] ...] at given time.
  val can be a number or a vector of numbers."
  [keyframes ^double t]
  (let [t (mod t DAY-LENGTH)
        n (count keyframes)]
    (loop [i 0]
      (if (>= i (dec n))
        (second (nth keyframes (dec n)))
        (let [[t0 v0] (nth keyframes i)
              [t1 v1] (nth keyframes (inc i))
              t0 (double t0) t1 (double t1)]
          (if (and (>= t t0) (< t t1))
            (let [frac (/ (- t t0) (- t1 t0))]
              (if (number? v0)
                (+ (* (- 1.0 frac) (double v0)) (* frac (double v1)))
                (mapv (fn [a b] (+ (* (- 1.0 frac) (double a)) (* frac (double b)))) v0 v1)))
            (recur (inc i))))))))

(defn get-day-night-ratio
  "Get day/night ratio (0.0-1.0) for given game time."
  ^double [^double game-time]
  (double (lerp-keyframes day-night-keyframes game-time)))

(defn get-sky-colors
  "Get [top-color bot-color] for given game time. Each is [r g b]."
  [^double game-time]
  [(lerp-keyframes sky-top-keyframes game-time)
   (lerp-keyframes sky-bot-keyframes game-time)])

;; ================================================================
;; Shaders (reuse Doom format — textured with texture array)
;; ================================================================

(def vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float dayNightRatio;
    float gameTime;
    float skyYaw;
    float skyPitch;
    float skyTopR, skyTopG, skyTopB;
    float skyBotR, skyBotG, skyBotB;
    float camX, camY, camZ;
};

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in float inDayLight;
layout(location = 3) in float inNightLight;
layout(location = 4) in float inTexLayer;
layout(location = 5) in float inAO;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float fragDayLight;
layout(location = 2) out float fragNightLight;
layout(location = 3) out float fragTexLayer;
layout(location = 4) out float fragDist;
layout(location = 5) out float fragAO;

void main() {
    vec4 clipPos = vp * vec4(inPos, 1.0);
    gl_Position = clipPos;
    fragUV = inUV;
    fragDayLight = inDayLight;
    fragNightLight = inNightLight;
    fragTexLayer = inTexLayer;
    fragDist = length(clipPos.xyz);
    fragAO = inAO;
}
")

(def frag-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float dayNightRatio;
    float gameTime;
    float skyYaw;
    float skyPitch;
    float skyTopR, skyTopG, skyTopB;
    float skyBotR, skyBotG, skyBotB;
    float camX, camY, camZ;
};

layout(set = 0, binding = 0) uniform sampler2DArray texArray;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float fragDayLight;
layout(location = 2) in float fragNightLight;
layout(location = 3) in float fragTexLayer;
layout(location = 4) in float fragDist;
layout(location = 5) in float fragAO;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 texColor = texture(texArray, vec3(fract(fragUV), fragTexLayer));
    if (texColor.a < 0.1) discard;

    // Blend day/night light based on time of day
    float light = mix(fragNightLight, fragDayLight, dayNightRatio);
    // Apply ambient occlusion
    light *= fragAO;
    // Minimum ambient so caves aren't pitch black
    light = max(light, 0.02);

    vec3 lit = texColor.rgb * light;

    // Distance fog — blend to sky horizon color
    float fogFactor = clamp(fragDist / 300.0, 0.0, 1.0);
    fogFactor = fogFactor * fogFactor;
    vec3 fogColor = vec3(skyBotR, skyBotG, skyBotB);
    outColor = vec4(mix(lit, fogColor, fogFactor), texColor.a);
}
")

;; Sky shader
(def sky-vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float dayNightRatio;
    float gameTime;
    float skyYaw;
    float skyPitch;
    float skyTopR, skyTopG, skyTopB;
    float skyBotR, skyBotG, skyBotB;
    float camX, camY, camZ;
};

layout(location = 0) out vec3 fragDir;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec4 clip = vec4(pos * 2.0 - 1.0, 0.9999, 1.0);
    gl_Position = clip;

    // Reconstruct world-space ray direction from inverse VP
    mat4 invVP = inverse(vp);
    vec4 worldNear = invVP * vec4(clip.xy, -1.0, 1.0);
    vec4 worldFar  = invVP * vec4(clip.xy,  1.0, 1.0);
    fragDir = normalize(worldFar.xyz / worldFar.w - worldNear.xyz / worldNear.w);
}
")

(def sky-frag-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float dayNightRatio;
    float gameTime;
    float skyYaw;
    float skyPitch;
    float skyTopR, skyTopG, skyTopB;
    float skyBotR, skyBotG, skyBotB;
    float camX, camY, camZ;
};

layout(location = 0) in vec3 fragDir;
layout(location = 0) out vec4 outColor;

// Hash for star field — takes a 3D cell coordinate
float starHash(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

void main() {
    vec3 dir = normalize(fragDir);

    // Sky gradient based on world-space elevation
    float elevation = dir.y;  // -1 (down) to +1 (up)
    float t = clamp(elevation * 0.5 + 0.5, 0.0, 1.0);  // 0 at horizon, 1 at zenith
    vec3 topColor = vec3(skyTopR, skyTopG, skyTopB);
    vec3 botColor = vec3(skyBotR, skyBotG, skyBotB);
    vec3 skyColor = mix(botColor, topColor, t);

    // Sun direction in world space — orbits in the XY plane
    float sunPhase = (gameTime / 24000.0) * 6.28318;
    vec3 sunDir = normalize(vec3(sin(sunPhase), -cos(sunPhase), 0.3));
    float sunElevation = sunDir.y;

    // Sun disc + glow via dot product with ray direction
    float sunDot = dot(dir, sunDir);
    float sunVisible = smoothstep(-0.05, 0.05, sunElevation);
    float sun = smoothstep(0.999, 0.9995, sunDot) * sunVisible;  // ~2.5 deg disc
    float glow = pow(max(sunDot, 0.0), 64.0) * 0.6 * sunVisible;
    float softGlow = pow(max(sunDot, 0.0), 8.0) * 0.12 * sunVisible;

    vec3 sunColor = mix(vec3(1.0, 0.6, 0.2), vec3(1.0, 1.0, 0.9),
                        clamp(sunElevation * 2.0, 0.0, 1.0));
    // Boost sun to stand out against bright daytime sky
    float sunBoost = 1.0 + sunElevation * 1.5;  // brighter when high
    sunColor *= sunBoost;

    // Moon — opposite side of sky from sun
    vec3 moonDir = -sunDir;
    float moonDot = dot(dir, moonDir);
    float moonVisible = smoothstep(-0.05, 0.05, moonDir.y);
    float moonDisc = smoothstep(0.9996, 0.9999, moonDot) * moonVisible;
    // Crescent: darken one side
    vec3 moonRight = normalize(cross(moonDir, vec3(0.0, 1.0, 0.0)));
    float crescent = smoothstep(-0.3, 0.5, dot(dir - moonDir, moonRight));
    moonDisc *= crescent;
    float moonGlow = pow(max(moonDot, 0.0), 128.0) * 0.08 * moonVisible;

    // Stars — rotate with sun/moon (celestial sphere)
    float starFade = smoothstep(0.3, 0.05, dayNightRatio);
    float stars = 0.0;
    if (starFade > 0.001 && elevation > -0.1) {
        // Rotate direction by sun phase so stars track the celestial rotation
        float cp = cos(-sunPhase); float sp2 = sin(-sunPhase);
        vec3 rotDir = vec3(
            dir.x * cp + dir.y * sp2,
            -dir.x * sp2 + dir.y * cp,
            dir.z);
        vec3 sp = rotDir * 200.0;
        vec3 cell = floor(sp);
        vec3 f = fract(sp);
        // Check if this cell has a star
        float h = starHash(cell);
        if (h > 0.985) {
            // Star brightness varies
            float brightness = (h - 0.985) / 0.015;
            // Point-like: fade toward cell edges
            vec3 center = vec3(0.5);
            float d = length(f - center);
            float point = smoothstep(0.35, 0.1, d);
            // Twinkle
            float twinkle = 0.7 + 0.3 * sin(gameTime * 0.01 + h * 100.0);
            stars = point * brightness * twinkle * starFade * 0.8;
        }
    }

    vec3 color = skyColor
        + sunColor * (sun + glow + softGlow)
        + vec3(0.7, 0.75, 0.9) * (moonDisc * 0.6 + moonGlow)
        + vec3(stars);

    outColor = vec4(color, 1.0);
}
")

;; Water shader — animated normals, Fresnel, sky reflection
(def water-vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float dayNightRatio;
    float gameTime;
    float skyYaw;
    float skyPitch;
    float skyTopR, skyTopG, skyTopB;
    float skyBotR, skyBotG, skyBotB;
    float camX, camY, camZ;
};

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in float inDayLight;
layout(location = 3) in float inNightLight;
layout(location = 4) in float inTexLayer;
layout(location = 5) in float inAO;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out float fragDayLight;
layout(location = 2) out float fragNightLight;
layout(location = 3) out float fragDist;

void main() {
    vec4 clipPos = vp * vec4(inPos, 1.0);
    gl_Position = clipPos;
    fragWorldPos = inPos;
    fragDayLight = inDayLight;
    fragNightLight = inNightLight;
    fragDist = length(clipPos.xyz);
}
")

(def water-frag-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float dayNightRatio;
    float gameTime;
    float skyYaw;
    float skyPitch;
    float skyTopR, skyTopG, skyTopB;
    float skyBotR, skyBotG, skyBotB;
    float camX, camY, camZ;
};

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in float fragDayLight;
layout(location = 2) in float fragNightLight;
layout(location = 3) in float fragDist;
layout(location = 0) out vec4 outColor;

// Sum-of-sines wave function for animated water normals
vec3 waterNormal(vec2 pos, float time) {
    // Multiple wave octaves at different frequencies and directions
    float t = time * 0.5;
    float h = 0.0;
    float dx = 0.0;
    float dz = 0.0;

    // Wave 1: large slow
    float f1 = 0.4;
    float a1 = 0.08;
    vec2 d1 = vec2(0.6, 0.8);
    float phase1 = dot(d1, pos) * f1 + t * 1.2;
    h += a1 * sin(phase1);
    dx += a1 * f1 * d1.x * cos(phase1);
    dz += a1 * f1 * d1.y * cos(phase1);

    // Wave 2: medium
    float f2 = 0.9;
    float a2 = 0.04;
    vec2 d2 = vec2(-0.4, 0.9);
    float phase2 = dot(d2, pos) * f2 + t * 2.0;
    h += a2 * sin(phase2);
    dx += a2 * f2 * d2.x * cos(phase2);
    dz += a2 * f2 * d2.y * cos(phase2);

    // Wave 3: small ripples
    float f3 = 2.2;
    float a3 = 0.015;
    vec2 d3 = vec2(0.8, -0.6);
    float phase3 = dot(d3, pos) * f3 + t * 3.5;
    h += a3 * sin(phase3);
    dx += a3 * f3 * d3.x * cos(phase3);
    dz += a3 * f3 * d3.y * cos(phase3);

    // Wave 4: tiny detail
    float f4 = 4.0;
    float a4 = 0.008;
    vec2 d4 = vec2(-0.7, -0.7);
    float phase4 = dot(d4, pos) * f4 + t * 5.0;
    h += a4 * sin(phase4);
    dx += a4 * f4 * d4.x * cos(phase4);
    dz += a4 * f4 * d4.y * cos(phase4);

    return normalize(vec3(-dx, 1.0, -dz));
}

void main() {
    float time = gameTime / 72.0;  // Convert game ticks to seconds

    // Animated normal
    vec3 N = waterNormal(fragWorldPos.xz, time);

    // Per-fragment view direction from camera to fragment
    vec3 camPos = vec3(camX, camY, camZ);
    vec3 V = normalize(camPos - fragWorldPos);

    // Fresnel (Schlick approximation)
    float NdotV = max(dot(N, V), 0.0);
    float F0 = 0.02;  // Water IOR ~1.33
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // Sky reflection
    vec3 R = reflect(-V, N);
    float skyT = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 skyTop = vec3(skyTopR, skyTopG, skyTopB);
    vec3 skyBot = vec3(skyBotR, skyBotG, skyBotB);
    vec3 reflectedSky = mix(skyBot, skyTop, skyT);

    // Water body color (blue-green, darker with depth implied)
    vec3 waterColor = vec3(0.05, 0.15, 0.25);

    // Blend day/night light
    float light = mix(fragNightLight, fragDayLight, dayNightRatio);
    light = max(light, 0.02);

    // Final color: mix water body with sky reflection via Fresnel
    vec3 color = mix(waterColor * light, reflectedSky, fresnel);

    // Specular highlight from sun
    float sunPhase = (gameTime / 24000.0) * 6.28318;
    float sunElev = -cos(sunPhase);
    if (sunElev > 0.0) {
        vec3 sunDir = normalize(vec3(sin(sunPhase) * 0.5, sunElev, cos(sunPhase) * 0.5));
        vec3 H = normalize(V + sunDir);
        float spec = pow(max(dot(N, H), 0.0), 128.0);
        color += vec3(1.0, 0.95, 0.8) * spec * sunElev * 0.8;
    }

    // Distance fog
    float fogFactor = clamp(fragDist / 300.0, 0.0, 1.0);
    fogFactor = fogFactor * fogFactor;
    vec3 fogColor = skyBot;
    color = mix(color, fogColor, fogFactor);

    // Semi-transparent
    float alpha = mix(0.6, 0.85, fresnel);

    outColor = vec4(color, alpha);
}
")

;; Crosshair shader
(def crosshair-vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float dayNightRatio;
    float gameTime;
    float skyYaw;
    float skyPitch;
    float skyTopR, skyTopG, skyTopB;
    float skyBotR, skyBotG, skyBotB;
    float camX, camY, camZ;
};

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in float inDayLight;
layout(location = 3) in float inNightLight;
layout(location = 4) in float inTexLayer;
layout(location = 5) in float inAO;

void main() {
    gl_Position = vec4(inPos.xy, 0.0, 1.0);
}
")

(def crosshair-frag-glsl
  "#version 450

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 1.0, 1.0, 0.8);
}
")

;; ================================================================
;; Vertex format helpers
;; ================================================================

(def vertex-bindings
  [{:binding 0 :stride (* 9 4)}])  ;; 9 floats × 4 bytes

(def vertex-attributes
  [{:location 0 :binding 0 :format VK13/VK_FORMAT_R32G32B32_SFLOAT :offset 0}     ;; pos
   {:location 1 :binding 0 :format VK13/VK_FORMAT_R32G32_SFLOAT :offset 12}        ;; uv
   {:location 2 :binding 0 :format VK13/VK_FORMAT_R32_SFLOAT :offset 20}           ;; dayLight
   {:location 3 :binding 0 :format VK13/VK_FORMAT_R32_SFLOAT :offset 24}           ;; nightLight
   {:location 4 :binding 0 :format VK13/VK_FORMAT_R32_SFLOAT :offset 28}           ;; texLayer
   {:location 5 :binding 0 :format VK13/VK_FORMAT_R32_SFLOAT :offset 32}])         ;; ao

;; ================================================================
;; Crosshair mesh
;; ================================================================

(defn make-crosshair-data []
  (let [s 0.015  ;; half-size of crosshair lines
        w 0.003  ;; line width
        ;; 9 floats: pos(3) uv(2) dayLight nightLight texLayer ao
        verts (float-array
                [;; Horizontal bar
                 (- s) (- w) 0  0 0  1 0 0 1
                 (- s) w     0  0 0  1 0 0 1
                 s     w     0  0 0  1 0 0 1
                 s     (- w) 0  0 0  1 0 0 1
                 ;; Vertical bar
                 (- w) (- s) 0  0 0  1 0 0 1
                 (- w) s     0  0 0  1 0 0 1
                 w     s     0  0 0  1 0 0 1
                 w     (- s) 0  0 0  1 0 0 1])
        indices (int-array [0 1 2  0 2 3   4 5 6  4 6 7])]
    {:vertices verts :indices indices}))

;; ================================================================
;; Game state
;; ================================================================

(defonce game-state-ref (atom nil))
(defonce ^:private game-thread (atom nil))
(defonce ^:private game-window (atom nil))
(defonce last-crash (atom nil))  ;; stores last crash info {:time :error :stacktrace}

;; ================================================================
;; Background chunk generation (Minetest-style emerge pattern)
;; ================================================================

(defonce ^:private chunk-gen-executor (atom nil))
(defonce ^:private chunk-gen-pending (atom #{}))   ;; positions in-flight
(defonce ^:private chunk-gen-ready (java.util.concurrent.ConcurrentLinkedQueue.))

(defn- start-chunk-gen-pool! []
  (reset! chunk-gen-pending #{})
  (.clear chunk-gen-ready)
  (reset! chunk-gen-executor
    (java.util.concurrent.Executors/newFixedThreadPool 4)))

(defn- stop-chunk-gen-pool! []
  (when-let [^java.util.concurrent.ExecutorService ex @chunk-gen-executor]
    (.shutdownNow ex)
    (reset! chunk-gen-executor nil))
  (reset! chunk-gen-pending #{})
  (.clear chunk-gen-ready))

(defn- submit-chunk-gen!
  "Submit a chunk position for background generation."
  [pos gen world-path]
  (when-not (contains? @chunk-gen-pending pos)
    (swap! chunk-gen-pending conj pos)
    (let [^java.util.concurrent.ExecutorService ex @chunk-gen-executor]
      (when ex
        (.submit ex ^Runnable
          (fn []
            (try
              (let [[cx cy cz] pos
                    saved (when world-path (persist/load-chunk world-path pos))
                    chunk (or saved (terrain/generate-chunk gen cx cy cz))]
                (.add ^java.util.concurrent.ConcurrentLinkedQueue chunk-gen-ready
                  {:pos pos :chunk chunk}))
              (catch Throwable e
                (println "Chunk gen error:" pos (.getMessage e))
                (swap! chunk-gen-pending disj pos)))))))))

(defn- drain-ready-chunks!
  "Drain up to n ready chunks from the queue. Returns vec of {:pos :chunk}."
  ([] (drain-ready-chunks! 8))
  ([^long max-n]
   (loop [results (transient [])
          i 0]
     (if (>= i max-n)
       (persistent! results)
       (if-let [entry (.poll ^java.util.concurrent.ConcurrentLinkedQueue chunk-gen-ready)]
         (do (swap! chunk-gen-pending disj (:pos entry))
             (recur (conj! results entry) (inc i)))
         (persistent! results))))))

(defn upload-chunk-meshes
  "Upload mesh data to GPU for all meshed chunks."
  [ctx world]
  (reduce-kv
    (fn [w pos chunk]
      (let [w (if-let [md (:mesh-data chunk)]
                (if (:gpu-mesh chunk)
                  w
                  (let [m (mesh/create-static-mesh ctx
                            (:vertices md) (:indices md) (* 9 4))]
                    (assoc-in w [pos :gpu-mesh] m)))
                w)
            w (if-let [wmd (:water-mesh-data chunk)]
                (if (:water-gpu-mesh (get w pos))
                  w
                  (let [m (mesh/create-static-mesh ctx
                            (:vertices wmd) (:indices wmd) (* 9 4))]
                    (assoc-in w [pos :water-gpu-mesh] m)))
                w)]
        w))
    world
    world))

(defn needed-chunks
  "Compute set of chunk positions that should be loaded around player."
  [^double px ^double pz]
  (let [pcx (Math/floorDiv (long px) (long w/CHUNK-SIZE))
        pcz (Math/floorDiv (long pz) (long w/CHUNK-SIZE))]
    (into #{}
      (for [cx (range (- pcx LOAD-RADIUS) (+ pcx LOAD-RADIUS 1))
            cz (range (- pcz LOAD-RADIUS) (+ pcz LOAD-RADIUS 1))
            cy (range HEIGHT-CHUNKS)]
        [cx cy cz]))))

(defn stream-chunks
  "Stream chunks using background generation.
  Game thread: submit missing → collect ready → trees+light+mesh+upload → unload."
  [state ctx]
  (let [t-start (System/nanoTime)
        world (:world state)
        gen (:gen state)
        [px _py pz] (:pos (:cam state))
        needed (needed-chunks (double px) (double pz))
        pcx (Math/floorDiv (long px) (long w/CHUNK-SIZE))
        pcz (Math/floorDiv (long pz) (long w/CHUNK-SIZE))
        world-path (:world-path state)

        ;; 1. Submit missing chunks to background pool
        missing (remove #(or (contains? world %)
                             (contains? @chunk-gen-pending %))
                        needed)
        _ (doseq [pos missing]
            (submit-chunk-gen! pos gen world-path))

        ;; 2. Collect ready chunks from background workers
        ready (drain-ready-chunks!)
        world (reduce (fn [w {:keys [pos chunk]}]
                        (assoc w pos chunk))
                      world ready)
        new-positions (mapv :pos ready)

        ;; 3. Place trees on new XZ columns
        new-xz-cols (into #{} (map (fn [[cx _ cz]] [cx cz]) new-positions))
        world (reduce (fn [w [cx cz]]
                        (terrain/place-trees w gen cx cz))
                      world new-xz-cols)

        ;; 4. Light new columns
        _ (when (seq new-xz-cols)
            (doseq [[cx cz] new-xz-cols]
              (lighting/propagate-sunlight-column! world cx cz HEIGHT-CHUNKS))
            (doseq [pos new-positions]
              (when-let [chunk (get world pos)]
                (lighting/light-chunk! chunk))))

        ;; 5. Mesh new chunks + dirty neighbors (cap at 8 per tick)
        mesh-positions (into (set new-positions)
                         (for [pos new-positions
                               [dx dy dz] [[1 0 0] [-1 0 0] [0 1 0] [0 -1 0] [0 0 1] [0 0 -1]]
                               :let [npos (mapv + pos [dx dy dz])]
                               :when (contains? world npos)]
                           npos))
        mesh-batch (vec (take 8 (filter #(and (contains? world %)
                                              (:dirty? (get world %)))
                                        mesh-positions)))
        world (reduce (fn [w pos]
                        (let [chunk (get w pos)
                              md (mesher/mesh-chunk chunk w)
                              wmd (mesher/mesh-chunk-water chunk w)]
                          (assoc w pos (assoc chunk :mesh-data md :water-mesh-data wmd :dirty? false))))
                      world mesh-batch)

        ;; 6. Upload new meshes to GPU
        world (reduce (fn [w pos]
                        (let [chunk (get w pos)
                              w (if (and (:mesh-data chunk) (not (:gpu-mesh chunk)))
                                  (let [md (:mesh-data chunk)
                                        m (mesh/create-static-mesh ctx
                                            (:vertices md) (:indices md) (* 9 4))]
                                    (assoc-in w [pos :gpu-mesh] m))
                                  w)
                              chunk (get w pos)
                              w (if (and (:water-mesh-data chunk) (not (:water-gpu-mesh chunk)))
                                  (let [wmd (:water-mesh-data chunk)
                                        m (mesh/create-static-mesh ctx
                                            (:vertices wmd) (:indices wmd) (* 9 4))]
                                    (assoc-in w [pos :water-gpu-mesh] m))
                                  w)]
                          w))
                      world (keys world))

        ;; 7. Unload distant chunks
        unload-dist-sq (* UNLOAD-RADIUS UNLOAD-RADIUS)
        to-unload (vec (filter
                    (fn [[cx _cy cz]]
                      (let [dx (- cx pcx) dz (- cz pcz)]
                        (> (+ (* dx dx) (* dz dz)) unload-dist-sq)))
                    (keys world)))
        _ (when (seq to-unload)
            (VK13/vkDeviceWaitIdle (:device ctx)))
        _ (doseq [pos to-unload]
            (let [chunk (get world pos)]
              (when (and (:blocks-dirty? chunk) (:world-path state))
                (persist/queue-save!
                  (fn [conn world-path]
                    (persist/save-chunk! conn world-path chunk))))
              (when-let [m (:gpu-mesh chunk)]
                (mesh/destroy-mesh! ctx m))
              (when-let [m (:water-gpu-mesh chunk)]
                (mesh/destroy-mesh! ctx m))))
        world (apply dissoc world to-unload)

        total-ms (/ (double (- (System/nanoTime) t-start)) 1e6)]
    (when (and (> total-ms 10.0) (or (seq new-positions) (seq mesh-batch)))
      (println (format "[stream] %.0f ms — %d ready, %d meshed, %d pending, %d unloaded"
                 total-ms (count new-positions) (count mesh-batch)
                 (count @chunk-gen-pending) (count to-unload))))
    (assoc state :world world)))

;; ================================================================
;; Hot shader reload (development tool)
;; ================================================================

(defn- shader-hashes
  "Compute hashes of all shader source strings for change detection."
  []
  {:terrain [(hash vert-glsl) (hash frag-glsl)]
   :sky     [(hash sky-vert-glsl) (hash sky-frag-glsl)]
   :water   [(hash water-vert-glsl) (hash water-frag-glsl)]
   :cross   [(hash crosshair-vert-glsl) (hash crosshair-frag-glsl)]
   :hud     [(hash hud/hud-vert-glsl) (hash hud/hud-frag-glsl)]})

(defn- rebuild-pipeline!
  "Recreate a single pipeline with new shader source. Returns new pipeline map."
  [device sc layout-handle vert-src frag-src opts]
  (pipeline/create-graphics-pipeline device
    (merge {:vert-glsl vert-src
            :frag-glsl frag-src
            :topology :triangles
            :color-format (:format sc)
            :depth-format VK13/VK_FORMAT_D32_SFLOAT
            :cull-mode :none
            :layout layout-handle}
           opts)))

(defn- maybe-reload-shaders
  "Check if any shader source changed; if so, recreate affected pipelines.
  Called once per second from update-valley. Old pipelines are deferred
  for destruction on a subsequent frame to avoid GPU use-after-free."
  [state]
  ;; First, destroy any pipelines queued for cleanup on a previous frame
  ;; NOTE: we only destroy the pipeline + shader modules, NOT the layout
  ;; because the layout is shared across all pipelines.
  (let [device (:device (:ctx state))
        old-pipes (:pipelines-to-destroy state)
        state (if (seq old-pipes)
                (do (VK13/vkDeviceWaitIdle device)
                    (doseq [p old-pipes]
                      (VK13/vkDestroyPipeline device (long (:pipeline p)) nil)
                      (shader/destroy-shader-module! device (:vert-module p))
                      (shader/destroy-shader-module! device (:frag-module p)))
                    (dissoc state :pipelines-to-destroy))
                state)
        old-hashes (:shader-hashes state)
        new-hashes (shader-hashes)]
    (if (= old-hashes new-hashes)
      state
      (let [sc (:swapchain-info (:ctx state))
            lh (long (:vk-layout state))
            changed (into {} (filter (fn [[k v]] (not= v (get old-hashes k)))) new-hashes)]
        (println "Hot-reloading shaders:" (keys changed))
        (try
          (VK13/vkDeviceWaitIdle device)
          (let [retired (transient [])

                state
                (if (:terrain changed)
                  (let [old (:terrain-pipe state)
                        new-pipe (rebuild-pipeline! device sc lh vert-glsl frag-glsl
                                   {:vertex-bindings vertex-bindings
                                    :vertex-attributes vertex-attributes})]
                    (conj! retired old)
                    (assoc state :terrain-pipe new-pipe))
                  state)

                state
                (if (:sky changed)
                  (let [old (:sky-pipe state)
                        new-pipe (rebuild-pipeline! device sc lh sky-vert-glsl sky-frag-glsl
                                   {:depth-test? false :depth-write? false})]
                    (conj! retired old)
                    (assoc state :sky-pipe new-pipe))
                  state)

                state
                (if (:water changed)
                  (let [old (:water-pipe state)
                        new-pipe (rebuild-pipeline! device sc lh water-vert-glsl water-frag-glsl
                                   {:depth-write? false :blend? true
                                    :vertex-bindings vertex-bindings
                                    :vertex-attributes vertex-attributes})]
                    (conj! retired old)
                    (assoc state :water-pipe new-pipe))
                  state)

                state
                (if (:cross changed)
                  (let [old (:crosshair-pipe state)
                        new-pipe (rebuild-pipeline! device sc lh
                                   crosshair-vert-glsl crosshair-frag-glsl
                                   {:depth-test? false :depth-write? false :blend? true
                                    :vertex-bindings vertex-bindings
                                    :vertex-attributes vertex-attributes})]
                    (conj! retired old)
                    (assoc state :crosshair-pipe new-pipe))
                  state)

                state
                (if (:hud changed)
                  (let [old (:hud-pipe state)
                        new-pipe (rebuild-pipeline! device sc lh
                                   hud/hud-vert-glsl hud/hud-frag-glsl
                                   {:depth-test? false :depth-write? false :blend? true
                                    :vertex-bindings vertex-bindings
                                    :vertex-attributes vertex-attributes})]
                    (conj! retired old)
                    (assoc state :hud-pipe new-pipe))
                  state)]
            (println "Shader reload complete.")
            (assoc state
              :shader-hashes new-hashes
              :pipelines-to-destroy (persistent! retired)))
          (catch Exception e
            (println "Shader reload FAILED:" (.getMessage e))
            state))))))

(defn init-valley [ctx loading-msg]
  (let [device (:device ctx)
        sc (:swapchain-info ctx)

        ;; Generate textures
        tex-images (textures/generate-block-textures)
        tex-array (texture/create-texture-array ctx tex-images
                    :filter-mode :nearest)

        ;; Texture descriptor
        tex-pool (texture/create-texture-descriptor-pool device 4)
        tex-layout (texture/create-texture-descriptor-layout device)
        tex-desc (texture/allocate-texture-array-descriptor-set
                   device tex-pool tex-layout tex-array)

        ;; Pipeline layout — shared across all pipelines
        ;; 116 bytes: mat4(64) + ratio(4) + time(4) + yaw(4) + pitch(4) + skyColors(24) + camPos(12)
        pipe-layout (texture/create-textured-pipeline-layout device
                      {:size 116 :stages (bit-or VK13/VK_SHADER_STAGE_VERTEX_BIT
                                                 VK13/VK_SHADER_STAGE_FRAGMENT_BIT)})

        terrain-pipe (pipeline/create-graphics-pipeline device
                       {:vert-glsl vert-glsl
                        :frag-glsl frag-glsl
                        :topology :triangles
                        :color-format (:format sc)
                        :depth-format VK13/VK_FORMAT_D32_SFLOAT
                        :cull-mode :none
                        :vertex-bindings vertex-bindings
                        :vertex-attributes vertex-attributes
                        :layout (:layout pipe-layout)})

        sky-pipe (pipeline/create-graphics-pipeline device
                   {:vert-glsl sky-vert-glsl
                    :frag-glsl sky-frag-glsl
                    :topology :triangles
                    :color-format (:format sc)
                    :depth-format VK13/VK_FORMAT_D32_SFLOAT
                    :depth-test? false
                    :depth-write? false
                    :cull-mode :none
                    :layout (:layout pipe-layout)})

        water-pipe (pipeline/create-graphics-pipeline device
                     {:vert-glsl water-vert-glsl
                      :frag-glsl water-frag-glsl
                      :topology :triangles
                      :color-format (:format sc)
                      :depth-format VK13/VK_FORMAT_D32_SFLOAT
                      :depth-write? false  ;; no depth write for transparent water
                      :blend? true
                      :cull-mode :none
                      :vertex-bindings vertex-bindings
                      :vertex-attributes vertex-attributes
                      :layout (:layout pipe-layout)})

        crosshair-pipe (pipeline/create-graphics-pipeline device
                         {:vert-glsl crosshair-vert-glsl
                          :frag-glsl crosshair-frag-glsl
                          :topology :triangles
                          :color-format (:format sc)
                          :depth-format VK13/VK_FORMAT_D32_SFLOAT
                          :depth-test? false
                          :depth-write? false
                          :blend? true
                          :cull-mode :none
                          :vertex-bindings vertex-bindings
                          :vertex-attributes vertex-attributes
                          :layout (:layout pipe-layout)})

        ;; Crosshair mesh
        ch-data (make-crosshair-data)
        crosshair-mesh (mesh/create-static-mesh ctx
                         (:vertices ch-data) (:indices ch-data) (* 9 4))

        ;; Persistence: try to load existing world
        world-path (persist/world-dir "valley-save")
        db-conn (or (persist/open-world-db world-path)
                    (persist/create-world-db! world-path))
        saved-meta (persist/load-world-meta db-conn)
        saved-player (persist/load-player db-conn)
        seed (long (or (:seed saved-meta) 42))

        ;; Generate initial world (small, streaming fills the rest)
        _ (reset! loading-msg "Generating terrain...")
        _ (println "Generating terrain..." (if saved-meta "(restoring saved world)" "(new world)"))
        gen (terrain/make-terrain-gen seed)
        world (terrain/generate-world gen 3 HEIGHT-CHUNKS)
        _ (reset! loading-msg (str "Lighting " (count world) " chunks..."))
        _ (println "Lighting" (count world) "chunks...")
        _ (lighting/light-world! world HEIGHT-CHUNKS)
        _ (reset! loading-msg (str "Meshing " (count world) " chunks..."))
        _ (println "Meshing" (count world) "chunks...")
        world (mesher/mesh-world world)
        _ (reset! loading-msg "Uploading to GPU...")
        _ (println "Uploading to GPU...")
        world (upload-chunk-meshes ctx world)
        _ (println "Ready!")

        ;; HUD pipeline
        hud-pipe (pipeline/create-graphics-pipeline device
                   {:vert-glsl hud/hud-vert-glsl
                    :frag-glsl hud/hud-frag-glsl
                    :topology :triangles
                    :color-format (:format sc)
                    :depth-format VK13/VK_FORMAT_D32_SFLOAT
                    :depth-test? false
                    :depth-write? false
                    :blend? true
                    :cull-mode :none
                    :vertex-bindings vertex-bindings
                    :vertex-attributes vertex-attributes
                    :layout (:layout pipe-layout)})

        ;; Camera start position — use saved or surface near origin
        start-y (+ (terrain/surface-height gen 0 0) 2)
        default-pos [0.0 (double start-y) 0.0]
        start-pos (or (:pos saved-player) default-pos)
        cam (camera/create-fps-camera start-pos 0.0 0.0)

        ;; Player entity + inventory — restore from save or create fresh
        player (if saved-player
                 (assoc (entity/create-player start-pos)
                   :health (:health saved-player))
                 (entity/create-player start-pos))
        inventory (or (:inventory saved-player)
                      (-> (inventory/create-inventory)
                          (inventory/add-item :wood-pickaxe 1)
                          (inventory/add-item :torch 16)))
        hunger-state (or (:hunger-state saved-player)
                         (hunger/create-hunger))
        game-time (double (or (:game-time saved-meta) 12000.0))

        ;; Initial HUD mesh
        health (long (:health player))
        hunger (long (:hunger hunger-state))
        hud-mesh-data (hud/build-hud-mesh health 20 hunger inventory 0)
        hud-mesh (mesh/create-static-mesh ctx
                   (:vertices hud-mesh-data) (:indices hud-mesh-data) (* 9 4))

        ;; Start persistence writer
        _ (persist/start-writer! db-conn world-path)
        ;; Save world meta if new
        _ (when-not saved-meta
            (persist/save-world-meta! db-conn seed 12000.0 default-pos))

        ;; Load saved mobs
        saved-mobs (persist/load-entities db-conn)]

    {:world world
     :gen gen
     :cam cam
     :ctx ctx          ;; needed for chunk streaming (GPU uploads)
     :vy 0.0          ;; vertical velocity
     :on-ground? true
     :fly-mode? true   ;; start in fly mode for easy exploration
     :frame-count 0
     :day-night-ratio 1.0
     :game-time game-time
     :sky-top-color [0.3 0.5 0.9]
     :sky-bot-color [0.7 0.8 1.0]
     :shader-hashes (shader-hashes)
     :terrain-pipe terrain-pipe
     :water-pipe water-pipe
     :sky-pipe sky-pipe
     :crosshair-pipe crosshair-pipe
     :crosshair-mesh crosshair-mesh
     :hud-pipe hud-pipe
     :hud-mesh hud-mesh
     :prev-hud-health 20
     :player player
     :inventory inventory
     :hunger-state hunger-state
     :mobs (or saved-mobs [])
     :solid-arr (tc/make-solid-bytes w/solid-arr)
     :world-state (vs/create-world)
     :mob-mesh nil
     :tex-array tex-array
     :tex-desc tex-desc
     :tex-pool tex-pool
     :tex-layout tex-layout
     :pipe-layout pipe-layout
     :vk-layout (:layout pipe-layout)
     :db-conn db-conn
     :world-path world-path}))

;; ================================================================
;; Frustum culling
;; ================================================================

(defn extract-frustum-planes
  "Extract 6 frustum planes from a column-major VP matrix.
  Each plane is [a b c d] where ax+by+cz+d >= 0 is inside."
  [^floats vp]
  ;; VP is column-major: vp[col*4+row]
  ;; Row i of the matrix: [vp[i] vp[4+i] vp[8+i] vp[12+i]]
  (let [r0 [(aget vp 0) (aget vp 4) (aget vp 8)  (aget vp 12)]
        r1 [(aget vp 1) (aget vp 5) (aget vp 9)  (aget vp 13)]
        r2 [(aget vp 2) (aget vp 6) (aget vp 10) (aget vp 14)]
        r3 [(aget vp 3) (aget vp 7) (aget vp 11) (aget vp 15)]
        add (fn [[a b c d] [e f g h]] [(+ a e) (+ b f) (+ c g) (+ d h)])
        sub (fn [[a b c d] [e f g h]] [(- a e) (- b f) (- c g) (- d h)])]
    [(add r3 r0)    ;; left
     (sub r3 r0)    ;; right
     (add r3 r1)    ;; bottom
     (sub r3 r1)    ;; top
     (add r3 r2)    ;; near
     (sub r3 r2)])) ;; far

(defn- aabb-outside-plane?
  "Test if an AABB is entirely outside a frustum plane [a b c d]."
  [plane minx miny minz maxx maxy maxz]
  (let [a (double (nth plane 0)) b (double (nth plane 1))
        c (double (nth plane 2)) d (double (nth plane 3))
        minx (double minx) miny (double miny) minz (double minz)
        maxx (double maxx) maxy (double maxy) maxz (double maxz)
        ;; Pick the vertex most in the direction of the plane normal
        px (if (pos? a) maxx minx)
        py (if (pos? b) maxy miny)
        pz (if (pos? c) maxz minz)]
    (neg? (+ (* a px) (* b py) (* c pz) d))))

(defn chunk-in-frustum?
  "Test if a chunk's AABB is at least partially inside the frustum."
  [planes cx cy cz]
  (let [cx (long cx) cy (long cy) cz (long cz)
        minx (double (* cx w/CHUNK-SIZE))
        miny (double (* cy w/CHUNK-SIZE))
        minz (double (* cz w/CHUNK-SIZE))
        maxx (+ minx (double w/CHUNK-SIZE))
        maxy (+ miny (double w/CHUNK-SIZE))
        maxz (+ minz (double w/CHUNK-SIZE))]
    (not (or (aabb-outside-plane? (nth planes 0) minx miny minz maxx maxy maxz)
             (aabb-outside-plane? (nth planes 1) minx miny minz maxx maxy maxz)
             (aabb-outside-plane? (nth planes 2) minx miny minz maxx maxy maxz)
             (aabb-outside-plane? (nth planes 3) minx miny minz maxx maxy maxz)
             (aabb-outside-plane? (nth planes 4) minx miny minz maxx maxy maxz)
             (aabb-outside-plane? (nth planes 5) minx miny minz maxx maxy maxz)))))

;; ================================================================
;; Player physics
;; ================================================================

(defn get-block-at
  "Get block at world position (float coords → block coords)."
  [world ^double x ^double y ^double z]
  (w/get-world-block world (long (Math/floor x)) (long (Math/floor y)) (long (Math/floor z))))

(defn collides?
  "Check if a player AABB at (x, y, z) with eye-height and radius collides with solid blocks."
  [world x y z]
  (let [x (double x) y (double y) z (double z)
        r PLAYER-RADIUS
        foot-y (- y PLAYER-HEIGHT)
        ;; Check blocks the player AABB overlaps
        bx0 (long (Math/floor (- x r)))
        bx1 (long (Math/floor (+ x r)))
        bz0 (long (Math/floor (- z r)))
        bz1 (long (Math/floor (+ z r)))
        by0 (long (Math/floor foot-y))
        by1 (long (Math/floor y))]
    (loop [bx bx0]
      (if (> bx bx1) false
        (or (loop [bz bz0]
              (if (> bz bz1) false
                (or (loop [by by0]
                      (if (> by by1) false
                        (or (w/solid? (w/get-world-block world bx by bz))
                            (recur (inc by)))))
                    (recur (inc bz)))))
            (recur (inc bx)))))))

(defn update-physics
  "Apply gravity, jumping, and collision to player state."
  [state inp ^double dt]
  (let [cam (:cam state)
        [px py pz] (:pos cam)
        px (double px) py (double py) pz (double pz)
        vy (double (:vy state))
        world (:world state)
        on-ground? (:on-ground? state)

        ;; Jump
        vy (if (and on-ground? (input/key-pressed? inp :space))
             JUMP-VELOCITY
             vy)

        ;; Gravity
        vy (- vy (* GRAVITY dt))

        ;; Vertical movement
        new-py (+ py (* vy dt))

        ;; Ground collision: check if feet would be inside a solid block
        foot-y (- new-py PLAYER-HEIGHT)
        ground-block (get-block-at world px foot-y pz)
        on-ground? (w/solid? ground-block)

        ;; Snap to ground if colliding
        [new-py vy on-ground?]
        (if (and on-ground? (neg? vy))
          (let [ground-top (+ (Math/floor foot-y) 1.0)]
            [(+ ground-top PLAYER-HEIGHT) 0.0 true])
          [new-py vy false])

        ;; Head collision
        [new-py vy] (let [head-block (get-block-at world px new-py pz)]
                      (if (and (w/solid? head-block) (pos? vy))
                        [py 0.0]
                        [new-py vy]))

        ;; Horizontal collision (simple: try each axis separately)
        ;; fps-camera-update handles mouse look + WASD; we override Y with physics
        cam-moved (camera/fps-camera-update cam inp dt MOVE-SPEED MOUSE-SENSITIVITY)
        [nx _cam-y nz] (:pos cam-moved)
        nx (double nx) nz (double nz)

        ;; Try X movement
        nx (if (collides? world nx new-py pz) px nx)
        ;; Try Z movement
        nz (if (collides? world nx new-py nz) pz nz)

        cam (assoc cam-moved :pos [nx new-py nz])

        ;; Fall damage tracking
        dy-moved (- new-py py)
        player (:player state)
        player (if player
                 (entity/update-fall-distance player vy dy-moved on-ground?)
                 player)]
    (cond-> (assoc state :cam cam :vy vy :on-ground? on-ground?)
      player (assoc :player player))))

;; ================================================================
;; Block interaction (raycast + break/place)
;; ================================================================

(def ^:const REACH 6.0)     ;; max interaction distance in blocks
(def ^:const RAY-STEP 0.05) ;; raycast step size

(defn raycast
  "Cast a ray from eye position along direction. Returns [hit-pos face-normal]
  where hit-pos is the block coord that was hit, face-normal is the entry direction,
  or nil if nothing hit within REACH."
  [world eye-pos dir]
  (let [[ex ey ez] eye-pos
        [dx dy dz] dir
        ex (double ex) ey (double ey) ez (double ez)
        dx (double dx) dy (double dy) dz (double dz)
        steps (long (/ REACH RAY-STEP))]
    (loop [i (long 1)
           prev-bx (long (Math/floor ex))
           prev-by (long (Math/floor ey))
           prev-bz (long (Math/floor ez))]
      (when (< i steps)
        (let [t (* (double i) RAY-STEP)
              px (+ ex (* dx t))
              py (+ ey (* dy t))
              pz (+ ez (* dz t))
              bx (long (Math/floor px))
              by (long (Math/floor py))
              bz (long (Math/floor pz))]
          (if (and (not= bx prev-bx) (not= by prev-by) (not= bz prev-bz))
            ;; Entered a new block — check if solid
            (recur (inc i) bx by bz)
            (let [block (w/get-world-block world bx by bz)]
              (if (w/solid? block)
                {:hit [bx by bz]
                 :face [(- prev-bx bx) (- prev-by by) (- prev-bz bz)]}
                (recur (inc i) bx by bz)))))))))

(defn camera-ray-dir
  "Get the forward direction vector from camera yaw/pitch."
  [cam]
  (let [yaw (double (:yaw cam))
        pitch (double (:pitch cam))
        cp (Math/cos pitch)]
    [(* (Math/sin yaw) cp)
     (Math/sin pitch)
     (* (- (Math/cos yaw)) cp)]))

(defn remesh-chunk-at
  "Re-mesh the chunk containing world block (wx, wy, wz) and upload to GPU."
  [state wx wy wz]
  (let [[cpos _] (w/world-to-chunk wx wy wz)
        world (:world state)
        ctx (:ctx state)]
    (if-let [chunk (get world cpos)]
      (let [md (mesher/mesh-chunk chunk world)
            wmd (mesher/mesh-chunk-water chunk world)
            ;; Wait for GPU before destroying in-flight meshes
            _ (when (or (:gpu-mesh chunk) (:water-gpu-mesh chunk))
                (VK13/vkDeviceWaitIdle (:device ctx)))
            _ (when-let [m (:gpu-mesh chunk)]
                (mesh/destroy-mesh! ctx m))
            _ (when-let [m (:water-gpu-mesh chunk)]
                (mesh/destroy-mesh! ctx m))
            ;; Upload new meshes
            chunk (assoc chunk :mesh-data md :water-mesh-data wmd :dirty? false)
            chunk (if md
                    (assoc chunk :gpu-mesh
                      (mesh/create-static-mesh ctx
                        (:vertices md) (:indices md) (* 9 4)))
                    (dissoc chunk :gpu-mesh))
            chunk (if wmd
                    (assoc chunk :water-gpu-mesh
                      (mesh/create-static-mesh ctx
                        (:vertices wmd) (:indices wmd) (* 9 4)))
                    (dissoc chunk :water-gpu-mesh))]
        (assoc-in state [:world cpos] chunk))
      state)))

(defn remesh-affected
  "Relight and re-mesh the chunk at (wx,wy,wz) and neighbors affected by the change."
  [state ^long wx ^long wy ^long wz]
  (let [[cpos _] (w/world-to-chunk wx wy wz)
        [cx _cy cz] cpos
        lx (Math/floorMod wx (long w/CHUNK-SIZE))
        ly (Math/floorMod wy (long w/CHUNK-SIZE))
        lz (Math/floorMod wz (long w/CHUNK-SIZE))
        ;; Relight the column containing the modified block
        world (:world state)
        _ (lighting/relight-column! world cx cz)
        ;; Also relight neighbor columns if the block is on an X or Z boundary
        _ (when (zero? lx) (lighting/relight-column! world (dec cx) cz))
        _ (when (= lx (dec w/CHUNK-SIZE)) (lighting/relight-column! world (inc cx) cz))
        _ (when (zero? lz) (lighting/relight-column! world cx (dec cz)))
        _ (when (= lz (dec w/CHUNK-SIZE)) (lighting/relight-column! world cx (inc cz)))
        ;; Remesh the affected chunk and boundary neighbors
        state (remesh-chunk-at state wx wy wz)]
    (cond-> state
      (zero? lx)                     (remesh-chunk-at (dec wx) wy wz)
      (= lx (dec w/CHUNK-SIZE))     (remesh-chunk-at (inc wx) wy wz)
      (zero? ly)                     (remesh-chunk-at wx (dec wy) wz)
      (= ly (dec w/CHUNK-SIZE))     (remesh-chunk-at wx (inc wy) wz)
      (zero? lz)                     (remesh-chunk-at wx wy (dec wz))
      (= lz (dec w/CHUNK-SIZE))     (remesh-chunk-at wx wy (inc wz)))))

(defn handle-block-interaction
  "Handle left-click (break/attack) and right-click (place) on blocks and mobs."
  [state inp]
  (let [cam (:cam state)
        world (:world state)
        ray-dir (camera-ray-dir cam)
        [rdx rdy rdz] ray-dir
        [ex ey ez] (:pos cam)]
    (cond
      ;; Left click — attack mob or break block
      (input/mouse-pressed? inp :left)
      (let [mob-hit (mobs/find-hit-mob (:mobs state) ex ey ez rdx rdy rdz)]
        (if mob-hit
          ;; Hit a mob — deal damage
          (let [[hit-mob _dist] mob-hit
                mob-id (:id hit-mob)
                damage (long (items/attack-damage (inventory/get-held-item (:inventory state))))
                state (update state :mobs
                        (fn [ms]
                          (mapv (fn [m]
                                  (if (= (:id m) mob-id)
                                    (mobs/damage-mob m damage (:pos cam))
                                    m))
                                ms)))
                ;; Check for dead mobs → drop items
                state (let [dead (filter mobs/mob-dead? (:mobs state))
                            drops (mapcat mobs/mob-drops dead)
                            inv (reduce (fn [inv d]
                                          (inventory/add-item inv (:item d) (:count d)))
                                        (:inventory state) drops)]
                        (-> state
                            (assoc :inventory inv)
                            (update :mobs (fn [ms] (filterv (complement mobs/mob-dead?) ms)))))
                ;; Damage held tool
                state (if (items/tool? (inventory/get-held-item (:inventory state)))
                        (update state :inventory inventory/damage-held-tool)
                        state)
                ;; Attack exhaustion
                state (if (:hunger-state state)
                        (update state :hunger-state hunger/exhaust hunger/ATTACK-EXHAUSTION)
                        state)]
            state)
          ;; No mob hit — try block
          (let [hit (raycast world (:pos cam) ray-dir)]
            (if hit
              (let [[bx by bz] (:hit hit)
                    block-id (w/get-world-block world bx by bz)
                    ;; Check if it's a crop — harvest drops differ
                    crop-drops (farming/harvest-crop block-id)
                    drop-item (when-not crop-drops (items/block->drop block-id))
                    state (update state :world w/set-world-block! bx by bz w/AIR)
                    state (if crop-drops
                            (reduce (fn [s {:keys [item count]}]
                                      (update s :inventory inventory/add-item item count))
                                    state crop-drops)
                            (if drop-item
                              (update state :inventory inventory/add-item drop-item 1)
                              state))
                    state (if (items/tool? (inventory/get-held-item (:inventory state)))
                            (update state :inventory inventory/damage-held-tool)
                            state)
                    ;; Mining exhaustion
                    state (if (:hunger-state state)
                            (update state :hunger-state hunger/exhaust hunger/MINE-EXHAUSTION)
                            state)]
                (remesh-affected state bx by bz))
              state))))

      ;; Right click — eat food, use hoe, plant seed, or place block
      (input/mouse-pressed? inp :right)
      (let [hit (raycast world (:pos cam) ray-dir)
            inv (:inventory state)
            held (inventory/get-held-item inv)]
        (cond
          ;; Eat food if holding food and hungry
          (and held (hunger/food? held) (:hunger-state state))
          (if-let [new-hunger (hunger/eat-food (:hunger-state state) held)]
            (-> state
                (assoc :hunger-state new-hunger)
                (update :inventory inventory/remove-item held 1))
            state)

          ;; Hoe: till dirt/grass into farmland
          (and hit held
               (= :hoe (:tool-type (get items/item-props held))))
          (let [[bx by bz] (:hit hit)
                block-id (w/get-world-block world bx by bz)]
            (if-let [new-block (farming/use-hoe block-id)]
              (let [state (update state :world w/set-world-block! bx by bz new-block)
                    state (update state :inventory inventory/damage-held-tool)]
                (remesh-affected state bx by bz))
              state))

          ;; Plant seed on farmland
          (and hit held (get farming/seed->crop held))
          (let [[bx by bz] (:hit hit)
                [fx fy fz] (:face hit)
                px (+ (long bx) (long fx))
                py (+ (long by) (long fy))
                pz (+ (long bz) (long fz))
                block-below (w/get-world-block world bx by bz)]
            (if-let [crop-block (farming/plant-seed held block-below)]
              ;; Place crop on top of farmland
              (let [state (update state :world w/set-world-block! px py pz crop-block)
                    state (update state :inventory inventory/remove-item held 1)]
                (remesh-affected state px py pz))
              state))

          ;; Place block from held item
          :else
          (let [block-id (when held (items/item->block held))]
            (if (and hit block-id)
              (let [[bx by bz] (:hit hit)
                    [fx fy fz] (:face hit)
                    px (+ (long bx) (long fx))
                    py (+ (long by) (long fy))
                    pz (+ (long bz) (long fz))
                    [cx cy cz] (:pos cam)
                    foot-y (- (double cy) PLAYER-HEIGHT)
                    player-block-y (long (Math/floor (double cy)))
                    player-block-fy (long (Math/floor foot-y))]
                (if (and (= px (long (Math/floor (double cx))))
                         (= pz (long (Math/floor (double cz))))
                         (or (= py player-block-y) (= py player-block-fy)))
                  state
                  (let [state (update state :world w/set-world-block! px py pz block-id)
                        state (update state :inventory inventory/remove-item held 1)]
                    (remesh-affected state px py pz))))
              state))))

      :else state)))

;; ================================================================
;; Update
;; ================================================================

(defn update-valley [state inp dt]
  (let [frame-t0 (System/nanoTime)
        dt (double (min dt 0.05))
        fly? (:fly-mode? state)
        fc (long (:frame-count state 0))

        ;; Toggle fly mode
        state (if (input/key-pressed? inp :f)
                (do (println (if fly? "Walk mode" "Fly mode"))
                    (cond-> (assoc state :fly-mode? (not fly?) :vy 0.0)
                      ;; Reset fall distance when entering walk mode so
                      ;; landing from fly height doesn't cause instant death
                      (and fly? (:player state))
                      (assoc-in [:player :data :fall-distance] 0.0)))
                state)
        fly? (:fly-mode? state)

        ;; Movement
        state (with-prof :movement
                (if fly?
                  (let [cam (camera/fps-camera-update (:cam state) inp dt MOVE-SPEED MOUSE-SENSITIVITY)
                        pos (:pos cam)
                        dy (* dt MOVE-SPEED
                               (cond
                                 (input/key-down? inp :space) 1.0
                                 (input/key-down? inp :left-shift) -1.0
                                 :else 0.0))
                        cam (assoc cam :pos [(nth pos 0)
                                             (+ (double (nth pos 1)) dy)
                                             (nth pos 2)])]
                    (assoc state :cam cam))
                  (update-physics state inp dt)))

        ;; Environmental damage (lava, drowning) + entity tick
        state (if (and (not fly?) (:player state))
                (let [cam (:cam state)
                      [cx cy cz] (:pos cam)
                      foot-y (- (double cy) PLAYER-HEIGHT)
                      feet-block (get-block-at (:world state) (double cx) foot-y (double cz))
                      head-block (get-block-at (:world state) (double cx) (double cy) (double cz))
                      player (-> (:player state)
                                 (entity/tick-entity dt)
                                 (entity/tick-environmental-damage feet-block head-block dt))]
                  (if (entity/dead? player)
                    ;; Death → respawn
                    (let [player (entity/respawn player)
                          spawn-pos (:pos player)
                          cam (assoc (:cam state) :pos spawn-pos)]
                      (assoc state :player player :cam cam :vy 0.0 :on-ground? true
                             :hunger-state (hunger/create-hunger)))
                    (assoc state :player player)))
                state)

        ;; Hunger ticking
        state (if (and (not fly?) (:hunger-state state) (:player state))
                (let [;; Sprint exhaustion (moving + shift)
                      moving? (or (input/key-down? inp :w) (input/key-down? inp :a)
                                  (input/key-down? inp :s) (input/key-down? inp :d))
                      sprinting? (and moving? (input/key-down? inp :left-control))
                      hs (:hunger-state state)
                      hs (if sprinting? (hunger/exhaust hs (* hunger/SPRINT-EXHAUSTION dt 4.0)) hs)
                      ;; Jump exhaustion
                      hs (if (and (:on-ground? state) (input/key-pressed? inp :space))
                           (hunger/exhaust hs hunger/JUMP-EXHAUSTION) hs)
                      ;; Tick hunger
                      {:keys [hunger-state heal damage]} (hunger/tick-hunger hs dt)
                      ;; Apply heal/damage to player
                      player (:player state)
                      player (if (pos? (long heal)) (entity/heal-entity player heal) player)
                      player (if (pos? (long damage)) (entity/damage-entity player damage) player)]
                  (assoc state :hunger-state hunger-state :player player))
                state)

        ;; Hotbar selection (number keys 1-9 + scroll wheel)
        state (let [inv (:inventory state)]
                (cond
                  (input/key-pressed? inp :1) (assoc state :inventory (inventory/select-slot inv 0))
                  (input/key-pressed? inp :2) (assoc state :inventory (inventory/select-slot inv 1))
                  (input/key-pressed? inp :3) (assoc state :inventory (inventory/select-slot inv 2))
                  (input/key-pressed? inp :4) (assoc state :inventory (inventory/select-slot inv 3))
                  (input/key-pressed? inp :5) (assoc state :inventory (inventory/select-slot inv 4))
                  (input/key-pressed? inp :6) (assoc state :inventory (inventory/select-slot inv 5))
                  (input/key-pressed? inp :7) (assoc state :inventory (inventory/select-slot inv 6))
                  (input/key-pressed? inp :8) (assoc state :inventory (inventory/select-slot inv 7))
                  (input/key-pressed? inp :9) (assoc state :inventory (inventory/select-slot inv 8))
                  :else
                  (let [[_sx sy] (input/scroll-delta inp)
                        sy (double (or sy 0.0))]
                    (if (not (zero? sy))
                      (assoc state :inventory (inventory/cycle-selection inv (if (pos? (double sy)) -1 1)))
                      state))))

        ;; Block interaction
        state (with-prof :block-interact (handle-block-interaction state inp))

        ;; Tick mobs via Datahike WorldState
        player-pos (get-in state [:cam :pos])
        day-ratio (double (get state :day-night-ratio 1.0))
        solid-arr (:solid-arr state)
        state (with-prof :mobs (if-let [ws (:world-state state)]
                (let [db (vs/world-db ws)
                      mob-tuples (vs/pull-all-mobs db)]
                  (if (seq mob-tuples)
                    ;; Datahike path: query → tick → fast-transact
                    (let [mob-maps (vs/mobs->maps mob-tuples)
                          ticked (mapv (fn [m]
                                         (if (:hostile? m)
                                           (hostile/tick-hostile m (:world state) solid-arr player-pos day-ratio dt)
                                           (mobs/tick-mob m (:world state) solid-arr dt)))
                                       mob-maps)
                          triples (vs/build-full-updates db ticked)]
                      (when (seq triples)
                        (vs/fast-transact! ws triples))
                      ;; Keep :mobs in sync for rendering/combat (legacy compat)
                      (assoc state :mobs ticked))
                    ;; No mobs in Datahike yet — populate from mob vector
                    (do (when (seq (:mobs state))
                          (doseq [m (:mobs state)]
                            (vs/add-mob! ws (:mob-type m)
                              (mobs/mob-x m) (mobs/mob-y m) (mobs/mob-z m)
                              (boolean (:hostile? m)))))
                        state)))
                ;; No WorldState — fallback to old vector path
                (update state :mobs
                  (fn [ms]
                    (mapv (fn [m]
                            (if (:hostile? m)
                              (hostile/tick-hostile m (:world state) solid-arr player-pos day-ratio dt)
                              (mobs/tick-mob m (:world state) solid-arr dt)))
                          ms)))))

        ;; Hostile mob attacks → player damage
        state (if (and (not fly?) (:player state))
                (reduce (fn [s m]
                          (if (:hostile? m)
                            (let [{:keys [mob player-damage]} (hostile/tick-hostile-attack m player-pos dt)]
                              (cond-> (update s :mobs (fn [ms] (mapv #(if (= (:id %) (:id mob)) mob %) ms)))
                                (and (pos? (long player-damage))
                                     (entity/can-hurt? (:player s)))
                                (update :player entity/damage-entity player-damage)))
                            s))
                        state (:mobs state))
                state)

        ;; Lurker explosions
        state (let [explosions (keep (fn [m]
                                       (when (= (:mob-type m) :lurker)
                                         (let [result (hostile/tick-lurker-fuse m player-pos dt)]
                                           (when (:explode? result) result))))
                                     (:mobs state))]
                (if (seq explosions)
                  (reduce (fn [s {:keys [pos]}]
                            (let [destroyed (hostile/explode-blocks (:world s) pos 3)
                                  ;; Remove blocks
                                  s (reduce (fn [s [bx by bz]]
                                              (update s :world w/set-world-block! bx by bz w/AIR))
                                            s destroyed)
                                  ;; Remesh affected chunks
                                  s (reduce (fn [s [bx by bz]]
                                              (remesh-affected s bx by bz))
                                            s destroyed)
                                  ;; Damage player if nearby
                                  [ex _ey ez] pos
                                  [px _py pz] player-pos
                                  blast-dist (Math/sqrt (+ (* (- (double ex) (double px)) (- (double ex) (double px)))
                                                          (* (- (double ez) (double pz)) (- (double ez) (double pz)))))]
                              (if (and (< blast-dist 5.0) (:player s) (entity/can-hurt? (:player s)))
                                (update s :player entity/damage-entity (long (max 1 (- 10 (* 2 (long blast-dist))))))
                                s)))
                          ;; Remove exploded lurkers
                          (update state :mobs
                            (fn [ms] (filterv (fn [m] (not (and (= (:mob-type m) :lurker)
                                                                (some #(= (:id m) (:id %))
                                                                      (map :mob explosions)))))
                                              ms)))
                          explosions)
                  state))

        ;; Spawn/despawn mobs + tick crops (every 200 frames ≈ 3 sec)
        state (if (zero? (rem fc 200))
                (let [player-pos (get-in state [:cam :pos])
                      ;; Spawn/despawn via legacy mob vector (both paths sync)
                      state (-> state
                                (update :mobs mobs/despawn-mobs player-pos)
                                (update :mobs mobs/spawn-mobs (:world state) player-pos)
                                (update :mobs hostile/spawn-hostiles (:world state) player-pos day-ratio))
                      ;; Sync new mobs to Datahike
                      state (if-let [ws (:world-state state)]
                              (let [db (vs/world-db ws)
                                    dh-eids (set (vs/all-mob-eids db))
                                    ;; Add mobs that exist in vector but not in Datahike
                                    new-mobs (filter #(not (contains? dh-eids (:db/id %))) (:mobs state))]
                                (doseq [m new-mobs]
                                  (when (and (:mob-type m) (not (:db/id m)))
                                    (vs/add-mob! ws (:mob-type m)
                                      (mobs/mob-x m) (mobs/mob-y m) (mobs/mob-z m)
                                      (boolean (:hostile? m)))))
                                state)
                              state)
                      ;; Tick crops and farmland near player
                      [px _py pz] player-pos
                      px (long (Math/floor (double px)))
                      pz (long (Math/floor (double pz)))
                      world (:world state)]
                  ;; Scan blocks in 16-block radius around player for crops/farmland
                  (reduce (fn [s dx]
                            (reduce (fn [s dz]
                                      (let [bx (+ px dx) bz (+ pz dz)]
                                        (reduce (fn [s by]
                                                  (let [block (w/get-world-block world bx by bz)]
                                                    (cond
                                                      (farming/crop-block? block)
                                                      (if-let [new-block (farming/tick-crop world bx by bz block)]
                                                        (let [s (update s :world w/set-world-block! bx by bz new-block)]
                                                          (remesh-affected s bx by bz))
                                                        s)

                                                      (or (= block farming/FARMLAND) (= block farming/FARMLAND-WET))
                                                      (if-let [new-block (farming/tick-farmland world bx by bz block)]
                                                        (let [s (update s :world w/set-world-block! bx by bz new-block)]
                                                          (remesh-affected s bx by bz))
                                                        s)

                                                      :else s)))
                                                s (range 1 80))))
                                    s (range -16 17)))
                          state (range -16 17)))
                state)

        ;; Rebuild mob mesh (every 4 frames — mobs move slowly)
        state (with-prof :mob-mesh (if (zero? (rem fc 4))
                (let [mob-mesh-data (mob-models/build-all-mobs-mesh (:mobs state))
                      old-mesh (:mob-mesh state)
                      ;; Queue old mesh for deferred destruction (2 frames for in-flight safety)
                      stale (get state :stale-meshes [])
                      stale (if old-mesh (conj stale {:mesh old-mesh :frame fc}) stale)
                      ;; Destroy meshes that are old enough (>= 4 frames)
                      to-destroy (filterv #(> (- fc (long (:frame %))) 3) stale)
                      stale (filterv #(<= (- fc (long (:frame %))) 3) stale)
                      _ (doseq [{:keys [mesh]} to-destroy]
                          (mesh/destroy-mesh! (:ctx state) mesh))]
                  (if mob-mesh-data
                    (assoc state :mob-mesh
                      (mesh/create-static-mesh (:ctx state)
                        (:vertices mob-mesh-data) (:indices mob-mesh-data) (* 9 4))
                      :stale-meshes stale)
                    (assoc state :mob-mesh nil :stale-meshes stale)))
                state))

        ;; Update HUD mesh (hearts)
        state (if (:player state)
                (hud/update-hud-mesh state (:ctx state))
                state)

        ;; Chunk streaming (every STREAM-INTERVAL frames)
        state (with-prof :streaming
                (if (zero? (rem fc STREAM-INTERVAL))
                  (try
                    (stream-chunks state (:ctx state))
                    (catch Exception e
                      (println "Streaming error:" (.getMessage e))
                      state))
                  state))

        ;; Hot shader reload (check every 60 frames ≈ 1/sec)
        state (if (zero? (rem fc 60))
                (try
                  (maybe-reload-shaders state)
                  (catch Exception e
                    (println "Shader reload error:" (.getMessage e))
                    state))
                state)

        ;; Periodic save (every 300 frames ≈ 5 sec for player, 1800 ≈ 30 sec for entities)
        _ (when-let [conn (:db-conn state)]
            (when (zero? (rem fc 300))
              (persist/queue-save!
                (fn [c wp]
                  (persist/save-player! c (:player state) (:inventory state)
                    (:hunger-state state) (get-in state [:cam :pos]))
                  (persist/save-world-meta! c 42 (get state :game-time 12000.0)
                    [0.0 0.0 0.0]))))
            (when (zero? (rem fc 1800))
              (persist/queue-save!
                (fn [c _wp]
                  (persist/save-entities! c (:mobs state))))))

        ;; Advance game time and compute day/night ratio
        game-time (double (get state :game-time 12000.0))
        game-time (mod (+ game-time (* (double dt) TIME-SPEED)) DAY-LENGTH)
        day-ratio (get-day-night-ratio game-time)
        [sky-top sky-bot] (get-sky-colors game-time)

        state (assoc state
                :frame-count (inc fc)
                :game-time game-time
                :day-night-ratio day-ratio
                :sky-top-color sky-top
                :sky-bot-color sky-bot)]

    ;; Log slow frames
    (let [frame-ms (/ (double (- (System/nanoTime) frame-t0)) 1e6)]
      (when (> frame-ms 16.0)
        (swap! slow-frames conj {:frame fc :total-ms frame-ms})))
    (if (input/key-pressed? inp :escape)
      (assoc state :quit? true)
      state)))

;; ================================================================
;; Render
;; ================================================================

(defn- push-constants!
  "Write 116 bytes of push constants: mat4(64) + ratio(4) + time(4) + yaw(4) + pitch(4) + skyColors(24) + camPos(12)."
  [^VkCommandBuffer cb ^long layout state ^floats vp]
  (let [stack (MemoryStack/stackPush)
        buf (.malloc stack 116)
        cam (:cam state)
        cam-pos (:pos cam)
        day-ratio (float (get state :day-night-ratio 1.0))
        game-time (float (get state :game-time 12000.0))
        sky-top (get state :sky-top-color [0.3 0.5 0.9])
        sky-bot (get state :sky-bot-color [0.7 0.8 1.0])]
    (dotimes [i 16] (.putFloat buf (* i 4) (aget vp i)))
    (.putFloat buf 64 day-ratio)
    (.putFloat buf 68 game-time)
    (.putFloat buf 72 (float (:yaw cam)))
    (.putFloat buf 76 (float (:pitch cam)))
    (.putFloat buf 80 (float (nth sky-top 0)))
    (.putFloat buf 84 (float (nth sky-top 1)))
    (.putFloat buf 88 (float (nth sky-top 2)))
    (.putFloat buf 92 (float (nth sky-bot 0)))
    (.putFloat buf 96 (float (nth sky-bot 1)))
    (.putFloat buf 100 (float (nth sky-bot 2)))
    (.putFloat buf 104 (float (nth cam-pos 0)))
    (.putFloat buf 108 (float (nth cam-pos 1)))
    (.putFloat buf 112 (float (nth cam-pos 2)))
    (VK13/vkCmdPushConstants cb layout
      (bit-or VK13/VK_SHADER_STAGE_VERTEX_BIT VK13/VK_SHADER_STAGE_FRAGMENT_BIT)
      0 buf)
    (MemoryStack/stackPop)))

(defn render-valley [state ^VkCommandBuffer cb ctx {:keys [extent]}]
  (let [[w h] extent
        cam (:cam state)
        proj (math/perspective (/ Math/PI 3.0) (/ (double w) (double h)) 0.1 500.0)
        view (camera/camera-view-matrix cam)
        vp (math/mat4-mul proj view)]

    ;; Set viewport/scissor
    (pipeline/cmd-set-viewport! cb w h)
    (pipeline/cmd-set-scissor! cb w h)

    (let [layout (long (:vk-layout state))]

      ;; Sky (drawn first, no depth)
      (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
        (:pipeline (:sky-pipe state)))
      (push-constants! cb layout state vp)
      (VK13/vkCmdDraw cb 3 1 0 0)

      ;; Terrain chunks
      (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
        (:pipeline (:terrain-pipe state)))

      ;; Bind texture array
      (let [stack (MemoryStack/stackPush)]
        (VK13/vkCmdBindDescriptorSets cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
          layout 0 (.longs stack (long (:tex-desc state))) nil)
        (MemoryStack/stackPop))

      (push-constants! cb layout state vp)

    ;; Draw visible chunk meshes (frustum culled)
    (let [planes (extract-frustum-planes vp)]
      (doseq [[[cx cy cz] chunk] (:world state)]
        (when (and (:gpu-mesh chunk)
                   (chunk-in-frustum? planes cx cy cz))
          (let [m (:gpu-mesh chunk)]
            (mesh/cmd-bind-mesh! cb m)
            (mesh/cmd-draw-mesh! cb m))))

      ;; Mob mesh (uses terrain pipeline, same vertex format)
      (when-let [mob-mesh (:mob-mesh state)]
        (mesh/cmd-bind-mesh! cb mob-mesh)
        (mesh/cmd-draw-mesh! cb mob-mesh))

      ;; Water pass (alpha blend, no depth write)
      (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
        (:pipeline (:water-pipe state)))

      (let [stack (MemoryStack/stackPush)]
        (VK13/vkCmdBindDescriptorSets cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
          layout 0 (.longs stack (long (:tex-desc state))) nil)
        (MemoryStack/stackPop))
      (push-constants! cb layout state vp)

      (doseq [[[cx cy cz] chunk] (:world state)]
        (when (and (:water-gpu-mesh chunk)
                   (chunk-in-frustum? planes cx cy cz))
          (let [m (:water-gpu-mesh chunk)]
            (mesh/cmd-bind-mesh! cb m)
            (mesh/cmd-draw-mesh! cb m)))))

      ;; Crosshair
      (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
        (:pipeline (:crosshair-pipe state)))
      (push-constants! cb layout state vp)
      (mesh/cmd-bind-mesh! cb (:crosshair-mesh state))
      (mesh/cmd-draw-mesh! cb (:crosshair-mesh state))

      ;; HUD (hearts)
      (when (:hud-mesh state)
        (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
          (:pipeline (:hud-pipe state)))
        (let [stack (MemoryStack/stackPush)]
          (VK13/vkCmdBindDescriptorSets cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
            layout 0 (.longs stack (long (:tex-desc state))) nil)
          (MemoryStack/stackPop))
        (push-constants! cb layout state vp)
        (hud/render-hud! cb state)))))

;; ================================================================
;; Entry point
;; ================================================================

(defn- run-game! []
  (try
    (println "Starting Living Valley...")
    (start-chunk-gen-pool!)
    (gpu/with-gpu-context [ctx {:width W :height H
                                :title "Living Valley — Raster Voxel Engine"
                                :depth? true}]
      (reset! game-window (:window ctx))
      (let [inp (input/create-input-system (:window ctx))
            sync (frame/create-frame-sync ctx)
            ;; Run heavy init in background — frame loop starts immediately
            init-result (atom nil)
            loading-msg (atom "Loading...")
            _ (future
                (try
                  (let [result (init-valley ctx loading-msg)]
                    (reset! init-result result)
                    (println "World ready!"))
                  (catch Throwable e
                    (println "Init error:" (.getMessage e))
                    (.printStackTrace e))))
            ;; Loading state: minimal, just enough to render a clear screen
            loading-state {:loading? true :world {} :mobs []}]
        (try
          (frame/run-game-loop! ctx
            {:frame-sync sync
             :init-state loading-state
             :input-system inp
             :update-fn (fn [state inp dt]
                          (if (:loading? state)
                            ;; Check if init is done
                            (if-let [full-state @init-result]
                              (do (camera/set-cursor-captured! (:window ctx) true)
                                  full-state)
                              (do (GLFW/glfwSetWindowTitle
                                    (long (:window ctx))
                                    (str "Living Valley — " @loading-msg))
                                  state))
                            ;; Normal gameplay
                            (update-valley state inp dt)))
             :render-fn (fn [state cb ctx opts]
                          (when-not (:loading? state)
                            (render-valley state cb ctx opts)))
             :clear-color [0.4 0.6 0.9 1.0]
             :title "Living Valley"
             :state-atom game-state-ref})
          (finally
            (VK13/vkDeviceWaitIdle (:device ctx))
            ;; Save all state before cleanup
            (let [final-state (or @game-state-ref loading-state)]
              (when-let [conn (:db-conn final-state)]
                (println "Saving world...")
                ;; Save only modified chunks
                (let [dirty (filter (fn [[_ chunk]] (:blocks-dirty? chunk)) (:world final-state))]
                  (println "Saving" (count dirty) "dirty chunks out of" (count (:world final-state)) "loaded...")
                  (doseq [[pos chunk] dirty]
                    (try
                      (persist/save-chunk! conn (:world-path final-state) chunk)
                      (catch Exception e
                        (println "Chunk save error:" (.getMessage e))))))
                ;; Save player
                (persist/save-player! conn (:player final-state) (:inventory final-state)
                  (:hunger-state final-state) (get-in final-state [:cam :pos]))
                ;; Save entities
                (persist/save-entities! conn (:mobs final-state))
                ;; Save world meta
                (persist/save-world-meta! conn 42 (:game-time final-state) [0.0 0.0 0.0])
                ;; Stop writer thread and close DB
                (persist/stop-writer! conn (:world-path final-state))
                (persist/close-world-db! conn)
                (println "World saved.")))
            ;; Clean up all chunks (including streamed ones)
            (let [final-state (or @game-state-ref loading-state)]
            (doseq [[_ chunk] (:world final-state)]
              (when-let [m (:gpu-mesh chunk)]
                (mesh/destroy-mesh! ctx m))
              (when-let [m (:water-gpu-mesh chunk)]
                (mesh/destroy-mesh! ctx m)))
            (mesh/destroy-mesh! ctx (:crosshair-mesh final-state))
            (when-let [m (:hud-mesh final-state)]
              (mesh/destroy-mesh! ctx m))
            (when-let [m (:mob-mesh final-state)]
              (mesh/destroy-mesh! ctx m))
            (pipeline/destroy-graphics-pipeline! (:device ctx) (:terrain-pipe final-state))
            (pipeline/destroy-graphics-pipeline! (:device ctx) (:water-pipe final-state))
            (pipeline/destroy-graphics-pipeline! (:device ctx) (:sky-pipe final-state))
            (pipeline/destroy-graphics-pipeline! (:device ctx) (:crosshair-pipe final-state))
            (pipeline/destroy-graphics-pipeline! (:device ctx) (:hud-pipe final-state))
            (texture/destroy-texture-array! (:device ctx) (:allocator ctx) (:tex-array final-state))
            (frame/destroy-frame-sync! ctx sync)
            (input/destroy-input-system! inp)
            (stop-chunk-gen-pool!)
            (reset! game-window nil)
            (println "Valley stopped."))))))
    (catch Throwable t
      (let [sw (java.io.StringWriter.)
            pw (java.io.PrintWriter. sw)]
        (.printStackTrace t pw)
        (let [info {:time (java.time.Instant/now)
                    :error (.getMessage t)
                    :type (str (class t))
                    :stacktrace (str sw)}]
          (reset! last-crash info)
          (spit "/tmp/valley_crash.log"
            (str "\n=== CRASH " (:time info) " ===\n"
                 (:type info) ": " (:error info) "\n"
                 (:stacktrace info) "\n")
            :append true)
          (println "CRASH:" (:type info) "-" (:error info))
          (println "Full trace saved to /tmp/valley_crash.log")
          (println "Check (deref valley.game/last-crash) for details"))))))

(defn start! []
  (when-let [t @game-thread]
    (when (.isAlive ^Thread t)
      (println "Game already running. Use (stop!) first.")
      (throw (ex-info "Game already running" {}))))
  (let [t (Thread. ^Runnable run-game! "valley-game")]
    (.setDaemon t true)
    (.start t)
    (reset! game-thread t)
    (println "Game started on background thread.")
    :started))

(defn stop! []
  (when-let [w @game-window]
    (GLFW/glfwSetWindowShouldClose w true)
    (println "Requested window close."))
  (when-let [t @game-thread]
    (when (.isAlive ^Thread t)
      (.join ^Thread t 3000))
    (reset! game-thread nil))
  :stopped)
