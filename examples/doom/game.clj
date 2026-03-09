(ns doom.game
  "Doom WAD renderer — full game loop with sky, sprites, monsters, weapons,
  sectors, HUD, sound, and jumping."
  (:require
   [doom.wad :as wad]
   [doom.level :as level]
   [doom.bsp :as bsp]
   [doom.sprite :as sprite]
   [doom.player :as player]
   [doom.monster :as monster]
   [doom.sector :as sector]
   [doom.hud :as hud]
   [doom.sound :as sound]
   [doom.music :as music]
   [raster.vk.gpu :as gpu]
   [raster.vk.input :as input]
   [raster.vk.frame :as frame]
   [raster.vk.mesh :as mesh]
   [raster.vk.pipeline :as pipeline]
   [raster.vk.texture :as texture]
   [raster.vk.camera :as camera]
   [raster.vk.audio :as audio])
  (:import
   [org.lwjgl.vulkan VK13 VkCommandBuffer]
   [org.lwjgl.system MemoryStack]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Constants
;; ================================================================

(def ^:const W 1280)
(def ^:const H 720)
(def ^:const PLAYER-HEIGHT 41.0)  ;; viewheight (eye level above feet)
(def ^:const PLAYER-BODY-HEIGHT 56.0) ;; full player body height for collision
(def ^:const PLAYER-RADIUS 16.0)
(def ^:const MOVE-SPEED 6.0)
(def ^:const MOUSE-SENSITIVITY 0.15)
(def ^:const GRAVITY 800.0) ;; Doom units/sec^2
(def ^:const JUMP-VELOCITY 280.0) ;; Doom units/sec

;; ================================================================
;; Shaders
;; ================================================================

(def vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;
};

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in float inLight;
layout(location = 3) in float inTexLayer;
layout(location = 4) in vec2 inTexScale;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float fragLight;
layout(location = 2) out float fragTexLayer;
layout(location = 3) out vec2 fragTexScale;

void main() {
    gl_Position = vp * vec4(inPos, 1.0);
    fragUV = inUV;
    fragLight = inLight;
    fragTexLayer = inTexLayer;
    fragTexScale = inTexScale;
}
")

(def frag-textured-glsl
  "#version 450

layout(set = 0, binding = 0) uniform sampler2DArray texArray;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float fragLight;
layout(location = 2) in float fragTexLayer;
layout(location = 3) in vec2 fragTexScale;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 dx = dFdx(fragUV) * fragTexScale;
    vec2 dy = dFdy(fragUV) * fragTexScale;
    vec2 uv = fract(fragUV) * fragTexScale;
    vec4 texColor = textureGrad(texArray, vec3(uv, fragTexLayer), dx, dy);
    if (texColor.a < 0.1) discard;
    float l = fragLight * fragLight;
    outColor = vec4(texColor.rgb * l, texColor.a);
}
")

(def frag-wireframe-glsl
  "#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float fragLight;
layout(location = 2) in float fragTexLayer;
layout(location = 3) in vec2 fragTexScale;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.0, fragLight, 0.0, 1.0);
}
")

;; Sky shader — fullscreen triangle, cylindrical projection
(def sky-vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;    // unused for sky, but must match layout
};

layout(location = 0) out vec2 fragUV;

void main() {
    // Fullscreen triangle trick (3 vertices, no vertex buffer)
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.9999, 1.0);
    fragUV = pos;
}
")

(def sky-frag-glsl
  "#version 450

layout(set = 0, binding = 0) uniform sampler2DArray texArray;

layout(push_constant) uniform PushConstants {
    mat4 vp;
    float skyYaw;
    float skyPitch;
    float skyLayer;
    float skyScaleU;
    float skyScaleV;
};

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    // Doom sky: 256 columns = 360 degrees, wraps horizontally
    // skyYaw is in radians, convert to [0,1] fraction of full rotation
    float u = fragUV.x * 0.25 + skyYaw / (2.0 * 3.14159265);
    // Vertical: screen top=0, bottom=1
    // Map full screen height to full sky texture height
    // skyPitch shifts the sky up/down when looking up/down
    float v = fragUV.y - skyPitch * 0.3;
    // Horizontal: wrap (fract), scale to valid texture portion
    // Vertical: clamp to [0,1], scale to valid texture portion
    float texU = fract(u) * skyScaleU;
    float texV = clamp(v, 0.0, 1.0) * skyScaleV;
    outColor = texture(texArray, vec3(texU, texV, skyLayer));
}
")

;; HUD shader — screen-space overlay with orthographic projection
(def hud-vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 vp;  // unused for HUD
};

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in float inLight;
layout(location = 3) in float inTexLayer;
layout(location = 4) in vec2 inTexScale;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float fragLight;
layout(location = 2) out float fragTexLayer;
layout(location = 3) out vec2 fragTexScale;

void main() {
    // inPos.xy are NDC coords [-1,1]
    gl_Position = vec4(inPos.xy, 0.0, 1.0);
    fragUV = inUV;
    fragLight = inLight;
    fragTexLayer = inTexLayer;
    fragTexScale = inTexScale;
}
")

;; ================================================================
;; Texture array building (enhanced: includes sprites, sky, HUD)
;; ================================================================

(defn- build-texture-images
  "Extract all WAD textures, flats, sprites, sky, and HUD images."
  [wad-data map-data]
  (let [palette (wad/parse-playpal wad-data)
        pnames (wad/parse-pnames wad-data)
        tex-defs (concat (or (wad/parse-texture-defs wad-data "TEXTURE1") [])
                         (or (wad/parse-texture-defs wad-data "TEXTURE2") []))
        tex-by-name (into {} (map (fn [td] [(:name td) td]) tex-defs))
        sidedefs (:sidedefs map-data)
        sectors (:sectors map-data)
        wall-tex-names (->> sidedefs
                            (mapcat (fn [sd] [(:upper sd) (:lower sd) (:middle sd)]))
                            (remove #(or (nil? %) (= % "-")))
                            (into #{}))
        flat-names (->> sectors
                        (mapcat (fn [s] [(:floor-tex s) (:ceil-tex s)]))
                        (remove #(or (nil? %) (= % "-") (= % "F_SKY1")))
                        (into #{})
                        ;; Add animation frames that might not be directly referenced
                        (into (into #{} ["NUKAGE1" "NUKAGE2" "NUKAGE3"])))
        images (atom [])
        missing (atom #{})]
    (println "Loading textures:" (count wall-tex-names) "wall textures,"
             (count flat-names) "flats")
    ;; Wall textures
    (doseq [name wall-tex-names]
      (if-let [td (get tex-by-name name)]
        (try
          (let [rgba (wad/compose-texture wad-data td pnames palette)]
            (swap! images conj {:name name :pixels rgba
                                :width (:width td) :height (:height td)}))
          (catch Exception e
            (println "Warning: failed to compose texture" name (.getMessage e))
            (swap! missing conj name)))
        (swap! missing conj name)))
    ;; Flat textures
    (doseq [name flat-names]
      (try
        (if-let [rgba (wad/decode-flat wad-data name palette)]
          (swap! images conj {:name name :pixels rgba :width 64 :height 64})
          (swap! missing conj name))
        (catch Exception e
          (println "Warning: failed to decode flat" name (.getMessage e))
          (swap! missing conj name))))
    ;; Log missing textures
    (when (seq @missing)
      (println "Missing textures:" @missing))
    ;; Sky texture
    (when-let [sky (wad/compose-sky-texture wad-data palette)]
      (swap! images conj {:name "SKY1" :pixels (:pixels sky)
                          :width (:width sky) :height (:height sky)}))
    ;; Sprite textures
    (let [sprite-imgs (sprite/build-sprite-images wad-data map-data palette)]
      (println "Loading" (count sprite-imgs) "sprite images")
      (swap! images into sprite-imgs))
    ;; HUD textures
    (let [hud-imgs (hud/build-hud-images wad-data palette)]
      (println "Loading" (count hud-imgs) "HUD images")
      (swap! images into hud-imgs))
    @images))

;; ================================================================
;; Initialization
;; ================================================================

(defn init-doom
  "Initialize Doom renderer with all gameplay systems."
  [ctx ^String wad-path ^String map-name & {:keys [textured?] :or {textured? true}}]
  (let [device (:device ctx)
        _allocator (:allocator ctx)
        sc (:swapchain-info ctx)
        depth-format VK13/VK_FORMAT_D32_SFLOAT

        _ (println "Loading WAD:" wad-path)
        wad-data (wad/load-wad wad-path)
        _ (println "Available maps:" (wad/list-maps wad-data))
        map-data (wad/load-map wad-data map-name)
        _ (when-not map-data (throw (ex-info "Map not found" {:map map-name})))
        _ (println "Loaded" map-name ":"
                   (count (:vertexes map-data)) "vertices,"
                   (count (:linedefs map-data)) "linedefs,"
                   (count (:sectors map-data)) "sectors,"
                   (count (:things map-data)) "things")

        ;; Texture array
        tex-images (when textured? (build-texture-images wad-data map-data))
        tex-pool (when textured? (texture/create-texture-descriptor-pool device 4))
        tex-layout (when textured? (texture/create-texture-descriptor-layout device))
        tex-array (when (and textured? (seq tex-images))
                    (texture/create-texture-array ctx tex-images :filter-mode :nearest))
        tex-desc-set (when (and tex-array tex-pool tex-layout)
                       (texture/allocate-texture-array-descriptor-set
                         device tex-pool tex-layout tex-array))

        ;; Level geometry
        _ (println "Building geometry...")
        geom (if textured?
               (level/build-level-geometry map-data
                 :tex-array-map (when tex-array (:layer-map tex-array)))
               (level/build-wireframe-lines map-data))
        _ (println "Geometry:" (/ (alength ^floats (:vertices geom)) level/FLOATS-PER-VERTEX) "vertices,"
                   (alength ^ints (:indices geom)) "indices")
        ;; Dynamic mesh for level (rebuilt when doors/lifts move)
        level-mesh (mesh/create-dynamic-mesh ctx 20000 30000 level/VERTEX-STRIDE)
        level-mesh (mesh/update-dynamic-mesh! ctx level-mesh
                     (:vertices geom) (:indices geom))

        ;; Shared pipeline layout (push constants + texture descriptor)
        pipe-layout (if textured?
                      (texture/create-textured-pipeline-layout device
                        {:size 84 :stages (bit-or VK13/VK_SHADER_STAGE_VERTEX_BIT
                                                  VK13/VK_SHADER_STAGE_FRAGMENT_BIT)})
                      {:layout (pipeline/create-graphics-pipeline-layout device
                                 {:size 64 :stages VK13/VK_SHADER_STAGE_VERTEX_BIT})})

        vertex-bindings [{:binding 0 :stride level/VERTEX-STRIDE}]
        vertex-attributes [{:location 0 :binding 0
                            :format VK13/VK_FORMAT_R32G32B32_SFLOAT :offset 0}
                           {:location 1 :binding 0
                            :format VK13/VK_FORMAT_R32G32_SFLOAT :offset 12}
                           {:location 2 :binding 0
                            :format VK13/VK_FORMAT_R32_SFLOAT :offset 20}
                           {:location 3 :binding 0
                            :format VK13/VK_FORMAT_R32_SFLOAT :offset 24}
                           {:location 4 :binding 0
                            :format VK13/VK_FORMAT_R32G32_SFLOAT :offset 28}]

        ;; Level pipeline (depth test + depth write)
        level-pipe (pipeline/create-graphics-pipeline device
                     {:vert-glsl vert-glsl
                      :frag-glsl (if textured? frag-textured-glsl frag-wireframe-glsl)
                      :topology (if textured? :triangles :lines)
                      :color-format (:format sc)
                      :depth-format depth-format
                      :cull-mode :none
                      :polygon-mode :fill
                      :blend? false
                      :layout (:layout pipe-layout)
                      :vertex-bindings vertex-bindings
                      :vertex-attributes vertex-attributes})

        ;; Sprite pipeline (depth test ON, alpha blend, no depth write via discard)
        sprite-pipe (when textured?
                      (pipeline/create-graphics-pipeline device
                        {:vert-glsl vert-glsl
                         :frag-glsl frag-textured-glsl
                         :topology :triangles
                         :color-format (:format sc)
                         :depth-format depth-format
                         :cull-mode :none
                         :polygon-mode :fill
                         :blend? true
                         :layout (:layout pipe-layout)
                         :vertex-bindings vertex-bindings
                         :vertex-attributes vertex-attributes}))

        ;; HUD pipeline (no depth test, alpha blend)
        hud-pipe (when textured?
                   (pipeline/create-graphics-pipeline device
                     {:vert-glsl hud-vert-glsl
                      :frag-glsl frag-textured-glsl
                      :topology :triangles
                      :color-format (:format sc)
                      :depth-format depth-format
                      :depth-test? false
                      :depth-write? false
                      :cull-mode :none
                      :polygon-mode :fill
                      :blend? true
                      :layout (:layout pipe-layout)
                      :vertex-bindings vertex-bindings
                      :vertex-attributes vertex-attributes}))

        ;; Sky pipeline (no depth write, drawn first, no vertex input)
        sky-pipe (when textured?
                   (pipeline/create-graphics-pipeline device
                     {:vert-glsl sky-vert-glsl
                      :frag-glsl sky-frag-glsl
                      :topology :triangles
                      :color-format (:format sc)
                      :depth-format depth-format
                      :depth-test? false
                      :depth-write? false
                      :cull-mode :none
                      :polygon-mode :fill
                      :blend? false
                      :layout (:layout pipe-layout)}))

        ;; Dynamic mesh for sprites (updated each frame)
        sprite-mesh (when textured?
                      (mesh/create-dynamic-mesh ctx 2000 3000 level/VERTEX-STRIDE))

        ;; Dynamic mesh for HUD
        hud-mesh (when textured?
                   (mesh/create-dynamic-mesh ctx 400 600 level/VERTEX-STRIDE))

        ;; Sprite data
        sprite-map (when textured? (wad/parse-sprites wad-data))
        sprite-tex-map (when (and textured? tex-array)
                         (:layer-map tex-array))

        ;; Camera setup
        player-thing (wad/player-start map-data)
        _ (println "Player start:" player-thing)
        spawn-x (if player-thing (double (:x player-thing)) 0.0)
        spawn-y (if player-thing (double (:y player-thing)) 0.0)
        spawn-angle (if player-thing (* (double (:angle player-thing)) (/ Math/PI 180.0)) 0.0)
        cam-x (level/doom->vk-x spawn-x)
        cam-z (level/doom->vk-z spawn-y)
        spawn-sector-idx (bsp/point-sector map-data spawn-x spawn-y)
        spawn-floor (if spawn-sector-idx
                      (double (:floor-height (nth (:sectors map-data) spawn-sector-idx)))
                      0.0)
        cam-y (level/doom->vk-y (+ spawn-floor PLAYER-HEIGHT))
        cam (-> (camera/create-fps-camera [cam-x cam-y cam-z]
                  (- spawn-angle (/ Math/PI 2.0)) 0.0)
                (assoc :aspect (/ (double W) (double H))))

        ;; Game state initialization
        player-state (player/create-player-state)
        monsters (monster/init-monsters map-data)
        sector-states (sector/init-sector-states map-data)

        ;; Audio
        audio-ctx (try (audio/create-audio-context) (catch Exception _ nil))
        _ (when audio-ctx
            (println "Audio initialized")
            (sound/load-all-sounds! audio-ctx wad-data))

        ;; Music
        music-seq (try (music/play-music! wad-data (str "D_" map-name))
                       (catch Exception e
                         (println "Music init failed:" (.getMessage e))
                         nil))

        ;; Sky layer info
        sky-info (when (and tex-array (get (:layer-map tex-array) "SKY1"))
                   (get (:layer-map tex-array) "SKY1"))]

    (println "Camera at:" (:pos cam) "yaw:" (:yaw cam))
    (println "Monsters:" (count monsters) "Sector specials:" (count sector-states))

    {:wad-data wad-data
     :map-data map-data
     :level-mesh level-mesh
     :tex-array-map (when tex-array (:layer-map tex-array))
     :level-pipe level-pipe
     :sprite-pipe sprite-pipe
     :hud-pipe hud-pipe
     :sky-pipe sky-pipe
     :pipe-layout pipe-layout
     :tex-pool tex-pool
     :tex-layout tex-layout
     :tex-array tex-array
     :tex-desc-set tex-desc-set
     :textured? textured?
     :sprite-mesh sprite-mesh
     :hud-mesh hud-mesh
     :sprite-map sprite-map
     :sprite-tex-map sprite-tex-map
     :sky-info sky-info
     :cam cam
     :spawn-cam cam  ;; remember spawn camera for respawn
     :player-state player-state
     :monsters monsters
     :projectiles []
     :sector-states sector-states
     :active-things (vec (:things map-data))
     :audio-ctx audio-ctx
     :music-seq music-seq
     :cursor-captured? false
     :mode :fps
     ;; Volume (0-15, like Doom)
     :sfx-volume 12
     ;; Jumping
     :vy 0.0          ;; vertical velocity in Doom units/sec
     :on-ground? true
     :death-timer 0.0
     :anim-tic 0.0}))

;; ================================================================
;; Update
;; ================================================================

(defn- fps-move-only
  "Apply only WASD movement to FPS camera (no mouse look)."
  [cam input ^double dt ^double speed]
  (let [yaw (double (:yaw cam))
        fx (Math/sin yaw)
        fz (- (Math/cos yaw))
        rx (Math/cos yaw)
        rz (Math/sin yaw)
        move-f (cond (input/key-down? input :w) 1.0
                     (input/key-down? input :s) -1.0
                     :else 0.0)
        move-r (cond (input/key-down? input :d) 1.0
                     (input/key-down? input :a) -1.0
                     :else 0.0)
        s (* speed dt)
        [px py pz] (:pos cam)
        px (double px) py (double py) pz (double pz)]
    (assoc cam :pos [(+ px (* fx move-f s) (* rx move-r s))
                     py
                     (+ pz (* fz move-f s) (* rz move-r s))])))

(defn update-doom [state input ^double dt]
  (let [;; ESC toggles menu
        menu-open? (:menu-open? state)
        menu-open? (if (input/key-pressed? input :escape)
                     (not menu-open?) menu-open?)
        menu-sel (long (or (:menu-sel state) 0))]

    ;; If menu is open, handle menu input only
    (if menu-open?
      (let [sfx-volume (long (or (:sfx-volume state) 12))
            mus-volume (long (or (:mus-volume state) 12))
            ;; Menu items: 0=SFX Volume, 1=Music Volume
            menu-sel (cond
                       (input/key-pressed? input :up) (max 0 (dec menu-sel))
                       (input/key-pressed? input :down) (min 1 (inc menu-sel))
                       :else menu-sel)
            ;; Adjust volume with left/right on selected item
            [sfx-volume mus-volume]
            (cond
              (= menu-sel 0)
              [(cond (input/key-pressed? input :left) (max 0 (dec sfx-volume))
                     (input/key-pressed? input :right) (min 15 (inc sfx-volume))
                     :else sfx-volume)
               mus-volume]
              (= menu-sel 1)
              [sfx-volume
               (cond (input/key-pressed? input :left) (max 0 (dec mus-volume))
                     (input/key-pressed? input :right) (min 15 (inc mus-volume))
                     :else mus-volume)]
              :else [sfx-volume mus-volume])]
        ;; Apply music volume change
        (when (not= mus-volume (long (or (:mus-volume state) 12)))
          (music/set-music-volume! (:music-seq state) mus-volume))
        ;; menu-open? was toggled by ESC at the top
        (assoc state :menu-open? menu-open? :menu-sel menu-sel
               :sfx-volume sfx-volume :mus-volume mus-volume))

      ;; Normal gameplay
      (let [cam (:cam state)
            mode (:mode state)

            ;; Toggle cursor capture with Tab
            captured? (:cursor-captured? state)
            toggle-capture? (input/key-pressed? input :tab)
            captured? (if toggle-capture? (not captured?) captured?)

            ;; Toggle mode with F1
            toggle-mode? (input/key-pressed? input :f1)
            mode (if toggle-mode?
                   (case mode :fps :orbit :orbit :fps)
                   mode)
            cam (if toggle-mode?
                  (case mode
                    :orbit (-> (camera/create-orbit-camera [0 0 0] 100.0 0.0 0.8)
                               (assoc :aspect (/ (double W) (double H))))
                    :fps (-> (camera/create-fps-camera (or (:pos cam) [0 1 0]) 0.0 0.0)
                             (assoc :aspect (/ (double W) (double H)))))
                  cam)

            ;; Volume control: +/- keys (like Doom's F4 menu)
            sfx-volume (long (or (:sfx-volume state) 12))
            sfx-volume (cond
                         (input/key-pressed? input :equal) (min 15 (inc sfx-volume))
                         (input/key-pressed? input :minus) (max 0 (dec sfx-volume))
                         :else sfx-volume)

        ;; Mouse look — apply ONLY on first fixed-step of this frame
        cur-mouse-pos (when input (:mouse-pos input))
        prev-mouse-pos (:last-mouse-pos state)
        first-step? (not= cur-mouse-pos prev-mouse-pos)

        cam (case mode
              :fps (if captured?
                     (if first-step?
                       (camera/fps-camera-update cam input dt MOVE-SPEED MOUSE-SENSITIVITY)
                       (fps-move-only cam input dt MOVE-SPEED))
                     cam)
              :orbit (camera/orbit-camera-update cam input dt 2.0 20.0))

        ;; FPS mode: collision detection, floor tracking, jumping
        old-cam (:cam state)
        vy (double (or (:vy state) 0.0))
        on-ground? (boolean (:on-ground? state))

        [cam vy on-ground?]
        (if (and (= mode :fps) captured?)
          (let [map-data (:map-data state)
                sector-states (:sector-states state)
                ;; Old position in Doom coords
                old-doom-x (/ (double (nth (:pos old-cam) 0)) level/SCALE)
                old-doom-y (/ (- (double (nth (:pos old-cam) 2))) level/SCALE)
                ;; New position in Doom coords
                new-doom-x (/ (double (nth (:pos cam) 0)) level/SCALE)
                new-doom-y (/ (- (double (nth (:pos cam) 2))) level/SCALE)
                ddx (- new-doom-x old-doom-x)
                ddy (- new-doom-y old-doom-y)
                sector-overrides (sector/get-sector-overrides sector-states (:sectors map-data))
                [final-doom-x final-doom-y]
                (bsp/move-and-slide map-data [old-doom-x old-doom-y]
                                    [ddx ddy] PLAYER-RADIUS
                                    :sector-overrides sector-overrides)
                final-x (level/doom->vk-x final-doom-x)
                final-z (level/doom->vk-z final-doom-y)
                ;; Floor tracking with sector overrides (doors/lifts)
                sector-idx (bsp/point-sector map-data final-doom-x final-doom-y)
                base-floor (if sector-idx
                             (double (:floor-height (nth (:sectors map-data) sector-idx)))
                             0.0)
                override (when sector-idx (get sector-states sector-idx))
                floor-h (if (and override (:floor-height override))
                          (double (:floor-height override))
                          base-floor)
                ;; Ceiling tracking
                base-ceil (if sector-idx
                            (double (:ceil-height (nth (:sectors map-data) sector-idx)))
                            256.0)
                ceil-h (if (and override (:ceil-height override))
                         (double (:ceil-height override))
                         base-ceil)
                ;; Jumping physics
                current-doom-h (/ (double (nth (:pos old-cam) 1)) level/SCALE)
                current-feet (- current-doom-h PLAYER-HEIGHT)
                ;; Apply jump if on ground and space pressed
                vy (if (and on-ground? (input/key-down? input :space))
                     JUMP-VELOCITY
                     vy)
                ;; Apply gravity
                vy (- vy (* GRAVITY dt))
                new-feet (+ current-feet (* vy dt))
                ;; Ceiling collision (top of player body hits ceiling)
                new-head (+ new-feet PLAYER-BODY-HEIGHT)
                [new-feet vy] (if (>= new-head ceil-h)
                                [(- ceil-h PLAYER-BODY-HEIGHT) (min vy 0.0)]
                                [new-feet vy])
                ;; Floor collision
                [new-feet vy on-ground?]
                (if (<= new-feet floor-h)
                  [floor-h 0.0 true]
                  [new-feet vy false])
                target-y (level/doom->vk-y (+ new-feet PLAYER-HEIGHT))]
            [(assoc cam :pos [final-x target-y final-z]) vy on-ground?])
          [cam vy on-ground?])

        ;; Player position in Doom coords (for gameplay systems)
        px (/ (double (nth (:pos cam) 0)) level/SCALE)
        py (/ (- (double (nth (:pos cam) 2))) level/SCALE)

        ;; Use key (E) — activate doors, switches, lifts
        sector-states (:sector-states state)
        sector-states (if (and (input/key-pressed? input :e) (= mode :fps))
                        (sector/try-use-line (:map-data state) sector-states
                                             px py (double (:yaw cam))
                                             (:player-state state))
                        sector-states)

        ;; Update sector states (doors, lifts, lights)
        sector-states (sector/update-sector-states sector-states dt)

        ;; Damage floors
        floor-dmg (sector/check-damage-floor sector-states (:map-data state) px py)
        player-state (if (pos? floor-dmg)
                       (player/apply-damage (:player-state state) floor-dmg)
                       (:player-state state))

        ;; Sound: door activation
        _ (when (and (:audio-ctx state) (input/key-pressed? input :e) (= mode :fps))
            (sound/play-ui! (:audio-ctx state) :door-open sfx-volume))

        ;; Fire weapon
        fire? (and captured? (= mode :fps) (input/mouse-down? input :left))
        ;; Weapon switch (1-7 keys)
        switch-weapon (cond
                        (input/key-pressed? input :1) :fist
                        (input/key-pressed? input :2) :pistol
                        (input/key-pressed? input :3) :shotgun
                        (input/key-pressed? input :4) :chaingun
                        (input/key-pressed? input :5) :rocket-launcher
                        (input/key-pressed? input :6) :chainsaw
                        :else nil)
        weapon-input {:fire? fire? :switch-weapon switch-weapon}
        player-state (player/update-weapon player-state weapon-input)

        ;; Weapon fire sound
        _ (when (and fire? (:audio-ctx state)
                    (= (:weapon-state player-state) :firing)
                    (= (:weapon-frame player-state) 0))
            (let [snd (case (:current-weapon player-state)
                        :pistol :pistol :shotgun :shotgun
                        :chaingun :pistol :chainsaw :pistol
                        :pistol)]
              (sound/play-ui! (:audio-ctx state) snd sfx-volume)))

        ;; Pickups
        active-things (:active-things state)
        _prev-items (:items player-state)
        [player-state active-things]
        (player/check-pickups player-state active-things px py)
        ;; Pickup sound
        _ (when (and (:audio-ctx state) (not= (count active-things) (count (:active-things state))))
            (sound/play-ui! (:audio-ctx state) :pickup-item sfx-volume))

        ;; Monster AI
        old-monsters (:monsters state)
        [monsters player-state new-projs]
        (monster/update-monsters old-monsters (:map-data state)
                                  player-state px py dt)

        ;; Projectiles
        projectiles (:projectiles state)
        projectiles (into projectiles new-projs)
        [projectiles player-state]
        (monster/update-projectiles projectiles (:map-data state) player-state px py dt)

        ;; Hitscan attack when weapon fires
        player-state
        (if (and fire? (= (:weapon-state player-state) :firing)
                 (= (:weapon-frame player-state) 0))
          (let [w (get player/weapons (:current-weapon player-state))
                yaw (double (:yaw cam))
                ;; Ray direction in Doom coords
                rdx (Math/cos (- (/ Math/PI 2.0) yaw))
                rdy (Math/sin (- (/ Math/PI 2.0) yaw))
                max-range (double (or (:range w) 2048))
                ;; Check monster hits — find closest monster in aim cone
                hit-monster-idx
                (when (:hitscan? w)
                  (let [candidates
                        (keep-indexed
                          (fn [i m]
                            (when (and (not= (:state m) :dead)
                                       (not= (:state m) :death))
                              (let [mx (double (:x m))
                                    my (double (:y m))
                                    dx (- mx px) dy (- my py)
                                    dist (Math/sqrt (+ (* dx dx) (* dy dy)))
                                    dot (+ (* rdx (/ dx (max dist 1.0)))
                                           (* rdy (/ dy (max dist 1.0))))]
                                (when (and (< dist max-range) (> dot 0.7))
                                  [i dist]))))
                          monsters)]
                    (when (seq candidates)
                      (first (apply min-key second candidates)))))]
            (if hit-monster-idx
              (let [dmg (player/roll-damage (:damage w))
                    m (nth monsters hit-monster-idx)
                    m' (monster/damage-monster m dmg)]
                (assoc player-state :_hit-monster [hit-monster-idx m']))
              player-state))
          player-state)

        ;; Apply monster hit
        monsters (if-let [[idx m'] (:_hit-monster player-state)]
                   (assoc monsters idx m')
                   monsters)
        player-state (dissoc player-state :_hit-monster)

        ;; Sound propagation — firing alerts monsters in connected sectors
        monsters (if (and fire? (= (:weapon-state player-state) :firing))
                   (let [sound-sectors (monster/propagate-sound (:map-data state) px py)]
                     (if sound-sectors
                       (monster/alert-monsters-by-sound monsters (:map-data state) sound-sectors)
                       monsters))
                   monsters)

        ;; Monster sound effects (sight, pain, death)
        _ (when-let [ac (:audio-ctx state)]
            (dotimes [i (min (count old-monsters) (count monsters))]
              (let [old-st (:state (nth old-monsters i))
                    new-m (nth monsters i)
                    new-st (:state new-m)
                    mx (double (:x new-m))
                    my (double (:y new-m))]
                (when (not= old-st new-st)
                  (let [is-imp? (= (:type new-m) 3001)
                        snd (cond
                              (and (= old-st :idle) (= new-st :chase))
                              (if is-imp? :imp-sight
                                (if (zero? (rand-int 2)) :monster-sight-1 :monster-sight-2))
                              (= new-st :pain)
                              (if is-imp? :imp-pain :monster-pain)
                              (= new-st :death)
                              (if is-imp? :imp-death
                                (if (zero? (rand-int 2)) :monster-death-1 :monster-death-2))
                              (= new-st :attack)
                              (when is-imp? :imp-attack)
                              :else nil)]
                    (when snd
                      (sound/play-at! ac snd mx my 0.0 sfx-volume)))))))

        ;; Audio listener update
        _ (when-let [ac (:audio-ctx state)]
            (sound/update-listener! ac cam))

        ;; Death/respawn handling
        death-timer (double (or (:death-timer state) 0.0))
        [player-state cam vy on-ground? death-timer monsters projectiles]
        (if (not (player/alive? player-state))
          (let [new-timer (+ death-timer dt)]
            (if (> new-timer 2.0)
              ;; Respawn: reset player, camera, clear projectiles
              [(player/create-player-state)
               (or (:spawn-cam state) cam)
               0.0 true 0.0
               ;; Reset monster states to idle
               (mapv (fn [m] (if (not= (:state m) :dead)
                               (assoc m :state :idle :tics (+ 10 (rand-int 30)) :active? false)
                               m))
                     monsters)
               []]
              [player-state cam vy on-ground? new-timer monsters projectiles]))
          [player-state cam vy on-ground? 0.0 monsters projectiles])]

    (assoc state
           :cam cam
           :mode mode
           :menu-open? false
           :cursor-captured? captured?
           :last-mouse-pos cur-mouse-pos
           :vy vy
           :on-ground? on-ground?
           :player-state player-state
           :monsters monsters
           :projectiles projectiles
           :sector-states sector-states
           :active-things active-things
           :death-timer death-timer
           :sfx-volume sfx-volume
           :anim-tic (+ (double (or (:anim-tic state) 0.0)) (* dt 35.0)))))))

;; ================================================================
;; Render
;; ================================================================

(defn- push-vp! [^VkCommandBuffer cb ^long layout ^floats vp]
  (let [stack (MemoryStack/stackPush)
        buf (.mallocFloat stack 16)]
    (.put buf vp)
    (.flip buf)
    (VK13/vkCmdPushConstants cb layout
      (bit-or VK13/VK_SHADER_STAGE_VERTEX_BIT VK13/VK_SHADER_STAGE_FRAGMENT_BIT)
      0 buf)
    (MemoryStack/stackPop)))

(defn- push-sky-constants! [^VkCommandBuffer cb layout
                            ^floats vp yaw pitch
                            sky-layer sky-su sky-sv]
  (let [layout (long layout)
        yaw (float yaw) pitch (float pitch)
        sky-layer (float sky-layer) sky-su (float sky-su) sky-sv (float sky-sv)
        stack (MemoryStack/stackPush)
        ;; mat4 (64 bytes) + 5 floats (20 bytes) = 84 bytes
        buf (.malloc stack 84)]
    ;; VP matrix (16 floats = 64 bytes)
    (dotimes [i 16]
      (.putFloat buf (* i 4) (aget vp i)))
    ;; Sky params
    (.putFloat buf 64 yaw)
    (.putFloat buf 68 pitch)
    (.putFloat buf 72 sky-layer)
    (.putFloat buf 76 sky-su)
    (.putFloat buf 80 sky-sv)
    (VK13/vkCmdPushConstants cb layout
      (bit-or VK13/VK_SHADER_STAGE_VERTEX_BIT VK13/VK_SHADER_STAGE_FRAGMENT_BIT)
      0 buf)
    (MemoryStack/stackPop)))

(defn render-doom [state ^VkCommandBuffer cb ctx _frame-info]
  (let [{:keys [level-mesh level-pipe sprite-pipe hud-pipe sky-pipe
                pipe-layout tex-desc-set textured? cam
                sprite-mesh hud-mesh sprite-map sprite-tex-map
                sky-info player-state sector-states]} state
        layout (long (:layout pipe-layout))
        vp (camera/camera-vp-matrix cam)
        ;; Rebuild level geometry with door/lift overrides + animated flats
        sector-overrides (when sector-states
                           (sector/get-sector-overrides sector-states
                             (:sectors (:map-data state))))
        ;; Animated flat cycling (NUKAGE1/2/3 every 8 tics)
        anim-tic (double (or (:anim-tic state) 0.0))
        nukage-frame (mod (int (/ anim-tic 8.0)) 3)
        nukage-name (str "NUKAGE" (inc nukage-frame))
        flat-remap {"NUKAGE1" nukage-name "NUKAGE2" nukage-name "NUKAGE3" nukage-name}
        level-mesh (if textured?
                     (let [geom (level/build-level-geometry
                                  (:map-data state)
                                  :tex-array-map (:tex-array-map state)
                                  :sector-overrides sector-overrides
                                  :flat-remap flat-remap)]
                       (mesh/update-dynamic-mesh! ctx level-mesh
                         (:vertices geom) (:indices geom)))
                     level-mesh)]

    ;; Bind texture array descriptor (shared by all pipelines)
    (when (and textured? tex-desc-set)
      (let [stack (MemoryStack/stackPush)]
        (VK13/vkCmdBindDescriptorSets cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                       layout 0
                                       (.longs stack (long tex-desc-set)) nil)
        (MemoryStack/stackPop)))

    ;; 1. Sky (drawn first, no depth test)
    (when (and sky-pipe sky-info)
      (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                              (long (:pipeline sky-pipe)))
      (push-sky-constants! cb layout vp
                           (double (:yaw cam)) (double (:pitch cam))
                           (double (:layer sky-info))
                           (double (:scale-u sky-info))
                           (double (:scale-v sky-info)))
      ;; Draw fullscreen triangle (3 vertices, no vertex buffer)
      (VK13/vkCmdDraw cb 3 1 0 0))

    ;; 2. Level geometry (depth test + depth write)
    (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                            (long (:pipeline level-pipe)))
    (push-vp! cb layout vp)
    (mesh/cmd-draw-mesh! cb level-mesh)

    ;; 3. Sprites (depth test, alpha blend)
    (when (and sprite-pipe sprite-mesh sprite-tex-map)
      (let [proj-entities (mapv (fn [p] {:x (:x p) :y (:y p)
                                         :sprite (or (:sprite p) "BAL1")
                                         :angle 0 :z (:z p)})
                                (:projectiles state))
            quads (sprite/build-sprite-quads
                    (:map-data state) sprite-map sprite-tex-map
                    (:pos cam) (:yaw cam)
                    :things (:active-things state)
                    :monsters (:monsters state)
                    :extra-entities proj-entities)]
        (when quads
          (let [updated-sprite-mesh
                (mesh/update-dynamic-mesh! ctx sprite-mesh
                                           (:vertices quads) (:indices quads))]
            (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                    (long (:pipeline sprite-pipe)))
            (push-vp! cb layout vp)
            (mesh/cmd-draw-mesh! cb updated-sprite-mesh)))))

    ;; 4. HUD overlay (no depth test, alpha blend)
    (when (and hud-pipe hud-mesh player-state sprite-tex-map)
      ;; Choose HUD or menu quads depending on menu state
      (let [hud-quads (if (:menu-open? state)
                        (hud/build-menu-quads sprite-tex-map
                          (long (or (:menu-sel state) 0))
                          (long (or (:sfx-volume state) 12))
                          (long (or (:mus-volume state) 12)))
                        (hud/build-hud-quads player-state sprite-tex-map))]
        (when hud-quads
          (let [updated-hud-mesh
                (mesh/update-dynamic-mesh! ctx hud-mesh
                                           (:vertices hud-quads) (:indices hud-quads))]
            (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                    (long (:pipeline hud-pipe)))
            (let [identity-vp (float-array [1 0 0 0, 0 1 0 0, 0 0 1 0, 0 0 0 1])]
              (push-vp! cb layout identity-vp))
            (mesh/cmd-draw-mesh! cb updated-hud-mesh)))))))

;; ================================================================
;; Main entry point
;; ================================================================

(defonce ^:private game-thread (atom nil))
(defonce ^:private game-window (atom nil))
(defonce game-state-ref (atom nil)) ;; public for REPL inspection

(defn press-key!
  "Simulate a key press from the REPL. Key is a keyword like :escape, :up, :left.
  Injects key-down + key-up events into the input system."
  [k]
  (when-let [inp (:input-system @game-state-ref)]
    (swap! (:events inp) conj [:key-down k])
    (Thread/sleep 30)
    (swap! (:events inp) conj [:key-up k])
    k))

(defn screenshot!
  "Take a screenshot, save to path. Returns path on success."
  ([] (screenshot! "/tmp/doom-screenshot.png"))
  ([^String path]
   (require 'raster.vk.screenshot)
   ((resolve 'raster.vk.screenshot/request-screenshot!) path)
   (Thread/sleep 500)
   path))

(defn- run-game!
  "Internal: run the game loop (blocking). Called on game thread."
  [wad map-name textured?]
  (println "=== DOOM on Raster ===")
  (println "Controls: WASD move, Mouse look, Space jump, E use/open")
  (println "          Tab toggle mouse, F1 toggle FPS/orbit, ESC quit")
  (println "          1-6 switch weapons, Left click fire")
  (gpu/with-gpu-context [ctx {:width W :height H
                              :title (str "Doom — " map-name)
                              :depth? true :debug? true}]
    (reset! game-window (:window ctx))
    (let [inp-sys (input/create-input-system (:window ctx))
          sync (frame/create-frame-sync ctx)
          state (init-doom ctx wad map-name :textured? textured?)]
      (camera/set-cursor-captured! (:window ctx) true)
      (try
        (frame/run-game-loop! ctx
          {:frame-sync sync
           :init-state (assoc state :cursor-captured? true :input-system inp-sys)
           :input-system inp-sys
           :update-fn #'update-doom
           :render-fn #'render-doom
           :clear-color [0.0 0.0 0.0 1.0]
           :title (str "Doom — " map-name)
           :state-atom game-state-ref})
        (finally
          (VK13/vkDeviceWaitIdle (:device ctx))
          ;; Cleanup
          (frame/destroy-frame-sync! ctx sync)
          (input/destroy-input-system! inp-sys)
          ;; Music cleanup
          (when (:music-seq state)
            (try (music/stop-music! (:music-seq state))
                 (catch Exception _)))
          ;; Audio cleanup
          (when (:audio-ctx state)
            (try (audio/destroy-audio-context! (:audio-ctx state))
                 (catch Exception _)))
          ;; Meshes
          (mesh/destroy-mesh! ctx (:level-mesh state))
          (when (:sprite-mesh state) (mesh/destroy-mesh! ctx (:sprite-mesh state)))
          (when (:hud-mesh state) (mesh/destroy-mesh! ctx (:hud-mesh state)))
          ;; Pipelines
          (let [device (:device ctx)]
            (when (:level-pipe state)
              (VK13/vkDestroyPipeline device (:pipeline (:level-pipe state)) nil)
              (when (:vert-module (:level-pipe state))
                (VK13/vkDestroyShaderModule device (:vert-module (:level-pipe state)) nil))
              (when (:frag-module (:level-pipe state))
                (VK13/vkDestroyShaderModule device (:frag-module (:level-pipe state)) nil)))
            (when (:sprite-pipe state)
              (VK13/vkDestroyPipeline device (:pipeline (:sprite-pipe state)) nil)
              (when (:vert-module (:sprite-pipe state))
                (VK13/vkDestroyShaderModule device (:vert-module (:sprite-pipe state)) nil))
              (when (:frag-module (:sprite-pipe state))
                (VK13/vkDestroyShaderModule device (:frag-module (:sprite-pipe state)) nil)))
            (when (:hud-pipe state)
              (VK13/vkDestroyPipeline device (:pipeline (:hud-pipe state)) nil)
              (when (:vert-module (:hud-pipe state))
                (VK13/vkDestroyShaderModule device (:vert-module (:hud-pipe state)) nil))
              (when (:frag-module (:hud-pipe state))
                (VK13/vkDestroyShaderModule device (:frag-module (:hud-pipe state)) nil)))
            (when (:sky-pipe state)
              (VK13/vkDestroyPipeline device (:pipeline (:sky-pipe state)) nil)
              (when (:vert-module (:sky-pipe state))
                (VK13/vkDestroyShaderModule device (:vert-module (:sky-pipe state)) nil))
              (when (:frag-module (:sky-pipe state))
                (VK13/vkDestroyShaderModule device (:frag-module (:sky-pipe state)) nil))))
          ;; Textures
          (when (:textured? state)
            (when (:tex-array state)
              (texture/destroy-texture-array! (:device ctx) (:allocator ctx) (:tex-array state)))
            (when (:tex-pool state)
              (VK13/vkDestroyDescriptorPool (:device ctx) (:tex-pool state) nil))
            (when (:tex-layout state)
              (VK13/vkDestroyDescriptorSetLayout (:device ctx) (:tex-layout state) nil)))
          ;; Pipeline layout
          (VK13/vkDestroyPipelineLayout (:device ctx) (:layout (:pipe-layout state)) nil)
          (println "Done."))))))

(defn start!
  "Launch Doom renderer on a background thread. REPL stays interactive.
  Options:
    :wad — path to WAD file (default: doom1.wad)
    :map — map name (default: E1M1)
    :textured? — render with textures (default: true)"
  [& {:keys [wad map textured?]
      :or {wad "/usr/share/games/doom/doom1.wad"
           map "E1M1"
           textured? true}}]
  (when-let [t @game-thread]
    (when (.isAlive ^Thread t)
      (println "Game already running. Use (stop!) first.")
      (throw (ex-info "Game already running" {}))))
  (let [t (Thread.
            (fn []
              (try
                (run-game! wad map textured?)
                (catch Exception e
                  (println "Game error:" (.getMessage e))
                  (.printStackTrace e))))
            "doom-game")]
    (.setDaemon t true)
    (.start t)
    (reset! game-thread t)
    (println "Game started on background thread.")
    :started))

(defn stop!
  "Close the game window, causing the game loop to exit cleanly."
  []
  (when-let [w @game-window]
    (org.lwjgl.glfw.GLFW/glfwSetWindowShouldClose w true)
    (println "Requested window close."))
  ;; Wait for thread to finish
  (when-let [t @game-thread]
    (when (.isAlive ^Thread t)
      (.join ^Thread t 3000))
    (reset! game-thread nil)
    (reset! game-window nil))
  :stopped)

(defn wireframe!
  "Quick-start wireframe mode."
  [& {:keys [wad map] :or {wad "/usr/share/games/doom/doom1.wad" map "E1M1"}}]
  (start! :wad wad :map map :textured? false))
