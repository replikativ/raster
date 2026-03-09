(ns asteroids.core
  "Geometric Asteroids — typed dispatch showcase.

  All shapes are regular polygons rendered as procedural textures.
  Splitting dispatches via deftm on concrete shape type:
    Octagon → Squares, Hexagon → Pentagons, Pentagon → Square+Triangle,
    Square → Triangles, Triangle → destroyed.

  Add new polygon types from the REPL and the game picks them up live."
  (:refer-clojure :exclude [defn])
  (:require
   [asteroids.shapes :as shapes]
   [raster.core :refer [defn]]
   [raster.vk.gpu :as gpu]
   [raster.vk.input :as input]
   [raster.vk.frame :as frame]
   [raster.vk.math :as math]
   [raster.vk.mem :as mem]
   [raster.vk.pipeline :as pipeline]
   [raster.vk.shader]
   [raster.vk.texture :as texture]
   [raster.vk.text :as text])
  (:import
   [org.lwjgl.vulkan VK13 VkCommandBuffer]
   [org.lwjgl.system MemoryStack MemoryUtil]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Constants
;; ================================================================

(def ^:const W 1024)
(def ^:const H 768)
(def ^:const MAX-QUADS 512)
(def ^:const FLOATS-PER-VERTEX 4)  ;; x y u v
(def ^:const VERTS-PER-QUAD 6)     ;; 2 triangles
(def ^:const VB-FLOATS (* MAX-QUADS VERTS-PER-QUAD FLOATS-PER-VERTEX))

(def ^:const SHIP-ACCEL 300.0)
(def ^:const SHIP-DRAG 0.99)
(def ^:const SHIP-TURN-SPEED 4.0)
(def ^:const BULLET-SPEED 500.0)
(def ^:const BULLET-TTL 1.5)
(def ^:const BULLET-COOLDOWN 0.15)

;; ================================================================
;; Shaders
;; ================================================================

(def vert-glsl
  "#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
};

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 fragUV;

void main() {
    gl_Position = mvp * vec4(inPos, 0.0, 1.0);
    fragUV = inUV;
}
")

(def frag-glsl
  "#version 450

layout(set = 0, binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 texColor = texture(texSampler, fragUV);
    if (texColor.a < 0.05) discard;
    outColor = texColor;
}
")

(def bullet-frag-glsl
  "#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.85, 0.85, 0.85, 1.0);
}
")

;; ================================================================
;; Quad building
;; ================================================================

(defn emit-quad!
  "Write 6 vertices (2 triangles) for a rotated textured quad into float array."
  [^floats verts offset cx cy hw hh angle u0 v0 u1 v1]
  (let [offset (long offset)
        cx (double cx) cy (double cy) hw (double hw) hh (double hh)
        angle (double angle)
        u0 (double u0) v0 (double v0) u1 (double u1) v1 (double v1)
        ca (Math/cos angle)
        sa (Math/sin angle)
        tlx (+ cx (- (* (- hw) ca) (* hh sa)))
        tly (+ cy (+ (* (- hw) sa) (* hh ca)))
        trx (+ cx (- (* hw ca) (* hh sa)))
        try_ (+ cy (+ (* hw sa) (* hh ca)))
        brx (+ cx (+ (* hw ca) (* hh sa)))
        bry (+ cy (- (* hw sa) (* hh ca)))
        blx (+ cx (- (* (- hw) ca) (* (- hh) sa)))
        bly (+ cy (+ (* (- hw) sa) (* (- hh) ca)))
        o (int offset)]
    (aset verts o (float tlx)) (aset verts (+ o 1) (float tly))
    (aset verts (+ o 2) (float u0)) (aset verts (+ o 3) (float v0))
    (aset verts (+ o 4) (float trx)) (aset verts (+ o 5) (float try_))
    (aset verts (+ o 6) (float u1)) (aset verts (+ o 7) (float v0))
    (aset verts (+ o 8) (float brx)) (aset verts (+ o 9) (float bry))
    (aset verts (+ o 10) (float u1)) (aset verts (+ o 11) (float v1))
    (aset verts (+ o 12) (float tlx)) (aset verts (+ o 13) (float tly))
    (aset verts (+ o 14) (float u0)) (aset verts (+ o 15) (float v0))
    (aset verts (+ o 16) (float brx)) (aset verts (+ o 17) (float bry))
    (aset verts (+ o 18) (float u1)) (aset verts (+ o 19) (float v1))
    (aset verts (+ o 20) (float blx)) (aset verts (+ o 21) (float bly))
    (aset verts (+ o 22) (float u0)) (aset verts (+ o 23) (float v1))
    (+ offset 24)))

;; ================================================================
;; Game state
;; ================================================================

(defn initial-state []
  {:ship {:x (/ W 2.0) :y (/ H 2.0) :angle (/ Math/PI -2.0) :vx 0.0 :vy 0.0}
   :asteroids (vec (repeatedly 5 #(shapes/random-shape 100.0)))
   :bullets []
   :score 0
   :lives 3
   :phase :playing
   :wave 1
   :cooldown 0.0
   :respawn-timer 0.0
   :invincible 0.0})

;; ================================================================
;; Game update
;; ================================================================

(defn- wrap-coord ^double [^double v ^double max-v]
  (let [v (rem v max-v)] (if (neg? v) (+ v max-v) v)))

(defn- move-ship [ship ^double dt]
  (-> ship
      (update :x #(wrap-coord (+ (double %) (* (double (:vx ship)) dt)) (double W)))
      (update :y #(wrap-coord (+ (double %) (* (double (:vy ship)) dt)) (double H)))
      (update :angle #(+ (double %) (* (double (get ship :spin 0.0)) dt)))))

(defn- move-bullet [b ^double dt]
  (-> b
      (update :x #(wrap-coord (+ (double %) (* (double (:vx b)) dt)) (double W)))
      (update :y #(wrap-coord (+ (double %) (* (double (:vy b)) dt)) (double H)))
      (update :ttl - dt)))

(defn- circle-collision? [a b ^double r]
  (let [dx (- (double (:x a)) (double (:x b)))
        dy (- (double (:y a)) (double (:y b)))]
    (< (+ (* dx dx) (* dy dy)) (* r r))))

(defn update-game [state inp ^double dt]
  (if (= (:phase state) :game-over)
    (if (input/key-pressed? inp :enter)
      (initial-state)
      state)
    (let [{:keys [ship asteroids bullets score lives phase wave cooldown respawn-timer invincible]} state
          invincible (max 0.0 (- (double (or invincible 0.0)) dt))

          ;; Ship controls
          turning-left (input/key-down? inp :left)
          turning-right (input/key-down? inp :right)
          thrusting (input/key-down? inp :up)
          shooting (input/key-down? inp :space)

          ship-angle (double (:angle ship))
          ship-angle (cond-> ship-angle
                       turning-left (- (* SHIP-TURN-SPEED dt))
                       turning-right (+ (* SHIP-TURN-SPEED dt)))

          svx (double (:vx ship))
          svy (double (:vy ship))
          svx (if thrusting (+ svx (* (Math/cos ship-angle) SHIP-ACCEL dt)) svx)
          svy (if thrusting (+ svy (* (Math/sin ship-angle) SHIP-ACCEL dt)) svy)
          svx (* svx SHIP-DRAG)
          svy (* svy SHIP-DRAG)

          ship' (if (= phase :playing)
                  (-> ship (assoc :angle ship-angle :vx svx :vy svy) (move-ship dt))
                  ship)

          cooldown' (max 0.0 (- cooldown dt))

          new-bullet (when (and shooting (= phase :playing) (<= cooldown' 0.0))
                       {:x (+ (double (:x ship')) (* 20.0 (Math/cos ship-angle)))
                        :y (+ (double (:y ship')) (* 20.0 (Math/sin ship-angle)))
                        :vx (+ svx (* BULLET-SPEED (Math/cos ship-angle)))
                        :vy (+ svy (* BULLET-SPEED (Math/sin ship-angle)))
                        :ttl BULLET-TTL})
          cooldown' (if new-bullet BULLET-COOLDOWN cooldown')

          bullets' (->> (if new-bullet (conj bullets new-bullet) bullets)
                        (map #(move-bullet % dt))
                        (filter #(pos? (:ttl %)))
                        vec)

          ;; Move asteroids via shapes/move-shape (wraps + rotates)
          asteroids' (mapv #(shapes/move-shape % dt (double W) (double H)) asteroids)

          ;; Collision: bullets vs asteroids — deftm split dispatch
          [asteroids'' bullets'' new-score]
          (loop [asts asteroids' blts bullets' sc (long score) new-asts [] dead-bullets #{}]
            (if (empty? asts)
              [(vec new-asts) (vec (remove #(contains? dead-bullets %) blts)) sc]
              (let [a (first asts)
                    hit (first (filter #(and (not (contains? dead-bullets %))
                                            (circle-collision? a % (shapes/shape-size a)))
                                       blts))]
                (if hit
                  ;; deftm dispatch: split shape into children (or nil if terminal)
                  (let [children (or (shapes/split a) [])
                        pts (shapes/score-value a)]
                    (recur (rest asts) blts (+ sc (long pts))
                           (into new-asts children) (conj dead-bullets hit)))
                  (recur (rest asts) blts sc (conj new-asts a) dead-bullets)))))

          ;; Ship collision (immune while invincible > 0)
          ship-hit (and (= phase :playing)
                        (<= invincible 0.0)
                        (some #(circle-collision? ship' % (+ 12.0 (shapes/shape-size %))) asteroids''))

          [phase' lives' ship'' respawn-timer' invincible']
          (cond
            ship-hit
            (if (<= lives 1)
              [:game-over 0 ship' 0.0 0.0]
              [:dead (dec lives) (assoc ship' :x (/ W 2.0) :y (/ H 2.0) :vx 0 :vy 0) 2.0 0.0])
            (= phase :dead)
            (if (<= respawn-timer dt)
              [:playing lives ship' 0.0 2.5]  ;; 2.5s invincibility on respawn
              [:dead lives ship' (- respawn-timer dt) 0.0])
            :else [phase lives ship' respawn-timer invincible])

          ;; Wave clear
          cur-wave (long (or wave 1))
          asteroids''' (if (empty? asteroids'')
                         (vec (repeatedly (+ 4 (inc cur-wave)) #(shapes/random-shape 100.0)))
                         asteroids'')
          wave' (if (empty? asteroids'') (inc cur-wave) cur-wave)]

      {:ship ship'' :asteroids asteroids''' :bullets bullets''
       :score new-score :lives lives' :phase phase' :wave wave'
       :cooldown cooldown' :respawn-timer respawn-timer'
       :invincible invincible'})))

;; ================================================================
;; Rendering — vertex builders
;; ================================================================

(defn build-shape-vertices
  "Build vertex data for a single shape. Returns new offset.
  Visual size is dispatched via shapes/render-scale per type."
  [^floats verts ^long offset shape]
  (let [sz (shapes/shape-size shape)
        hw (* sz (shapes/render-scale shape))]
    (emit-quad! verts offset
      (shapes/shape-x shape) (shapes/shape-y shape) hw hw
      (shapes/shape-angle shape) 0.0 0.0 1.0 1.0)))

(defn build-ship-vertices
  "Build vertex data for the ship. Returns [float-array vertex-count] or nil."
  [state]
  (let [{:keys [ship phase invincible]} state
        inv (double (or invincible 0.0))]
    (when (and (= phase :playing)
               ;; Blink during invincibility (visible 60% of the time)
               (or (<= inv 0.0)
                   (> (mod (* inv 8.0) 1.0) 0.4)))
      (let [verts (float-array 24)
            offset (emit-quad! verts 0
                     (double (:x ship)) (double (:y ship)) 28.0 28.0
                     (double (:angle ship)) 0.0 0.0 1.0 1.0)]
        [verts (quot (long offset) FLOATS-PER-VERTEX)]))))

(defn build-bullet-vertices ^floats [state]
  (let [bullets (:bullets state)
        verts (float-array (* (count bullets) VERTS-PER-QUAD FLOATS-PER-VERTEX))]
    (reduce (fn [^long off b]
              (emit-quad! verts off
                (double (:x b)) (double (:y b)) 2.5 2.5 0.0
                0.0 0.0 1.0 1.0))
            0 bullets)
    verts))

;; ================================================================
;; Texture registry — maps side-count to Vulkan texture + descriptor
;; ================================================================

(def ^:private tex-registry
  "Atom: {side-count {:texture tex :descriptor desc}}. Rebuilt on demand."
  (atom {}))

(defn- ensure-shape-texture!
  "Generate and register texture for N-sided polygon if not already present."
  [ctx tex-pool tex-layout n-sides]
  (when-not (get @tex-registry n-sides)
    (let [img (shapes/generate-polygon-texture n-sides 128)
          tex (texture/create-texture ctx img :filter-mode :linear :address-mode :clamp)
          _ (MemoryUtil/memFree ^java.nio.ByteBuffer (:pixels img))
          desc (texture/allocate-texture-descriptor-set
                 (:device ctx) tex-pool tex-layout tex)]
      (swap! tex-registry assoc n-sides {:texture tex :descriptor desc}))))

;; ================================================================
;; GPU resource init
;; ================================================================

(defn init-game [ctx]
  (let [device (:device ctx)
        allocator (:allocator ctx)
        sc (:swapchain-info ctx)

        ;; Pipeline layout (creates descriptor set layout internally)
        pipe-layout-info (texture/create-textured-pipeline-layout device
                           {:size 64 :stages VK13/VK_SHADER_STAGE_VERTEX_BIT})
        tex-layout (:desc-layout pipe-layout-info)

        ;; Descriptor pool — enough for all shape types + ship + font
        tex-pool (texture/create-texture-descriptor-pool device 16)

        ;; Generate shape textures for all known side counts
        _ (reset! tex-registry {})
        _ (doseq [n [3 4 5 6 8]]
            (ensure-shape-texture! ctx tex-pool tex-layout n))

        ;; Ship texture
        ship-img (shapes/generate-ship-texture 128)
        ship-tex (texture/create-texture ctx ship-img :filter-mode :linear :address-mode :clamp)
        _ (MemoryUtil/memFree ^java.nio.ByteBuffer (:pixels ship-img))
        ship-desc (texture/allocate-texture-descriptor-set device tex-pool tex-layout ship-tex)

        ;; Pipelines
        sprite-pipe (pipeline/create-graphics-pipeline device
                      {:vert-glsl vert-glsl :frag-glsl frag-glsl
                       :color-format (:format sc) :depth-format 0 :blend? true
                       :layout (:layout pipe-layout-info)
                       :vertex-bindings [{:binding 0 :stride 16}]
                       :vertex-attributes [{:location 0 :binding 0
                                            :format VK13/VK_FORMAT_R32G32_SFLOAT :offset 0}
                                           {:location 1 :binding 0
                                            :format VK13/VK_FORMAT_R32G32_SFLOAT :offset 8}]})
        bullet-pipe (pipeline/create-graphics-pipeline device
                      {:vert-glsl vert-glsl :frag-glsl bullet-frag-glsl
                       :color-format (:format sc) :depth-format 0 :blend? true
                       :push-constants {:size 64 :stages VK13/VK_SHADER_STAGE_VERTEX_BIT}
                       :vertex-bindings [{:binding 0 :stride 16}]
                       :vertex-attributes [{:location 0 :binding 0
                                            :format VK13/VK_FORMAT_R32G32_SFLOAT :offset 0}
                                           {:location 1 :binding 0
                                            :format VK13/VK_FORMAT_R32G32_SFLOAT :offset 8}]})

        vb (mem/create-buffer allocator (* VB-FLOATS 4) :vertex :cpu-to-gpu)
        ship-vb (mem/create-buffer allocator (* VERTS-PER-QUAD FLOATS-PER-VERTEX 4) :vertex :cpu-to-gpu)
        bullet-vb (mem/create-buffer allocator (* 128 VERTS-PER-QUAD FLOATS-PER-VERTEX 4) :vertex :cpu-to-gpu)
        proj (math/ortho 0.0 (double W) 0.0 (double H) -1.0 1.0)
        font (text/create-bitmap-font ctx tex-pool tex-layout :font-size 28.0)
        text-vb (mem/create-buffer allocator (* 256 6 4 4) :vertex :cpu-to-gpu)]

    {:ship-tex ship-tex :ship-desc ship-desc
     :tex-pool tex-pool :tex-layout tex-layout
     :pipe-layout-info pipe-layout-info
     :sprite-pipe sprite-pipe :bullet-pipe bullet-pipe
     :vb vb :ship-vb ship-vb :bullet-vb bullet-vb :proj proj
     :font font :text-vb text-vb
     :game (initial-state)
     :menu {:open? false :selection 0}}))

;; ================================================================
;; Audio
;; ================================================================

(defn init-audio []
  (try
    (require 'raster.vk.audio)
    (let [create-ctx (resolve 'raster.vk.audio/create-audio-context)
          load-snd (resolve 'raster.vk.audio/load-sound)
          ctx (create-ctx)
          base "examples/asteroids/assets/sounds/"]
      (load-snd ctx :laser (str base "laser1.ogg"))
      (load-snd ctx :explosion-big (str base "explosion5.ogg"))
      (load-snd ctx :explosion-small (str base "explosion2.ogg"))
      (load-snd ctx :warp (str base "warpout.ogg"))
      ctx)
    (catch Exception e
      (println "Audio init failed (continuing without sound):" (.getMessage e))
      nil)))

;; ================================================================
;; Frame callbacks
;; ================================================================

(defn update-fn [state inp dt]
  (let [game' (update-game (:game state) inp dt)
        old-score (get-in state [:game :score])
        new-score (:score game')
        scored (> new-score old-score)
        old-phase (get-in state [:game :phase])
        ship-died (and (= old-phase :playing) (= (:phase game') :dead))]
    (cond-> (assoc state :game game')
      ;; Audio triggers
      (and (:audio-ctx state) scored (not ship-died))
      (assoc :play-explosion true)
      (and (:audio-ctx state) ship-died)
      (assoc :play-death true))))

(defn- push-mvp! [^VkCommandBuffer cb ^long layout ^floats proj]
  (let [stack (MemoryStack/stackPush)]
    (let [buf (.mallocFloat stack 16)]
      (.put buf proj 0 16) (.flip buf)
      (VK13/vkCmdPushConstants cb layout VK13/VK_SHADER_STAGE_VERTEX_BIT 0 buf))
    (MemoryStack/stackPop)))

(defn- bind-vb! [^VkCommandBuffer cb ^long buffer-handle]
  (let [stack (MemoryStack/stackPush)]
    (VK13/vkCmdBindVertexBuffers cb 0 (.longs stack buffer-handle) (.longs stack 0))
    (MemoryStack/stackPop)))

(defn- bind-desc! [^VkCommandBuffer cb ^long layout ^long desc-set]
  (let [stack (MemoryStack/stackPush)]
    (VK13/vkCmdBindDescriptorSets cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                   layout 0 (.longs stack desc-set) nil)
    (MemoryStack/stackPop)))

(def menu-items ["RESUME" "RESTART" "QUIT"])

(defn render-fn [state ^VkCommandBuffer cb ctx {:keys [_extent]}]
  (let [{:keys [game sprite-pipe bullet-pipe vb bullet-vb proj
                ship-desc pipe-layout-info]} state
        layout (long (:layout pipe-layout-info))
        n-bullets (count (:bullets game))]

    ;; Bind sprite pipeline once
    (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                            (long (:pipeline sprite-pipe)))
    (push-mvp! cb layout proj)
    (bind-vb! cb (long (:buffer vb)))

    ;; === 1. Draw asteroids — all batches into one VB upload, draw with firstVertex ===
    (let [by-type (group-by #(shapes/sides %) (:asteroids game))
          verts (float-array VB-FLOATS)
          ;; Write ALL batches into verts at increasing offsets
          batch-info (loop [batches (seq by-type) float-off 0 infos []]
                       (if-not batches
                         infos
                         (let [[n-sides shapes] (first batches)
                               end-off (reduce (fn [^long off ast]
                                                 (build-shape-vertices verts off ast))
                                               float-off shapes)
                               n-verts (quot (- (long end-off) (long float-off)) FLOATS-PER-VERTEX)
                               first-vert (quot (long float-off) FLOATS-PER-VERTEX)]
                           (recur (next batches) end-off
                                  (conj infos {:n-sides n-sides :first-vert first-vert :n-verts n-verts})))))
          total-floats (reduce + 0 (map #(* (:n-verts %) (long FLOATS-PER-VERTEX)) batch-info))]
      (when (pos? total-floats)
        ;; Single upload of all asteroid vertices
        (mem/with-mapped-buffer [bb (:allocator ctx) vb]
          (let [^java.nio.ByteBuffer bb bb
                fb (.asFloatBuffer bb)]
            (.put fb verts 0 (int total-floats))
            (.flip fb)))
        (bind-vb! cb (long (:buffer vb)))
        ;; Draw each batch with its own descriptor + firstVertex offset
        (doseq [{:keys [n-sides first-vert n-verts]} batch-info]
          (when (pos? (int n-verts))
            (bind-desc! cb layout (long (:descriptor (get @tex-registry n-sides))))
            (VK13/vkCmdDraw cb (int n-verts) 1 (int first-vert) 0)))))

    ;; === 2. Draw ship (own vertex buffer to avoid data race with asteroid VB) ===
    (when-let [[ship-verts n-ship-verts] (build-ship-vertices game)]
      (let [svb (:ship-vb state)]
        (mem/with-mapped-buffer [bb (:allocator ctx) svb]
          (let [^java.nio.ByteBuffer bb bb
                fb (.asFloatBuffer bb)]
            (.put fb ^floats ship-verts 0 (* (int n-ship-verts) (int FLOATS-PER-VERTEX)))
            (.flip fb)))
        (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                (long (:pipeline sprite-pipe)))
        (bind-desc! cb layout (long ship-desc))
        (push-mvp! cb layout proj)
        (bind-vb! cb (long (:buffer svb)))
        (VK13/vkCmdDraw cb (int n-ship-verts) 1 0 0)))

    ;; === 3. Draw bullets ===
    (when (pos? (int n-bullets))
      (let [bv (build-bullet-vertices game)
            n-verts (* (int n-bullets) (int VERTS-PER-QUAD))]
        (mem/with-mapped-buffer [bb (:allocator ctx) bullet-vb]
          (let [^java.nio.ByteBuffer bb bb
                fb (.asFloatBuffer bb)]
            (.put fb bv 0 (* (int n-verts) (int FLOATS-PER-VERTEX)))
            (.flip fb)))
        (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                (long (:pipeline bullet-pipe)))
        (push-mvp! cb (long (:layout bullet-pipe)) proj)
        (bind-vb! cb (long (:buffer bullet-vb)))
        (VK13/vkCmdDraw cb (int n-verts) 1 0 0)))

    ;; === 4. HUD ===
    (let [{:keys [font text-vb]} state
          score-text (str "Score: " (:score game) "   Wave " (:wave game 1))
          lives-text (str "Lives: " (:lives game))
          [^floats sv sn] (text/build-text-vertices font score-text [10.0 (- H 36.0)] 1.0)
          [^floats lv ln] (text/build-text-vertices font lives-text [10.0 (- H 66.0)] 1.0)
          sn (int sn) ln (int ln)
          total-floats (+ (* sn 4) (* ln 4))
          merged (float-array total-floats)]
      (System/arraycopy sv 0 merged 0 (* sn 4))
      (System/arraycopy lv 0 merged (* sn 4) (* ln 4))
      (let [total-verts (+ sn ln)]
        (when (pos? total-verts)
          (mem/with-mapped-buffer [bb (:allocator ctx) text-vb]
            (let [^java.nio.ByteBuffer bb bb
                  fb (.asFloatBuffer bb)]
              (.put fb merged 0 total-floats)
              (.flip fb)))
          (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                  (long (:pipeline sprite-pipe)))
          (bind-desc! cb layout (long (:desc-set font)))
          (push-mvp! cb layout proj)
          (bind-vb! cb (long (:buffer text-vb)))
          (VK13/vkCmdDraw cb total-verts 1 0 0))))

    ;; === 5. Game over / menu overlays ===
    (when (= (:phase game) :game-over)
      (let [{:keys [font text-vb]} state
            text "Game Over  --  press Enter"
            [^floats gv gn] (text/build-text-vertices font text [(- (/ W 2.0) 180.0) (/ H 2.0)] 1.0)
            gn (int gn)]
        (when (pos? gn)
          (mem/with-mapped-buffer [bb (:allocator ctx) text-vb]
            (let [^java.nio.ByteBuffer bb bb
                  fb (.asFloatBuffer bb)]
              (.put fb gv 0 (* gn 4))
              (.flip fb)))
          (VK13/vkCmdBindPipeline cb VK13/VK_PIPELINE_BIND_POINT_GRAPHICS
                                  (long (:pipeline sprite-pipe)))
          (bind-desc! cb layout (long (:desc-set font)))
          (push-mvp! cb layout proj)
          (bind-vb! cb (long (:buffer text-vb)))
          (VK13/vkCmdDraw cb gn 1 0 0))))))

;; ================================================================
;; Cleanup
;; ================================================================

(defn destroy-game-resources! [ctx state]
  (let [device (:device ctx)
        allocator (:allocator ctx)]
    (mem/destroy-buffer! allocator (:vb state))
    (mem/destroy-buffer! allocator (:ship-vb state))
    (mem/destroy-buffer! allocator (:bullet-vb state))
    (pipeline/destroy-graphics-pipeline! device (:bullet-pipe state))
    (let [pipe (:sprite-pipe state)]
      (VK13/vkDestroyPipeline device (:pipeline pipe) nil)
      (when-let [s (:vert-spirv pipe)] (MemoryUtil/memFree ^java.nio.ByteBuffer s))
      (when-let [s (:frag-spirv pipe)] (MemoryUtil/memFree ^java.nio.ByteBuffer s))
      (raster.vk.shader/destroy-shader-module! device (:vert-module pipe))
      (raster.vk.shader/destroy-shader-module! device (:frag-module pipe)))
    ;; Textures
    (doseq [[_ {:keys [texture]}] @tex-registry]
      (texture/destroy-texture! device allocator texture))
    (texture/destroy-texture! device allocator (:ship-tex state))
    (VK13/vkDestroyDescriptorPool device (:tex-pool state) nil)
    (VK13/vkDestroyDescriptorSetLayout device (long (:tex-layout state)) nil)
    (VK13/vkDestroyPipelineLayout device (:layout (:pipe-layout-info state)) nil)
    (when-let [font (:font state)]
      (text/destroy-bitmap-font! ctx font))
    (when-let [tvb (:text-vb state)]
      (mem/destroy-buffer! allocator tvb))))

;; ================================================================
;; Menu
;; ================================================================

(defn- update-menu [state inp]
  (let [menu (:menu state)
        sel (long (:selection menu 0))
        n (count menu-items)]
    (cond
      (input/key-pressed? inp :escape)
      (update state :menu assoc :open? (not (:open? menu)) :selection 0)

      (not (:open? menu)) state

      (input/key-pressed? inp :up)
      (update state :menu assoc :selection (mod (dec sel) n))

      (input/key-pressed? inp :down)
      (update state :menu assoc :selection (mod (inc sel) n))

      (input/key-pressed? inp :enter)
      (case (int sel)
        0 (update state :menu assoc :open? false)
        1 (-> state (assoc :game (initial-state)) (update :menu assoc :open? false))
        2 (do (System/exit 0) state)
        state)

      :else state)))

(defn full-update-fn [state inp dt]
  (let [state (update-menu state inp)]
    (if (get-in state [:menu :open?])
      state
      (let [state' (update-fn state inp dt)]
        ;; Audio
        (when-let [ac (:audio-ctx state')]
          (when (:play-explosion state')
            (try ((resolve 'raster.vk.audio/play-sound!) ac :explosion-small)
                 (catch Exception _)))
          (when (:play-death state')
            (try ((resolve 'raster.vk.audio/play-sound!) ac :explosion-big)
                 (catch Exception _))))
        (dissoc state' :play-explosion :play-death)))))

;; ================================================================
;; Entry point
;; ================================================================

(defn start! []
  (println "=== GEOMETRIC ASTEROIDS ===")
  (println "Controls: Arrow keys to move, Space to shoot, Enter to restart")
  (println "Add new polygon types from the REPL with deftm dispatch!")
  (gpu/with-gpu-context [ctx {:width W :height H
                              :title "Raster — Geometric Asteroids"
                              :depth? false :debug? true}]
    (let [inp-sys (input/create-input-system (:window ctx))
          sync (frame/create-frame-sync ctx)
          resources (init-game ctx)
          audio-ctx (init-audio)]
      (try
        (frame/run-game-loop! ctx
          {:frame-sync sync
           :init-state (assoc resources :audio-ctx audio-ctx)
           :input-system inp-sys
           :update-fn #'full-update-fn
           :render-fn #'render-fn
           :clear-color [0.0 0.0 0.0 1.0]  ;; pure black — paper figure background
           :title "Geometric Asteroids"})
        (finally
          (VK13/vkDeviceWaitIdle (:device ctx))
          (frame/destroy-frame-sync! ctx sync)
          (input/destroy-input-system! inp-sys)
          (destroy-game-resources! ctx resources)
          (when audio-ctx ((resolve 'raster.vk.audio/destroy-audio-context!) audio-ctx))
          (println "Done."))))))
