(ns valley.hud
  "HUD rendering: health hearts + hotbar overlay.
  Uses the same vertex format as terrain (9 floats) in clip space, no depth.
  Health displayed as hearts. Hunger is hidden until low (< 30%), then
  a flashing warning bar appears."
  (:require [valley.items :as items]
            [raster.vk.mesh :as mesh])
  (:import [org.lwjgl.vulkan VK13 VkCommandBuffer]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Constants
;; ================================================================

;; Heart texture layers (appended after block textures)
(def ^:const HEART-FULL-LAYER 62)
(def ^:const HEART-HALF-LAYER 63)
(def ^:const HEART-EMPTY-LAYER 64)

;; HUD layout (clip space: -1 to 1, Vulkan Y-down: -1=top, +1=bottom)
(def ^:const HEART-SIZE 0.04)
(def ^:const HEART-SPACING 0.045)
(def ^:const HEARTS-X -0.225)       ;; centered above hotbar
(def ^:const HEARTS-Y 0.82)

;; Hunger warning bar (only visible when hunger < 30%)
(def ^:const HUNGER-BAR-WIDTH 0.25)
(def ^:const HUNGER-BAR-HEIGHT 0.016)
(def ^:const HUNGER-BAR-X -0.125)       ;; centered
(def ^:const HUNGER-BAR-Y 0.80)         ;; just above hearts
(def ^:const HUNGER-WARN-THRESHOLD 6)   ;; out of 20 — show warning below this

;; Hotbar layout
(def ^:const HOTBAR-SLOT-SIZE 0.08)
(def ^:const HOTBAR-SPACING 0.088)
(def ^:const HOTBAR-X -0.396)
(def ^:const HOTBAR-Y 0.88)
(def ^:const HOTBAR-ITEM-SIZE 0.064)
(def ^:const HOTBAR-ITEM-OFFSET 0.008)

;; Slot background texture layer
(def ^:const SLOT-BG-LAYER 30)

;; ================================================================
;; Vertex generation
;; ================================================================

(defn- add-quad!
  "Add a textured quad. Vertex format: pos(3) uv(2) R(1) G(1) texLayer(1) B(1).
  texLayer < 0 signals flat color mode (R,G,B used directly)."
  [^floats verts ^ints indices vi ii x y w h tex-layer r g b]
  (let [vi (long vi) ii (long ii)
        x (double x) y (double y) w (double w) h (double h)
        tex-layer (double tex-layer)
        r (double r) g (double g) b (double b)
        x1 (+ x w)
        y1 (+ y h)
        base (* vi 9)]
    ;; Vulkan Y-down: y < y1, so y is top, y1 is bottom
    ;; UV: v=0 at top (y), v=1 at bottom (y1)
    (doseq [[off vx vy u v] [[0 x y 0.0 0.0]
                              [9 x y1 0.0 1.0]
                              [18 x1 y1 1.0 1.0]
                              [27 x1 y 1.0 0.0]]]
      (let [b-off (+ base (long off))]
        (aset verts (+ b-off 0) (float vx))
        (aset verts (+ b-off 1) (float vy))
        (aset verts (+ b-off 2) (float 0.0))
        (aset verts (+ b-off 3) (float u))
        (aset verts (+ b-off 4) (float v))
        (aset verts (+ b-off 5) (float r))
        (aset verts (+ b-off 6) (float g))
        (aset verts (+ b-off 7) (float tex-layer))
        (aset verts (+ b-off 8) (float b))))
    (aset indices (+ ii 0) (int vi))
    (aset indices (+ ii 1) (int (+ vi 1)))
    (aset indices (+ ii 2) (int (+ vi 2)))
    (aset indices (+ ii 3) (int vi))
    (aset indices (+ ii 4) (int (+ vi 2)))
    (aset indices (+ ii 5) (int (+ vi 3)))))

(defn build-hud-mesh
  "Build mesh data for HUD: hearts + hunger warning + hotbar.
  Returns {:vertices float[] :indices int[]}."
  [health max-health hunger inventory frame-count]
  (let [health (long health) max-health (long max-health) hunger (long hunger)
        hotbar (:hotbar inventory)
        selected (long (:selected inventory 0))
        num-hearts (long (/ max-health 2))
        ;; Max quads: 10 hearts + 2 hunger bars + 9 slots + 9 items + 1 selection
        max-quads 40
        max-verts (* max-quads 4)
        max-indices (* max-quads 6)
        verts (float-array (* max-verts 9))
        indices (int-array max-indices)
        qi (atom 0)
        emit (fn [x y w h tex-layer r g b]
               (let [i (long @qi)
                     vi (* i 4)
                     ii (* i 6)]
                 (add-quad! verts indices vi ii x y w h tex-layer r g b)
                 (reset! qi (inc i))))]

    ;; Hearts
    (dotimes [i num-hearts]
      (let [x (+ HEARTS-X (* (double i) HEART-SPACING))
            heart-hp (- health (* i 2))
            tex-layer (cond
                        (>= heart-hp 2) HEART-FULL-LAYER
                        (= heart-hp 1) HEART-HALF-LAYER
                        :else HEART-EMPTY-LAYER)]
        (emit x HEARTS-Y HEART-SIZE HEART-SIZE (double tex-layer) 1.0 1.0 1.0)))

    ;; Hunger warning — only shows when hunger drops below threshold
    (when (< hunger HUNGER-WARN-THRESHOLD)
      (let [hunger-frac (/ (double hunger) 20.0)
            fc (long frame-count)
            ;; Flash: visible 70% of time, hidden 30%
            visible? (< (rem fc 30) 21)]
        (when visible?
          ;; Background (dark)
          (emit HUNGER-BAR-X HUNGER-BAR-Y HUNGER-BAR-WIDTH HUNGER-BAR-HEIGHT
                -1.0  0.15 0.08 0.02)
          ;; Fill (amber, shrinks as hunger drops)
          (let [fill-w (* HUNGER-BAR-WIDTH hunger-frac)]
            (when (pos? fill-w)
              (emit HUNGER-BAR-X HUNGER-BAR-Y fill-w HUNGER-BAR-HEIGHT
                    -1.0  0.9 0.55 0.1))))))

    ;; Hotbar slot backgrounds
    (dotimes [i 9]
      (let [x (+ HOTBAR-X (* (double i) HOTBAR-SPACING))]
        (emit x HOTBAR-Y HOTBAR-SLOT-SIZE HOTBAR-SLOT-SIZE
              (double SLOT-BG-LAYER) 1.0 1.0 1.0)))

    ;; Hotbar item icons
    (dotimes [i 9]
      (when-let [slot (nth hotbar i)]
        (let [item-id (:item slot)
              tex-layer (get-in items/item-props [item-id :tex-layer])]
          (when tex-layer
            (let [x (+ HOTBAR-X (* (double i) HOTBAR-SPACING) HOTBAR-ITEM-OFFSET)]
              (emit x (+ HOTBAR-Y HOTBAR-ITEM-OFFSET)
                    HOTBAR-ITEM-SIZE HOTBAR-ITEM-SIZE
                    (double tex-layer) 1.0 1.0 1.0))))))

    ;; Selection highlight
    (let [sel-x (+ HOTBAR-X (* (double selected) HOTBAR-SPACING) -0.003)]
      (emit sel-x (- HOTBAR-Y 0.003)
            (+ HOTBAR-SLOT-SIZE 0.006) (+ HOTBAR-SLOT-SIZE 0.006)
            11.0 1.0 1.0 1.0))

    (let [total-quads (long @qi)
          total-verts (* total-quads 4)
          total-indices (* total-quads 6)]
      {:vertices (java.util.Arrays/copyOf verts (int (* total-verts 9)))
       :indices (java.util.Arrays/copyOf indices (int total-indices))})))

;; ================================================================
;; HUD shaders
;; ================================================================

(def hud-vert-glsl
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
layout(location = 2) in float inR;
layout(location = 3) in float inG;
layout(location = 4) in float inTexLayer;
layout(location = 5) in float inB;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float fragTexLayer;
layout(location = 2) out vec3 fragColor;

void main() {
    gl_Position = vec4(inPos.xy, 0.0, 1.0);
    fragUV = inUV;
    fragTexLayer = inTexLayer;
    fragColor = vec3(inR, inG, inB);
}
")

(def hud-frag-glsl
  "#version 450

layout(set = 0, binding = 0) uniform sampler2DArray texArray;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float fragTexLayer;
layout(location = 2) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    if (fragTexLayer < 0.0) {
        // Flat colored bar (hunger warning)
        outColor = vec4(fragColor, 0.85);
    } else {
        vec4 texColor = texture(texArray, vec3(fragUV, fragTexLayer));
        if (texColor.a < 0.1) discard;
        outColor = texColor;
    }
}
")

;; ================================================================
;; Render HUD
;; ================================================================

(defn render-hud!
  "Render the HUD into the command buffer."
  [^VkCommandBuffer cb state]
  (when-let [hud-mesh (:hud-mesh state)]
    (mesh/cmd-bind-mesh! cb hud-mesh)
    (mesh/cmd-draw-mesh! cb hud-mesh)))

(defn update-hud-mesh
  "Rebuild HUD mesh if health, hunger warning state, or inventory changed."
  [state ctx]
  (let [player (:player state)
        health (long (or (:health player) 20))
        max-health (long (or (:max-health player) 20))
        hunger-state (:hunger-state state)
        hunger (long (or (:hunger hunger-state) 20))
        inventory (:inventory state)
        fc (long (:frame-count state 0))
        ;; Include hunger-warning visibility in the key so it updates when flashing
        hunger-warn? (< hunger HUNGER-WARN-THRESHOLD)
        flash-phase (when hunger-warn? (< (rem fc 30) 21))
        hud-key [health hunger-warn? flash-phase (:selected inventory)
                 (mapv #(when % [(:item %) (:count %)]) (:hotbar inventory))]
        prev-key (:prev-hud-key state)]
    (if (= hud-key prev-key)
      state
      (let [_ (when-let [m (:hud-mesh state)]
                (VK13/vkDeviceWaitIdle (:device ctx))
                (mesh/destroy-mesh! ctx m))
            mesh-data (build-hud-mesh health max-health hunger
                        (or inventory {:hotbar (vec (repeat 9 nil)) :selected 0})
                        fc)
            gpu-mesh (mesh/create-static-mesh ctx
                       (:vertices mesh-data) (:indices mesh-data) (* 9 4))]
        (assoc state
          :hud-mesh gpu-mesh
          :prev-hud-key hud-key)))))
