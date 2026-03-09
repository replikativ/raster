(ns doom.sprite
  "Sprite rendering: billboarded quads for things (items, decorations, monsters).
  Each visible thing becomes a camera-facing quad with the appropriate sprite frame."
  (:require
   [doom.wad :as wad]
   [doom.bsp :as bsp]
   [doom.level :as level]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Sprite atlas building
;; ================================================================

(defn build-sprite-images
  "Build sprite images for all things present in the map.
  Returns vector of {:name :pixels :width :height :left-offset :top-offset}."
  [wad-data map-data ^bytes palette]
  (let [sprite-map (wad/parse-sprites wad-data)
        things (:things map-data)
        ;; Collect unique sprite prefixes needed
        needed-prefixes (->> things
                             (keep (fn [t]
                                     (when-let [info (get wad/thing-types (:type t))]
                                       (:sprite info))))
                             (into #{})
                             ;; Add projectile and weapon sprites
                             (into #{"BAL1" "BAL2" "PISG" "PISF" "SHTG" "SHTF"
                                     "PUNG" "CHGG" "CHGF" "MISG" "MISF" "SAWG"}))
        images (atom [])
        seen (atom #{})]
    ;; For each needed prefix, decode all frames/rotations
    (doseq [prefix needed-prefixes]
      (when-let [frames (get sprite-map prefix)]
        (doseq [[_frame rotations] frames]
          (doseq [[_rotation sprite-info] rotations]
            (let [lump-name (:name sprite-info)]
              (when-not (contains? @seen lump-name)
                (swap! seen conj lump-name)
                (when-let [decoded (wad/decode-sprite wad-data (:lump sprite-info) palette)]
                  (swap! images conj
                         (assoc decoded :name lump-name)))))))))
    @images))

;; ================================================================
;; Sprite frame selection
;; ================================================================

(defn select-sprite-frame
  "Select the correct sprite frame name for a thing given camera position.
  thing-angle: angle the thing faces (0-360 degrees).
  camera-pos: [x y] in Doom coords. thing-pos: [x y] in Doom coords.
  frame-char: optional frame char (default \\A). Use for animated monsters.
  Returns sprite lump name string, or nil."
  [sprite-map prefix thing-angle camera-pos thing-pos & {:keys [frame-char]}]
  (when-let [frames (get sprite-map prefix)]
    (let [cam-x (double (nth camera-pos 0))
          cam-y (double (nth camera-pos 1))
          tx (double (nth thing-pos 0))
          ty (double (nth thing-pos 1))
          thing-angle (double thing-angle)
          frame (or frame-char \A)
          ;; Fall back to frame A if requested frame doesn't exist
          rotations (or (get frames frame) (get frames \A))]
      (when rotations
        (if (contains? rotations 0)
          ;; Rotation 0 = single angle for all views
          (:name (get rotations 0))
          ;; Pick rotation based on angle from thing to camera
          (let [;; Angle from thing to camera
                dx (- cam-x tx)
                dy (- cam-y ty)
                angle-to-cam (Math/toDegrees (Math/atan2 dy dx))
                ;; Relative angle (thing's facing vs camera direction)
                rel-angle (mod (+ (- angle-to-cam thing-angle) 360.0 22.5) 360.0)
                ;; Map to rotation 1-8 (Doom convention: 1=front, 5=back, clockwise)
                rotation (inc (int (/ rel-angle 45.0)))
                rotation (min 8 (max 1 rotation))]
            (when-let [info (get rotations rotation)]
              (:name info))))))))

;; ================================================================
;; Billboard quad generation
;; ================================================================

(def ^:const SPRITE-FLOATS-PER-VERTEX 9)

(defn- monster-frame-char
  "Select sprite frame char based on monster AI state and frame counter."
  [state frame]
  (let [frame (int (or frame 0))]
    (case state
      :idle    (char (+ (int \A) (mod frame 2)))     ;; A-B idle
      :chase   (char (+ (int \A) (mod frame 4)))     ;; A-D walk cycle
      :attack  (char (+ (int \E) (min frame 2)))     ;; E-F-G attack
      :pain    \G                                     ;; G pain
      :death   (char (+ (int \H) (min frame 4)))     ;; H-I-J-K-L death
      :dead    \L                                     ;; L final corpse
      \A)))

(defn build-sprite-quads
  "Build billboard quads for all visible things + extra entities (projectiles).
  Returns {:vertices float[] :indices int[] :count int} or nil if no sprites.
  things: active thing list (or nil to use map-data things).
  monsters: [{:x :y :angle :state :frame :info ...}] for monster sprites.
  extra-entities: [{:x :y :sprite :angle}] for projectiles etc."
  [map-data sprite-map sprite-tex-map cam-pos cam-yaw
   & {:keys [things extra-entities monsters]}]
  (let [things (or things (:things map-data))
        sectors (:sectors map-data)
        ;; Camera position in Doom coords
        [^double cam-vx _ ^double cam-vz] cam-pos
        cam-dx (/ cam-vx level/SCALE)
        cam-dy (/ (- cam-vz) level/SCALE)
        ;; Right vector from camera yaw for billboarding
        yaw (double cam-yaw)
        right-x (Math/cos yaw)
        right-z (Math/sin yaw)
        ;; Pre-allocate (generous: 4 verts × 9 floats per thing + extras + monsters)
        max-sprites (+ (count things) (count extra-entities) (count monsters))
        verts (float-array (* max-sprites 4 SPRITE-FLOATS-PER-VERTEX))
        indices (int-array (* max-sprites 6))
        vi (volatile! 0)
        ii (volatile! 0)
        vc (volatile! 0)
        sprite-count (volatile! 0)]
    (doseq [thing things]
      (let [thing-type (:type thing)
            type-info (get wad/thing-types thing-type)]
        (when (and type-info (:sprite type-info)
                   (not= (:category type-info) :player)
                   (not= (:category type-info) :deathmatch)
                   (not= (:category type-info) :monster))
          (let [tx (double (:x thing))
                ty (double (:y thing))
                ddx (- tx cam-dx) ddy (- ty cam-dy)
                thing-dist (Math/sqrt (+ (* ddx ddx) (* ddy ddy)))]
           (when (< thing-dist 1500.0)
            (let [thing-angle (double (:angle thing))
                prefix (:sprite type-info)
                ;; Select correct frame/rotation
                sprite-name (select-sprite-frame
                              sprite-map prefix thing-angle
                              [cam-dx cam-dy] [tx ty])
                ;; Look up in texture array map
                tex-info (when sprite-name (get sprite-tex-map sprite-name))]
            (when tex-info
              ;; Get floor height for Z positioning
              (let [sector-idx (bsp/point-sector map-data tx ty)
                    sector (when sector-idx (nth sectors sector-idx))
                    floor-h (if sector (double (:floor-height sector)) 0.0)
                    light (if sector (/ (double (:light-level sector)) 255.0) 0.8)
                    ;; Sprite dimensions in Doom units
                    sw (double (:width tex-info))
                    sh (double (:height tex-info))
                    ;; Center offset (left-offset positions the sprite center)
                    ;; bottom of sprite sits on the floor
                    half-w (* sw 0.5)
                    ;; VK coordinates
                    cx (level/doom->vk-x tx)
                    cz (level/doom->vk-z ty)
                    bottom-y (level/doom->vk-y floor-h)
                    top-y (level/doom->vk-y (+ floor-h sh))
                    ;; Billboard: expand left/right using camera right vector
                    hw (* half-w level/SCALE)
                    lx (- cx (* right-x hw))
                    lz (- cz (* right-z hw))
                    rx (+ cx (* right-x hw))
                    rz (+ cz (* right-z hw))
                    ;; Texture array info
                    tex-layer (float (double (:layer tex-info)))
                    su (float (double (:scale-u tex-info)))
                    sv (float (double (:scale-v tex-info)))
                    base-v (long @vc)
                    vo (long @vi)
                    io (long @ii)]
                ;; Emit 4 vertices: BL BR TR TL
                ;; BL
                (aset verts vo (float lx))
                (aset verts (+ vo 1) (float bottom-y))
                (aset verts (+ vo 2) (float lz))
                (aset verts (+ vo 3) (float 0.0))
                (aset verts (+ vo 4) (float 1.0))
                (aset verts (+ vo 5) (float light))
                (aset verts (+ vo 6) tex-layer)
                (aset verts (+ vo 7) su)
                (aset verts (+ vo 8) sv)
                ;; BR
                (aset verts (+ vo 9) (float rx))
                (aset verts (+ vo 10) (float bottom-y))
                (aset verts (+ vo 11) (float rz))
                (aset verts (+ vo 12) (float 1.0))
                (aset verts (+ vo 13) (float 1.0))
                (aset verts (+ vo 14) (float light))
                (aset verts (+ vo 15) tex-layer)
                (aset verts (+ vo 16) su)
                (aset verts (+ vo 17) sv)
                ;; TR
                (aset verts (+ vo 18) (float rx))
                (aset verts (+ vo 19) (float top-y))
                (aset verts (+ vo 20) (float rz))
                (aset verts (+ vo 21) (float 1.0))
                (aset verts (+ vo 22) (float 0.0))
                (aset verts (+ vo 23) (float light))
                (aset verts (+ vo 24) tex-layer)
                (aset verts (+ vo 25) su)
                (aset verts (+ vo 26) sv)
                ;; TL
                (aset verts (+ vo 27) (float lx))
                (aset verts (+ vo 28) (float top-y))
                (aset verts (+ vo 29) (float lz))
                (aset verts (+ vo 30) (float 0.0))
                (aset verts (+ vo 31) (float 0.0))
                (aset verts (+ vo 32) (float light))
                (aset verts (+ vo 33) tex-layer)
                (aset verts (+ vo 34) su)
                (aset verts (+ vo 35) sv)
                ;; 6 indices (2 triangles)
                (aset indices io (int base-v))
                (aset indices (+ io 1) (int (+ base-v 1)))
                (aset indices (+ io 2) (int (+ base-v 2)))
                (aset indices (+ io 3) (int base-v))
                (aset indices (+ io 4) (int (+ base-v 2)))
                (aset indices (+ io 5) (int (+ base-v 3)))
                (vreset! vi (+ vo 36))
                (vreset! ii (+ io 6))
                (vreset! vc (+ base-v 4))
                (vswap! sprite-count inc)))))))))
    ;; Monsters (animated sprites based on AI state, including dead as corpses)
    (doseq [m monsters]
      (let [tx (double (:x m))
            ty (double (:y m))
            ddx (- tx cam-dx) ddy (- ty cam-dy)
            mdist (Math/sqrt (+ (* ddx ddx) (* ddy ddy)))]
        (when (< mdist 1500.0)
          (let [thing-angle (double (:angle m))
                prefix (get-in m [:info :sprite])
                frame-char (monster-frame-char (:state m) (:frame m))
                sprite-name (when prefix
                              (select-sprite-frame
                                sprite-map prefix thing-angle
                                [cam-dx cam-dy] [tx ty]
                                :frame-char frame-char))
                tex-info (when sprite-name (get sprite-tex-map sprite-name))]
            (when tex-info
            (let [sector-idx (bsp/point-sector map-data tx ty)
                  sector (when sector-idx (nth sectors sector-idx))
                  floor-h (if sector (double (:floor-height sector)) 0.0)
                  light (if sector (/ (double (:light-level sector)) 255.0) 0.8)
                  sw (double (:width tex-info))
                  sh (double (:height tex-info))
                  half-w (* sw 0.5)
                  cx (level/doom->vk-x tx)
                  cz (level/doom->vk-z ty)
                  bottom-y (level/doom->vk-y floor-h)
                  top-y (level/doom->vk-y (+ floor-h sh))
                  hw (* half-w level/SCALE)
                  lx (- cx (* right-x hw))
                  lz (- cz (* right-z hw))
                  rx (+ cx (* right-x hw))
                  rz (+ cz (* right-z hw))
                  tex-layer (float (double (:layer tex-info)))
                  su (float (double (:scale-u tex-info)))
                  sv (float (double (:scale-v tex-info)))
                  base-v (long @vc)
                  vo (long @vi)
                  io (long @ii)]
              ;; BL BR TR TL
              (aset verts vo (float lx))
              (aset verts (+ vo 1) (float bottom-y))
              (aset verts (+ vo 2) (float lz))
              (aset verts (+ vo 3) (float 0.0))
              (aset verts (+ vo 4) (float 1.0))
              (aset verts (+ vo 5) (float light))
              (aset verts (+ vo 6) tex-layer)
              (aset verts (+ vo 7) su)
              (aset verts (+ vo 8) sv)
              (aset verts (+ vo 9) (float rx))
              (aset verts (+ vo 10) (float bottom-y))
              (aset verts (+ vo 11) (float rz))
              (aset verts (+ vo 12) (float 1.0))
              (aset verts (+ vo 13) (float 1.0))
              (aset verts (+ vo 14) (float light))
              (aset verts (+ vo 15) tex-layer)
              (aset verts (+ vo 16) su)
              (aset verts (+ vo 17) sv)
              (aset verts (+ vo 18) (float rx))
              (aset verts (+ vo 19) (float top-y))
              (aset verts (+ vo 20) (float rz))
              (aset verts (+ vo 21) (float 1.0))
              (aset verts (+ vo 22) (float 0.0))
              (aset verts (+ vo 23) (float light))
              (aset verts (+ vo 24) tex-layer)
              (aset verts (+ vo 25) su)
              (aset verts (+ vo 26) sv)
              (aset verts (+ vo 27) (float lx))
              (aset verts (+ vo 28) (float top-y))
              (aset verts (+ vo 29) (float lz))
              (aset verts (+ vo 30) (float 0.0))
              (aset verts (+ vo 31) (float 0.0))
              (aset verts (+ vo 32) (float light))
              (aset verts (+ vo 33) tex-layer)
              (aset verts (+ vo 34) su)
              (aset verts (+ vo 35) sv)
              (aset indices io (int base-v))
              (aset indices (+ io 1) (int (+ base-v 1)))
              (aset indices (+ io 2) (int (+ base-v 2)))
              (aset indices (+ io 3) (int base-v))
              (aset indices (+ io 4) (int (+ base-v 2)))
              (aset indices (+ io 5) (int (+ base-v 3)))
              (vreset! vi (+ vo 36))
              (vreset! ii (+ io 6))
              (vreset! vc (+ base-v 4))
              (vswap! sprite-count inc)))))))
    ;; Extra entities (projectiles, etc.)
    (doseq [ent extra-entities]
      (let [tx (double (:x ent))
            ty (double (:y ent))
            prefix (:sprite ent)
            sprite-name (select-sprite-frame
                          sprite-map prefix (double (or (:angle ent) 0))
                          [cam-dx cam-dy] [tx ty])
            tex-info (when sprite-name (get sprite-tex-map sprite-name))]
        (when tex-info
          (let [sector-idx (bsp/point-sector map-data tx ty)
                sector (when sector-idx (nth sectors sector-idx))
                floor-h (if sector (double (:floor-height sector)) 0.0)
                light (if sector (/ (double (:light-level sector)) 255.0) 0.8)
                sz (double (or (:z ent) (+ floor-h 32.0))) ;; default: 32 above floor
                sw (double (:width tex-info))
                sh (double (:height tex-info))
                half-w (* sw 0.5)
                cx (level/doom->vk-x tx)
                cz (level/doom->vk-z ty)
                bottom-y (level/doom->vk-y sz)
                top-y (level/doom->vk-y (+ sz sh))
                hw (* half-w level/SCALE)
                lx (- cx (* right-x hw))
                lz (- cz (* right-z hw))
                rx (+ cx (* right-x hw))
                rz (+ cz (* right-z hw))
                tex-layer (float (double (:layer tex-info)))
                su (float (double (:scale-u tex-info)))
                sv (float (double (:scale-v tex-info)))
                base-v (long @vc)
                vo (long @vi)
                io (long @ii)]
            ;; BL BR TR TL (same vertex layout as things)
            (aset verts vo (float lx))
            (aset verts (+ vo 1) (float bottom-y))
            (aset verts (+ vo 2) (float lz))
            (aset verts (+ vo 3) (float 0.0))
            (aset verts (+ vo 4) (float 1.0))
            (aset verts (+ vo 5) (float light))
            (aset verts (+ vo 6) tex-layer)
            (aset verts (+ vo 7) su)
            (aset verts (+ vo 8) sv)
            (aset verts (+ vo 9) (float rx))
            (aset verts (+ vo 10) (float bottom-y))
            (aset verts (+ vo 11) (float rz))
            (aset verts (+ vo 12) (float 1.0))
            (aset verts (+ vo 13) (float 1.0))
            (aset verts (+ vo 14) (float light))
            (aset verts (+ vo 15) tex-layer)
            (aset verts (+ vo 16) su)
            (aset verts (+ vo 17) sv)
            (aset verts (+ vo 18) (float rx))
            (aset verts (+ vo 19) (float top-y))
            (aset verts (+ vo 20) (float rz))
            (aset verts (+ vo 21) (float 1.0))
            (aset verts (+ vo 22) (float 0.0))
            (aset verts (+ vo 23) (float light))
            (aset verts (+ vo 24) tex-layer)
            (aset verts (+ vo 25) su)
            (aset verts (+ vo 26) sv)
            (aset verts (+ vo 27) (float lx))
            (aset verts (+ vo 28) (float top-y))
            (aset verts (+ vo 29) (float lz))
            (aset verts (+ vo 30) (float 0.0))
            (aset verts (+ vo 31) (float 0.0))
            (aset verts (+ vo 32) (float light))
            (aset verts (+ vo 33) tex-layer)
            (aset verts (+ vo 34) su)
            (aset verts (+ vo 35) sv)
            (aset indices io (int base-v))
            (aset indices (+ io 1) (int (+ base-v 1)))
            (aset indices (+ io 2) (int (+ base-v 2)))
            (aset indices (+ io 3) (int base-v))
            (aset indices (+ io 4) (int (+ base-v 2)))
            (aset indices (+ io 5) (int (+ base-v 3)))
            (vreset! vi (+ vo 36))
            (vreset! ii (+ io 6))
            (vreset! vc (+ base-v 4))
            (vswap! sprite-count inc)))))
    (let [n-vfloats (long @vi)
          n-indices (long @ii)]
      (when (pos? n-vfloats)
        {:vertices (java.util.Arrays/copyOf verts (int n-vfloats))
         :indices (java.util.Arrays/copyOf indices (int n-indices))
         :count @sprite-count}))))
