(ns doom.hud
  "HUD / Status bar rendering as screen-space textured overlay.
  Draws health, armor, ammo, keys as overlay quads."
  (:require
   [doom.wad :as wad]
   [doom.player :as player]))

(set! *warn-on-reflection* true)

;; ================================================================
;; HUD number rendering
;; ================================================================

;; Doom status bar numbers are STTNUM0-9 (big) and STYSNUM0-9 (small)
;; We render using overlay quads with the sprite texture array

;; Doom's screen is 320x200. Status bar is 32px at bottom (y=168-199).
;; Map pixel coords to NDC: ndc_x = (px / 320) * 2 - 1
;;                          ndc_y = 1 - (py / 200) * 2  (Y flipped)
(def ^:const NUMBER-WIDTH 14.0) ;; Pixel width of a STTNUM digit
(def ^:const NUMBER-HEIGHT 16.0) ;; Pixel height

(defn- px->ndc-x [^double px] (- (* (/ px 320.0) 2.0) 1.0))
(defn- px->ndc-y [^double py] (- (* (/ py 200.0) 2.0) 1.0)) ;; Vulkan: Y+ is down, y=0→-1 (top), y=200→+1 (bottom)
(defn- pw->ndc-w [^double pw] (* (/ pw 320.0) 2.0))
(defn- ph->ndc-h [^double ph] (* (/ ph 200.0) 2.0))

(defn- digit-quads
  "Generate quads for a multi-digit number at Doom pixel position.
  x,y in Doom 320x200 pixel coords. Returns vector of [ndc-x ndc-y ndc-w ndc-h digit]."
  [x y value min-digits]
  (let [x (double x) y (double y)
        value (long value) min-digits (long min-digits)
        s (str (Math/abs value))
        s (if (< (count s) min-digits)
            (apply str (repeat (- min-digits (count s)) "0") [s])
            s)
        n (count s)
        ;; Right-justified: x is the RIGHT edge of the last digit (Doom convention)
        ;; Start drawing from x - n*NUMBER_WIDTH
        start-x (- x (* n NUMBER-WIDTH))]
    (vec
      (map-indexed
        (fn [i ch]
          {:x (px->ndc-x (+ start-x (* i NUMBER-WIDTH)))
           :y (px->ndc-y y)
           :w (pw->ndc-w NUMBER-WIDTH)
           :h (ph->ndc-h NUMBER-HEIGHT)
           :digit (- (int ch) (int \0))})
        s))))

;; ================================================================
;; HUD state
;; ================================================================

(defn build-hud-images
  "Build HUD sprite images from WAD. Returns vector of {:name :pixels :width :height}."
  [wad-data ^bytes palette]
  (let [images (atom [])
        ;; Number sprites STTNUM0-9
        _ (doseq [d (range 10)]
            (let [name (str "STTNUM" d)]
              (when-let [lump (wad/find-lump wad-data name)]
                (when-let [rgba (wad/decode-picture-at
                                  (:buf wad-data)
                                  (:offset lump) (:size lump) palette)]
                  (swap! images conj (assoc rgba :name name))))))
        ;; Percent sign
        _ (when-let [lump (wad/find-lump wad-data "STTPRCNT")]
            (when-let [rgba (wad/decode-picture-at (:buf wad-data) (:offset lump) (:size lump) palette)]
              (swap! images conj (assoc rgba :name "STTPRCNT"))))
        ;; Minus sign
        _ (when-let [lump (wad/find-lump wad-data "STTMINUS")]
            (when-let [rgba (wad/decode-picture-at (:buf wad-data) (:offset lump) (:size lump) palette)]
              (swap! images conj (assoc rgba :name "STTMINUS"))))
        ;; Status bar background
        _ (when-let [lump (wad/find-lump wad-data "STBAR")]
            (when-let [rgba (wad/decode-picture-at (:buf wad-data) (:offset lump) (:size lump) palette)]
              (swap! images conj (assoc rgba :name "STBAR"))))
        ;; Key indicators
        _ (doseq [kname ["STKEYS0" "STKEYS1" "STKEYS2" "STKEYS3" "STKEYS4" "STKEYS5"]]
            (when-let [lump (wad/find-lump wad-data kname)]
              (when-let [rgba (wad/decode-picture-at (:buf wad-data) (:offset lump) (:size lump) palette)]
                (swap! images conj (assoc rgba :name kname)))))
        ;; Face sprites (5 health levels × straight/left/right + special)
        _ (doseq [h (range 5)
                  dir ["0" "1" "2"]]
            (let [name (str "STFST" h dir)]
              (when-let [lump (wad/find-lump wad-data name)]
                (when-let [rgba (wad/decode-picture-at (:buf wad-data) (:offset lump) (:size lump) palette)]
                  (swap! images conj (assoc rgba :name name))))))
        ;; Arms numbers (small grey/yellow)
        _ (doseq [n (range 10)]
            (doseq [prefix ["STGNUM" "STYSNUM"]]
              (let [name (str prefix n)]
                (when-let [lump (wad/find-lump wad-data name)]
                  (when-let [rgba (wad/decode-picture-at (:buf wad-data) (:offset lump) (:size lump) palette)]
                    (swap! images conj (assoc rgba :name name)))))))
        ;; Menu graphics
        _ (doseq [mname ["M_SKULL1" "M_SKULL2" "M_OPTTTL" "M_SFXVOL" "M_MUSVOL"
                         "M_THERML" "M_THERMM" "M_THERMR" "M_THERMO" "M_PAUSE"
                         "M_DOOM" "M_OPTION" "M_SVOL"]]
            (when-let [lump (wad/find-lump wad-data mname)]
              (when-let [rgba (wad/decode-picture-at (:buf wad-data) (:offset lump) (:size lump) palette)]
                (swap! images conj (assoc rgba :name mname)))))]
    @images))

;; ================================================================
;; HUD overlay quad generation
;; ================================================================

(def ^:const HUD-FLOATS-PER-VERTEX 9)

(defn build-hud-quads
  "Build screen-space overlay quads for HUD.
  Returns {:vertices float[] :indices int[]} in NDC coords.
  hud-tex-map: {name → {:layer :scale-u :scale-v :width :height}}."
  [player-state hud-tex-map]
  (let [verts (float-array (* 100 4 HUD-FLOATS-PER-VERTEX)) ;; generous
        indices (int-array (* 100 6))
        vi (volatile! 0)
        ii (volatile! 0)
        vc (volatile! 0)
        emit-quad! (fn [x y w h u0 v0 u1 v1 tex-layer su sv]
                     (let [x (double x) y (double y) w (double w) h (double h)
                           u0 (double u0) v0 (double v0) u1 (double u1) v1 (double v1)
                           tex-layer (float tex-layer) su (float su) sv (float sv)
                           base-v (long @vc)
                           vo (long @vi)
                           io (long @ii)
                           light (float 1.0)]
                       ;; BL
                       (aset verts vo (float x))
                       (aset verts (+ vo 1) (float (+ y h)))
                       (aset verts (+ vo 2) (float 0.0))
                       (aset verts (+ vo 3) (float u0))
                       (aset verts (+ vo 4) (float v1))
                       (aset verts (+ vo 5) light)
                       (aset verts (+ vo 6) tex-layer)
                       (aset verts (+ vo 7) su)
                       (aset verts (+ vo 8) sv)
                       ;; BR
                       (aset verts (+ vo 9) (float (+ x w)))
                       (aset verts (+ vo 10) (float (+ y h)))
                       (aset verts (+ vo 11) (float 0.0))
                       (aset verts (+ vo 12) (float u1))
                       (aset verts (+ vo 13) (float v1))
                       (aset verts (+ vo 14) light)
                       (aset verts (+ vo 15) tex-layer)
                       (aset verts (+ vo 16) su)
                       (aset verts (+ vo 17) sv)
                       ;; TR
                       (aset verts (+ vo 18) (float (+ x w)))
                       (aset verts (+ vo 19) (float y))
                       (aset verts (+ vo 20) (float 0.0))
                       (aset verts (+ vo 21) (float u1))
                       (aset verts (+ vo 22) (float v0))
                       (aset verts (+ vo 23) light)
                       (aset verts (+ vo 24) tex-layer)
                       (aset verts (+ vo 25) su)
                       (aset verts (+ vo 26) sv)
                       ;; TL
                       (aset verts (+ vo 27) (float x))
                       (aset verts (+ vo 28) (float y))
                       (aset verts (+ vo 29) (float 0.0))
                       (aset verts (+ vo 30) (float u0))
                       (aset verts (+ vo 31) (float v0))
                       (aset verts (+ vo 32) light)
                       (aset verts (+ vo 33) tex-layer)
                       (aset verts (+ vo 34) su)
                       (aset verts (+ vo 35) sv)
                       ;; Indices
                       (aset indices io (int base-v))
                       (aset indices (+ io 1) (int (+ base-v 1)))
                       (aset indices (+ io 2) (int (+ base-v 2)))
                       (aset indices (+ io 3) (int base-v))
                       (aset indices (+ io 4) (int (+ base-v 2)))
                       (aset indices (+ io 5) (int (+ base-v 3)))
                       (vreset! vi (+ vo 36))
                       (vreset! ii (+ io 6))
                       (vreset! vc (+ base-v 4))))
        ;; Emit number digits for a value at Doom pixel position
        emit-number! (fn [x y value min-digits]
                       (let [x (double x) y (double y)
                             value (long value) min-digits (long min-digits)
                             digits (digit-quads x y value min-digits)]
                         (doseq [{:keys [x y w h digit]} digits]
                           (let [tex-name (str "STTNUM" digit)
                                 tex-info (get hud-tex-map tex-name)]
                             (when tex-info
                               (emit-quad! x y w h
                                           0.0 0.0 1.0 1.0
                                           (float (double (:layer tex-info)))
                                           (float (double (:scale-u tex-info)))
                                           (float (double (:scale-v tex-info)))))))))]

    ;; Status bar background
    (when-let [tex-info (get hud-tex-map "STBAR")]
      (emit-quad! (px->ndc-x 0.0) (px->ndc-y 168.0)
                  (pw->ndc-w 320.0) (ph->ndc-h 32.0)
                  0.0 0.0 1.0 1.0
                  (float (double (:layer tex-info)))
                  (float (double (:scale-u tex-info)))
                  (float (double (:scale-v tex-info)))))

    ;; Ammo count (Doom x=44, y=171 — right-justified)
    (let [w (get (:ammo player-state)
                 (get-in player/weapons [(:current-weapon player-state) :ammo-type])
                 0)]
      (emit-number! 44.0 171.0 w 3))

    ;; Health number (Doom x=90, y=171)
    (emit-number! 90.0 171.0 (:health player-state) 3)

    ;; Armor number (Doom x=221, y=171)
    (emit-number! 221.0 171.0 (:armor player-state) 3)

    ;; Percent signs after health and armor
    (when-let [pct (get hud-tex-map "STTPRCNT")]
      (let [pw (double (or (:width pct) 14))
            ph (double (or (:height pct) 16))]
        ;; After health (x=90, y=171)
        (emit-quad! (px->ndc-x 90.0) (px->ndc-y 171.0)
                    (pw->ndc-w pw) (ph->ndc-h ph)
                    0.0 0.0 1.0 1.0
                    (float (double (:layer pct)))
                    (float (double (:scale-u pct)))
                    (float (double (:scale-v pct))))
        ;; After armor (x=221, y=171)
        (emit-quad! (px->ndc-x 221.0) (px->ndc-y 171.0)
                    (pw->ndc-w pw) (ph->ndc-h ph)
                    0.0 0.0 1.0 1.0
                    (float (double (:layer pct)))
                    (float (double (:scale-u pct)))
                    (float (double (:scale-v pct))))))

    ;; Face widget (Doom x=143, y=168, 24x29 pixels)
    (let [health (long (or (:health player-state) 100))
          hlevel (cond (>= health 80) 0 (>= health 60) 1 (>= health 40) 2
                       (>= health 20) 3 :else 4)
          ;; Cycle direction 0/1/2 based on game time (if available)
          face-dir (mod (int (/ (double (or (:weapon-tics player-state) 0)) 5)) 3)
          face-name (str "STFST" hlevel face-dir)
          tex-info (get hud-tex-map face-name)]
      (when tex-info
        (let [fw (double (or (:width tex-info) 24))
              fh (double (or (:height tex-info) 29))]
          (emit-quad! (px->ndc-x 143.0) (px->ndc-y 168.0)
                      (pw->ndc-w fw) (ph->ndc-h fh)
                      0.0 0.0 1.0 1.0
                      (float (double (:layer tex-info)))
                      (float (double (:scale-u tex-info)))
                      (float (double (:scale-v tex-info)))))))

    ;; Key indicators (Doom x=239, y=171/181/191)
    (doseq [[k idx ky] [[:blue 0 171.0] [:yellow 1 181.0] [:red 2 191.0]]]
      (let [has-key? (contains? (:keys player-state) k)
            key-name (str "STKEYS" (if has-key? idx (+ idx 3)))
            tex-info (get hud-tex-map key-name)]
        (when tex-info
          (let [kw (double (or (:width tex-info) 16))
                kh (double (or (:height tex-info) 10))]
            (emit-quad! (px->ndc-x 239.0) (px->ndc-y (double ky))
                        (pw->ndc-w kw) (ph->ndc-h kh)
                        0.0 0.0 1.0 1.0
                        (float (double (:layer tex-info)))
                        (float (double (:scale-u tex-info)))
                        (float (double (:scale-v tex-info))))))))

    ;; Weapon sprite overlay (bottom center of screen)
    (let [ws (:weapon-state player-state)
          wf (:weapon-frame player-state)
          wt (:weapon-tics player-state)
          cur-weapon (:current-weapon player-state)
          weapon-def (get player/weapons cur-weapon)
          ;; Weapon sprite name: prefix + frame letter + "0" (all rotations)
          frame-char (case ws
                       :firing (char (+ (int \B) (min (int (or wf 0)) 2)))
                       :ready \A
                       :lowering \A
                       :raising \A
                       \A)
          sprite-prefix (when weapon-def (:sprite weapon-def))
          sprite-name (when sprite-prefix
                        (str sprite-prefix frame-char "0"))
          tex-info (when sprite-name (get hud-tex-map sprite-name))]
      (when tex-info
        (let [sw (double (:width tex-info))
              sh (double (:height tex-info))
              ;; Center weapon horizontally, place at bottom
              ;; Doom weapon sprites are drawn centered at ~160,170
              wx (- 160.0 (* sw 0.5))
              wy (- 168.0 sh)
              ;; Lower/raise offset
              y-offset (case ws
                         :lowering (* (- 1.0 (/ (double (or wt 0)) 6.0)) sh)
                         :raising (* (/ (double (or wt 0)) 6.0) sh)
                         0.0)]
          (emit-quad! (px->ndc-x wx) (px->ndc-y (+ wy y-offset))
                      (pw->ndc-w sw) (ph->ndc-h sh)
                      0.0 0.0 1.0 1.0
                      (float (double (:layer tex-info)))
                      (float (double (:scale-u tex-info)))
                      (float (double (:scale-v tex-info)))))))

    (let [n-vfloats (long @vi)
          n-indices (long @ii)]
      (when (pos? n-vfloats)
        {:vertices (java.util.Arrays/copyOf verts (int n-vfloats))
         :indices (java.util.Arrays/copyOf indices (int n-indices))}))))

;; ================================================================
;; Menu overlay
;; ================================================================

(defn build-menu-quads
  "Build screen-space overlay for the pause menu using Doom's menu graphics.
  Shows: Sound Volume title with thermometer bar + skull cursor.
  menu-sel: 0=sfx-volume, 1=mus-volume. sfx-volume/mus-volume: 0-15."
  [hud-tex-map ^long menu-sel ^long sfx-volume ^long mus-volume]
  (let [verts (float-array (* 60 4 HUD-FLOATS-PER-VERTEX))
        indices (int-array (* 60 6))
        vi (volatile! 0)
        ii (volatile! 0)
        vc (volatile! 0)
        emit-quad! (fn [x y w h u0 v0 u1 v1 tex-layer su sv light]
                     (let [x (double x) y (double y) w (double w) h (double h)
                           u0 (double u0) v0 (double v0) u1 (double u1) v1 (double v1)
                           tex-layer (float tex-layer) su (float su) sv (float sv)
                           light (float light)
                           base-v (long @vc)
                           vo (long @vi)
                           io (long @ii)]
                       (aset verts vo (float x))
                       (aset verts (+ vo 1) (float (+ y h)))
                       (aset verts (+ vo 2) (float 0.0))
                       (aset verts (+ vo 3) (float u0))
                       (aset verts (+ vo 4) (float v1))
                       (aset verts (+ vo 5) light)
                       (aset verts (+ vo 6) tex-layer)
                       (aset verts (+ vo 7) su)
                       (aset verts (+ vo 8) sv)
                       (aset verts (+ vo 9) (float (+ x w)))
                       (aset verts (+ vo 10) (float (+ y h)))
                       (aset verts (+ vo 11) (float 0.0))
                       (aset verts (+ vo 12) (float u1))
                       (aset verts (+ vo 13) (float v1))
                       (aset verts (+ vo 14) light)
                       (aset verts (+ vo 15) tex-layer)
                       (aset verts (+ vo 16) su)
                       (aset verts (+ vo 17) sv)
                       (aset verts (+ vo 18) (float (+ x w)))
                       (aset verts (+ vo 19) (float y))
                       (aset verts (+ vo 20) (float 0.0))
                       (aset verts (+ vo 21) (float u1))
                       (aset verts (+ vo 22) (float v0))
                       (aset verts (+ vo 23) light)
                       (aset verts (+ vo 24) tex-layer)
                       (aset verts (+ vo 25) su)
                       (aset verts (+ vo 26) sv)
                       (aset verts (+ vo 27) (float x))
                       (aset verts (+ vo 28) (float y))
                       (aset verts (+ vo 29) (float 0.0))
                       (aset verts (+ vo 30) (float u0))
                       (aset verts (+ vo 31) (float v0))
                       (aset verts (+ vo 32) light)
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
                       (vreset! vc (+ base-v 4))))
        emit-graphic! (fn [tex-name doom-x doom-y & {:keys [light] :or {light 1.0}}]
                        (when-let [tex-info (get hud-tex-map tex-name)]
                          (let [w (double (or (:width tex-info) 16))
                                h (double (or (:height tex-info) 16))]
                            (emit-quad! (px->ndc-x (double doom-x)) (px->ndc-y (double doom-y))
                                        (pw->ndc-w w) (ph->ndc-h h)
                                        0.0 0.0 1.0 1.0
                                        (float (double (:layer tex-info)))
                                        (float (double (:scale-u tex-info)))
                                        (float (double (:scale-v tex-info)))
                                        (float light)))))

        emit-therm! (fn [therm-x therm-y vol]
                      (let [therm-x (double therm-x) therm-y (double therm-y) vol (long vol)
                            therml-w (double (or (:width (get hud-tex-map "M_THERML")) 6))
                            thermm-w (double (or (:width (get hud-tex-map "M_THERMM")) 9))]
                        ;; Left cap
                        (emit-graphic! "M_THERML" therm-x therm-y)
                        ;; 16 middle segments
                        (dotimes [i 16]
                          (emit-graphic! "M_THERMM" (+ therm-x therml-w (* (double i) thermm-w)) therm-y))
                        ;; Right cap
                        (emit-graphic! "M_THERMR" (+ therm-x therml-w (* 16.0 thermm-w)) therm-y)
                        ;; Notch at volume position
                        (emit-graphic! "M_THERMO" (+ therm-x therml-w (* (double vol) thermm-w)) therm-y)))

        ;; Layout constants — Doom screen is 320x200
        ;; Thermometer: 6 + 16*9 + 6 = 156px wide
        therm-total-w 156.0
        label-x 82.0         ;; left edge of labels
        therm-x (double (/ (- 320.0 therm-total-w) 2.0)) ;; centered

        ;; Dark background using STBAR
        _ (when-let [tex-info (get hud-tex-map "STBAR")]
            (emit-quad! (px->ndc-x 30.0) (px->ndc-y 20.0)
                        (pw->ndc-w 260.0) (ph->ndc-h 150.0)
                        0.0 0.0 0.5 0.5
                        (float (double (:layer tex-info)))
                        (float (double (:scale-u tex-info)))
                        (float (double (:scale-v tex-info)))
                        (float 0.12)))

        ;; Title: M_SVOL ("Sound Volume") — centered
        _ (let [title-w (double (or (:width (get hud-tex-map "M_SVOL")) 168))
                title-x (/ (- 320.0 title-w) 2.0)]
            (emit-graphic! "M_SVOL" title-x 28.0))

        ;; --- SFX Volume ---
        sfx-label-y 62.0
        _ (emit-graphic! "M_SFXVOL" label-x sfx-label-y)
        sfx-therm-y 78.0
        _ (emit-therm! therm-x sfx-therm-y sfx-volume)

        ;; --- Music Volume ---
        mus-label-y 105.0
        _ (emit-graphic! "M_MUSVOL" label-x mus-label-y)
        mus-therm-y 121.0
        _ (emit-therm! therm-x mus-therm-y mus-volume)

        ;; Skull cursor (M_SKULL1) next to selected item label
        cursor-y (case menu-sel 0 sfx-label-y 1 mus-label-y sfx-label-y)
        _ (emit-graphic! "M_SKULL1" (- label-x 24.0) cursor-y)
        n-vfloats (long @vi)
        n-indices (long @ii)]
    (when (pos? n-vfloats)
      {:vertices (java.util.Arrays/copyOf verts (int n-vfloats))
       :indices (java.util.Arrays/copyOf indices (int n-indices))})))
