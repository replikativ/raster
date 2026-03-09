(ns doom.sound
  "Sound system: load WAD sounds into OpenAL, spatial playback."
  (:require
   [doom.wad :as wad]
   [doom.level :as level]
   [raster.vk.audio :as audio])
  (:import
   [org.lwjgl.openal AL10]
   [org.lwjgl.system MemoryUtil]
   [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

;; ================================================================
;; WAD sound loading into OpenAL
;; ================================================================

(def doom-sounds
  "Sound names needed for E1M1 gameplay."
  {:pistol "DSPISTOL"
   :shotgun "DSSHOTGN"
   :door-open "DSDOROPN"
   :door-close "DSDORCLS"
   :switch "DSSWTCHN"
   :pickup-item "DSITEMUP"
   :pickup-weapon "DSWPNUP"
   :player-pain "DSPLPAIN"
   :player-death "DSPLDETH"
   :monster-sight-1 "DSPOSIT1"
   :monster-sight-2 "DSPOSIT2"
   :monster-pain "DSPOPAIN"
   :monster-death-1 "DSPODTH1"
   :monster-death-2 "DSPODTH2"
   :imp-sight "DSBGSIT1"
   :imp-attack "DSFIRSHT"
   :imp-pain "DSDMPAIN"
   :imp-death "DSBGDTH1"
   :fireball-hit "DSFIRXPL"
   :barrel-explode "DSBAREXP"
   :lift "DSPSTART"})

(defn load-wad-sound!
  "Load a single WAD sound into OpenAL. Returns buffer-id or nil.
  Doom sounds are 8-bit unsigned PCM. We convert to 16-bit signed for OpenAL."
  [audio-ctx wad-data sound-kw ^String lump-name]
  (when-let [snd (wad/parse-sound-lump wad-data lump-name)]
    (let [^bytes samples (:samples snd)
          n (:num-samples snd)
          rate (:sample-rate snd)
          ;; Convert 8-bit unsigned to 16-bit signed
          buf (MemoryUtil/memAlloc (* n 2))
          _ (dotimes [i n]
              (let [u8 (bit-and (aget samples i) 0xFF)
                    ;; 8-bit unsigned (128=silence) → 16-bit signed
                    s16 (short (* (- u8 128) 256))]
                (.putShort ^ByteBuffer buf s16)))
          _ (.flip ^ByteBuffer buf)
          al-buf (AL10/alGenBuffers)]
      (AL10/alBufferData al-buf AL10/AL_FORMAT_MONO16
                         ^ByteBuffer buf (int rate))
      (MemoryUtil/memFree buf)
      (swap! (:sounds audio-ctx) assoc sound-kw al-buf)
      al-buf)))

(defn load-all-sounds!
  "Load all gameplay sounds from WAD into OpenAL context."
  [audio-ctx wad-data]
  (doseq [[kw lump-name] doom-sounds]
    (try
      (load-wad-sound! audio-ctx wad-data kw lump-name)
      (catch Exception e
        (println "Warning: failed to load sound" lump-name (.getMessage e)))))
  (println "Loaded" (count @(:sounds audio-ctx)) "sounds"))

;; ================================================================
;; Sound playback helpers
;; ================================================================

(defn volume->gain
  "Convert Doom volume (0-15) to OpenAL gain (0.0-1.0)."
  ^double [^long vol]
  (/ (double (max 0 (min vol 15))) 15.0))

(defn play-at!
  "Play a sound at a position in Doom coordinates."
  ([audio-ctx sound-kw doom-x doom-y doom-z]
   (play-at! audio-ctx sound-kw doom-x doom-y doom-z 12))
  ([audio-ctx sound-kw doom-x doom-y doom-z sfx-vol]
   (let [doom-x (double doom-x) doom-y (double doom-y) doom-z (double doom-z)
         vx (level/doom->vk-x doom-x)
         vy (level/doom->vk-y doom-z)
         vz (level/doom->vk-z doom-y)
         gain (volume->gain (long sfx-vol))]
     (try
       (audio/play-sound! audio-ctx sound-kw
                          :position [vx vy vz]
                          :gain gain)
       (catch Exception _ nil)))))

(defn play-ui!
  "Play a non-positional UI sound (pickup, weapon)."
  ([audio-ctx sound-kw]
   (play-ui! audio-ctx sound-kw 12))
  ([audio-ctx sound-kw sfx-vol]
   (try
     (audio/play-sound! audio-ctx sound-kw :gain (volume->gain (long sfx-vol)))
     (catch Exception _ nil))))

(defn update-listener!
  "Update the OpenAL listener position/orientation from camera."
  [audio-ctx cam]
  (let [[px py pz] (:pos cam)
        yaw (double (:yaw cam))
        pitch (double (:pitch cam))
        cp (Math/cos pitch)
        fx (* (Math/sin yaw) cp)
        fy (Math/sin pitch)
        fz (* (- (Math/cos yaw)) cp)]
    (audio/set-listener! audio-ctx
                         [px py pz]
                         [fx fy fz]
                         [0.0 1.0 0.0])))
