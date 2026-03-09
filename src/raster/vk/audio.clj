(ns raster.vk.audio
  "Audio system via OpenAL (LWJGL). Supports loading OGG, positional audio."
  (:import
   [org.lwjgl.openal AL ALC AL10 ALC10]
   [org.lwjgl.stb STBVorbis]
   [org.lwjgl.system MemoryStack MemoryUtil]
   [java.nio ByteBuffer ShortBuffer IntBuffer]))

(set! *warn-on-reflection* true)

(defrecord AudioContext [device al-context sounds])

;; ================================================================
;; Context
;; ================================================================

(defn create-audio-context
  "Open default AL device, create context, make current."
  []
  (let [^CharSequence default-dev nil
        device (ALC10/alcOpenDevice default-dev)]
    (when (zero? device)
      (throw (ex-info "Failed to open OpenAL device" {})))
    (let [^IntBuffer no-attrs nil
          al-ctx (ALC10/alcCreateContext device no-attrs)]
      (when (zero? al-ctx)
        (ALC10/alcCloseDevice device)
        (throw (ex-info "Failed to create OpenAL context" {})))
      (ALC10/alcMakeContextCurrent al-ctx)
      (AL/createCapabilities (ALC/createCapabilities device))
      (->AudioContext device al-ctx (atom {})))))

(defn destroy-audio-context!
  [^AudioContext ctx]
  ;; Delete all buffers
  (doseq [[_ buf-id] @(:sounds ctx)]
    (AL10/alDeleteBuffers (int buf-id)))
  (ALC10/alcMakeContextCurrent 0)
  (ALC10/alcDestroyContext (:al-context ctx))
  (ALC10/alcCloseDevice (:device ctx)))

;; ================================================================
;; Sound loading
;; ================================================================

(defn load-sound
  "Decode OGG file via STBVorbis, upload to AL buffer. Associates name-kw with buffer ID."
  [^AudioContext ctx name-kw ^String path]
  (let [stack (MemoryStack/stackPush)]
    (try
      (let [p-channels (.mallocInt stack 1)
            p-rate (.mallocInt stack 1)
            ;; STBVorbis decodes to interleaved short samples
            ^ShortBuffer samples (STBVorbis/stb_vorbis_decode_filename path p-channels p-rate)
            _ (when-not samples
                (throw (ex-info (str "Failed to decode OGG: " path) {:path path})))
            channels (.get p-channels 0)
            sample-rate (.get p-rate 0)
            format (case (int channels)
                     1 AL10/AL_FORMAT_MONO16
                     2 AL10/AL_FORMAT_STEREO16)
            buf-id (AL10/alGenBuffers)]
        (AL10/alBufferData buf-id format samples sample-rate)
        (MemoryUtil/memFree samples)
        (swap! (:sounds ctx) assoc name-kw buf-id)
        buf-id)
      (finally
        (MemoryStack/stackPop)))))

;; ================================================================
;; Playback
;; ================================================================

(def ^:private source-pool
  "Pool of reusable OpenAL sources to prevent source leaking."
  (atom {:sources [] :next 0}))

(defn- get-or-create-source
  "Get a reusable source, creating if needed. Recycles stopped sources."
  ^long []
  (let [{:keys [sources]} @source-pool]
    ;; Try to find a stopped source to reuse
    (loop [i 0]
      (if (< i (count sources))
        (let [src (int (nth sources i))
              state (AL10/alGetSourcei src AL10/AL_SOURCE_STATE)]
          (if (or (= state AL10/AL_STOPPED) (= state AL10/AL_INITIAL))
            (do (AL10/alSourcei src AL10/AL_BUFFER 0) ;; detach old buffer
                src)
            (recur (inc i))))
        ;; No free source, create new one (up to 32)
        (if (< (count sources) 32)
          (let [src (AL10/alGenSources)]
            (when (pos? src)
              (swap! source-pool update :sources conj src))
            src)
          ;; All 32 in use, force-recycle oldest
          (let [src (int (nth sources (mod (:next @source-pool) 32)))]
            (AL10/alSourceStop src)
            (AL10/alSourcei src AL10/AL_BUFFER 0)
            (swap! source-pool update :next inc)
            src))))))

(defn play-sound!
  "Attach buffer to a pooled source, play. Returns source-id."
  [^AudioContext ctx name-kw & {:keys [gain pitch position loop?]
                                :or {gain 1.0 pitch 1.0 loop? false}}]
  ;; Ensure AL context is current on this thread
  (ALC10/alcMakeContextCurrent (:al-context ctx))
  (let [buf-id (get @(:sounds ctx) name-kw)]
    (when-not buf-id
      (throw (ex-info (str "Sound not loaded: " name-kw) {:name name-kw})))
    (let [src (get-or-create-source)]
      (when (pos? src)
        (AL10/alSourcei src AL10/AL_BUFFER (int buf-id))
        (AL10/alSourcef src AL10/AL_GAIN (float gain))
        (AL10/alSourcef src AL10/AL_PITCH (float pitch))
        (AL10/alSourcei src AL10/AL_LOOPING (if loop? AL10/AL_TRUE AL10/AL_FALSE))
        (if position
          (let [[x y z] position]
            (AL10/alSourcei src AL10/AL_SOURCE_RELATIVE AL10/AL_FALSE)
            (AL10/alSource3f src AL10/AL_POSITION (float x) (float y) (float z)))
          ;; No position = relative to listener (UI sounds)
          (do (AL10/alSourcei src AL10/AL_SOURCE_RELATIVE AL10/AL_TRUE)
              (AL10/alSource3f src AL10/AL_POSITION (float 0) (float 0) (float 0))))
        (AL10/alSourcePlay src)
        src))))

(defn stop-sound!
  "Stop and delete a source."
  [_ctx ^long source-id]
  (AL10/alSourceStop (int source-id))
  (AL10/alDeleteSources (int source-id)))

(defn set-listener!
  "Set listener position + orientation for 3D audio.
  forward/up are [x y z] vectors."
  [_ctx pos forward up]
  (let [[px py pz] pos
        [fx fy fz] forward
        [ux uy uz] up]
    (AL10/alListener3f AL10/AL_POSITION (float px) (float py) (float pz))
    (AL10/alListenerfv AL10/AL_ORIENTATION
                       (float-array [(float fx) (float fy) (float fz)
                                     (float ux) (float uy) (float uz)]))))

(defn update-source!
  "Update source properties."
  [_ctx source-id & {:keys [position gain]}]
  (when position
    (let [[x y z] position]
      (AL10/alSource3f (int source-id) AL10/AL_POSITION (float x) (float y) (float z))))
  (when gain
    (AL10/alSourcef (int source-id) AL10/AL_GAIN (float gain))))
