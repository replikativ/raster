(ns raster.vk.text
  "Bitmap font rendering via STB TrueType.
  Bakes a TTF font into a texture atlas, then builds vertex quads for text strings.
  Vertex format matches the sprite pipeline: [x y u v] per vertex, 6 verts per glyph."
  (:require
   [raster.vk.texture :as texture])
  (:import
   [org.lwjgl.stb STBTruetype STBTTBakedChar STBTTAlignedQuad]
   [org.lwjgl.system MemoryUtil MemoryStack]
   [java.nio ByteBuffer]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Font atlas baking
;; ================================================================

(defrecord BitmapFont [texture desc-set char-data
                       ^long first-char ^long num-chars
                       ^long atlas-w ^long atlas-h
                       ^double font-size])

(defn- read-font-file ^ByteBuffer [^String path]
  (let [bytes (java.nio.file.Files/readAllBytes (java.nio.file.Paths/get path (into-array String [])))
        buf (MemoryUtil/memAlloc (alength bytes))]
    (.put buf bytes)
    (.flip buf)
    buf))

(defn- find-system-font
  "Find a TTF font file on the system. Tries serif first, then sans, then mono.
  Works on Linux, macOS, and Windows."
  [style]
  (let [candidates
        (case style
          :serif
          [;; Linux
           "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
           "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
           "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"
           ;; macOS
           "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
           "/System/Library/Fonts/NewYork.ttf"
           "/Library/Fonts/Georgia.ttf"
           ;; Windows
           "C:/Windows/Fonts/times.ttf"
           "C:/Windows/Fonts/georgia.ttf"
           "C:/Windows/Fonts/cambria.ttc"]
          :mono
          ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
           "/System/Library/Fonts/Menlo.ttc"
           "/System/Library/Fonts/SFMono-Regular.otf"
           "C:/Windows/Fonts/consola.ttf"
           "C:/Windows/Fonts/cour.ttf"]
          ;; default: sans
          ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
           "/System/Library/Fonts/Helvetica.ttc"
           "C:/Windows/Fonts/arial.ttf"])]
    (or (first (filter #(.exists (java.io.File. ^String %)) candidates))
        ;; Ultimate fallback: try DejaVu (most common on Linux)
        (let [fallback "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
          (when (.exists (java.io.File. fallback)) fallback)))))

(defn create-bitmap-font
  "Bake a TTF font into a texture atlas and upload to GPU.
  Returns BitmapFont record with texture, descriptor set, and char metrics.

  Options:
    :font-size  - pixel height (default 32)
    :font-style - :serif (default), :mono, or :sans
    :font-path  - explicit TTF path (overrides style lookup)
    :atlas-size - atlas width/height in pixels (default 512)
    :first-char - first ASCII char to bake (default 32, space)
    :num-chars  - number of chars (default 96, ASCII 32-127)"
  [ctx pool layout & {:keys [font-size font-style font-path atlas-size first-char num-chars]
                      :or {font-size 32.0 font-style :serif atlas-size 512 first-char 32 num-chars 96}}]
  (let [path (or font-path
                 (find-system-font font-style)
                 (throw (ex-info "No suitable font found on this system"
                                 {:style font-style})))
        font-buf (read-font-file path)
        atlas-w (int atlas-size)
        atlas-h (int atlas-size)
        char-data (STBTTBakedChar/malloc (int num-chars))
        bitmap (MemoryUtil/memAlloc (* atlas-w atlas-h))]
    (try
      ;; Bake font bitmap (single channel)
      (STBTruetype/stbtt_BakeFontBitmap font-buf (float font-size)
                                        bitmap atlas-w atlas-h (int first-char) char-data)

      ;; Convert 1-channel bitmap to RGBA
      (let [rgba-size (* atlas-w atlas-h 4)
            rgba (MemoryUtil/memAlloc rgba-size)]
        (dotimes [i (* atlas-w atlas-h)]
          (let [v (.get bitmap i)]
            (.put rgba (int (* i 4)) (unchecked-byte -1))       ;; R = 255 (white)
            (.put rgba (int (+ (* i 4) 1)) (unchecked-byte -1)) ;; G = 255
            (.put rgba (int (+ (* i 4) 2)) (unchecked-byte -1)) ;; B = 255
            (.put rgba (int (+ (* i 4) 3)) v)))                 ;; A = glyph coverage

        ;; Upload as texture
        (let [image-data {:pixels rgba :width atlas-w :height atlas-h :channels 4}
              tex (texture/create-texture ctx image-data :filter-mode :linear :address-mode :clamp)
              desc-set (texture/allocate-texture-descriptor-set
                        (:device ctx) pool layout tex)]
          (MemoryUtil/memFree rgba)
          (->BitmapFont tex desc-set char-data
                        first-char num-chars atlas-w atlas-h font-size)))
      (finally
        (MemoryUtil/memFree bitmap)
        (MemoryUtil/memFree font-buf)))))

(defn destroy-bitmap-font!
  [ctx ^BitmapFont font]
  (texture/destroy-texture! (:device ctx) (:allocator ctx) (:texture font))
  (.free ^STBTTBakedChar (:char-data font)))

;; ================================================================
;; Text vertex generation
;; ================================================================

(defn build-text-vertices
  "Build vertex data for a text string. Returns [float-array vertex-count].
  Vertices are [x y u v] per vertex, 6 vertices per character (2 triangles).

  pos: [x y] top-left corner in screen pixels
  scale: size multiplier (1.0 = font-size pixels tall)"
  ^objects [^BitmapFont font ^String text pos ^double scale]
  (let [pos-x (double (nth pos 0))
        pos-y (double (nth pos 1))
        char-data ^STBTTBakedChar (:char-data font)
        first-char (long (:first-char font))
        num-chars (long (:num-chars font))
        n (.length text)
        max-floats (* n 6 4) ;; 6 verts * 4 floats per char
        verts (float-array max-floats)
        stack (MemoryStack/stackPush)
        px (.floats stack (float pos-x))
        py (.floats stack (float pos-y))
        q (STBTTAlignedQuad/malloc stack)]
    (try
      ;; STB lays out at px,py in top-down screen coords.
      ;; We init px=0, py=0, then scale + translate + flip Y for bottom-up ortho.
      (.put px 0 (float 0.0))
      (.put py 0 (float 0.0))
      (loop [i 0 vi 0]
        (if (>= i n)
          (object-array [verts (/ vi 4)])  ;; [float-array vertex-count]
          (let [c (- (int (.charAt text i)) first-char)]
            (if (or (< c 0) (>= c num-chars))
              ;; Skip unprintable chars, advance cursor
              (do (.put px 0 (float (+ (double (.get px 0)) (* (double (:font-size font)) 0.5))))
                  (recur (inc i) vi))
              (do
                (STBTruetype/stbtt_GetBakedQuad char-data
                                                (int (:atlas-w font)) (int (:atlas-h font))
                                                (int c) px py q true)
                ;; STB gives quads in top-down coords at origin.
                ;; Transform: x' = pos_x + stb_x * scale, y' = pos_y - stb_y * scale
                (let [sx0 (+ pos-x (* (double (.x0 q)) scale))
                      sx1 (+ pos-x (* (double (.x1 q)) scale))
                      sy0 (- pos-y (* (double (.y0 q)) scale))
                      sy1 (- pos-y (* (double (.y1 q)) scale))
                      u0 (double (.s0 q))
                      v0 (double (.t0 q))
                      u1 (double (.s1 q))
                      v1 (double (.t1 q))
                      vi (int vi)]
                  ;; Triangle 1: top-left, top-right, bottom-left
                  (aset verts vi (float sx0))
                  (aset verts (+ vi 1) (float sy0))
                  (aset verts (+ vi 2) (float u0))
                  (aset verts (+ vi 3) (float v0))

                  (aset verts (+ vi 4) (float sx1))
                  (aset verts (+ vi 5) (float sy0))
                  (aset verts (+ vi 6) (float u1))
                  (aset verts (+ vi 7) (float v0))

                  (aset verts (+ vi 8) (float sx0))
                  (aset verts (+ vi 9) (float sy1))
                  (aset verts (+ vi 10) (float u0))
                  (aset verts (+ vi 11) (float v1))

                  ;; Triangle 2: top-right, bottom-right, bottom-left
                  (aset verts (+ vi 12) (float sx1))
                  (aset verts (+ vi 13) (float sy0))
                  (aset verts (+ vi 14) (float u1))
                  (aset verts (+ vi 15) (float v0))

                  (aset verts (+ vi 16) (float sx1))
                  (aset verts (+ vi 17) (float sy1))
                  (aset verts (+ vi 18) (float u1))
                  (aset verts (+ vi 19) (float v1))

                  (aset verts (+ vi 20) (float sx0))
                  (aset verts (+ vi 21) (float sy1))
                  (aset verts (+ vi 22) (float u0))
                  (aset verts (+ vi 23) (float v1))

                  (recur (inc i) (+ vi 24))))))))
      (finally
        (MemoryStack/stackPop)))))

(defn text-width
  "Estimate text width in pixels at given scale."
  ^double [^BitmapFont font ^String text ^double scale]
  (let [char-data ^STBTTBakedChar (:char-data font)
        first-char (long (:first-char font))
        num-chars (long (:num-chars font))
        n (.length text)]
    (loop [i 0 w 0.0]
      (if (>= i n)
        (* w scale)
        (let [c (- (int (.charAt text i)) first-char)]
          (if (or (< c 0) (>= c num-chars))
            (recur (inc i) (+ w (* (:font-size font) 0.5)))
            (let [^org.lwjgl.stb.STBTTBakedChar cd (.get char-data (int c))]
              (recur (inc i) (+ w (double (.xadvance cd)))))))))))
