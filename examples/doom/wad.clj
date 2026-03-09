(ns doom.wad
  "Pure-data WAD binary parser. No GPU dependencies.
  Parses all map lumps, textures, flats, and palettes from Doom WAD files."
  (:import
   [java.io RandomAccessFile]
   [java.nio ByteBuffer ByteOrder]
   [java.nio.channels FileChannel$MapMode]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Core WAD loading
;; ================================================================

(defn- read-char8
  "Read an 8-byte null-terminated ASCII string from ByteBuffer at current position."
  ^String [^ByteBuffer buf]
  (let [sb (StringBuilder. 8)]
    (dotimes [_ 8]
      (let [b (.get buf)]
        (when (not= b 0)
          (.append sb (char (bit-and b 0xFF))))))
    (.toString sb)))

(defn- read-char8-at
  "Read an 8-byte null-terminated ASCII string at absolute offset."
  ^String [^ByteBuffer buf ^long offset]
  (let [sb (StringBuilder. 8)]
    (dotimes [i 8]
      (let [b (.get buf (int (+ offset i)))]
        (when (not= b 0)
          (.append sb (char (bit-and b 0xFF))))))
    (.toString sb)))

(defn load-wad
  "Memory-map a WAD file and parse its directory.
  Returns {:type :IWAD/:PWAD :directory [{:name :offset :size}...] :buf ByteBuffer}"
  [^String path]
  (let [raf (RandomAccessFile. path "r")
        ch (.getChannel raf)
        size (.size ch)
        buf (.map ch FileChannel$MapMode/READ_ONLY 0 size)]
    (.close ch)
    (.close raf)
    (.order buf ByteOrder/LITTLE_ENDIAN)
    ;; Parse header
    (let [type-bytes (byte-array 4)]
      (.get buf type-bytes)
      (let [wad-type (String. type-bytes "ASCII")
            numlumps (.getInt buf)
            infotableofs (.getInt buf)]
        ;; Parse directory
        (.position buf (int infotableofs))
        (let [dir (into []
                    (map (fn [_]
                           (let [filepos (.getInt buf)
                                 lsize (.getInt buf)
                                 name (read-char8 buf)]
                             {:name name :offset filepos :size lsize})))
                    (range numlumps))]
          (.position buf 0)
          {:type (keyword wad-type)
           :directory dir
           :buf buf})))))

(defn find-lump
  "Find a lump by name in the WAD directory (case-insensitive). Returns {:name :offset :size} or nil."
  [wad ^String name]
  (let [uname (.toUpperCase name)]
    (first (filter #(= (.toUpperCase ^String (:name %)) uname) (:directory wad)))))

(defn find-lump-between
  "Find a lump by name between start/end marker lumps (e.g. F_START/F_END)."
  [wad ^String name ^String start-marker ^String end-marker]
  (let [dir (:directory wad)
        start-idx (some (fn [[i entry]] (when (= (:name entry) start-marker) i))
                    (map-indexed vector dir))
        end-idx (some (fn [[i entry]] (when (= (:name entry) end-marker) i))
                  (map-indexed vector dir))]
    (when (and start-idx end-idx)
      (first (filter #(= (:name %) name)
               (subvec dir (inc (int start-idx)) (int end-idx)))))))

;; ================================================================
;; Map lump discovery
;; ================================================================

(def ^:private map-lump-names
  ["THINGS" "LINEDEFS" "SIDEDEFS" "VERTEXES" "SEGS"
   "SSECTORS" "NODES" "SECTORS" "REJECT" "BLOCKMAP"])

(defn find-map-lumps
  "Find all map lumps for a given map name (e.g. \"E1M1\").
  Returns map of keyword->lump: {:THINGS lump :LINEDEFS lump ...}"
  [wad ^String map-name]
  (let [dir (:directory wad)
        ;; Find the map marker lump
        marker-idx (some (fn [[i entry]]
                           (when (= (:name entry) map-name) i))
                     (map-indexed vector dir))]
    (when marker-idx
      (let [marker-idx (int marker-idx)]
        (into {}
          (keep (fn [^String lname]
                  (let [lump (nth dir (+ marker-idx 1
                                        (.indexOf ^java.util.List
                                          (mapv :name (subvec dir (inc marker-idx)
                                                        (min (+ marker-idx 11) (count dir))))
                                          lname))
                              nil)]
                    (when (and lump (= (:name lump) lname))
                      [(keyword lname) lump]))))
          map-lump-names)))))

(defn find-map-lumps
  "Find all map lumps for a given map name (e.g. \"E1M1\").
  Returns map of keyword->lump: {:THINGS lump :LINEDEFS lump ...}"
  [wad ^String map-name]
  (let [dir (:directory wad)
        marker-idx (some (fn [[i entry]]
                           (when (= (:name entry) map-name) i))
                     (map-indexed vector dir))]
    (when marker-idx
      (let [mi (int marker-idx)]
        (into {}
          (for [offset (range 1 11)
                :let [idx (+ mi offset)]
                :when (< idx (count dir))
                :let [entry (nth dir idx)]
                :when (some #(= (:name entry) %) map-lump-names)]
            [(keyword (:name entry)) entry]))))))

;; ================================================================
;; Map lump parsers
;; ================================================================

(defn parse-vertexes
  "Parse VERTEXES lump. Returns vector of [x y] (short values)."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 4)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 4))]
               [(.getShort buf (int pos))
                (.getShort buf (int (+ pos 2)))])))
      (range count))))

(defn parse-linedefs
  "Parse LINEDEFS lump. Returns vector of linedef maps."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 14)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 14))
                   ;; Read as unsigned shorts
                   v1 (bit-and (.getShort buf (int pos)) 0xFFFF)
                   v2 (bit-and (.getShort buf (int (+ pos 2))) 0xFFFF)
                   flags (bit-and (.getShort buf (int (+ pos 4))) 0xFFFF)
                   special (bit-and (.getShort buf (int (+ pos 6))) 0xFFFF)
                   tag (bit-and (.getShort buf (int (+ pos 8))) 0xFFFF)
                   s0 (bit-and (.getShort buf (int (+ pos 10))) 0xFFFF)
                   s1 (bit-and (.getShort buf (int (+ pos 12))) 0xFFFF)]
               {:v1 v1 :v2 v2 :flags flags :special special :tag tag
                :front-sidedef (when (not= s0 0xFFFF) s0)
                :back-sidedef (when (not= s1 0xFFFF) s1)})))
      (range count))))

(defn parse-sidedefs
  "Parse SIDEDEFS lump. Returns vector of sidedef maps."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 30)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 30))]
               {:x-offset (.getShort buf (int pos))
                :y-offset (.getShort buf (int (+ pos 2)))
                :upper (read-char8-at buf (+ pos 4))
                :lower (read-char8-at buf (+ pos 12))
                :middle (read-char8-at buf (+ pos 20))
                :sector (bit-and (.getShort buf (int (+ pos 28))) 0xFFFF)})))
      (range count))))

(defn parse-sectors
  "Parse SECTORS lump. Returns vector of sector maps."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 26)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 26))]
               {:floor-height (.getShort buf (int pos))
                :ceil-height (.getShort buf (int (+ pos 2)))
                :floor-tex (read-char8-at buf (+ pos 4))
                :ceil-tex (read-char8-at buf (+ pos 12))
                :light-level (bit-and (.getShort buf (int (+ pos 20))) 0xFFFF)
                :special (bit-and (.getShort buf (int (+ pos 22))) 0xFFFF)
                :tag (bit-and (.getShort buf (int (+ pos 24))) 0xFFFF)})))
      (range count))))

(defn parse-things
  "Parse THINGS lump. Returns vector of thing maps."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 10)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 10))]
               {:x (.getShort buf (int pos))
                :y (.getShort buf (int (+ pos 2)))
                :angle (bit-and (.getShort buf (int (+ pos 4))) 0xFFFF)
                :type (bit-and (.getShort buf (int (+ pos 6))) 0xFFFF)
                :flags (bit-and (.getShort buf (int (+ pos 8))) 0xFFFF)})))
      (range count))))

(defn parse-segs
  "Parse SEGS lump. Returns vector of seg maps."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 12)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 12))]
               {:v1 (bit-and (.getShort buf (int pos)) 0xFFFF)
                :v2 (bit-and (.getShort buf (int (+ pos 2))) 0xFFFF)
                :angle (.getShort buf (int (+ pos 4)))
                :linedef (bit-and (.getShort buf (int (+ pos 6))) 0xFFFF)
                :direction (bit-and (.getShort buf (int (+ pos 8))) 0xFFFF)
                :offset (.getShort buf (int (+ pos 10)))})))
      (range count))))

(defn parse-ssectors
  "Parse SSECTORS lump. Returns vector of {:num-segs :first-seg}."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 4)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 4))]
               {:num-segs (bit-and (.getShort buf (int pos)) 0xFFFF)
                :first-seg (bit-and (.getShort buf (int (+ pos 2))) 0xFFFF)})))
      (range count))))

(defn parse-nodes
  "Parse NODES lump. Returns vector of BSP node maps."
  [wad lump]
  (let [^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (/ (:size lump) 28)]
    (into []
      (map (fn [i]
             (let [pos (+ offset (* i 28))]
               {:x (.getShort buf (int pos))
                :y (.getShort buf (int (+ pos 2)))
                :dx (.getShort buf (int (+ pos 4)))
                :dy (.getShort buf (int (+ pos 6)))
                ;; Bounding boxes: top bottom left right for each child
                :bbox-right [(bit-and (.getShort buf (int (+ pos 8))) 0xFFFF)
                             (bit-and (.getShort buf (int (+ pos 10))) 0xFFFF)
                             (bit-and (.getShort buf (int (+ pos 12))) 0xFFFF)
                             (bit-and (.getShort buf (int (+ pos 14))) 0xFFFF)]
                :bbox-left [(bit-and (.getShort buf (int (+ pos 16))) 0xFFFF)
                            (bit-and (.getShort buf (int (+ pos 18))) 0xFFFF)
                            (bit-and (.getShort buf (int (+ pos 20))) 0xFFFF)
                            (bit-and (.getShort buf (int (+ pos 22))) 0xFFFF)]
                :child-right (bit-and (.getShort buf (int (+ pos 24))) 0xFFFF)
                :child-left (bit-and (.getShort buf (int (+ pos 26))) 0xFFFF)})))
      (range count))))

;; ================================================================
;; Full map data loader
;; ================================================================

(defn load-map
  "Load and parse all lumps for a given map. Returns a map with all parsed data."
  [wad ^String map-name]
  (let [lumps (find-map-lumps wad map-name)]
    (when lumps
      {:name map-name
       :vertexes (parse-vertexes wad (:VERTEXES lumps))
       :linedefs (parse-linedefs wad (:LINEDEFS lumps))
       :sidedefs (parse-sidedefs wad (:SIDEDEFS lumps))
       :sectors (parse-sectors wad (:SECTORS lumps))
       :things (parse-things wad (:THINGS lumps))
       :segs (parse-segs wad (:SEGS lumps))
       :ssectors (parse-ssectors wad (:SSECTORS lumps))
       :nodes (parse-nodes wad (:NODES lumps))})))

;; ================================================================
;; Texture system parsing
;; ================================================================

(defn parse-playpal
  "Parse PLAYPAL lump. Returns first palette as byte[768] (256 RGB triplets)."
  [wad]
  (let [lump (find-lump wad "PLAYPAL")
        ^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        palette (byte-array 768)]
    (dotimes [i 768]
      (aset palette i (.get buf (int (+ offset i)))))
    palette))

(defn parse-pnames
  "Parse PNAMES lump. Returns vector of patch name strings."
  [wad]
  (let [lump (find-lump wad "PNAMES")
        ^ByteBuffer buf (:buf wad)
        offset (int (:offset lump))
        count (.getInt buf (int offset))]
    (into []
      (map (fn [i]
             (read-char8-at buf (+ offset 4 (* i 8)))))
      (range count))))

(defn parse-texture-defs
  "Parse TEXTURE1 or TEXTURE2 lump. Returns vector of texture definition maps."
  [wad ^String lump-name]
  (when-let [lump (find-lump wad lump-name)]
    (let [^ByteBuffer buf (:buf wad)
          base (int (:offset lump))
          numtex (.getInt buf base)]
      (into []
        (map (fn [i]
               (let [tex-offset (+ base (.getInt buf (int (+ base 4 (* i 4)))))
                     name (read-char8-at buf tex-offset)
                     width (bit-and (.getShort buf (int (+ tex-offset 12))) 0xFFFF)
                     height (bit-and (.getShort buf (int (+ tex-offset 14))) 0xFFFF)
                     patchcount (bit-and (.getShort buf (int (+ tex-offset 20))) 0xFFFF)]
                 {:name name
                  :width width
                  :height height
                  :patches (into []
                             (map (fn [j]
                                    (let [pp (+ tex-offset 22 (* j 10))]
                                      {:originx (.getShort buf (int pp))
                                       :originy (.getShort buf (int (+ pp 2)))
                                       :patch-idx (bit-and (.getShort buf (int (+ pp 4))) 0xFFFF)})))
                             (range patchcount))})))
        (range numtex)))))

(defn decode-picture
  "Decode a Doom picture-format lump (patch/sprite) to palette-indexed pixels.
  Returns {:width :height :pixels byte[] :left-offset :top-offset} or nil."
  [wad ^String lump-name]
  (when-let [lump (find-lump wad lump-name)]
    (let [^ByteBuffer buf (:buf wad)
          base (int (:offset lump))
          width (bit-and (.getShort buf base) 0xFFFF)
          height (bit-and (.getShort buf (int (+ base 2))) 0xFFFF)
          left-offset (.getShort buf (int (+ base 4)))
          top-offset (.getShort buf (int (+ base 6)))
          pixels (byte-array (* width height))
          ;; Track which pixels are opaque (from posts) vs transparent (gaps)
          opaque (boolean-array (* width height))]
      ;; Read column offsets
      (dotimes [col width]
        (let [col-offset (.getInt buf (int (+ base 8 (* col 4))))
              col-start (+ base col-offset)]
          ;; Read posts
          (loop [pos (int col-start)]
            (let [topdelta (bit-and (.get buf pos) 0xFF)]
              (when (not= topdelta 0xFF)
                (let [length (bit-and (.get buf (int (+ pos 1))) 0xFF)]
                  ;; Skip padding byte, read pixels, skip trailing padding
                  (dotimes [row length]
                    (when (< (+ topdelta row) height)
                      (let [idx (+ (* (+ topdelta row) width) col)]
                        (aset pixels idx (.get buf (int (+ pos 3 row))))
                        (aset opaque idx true))))
                  (recur (int (+ pos 4 length)))))))))
      {:width width :height height :pixels pixels :opaque opaque
       :left-offset left-offset :top-offset top-offset})))

(defn compose-texture
  "Compose a wall texture from its patches using the palette.
  Returns RGBA byte[] (4 bytes per pixel)."
  [wad tex-def pnames ^bytes palette]
  (let [tw (int (:width tex-def))
        th (int (:height tex-def))
        rgba (byte-array (* tw th 4))
        ;; Track which pixels have been written
        written (boolean-array (* tw th))]
    (doseq [patch-info (:patches tex-def)]
      (let [patch-name (nth pnames (:patch-idx patch-info))
            patch-data (decode-picture wad patch-name)]
        (when patch-data
          (let [ox (int (:originx patch-info))
                oy (int (:originy patch-info))
                pw (int (:width patch-data))
                ph (int (:height patch-data))
                ^bytes ppix (:pixels patch-data)
                ^booleans popaque (:opaque patch-data)]
            (dotimes [py ph]
              (dotimes [px pw]
                (let [tx (+ ox px)
                      ty (+ oy py)
                      src-idx (+ (* py pw) px)]
                  (when (and (>= tx 0) (< tx tw) (>= ty 0) (< ty th)
                             (aget popaque src-idx)) ;; only copy opaque post pixels
                    (let [pidx (bit-and (aget ppix src-idx) 0xFF)
                          dst (* (+ (* ty tw) tx) 4)
                          r (bit-and (aget palette (* pidx 3)) 0xFF)
                          g (bit-and (aget palette (+ (* pidx 3) 1)) 0xFF)
                          b (bit-and (aget palette (+ (* pidx 3) 2)) 0xFF)]
                      (aset rgba dst (unchecked-byte r))
                      (aset rgba (+ dst 1) (unchecked-byte g))
                      (aset rgba (+ dst 2) (unchecked-byte b))
                      (aset rgba (+ dst 3) (unchecked-byte -1)) ;; alpha=255
                      (aset written (+ (* ty tw) tx) true))))))))))
    rgba))

(defn decode-flat
  "Decode a flat texture (64x64 floor/ceiling). Returns RGBA byte[]."
  [wad ^String flat-name ^bytes palette]
  (let [;; Flats are between F_START and F_END (or FF_START/FF_END)
        lump (or (find-lump-between wad flat-name "F_START" "F_END")
                 (find-lump-between wad flat-name "FF_START" "FF_END")
                 ;; Freedoom uses F1_START/F1_END etc.
                 (find-lump-between wad flat-name "F1_START" "F1_END")
                 (find-lump-between wad flat-name "F2_START" "F2_END"))]
    (when (and lump (= (:size lump) 4096))
      (let [^ByteBuffer buf (:buf wad)
            offset (int (:offset lump))
            rgba (byte-array (* 64 64 4))]
        (dotimes [i 4096]
          (let [pidx (bit-and (.get buf (int (+ offset i))) 0xFF)
                dst (* i 4)
                r (bit-and (aget palette (* pidx 3)) 0xFF)
                g (bit-and (aget palette (+ (* pidx 3) 1)) 0xFF)
                b (bit-and (aget palette (+ (* pidx 3) 2)) 0xFF)]
            (aset rgba dst (unchecked-byte r))
            (aset rgba (+ dst 1) (unchecked-byte g))
            (aset rgba (+ dst 2) (unchecked-byte b))
            (aset rgba (+ dst 3) (unchecked-byte -1))))
        rgba))))

;; ================================================================
;; Sprite parsing
;; ================================================================

(defn find-lumps-between
  "Find all lumps between start/end markers. Returns vector of lumps."
  [wad ^String start-marker ^String end-marker]
  (let [dir (:directory wad)
        start-idx (some (fn [[i entry]] (when (= (:name entry) start-marker) i))
                    (map-indexed vector dir))
        end-idx (some (fn [[i entry]] (when (= (:name entry) end-marker) i))
                  (map-indexed vector dir))]
    (when (and start-idx end-idx (< start-idx end-idx))
      (subvec dir (inc (int start-idx)) (int end-idx)))))

(defn decode-picture-at
  "Decode a Doom picture-format lump at a given offset/size to RGBA.
  Returns {:width :height :pixels byte[] :left-offset :top-offset} or nil."
  [^ByteBuffer buf ^long base ^long lump-size ^bytes palette]
  (when (> lump-size 8)
    (let [width (bit-and (.getShort buf (int base)) 0xFFFF)
          height (bit-and (.getShort buf (int (+ base 2))) 0xFFFF)]
      ;; Basic validation
      (when (and (> width 0) (<= width 320) (> height 0) (<= height 200)
                 (>= lump-size (+ 8 (* width 4))))
        (let [left-offset (.getShort buf (int (+ base 4)))
              top-offset (.getShort buf (int (+ base 6)))
              rgba (byte-array (* width height 4))]
          ;; Fill with transparent
          (java.util.Arrays/fill rgba (byte 0))
          ;; Read column offsets and posts
          (try
            (dotimes [col width]
              (let [col-offset (.getInt buf (int (+ base 8 (* col 4))))
                    col-start (+ base col-offset)]
                (loop [pos (int col-start)]
                  (let [topdelta (bit-and (.get buf pos) 0xFF)]
                    (when (not= topdelta 0xFF)
                      (let [length (bit-and (.get buf (int (+ pos 1))) 0xFF)]
                        (dotimes [row length]
                          (when (< (+ topdelta row) height)
                            (let [pidx (bit-and (.get buf (int (+ pos 3 row))) 0xFF)
                                  dst (* (+ (* (+ topdelta row) width) col) 4)
                                  r (bit-and (aget palette (* pidx 3)) 0xFF)
                                  g (bit-and (aget palette (+ (* pidx 3) 1)) 0xFF)
                                  b (bit-and (aget palette (+ (* pidx 3) 2)) 0xFF)]
                              (aset rgba dst (unchecked-byte r))
                              (aset rgba (+ dst 1) (unchecked-byte g))
                              (aset rgba (+ dst 2) (unchecked-byte b))
                              (aset rgba (+ dst 3) (unchecked-byte -1)))))
                        (recur (int (+ pos 4 length)))))))))
            (catch Exception _e nil))
          {:width width :height height :pixels rgba
           :left-offset left-offset :top-offset top-offset})))))

(defn parse-sprites
  "Parse sprite lumps from WAD (between S_START/S_END or SS_START/SS_END).
  Returns map: {\"TROO\" {\\A {0 {:lump lump :name \"TROOA0\"} 1 {...}} ...} ...}
  where rotation 0 means all angles."
  [wad]
  (let [sprite-lumps (or (find-lumps-between wad "S_START" "S_END")
                         (find-lumps-between wad "SS_START" "SS_END")
                         [])]
    (reduce
      (fn [acc lump]
        (let [^String name (:name lump)]
          (if (>= (.length name) 6)
            (let [prefix (.substring name 0 4)
                  frame (char (.charAt name 4))
                  rotation (- (int (.charAt name 5)) (int \0))]
              (-> acc
                  (assoc-in [prefix frame rotation]
                            {:lump lump :name name})
                  ;; Handle mirrored frames (e.g. TROOA2A8 = frame A rot 2 + mirrored rot 8)
                  (cond->
                    (>= (.length name) 8)
                    (assoc-in [prefix (char (.charAt name 6))
                               (- (int (.charAt name 7)) (int \0))]
                              {:lump lump :name name :mirror? true}))))
            acc)))
      {}
      sprite-lumps)))

(defn decode-sprite
  "Decode a sprite lump to RGBA using the palette.
  Returns {:width :height :pixels byte[] :left-offset :top-offset} or nil."
  [wad lump ^bytes palette]
  (let [^ByteBuffer buf (:buf wad)]
    (decode-picture-at buf (:offset lump) (:size lump) palette)))

;; ================================================================
;; Sound parsing
;; ================================================================

(defn parse-sound-lump
  "Parse a Doom sound effect lump (DS* format).
  Doom sounds: 8-byte header (format 3, sample_rate, num_samples, pad), then unsigned 8-bit PCM.
  Returns {:sample-rate int :samples byte[] :num-samples int} or nil."
  [wad ^String name]
  (when-let [lump (find-lump wad name)]
    (let [^ByteBuffer buf (:buf wad)
          base (int (:offset lump))
          format (bit-and (.getShort buf base) 0xFFFF)
          sample-rate (bit-and (.getShort buf (int (+ base 2))) 0xFFFF)
          num-samples (int (.getInt buf (int (+ base 4))))]
      (when (= format 3)
        (let [;; Skip padding bytes at start and end (16 bytes before, 16 after in some WADs)
              data-start (+ base 8)
              samples (byte-array num-samples)]
          (dotimes [i num-samples]
            (aset samples i (.get buf (int (+ data-start i)))))
          {:sample-rate sample-rate
           :samples samples
           :num-samples num-samples})))))

(defn parse-mus-lump
  "Parse MUS music lump header. Returns {:data-offset :length :channels :instruments}."
  [wad ^String name]
  (when-let [lump (find-lump wad name)]
    (let [^ByteBuffer buf (:buf wad)
          base (int (:offset lump))
          id0 (bit-and (.get buf base) 0xFF)
          id1 (bit-and (.get buf (int (+ base 1))) 0xFF)
          id2 (bit-and (.get buf (int (+ base 2))) 0xFF)
          id3 (bit-and (.get buf (int (+ base 3))) 0xFF)]
      (when (and (= id0 (int \M)) (= id1 (int \U)) (= id2 (int \S)) (= id3 0x1A))
        (let [score-len (bit-and (.getShort buf (int (+ base 4))) 0xFFFF)
              score-start (bit-and (.getShort buf (int (+ base 6))) 0xFFFF)
              channels (bit-and (.getShort buf (int (+ base 8))) 0xFFFF)
              sec-channels (bit-and (.getShort buf (int (+ base 10))) 0xFFFF)
              num-instruments (bit-and (.getShort buf (int (+ base 12))) 0xFFFF)]
          {:lump lump
           :score-len score-len
           :score-start (+ base score-start)
           :channels channels
           :sec-channels sec-channels
           :num-instruments num-instruments})))))

;; ================================================================
;; Utility
;; ================================================================

(defn player-start
  "Find Player 1 start position from things. Returns {:x :y :angle} or nil."
  [map-data]
  (first (filter #(= (:type %) 1) (:things map-data))))

(defn list-maps
  "List all map names in a WAD (E1M1 format or MAP01 format)."
  [wad]
  (let [names (map :name (:directory wad))]
    (filter (fn [n]
              (or (re-matches #"E\d+M\d+" n)
                  (re-matches #"MAP\d+" n)))
      names)))

(defn compose-sky-texture
  "Compose SKY1 (or named sky) texture to RGBA. Returns {:pixels byte[] :width :height}."
  [wad ^bytes palette]
  (let [pnames (parse-pnames wad)
        tex-defs (concat (or (parse-texture-defs wad "TEXTURE1") [])
                         (or (parse-texture-defs wad "TEXTURE2") []))
        sky-def (first (filter #(= (:name %) "SKY1") tex-defs))]
    (when sky-def
      (let [rgba (compose-texture wad sky-def pnames palette)]
        {:pixels rgba :width (:width sky-def) :height (:height sky-def)}))))

;; ================================================================
;; Thing type definitions
;; ================================================================

(def thing-types
  "Map of thing type number to {:name :sprite :category :radius :height}."
  {;; Player starts
   1 {:name "Player 1" :category :player}
   2 {:name "Player 2" :category :player}
   3 {:name "Player 3" :category :player}
   4 {:name "Player 4" :category :player}
   11 {:name "Deathmatch" :category :deathmatch}
   ;; Monsters
   3004 {:name "Zombieman" :sprite "POSS" :category :monster :radius 20 :height 56
         :health 20 :speed 8 :pain-chance 200 :melee-dmg 0 :ranged-dmg [3 15]}
   9 {:name "Shotgun Guy" :sprite "SPOS" :category :monster :radius 20 :height 56
      :health 30 :speed 8 :pain-chance 170 :melee-dmg 0 :ranged-dmg [3 15]}
   3001 {:name "Imp" :sprite "TROO" :category :monster :radius 20 :height 56
         :health 60 :speed 8 :pain-chance 200 :melee-dmg [3 24] :ranged-dmg [3 24]
         :projectile "BAL1"}
   ;; Decorations
   2035 {:name "Barrel" :sprite "BAR1" :category :decoration :radius 10 :height 42 :blocking? true}
   30 {:name "Tall green pillar" :sprite "COL1" :category :decoration :radius 16 :height 52 :blocking? true}
   31 {:name "Short green pillar" :sprite "COL2" :category :decoration :radius 16 :height 40 :blocking? true}
   32 {:name "Tall red pillar" :sprite "COL3" :category :decoration :radius 16 :height 52 :blocking? true}
   33 {:name "Short red pillar" :sprite "COL4" :category :decoration :radius 16 :height 40 :blocking? true}
   35 {:name "Candelabra" :sprite "CBRA" :category :decoration :radius 16 :height 52 :blocking? true}
   2028 {:name "Floor lamp" :sprite "COLU" :category :decoration :radius 16 :height 48 :blocking? true}
   85 {:name "Tech lamp" :sprite "TLMP" :category :decoration :radius 16 :height 80 :blocking? true}
   86 {:name "Tall tech lamp" :sprite "TLP2" :category :decoration :radius 16 :height 60 :blocking? true}
   34 {:name "Candle" :sprite "CAND" :category :decoration :radius 16 :height 16 :blocking? false}
   44 {:name "Tall blue torch" :sprite "TBLU" :category :decoration :radius 16 :height 96 :blocking? true}
   45 {:name "Tall green torch" :sprite "TGRN" :category :decoration :radius 16 :height 96 :blocking? true}
   46 {:name "Tall red torch" :sprite "TRED" :category :decoration :radius 16 :height 96 :blocking? true}
   55 {:name "Short blue torch" :sprite "SMBT" :category :decoration :radius 16 :height 72 :blocking? true}
   56 {:name "Short green torch" :sprite "SMGT" :category :decoration :radius 16 :height 72 :blocking? true}
   57 {:name "Short red torch" :sprite "SMRT" :category :decoration :radius 16 :height 72 :blocking? true}
   48 {:name "Tech column" :sprite "ELEC" :category :decoration :radius 16 :height 128 :blocking? true}
   10 {:name "Bloody mess" :sprite "PLAY" :category :decoration :radius 16 :height 16 :blocking? false}
   12 {:name "Bloody mess 2" :sprite "PLAY" :category :decoration :radius 16 :height 16 :blocking? false}
   15 {:name "Dead player" :sprite "PLAY" :category :decoration :radius 16 :height 16 :blocking? false}
   24 {:name "Pool of blood" :sprite "POL5" :category :decoration :radius 16 :height 16 :blocking? false}
   ;; Items
   2014 {:name "Health bonus" :sprite "BON1" :category :item :pickup :health-bonus}
   2011 {:name "Stimpack" :sprite "STIM" :category :item :pickup :stimpack}
   2012 {:name "Medikit" :sprite "MEDI" :category :item :pickup :medikit}
   2015 {:name "Armor bonus" :sprite "BON2" :category :item :pickup :armor-bonus}
   2018 {:name "Green armor" :sprite "ARM1" :category :item :pickup :green-armor}
   2019 {:name "Blue armor" :sprite "ARM2" :category :item :pickup :blue-armor}
   8 {:name "Backpack" :sprite "BPAK" :category :item :pickup :backpack}
   ;; Weapons
   2001 {:name "Shotgun" :sprite "SHOT" :category :weapon :pickup :shotgun}
   2002 {:name "Chaingun" :sprite "MGUN" :category :weapon :pickup :chaingun}
   2003 {:name "Rocket launcher" :sprite "LAUN" :category :weapon :pickup :rocket-launcher}
   2006 {:name "BFG9000" :sprite "BFUG" :category :weapon :pickup :bfg}
   2005 {:name "Chainsaw" :sprite "CSAW" :category :weapon :pickup :chainsaw}
   ;; Ammo
   2007 {:name "Clip" :sprite "CLIP" :category :ammo :pickup :clip}
   2048 {:name "Box of ammo" :sprite "AMMO" :category :ammo :pickup :ammo-box}
   2008 {:name "Shells" :sprite "SHEL" :category :ammo :pickup :shells}
   2049 {:name "Box of shells" :sprite "SBOX" :category :ammo :pickup :shell-box}
   2010 {:name "Rocket" :sprite "ROCK" :category :ammo :pickup :rocket}
   2046 {:name "Box of rockets" :sprite "BROK" :category :ammo :pickup :rocket-box}
   2047 {:name "Cell charge" :sprite "CELL" :category :ammo :pickup :cell}
   17 {:name "Cell pack" :sprite "CELP" :category :ammo :pickup :cell-pack}
   ;; Keys
   5 {:name "Blue keycard" :sprite "BKEY" :category :key :pickup :blue-key}
   13 {:name "Red keycard" :sprite "RKEY" :category :key :pickup :red-key}
   6 {:name "Yellow keycard" :sprite "YKEY" :category :key :pickup :yellow-key}
   40 {:name "Blue skull" :sprite "BSKU" :category :key :pickup :blue-skull}
   38 {:name "Red skull" :sprite "RSKU" :category :key :pickup :red-skull}
   39 {:name "Yellow skull" :sprite "YSKU" :category :key :pickup :yellow-skull}})
