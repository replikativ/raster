(ns doom.level
  "Geometry builder for Doom maps. Converts WAD map data into vertex/index arrays
  suitable for GPU rendering. Supports walls, floors, and ceilings.
  Coordinate transforms and geometry kernels use deftm."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn deftm]])
  (:import [earcut4j Earcut]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Coordinate mapping: Doom → Vulkan
;; Doom X → VK X, Doom Y → VK -Z, Doom height → VK Y
;; Scale: 1 Doom unit = 1/32 VK units
;; ================================================================

(def ^:const SCALE (/ 1.0 32.0))

(deftm doom->vk-x [x :- Double] :- Double (* x SCALE))
(deftm doom->vk-x [x :- Long]   :- Double (* (double x) SCALE))
(deftm doom->vk-y [h :- Double] :- Double (* h SCALE))
(deftm doom->vk-y [h :- Long]   :- Double (* (double h) SCALE))
(deftm doom->vk-z [y :- Double] :- Double (* (- y) SCALE))
(deftm doom->vk-z [y :- Long]   :- Double (* (- (double y)) SCALE))

;; ================================================================
;; Vertex format: [x y z u v light texLayer scaleU scaleV] = 9 floats = 36 bytes
;; For texture arrays: texLayer is the array layer index,
;; scaleU/scaleV = actualTexSize / paddedTexSize for UV correction.
;; ================================================================

(def ^:const FLOATS-PER-VERTEX 9)
(def ^:const VERTEX-STRIDE (* FLOATS-PER-VERTEX 4))

;; ================================================================
;; Wall geometry
;; ================================================================

(deftm wall-length
  "Compute 2D wall length in Doom units."
  [x1 :- Long, y1 :- Long, x2 :- Long, y2 :- Long] :- Double
  (let [dx (double (- x2 x1))
        dy (double (- y2 y1))]
    (Math/sqrt (+ (* dx dx) (* dy dy)))))

(defn build-wall-geometry
  "Build wall quads from linedefs/sidedefs/sectors/vertices.
  Returns {:vertices float[] :indices int[] :wall-count int}.
  tex-array-map: {name → {:layer int :scale-u float :scale-v float}} from TextureArray.
  sector-overrides: {sector-idx {:floor-height :ceil-height}} for doors/lifts."
  [map-data & {:keys [tex-array-map sector-overrides]}]
  (let [vertexes (:vertexes map-data)
        linedefs (:linedefs map-data)
        sidedefs (:sidedefs map-data)
        sectors (:sectors map-data)
        ;; Pre-allocate generous arrays (4 verts per quad, up to 3 quads per linedef)
        max-quads (* (count linedefs) 3)
        verts (float-array (* max-quads 4 FLOATS-PER-VERTEX))
        indices (int-array (* max-quads 6))
        vi (volatile! 0) ;; vertex float offset
        ii (volatile! 0) ;; index offset
        vc (volatile! 0) ;; vertex count
        wall-count (volatile! 0)]
    (doseq [linedef linedefs]
      (let [v1-idx (:v1 linedef)
            v2-idx (:v2 linedef)
            [x1 y1] (nth vertexes v1-idx)
            [x2 y2] (nth vertexes v2-idx)
            x1 (long x1) y1 (long y1)
            x2 (long x2) y2 (long y2)
            wlen (wall-length x1 y1 x2 y2)
            ;; Vulkan coordinates
            vx1 (doom->vk-x x1) vz1 (doom->vk-z y1)
            vx2 (doom->vk-x x2) vz2 (doom->vk-z y2)]
        ;; Helper: effective sector heights with overrides
        (letfn [(eff-floor [sector si]
                  (if-let [o (when sector-overrides (get sector-overrides si))]
                    (double (or (:floor-height o) (:floor-height sector)))
                    (double (:floor-height sector))))
                (eff-ceil [sector si]
                  (if-let [o (when sector-overrides (get sector-overrides si))]
                    (double (or (:ceil-height o) (:ceil-height sector)))
                    (double (:ceil-height sector))))
                ;; Helper to emit a wall quad
                (emit-quad! [bottom top light
                             tex-name x-offset y-offset
                             default-tw default-th upper-peg?]
                  (let [bottom (double bottom) top (double top) light (double light)
                        x-offset (double x-offset) y-offset (double y-offset)
                        ;; Look up texture in array map for dimensions + layer
                        tex-info (when (and tex-array-map tex-name)
                                   (get tex-array-map tex-name))
                        tex-layer (float (if tex-info (double (:layer tex-info)) 0.0))
                        su (float (if tex-info (double (:scale-u tex-info)) 1.0))
                        sv (float (if tex-info (double (:scale-v tex-info)) 1.0))
                        ;; Get actual texture dimensions from array map
                        tex-w (double (if tex-info (:width tex-info) default-tw))
                        tex-h (double (if tex-info (:height tex-info) default-th))
                        h (- top bottom)
                        ;; Raw UV in texture-normalized coords (0-1 = one repeat)
                        u-scale (/ 1.0 tex-w)
                        v-scale (/ 1.0 tex-h)
                        u0 (* x-offset u-scale)
                        u1 (* (+ x-offset wlen) u-scale)
                        v0 (if upper-peg?
                             (* y-offset v-scale)
                             (* (+ y-offset (- tex-h h)) v-scale))
                        v1 (+ v0 (* h v-scale))
                        vy-bot (doom->vk-y bottom)
                        vy-top (doom->vk-y top)
                        base-v (long @vc)
                        vo (long @vi)
                        io (long @ii)]
                    (when (> h 0.0)
                      ;; 4 vertices: BL BR TR TL (9 floats each)
                      ;; BL
                      (aset verts vo (float vx1))
                      (aset verts (+ vo 1) (float vy-bot))
                      (aset verts (+ vo 2) (float vz1))
                      (aset verts (+ vo 3) (float u0))
                      (aset verts (+ vo 4) (float v1))
                      (aset verts (+ vo 5) (float light))
                      (aset verts (+ vo 6) tex-layer)
                      (aset verts (+ vo 7) su)
                      (aset verts (+ vo 8) sv)
                      ;; BR
                      (aset verts (+ vo 9) (float vx2))
                      (aset verts (+ vo 10) (float vy-bot))
                      (aset verts (+ vo 11) (float vz2))
                      (aset verts (+ vo 12) (float u1))
                      (aset verts (+ vo 13) (float v1))
                      (aset verts (+ vo 14) (float light))
                      (aset verts (+ vo 15) tex-layer)
                      (aset verts (+ vo 16) su)
                      (aset verts (+ vo 17) sv)
                      ;; TR
                      (aset verts (+ vo 18) (float vx2))
                      (aset verts (+ vo 19) (float vy-top))
                      (aset verts (+ vo 20) (float vz2))
                      (aset verts (+ vo 21) (float u1))
                      (aset verts (+ vo 22) (float v0))
                      (aset verts (+ vo 23) (float light))
                      (aset verts (+ vo 24) tex-layer)
                      (aset verts (+ vo 25) su)
                      (aset verts (+ vo 26) sv)
                      ;; TL
                      (aset verts (+ vo 27) (float vx1))
                      (aset verts (+ vo 28) (float vy-top))
                      (aset verts (+ vo 29) (float vz1))
                      (aset verts (+ vo 30) (float u0))
                      (aset verts (+ vo 31) (float v0))
                      (aset verts (+ vo 32) (float light))
                      (aset verts (+ vo 33) tex-layer)
                      (aset verts (+ vo 34) su)
                      (aset verts (+ vo 35) sv)
                      ;; 6 indices: 2 triangles
                      (aset indices io (int base-v))
                      (aset indices (+ io 1) (int (+ base-v 1)))
                      (aset indices (+ io 2) (int (+ base-v 2)))
                      (aset indices (+ io 3) (int base-v))
                      (aset indices (+ io 4) (int (+ base-v 2)))
                      (aset indices (+ io 5) (int (+ base-v 3)))
                      (vreset! vi (+ vo 36))
                      (vreset! ii (+ io 6))
                      (vreset! vc (+ base-v 4))
                      (vswap! wall-count inc))))]
          ;; Front side
          (when-let [fs-idx (:front-sidedef linedef)]
            (let [front-side (nth sidedefs fs-idx)
                  front-si (:sector front-side)
                  front-sector (nth sectors front-si)
                  light (/ (double (:light-level front-sector)) 255.0)]
              (if-let [bs-idx (:back-sidedef linedef)]
                ;; Two-sided linedef
                (let [back-side (nth sidedefs bs-idx)
                      back-si (:sector back-side)
                      back-sector (nth sectors back-si)
                      ff (eff-floor front-sector front-si)
                      fc (eff-ceil front-sector front-si)
                      bf (eff-floor back-sector back-si)
                      bc (eff-ceil back-sector back-si)]
                  ;; Upper wall (front ceil → back ceil)
                  (when (> fc bc)
                    (let [tex (:upper front-side)
                          upper-peg? (zero? (bit-and (:flags linedef) 8))]
                      (emit-quad! bc fc light tex
                                  (double (:x-offset front-side))
                                  (double (:y-offset front-side))
                                  128.0 128.0 upper-peg?)))
                  ;; Lower wall (back floor → front floor)
                  (when (> bf ff)
                    (let [tex (:lower front-side)
                          lower-peg? (not (zero? (bit-and (:flags linedef) 16)))]
                      (emit-quad! ff bf light tex
                                  (double (:x-offset front-side))
                                  (double (:y-offset front-side))
                                  128.0 128.0 lower-peg?)))
                  ;; Middle (transparent door, etc.)
                  (when (and (:middle front-side)
                             (not= (:middle front-side) "-"))
                    (let [mid-bottom (max ff bf)
                          mid-top (min fc bc)]
                      (when (> mid-top mid-bottom)
                        (emit-quad! mid-bottom mid-top light
                                    (:middle front-side)
                                    (double (:x-offset front-side))
                                    (double (:y-offset front-side))
                                    128.0 128.0 true))))
                  ;; === Back side walls (viewed from back sector) ===
                  (let [back-light (/ (double (:light-level back-sector)) 255.0)]
                    ;; Back upper wall (back ceil → front ceil, where bc > fc)
                    (when (> bc fc)
                      (let [tex (:upper back-side)
                            upper-peg? (zero? (bit-and (:flags linedef) 8))]
                        (emit-quad! fc bc back-light tex
                                    (double (:x-offset back-side))
                                    (double (:y-offset back-side))
                                    128.0 128.0 upper-peg?)))
                    ;; Back lower wall (front floor → back floor, where ff > bf)
                    (when (> ff bf)
                      (let [tex (:lower back-side)
                            lower-peg? (not (zero? (bit-and (:flags linedef) 16)))]
                        (emit-quad! bf ff back-light tex
                                    (double (:x-offset back-side))
                                    (double (:y-offset back-side))
                                    128.0 128.0 lower-peg?)))
                    ;; Back middle
                    (when (and (:middle back-side)
                               (not= (:middle back-side) "-"))
                      (let [mid-bottom (max ff bf)
                            mid-top (min fc bc)]
                        (when (> mid-top mid-bottom)
                          (emit-quad! mid-bottom mid-top back-light
                                      (:middle back-side)
                                      (double (:x-offset back-side))
                                      (double (:y-offset back-side))
                                      128.0 128.0 true))))))
                ;; One-sided linedef
                (let [ff (eff-floor front-sector front-si)
                      fc (eff-ceil front-sector front-si)
                      tex (:middle front-side)]
                  (emit-quad! ff fc light tex
                              (double (:x-offset front-side))
                              (double (:y-offset front-side))
                              128.0 128.0 true))))))))
    ;; Trim arrays to actual size
    (let [n-vfloats (long @vi)
          n-indices (long @ii)]
      {:vertices (java.util.Arrays/copyOf verts (int n-vfloats))
       :indices (java.util.Arrays/copyOf indices (int n-indices))
       :wall-count @wall-count})))

;; ================================================================
;; Floor/ceiling geometry via sector boundary tracing + earcut
;; ================================================================

(defn- collect-sector-lines
  "Collect all linedef edges belonging to a sector, oriented so the sector
  is on the left side (front). Returns vector of [v1-idx v2-idx]."
  [map-data ^long sector-idx]
  (let [linedefs (:linedefs map-data)
        sidedefs (:sidedefs map-data)]
    (reduce
      (fn [edges linedef]
        (let [front-sd (:front-sidedef linedef)
              back-sd (:back-sidedef linedef)
              front-sector (when front-sd (:sector (nth sidedefs front-sd)))
              back-sector (when back-sd (:sector (nth sidedefs back-sd)))]
          (cond-> edges
            (= front-sector sector-idx)
            (conj [(:v1 linedef) (:v2 linedef)])
            (= back-sector sector-idx)
            (conj [(:v2 linedef) (:v1 linedef)]))))
      []
      linedefs)))

(defn- trace-loops
  "Trace directed edges into closed loops. Returns vector of loops,
  where each loop is a vector of vertex indices in order."
  [edges]
  (if (empty? edges)
    []
    (let [;; Build adjacency: v1 → [v2 ...]
          adj (reduce (fn [m [v1 v2]]
                        (update m v1 (fnil conj []) v2))
                {} edges)
          adj (atom adj)
          loops (atom [])]
      ;; Trace loops by following edges
      (while (seq @adj)
        (let [start (ffirst @adj)
              loop-verts (atom [start])
              current (atom start)]
          (loop []
            (let [nexts (get @adj @current)]
              (when (seq nexts)
                (let [nxt (first nexts)]
                  ;; Remove used edge
                  (swap! adj update @current rest)
                  (when (empty? (get @adj @current))
                    (swap! adj dissoc @current))
                  (when (not= nxt start)
                    (swap! loop-verts conj nxt)
                    (reset! current nxt)
                    (recur))))))
          (when (> (count @loop-verts) 2)
            (swap! loops conj @loop-verts))))
      @loops)))

(defn- signed-area-2d
  "Compute signed area of a 2D polygon. Positive = CCW."
  ^double [loop-verts vertexes]
  (let [n (count loop-verts)]
    (* 0.5
       (reduce
         (fn [^double sum ^long i]
           (let [vi (nth loop-verts i)
                 vj (nth loop-verts (mod (inc i) n))
                 [x1 y1] (nth vertexes vi)
                 [x2 y2] (nth vertexes vj)]
             (+ sum (- (* (double x1) (double y2))
                       (* (double x2) (double y1))))))
         0.0
         (range n)))))

(defn build-floor-ceil-geometry
  "Build floor and ceiling triangles for all sectors using earcut triangulation.
  Returns {:vertices float[] :indices int[]}.
  tex-array-map: {name → {:layer int :scale-u float :scale-v float :width int :height int}}
  sector-overrides: {sector-idx {:floor-height :ceil-height}} for doors/lifts.
  flat-remap: {name → name} for animated texture cycling."
  [map-data & {:keys [tex-array-map sector-overrides flat-remap]}]
  (let [vertexes (:vertexes map-data)
        sectors (:sectors map-data)
        all-verts (java.util.ArrayList.)
        all-indices (java.util.ArrayList.)
        vert-offset (volatile! 0)]
    (doseq [[sector-idx sector] (map-indexed vector sectors)]
      (let [edges (collect-sector-lines map-data sector-idx)
            loops (trace-loops edges)]
        (when (seq loops)
          ;; Sort loops by area: largest (outer) first, holes after
          (let [sorted-loops (sort-by
                               (fn [loop-v]
                                 (- (Math/abs (signed-area-2d loop-v vertexes))))
                               loops)
                outer-loop (first sorted-loops)
                hole-loops (rest sorted-loops)
                ;; Build flat coord array for earcut
                ;; Outer ring first, then holes
                all-loop-verts (into (vec outer-loop) (mapcat identity hole-loops))
                coords (double-array (* 2 (count all-loop-verts)))
                hole-indices (when (seq hole-loops)
                               (int-array (count hole-loops)))
                ;; Map local earcut vertex index → WAD vertex index
                vert-map (int-array (count all-loop-verts))]
            ;; Fill coords and vert-map
            (let [idx (volatile! 0)]
              ;; Outer loop
              (doseq [vi outer-loop]
                (let [i (long @idx)
                      [x y] (nth vertexes vi)]
                  (aset coords (* i 2) (double x))
                  (aset coords (+ (* i 2) 1) (double y))
                  (aset vert-map i (int vi))
                  (vreset! idx (inc i))))
              ;; Hole loops
              (dotimes [hi (count hole-loops)]
                (when hole-indices
                  (aset hole-indices hi (int @idx)))
                (doseq [vi (nth hole-loops hi)]
                  (let [i (long @idx)
                        [x y] (nth vertexes vi)]
                    (aset coords (* i 2) (double x))
                    (aset coords (+ (* i 2) 1) (double y))
                    (aset vert-map i (int vi))
                    (vreset! idx (inc i))))))
            ;; Triangulate
            (let [tri-indices (Earcut/earcut coords hole-indices 2)
                  light (/ (double (:light-level sector)) 255.0)
                  override (when sector-overrides (get sector-overrides sector-idx))
                  floor-h (if (and override (:floor-height override))
                            (double (:floor-height override))
                            (double (:floor-height sector)))
                  ceil-h (if (and override (:ceil-height override))
                           (double (:ceil-height override))
                           (double (:ceil-height sector)))
                  floor-tex (let [ft (:floor-tex sector)]
                              (if flat-remap (get flat-remap ft ft) ft))
                  ceil-tex (let [ct (:ceil-tex sector)]
                             (if flat-remap (get flat-remap ct ct) ct))
                  base (long @vert-offset)]
              (when (and tri-indices (pos? (.size tri-indices)))
                ;; Emit vertices for each unique vertex used in this sector
                ;; Use a local vertex map: wad-vert-idx → local-idx
                (let [local-map (java.util.HashMap.)
                      local-count (volatile! 0)]
                  ;; Process all triangulated indices
                  (let [emit-vert! (fn [^long wad-vi ^double height tex-name _flip-winding?]
                                     (let [key (long (+ (* wad-vi 1000000) (long (* height 100))))
                                           existing (.get local-map key)]
                                       (if existing
                                         (int existing)
                                         (let [li (long @local-count)
                                               [x y] (nth vertexes wad-vi)
                                               vx (doom->vk-x (double x))
                                               vy (doom->vk-y height)
                                               vz (doom->vk-z (double y))
                                               ;; Flat UV: tile every 64 Doom units (raw, shader does fract)
                                               u (/ (double x) 64.0)
                                               v (/ (double y) 64.0)
                                               ;; Texture array layer + scale
                                               tex-info (when (and tex-array-map tex-name)
                                                          (get tex-array-map tex-name))
                                               tex-layer (float (if tex-info (double (:layer tex-info)) 0.0))
                                               su (float (if tex-info (double (:scale-u tex-info)) 1.0))
                                               sv (float (if tex-info (double (:scale-v tex-info)) 1.0))]
                                           (.add all-verts (float vx))
                                           (.add all-verts (float vy))
                                           (.add all-verts (float vz))
                                           (.add all-verts (float u))
                                           (.add all-verts (float v))
                                           (.add all-verts (float light))
                                           (.add all-verts tex-layer)
                                           (.add all-verts su)
                                           (.add all-verts sv)
                                           (.put local-map key (int li))
                                           (vreset! local-count (inc li))
                                           (int li)))))]
                    ;; Floor triangles (CCW when viewed from above → flip for VK Y-up)
                    (let [n-tris (/ (.size tri-indices) 3)]
                      (dotimes [t n-tris]
                        (let [i0 (.get tri-indices (* t 3))
                              i1 (.get tri-indices (+ (* t 3) 1))
                              i2 (.get tri-indices (+ (* t 3) 2))
                              w0 (aget vert-map i0)
                              w1 (aget vert-map i1)
                              w2 (aget vert-map i2)
                              ;; Floor - emit with reversed winding
                              fi0 (emit-vert! w0 floor-h floor-tex false)
                              fi1 (emit-vert! w1 floor-h floor-tex false)
                              fi2 (emit-vert! w2 floor-h floor-tex false)]
                          (.add all-indices (int (+ base fi0)))
                          (.add all-indices (int (+ base fi2)))
                          (.add all-indices (int (+ base fi1))))))
                    ;; Ceiling triangles (normal winding)
                    (when (not= ceil-tex "F_SKY1")
                      (let [n-tris (/ (.size tri-indices) 3)]
                        (dotimes [t n-tris]
                          (let [i0 (.get tri-indices (* t 3))
                                i1 (.get tri-indices (+ (* t 3) 1))
                                i2 (.get tri-indices (+ (* t 3) 2))
                                w0 (aget vert-map i0)
                                w1 (aget vert-map i1)
                                w2 (aget vert-map i2)
                                ci0 (emit-vert! w0 ceil-h ceil-tex false)
                                ci1 (emit-vert! w1 ceil-h ceil-tex false)
                                ci2 (emit-vert! w2 ceil-h ceil-tex false)]
                            (.add all-indices (int (+ base ci0)))
                            (.add all-indices (int (+ base ci1)))
                            (.add all-indices (int (+ base ci2))))))))
                  (vreset! vert-offset (+ base (long @local-count))))))))))
    ;; Convert ArrayLists to arrays
    (let [nv (.size all-verts)
          ni (.size all-indices)
          va (float-array nv)
          ia (int-array ni)]
      (dotimes [i nv] (aset va i (float (.get all-verts i))))
      (dotimes [i ni] (aset ia i (int (.get all-indices i))))
      {:vertices va :indices ia})))

;; ================================================================
;; Combined level mesh
;; ================================================================

(defn build-level-geometry
  "Build complete level geometry (walls + floors/ceilings).
  Returns {:vertices float[] :indices int[] :wall-index-count int :total-index-count int}.
  tex-array-map: {name → {:layer :scale-u :scale-v :width :height}} from TextureArray.
  sector-overrides: {sector-idx {:floor-height :ceil-height}} for doors/lifts.
  flat-remap: {name → name} for animated texture cycling."
  [map-data & {:keys [tex-array-map sector-overrides flat-remap]}]
  (let [walls (build-wall-geometry map-data :tex-array-map tex-array-map
                                   :sector-overrides sector-overrides)
        floors (build-floor-ceil-geometry map-data :tex-array-map tex-array-map
                                          :sector-overrides sector-overrides
                                          :flat-remap flat-remap)
        ;; Merge vertex arrays
        ^floats wv (:vertices walls)
        ^floats fv (:vertices floors)
        total-vfloats (+ (alength wv) (alength fv))
        merged-verts (float-array total-vfloats)
        _ (System/arraycopy wv 0 merged-verts 0 (alength wv))
        _ (System/arraycopy fv 0 merged-verts (alength wv) (alength fv))
        ;; Merge index arrays (floor indices need vertex offset)
        ^ints wi (:indices walls)
        ^ints fi (:indices floors)
        wall-vert-count (/ (alength wv) FLOATS-PER-VERTEX)
        total-indices (+ (alength wi) (alength fi))
        merged-indices (int-array total-indices)]
    (System/arraycopy wi 0 merged-indices 0 (alength wi))
    ;; Offset floor indices by wall vertex count
    (dotimes [i (alength fi)]
      (aset merged-indices (+ (alength wi) i)
            (int (+ (aget fi i) wall-vert-count))))
    {:vertices merged-verts
     :indices merged-indices
     :wall-index-count (alength wi)
     :total-index-count total-indices}))

;; ================================================================
;; Wireframe geometry (Phase 1 - simple line segments)
;; ================================================================

(defn build-wireframe-lines
  "Build line geometry for wireframe rendering. Each linedef becomes
  4 line segments (floor and ceiling edges + 2 verticals).
  Returns {:vertices float[] :indices int[]}."
  [map-data]
  (let [vertexes (:vertexes map-data)
        linedefs (:linedefs map-data)
        sidedefs (:sidedefs map-data)
        sectors (:sectors map-data)
        ;; Each wall: 4 verts, 8 indices (4 line segments × 2 verts each)
        max-walls (count linedefs)
        verts (float-array (* max-walls 4 FLOATS-PER-VERTEX))
        indices (int-array (* max-walls 8))
        vi (volatile! 0)
        ii (volatile! 0)
        vc (volatile! 0)]
    (doseq [linedef linedefs]
      (when-let [fs-idx (:front-sidedef linedef)]
        (let [[x1 y1] (nth vertexes (:v1 linedef))
              [x2 y2] (nth vertexes (:v2 linedef))
              front-side (nth sidedefs fs-idx)
              sector (nth sectors (:sector front-side))
              floor-h (double (:floor-height sector))
              ceil-h (double (:ceil-height sector))
              light (/ (double (:light-level sector)) 255.0)
              vx1 (doom->vk-x (double x1))
              vz1 (doom->vk-z (double y1))
              vx2 (doom->vk-x (double x2))
              vz2 (doom->vk-z (double y2))
              vy-f (doom->vk-y floor-h)
              vy-c (doom->vk-y ceil-h)
              base (long @vc)]
          ;; 4 verts: BL BR TR TL
          (doseq [[vx vy vz] [[vx1 vy-f vz1] [vx2 vy-f vz2]
                               [vx2 vy-c vz2] [vx1 vy-c vz1]]]
            (let [o (long @vi)]
              (aset verts o (float vx))
              (aset verts (+ o 1) (float vy))
              (aset verts (+ o 2) (float vz))
              (aset verts (+ o 3) (float 0.0))
              (aset verts (+ o 4) (float 0.0))
              (aset verts (+ o 5) (float light))
              (aset verts (+ o 6) (float 0.0))  ;; texLayer
              (aset verts (+ o 7) (float 1.0))  ;; scaleU
              (aset verts (+ o 8) (float 1.0))  ;; scaleV
              (vreset! vi (+ o FLOATS-PER-VERTEX))))
          ;; 4 line segments: bottom, right, top, left
          (let [io (long @ii)]
            ;; Bottom: BL-BR
            (aset indices io (int base))
            (aset indices (+ io 1) (int (+ base 1)))
            ;; Right: BR-TR
            (aset indices (+ io 2) (int (+ base 1)))
            (aset indices (+ io 3) (int (+ base 2)))
            ;; Top: TR-TL
            (aset indices (+ io 4) (int (+ base 2)))
            (aset indices (+ io 5) (int (+ base 3)))
            ;; Left: TL-BL
            (aset indices (+ io 6) (int (+ base 3)))
            (aset indices (+ io 7) (int base))
            (vreset! ii (+ io 8)))
          (vreset! vc (+ base 4)))))
    (let [nv (long @vi)
          ni (long @ii)]
      {:vertices (java.util.Arrays/copyOf verts (int nv))
       :indices (java.util.Arrays/copyOf indices (int ni))})))
