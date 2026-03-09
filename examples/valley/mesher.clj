(ns valley.mesher
  "Face-culling mesh generation for voxel chunks.
  Vertex format: [pos.x pos.y pos.z u v dayLight nightLight texLayer ao] = 9 floats/vertex.
  Includes per-vertex ambient occlusion and dual light channels.
  Uses clojure.core/defn — mesh assembly is rendering code, not simulation."
  (:require [valley.world :as w]
            [valley.lighting :as light]))

(set! *warn-on-reflection* true)

;; Vertex stride: 9 floats
;; pos(3) + uv(2) + dayLight(1) + nightLight(1) + texLayer(1) + ao(1)
(def ^:const VERTEX-STRIDE 9)

;; Face vertex offsets: 4 corners for each face (CCW winding when viewed from outside)
;; Each is [x y z u v] relative to block origin
(def face-vertices
  [;; TOP (+Y face at y+1)
   [[0 1 0  0 0] [0 1 1  0 1] [1 1 1  1 1] [1 1 0  1 0]]
   ;; BOTTOM (-Y face at y)
   [[0 0 1  0 0] [0 0 0  0 1] [1 0 0  1 1] [1 0 1  1 0]]
   ;; NORTH (+Z face at z+1)
   [[1 0 1  0 0] [1 1 1  0 1] [0 1 1  1 1] [0 0 1  1 0]]
   ;; SOUTH (-Z face at z)
   [[0 0 0  0 0] [0 1 0  0 1] [1 1 0  1 1] [1 0 0  1 0]]
   ;; EAST (+X face at x+1)
   [[1 0 0  0 0] [1 1 0  0 1] [1 1 1  1 1] [1 0 1  1 0]]
   ;; WEST (-X face at x)
   [[0 0 1  0 0] [0 1 1  0 1] [0 1 0  1 1] [0 0 0  1 0]]])

;; Neighbor offsets for each face
(def neighbor-offsets
  [[0 1 0] [0 -1 0] [0 0 1] [0 0 -1] [1 0 0] [-1 0 0]])

;; Per-face directional shading multiplier
;; TOP=1.0, BOTTOM=0.5, NORTH/SOUTH=0.8, EAST/WEST=0.7
(def face-shade
  (float-array [1.0 0.5 0.8 0.8 0.7 0.7]))

;; AO brightness multipliers for AO values 0-3
(def ao-multiplier
  (float-array [0.3 0.55 0.8 1.0]))

;; ================================================================
;; AO corner offsets per face per vertex
;; For each face direction, for each of the 4 vertices:
;;   [side1-dx side1-dy side1-dz  side2-dx side2-dy side2-dz  corner-dx corner-dy corner-dz]
;; The offsets are in the normal direction + edge directions for that face.
;; ================================================================

;; For each face, each vertex has 3 AO neighbor offsets relative to the neighbor block position.
;; We precompute these as flat arrays for speed.
;; Face normal + two tangent axes define the corner sampling directions.

(defn- compute-ao-offsets
  "Compute AO sampling offsets for all 6 faces, 4 vertices each.
  Returns a vector of 6 vectors of 4 vectors of [s1 s2 corner] offsets."
  []
  ;; For each face, the 4 vertex positions define corners.
  ;; The AO for a vertex samples 3 neighbors:
  ;;   side1: in tangent direction 1
  ;;   side2: in tangent direction 2
  ;;   corner: in both tangent directions
  ;; All relative to the face's neighbor position.
  ;;
  ;; Face normal and tangent axes:
  ;; TOP(+Y):    tangents = X, Z
  ;; BOTTOM(-Y): tangents = X, Z
  ;; NORTH(+Z):  tangents = X, Y
  ;; SOUTH(-Z):  tangents = X, Y
  ;; EAST(+X):   tangents = Z, Y
  ;; WEST(-X):   tangents = Z, Y

  ;; face-vertex corners in tangent space: (sign-t1, sign-t2) for each vertex
  ;; vertex 0: (-,-), vertex 1: (-,+), vertex 2: (+,+), vertex 3: (+,-)
  ;; But actual mapping depends on the face vertex order.
  ;; We derive from the face-vertices: the vertex offset along each tangent axis.

  (let [;; For each face: [normal-axis, tangent1-axis, tangent2-axis]
        ;; axis: 0=X, 1=Y, 2=Z, sign from face-normals
        face-tangents
        [[[ 0  0  0] ;; TOP: normal=+Y
          [ 1  0  0] ;; t1 = +X
          [ 0  0  1]] ;; t2 = +Z
         [[ 0  0  0] ;; BOTTOM: normal=-Y
          [ 1  0  0]
          [ 0  0  1]]
         [[ 0  0  0] ;; NORTH: normal=+Z
          [ 1  0  0]
          [ 0  1  0]]
         [[ 0  0  0] ;; SOUTH: normal=-Z
          [ 1  0  0]
          [ 0  1  0]]
         [[ 0  0  0] ;; EAST: normal=+X
          [ 0  0  1]
          [ 0  1  0]]
         [[ 0  0  0] ;; WEST: normal=-X
          [ 0  0  1]
          [ 0  1  0]]]]
    ;; For each face, for each vertex, compute the AO corner direction
    ;; based on which "corner" of the face the vertex is at.
    ;; We look at the vertex offset and determine its tangent-space position.
    (vec
      (for [face (range 6)]
        (let [fverts (nth face-vertices face)
              [_ t1 t2] (nth face-tangents face)
              ;; Find the center of the face vertices to determine corner signs
              cx (/ (reduce + (map #(nth % 0) fverts)) 4.0)
              cy (/ (reduce + (map #(nth % 1) fverts)) 4.0)
              cz (/ (reduce + (map #(nth % 2) fverts)) 4.0)]
          (vec
            (for [vi (range 4)]
              (let [[vx vy vz] (nth fverts vi)
                    ;; Sign along each tangent: is this vertex above or below center?
                    s1 (long (Math/signum (double (- (+ (* vx (nth t1 0))
                                                        (* vy (nth t1 1))
                                                        (* vz (nth t1 2)))
                                                     (+ (* cx (nth t1 0))
                                                        (* cy (nth t1 1))
                                                        (* cz (nth t1 2)))))))
                    s2 (long (Math/signum (double (- (+ (* vx (nth t2 0))
                                                        (* vy (nth t2 1))
                                                        (* vz (nth t2 2)))
                                                     (+ (* cx (nth t2 0))
                                                        (* cy (nth t2 1))
                                                        (* cz (nth t2 2)))))))]
                ;; side1 offset = s1 * t1, side2 offset = s2 * t2, corner = s1*t1 + s2*t2
                {:side1 [(* s1 (nth t1 0)) (* s1 (nth t1 1)) (* s1 (nth t1 2))]
                 :side2 [(* s2 (nth t2 0)) (* s2 (nth t2 1)) (* s2 (nth t2 2))]
                 :corner [(+ (* s1 (nth t1 0)) (* s2 (nth t2 0)))
                          (+ (* s1 (nth t1 1)) (* s2 (nth t2 1)))
                          (+ (* s1 (nth t1 2)) (* s2 (nth t2 2)))]}))))))))

(def ao-offsets (compute-ao-offsets))

;; ================================================================
;; Block neighbor query
;; ================================================================

(defn get-neighbor-block
  "Get block at local+offset, looking into adjacent chunks if needed."
  [chunk world lx ly lz dx dy dz]
  (let [lx (long lx) ly (long ly) lz (long lz)
        dx (long dx) dy (long dy) dz (long dz)
        nx (+ lx dx)
        ny (+ ly dy)
        nz (+ lz dz)]
    (if (and (>= nx 0) (< nx w/CHUNK-SIZE)
             (>= ny 0) (< ny w/CHUNK-SIZE)
             (>= nz 0) (< nz w/CHUNK-SIZE))
      (w/get-block chunk nx ny nz)
      (let [[cx cy cz] (:pos chunk)
            wx (+ (* (long cx) (long w/CHUNK-SIZE)) nx)
            wy (+ (* (long cy) (long w/CHUNK-SIZE)) ny)
            wz (+ (* (long cz) (long w/CHUNK-SIZE)) nz)
            ncx (Math/floorDiv wx (long w/CHUNK-SIZE))
            ncy (Math/floorDiv wy (long w/CHUNK-SIZE))
            ncz (Math/floorDiv wz (long w/CHUNK-SIZE))]
        (if-let [nc (get world [ncx ncy ncz])]
          (w/get-block nc
            (Math/floorMod wx (long w/CHUNK-SIZE))
            (Math/floorMod wy (long w/CHUNK-SIZE))
            (Math/floorMod wz (long w/CHUNK-SIZE)))
          w/AIR)))))

(defn- get-neighbor-light
  "Get light byte at local+offset, looking into adjacent chunks if needed."
  [chunk world lx ly lz dx dy dz]
  (let [lx (long lx) ly (long ly) lz (long lz)
        dx (long dx) dy (long dy) dz (long dz)
        nx (+ lx dx)
        ny (+ ly dy)
        nz (+ lz dz)]
    (if (and (>= nx 0) (< nx w/CHUNK-SIZE)
             (>= ny 0) (< ny w/CHUNK-SIZE)
             (>= nz 0) (< nz w/CHUNK-SIZE))
      (let [idx (w/block-index nx ny nz)]
        (aget ^bytes (:light chunk) (int idx)))
      (let [[cx cy cz] (:pos chunk)
            wx (+ (* (long cx) (long w/CHUNK-SIZE)) nx)
            wy (+ (* (long cy) (long w/CHUNK-SIZE)) ny)
            wz (+ (* (long cz) (long w/CHUNK-SIZE)) nz)
            ncx (Math/floorDiv wx (long w/CHUNK-SIZE))
            ncy (Math/floorDiv wy (long w/CHUNK-SIZE))
            ncz (Math/floorDiv wz (long w/CHUNK-SIZE))]
        (if-let [nc (get world [ncx ncy ncz])]
          (let [nlx (Math/floorMod wx (long w/CHUNK-SIZE))
                nly (Math/floorMod wy (long w/CHUNK-SIZE))
                nlz (Math/floorMod wz (long w/CHUNK-SIZE))
                idx (w/block-index nlx nly nlz)]
            (aget ^bytes (:light nc) (int idx)))
          0)))))

(defn- compute-vertex-ao
  "Compute AO value (0-3) for a vertex given its face and vertex index.
  Uses the precomputed ao-offsets table.
  nx,ny,nz = position of the neighbor block (in the face normal direction)."
  [chunk world nx ny nz face vertex]
  (let [offsets (nth (nth ao-offsets face) vertex)
        [s1x s1y s1z] (:side1 offsets)
        [s2x s2y s2z] (:side2 offsets)
        [cx cy cz] (:corner offsets)
        side1-solid (w/solid? (get-neighbor-block chunk world nx ny nz s1x s1y s1z))
        side2-solid (w/solid? (get-neighbor-block chunk world nx ny nz s2x s2y s2z))
        corner-solid (w/solid? (get-neighbor-block chunk world nx ny nz cx cy cz))]
    (if (and side1-solid side2-solid)
      0  ;; fully occluded
      (- 3 (+ (if side1-solid 1 0) (if side2-solid 1 0) (if corner-solid 1 0))))))

;; ================================================================
;; Mesh generation
;; ================================================================

(defn- face-key
  "Compute a merge key for greedy meshing. Faces with the same key can be merged.
  Encodes: texture layer (8 bits), day light (8 bits), night light (8 bits),
  ao0-ao3 (2 bits each = 8 bits)."
  [tex-layer day-l-raw night-l-raw ao0 ao1 ao2 ao3]
  (let [tex-layer (long tex-layer) day-l-raw (long day-l-raw) night-l-raw (long night-l-raw)
        ao0 (long ao0) ao1 (long ao1) ao2 (long ao2) ao3 (long ao3)]
    (bit-or (bit-shift-left tex-layer 24)
            (bit-shift-left day-l-raw 16)
            (bit-shift-left night-l-raw 8)
            (bit-shift-left ao0 6)
            (bit-shift-left ao1 4)
            (bit-shift-left ao2 2)
            ao3)))

(defn mesh-chunk
  "Generate mesh data for a chunk with greedy meshing, AO, and dual light channels.
  For each of 6 face directions, sweeps through 16 slices and greedily merges
  adjacent coplanar faces with matching texture/light/AO into larger quads.
  Returns {:vertices float[] :indices int[] :vertex-count :index-count}
  or nil if chunk has no visible faces."
  [chunk world]
  (let [blocks ^ints (:blocks chunk)
        light-arr ^bytes (:light chunk)
        [cx cy cz] (:pos chunk)
        wx0 (double (* (long cx) (long w/CHUNK-SIZE)))
        wy0 (double (* (long cy) (long w/CHUNK-SIZE)))
        wz0 (double (* (long cz) (long w/CHUNK-SIZE)))
        cs (long w/CHUNK-SIZE)
        max-verts (* 4096 6 4)
        max-indices (* 4096 6 6)
        verts (float-array (* max-verts VERTEX-STRIDE))
        indices (int-array max-indices)
        vc (long-array 1)
        ic (long-array 1)
        ;; Greedy mask: 16x16 grid of face keys (0 = no face)
        mask (long-array (* cs cs))
        ;; Store AO per cell for later vertex emission
        ao-arr (int-array (* cs cs 4))]

    (dotimes [face 6]
      (let [[dx dy dz] (nth neighbor-offsets face)
            dx (long dx) dy (long dy) dz (long dz)
            shade (aget ^floats face-shade face)
            face-verts (nth face-vertices face)]

        ;; For each slice perpendicular to the face normal
        (dotimes [slice cs]
          ;; Clear mask
          (java.util.Arrays/fill mask (long 0))

          ;; Build the 2D mask for this slice
          ;; Face direction determines which axis is the slice axis and which two are the grid axes
          ;; TOP/BOTTOM: slice=Y, grid=(X,Z)
          ;; NORTH/SOUTH: slice=Z, grid=(X,Y)
          ;; EAST/WEST: slice=X, grid=(Z,Y)
          (dotimes [g1 cs]
            (dotimes [g0 cs]
              (let [;; Map (slice, g0, g1) to (lx, ly, lz) based on face
                    lx (long (case (int face)
                               (0 1) g0       ;; TOP/BOTTOM: g0=X
                               (2 3) g0       ;; NORTH/SOUTH: g0=X
                               (4 5) slice))   ;; EAST/WEST: slice=X
                    ly (long (case (int face)
                               (0 1) slice     ;; TOP/BOTTOM: slice=Y
                               (2 3) g1        ;; NORTH/SOUTH: g1=Y
                               (4 5) g1))      ;; EAST/WEST: g1=Y
                    lz (long (case (int face)
                               (0 1) g1        ;; TOP/BOTTOM: g1=Z
                               (2 3) slice     ;; NORTH/SOUTH: slice=Z
                               (4 5) g0))      ;; EAST/WEST: g0=Z
                    block-id (aget blocks (w/block-index lx ly lz))]
                (when (and (not= block-id w/AIR)
                           (not= block-id w/WATER))
                  (let [faces-arr (get w/block-faces block-id)]
                    (when faces-arr
                      (let [neighbor (get-neighbor-block chunk world lx ly lz dx dy dz)]
                        (when (w/transparent? neighbor)
                          (let [tex-layer (long (nth faces-arr face))
                                nlx (+ lx dx) nly (+ ly dy) nlz (+ lz dz)
                                light-byte (long (get-neighbor-light chunk world lx ly lz dx dy dz))
                                day-raw (bit-and light-byte 0x0F)
                                night-raw (bit-and (unsigned-bit-shift-right light-byte 4) 0x0F)
                                ao0 (long (compute-vertex-ao chunk world nlx nly nlz face 0))
                                ao1 (long (compute-vertex-ao chunk world nlx nly nlz face 1))
                                ao2 (long (compute-vertex-ao chunk world nlx nly nlz face 2))
                                ao3 (long (compute-vertex-ao chunk world nlx nly nlz face 3))
                                gi (+ g0 (* g1 cs))
                                key (face-key tex-layer day-raw night-raw ao0 ao1 ao2 ao3)]
                            (aset mask gi key)
                            (let [ai (* gi 4)]
                              (aset ao-arr ai (int ao0))
                              (aset ao-arr (+ ai 1) (int ao1))
                              (aset ao-arr (+ ai 2) (int ao2))
                              (aset ao-arr (+ ai 3) (int ao3))))))))))))

          ;; Greedy merge: sweep rows, extend rectangles
          (loop [g1 (long 0)]
            (when (< g1 cs)
              (loop [g0 (long 0)]
                (when (< g0 cs)
                  (let [gi (+ g0 (* g1 cs))
                        key (aget mask gi)]
                    (if (zero? key)
                      (recur (inc g0))
                      ;; Found a face — extend width (g0 direction)
                      (let [w0 (loop [w (long 1)]
                                 (if (and (< (+ g0 w) cs)
                                          (= (aget mask (+ g0 w (* g1 cs))) key))
                                   (recur (inc w))
                                   w))
                            ;; Extend height (g1 direction)
                            h0 (loop [h (long 1)]
                                 (if (< (+ g1 h) cs)
                                   (let [row-ok? (loop [c (long 0)]
                                                   (if (>= c w0) true
                                                     (if (= (aget mask (+ g0 c (* (+ g1 h) cs))) key)
                                                       (recur (inc c))
                                                       false)))]
                                     (if row-ok? (recur (inc h)) h))
                                   h))]
                        ;; Clear the merged region from mask
                        (dotimes [dh h0]
                          (dotimes [dw w0]
                            (aset mask (+ g0 dw (* (+ g1 dh) cs)) (long 0))))

                        ;; Emit quad for merged region [g0, g1] to [g0+w0, g1+h0]
                        (let [vi (aget vc 0)
                              ii (aget ic 0)
                              ;; Decode key back to properties
                              tex-layer (float (bit-and (unsigned-bit-shift-right key 24) 0xFF))
                              day-raw (bit-and (unsigned-bit-shift-right key 16) 0xFF)
                              night-raw (bit-and (unsigned-bit-shift-right key 8) 0xFF)
                              day-l (float (* (aget ^floats light/light-decode-table (int day-raw)) shade))
                              night-l (float (* (aget ^floats light/light-decode-table (int night-raw)) shade))
                              ;; AO from original cell
                              ai (* gi 4)
                              ao0 (aget ao-arr ai)
                              ao1 (aget ao-arr (+ ai 1))
                              ao2 (aget ao-arr (+ ai 2))
                              ao3 (aget ao-arr (+ ai 3))]

                          ;; Emit 4 vertices of the merged quad
                          ;; face-vertices gives unit offsets; we scale UVs by (w0, h0)
                          ;; and position by the block start + merged region size
                          (dotimes [v 4]
                            (let [[vx vy vz vu vv] (nth face-verts v)
                                  ;; Scale vertex offsets by merged quad size
                                  ;; The grid axes (g0, g1) map to specific world axes
                                  sx (double (if (zero? (long vu)) 0.0 (double w0)))
                                  sy (double (if (zero? (long vv)) 0.0 (double h0)))
                                  ;; Map grid position back to world
                                  ;; Convert (slice, g0, g1) + vertex offsets to world pos
                                  px (double (case (int face)
                                               (0 1) (+ wx0 g0 (* (double vx) (double w0)))
                                               (2 3) (+ wx0 g0 (* (double vx) (double w0)))
                                               (4 5) (+ wx0 slice (double vx))))
                                  py (double (case (int face)
                                               (0 1) (+ wy0 slice (double vy))
                                               (2 3) (+ wy0 g1 (* (double vy) (double h0)))
                                               (4 5) (+ wy0 g1 (* (double vy) (double h0)))))
                                  pz (double (case (int face)
                                               (0 1) (+ wz0 g1 (* (double vz) (double h0)))
                                               (2 3) (+ wz0 slice (double vz))
                                               (4 5) (+ wz0 g0 (* (double vz) (double w0)))))
                                  ;; UVs scale with merged quad size for texture tiling
                                  u (float (* (double vu) (double w0)))
                                  v-coord (float (* (double vv) (double h0)))
                                  base (* (+ vi v) VERTEX-STRIDE)
                                  ao-val (case (int v)
                                           0 ao0 1 ao1 2 ao2 3 ao3)]
                              (aset verts (+ base 0) (float px))
                              (aset verts (+ base 1) (float py))
                              (aset verts (+ base 2) (float pz))
                              (aset verts (+ base 3) u)
                              (aset verts (+ base 4) v-coord)
                              (aset verts (+ base 5) day-l)
                              (aset verts (+ base 6) night-l)
                              (aset verts (+ base 7) tex-layer)
                              (aset verts (+ base 8) (aget ^floats ao-multiplier (int ao-val)))))

                          ;; Indices with AO-based diagonal flip
                          (if (> (+ ao0 ao2) (+ ao1 ao3))
                            (do
                              (aset indices (+ ii 0) (int vi))
                              (aset indices (+ ii 1) (int (+ vi 1)))
                              (aset indices (+ ii 2) (int (+ vi 3)))
                              (aset indices (+ ii 3) (int (+ vi 1)))
                              (aset indices (+ ii 4) (int (+ vi 2)))
                              (aset indices (+ ii 5) (int (+ vi 3))))
                            (do
                              (aset indices (+ ii 0) (int vi))
                              (aset indices (+ ii 1) (int (+ vi 1)))
                              (aset indices (+ ii 2) (int (+ vi 2)))
                              (aset indices (+ ii 3) (int vi))
                              (aset indices (+ ii 4) (int (+ vi 2)))
                              (aset indices (+ ii 5) (int (+ vi 3)))))
                          (aset vc 0 (+ vi 4))
                          (aset ic 0 (+ ii 6)))
                        (recur (+ g0 w0)))))))
              (recur (inc g1)))))))

    (let [final-vc (aget vc 0)
          final-ic (aget ic 0)]
      (when (pos? final-vc)
        {:vertices (java.util.Arrays/copyOf verts (int (* final-vc VERTEX-STRIDE)))
         :indices (java.util.Arrays/copyOf indices (int final-ic))
         :vertex-count final-vc
         :index-count final-ic}))))

(defn mesh-chunk-water
  "Generate water mesh for a chunk. Only emits top faces of water blocks
  that are exposed to air. Returns {:vertices float[] :indices int[]} or nil."
  [chunk world]
  (let [blocks ^ints (:blocks chunk)
        light-arr ^bytes (:light chunk)
        [cx cy cz] (:pos chunk)
        wx0 (double (* (long cx) (long w/CHUNK-SIZE)))
        wy0 (double (* (long cy) (long w/CHUNK-SIZE)))
        wz0 (double (* (long cz) (long w/CHUNK-SIZE)))
        max-verts (* 4096 4)
        max-indices (* 4096 6)
        verts (float-array (* max-verts VERTEX-STRIDE))
        indices (int-array max-indices)
        vc (long-array 1)
        ic (long-array 1)
        water-layer (float 4.0)
        water-y-offset (double 0.875)]
    (dotimes [ly w/CHUNK-SIZE]
      (dotimes [lz w/CHUNK-SIZE]
        (dotimes [lx w/CHUNK-SIZE]
          (let [block-id (aget blocks (w/block-index lx ly lz))]
            (when (= block-id w/WATER)
              (let [above (get-neighbor-block chunk world lx ly lz 0 1 0)]
                (when (= above w/AIR)
                  (let [vi (aget vc 0)
                        ii (aget ic 0)
                        wy (+ wy0 (double ly) water-y-offset)
                        ;; Get light from block above the water
                        light-byte (get-neighbor-light chunk world lx ly lz 0 1 0)
                        day-l (float (aget ^floats light/light-decode-table (bit-and light-byte 0x0F)))
                        night-l (float (aget ^floats light/light-decode-table
                                         (bit-and (unsigned-bit-shift-right light-byte 4) 0x0F)))]
                    ;; 4 vertices — water top face
                    (doseq [[vi-off wx wz u v]
                            [[0 0.0 0.0 0.0 0.0]
                             [1 0.0 1.0 0.0 1.0]
                             [2 1.0 1.0 1.0 1.0]
                             [3 1.0 0.0 1.0 0.0]]]
                      (let [base (* (+ vi (long vi-off)) VERTEX-STRIDE)]
                        (aset verts (+ base 0) (float (+ wx0 lx (double wx))))
                        (aset verts (+ base 1) (float wy))
                        (aset verts (+ base 2) (float (+ wz0 lz (double wz))))
                        (aset verts (+ base 3) (float u))
                        (aset verts (+ base 4) (float v))
                        (aset verts (+ base 5) day-l)
                        (aset verts (+ base 6) night-l)
                        (aset verts (+ base 7) water-layer)
                        (aset verts (+ base 8) (float 1.0))))  ;; AO = 1.0 for water
                    (aset indices (+ ii 0) (int vi))
                    (aset indices (+ ii 1) (int (+ vi 1)))
                    (aset indices (+ ii 2) (int (+ vi 2)))
                    (aset indices (+ ii 3) (int vi))
                    (aset indices (+ ii 4) (int (+ vi 2)))
                    (aset indices (+ ii 5) (int (+ vi 3)))
                    (aset vc 0 (+ vi 4))
                    (aset ic 0 (+ ii 6))))))))))
    (let [final-vc (aget vc 0)
          final-ic (aget ic 0)]
      (when (pos? final-vc)
        {:vertices (java.util.Arrays/copyOf verts (int (* final-vc VERTEX-STRIDE)))
         :indices (java.util.Arrays/copyOf indices (int final-ic))
         :vertex-count final-vc
         :index-count final-ic}))))

(defn mesh-world
  "Mesh all dirty chunks in world. Returns world with updated :mesh-data on chunks."
  [world]
  (reduce-kv
    (fn [w pos chunk]
      (if (:dirty? chunk)
        (let [md (mesh-chunk chunk world)
              wmd (mesh-chunk-water chunk world)]
          (assoc w pos (assoc chunk :mesh-data md :water-mesh-data wmd :dirty? false)))
        w))
    world
    world))
