(ns valley.chunk
  "Portable streaming voxel world: deterministic terrain/cave/water/tree/ore generation
   (a pure function of world coords, via the valley.kernels noise facade), a moving
   active-buffer ring (recenter = copy overlap + gen new edge strips), per-column face
   meshing with skylight+AO in absolute world coords, and the block-property tables the
   physics kernels read. Cross-platform array helpers keep the JVM (Java primitive arrays)
   and the browser (typed arrays in wasm memory) on the same code."
  (:require [valley.kernels :as k]
            [valley.tex :as tex]))

(def ^:const CS 16)                       ; chunk size (valley CHUNK-SIZE)
(def ^:const STRIDE 24)                   ; pos3 + uv2 + layer (matches valley.slice)

;; --- cross-platform numeric arrays -------------------------------------------
(defn iarray [n] #?(:clj (int-array n)    :cljs (js/Int32Array. n)))
(defn barray [n] #?(:clj (byte-array n)   :cljs (js/Int8Array. n)))
(defn darray [n] #?(:clj (double-array n) :cljs (js/Float64Array. n)))
;; clojure.core aget/aset work on both Java arrays and JS typed arrays.

(defn idx ^long [^long x ^long y ^long z] (+ x (* z CS) (* y CS CS)))
(defn- iabs [n] (if (neg? n) (- n) n))

;; block ids (cf valley.tex/block-faces): 0 air, 1 grass, 2 dirt, 3 stone, 4 sand,
;; 5 snowy-grass, 6 snow, 7 podzol, 8 sandstone, 9 mountain-stone,
;; 10 water (transparent, non-solid), 11 log, 12 leaves,
;; 13 coal-ore, 14 iron-ore, 15 gold-ore, 16 diamond-ore.
(def ^:const WATER 10)
(def ^:const LOG 11)
(def ^:const LEAVES 12)
(def ^:const COAL 13)
(def ^:const IRON 14)
(def ^:const GOLD 15)
(def ^:const DIAMOND 16)
(def ^:private solid-ids [1 2 3 4 5 6 7 8 9 11 12 13 14 15 16])   ; everything except air + water
(defn solid-table []
  (let [s (barray 62)]
    (doseq [i solid-ids] (aset s (int i) #?(:clj (byte 1) :cljs 1)))
    s))
;; opaque = occludes neighbour faces (air + water are see-through for the mesher)
(defn opaque? [id] (and (pos? (long id)) (not= (long id) WATER)))

;; --- face mesh: per-face geometry + UVs (interior faces culled in mesh-region) ---
(def ^:private faces
  [[[0 0 1]  [[0 0 1] [1 0 1] [1 1 1] [0 1 1]]]
   [[0 0 -1] [[1 0 0] [0 0 0] [0 1 0] [1 1 0]]]
   [[1 0 0]  [[1 0 1] [1 0 0] [1 1 0] [1 1 1]]]
   [[-1 0 0] [[0 0 0] [0 0 1] [0 1 1] [0 1 0]]]
   [[0 1 0]  [[0 1 1] [1 1 1] [1 1 0] [0 1 0]]]
   [[0 -1 0] [[0 0 0] [1 0 0] [1 0 1] [0 0 1]]]])
(def ^:private face-uv [[0.0 0.0] [1.0 0.0] [1.0 1.0] [0.0 1.0]])

;; --- streaming world buffer ------------------------------------------------------
;; The physics kernel only ever sees ONE 16³ chunk; the player AABB is tiny, so each
;; frame we stitch a player-centred 16³ window out of the active buffer (stitch-window!)
;; and call the kernel against it at origin 0 — wasm kernel + marshaling stay identical
;; to a single chunk. World coords index a flat array: wx + wz*WX + wy*WX*WZ.

(defn world-idx [wx wy wz wx-dim wz-dim]   ; >4 args ⇒ no primitive hints (params or return)
  (+ (long wx) (* (long wz) (long wx-dim)) (* (long wy) (long wx-dim) (long wz-dim))))

;; biome (0..10) → surface / fill block id (cf valley.terrain/biomes)
;; 0 desert 1 savanna 2 plains 3 forest 4 jungle 5 swamp 6 taiga 7 tundra
;; 8 snowy-forest 9 mountains 10 mushroom
(def ^:private biome-surface [4 1 1 1 1 1 7 5 5 3 1])
(def ^:private biome-fill    [8 2 2 2 2 2 2 2 2 3 2])
;; chunk face index (faces vector order: +z -z +x -x +y -y) → block-faces index
;; (block-faces order: top bottom north south east west)
(def ^:private face->bf [2 3 4 5 0 1])

;; deterministic placement hash (platform-stable: stays within JS safe-integer range)
(defn- phash [x z]
  (mod (+ (* (+ (long x) 7) 374761393) (* (+ (long z) 13) 668265263)) 2147483647))

(defn- phash3 [x y z]
  (mod (+ (* (+ (long x) 7) 374761393) (* (+ (long y) 11) 1597334677)
          (* (+ (long z) 13) 668265263)) 2147483647))

;; Ores replace deep stone in small blobby veins. A coarse 2³ cell (>>1) decides the vein's
;; presence + type (rarity ↑ with value); a per-cell hash thins it to ~5-block clumps; depth
;; below the surface gates each type (diamond deepest). Deterministic → identical on reload.
(defn- ore-at
  "Ore block id (0 = plain stone) for deep-stone cell (wx,wy,wz) under surface height h."
  ^long [wx wy wz h]
  (let [d (- (long h) (long wy))]                       ; depth below surface
    (if (< d 4)
      0
      (let [vein (mod (phash3 (bit-shift-right (long wx) 1)
                              (bit-shift-right (long wy) 1)
                              (bit-shift-right (long wz) 1)) 1000)
            cell (phash3 wx wy wz)
            ore  (cond (and (< vein 4)  (>= d 14)) DIAMOND
                       (and (< vein 12) (>= d 11)) GOLD
                       (and (< vein 45) (>= d 7))  IRON
                       (< vein 120)                COAL
                       :else 0)]
        (if (and (pos? ore) (not= 0 (mod cell 3))) ore 0)))))   ; thin vein to ~2/3 of the blob

;; ============================================================================
;; Streaming world gen (Minetest-style: terrain is a PURE function of world (x,y,z) — no
;; global normalization — so every column generates identically regardless of load order).
;; ============================================================================
(def ^:const COL-H 48)         ; column height (blocks) — taller than the demo box for peaks/caves
(def ^:const STREAM-RG 5)      ; data radius (chunks): active buffer covers ±RG
(def ^:const STREAM-R 4)       ; mesh/render radius (< RG so meshed columns have neighbours)
(def ^:const STREAM-BUDGET 2)  ; columns meshed per frame (spread pop-in)
(def ^:const STREAM-AW (* (+ (* 2 STREAM-RG) 1) CS))   ; active-buffer width/depth (blocks)
(def ^:const TERRAIN-BASE 4)   ; lowest solid level
(def ^:const TERRAIN-REF 38)   ; fixed vertical reference (replaces the old global world-min)
(def ^:const SEA-LEVEL 14)     ; lakes fill below this
(def ^:const TREE-MARGIN 2)    ; canopy horizontal radius → neighbour-root scan margin

(defn column-height
  "Absolute surface height at world column (wx,wz): the deterministic kernel height placed at a
   FIXED reference (no world-min), clamped into [1, COL-H)."
  ^long [wx wz]
  (max 1 (min (dec COL-H) (+ TERRAIN-BASE (- (long (k/surface-height-biome wx wz)) TERRAIN-REF)))))

(defn- gen-into!
  "Fill region [rx0,rx1)×[0,COL-H)×[rz0,rz1) of array b (dims AW×COL-H×AD, world origin ox,oz)
   with terrain+caves+water, then trees (roots scanned in a ±TREE-MARGIN ring so neighbour-rooted
   canopies spill in identically). Writes are clipped to both the array and the region, so the
   same routine generates one column (AW=CS) or a whole region (oracle) bit-identically."
  [b AW AD ox oz rx0 rz0 rx1 rz1]
  (let [AW (long AW) AD (long AD) ox (long ox) oz (long oz)
        rx0 (long rx0) rz0 (long rz0) rx1 (long rx1) rz1 (long rz1)
        lidx (fn [lx ly lz] (+ (long lx) (* (long lz) AW) (* (long ly) AW AD)))
        in?  (fn [wx wy wz] (let [lx (- (long wx) ox) lz (- (long wz) oz) ly (long wy)]
                              (and (>= lx 0) (< lx AW) (>= lz 0) (< lz AD) (>= ly 0) (< ly COL-H)
                                   (>= (long wx) rx0) (< (long wx) rx1) (>= (long wz) rz0) (< (long wz) rz1))))
        put  (fn [wx wy wz id] (when (in? wx wy wz)
                                 (aset b (lidx (- (long wx) ox) (long wy) (- (long wz) oz)) (int id))))
        air? (fn [wx wy wz] (or (not (in? wx wy wz))
                                (zero? (aget #?(:clj ^ints b :cljs b)
                                             (lidx (- (long wx) ox) (long wy) (- (long wz) oz))))))]
    ;; terrain + caves + water for every column in the region
    (doseq [wx (range rx0 rx1) wz (range rz0 rz1)]
      (let [h    (column-height wx wz)
            bi   (long (k/biome-index (* wx 8) (* wz 8)))
            surf (nth biome-surface bi) fill (nth biome-fill bi)]
        (dotimes [wy COL-H]
          (when (and (< wy h)
                     (not (and (> wy 1) (< wy (dec h)) (= 1 (long (k/has-cave? wx wy wz))))))
            (put wx wy wz (cond (= wy (dec h)) surf
                                (>= wy (- h 3)) fill
                                :else (let [o (ore-at wx wy wz h)] (if (pos? o) o 3))))))
        (when (< h SEA-LEVEL)
          (loop [wy h] (when (< wy SEA-LEVEL) (put wx wy wz WATER) (recur (inc wy)))))))
    ;; trees: scan roots in the margin ring, x then z (matches global order), place spillover
    (doseq [wx (range (- rx0 TREE-MARGIN) (+ rx1 TREE-MARGIN))
            wz (range (- rz0 TREE-MARGIN) (+ rz1 TREE-MARGIN))]
      (let [h (column-height wx wz) bi (long (k/biome-index (* wx 8) (* wz 8)))]
        (when (and (>= h SEA-LEVEL) (= 1 (long (nth biome-surface bi)))
                   (zero? (mod (phash wx wz) 37)) (< (+ h 7) COL-H))
          (let [th (+ 4 (mod (phash wx wz) 3)) top (+ h th)]
            (dotimes [t th] (put wx (+ h t) wz LOG))
            (doseq [dy [-2 -1]] (let [ly (+ top dy -1)]
                                  (doseq [dx [-2 -1 0 1 2] dz [-2 -1 0 1 2]]
                                    (when (and (not (and (= (iabs dx) 2) (= (iabs dz) 2)))
                                               (air? (+ wx dx) ly (+ wz dz)))
                                      (put (+ wx dx) ly (+ wz dz) LEAVES)))))
            (doseq [dy [0 1]] (let [ly (+ top dy -1)]
                                (doseq [dx [-1 0 1] dz [-1 0 1]]
                                  (when (air? (+ wx dx) ly (+ wz dz))
                                    (put (+ wx dx) ly (+ wz dz) LEAVES)))))))))
    b))

(defn- skylight-into!
  "Top-down skylight (0/15) into the region [rx0,rx1)×[rz0,rz1) of light buffer (dims AW×COL-H×AD,
   origin ox,oz) from block buffer b. Per-column, no cross-column bleed."
  [b light AW AD ox oz rx0 rz0 rx1 rz1]
  (let [AW (long AW) AD (long AD) ox (long ox) oz (long oz)]
    (doseq [wx (range rx0 rx1) wz (range rz0 rz1)]
      (let [lx (- (long wx) ox) lz (- (long wz) oz)]
        (loop [ly (dec COL-H) lit true]
          (when (>= ly 0)
            (let [i (+ lx (* lz AW) (* ly AW AD))
                  air (zero? (aget #?(:clj ^ints b :cljs b) i))]
              (if (and lit air)
                (do (aset #?(:clj ^ints light :cljs light) i #?(:clj (int 15) :cljs 15)) (recur (dec ly) true))
                (recur (dec ly) false)))))))))

(defn- region-minus
  "Rectangle difference R \\ O (O ⊆ R) as ≤4 disjoint rects [x0 z0 x1 z1] (left/right/top/bottom)."
  [rx0 rz0 rx1 rz1 ox0 oz0 ox1 oz1]
  (filter (fn [[a b c d]] (and (< a c) (< b d)))
          [[rx0 rz0 ox0 rz1] [ox1 rz0 rx1 rz1] [ox0 rz0 ox1 oz0] [ox0 oz1 ox1 rz1]]))

;; --- active buffer: the loaded ring as one moving flat world ------------------------------
;; A "stream" is a world map {:blocks :light :wx :wy :wz :aox :aoz :rg :center :solid :scratch}
;; whose flat arrays cover chunks within `rg` of `center`. All consumers (mesher, stitch,
;; mob kernel) treat it as a flat world at buffer-local coords; world↔buffer offset = (aox,aoz).
(defn make-stream
  "Create an active buffer centred on chunk (pcx,pcz) with data radius `rg` (chunks)."
  [pcx pcz rg]
  (let [rg (long rg) AW (* (+ (* 2 rg) 1) CS)
        aox (* (- (long pcx) rg) CS) aoz (* (- (long pcz) rg) CS)
        buf (iarray (* AW COL-H AW)) light (iarray (* AW COL-H AW))]
    (gen-into! buf AW AW aox aoz aox aoz (+ aox AW) (+ aoz AW))
    (skylight-into! buf light AW AW aox aoz aox aoz (+ aox AW) (+ aoz AW))
    {:blocks buf :light light :wx AW :wy COL-H :wz AW :aox aox :aoz aoz
     :rg rg :center [(long pcx) (long pcz)] :solid (solid-table)
     :scratch (iarray (* CS CS CS))}))

(defn recenter-stream
  "Re-centre the active buffer on chunk (pcx,pcz): retained columns are copied, newly-entered
   strips generated (seamless — canopy radius = tree margin). Returns the updated stream (or the
   same one if the centre is unchanged)."
  [s pcx pcz]
  (let [pcx (long pcx) pcz (long pcz)]
    (if (= (:center s) [pcx pcz])
      s
      (let [rg (long (:rg s)) AW (* (+ (* 2 rg) 1) CS)
            naox (* (- pcx rg) CS) naoz (* (- pcz rg) CS)
            oaox (long (:aox s)) oaoz (long (:aoz s))
            ob #?(:clj ^ints (:blocks s) :cljs (:blocks s)) ol #?(:clj ^ints (:light s) :cljs (:light s))
            nb (iarray (* AW COL-H AW)) nl (iarray (* AW COL-H AW))
            ox0 (max naox oaox) oz0 (max naoz oaoz)
            ox1 (min (+ naox AW) (+ oaox AW)) oz1 (min (+ naoz AW) (+ oaoz AW))
            overlap? (and (< ox0 ox1) (< oz0 oz1))]
        (if-not overlap?
          ;; far jump (no overlap): generate the whole new region
          (do (gen-into! nb AW AW naox naoz naox naoz (+ naox AW) (+ naoz AW))
              (skylight-into! nb nl AW AW naox naoz naox naoz (+ naox AW) (+ naoz AW)))
          (do
            ;; copy retained overlap
            (doseq [wx (range ox0 ox1) wz (range oz0 oz1)]
              (let [olx (- wx oaox) olz (- wz oaoz) nlx (- wx naox) nlz (- wz naoz)]
                (dotimes [ly COL-H]
                  (let [oi (+ olx (* olz AW) (* ly AW AW)) ni (+ nlx (* nlz AW) (* ly AW AW))]
                    (aset #?(:clj ^ints nb :cljs nb) ni (aget ob oi))
                    (aset #?(:clj ^ints nl :cljs nl) ni (aget ol oi))))))
            ;; generate newly-entered strips
            (doseq [[sx0 sz0 sx1 sz1] (region-minus naox naoz (+ naox AW) (+ naoz AW) ox0 oz0 ox1 oz1)]
              (gen-into! nb AW AW naox naoz sx0 sz0 sx1 sz1)
              (skylight-into! nb nl AW AW naox naoz sx0 sz0 sx1 sz1))))
        (assoc s :blocks nb :light nl :aox naox :aoz naoz :center [pcx pcz])))))

;; world-block/set-block! take WORLD coords; on a stream world they subtract the active-buffer
;; origin (:aox/:aoz, default 0 for the fixed demo world) → buffer-local index.
(defn world-block ^long [world ^long wx ^long wy ^long wz]
  (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world))
        wx (- wx (long (:aox world 0))) wz (- wz (long (:aoz world 0)))]
    (if (and (>= wx 0) (< wx WX) (>= wy 0) (< wy WY) (>= wz 0) (< wz WZ))
      ;; ^ints hint → primitive aget on the JVM (meshing + AO + light hit this heavily);
      ;; cljs ignores it (typed-array access is already direct).
      (long (aget #?(:clj ^ints (:blocks world) :cljs (:blocks world)) (world-idx wx wy wz WX WZ)))
      0)))

(defn set-block!
  "Set the block at world (wx,wy,wz) to id (in-place, bounds-checked). Returns true if it
   changed a cell. Used by mining (break = id 0, place = a block id)."
  [world wx wy wz id]
  (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world))
        wx (- (long wx) (long (:aox world 0))) wy (long wy) wz (- (long wz) (long (:aoz world 0)))]
    (when (and (>= wx 0) (< wx WX) (>= wy 0) (< wy WY) (>= wz 0) (< wz WZ))
      (aset #?(:clj ^ints (:blocks world) :cljs (:blocks world)) (world-idx wx wy wz WX WZ) (int id))
      true)))

(defn stitch-window!
  "Fill a 16³ scratch int array from `world` centred at world origin (wox,woy,woz)
   — scratch local (i,j,k) ← world (wox+i, woy+j, woz+k). Returns scratch.

   Hot path: called every frame. The block array + dims are hoisted out of the 4096-cell
   loop and the flat index is computed primitively (no per-cell world-block call → no map
   lookups, no boxed world-idx). On the JVM the ^ints hints make aget/aset primitive (the
   untyped-array version was reflective → ~65ms/frame); cljs ignores the hints (typed-array
   access is already direct, which is why the browser never showed the regression)."
  [world scratch wox woy woz]   ; >4 params ⇒ no primitive hints on params/return
  (let [#?@(:clj  [b ^ints (:blocks world) scr ^ints scratch]
            :cljs [b (:blocks world) scr scratch])
        ;; world → buffer-local origin (stream worlds carry :aox/:aoz; demo world → 0)
        wox (- (long wox) (long (:aox world 0))) woy (long woy) woz (- (long woz) (long (:aoz world 0)))
        WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world))
        WXZ (* WX WZ)]
    (dotimes [j CS]
      (let [wy   (+ woy j)
            in-y (and (>= wy 0) (< wy WY))
            yoff (* wy WXZ)]
        (dotimes [k CS]
          (let [wz    (+ woz k)
                in-yz (and in-y (>= wz 0) (< wz WZ))
                zoff  (+ yoff (* wz WX))]
            (dotimes [i CS]
              (let [wx (+ wox i)
                    v  (if (and in-yz (>= wx 0) (< wx WX)) (aget b (+ zoff wx)) 0)]
                (aset scr (idx i j k) #?(:clj (int v) :cljs v)))))))))
  scratch)

;; --- lighting (port of valley.lighting skylight + valley.mesher AO) ------------------
(defn- pow* [a b] (#?(:clj Math/pow :cljs js/Math.pow) (double a) (double b)))

(defn- light-at
  "Skylight at world cell (clamped to 0 outside)."
  [skylight x y z WX WY WZ]                ; >4 args ⇒ no primitive hints
  (let [x (long x) y (long y) z (long z) WX (long WX) WY (long WY) WZ (long WZ)]
    (if (and (>= x 0) (< x WX) (>= y 0) (< y WY) (>= z 0) (< z WZ))
      (long (aget #?(:clj ^ints skylight :cljs skylight) (world-idx x y z WX WZ)))
      0)))

;; ambient-occlusion: 4 corner levels (0 darkest … 3 none) → brightness multiplier
(def ^:private ao-mult [0.45 0.68 0.85 1.0])
(defn- b2i [x] (if x 1 0))
(defn- ao-level [s1 s2 cor] (if (and s1 s2) 0 (- 3 (+ (b2i s1) (b2i s2) (b2i cor)))))

(defn- corner-ao
  "AO level (0..3) for one corner of a face, from the 3 blocks around it in the air layer
   one step along the face normal (standard voxel AO). `oc` = opaque-at reader."
  [oc x y z nx ny nz ox oy oz]
  (let [bx (+ x nx) by (+ y ny) bz (+ z nz)
        sx (- (* 2 ox) 1) sy (- (* 2 oy) 1) sz (- (* 2 oz) 1)]
    (cond
      (not (zero? nx)) (ao-level (oc bx (+ by sy) bz) (oc bx by (+ bz sz)) (oc bx (+ by sy) (+ bz sz)))
      (not (zero? ny)) (ao-level (oc (+ bx sx) by bz) (oc bx by (+ bz sz)) (oc (+ bx sx) by (+ bz sz)))
      :else            (ao-level (oc (+ bx sx) by bz) (oc bx (+ by sy) bz) (oc (+ bx sx) (+ by sy) bz)))))

(defn mesh-region
  "Textured face mesh of blocks in [x0,x1)×[0,WY)×[z0,z1) — neighbours are read across the
   box boundary (from the full world) so seam culling + AO are correct between chunks.
   Vertex = pos3 uv2 layer shade light ao (valley.tex/STRIDE = 36). {:verts :indices}.
   `skylight` is precomputed (global). The block array + dims are hoisted into a fast
   primitive reader so the AO inner loop does no map lookups."
  ([world skylight x0 z0 x1 z1] (mesh-region world skylight x0 z0 x1 z1 0 0))
  ([world skylight x0 z0 x1 z1 pxo pzo]
   (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world)) WXZ (* WX WZ)
         x0 (long x0) z0 (long z0) x1 (long x1) z1 (long z1) pxo (long pxo) pzo (long pzo)
         b   #?(:clj ^ints (:blocks world) :cljs (:blocks world))
         blk (fn [x y z] (let [x (long x) y (long y) z (long z)]
                           (if (and (>= x 0) (< x WX) (>= y 0) (< y WY) (>= z 0) (< z WZ))
                             (aget b (+ x (* z WX) (* y WXZ))) 0)))
         oc  (fn [x y z] (opaque? (blk x y z)))
         vb (volatile! (transient [])) ib (volatile! (transient [])) vc (volatile! 0)
         ;; water faces go in a separate sub-mesh (pos3 only) for the transparent water pass
         wvb (volatile! (transient [])) wib (volatile! (transient [])) wvc (volatile! 0)]
     (doseq [x (range x0 x1) y (range WY) z (range z0 z1)]
       (let [id (blk x y z)]
         (when (pos? id)
           (dotimes [fi 6]
             (let [[[nx ny nz] corners] (nth faces fi)
                   nid (blk (+ x nx) (+ y ny) (+ z nz))]
              ;; transparency-aware: show the face if the neighbour doesn't occlude
              ;; (air or water), but cull water↔water interiors.
               (when (and (not (opaque? nid)) (not (and (= id WATER) (= nid WATER))))
                 (if (= id WATER)
                   ;; water face: position only → water sub-mesh
                   (let [base @wvc]
                     (dotimes [ci 4]
                       (let [[ox oy oz] (nth corners ci)]
                         (vswap! wvb (fn [t] (reduce conj! t [(double (+ x ox pxo)) (double (+ y oy)) (double (+ z oz pzo))])))))
                     (vswap! wib (fn [t] (reduce conj! t [base (+ base 1) (+ base 2) base (+ base 2) (+ base 3)])))
                     (vswap! wvc + 4))
                   ;; opaque face: full textured/lit/AO vertex → terrain sub-mesh
                   (let [bf    (nth face->bf fi)
                         layer (double (tex/face-layer id bf))
                         shade (double (nth tex/face-shade bf))
                         lit   (pow* (/ (double (light-at skylight (+ x nx) (+ y ny) (+ z nz) WX WY WZ)) 15.0) 1.6)
                         base  @vc]
                     (dotimes [ci 4]
                       (let [[ox oy oz] (nth corners ci) [u v] (nth face-uv ci)
                             ao (double (nth ao-mult (corner-ao oc x y z nx ny nz ox oy oz)))]
                         (vswap! vb (fn [t] (reduce conj! t [(double (+ x ox pxo)) (double (+ y oy)) (double (+ z oz pzo))
                                                             u v layer shade lit ao])))))
                     (vswap! ib (fn [t] (reduce conj! t [base (+ base 1) (+ base 2) base (+ base 2) (+ base 3)])))
                     (vswap! vc + 4)))))))))
     {:opaque {:verts (persistent! @vb)  :indices (persistent! @ib)}
      :water  {:verts (persistent! @wvb) :indices (persistent! @wib)}})))

(defn tris
  "Expand an indexed mesh ({:verts :indices}, fpv floats/vertex) into a flat triangle-list
   vertex seq for the editable dynamic world mesh (re-uploaded on block edits — avoids
   re-allocating an indexed GPU buffer per edit)."
  [{:keys [verts indices]} fpv]
  (let [v (vec verts) fpv (long fpv)]
    (persistent!
     (reduce (fn [out i] (let [b (* (long i) fpv)]
                           (reduce (fn [o k] (conj! o (nth v (+ b k)))) out (range fpv))))
             (transient []) indices))))

;; --- per-column-chunk meshing (keystone for streaming + cheap edits) -----------------
;; A "column chunk" is the CS×WY×CS stack at chunk coord (cx,cz). Re-meshing one column on a
;; block edit is ~1/Ncols the cost of a whole-world re-mesh.
(defn chunk-key
  "Chunk coord [cx cz] containing world column (x,z)."
  [x z] [(long (#?(:clj Math/floor :cljs js/Math.floor) (/ (double x) CS)))
         (long (#?(:clj Math/floor :cljs js/Math.floor) (/ (double z) CS)))])

;; per-column GPU buffer capacity (verts). Avg surface column ≈ 8k; 32k is a safe ceiling.
(def ^:const MAX-COLUMN-VERTS 32768)

(defn edit-columns
  "Set of column keys whose mesh must be rebuilt after a block edit at world (x,z): the
   block's own column plus any horizontal neighbour column (an edge edit changes the
   adjacent column's seam faces). 1–3 columns in practice."
  [x z]
  (distinct (map (fn [[dx dz]] (chunk-key (+ (long x) dx) (+ (long z) dz)))
                 [[0 0] [1 0] [-1 0] [0 1] [0 -1]])))

(defn mesh-stream-column
  "Column (cx,cz) meshed from the active buffer `s` in ABSOLUTE world coords (so meshes never
   move on recentre). Returns {:opaque <tris 9 floats/vert> :water <tris 3 floats/vert>}."
  [s cx cz]
  (let [aox (long (:aox s)) aoz (long (:aoz s))
        bx0 (- (* (long cx) CS) aox) bz0 (- (* (long cz) CS) aoz)
        mr (mesh-region s (:light s) bx0 bz0 (+ bx0 CS) (+ bz0 CS) aox aoz)]
    {:opaque (tris (:opaque mr) 9) :water (tris (:water mr) 3)}))

(defn ring-cols
  "Column keys [cx cz] within Chebyshev radius `r` of chunk (pcx,pcz) (the mesh ring). Must be
   ≤ the stream's data radius (rg) so every meshed column has its neighbours loaded."
  [pcx pcz r]
  (for [cx (range (- (long pcx) r) (+ (long pcx) r 1))
        cz (range (- (long pcz) r) (+ (long pcz) r 1))] [cx cz]))
