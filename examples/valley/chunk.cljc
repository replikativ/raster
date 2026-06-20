(ns valley.chunk
  "Portable single-chunk model for the valley walk slice (Phase 2b): a 16³ block
   array generated from valley.kernels/terrain-height, a face mesh of it, and the
   block-property table the physics kernels read. Cross-platform array helpers keep
   the JVM (Java primitive arrays) and browser (typed arrays in wasm memory) on the
   same code. The player physics (integrate-physics!) runs against this chunk via
   the valley.kernels facade."
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
;; 10 water (transparent, non-solid), 11 log, 12 leaves.
(def ^:const WATER 10)
(def ^:const LOG 11)
(def ^:const LEAVES 12)
(def ^:private solid-ids [1 2 3 4 5 6 7 8 9 11 12])   ; everything except air + water
(defn solid-table []
  (let [s (barray 62)]
    (doseq [i solid-ids] (aset s (int i) #?(:clj (byte 1) :cljs 1)))
    s))
;; opaque = occludes neighbour faces (air + water are see-through for the mesher)
(defn opaque? [id] (and (pos? (long id)) (not= (long id) WATER)))

(defn gen-chunk
  "A self-contained 16³ chunk: per column the surface height is terrain-height
   remapped into 4..13 (so the terrain fits inside one chunk), filled grass/dirt/
   stone below. Returns the int block array."
  []
  (let [b (iarray (* CS CS CS))]
    (dotimes [x CS]
      (dotimes [z CS]
        (let [h (+ 4 (mod (k/terrain-height x z) 10))]   ; 4..13
          (dotimes [y CS]
            (when (< y h)
              (aset b (idx x y z)
                    (cond (= y (dec h)) 1 (>= y (- h 3)) 2 :else 3)))))))
    b))

(defn block-at ^long [blocks ^long x ^long y ^long z]
  (if (and (>= x 0) (< x CS) (>= y 0) (< y CS) (>= z 0) (< z CS))
    (long (aget blocks (idx x y z)))
    0))

;; --- face mesh of the chunk (interior faces culled) --------------------------
(def ^:private faces
  [[[0 0 1]  [[0 0 1] [1 0 1] [1 1 1] [0 1 1]]]
   [[0 0 -1] [[1 0 0] [0 0 0] [0 1 0] [1 1 0]]]
   [[1 0 0]  [[1 0 1] [1 0 0] [1 1 0] [1 1 1]]]
   [[-1 0 0] [[0 0 0] [0 0 1] [0 1 1] [0 1 0]]]
   [[0 1 0]  [[0 1 1] [1 1 1] [1 1 0] [0 1 0]]]
   [[0 -1 0] [[0 0 0] [1 0 0] [1 0 1] [0 0 1]]]])
(def ^:private face-uv [[0.0 0.0] [1.0 0.0] [1.0 1.0] [0.0 1.0]])

;; --- multi-chunk world (option C): a flat block grid spanning many chunks --------
;; The physics kernel only ever sees ONE 16³ chunk; for a world larger than that we
;; stitch a player-centred 16³ window each frame (the tiny player AABB can never
;; leave a centred window) and call the kernel against it at origin 0. This keeps the
;; wasm kernel + marshaling identical to the single-chunk path — only the host stitch
;; and a pos translation differ. World coords index a flat array: wx + wz*WX + wy*WX*WZ.

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

(defn gen-world
  "A flat cw×ch×cd-chunk world generated from valley.kernels/terrain-height over
   continuous world coords (so terrain flows across chunk seams). Caves carved by
   has-cave?, lakes filled to sea level, oak-like trees scattered on grassy biomes.
   Returns {:blocks :wx :wy :wz :solid}."
  [^long cw ^long ch ^long cd]
  (let [wx (* cw CS) wy (* ch CS) wz (* cd CS)
        b  (iarray (* wx wy wz))
        base 4
        sea  (+ base 3)                       ; lakes fill low columns up to here
        raw   (vec (for [x (range wx) z (range wz)] (long (k/surface-height-biome x z))))
        biome (vec (for [x (range wx) z (range wz)] (long (k/biome-index (* x 8) (* z 8)))))
        minh  (reduce min raw)
        hs    (vec (for [ci (range (* wx wz))] (max base (min (dec wy) (+ base (- (nth raw ci) minh))))))
        inb?  (fn [x y z] (and (>= x 0) (< x wx) (>= y 0) (< y wy) (>= z 0) (< z wz)))
        put   (fn [x y z id] (when (inb? x y z) (aset b (world-idx x y z wx wz) (int id))))
        airat? (fn [x y z] (or (not (inb? x y z))
                               (zero? (aget #?(:clj ^ints b :cljs b) (world-idx x y z wx wz)))))]
    ;; --- terrain + caves ---
    (dotimes [x wx]
      (dotimes [z wz]
        (let [ci   (+ (* x wz) z)
              h    (long (nth hs ci))
              bi   (nth biome ci)
              surf (nth biome-surface bi)
              fill (nth biome-fill bi)]
          (dotimes [y wy]
            (when (and (< y h)
                       (not (and (> y 1) (< y (- h 1)) (= 1 (long (k/has-cave? x y z))))))
              (aset b (world-idx x y z wx wz)
                    (int (cond (= y (dec h)) surf (>= y (- h 3)) fill :else 3)))))
          ;; --- lakes: fill air above terrain up to sea level ---
          (when (< h sea)
            (loop [y h] (when (< y sea) (put x y z WATER) (recur (inc y))))))))
    ;; --- trees: oak-like, only on dry grassy columns ---
    (dotimes [x wx]
      (dotimes [z wz]
        (let [ci (+ (* x wz) z) h (long (nth hs ci)) bi (nth biome ci)]
          (when (and (>= h sea) (= 1 (long (nth biome-surface bi)))   ; grass surface, above water
                     (zero? (mod (phash x z) 37))
                     (< (+ h 7) wy))
            (let [th (+ 4 (mod (phash x z) 3))
                  top (+ h th)]
              (dotimes [t th] (put x (+ h t) z LOG))                  ; trunk
              (doseq [dy [-2 -1]]                                     ; lower canopy 5×5 (skip far corners)
                (let [ly (+ top dy -1)]
                  (doseq [dx [-2 -1 0 1 2] dz [-2 -1 0 1 2]]
                    (when (and (not (and (= (iabs dx) 2) (= (iabs dz) 2))) (airat? (+ x dx) ly (+ z dz)))
                      (put (+ x dx) ly (+ z dz) LEAVES)))))
              (doseq [dy [0 1]]                                       ; upper canopy 3×3
                (let [ly (+ top dy -1)]
                  (doseq [dx [-1 0 1] dz [-1 0 1]]
                    (when (airat? (+ x dx) ly (+ z dz)) (put (+ x dx) ly (+ z dz) LEAVES))))))))))
    {:blocks b :wx wx :wy wy :wz wz :solid (solid-table)}))

(defn gen-flat-world
  "A flat cw×ch×cd-chunk world: solid up to `surface` (grass top, dirt, stone),
   walkable across all chunk seams without step-up. The walker milestone uses this;
   gen-world (real terrain) becomes walkable once the physics gains 1-block step-up."
  [cw ch cd surface]
  (let [wx (* cw CS) wy (* ch CS) wz (* cd CS)
        b  (iarray (* wx wy wz)) s (long surface)]
    (dotimes [x wx]
      (dotimes [z wz]
        (dotimes [y (min s wy)]
          (aset b (world-idx x y z wx wz)
                (cond (= y (dec s)) 1 (>= y (- s 3)) 2 :else 3)))))
    {:blocks b :wx wx :wy wy :wz wz :solid (solid-table)}))

(defn world-block ^long [world ^long wx ^long wy ^long wz]
  (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world))]
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
        wx (long wx) wy (long wy) wz (long wz)]
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
        wox (long wox) woy (long woy) woz (long woz)
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

(defn mesh-chunk
  "{:verts [floats] :indices [ints]} face mesh of the chunk."
  [blocks]
  (let [vb (volatile! (transient [])) ib (volatile! (transient [])) vc (volatile! 0)]
    (doseq [x (range CS) y (range CS) z (range CS)]
      (let [id (block-at blocks x y z)]
        (when (pos? id)
          (let [layer (double (dec id))]
            (doseq [[[nx ny nz] corners] faces]
              (when (zero? (block-at blocks (+ x nx) (+ y ny) (+ z nz)))
                (let [base @vc]
                  (dotimes [ci 4]
                    (let [[ox oy oz] (nth corners ci) [u v] (nth face-uv ci)]
                      (vswap! vb (fn [t] (reduce conj! t [(double (+ x ox)) (double (+ y oy)) (double (+ z oz)) u v layer])))))
                  (vswap! ib (fn [t] (reduce conj! t [base (+ base 1) (+ base 2) base (+ base 2) (+ base 3)])))
                  (vswap! vc + 4))))))))
    {:verts (persistent! @vb) :indices (persistent! @ib)}))

;; --- lighting (port of valley.lighting skylight + valley.mesher AO) ------------------
(defn- pow* [a b] (#?(:clj Math/pow :cljs js/Math.pow) (double a) (double b)))

(defn- light-at
  "Skylight at world cell (clamped to 0 outside)."
  [skylight x y z WX WY WZ]                ; >4 args ⇒ no primitive hints
  (let [x (long x) y (long y) z (long z) WX (long WX) WY (long WY) WZ (long WZ)]
    (if (and (>= x 0) (< x WX) (>= y 0) (< y WY) (>= z 0) (< z WZ))
      (long (aget #?(:clj ^ints skylight :cljs skylight) (world-idx x y z WX WZ)))
      0)))

(defn compute-skylight
  "Per-cell skylight 0..15: 15 in sky-open air, decaying under solids and bleeding
   sideways (8 relaxation passes ≈ flood-fill for unit decay). Solids stay 0; a face reads
   the light of the adjacent air cell. One-time at gen (cheap region recompute on edits)."
  [world]
  (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world))
        b     #?(:clj ^ints (:blocks world)     :cljs (:blocks world))
        light #?(:clj ^ints (iarray (* WX WY WZ)) :cljs (iarray (* WX WY WZ)))]
    ;; top-down: full sun until the first solid in each column
    (dotimes [x WX]
      (dotimes [z WZ]
        (loop [y (dec WY) lit true]
          (when (>= y 0)
            (let [i (world-idx x y z WX WZ) air (zero? (aget b i))]
              (if (and lit air)
                (do (aset light i #?(:clj (int 15) :cljs 15)) (recur (dec y) true))
                (recur (dec y) false)))))))
    ;; horizontal/under-overhang bleed
    (dotimes [_ 8]
      (dotimes [y WY] (dotimes [z WZ] (dotimes [x WX]
                                        (let [i (world-idx x y z WX WZ)]
                                          (when (zero? (aget b i))
                                            (let [l (long (aget light i))
                                                  n (max (light-at light (inc x) y z WX WY WZ) (light-at light (dec x) y z WX WY WZ)
                                                         (light-at light x (inc y) z WX WY WZ) (light-at light x (dec y) z WX WY WZ)
                                                         (light-at light x y (inc z) WX WY WZ) (light-at light x y (dec z) WX WY WZ))
                                                  m (max l (dec (long n)))]
                                              (when (> m l) (aset light i #?(:clj (int m) :cljs m))))))))))
    light))

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

(defn mesh-world
  "Textured face mesh of a multi-chunk world: per-face texture layer (valley.tex) +
   directional shade + baked skylight (per face) + ambient occlusion (per vertex).
   Interior faces culled across seams. Vertex = pos3 uv2 layer shade light ao
   (valley.tex/STRIDE = 36). {:verts :indices}. 1-arg computes skylight; 2-arg reuses a
   precomputed one (so block edits skip the global light pass). The block array + dims are
   hoisted into a fast primitive reader so the AO inner loop does no map lookups."
  ([world] (mesh-world world (compute-skylight world)))
  ([world skylight]
   (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world)) WXZ (* WX WZ)
         b   #?(:clj ^ints (:blocks world) :cljs (:blocks world))
         blk (fn [x y z] (let [x (long x) y (long y) z (long z)]
                           (if (and (>= x 0) (< x WX) (>= y 0) (< y WY) (>= z 0) (< z WZ))
                             (aget b (+ x (* z WX) (* y WXZ))) 0)))
         oc  (fn [x y z] (opaque? (blk x y z)))
         vb (volatile! (transient [])) ib (volatile! (transient [])) vc (volatile! 0)]
     (doseq [x (range WX) y (range WY) z (range WZ)]
       (let [id (blk x y z)]
         (when (pos? id)
           (dotimes [fi 6]
             (let [[[nx ny nz] corners] (nth faces fi)
                   nid (blk (+ x nx) (+ y ny) (+ z nz))]
              ;; transparency-aware: show the face if the neighbour doesn't occlude
              ;; (air or water), but cull water↔water interiors.
               (when (and (not (opaque? nid)) (not (and (= id WATER) (= nid WATER))))
                 (let [bf    (nth face->bf fi)
                       layer (double (tex/face-layer id bf))
                       shade (double (nth tex/face-shade bf))
                       lit   (pow* (/ (double (light-at skylight (+ x nx) (+ y ny) (+ z nz) WX WY WZ)) 15.0) 1.6)
                       base  @vc]
                   (dotimes [ci 4]
                     (let [[ox oy oz] (nth corners ci) [u v] (nth face-uv ci)
                           ao (double (nth ao-mult (corner-ao oc x y z nx ny nz ox oy oz)))]
                       (vswap! vb (fn [t] (reduce conj! t [(double (+ x ox)) (double (+ y oy)) (double (+ z oz))
                                                           u v layer shade lit ao])))))
                   (vswap! ib (fn [t] (reduce conj! t [base (+ base 1) (+ base 2) base (+ base 2) (+ base 3)])))
                   (vswap! vc + 4))))))))
     {:verts (persistent! @vb) :indices (persistent! @ib)})))

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
