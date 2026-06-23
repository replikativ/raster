(ns valley.kernels
  "Cross-platform kernel facade for the valley browser/JVM slice — the valley analog
   of voxel.kernels. Wraps valley.core's array-param deftm kernels behind scalar
   entry points with the permutation tables + biome params baked in, so the portable
   valley.chunk (.cljc) calls a uniform `(surface-height-biome x z)` on both platforms.

   JVM: this file — calls the deftm kernels directly with JVM int-array perms.
   Browser: a generated valley/kernels.cljs (same ns) backs the same fns with the
   wasm module (perms written into wasm memory at init). See valley/gen.clj."
  (:require [valley.core :as vc]
            [raster.noise :as noise]))

;; Deterministic permutation tables (seeded once). The cljs shim bakes the SAME
;; perms into wasm memory so terrain is identical in both shells.
(def ^:private hperm (noise/make-perm 1337))
(def ^:private dperm (noise/make-perm 9001))
(def ^:private cperm (noise/make-perm 4242))
;; biome noise perms + per-biome height-scale/base-offset, indexed by biome-index 0..10
;; (0 desert .. 9 mountains 10 mushroom; mountains get the tallest scale/offset).
(def ^:private tperm (noise/make-perm 2222))
(def ^:private uperm (noise/make-perm 3333))
(def ^:private mperm (noise/make-perm 6666))
(def ^:private SCALES  (double-array [8 12 16 20 28 6 20 10 18 48 12]))
(def ^:private OFFSETS (double-array [-4 0 0 2 4 -6 3 -2 2 15 0]))

(defn has-cave?
  "1 if (x,y,z) is carved out by cave noise."
  ^long [x y z]
  (vc/has-cave? cperm (double x) (double y) (double z)))

(defn surface-height-biome
  "Biome-aware surface height at world column (x,z)."
  ^long [x z]
  (vc/surface-height-biome hperm dperm tperm uperm mperm SCALES OFFSETS (double x) (double z)))

(defn biome-index
  "Biome 0..10 at world column (x,z)."
  ^long [x z]
  (vc/biome-index tperm uperm dperm mperm (double x) (double z)))

;; Player physics against one chunk. JVM: direct primitive-array call (zero-copy).
;; The cljs counterpart (generated kernels.cljs) marshals pos/vel/blocks/solid
;; through wasm memory and reads back the mutated pos/vel. Mutates pos + vel,
;; returns 1 if on ground.
(defn integrate-physics!
  [pos vel blocks solid cx cy cz hw h dx dz dt]   ; >4 args → no primitive hint
  (vc/integrate-physics! pos vel blocks solid cx cy cz hw h dx dz dt))

;; Batch mob physics. JVM: the world block array is used directly (zero-copy), so
;; upload-world! is a no-op; the browser shim uploads the resident world once.
(defn upload-world! [_blocks _solid] nil)

(defn integrate-physics-batch!
  [pos vel dxs dzs blocks solid n wxd wyd wzd hw h dt]
  (vc/integrate-physics-batch! pos vel dxs dzs blocks solid n wxd wyd wzd hw h dt))
