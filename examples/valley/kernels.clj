(ns valley.kernels
  "Cross-platform kernel facade for the valley browser/JVM slice — the valley analog
   of voxel.kernels. Wraps valley.core's array-param deftm kernels behind scalar
   entry points with the permutation tables + biome params baked in, so the portable
   valley.world (.cljc) calls a uniform `(terrain-height x z)` on both platforms.

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

;; Biome height shaping for the slice (a single rolling-hills profile for now;
;; full per-biome params come with the biome pass in a later phase).
(def ^:const HEIGHT-SCALE 18.0)
(def ^:const BASE-OFFSET -4.0)

(defn terrain-height
  "Surface block height at world column (x,z)."
  ^long [x z]
  (vc/terrain-height hperm dperm (double x) (double z) HEIGHT-SCALE BASE-OFFSET))

(defn has-cave?
  "1 if (x,y,z) is carved out by cave noise."
  ^long [x y z]
  (vc/has-cave? cperm (double x) (double y) (double z)))
