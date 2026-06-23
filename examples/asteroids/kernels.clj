(ns asteroids.kernels
  "Cross-platform asteroids — the numerical kernel layer.

   `Shape` is the kinematic value type; `move-shape` is the per-shape physics
   (integrate position + angle, toroidal wrap) — ONE monomorphic `deftm` (the
   kinematics are type-independent; polygon identity lives in the glue as a
   `sides` tag). On the JVM this is the typed-dispatch/bytecode path; in the
   browser `raster.compiler.cljs-emit` compiles it to wasm and generates a cljs
   counterpart of this namespace (a `defrecord Shape` + a `move-shape` wrapper
   marshaling the record over the value-type wasm boundary). `game.cljc` requires
   `asteroids.kernels` and gets the right one per platform.

   Per-element + immutable (naive); SoA batching is the later performance path."
  (:require [raster.core :refer [deftm defvalue]]
            [raster.numeric]
            [raster.arrays]))

(defvalue Shape [x :- Double, y :- Double, vx :- Double, vy :- Double,
                 angle :- Double, spin :- Double, size :- Double])

(deftm move-shape [s :- Shape, dt :- Double, w :- Double, h :- Double] :- Shape
  (->Shape (clojure.core/mod (raster.numeric/+ (.x s) (raster.numeric/* (.vx s) dt)) w)
           (clojure.core/mod (raster.numeric/+ (.y s) (raster.numeric/* (.vy s) dt)) h)
           (.vx s) (.vy s)
           (raster.numeric/+ (.angle s) (raster.numeric/* (.spin s) dt))
           (.spin s) (.size s)))
