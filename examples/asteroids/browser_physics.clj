(ns asteroids.browser-physics
  "Batch physics kernels for the browser build of Geometric Asteroids.

   The desktop game (core.clj / shapes.clj) uses per-shape `defvalue` typed
   dispatch + a Vulkan renderer. In the browser the rendering, input and the
   split hierarchy live in JS/Canvas2D (cljs-sandbox/asteroids), but the
   numerical hot loop — integrating every shape's position and angle each
   frame — is a raster `deftm` compiled to WebAssembly via `compile-wasm`.

   Shapes are stored Struct-of-Arrays in wasm linear memory (xs, ys, vxs, vys,
   angles, spins). `move-shapes!` integrates position+angle and applies the
   toroidal screen wrap entirely in wasm via `mod` (floored), so the whole of
   the desktop game's `move-shape` — including the wrap — lives in the compiled
   kernel with no JS fix-up. (The wrap uses `mod`, which has no SIMD lane op, so
   this kernel runs scalar; the simpler integrate-only maps still vectorize.)"
  (:require [raster.core :refer [deftm]]
            [raster.numeric]
            [raster.arrays]))

(deftm move-shapes!
  [xs :- (Array double), ys :- (Array double),
   vxs :- (Array double), vys :- (Array double),
   angs :- (Array double), spins :- (Array double),
   n :- Long, dt :- Double, w :- Double, h :- Double] :- nil
  (dotimes [i n]
    (raster.arrays/aset xs i
                        (clojure.core/mod (raster.numeric/+ (raster.arrays/aget xs i)
                                                            (raster.numeric/* (raster.arrays/aget vxs i) dt)) w))
    (raster.arrays/aset ys i
                        (clojure.core/mod (raster.numeric/+ (raster.arrays/aget ys i)
                                                            (raster.numeric/* (raster.arrays/aget vys i) dt)) h))
    (raster.arrays/aset angs i
                        (raster.numeric/+ (raster.arrays/aget angs i)
                                          (raster.numeric/* (raster.arrays/aget spins i) dt)))))

(deftm move-bullets!
  "Integrate bullet positions (no rotation) — same elementwise shape."
  [xs :- (Array double), ys :- (Array double),
   vxs :- (Array double), vys :- (Array double),
   n :- Long, dt :- Double] :- nil
  (dotimes [i n]
    (raster.arrays/aset xs i
                        (raster.numeric/+ (raster.arrays/aget xs i)
                                          (raster.numeric/* (raster.arrays/aget vxs i) dt)))
    (raster.arrays/aset ys i
                        (raster.numeric/+ (raster.arrays/aget ys i)
                                          (raster.numeric/* (raster.arrays/aget vys i) dt)))))
