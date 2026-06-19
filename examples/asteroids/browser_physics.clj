(ns asteroids.browser-physics
  "Batch physics kernels for the browser build of Geometric Asteroids.

   The desktop game (core.clj / shapes.clj) uses per-shape `defvalue` typed
   dispatch + a Vulkan renderer. In the browser the rendering, input and the
   split hierarchy live in JS/Canvas2D (cljs-sandbox/asteroids), but the
   numerical hot loop — integrating every shape's position and angle each
   frame — is a raster `deftm` compiled to WebAssembly via `compile-wasm`.

   Shapes are stored Struct-of-Arrays in wasm linear memory (xs, ys, vxs, vys,
   angles, spins), so `move-shapes!` is an elementwise counted map: it compiles
   to a clean wasm loop and SIMD-vectorizes (f64x2) under `:wasm-simd? true`.
   Wrap-around (toroidal screen) is applied in JS after the integrate step."
  (:require [raster.core :refer [deftm]]
            [raster.numeric]
            [raster.arrays]))

(deftm move-shapes!
  [xs :- (Array double), ys :- (Array double),
   vxs :- (Array double), vys :- (Array double),
   angs :- (Array double), spins :- (Array double),
   n :- Long, dt :- Double] :- nil
  (dotimes [i n]
    (raster.arrays/aset xs i
                        (raster.numeric/+ (raster.arrays/aget xs i)
                                          (raster.numeric/* (raster.arrays/aget vxs i) dt)))
    (raster.arrays/aset ys i
                        (raster.numeric/+ (raster.arrays/aget ys i)
                                          (raster.numeric/* (raster.arrays/aget vys i) dt)))
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
