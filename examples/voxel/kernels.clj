(ns voxel.kernels
  "Voxel terrain — the numerical kernel layer (the valley-shaped slice).

   `terrain-height` is a `deftm`: layered sine noise giving a surface height for a
   world column. On the JVM it's typed-dispatch/bytecode; in the browser
   `raster.compiler.cljs-emit` compiles it to WebAssembly (exercising the wasm
   sin/cos lowerings) and generates a cljs counterpart, so voxel.world's
   `(require [voxel.kernels])` gets the right impl per platform — exactly the
   asteroids pattern, but for terrain."
  (:require [raster.core :refer [deftm]]
            [raster.numeric]
            [raster.math]))

(deftm terrain-height
  "Surface height (world Y) at column (x, z), from summed sine octaves."
  [x :- Double, z :- Double] :- Double
  (let [a (raster.math/sin (raster.numeric/* x 0.18))
        b (raster.math/cos (raster.numeric/* z 0.22))
        c (raster.math/sin (raster.numeric/* (raster.numeric/+ x z) 0.09))
        d (raster.math/cos (raster.numeric/* (raster.numeric/- x z) 0.13))]
    (raster.numeric/+ (raster.numeric/* 3.5 a)
                      (raster.numeric/+ (raster.numeric/* 3.0 b)
                                        (raster.numeric/+ (raster.numeric/* 2.5 c)
                                                          (raster.numeric/* 1.5 d))))))
