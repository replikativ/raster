;; Compile voxel.kernels/terrain-height to wasm + generate the cljs counterpart.
;; Run from the repo root:
;;   clojure -Sdeps '{:paths ["src" "examples"]}' -M examples/voxel/gen.clj
;; Outputs into the shared asteroids/web playground build:
;;   examples/asteroids/web/public/voxel_kernels.wasm
;;   examples/asteroids/web/src/voxel/kernels.cljs
(require '[raster.compiler.cljs-emit :as cljs-emit]
         '[clojure.java.io :as io])
(require 'voxel.kernels)

(def web-dir "examples/asteroids/web")
(def out-dir (str web-dir "/src"))

(let [res (cljs-emit/emit!
           {:ns      'voxel.kernels
            :kernels [{:var #'voxel.kernels/terrain-height
                       :export "terrain_height" :fn "terrain-height" :dtype :double}]
            :out-dir out-dir
            :wasm-name "voxel_kernels"})
      src-wasm  (io/file out-dir "voxel_kernels.wasm")
      dest-wasm (io/file web-dir "public" "voxel_kernels.wasm")]
  (io/make-parents dest-wasm)
  (io/copy src-wasm dest-wasm)
  (.delete src-wasm)
  (println "wrote" (:cljs-file res))
  (println "wrote" (.getPath dest-wasm)))

(shutdown-agents)
