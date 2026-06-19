;; Regenerate the browser kernels for Geometric Asteroids from ../kernels.clj.
;;
;; Compiles the `deftm`/`defvalue` kernels in asteroids.kernels to ONE shared-
;; memory wasm module and (re)generates the cljs counterpart namespace. Run from
;; the repo root (needs raster's full deps + the examples on the path):
;;
;;   clojure -Sdeps '{:paths ["src" "examples"]}' -M examples/asteroids/web/gen.clj
;;
;; Outputs (committed so the playground builds without a JVM):
;;   examples/asteroids/web/public/kernels.wasm
;;   examples/asteroids/web/src/asteroids/kernels.cljs
(require '[raster.compiler.cljs-emit :as cljs-emit]
         '[clojure.java.io :as io])
(require 'asteroids.kernels)

(def web-dir "examples/asteroids/web")
(def out-dir (str web-dir "/src"))   ; cljs lands at src/asteroids/kernels.cljs

(let [res (cljs-emit/emit!
           {:ns      'asteroids.kernels
            :kernels [{:var      #'asteroids.kernels/move-shape
                       :export   "move_shape"
                       :fn       "move-shape"
                       :dtype    :double}]
            :out-dir out-dir
            :wasm-name "kernels"})
      ;; emit! writes <out-dir>/kernels.wasm — move it next to index.html
      src-wasm  (io/file out-dir "kernels.wasm")
      dest-wasm (io/file web-dir "public" "kernels.wasm")]
  (io/make-parents dest-wasm)
  (io/copy src-wasm dest-wasm)
  (.delete src-wasm)
  (println "wrote" (:cljs-file res))
  (println "wrote" (.getPath dest-wasm))
  (println "exports" (mapv :name (:exports res))))

(shutdown-agents)
