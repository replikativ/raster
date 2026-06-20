;; Compile valley's kernels to one wasm module + generate the cljs shim (valley.kernels)
;; via the general, declarative raster.compiler.cljs-emit (no hand-rolled strings, no
;; hand-computed memory offsets — emit! lays the regions out and marshals arrays through
;; the (ptr,len) ABI). Run from the repo root:
;;   clojure -Sdeps '{:paths ["src" "examples"]}' -M examples/valley/gen.clj
;; Outputs:
;;   examples/asteroids/web/public/valley_kernels.wasm
;;   examples/asteroids/web/src/valley/kernels.cljs   (same ns as kernels.clj)
(require '[raster.compiler.cljs-emit :as cljs-emit]
         '[raster.noise :as noise]
         '[clojure.java.io :as io])
(require 'valley.core)

(def web-dir "examples/asteroids/web")
;; perm tables — must match examples/valley/kernels.clj (the JVM facade)
(def HP (noise/make-perm 1337))
(def DP (noise/make-perm 9001))
(def CP (noise/make-perm 4242))
(def HS 18.0) (def BO -4.0)

(def K 1024)   ; CAP mobs for the batch columns

(let [res (cljs-emit/emit!
           {:ns        'valley.kernels
            :out-dir   (str web-dir "/src")
            :wasm-name "valley_kernels"
            ;; baked-at-init constant tables (written to memory by the generated init!)
            :consts   [{:sym 'HP :data (seq HP) :view :i32}
                       {:sym 'DP :data (seq DP) :view :i32}
                       {:sym 'CP :data (seq CP) :view :i32}]
            ;; per-call marshal buffers (single-player + batch mob columns)
            :scratch  [{:sym 'BLK  :view :i32 :bytes (* 4 16 16 16)}
                       {:sym 'SOL  :view :i8  :bytes 64}
                       {:sym 'POS  :view :f64 :bytes 24}
                       {:sym 'VEL  :view :f64 :bytes 24}
                       {:sym 'MPOS :view :f64 :bytes (* 8 3 K)}
                       {:sym 'MVEL :view :f64 :bytes (* 8 3 K)}
                       {:sym 'MDXS :view :f64 :bytes (* 8 K)}
                       {:sym 'MDZS :view :f64 :bytes (* 8 K)}]
            ;; uploaded once (the big resident world block array)
            :resident [{:sym 'WBLK :view :i32 :bytes (* 4 64 64 64)}
                       {:sym 'WSOL :view :i8  :bytes 64}]
            :uploads  [{:fn "upload-world!" :args '[blocks solid]
                        :set '[[WBLK blocks] [WSOL solid]]}]
            :kernels
            [{:var #'valley.core/terrain-height :export "terrain_height" :fn "terrain-height"
              :args '[x z]
              :call [[:const 'HP] [:const 'DP] 'x 'z [:lit HS] [:lit BO]]}
             {:var #'valley.core/has-cave? :export "has_cave" :fn "has-cave?"
              :args '[x y z]
              :call [[:const 'CP] 'x 'y 'z]}
             {:var #'valley.core/integrate-physics! :export "integrate_physics" :fn "integrate-physics!"
              :args '[pos vel blocks solid cx cy cz hw h dx dz dt]
              :call [[:inout 'POS 'pos] [:inout 'VEL 'vel] [:in 'BLK 'blocks] [:in 'SOL 'solid]
                     'cx 'cy 'cz 'hw 'h 'dx 'dz 'dt]}
             {:var #'valley.core/integrate-physics-batch! :export "batch" :fn "integrate-physics-batch!"
              :args '[pos vel dxs dzs blocks solid n wxd wyd wzd hw h dt]
              :call [[:inout 'MPOS 'pos] [:inout 'MVEL 'vel] [:in 'MDXS 'dxs] [:in 'MDZS 'dzs]
                     [:resident 'WBLK] [:resident 'WSOL] 'n 'wxd 'wyd 'wzd 'hw 'h 'dt]}]})
      src-wasm  (io/file (str web-dir "/src") "valley_kernels.wasm")
      dest-wasm (io/file web-dir "public" "valley_kernels.wasm")]
  (io/make-parents dest-wasm)
  (io/copy src-wasm dest-wasm)
  (.delete src-wasm)
  (println "wrote" (:cljs-file res))
  (println "wrote" (.getPath dest-wasm) (count (:exports res)) "exports")
  (println "layout" (into (sorted-map) (map (fn [[k v]] [k (:off v)]) (:layout res)))))

(shutdown-agents)
