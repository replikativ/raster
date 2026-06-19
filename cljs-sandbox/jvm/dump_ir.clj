;; Exploration II (step 2): dump saxpy post-pass IR as JS-friendly JSON
;; (preserving :raster.op/original + :raster.type/tag), plus a JVM reference
;; result, so a node translator can emit wasm and diff against it.
(require '[raster.core :refer [deftm]] '[raster.numeric] '[raster.arrays]
         '[raster.compiler.pipeline :as pl] '[clojure.string :as str])

(deftm saxpy! [a :- Double, x :- (Array double), y :- (Array double), n :- Long] :- Long
  (loop [i 0]
    (if (clojure.core/< (long i) (long n))
      (do (raster.arrays/aset! y (long i)
            (raster.numeric/+ (raster.numeric/* a (raster.arrays/aget x i))
                              (raster.arrays/aget y i)))
          (recur (clojure.core/inc (long i))))
      n)))

(defn post-pass-ir [v]
  (let [gwb (resolve 'raster.compiler.pipeline/get-walked-body)
        bpe (resolve 'raster.compiler.pipeline/build-param-env)
        cp  (resolve 'raster.compiler.pipeline/clean-params)
        gp  (resolve 'raster.compiler.pipeline/get-params)]
    (pl/run-passes (first (gwb v :double)) pl/forward-passes
      {:inline? true :simd? false :dtype :double
       :active-params (cp (gp v :double)) :param-env (bpe v :double)
       :source-ns (.ns ^clojure.lang.Var v)})))

;; --- IR S-expr -> plain Clojure JSON tree (preserve op/tag metadata) ---
(defn enc [x]
  (cond
    (seq? x)    (let [m (meta x)]
                  (cond-> {"node" "call" "head" (str (first x))
                           "args" (mapv enc (rest x))}
                    (:raster.op/original m)  (assoc "op"  (str (:raster.op/original m)))
                    (:raster.type/tag m)     (assoc "tag" (str (:raster.type/tag m)))))
    (vector? x) {"node" "vec" "items" (mapv enc x)}
    (symbol? x) {"node" "sym" "name" (str x)}
    (integer? x){"node" "int" "v" x}
    (float? x)  {"node" "float" "v" (double x)}
    (number? x) {"node" "float" "v" (double x)}
    :else       {"node" "lit" "v" (str x)}))

;; --- minimal JSON writer (no dep) ---
(defn jw [x]
  (cond
    (map? x)    (str "{" (str/join "," (map (fn [[k v]] (str (jw k) ":" (jw v))) x)) "}")
    (vector? x) (str "[" (str/join "," (map jw x)) "]")
    (string? x) (str \" (str/escape x {\" "\\\"" \\ "\\\\" \newline "\\n"}) \")
    (keyword? x)(jw (name x))
    (nil? x)    "null"
    :else       (str x)))

;; Run from the raster project root:
;;   clojure -M -e '(load-file "cljs-sandbox/jvm/dump_ir.clj")'
(let [ir (post-pass-ir #'saxpy!)]
  (spit "cljs-sandbox/jvm/ir_saxpy.sample.json" (jw (enc ir)))
  (println "wrote cljs-sandbox/jvm/ir_saxpy.sample.json"))

;; NOTE: compile-aot / lazy-JIT of this exact saxpy! kernel currently throws
;; "Cannot resolve class: raster.arrays" at bytecode.clj:2371
;; (emit-java-static-call) — a JVM-backend bug in raster.arrays/aset! emission
;; for this kernel shape, orthogonal to the wasm exploration. The IR itself
;; lowers fine (above); ir_to_wasm.mjs diffs the emitted wasm against the saxpy
;; oracle instead of the JVM result.
