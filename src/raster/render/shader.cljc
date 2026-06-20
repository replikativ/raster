(ns raster.render.shader
  "One shader source → GLSL (Vulkan) and WGSL (WebGPU).

   Scene code carries ONE shader description, not a GLSL/WGSL pair. A description
   is plain data: a uniform struct (named typed fields), vertex attributes,
   varyings, and a restricted expression body for the vertex and fragment stages.
   Both backends compile from it — the JVM Vulkan backend via `->glsl`, the browser
   WebGPU backend via `->wgsl` — so there are no hand-written per-platform shaders.

   NDC handedness is the emitter's job, not the author's: write `set-position` in
   Vulkan-style clip space (Y-down); the WGSL emitter flips Y for WebGPU. The
   uniform becomes a Vulkan push-constant block / a WebGPU uniform buffer — the
   size and the stages that reference it are derived from the description.

   Supported expression grammar (deliberately small; extend as shaders need it):
     number                      f32 literal
     symbol                      ref to an attribute / varying / uniform field / let
     (+ a b) (- a b) (* a b) (/ a b)   scalar/vector arithmetic (broadcast ok)
     (swizzle e xy)              e.xy
     (vec2|vec3|vec4 a ...)      constructor
   Statements:
     vertex:   (let n TYPE e)  (set-position e)  (out VARYING e)
     fragment: (let n TYPE e)  (color e)

   Description shape:
     {:uniform    {:name \"U\" :fields [[:viewport :vec4]]}   ; :uniform optional
      :attributes [[inPos vec2 0] [inColor vec3 1]]
      :varyings   [[vColor vec3 0]]
      :vertex     [(let ndc vec2 ...) (set-position ...) (out vColor inColor)]
      :fragment   [(color (vec4 vColor 1.0))]}"
  (:require [clojure.string :as str]))

(def ^:private type-bytes {:float 4 :vec2 8 :vec3 12 :vec4 16 :mat4 64})
(def ^:private type-align {:float 4 :vec2 8 :vec3 16 :vec4 16 :mat4 16})
(def ^:private glsl-type {:float "float" :vec2 "vec2" :vec3 "vec3" :vec4 "vec4" :mat4 "mat4"})
(def ^:private wgsl-type {:float "f32" :vec2 "vec2<f32>" :vec3 "vec3<f32>" :vec4 "vec4<f32>" :mat4 "mat4x4<f32>"})

(defn- tkey [t] (if (keyword? t) t (keyword (name t))))

(defn- fmt-num [x]
  ;; cljs prints 2.0 as "2"; GLSL/WGSL need an explicit decimal for f32 literals
  (let [s (str (double x))]
    (if (re-find #"[.eE]" s) s (str s ".0"))))

;; --- uniform layout ----------------------------------------------------------
(defn- uniform-fields [spec] (get-in spec [:uniform :fields]))
(defn- uniform-name [spec] (get-in spec [:uniform :name] "U"))

(defn uniform-bytes
  "std140-ish size of the uniform block (0 when there is no uniform)."
  [spec]
  (let [fields (uniform-fields spec)]
    (if (empty? fields)
      0
      (let [end (reduce (fn [off [_ t]]
                          (let [k (tkey t) a (type-align k)
                                base (* (quot (+ off (dec a)) a) a)]
                            (+ base (type-bytes k))))
                        0 fields)]
        (* 16 (quot (+ end 15) 16))))))

(defn- uniform-field-names [spec] (set (map (comp name first) (uniform-fields spec))))

(defn- all-syms [form]
  (cond (symbol? form) #{(name form)}
        (coll? form)   (reduce into #{} (map all-syms (seq form)))
        :else          #{}))

(defn- used-in? [stage-stmts ufields] (boolean (some ufields (all-syms stage-stmts))))

(defn uniform-stages
  "Set of stages (#{:vertex :fragment}) that reference the uniform."
  [spec]
  (let [uf (uniform-field-names spec)]
    (cond-> #{}
      (used-in? (:vertex spec) uf)   (conj :vertex)
      (used-in? (:fragment spec) uf) (conj :fragment))))

;; --- texture declarations ----------------------------------------------------
(defn- textures [spec] (:textures spec))
(defn- texture-arrays [spec] (:texture-arrays spec))
(defn- stage-textures
  "The [name binding] 2D-texture entries referenced in a stage's statements."
  [stage-stmts spec]
  (let [used (all-syms stage-stmts)]
    (filter (fn [[n _]] (used (name n))) (textures spec))))
(defn- stage-texture-arrays
  "The [name binding] 2D-array-texture entries referenced in a stage."
  [stage-stmts spec]
  (let [used (all-syms stage-stmts)]
    (filter (fn [[n _]] (used (name n))) (texture-arrays spec))))

;; --- expression emitter (dialect-parameterized) ------------------------------
(defn- emit-expr [e {:keys [vec-suffix kind] :as dial} ufields]
  (cond
    (number? e) (fmt-num e)
    (symbol? e) (let [n (name e)] (if (ufields n) (str "u." n) n))
    (seq? e)
    (let [[op & args] e]
      (case (name op)
        ("+" "-" "*" "/") (str "(" (str/join (str " " (name op) " ")
                                             (map #(emit-expr % dial ufields) args)) ")")
        "swizzle"         (str (emit-expr (first args) dial ufields) "." (name (second args)))
        ("vec2" "vec3" "vec4")
        (str (name op) vec-suffix "("
             (str/join ", " (map #(emit-expr % dial ufields) args)) ")")
        ;; (sample tex uv) → texture sample (combined sampler2D vs split texture+sampler)
        "sample" (let [tex (name (first args)) uv (emit-expr (second args) dial ufields)]
                   (case kind
                     :glsl (str "texture(" tex ", " uv ")")
                     :wgsl (str "textureSample(" tex ", " tex "_s, " uv ")")))
        ;; (sample-layer atlas uv layer) → 2D-array sample at integer layer
        "sample-layer" (let [tex (name (first args))
                             uv (emit-expr (second args) dial ufields)
                             ly (emit-expr (nth args 2) dial ufields)]
                         (case kind
                           :glsl (str "texture(" tex ", vec3(" uv ", " ly "))")
                           :wgsl (str "textureSample(" tex ", " tex "_s, " uv ", i32(" ly "))")))
        ;; built-in functions with identical names + semantics in GLSL and WGSL —
        ;; emit a plain call. (inverse is intentionally excluded: WGSL lacks it; pass a
        ;; CPU-computed inverse as a uniform instead.)
        ("mix" "clamp" "smoothstep" "step" "normalize" "dot" "cross" "length"
               "pow" "sin" "cos" "tan" "exp" "log" "sqrt" "abs" "sign" "floor" "ceil"
               "fract" "max" "min" "reflect" "distance")
        (str (name op) "(" (str/join ", " (map #(emit-expr % dial ufields) args)) ")")
        (throw (ex-info "unknown shader op" {:op op}))))
    :else (throw (ex-info "bad shader expr" {:expr e}))))

;; --- GLSL --------------------------------------------------------------------
(def ^:private glsl-dial {:vec-suffix "" :kind :glsl})

(defn- glsl-tex-decls [stage-stmts spec]
  (str (str/join "" (map (fn [[n b]] (str "layout(set=0, binding=" b ") uniform sampler2D " (name n) ";\n"))
                         (stage-textures stage-stmts spec)))
       (str/join "" (map (fn [[n b]] (str "layout(set=0, binding=" b ") uniform sampler2DArray " (name n) ";\n"))
                         (stage-texture-arrays stage-stmts spec)))))

(defn- glsl-uniform-decl [spec]
  (when (seq (uniform-fields spec))
    (str "layout(push_constant) uniform " (uniform-name spec) " {\n"
         (str/join "\n" (map (fn [[f t]] (str "  " (glsl-type (tkey t)) " " (name f) ";"))
                             (uniform-fields spec)))
         "\n} u;\n")))

(defn- glsl-stmt [s dial ufields]
  (let [[op & args] s]
    (case (name op)
      "let"          (let [[n t e] args]
                       (str "  " (glsl-type (tkey t)) " " (name n) " = " (emit-expr e dial ufields) ";"))
      "set-position" (str "  gl_Position = " (emit-expr (first args) dial ufields) ";")
      "out"          (str "  " (name (first args)) " = " (emit-expr (second args) dial ufields) ";")
      "color"        (str "  outColor = " (emit-expr (first args) dial ufields) ";"))))

(defn- glsl-vertex [spec]
  (let [uf (uniform-field-names spec)]
    (str "#version 450\n"
         (str/join "\n" (map (fn [[n t l]] (str "layout(location=" l ") in " (glsl-type (tkey t)) " " (name n) ";"))
                             (:attributes spec))) "\n"
         (str/join "\n" (map (fn [[n t l]] (str "layout(location=" l ") out " (glsl-type (tkey t)) " " (name n) ";"))
                             (:varyings spec))) "\n"
         (or (when (contains? (uniform-stages spec) :vertex) (glsl-uniform-decl spec)) "")
         (glsl-tex-decls (:vertex spec) spec)
         "void main() {\n"
         (str/join "\n" (map #(glsl-stmt % glsl-dial uf) (:vertex spec)))
         "\n}\n")))

(defn- glsl-fragment [spec]
  (let [uf (uniform-field-names spec)]
    (str "#version 450\n"
         (str/join "\n" (map (fn [[n t l]] (str "layout(location=" l ") in " (glsl-type (tkey t)) " " (name n) ";"))
                             (:varyings spec))) "\n"
         "layout(location=0) out vec4 outColor;\n"
         (or (when (contains? (uniform-stages spec) :fragment) (glsl-uniform-decl spec)) "")
         (glsl-tex-decls (:fragment spec) spec)
         "void main() {\n"
         (str/join "\n" (map #(glsl-stmt % glsl-dial uf) (:fragment spec)))
         "\n}\n")))

(defn ->glsl
  "Compile a shader description to {:vert glsl-string :frag glsl-string}."
  [spec]
  {:vert (glsl-vertex spec) :frag (glsl-fragment spec)})

;; --- WGSL --------------------------------------------------------------------
(def ^:private wgsl-dial {:vec-suffix "<f32>" :kind :wgsl})

(defn- wgsl-tex-decls [stage-stmts spec]
  (str (str/join "" (map (fn [[n b]]
                           (str "@group(1) @binding(" (* 2 b) ") var " (name n) ": texture_2d<f32>;\n"
                                "@group(1) @binding(" (inc (* 2 b)) ") var " (name n) "_s: sampler;\n"))
                         (stage-textures stage-stmts spec)))
       (str/join "" (map (fn [[n b]]
                           (str "@group(1) @binding(" (* 2 b) ") var " (name n) ": texture_2d_array<f32>;\n"
                                "@group(1) @binding(" (inc (* 2 b)) ") var " (name n) "_s: sampler;\n"))
                         (stage-texture-arrays stage-stmts spec)))))

(defn- wgsl-uniform-decl [spec]
  (when (seq (uniform-fields spec))
    (str "struct " (uniform-name spec) " {\n"
         (str/join "\n" (map (fn [[f t]] (str "  " (name f) ": " (wgsl-type (tkey t)) ",")) (uniform-fields spec)))
         "\n};\n@group(0) @binding(0) var<uniform> u: " (uniform-name spec) ";\n")))

(defn- wgsl-vertex [spec]
  (let [uf (uniform-field-names spec)
        params (str/join ", " (map (fn [[n t l]] (str "@location(" l ") " (name n) ": " (wgsl-type (tkey t))))
                                   (:attributes spec)))
        stmts (mapcat
               (fn [s]
                 (let [[op & args] s]
                   (case (name op)
                     "let"          [(str "  let " (name (first args)) " = " (emit-expr (nth args 2) wgsl-dial uf) ";")]
                     "set-position" (let [p (emit-expr (first args) wgsl-dial uf)]
                                      ;; Y-flip: WebGPU NDC is Y-up vs the author's Vulkan Y-down
                                      [(str "  let _p = " p ";")
                                       "  out.position = vec4<f32>(_p.x, -_p.y, _p.z, _p.w);"])
                     "out"          [(str "  out." (name (first args)) " = " (emit-expr (second args) wgsl-dial uf) ";")])))
               (:vertex spec))]
    (str (or (when (contains? (uniform-stages spec) :vertex) (wgsl-uniform-decl spec)) "")
         (wgsl-tex-decls (:vertex spec) spec)
         "struct VSOut {\n  @builtin(position) position: vec4<f32>,\n"
         (str/join "\n" (map (fn [[n t l]] (str "  @location(" l ") " (name n) ": " (wgsl-type (tkey t)) ",")) (:varyings spec)))
         "\n};\n@vertex\nfn main(" params ") -> VSOut {\n  var out: VSOut;\n"
         (str/join "\n" stmts)
         "\n  return out;\n}\n")))

(defn- wgsl-fragment [spec]
  (let [uf (uniform-field-names spec)
        params (str/join ", " (map (fn [[n t l]] (str "@location(" l ") " (name n) ": " (wgsl-type (tkey t))))
                                   (:varyings spec)))
        stmts (mapcat
               (fn [s]
                 (let [[op & args] s]
                   (case (name op)
                     "let"   [(str "  let " (name (first args)) " = " (emit-expr (nth args 2) wgsl-dial uf) ";")]
                     "color" [(str "  return " (emit-expr (first args) wgsl-dial uf) ";")])))
               (:fragment spec))]
    (str (or (when (contains? (uniform-stages spec) :fragment) (wgsl-uniform-decl spec)) "")
         (wgsl-tex-decls (:fragment spec) spec)
         "@fragment\nfn main(" params ") -> @location(0) vec4<f32> {\n"
         (str/join "\n" stmts)
         "\n}\n")))

(defn ->wgsl
  "Compile a shader description to {:vert wgsl-string :frag wgsl-string}."
  [spec]
  {:vert (wgsl-vertex spec) :frag (wgsl-fragment spec)})
