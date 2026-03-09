(ns raster.compiler.backend.gpu.par-vulkan
  "Vulkan compute backend for declarative parallel primitives.

  Generates GLSL 450 compute shader source from par forms using
  the shared c-emit expression emitter. Shaders are compiled to
  SPIR-V via shaderc (raster.vk.shader) and dispatched through
  the Vulkan compute pipeline (raster.vk.compute).

  GLSL scaffold:
    #version 450
    layout(std430, binding=N) buffer BufN { float data[]; } name;
    layout(push_constant) uniform PC { uint n; };
    layout(local_size_x = 256) in;
    void main() {
        for (uint idx = gl_GlobalInvocationID.x; idx < n;
             idx += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {
            // body from c-emit
        }
    }

  Usage:
    (generate-compute-map-kernel form :dtype :float)
    (generate-compute-map-void-kernel form :dtype :float)"
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [clojure.string :as str]))

;; ================================================================
;; GLSL type maps
;; ================================================================

(def glsl-type-map
  {:double "double"
   :float  "float"
   :int    "int"
   :long   "int"})  ;; GLSL has no 64-bit int by default; use int

;; ================================================================
;; GLSL buffer declaration helpers
;; ================================================================

(defn- glsl-buffer-decl
  "Generate a layout(std430, binding=N) buffer declaration.
  readonly? controls whether to add 'readonly' qualifier."
  [binding-idx ctype c-name readonly?]
  (str "layout(std430, binding = " binding-idx ") "
       (if readonly? "readonly " "")
       "buffer Buf" binding-idx " { " ctype " data[]; } " c-name ";"))

(defn- glsl-push-constants
  "Generate push constant block for scalar params + n."
  [scalar-params scl-type-fn]
  (let [members (concat
                 (map (fn [s] (str "    " (scl-type-fn s) " " (ce/c-symbol s) ";"))
                      scalar-params)
                 ["    uint n;"])]
    (str "layout(push_constant) uniform PC {\n"
         (str/join "\n" members)
         "\n};")))

;; ================================================================
;; SSBO suffix helper
;; ================================================================

(defn- apply-ssbo-suffix
  "Replace array[idx] with array.data[idx] for all plain and SoA arrays."
  [body-str plain-arr-params soa-expansions soa-arr-params]
  (let [body-str (reduce (fn [s sym]
                           (let [cname (ce/c-symbol sym)]
                             (str/replace s
                                          (re-pattern (str "(?<![\\w])" (java.util.regex.Pattern/quote cname) "\\["))
                                          (str cname ".data["))))
                         body-str plain-arr-params)]
    (reduce (fn [s sym]
              (if-let [soa-info (get soa-expansions (symbol (name sym)))]
                (reduce (fn [s2 {:keys [name]}]
                          (let [flat (str (ce/c-symbol sym) "_" name)]
                            (str/replace s2
                                         (re-pattern (str "(?<![\\w])" (java.util.regex.Pattern/quote flat) "\\["))
                                         (str flat ".data["))))
                        s (:fields soa-info))
                s))
            body-str soa-arr-params)))

;; ================================================================
;; Kernel generators
;; ================================================================

(defn generate-compute-map-kernel
  "Generate a GLSL 450 compute shader from a raster.par/map! form.

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [syms] :push-constant-size int
           :n-bindings int :dtype kw}."
  [form & {:keys [dtype kernel-name-prefix workgroup-size helpers scalar-types]
           :or {kernel-name-prefix "par_map"
                workgroup-size 256 helpers "" scalar-types {}}}]
  (let [info (par/extract-par-map-info form)
        {:keys [idx cast body]} info
        ;; Auto-derive dtype from cast symbol if not explicitly provided
        dtype (or dtype (get {'double :double 'float :float 'int :int 'long :long} cast :float))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        default-ctype (get glsl-type-map dtype "float")
        ;; SoA expansion
        soa-expansions (ce/collect-soa-expansions body)
        meta-types (ce/collect-array-types-from-meta body)
        array-syms (ce/collect-arrays-in-body body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        all-arr-params (vec (sort-by name array-syms))
        ;; Partition: SoA arrays vs plain arrays
        soa-arr-params (filterv #(contains? soa-expansions (symbol (name %))) all-arr-params)
        plain-arr-params (filterv #(not (contains? soa-expansions (symbol (name %)))) all-arr-params)
        scl-params (vec (sort-by name scalar-syms))
        ;; Per-array GLSL type
        arr-type (fn [s]
                   (let [t (get meta-types s (get meta-types (symbol (name s)) dtype))]
                     (get glsl-type-map t default-ctype)))
        ;; Assign SSBO bindings
        binding-idx (atom 0)
        ;; Plain array buffer declarations (readonly for map! inputs)
        plain-decls (mapv (fn [s]
                            (let [ct (arr-type s)
                                  i @binding-idx
                                  _ (swap! binding-idx inc)]
                              (glsl-buffer-decl i ct (ce/c-symbol s) true)))
                          plain-arr-params)
        ;; SoA array buffer declarations (one binding per field, readonly)
        soa-decls (vec (mapcat (fn [s]
                                 (let [soa-info (get soa-expansions (symbol (name s)))
                                       fields (:fields soa-info)]
                                   (mapv (fn [{:keys [name element-tag]}]
                                           (let [ct (get ce/element-tag->c element-tag "float")
                                                 flat-name (str (ce/c-symbol s) "_" name)
                                                 i @binding-idx
                                                 _ (swap! binding-idx inc)]
                                             (glsl-buffer-decl i ct flat-name true)))
                                         fields)))
                               soa-arr-params))
        out-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        n-bindings @binding-idx
        out-decl (glsl-buffer-decl out-binding default-ctype "out_buf" false)
        ;; Scalar type inference for push constants
        scl-type (fn [s]
                   (let [sname (name s)
                         explicit (get scalar-types s (get scalar-types (symbol sname)))]
                     (cond
                       (= explicit :int) "uint"
                       (or (re-find #"(?i)n[-_]|size|count|len|idx|offset" sname)
                           (contains? #{'long 'int} (:tag (meta s))))
                       "uint"
                       :else default-ctype)))
        push-block (glsl-push-constants scl-params scl-type)
        ;; Push constant size: sum of each member aligned per std430 rules
        push-constant-size (let [sizes (concat
                                        (map (fn [s] (if (= (scl-type s) "double") 8 4)) scl-params)
                                        [4])] ;; uint n
                             (reduce (fn [offset sz]
                                       (let [aligned (if (> (mod offset sz) 0)
                                                       (+ offset (- sz (mod offset sz)))
                                                       offset)]
                                         (+ aligned sz)))
                                     0 sizes))
        ;; Body emission
        adapted-body (ce/adapt-casts-for-dtype body dtype)
        all-arr-syms (set (map #(symbol (name %)) all-arr-params))
        ;; Detect if body is a loop (needs statement context in GLSL)
        body-is-loop? (and (seq? adapted-body) (= 'loop (first adapted-body)))
        ;; For loop bodies, extract the result variable name (first loop binding)
        loop-result-var (when body-is-loop?
                          (ce/c-symbol (first (take-nth 2 (second adapted-body)))))
        body-str (binding [ce/*emit-config* ce/glsl-config
                           ce/*scalar-type* default-ctype
                           ce/*soa-expansions* soa-expansions
                           ce/*idx-sym* idx]
                   ;; Both paths use emit-expr; for GLSL loops, emit-expr now
                   ;; returns inline declarations + while(true){} without outer braces
                   (ce/emit-expr adapted-body idx all-arr-syms "idx"))
        ;; Replace plain array[idx] with array.data[idx] for SSBO access
        body-str (apply-ssbo-suffix body-str plain-arr-params soa-expansions soa-arr-params)
        cast-str (if body-is-loop?
                   ;; Loop emitted as block; append output assignment after
                   (let [result (if cast
                                  (str default-ctype "(" loop-result-var ")")
                                  loop-result-var)]
                     (str body-str " out_buf.data[idx] = " result ";"))
                   (if cast (str default-ctype "(" body-str ")") body-str))
        ;; Helper functions
        gpu-fns (ce/collect-gpu-fn-calls body)
        helper-sources (binding [ce/*emit-config* ce/glsl-config
                                 ce/*scalar-type* default-ctype]
                         (str/join "\n" (map (comp :source ce/generate-c-helper) gpu-fns)))
        ;; Struct typedefs
        struct-typedefs (binding [ce/*emit-config* ce/glsl-config]
                          (ce/emit-struct-typedefs soa-expansions))
        source (str "#version 450\n"
                    (when (= dtype :double)
                      "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n")
                    "\n"
                    (str/join "\n" (concat plain-decls soa-decls)) "\n"
                    out-decl "\n"
                    "\n"
                    push-block "\n"
                    "\n"
                    "layout(local_size_x = " workgroup-size ") in;\n"
                    "\n"
                    (when (seq struct-typedefs) (str struct-typedefs "\n\n"))
                    (when (seq helper-sources) (str helper-sources "\n"))
                    (when (seq helpers) (str helpers "\n"))
                    "void main() {\n"
                    "    for (uint idx = gl_GlobalInvocationID.x; idx < n;\n"
                    "         idx += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {\n"
                    (if body-is-loop?
                      (str "        " cast-str "\n")
                      (str "        out_buf.data[idx] = " cast-str ";\n"))
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params all-arr-params
     :soa-expansions soa-expansions
     :scalar-params scl-params
     :out-binding out-binding
     :n-bindings n-bindings
     :push-constant-size push-constant-size
     :workgroup-size workgroup-size
     :dtype dtype}))

(defn generate-compute-map-void-kernel
  "Generate a GLSL 450 compute shader from a raster.par/map-void! form.
  No output array — body is executed for side effects (SSBO writes, atomics).

  array-types: {sym -> :float|:int|:long|:double} for mixed-type params.

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [syms] :written-arrays #{syms}
           :n-bindings int :push-constant-size int :dtype kw}."
  [form & {:keys [dtype kernel-name-prefix workgroup-size array-types helpers scalar-types]
           :or {dtype :float kernel-name-prefix "par_map_void"
                workgroup-size 256 array-types {} helpers "" scalar-types {}}}]
  (let [info (par/extract-par-map-void-info form)
        {:keys [idx body]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        default-ctype (get glsl-type-map dtype "float")
        ;; Auto-collect array types from walker :tag metadata, merge with explicit
        meta-types (ce/collect-array-types-from-meta body)
        array-types (merge meta-types array-types)
        ;; SoA expansion
        soa-expansions (ce/collect-soa-expansions body)
        array-syms (ce/collect-arrays-in-body body)
        written-syms (ce/collect-written-arrays body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        all-arr-params (vec (sort-by name array-syms))
        ;; Partition: SoA arrays vs plain arrays
        soa-arr-params (filterv #(contains? soa-expansions (symbol (name %))) all-arr-params)
        plain-arr-params (filterv #(not (contains? soa-expansions (symbol (name %)))) all-arr-params)
        scl-params (vec (sort-by name scalar-syms))
        ;; Per-array GLSL type
        arr-type (fn [s]
                   (let [t (get array-types s (get array-types (symbol (name s)) dtype))]
                     (get glsl-type-map t default-ctype)))
        written? (fn [s] (or (contains? written-syms s)
                             (contains? written-syms (symbol (name s)))))
        ;; Assign SSBO bindings
        binding-idx (atom 0)
        ;; Plain array buffer declarations
        plain-decls (mapv (fn [s]
                            (let [ct (arr-type s)
                                  i @binding-idx
                                  _ (swap! binding-idx inc)
                                  readonly (not (written? s))]
                              (glsl-buffer-decl i ct (ce/c-symbol s) readonly)))
                          plain-arr-params)
        ;; SoA array buffer declarations (one binding per field)
        soa-decls (vec (mapcat (fn [s]
                                 (let [soa-info (get soa-expansions (symbol (name s)))
                                       fields (:fields soa-info)
                                       writable (written? s)]
                                   (mapv (fn [{:keys [name element-tag]}]
                                           (let [ct (get ce/element-tag->c element-tag "float")
                                                 flat-name (str (ce/c-symbol s) "_" name)
                                                 i @binding-idx
                                                 _ (swap! binding-idx inc)]
                                             (glsl-buffer-decl i ct flat-name (not writable))))
                                         fields)))
                               soa-arr-params))
        n-bindings @binding-idx
        ;; Scalar type inference for push constants
        scl-type (fn [s]
                   (let [sname (name s)
                         explicit (get scalar-types s (get scalar-types (symbol sname)))]
                     (cond
                       (= explicit :int) "uint"
                       (or (re-find #"(?i)n[-_]|size|count|len|idx|offset" sname)
                           (contains? #{'long 'int} (:tag (meta s))))
                       "uint"
                       :else default-ctype)))
        push-block (glsl-push-constants scl-params scl-type)
        push-constant-size (let [sizes (concat
                                        (map (fn [s] (if (= (scl-type s) "double") 8 4)) scl-params)
                                        [4])] ;; uint n
                             (reduce (fn [offset sz]
                                       (let [aligned (if (> (mod offset sz) 0)
                                                       (+ offset (- sz (mod offset sz)))
                                                       offset)]
                                         (+ aligned sz)))
                                     0 sizes))
        ;; Struct typedefs
        struct-typedefs (binding [ce/*emit-config* ce/glsl-config]
                          (ce/emit-struct-typedefs soa-expansions))
        ;; Body emission
        adapted-body (ce/adapt-casts-for-dtype body dtype)
        all-arr-syms (set (map #(symbol (name %)) all-arr-params))
        body-str (binding [ce/*emit-config* ce/glsl-config
                           ce/*scalar-type* default-ctype
                           ce/*soa-expansions* soa-expansions
                           ce/*idx-sym* idx]
                   (ce/emit-stmt adapted-body idx all-arr-syms "idx"))
        body-str (apply-ssbo-suffix body-str plain-arr-params soa-expansions soa-arr-params)
        ;; Helper functions
        gpu-fns (ce/collect-gpu-fn-calls body)
        helper-sources (binding [ce/*emit-config* ce/glsl-config
                                 ce/*scalar-type* default-ctype]
                         (str/join "\n" (map (comp :source ce/generate-c-helper) gpu-fns)))
        source (str "#version 450\n"
                    (when (= dtype :double)
                      "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n")
                    "\n"
                    (str/join "\n" (concat plain-decls soa-decls)) "\n"
                    "\n"
                    push-block "\n"
                    "\n"
                    "layout(local_size_x = " workgroup-size ") in;\n"
                    "\n"
                    (when (seq struct-typedefs) (str struct-typedefs "\n\n"))
                    (when (seq helper-sources) (str helper-sources "\n"))
                    (when (seq helpers) (str helpers "\n"))
                    "void main() {\n"
                    "    for (uint idx = gl_GlobalInvocationID.x; idx < n;\n"
                    "         idx += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {\n"
                    "        " body-str "\n"
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params all-arr-params
     :soa-expansions soa-expansions
     :scalar-params scl-params
     :written-arrays written-syms
     :array-types array-types
     :n-bindings n-bindings
     :push-constant-size push-constant-size
     :workgroup-size workgroup-size
     :dtype dtype}))

;; ================================================================
;; Stencil kernel generator (GLSL)
;; ================================================================

(defn generate-compute-stencil-kernel
  "Generate a GLSL 450 compute shader from a raster.par/stencil! form.
  Boundary handling: Dirichlet (zero outside).

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [syms] :out-binding int :n-bindings int
           :push-constant-size int :dtype kw}."
  [form & {:keys [dtype kernel-name-prefix workgroup-size scalar-types]
           :or {kernel-name-prefix "par_stencil" workgroup-size 256 scalar-types {}}}]
  (let [info (par/extract-par-stencil-info form)
        {:keys [out radius boundary cast idx body]} info
        dtype (or dtype (get {'double :double 'float :float 'int :int} cast :float))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        default-ctype (get glsl-type-map dtype "float")
        array-syms (ce/collect-arrays-in-body body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        arr-params (vec (sort-by name array-syms))
        scl-params (vec (sort-by name scalar-syms))
        ;; Assign SSBO bindings: input arrays (readonly), then output
        binding-idx (atom 0)
        arr-decls (mapv (fn [s]
                          (let [i @binding-idx _ (swap! binding-idx inc)]
                            (glsl-buffer-decl i default-ctype (ce/c-symbol s) true)))
                        arr-params)
        out-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        out-decl (glsl-buffer-decl out-binding default-ctype (ce/c-symbol out) false)
        n-bindings @binding-idx
        scl-type (fn [s]
                   (let [sname (name s)
                         explicit (get scalar-types s (get scalar-types (symbol sname)))]
                     (cond
                       (= explicit :int) "uint"
                       (or (re-find #"(?i)n[-_]|size|count|len|idx|offset" sname)
                           (contains? #{'long 'int} (:tag (meta s))))
                       "uint"
                       :else default-ctype)))
        push-block (glsl-push-constants scl-params scl-type)
        push-constant-size (let [sizes (concat
                                        (map (fn [s] (if (= (scl-type s) "double") 8 4)) scl-params)
                                        [4])]
                             (reduce (fn [off sz]
                                       (let [al (if (> (mod off sz) 0) (+ off (- sz (mod off sz))) off)]
                                         (+ al sz)))
                                     0 sizes))
        adapted-body (ce/adapt-casts-for-dtype body dtype)
        body-str (binding [ce/*emit-config* ce/glsl-config
                           ce/*scalar-type* default-ctype]
                   (ce/emit-expr adapted-body idx (set (map #(symbol (name %)) arr-params)) "idx"))
        body-str (apply-ssbo-suffix body-str arr-params {} [])
        cast-str (if cast (str default-ctype "(" body-str ")") body-str)
        out-c (ce/c-symbol out)
        source (str "#version 450\n\n"
                    (str/join "\n" arr-decls) "\n"
                    out-decl "\n\n"
                    push-block "\n\n"
                    "layout(local_size_x = " workgroup-size ") in;\n\n"
                    "void main() {\n"
                    "    for (uint idx = gl_GlobalInvocationID.x; idx < n;\n"
                    "         idx += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {\n"
                    (case boundary
                      :dirichlet
                      (str "        if (idx < " radius " || idx >= n - " radius ") {\n"
                           "            " out-c ".data[idx] = 0.0;\n"
                           "        } else {\n"
                           "            " out-c ".data[idx] = " cast-str ";\n"
                           "        }\n")
                      (str "        " out-c ".data[idx] = " cast-str ";\n"))
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params arr-params
     :scalar-params scl-params
     :out-binding out-binding
     :n-bindings n-bindings
     :push-constant-size push-constant-size
     :workgroup-size workgroup-size
     :dtype dtype}))

;; ================================================================
;; Scatter kernel generator (GLSL)
;; ================================================================

(defn generate-compute-scatter-kernel
  "Generate a GLSL 450 compute shader from a raster.par/scatter! form.
  Scatter-add using atomicAdd for int arrays.
  Note: GLSL atomicAdd only supports int/uint; float scatter uses coherent + fence.

  Returns {:kernel-name str :source str :array-params [syms]
           :n-bindings int :push-constant-size int :dtype kw}."
  [form & {:keys [dtype kernel-name-prefix workgroup-size]
           :or {kernel-name-prefix "par_scatter" workgroup-size 256 dtype :float}}]
  (let [info (par/extract-par-scatter-info form)
        {:keys [out src index stride]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        default-ctype (get glsl-type-map dtype "float")
        strided? (some? stride)
        binding-idx (atom 0)
        out-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        out-decl (glsl-buffer-decl out-binding default-ctype (ce/c-symbol out) false)
        src-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        src-decl (glsl-buffer-decl src-binding default-ctype (ce/c-symbol src) true)
        idx-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        idx-decl (glsl-buffer-decl idx-binding "int" (ce/c-symbol index) true)
        n-bindings @binding-idx
        push-members (if strided? ["    uint n;" "    uint stride_val;"] ["    uint n;"])
        push-block (str "layout(push_constant) uniform PC {\n"
                        (str/join "\n" push-members) "\n};")
        push-constant-size (if strided? 8 4)
        out-c (ce/c-symbol out)
        src-c (ce/c-symbol src)
        index-c (ce/c-symbol index)
        source (str "#version 450\n\n"
                    out-decl "\n"
                    src-decl "\n"
                    idx-decl "\n\n"
                    push-block "\n\n"
                    "layout(local_size_x = " workgroup-size ") in;\n\n"
                    "void main() {\n"
                    "    for (uint i = gl_GlobalInvocationID.x; i < n;\n"
                    "         i += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {\n"
                    "        int dst_idx = " index-c ".data[i];\n"
                    (if strided?
                      (str "        for (uint d = 0; d < stride_val; d++) {\n"
                           "            uint src_pos = i * stride_val + d;\n"
                           "            uint dst_pos = uint(dst_idx) * stride_val + d;\n"
                           "            atomicAdd(" out-c ".data[dst_pos], " src-c ".data[src_pos]);\n"
                           "        }\n")
                      (str "        atomicAdd(" out-c ".data[dst_idx], " src-c ".data[i]);\n"))
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params [out src index]
     :n-bindings n-bindings
     :push-constant-size push-constant-size
     :strided? strided?
     :workgroup-size workgroup-size
     :dtype dtype}))

;; ================================================================
;; Reduce-by-key kernel generator (GLSL)
;; ================================================================

(defn generate-compute-reduce-by-key-kernel
  "Generate a GLSL 450 compute shader for reduce-by-key.
  output[keys[i]] += vals[i] using atomicAdd.

  Returns {:kernel-name str :source str :array-params [syms]
           :n-bindings int :push-constant-size int :dtype kw}."
  [form & {:keys [dtype kernel-name-prefix workgroup-size]
           :or {kernel-name-prefix "par_reduce_by_key" workgroup-size 256 dtype :float}}]
  (let [info (par/extract-par-reduce-by-key-info form)
        {:keys [out keys vals]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        default-ctype (get glsl-type-map dtype "float")
        binding-idx (atom 0)
        out-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        out-decl (glsl-buffer-decl out-binding default-ctype (ce/c-symbol out) false)
        keys-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        keys-decl (glsl-buffer-decl keys-binding "int" (ce/c-symbol keys) true)
        vals-binding (let [i @binding-idx] (swap! binding-idx inc) i)
        vals-decl (glsl-buffer-decl vals-binding default-ctype (ce/c-symbol vals) true)
        n-bindings @binding-idx
        out-c (ce/c-symbol out)
        keys-c (ce/c-symbol keys)
        vals-c (ce/c-symbol vals)
        source (str "#version 450\n\n"
                    out-decl "\n"
                    keys-decl "\n"
                    vals-decl "\n\n"
                    "layout(push_constant) uniform PC {\n    uint n;\n};\n\n"
                    "layout(local_size_x = " workgroup-size ") in;\n\n"
                    "void main() {\n"
                    "    for (uint i = gl_GlobalInvocationID.x; i < n;\n"
                    "         i += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {\n"
                    "        int k = " keys-c ".data[i];\n"
                    "        atomicAdd(" out-c ".data[k], " vals-c ".data[i]);\n"
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params [out keys vals]
     :n-bindings n-bindings
     :push-constant-size 4
     :workgroup-size workgroup-size
     :dtype dtype}))

;; ================================================================
;; Dispatch helpers
;; ================================================================

(defn compute-group-count
  "Calculate the number of workgroups for n elements."
  [^long n ^long workgroup-size]
  (long (Math/ceil (/ (double n) (double workgroup-size)))))
