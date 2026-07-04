(ns raster.compiler.backend.gpu.wgsl
  "Track C — WebGPU compute backend: emit a WGSL compute shader from a raster
   deftm's post-pass IR for an elementwise map.

   WebGPU/WGSL has no f64, so this backend is f32-only (the GPU numerical
   identity; f64 survives on the wasm/CPU path). Arrays become storage buffers
   (one @group(0) @binding per array); scalars + the element count ride in a
   uniform struct. Each invocation handles one element: `let i = gid.x; if (i <
   U._n) { out[i] = <expr>; }`. Operator lowering comes from the shared
   raster.compiler.backend.intrinsics table (:wgsl facet) — no per-backend op
   set here.

   v1 scope: elementwise maps (the (loop* [i 0] (if (< i n) (do <aset…> (recur
   (inc i))) els)) shape the passes lower dotimes into, and the bare dotimes SoA
   shape) — one or more `(aset arr i pure)` stores whose value is aget@i / arith
   / Math / scalar. Reductions + i32 fixed-point atomic scatter are a follow-on."
  (:require [raster.compiler.ir.form :as form]
            [raster.compiler.backend.intrinsics :as ix]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Extract the elementwise stores from the lowered IR
;; ---------------------------------------------------------------------------

(defn- unwrap
  "Peel let*/do wrappers to the meaningful inner form."
  [form]
  (cond
    (and (seq? form) (= 'let* (first form))) (recur (last form))
    (and (seq? form) (= 'do (first form)) (= 2 (count form))) (recur (second form))
    :else form))

(defn- find-counted-loop
  "Locate the (loop* [i init] (if (< i n) (do … (recur …)) els)) or (dotimes [i n] …)."
  [form]
  (let [f (unwrap form)]
    (cond
      (and (seq? f) (= 'let* (first f)))
      (some find-counted-loop (drop 2 f))
      (and (seq? f) (= 'do (first f)))
      (some find-counted-loop (rest f))
      (and (seq? f) ((some-fn form/loop-head? #{'dotimes}) (first f))) f
      :else nil)))

(defn- store? [x]
  (and (seq? x) (= 'clojure.core/aset (first x)) (= 4 (count x))))

(defn- flatten-body [forms]
  (mapcat (fn f [x] (if (and (seq? x) ((some-fn #{'do} form/let-head?) (first x)))
                      (mapcat f (if (= 'let* (first x)) (drop 2 x) (rest x))) [x]))
          forms))

(defn- as-tuple [s] [(nth s 1) (nth s 2) (nth s 3)])  ; (aset arr idx val) → [arr idx val]

(defn- loop-body-stores
  "Given the counted loop, return {:isym :nsym :stores [[arr idx-expr val]…]}."
  [loopf]
  (if (= 'dotimes (first loopf))
    (let [[_ [isym ncnt] & body] loopf]
      {:isym isym :nsym ncnt :stores (mapv as-tuple (filter store? (flatten-body body)))})
    (let [[_ [isym _] ifform] loopf
          [_ cnd then _] ifform]
      {:isym isym :nsym (nth cnd 2)
       :stores (mapv as-tuple (filter store? (flatten-body (rest then))))})))

;; ---------------------------------------------------------------------------
;; WGSL expression emission (registry-driven)
;; ---------------------------------------------------------------------------

(declare emit-expr)

(defn- wgsl-num [x]
  (let [s (str (double x))] (if (str/includes? s ".") s (str s ".0"))))

(defn- emit-call [op args env]
  (let [d (ix/descriptor op)
        w (:wgsl d)
        ea (mapv #(emit-expr % env) args)]
    (cond
      (nil? d) (throw (ex-info (str "no WGSL lowering for op " op) {:op op}))
      (= :floored-mod w) (let [[a b] ea] (str "(" a " - floor(" a " / " b ") * " b ")"))
      (= :cmp (:kind d)) (str "(" (first ea) " " w " " (second ea) ")")
      (string? w) (str "(" (str/join (str " " w " ") ea) ")")              ; infix
      (and (map? w) (:fn w)) (str (:fn w) "(" (str/join ", " ea) ")")      ; function
      (and (map? w) (:prefix w)) (str (:prefix w) "(" (first ea) ")")
      :else (throw (ex-info (str "no WGSL lowering for op " op) {:op op})))))

(defn- emit-expr
  "Emit a pure elementwise value expression as WGSL (f32). env = {:isym loop-var
   :scalars #{scalar-names}}. The loop var → the WGSL invocation index `gi`;
   scalar params → `U.<name>` (uniform struct members)."
  [expr {:keys [isym scalars] :as env}]
  (cond
    (number? expr) (wgsl-num expr)
    (= expr isym)  "gi"
    (symbol? expr) (let [n (name expr)] (if (contains? scalars n) (str "U." n) n))
    (and (seq? expr) (= 'clojure.core/aget (first expr)))
    (str (name (second expr)) "[" (emit-expr (nth expr 2) env) "]")
    (and (seq? expr) (#{'long 'int} (first expr))) (str "i32(" (emit-expr (second expr) env) ")")
    (and (seq? expr) (#{'float 'double} (first expr))) (str "f32(" (emit-expr (second expr) env) ")")
    (and (seq? expr) (= 'if (first expr)))
    (let [[_ c t e] expr] (str "select(" (emit-expr e env) ", " (emit-expr t env)
                               ", " (emit-expr c env) ")"))
    (and (seq? expr) (= '.invk (first expr)))
    (emit-call (:raster.op/original (meta expr)) (drop 2 expr) env)
    (and (seq? expr) (symbol? (first expr)) (ix/canonical (first expr)))
    (emit-call (first expr) (rest expr) env)
    :else (throw (ex-info (str "unhandled WGSL expr: " (pr-str expr)) {:expr expr}))))

;; ---------------------------------------------------------------------------
;; Module assembly
;; ---------------------------------------------------------------------------

(defn compile-kernel
  "Compile an elementwise-map kernel IR to a WGSL compute shader.
   {:name :params [{:sym :tag}] :ir [:workgroup]} →
   {:wgsl str :array-params [sym] :scalar-params [sym] :n-sym sym :workgroup int}.
   Arrays bind in param order at @binding 0…; the uniform struct (scalars + _n)
   binds last. The host writes arrays as f32 storage buffers + the uniform."
  [{:keys [params ir workgroup] :or {workgroup 64}}]
  (let [array?  (fn [t] (#{'doubles 'floats 'longs 'ints} t))
        intsc?  (fn [t] (#{'long 'int 'Long 'Integer} t))
        arr-params    (filterv #(array? (:tag %)) params)
        scl-params    (filterv #(not (array? (:tag %))) params)
        count-param   (first (filter #(intsc? (:tag %)) scl-params))     ; the element count → U._n
        float-scalars (filterv #(not (intsc? (:tag %))) scl-params)      ; broadcast f32 uniforms
        loopf  (or (find-counted-loop ir) (throw (ex-info "no elementwise loop found" {:ir ir})))
        {:keys [isym stores]} (loop-body-stores loopf)
        _ (when (empty? stores) (throw (ex-info "no aset stores in loop body" {})))
        env     {:isym isym :scalars (set (map #(name (:sym %)) float-scalars))}
        written (set (map #(name (first %)) stores))
        n-arr   (count arr-params)
        arr-decls (map-indexed
                   (fn [i p]
                     (let [nm (name (:sym p))
                           access (if (contains? written nm) "read_write" "read")]
                       (format "@group(0) @binding(%d) var<storage, %s> %s: array<f32>;" i access nm)))
                   arr-params)
        struct-fields (concat (map (fn [p] (str "  " (name (:sym p)) ": f32,")) float-scalars)
                              ["  _n: u32,"])
        uni-decl (str "struct Params {\n" (str/join "\n" struct-fields) "\n}\n"
                      (format "@group(0) @binding(%d) var<uniform> U: Params;" n-arr))
        body (str/join "\n      "
                       (map (fn [[arr idx-e val]]
                              (str (name arr) "[" (emit-expr idx-e env) "] = " (emit-expr val env) ";"))
                            stores))
        src (str (str/join "\n" arr-decls) "\n"
                 uni-decl "\n\n"
                 (format "@compute @workgroup_size(%d)\n" workgroup)
                 "fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n"
                 "  let gi: u32 = g.x;\n"
                 "  if (gi < U._n) {\n"
                 "      " body "\n"
                 "  }\n}\n")]
    {:wgsl src
     :array-params  (mapv :sym arr-params)
     :scalar-params (mapv :sym float-scalars)
     :n-sym         (:sym count-param)
     :workgroup     workgroup
     :written       written}))
