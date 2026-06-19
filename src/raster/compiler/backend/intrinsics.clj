(ns raster.compiler.backend.intrinsics
  "Single source of truth for how primitive numeric operators + intrinsics lower
   to each code-generation backend (wasm, C/OpenCL, GLSL, WGSL).

   Per the compiler design rule \"Centralize operator classification\": code-gen
   backends must not each carry their own hardcoded `#{…}` / `case` of operator
   symbols. They are thin dispatchers over this one table — looking up the
   canonical op and reading the lowering for the current element type. Adding a
   new primitive op = one row here, and the row states exactly which backends
   support it (a nil/absent facet = unsupported on that backend, surfaced as a
   clear error rather than a silent miscompile).

   Canonical key: a keyword naming the operation (:+ :< :sqrt :mod …). Callers
   normalize whatever they hold — a `raster.numeric/+` symbol, a bare `+`, a
   `Math/sqrt`, a comparison name, or a mangled devirtualized impl prefix
   (`_plus__m_double_double`) — to the canonical key via `canonical`.

   Lowering facets per row:
     :arity   1 or 2
     :kind    :infix (binary operator)  | :cmp (binary, bool result)
              :fn    (function call)    | :special (custom per-backend sequence)
     :wasm    {vt → encoder-opcode-keyword}  (vt ∈ #{:f64 :f32 :i32}); absent = unsupported
     :c       C/OpenCL form — infix string, or {:fn name} (GLSL fn override via :glsl)
     :wgsl    WGSL form — infix string or {:fn name}"
  (:require [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; The table — canonical op key → lowering facets
;; ---------------------------------------------------------------------------

(defn- vt3 [f64 f32 i32] (cond-> {} f64 (assoc :f64 f64) f32 (assoc :f32 f32) i32 (assoc :i32 i32)))
(defn- math1 [f64 f32 cfn] {:arity 1 :kind :fn :wasm (vt3 f64 f32 nil) :c {:fn cfn} :wgsl {:fn cfn}})

(def table
  {;; binary arithmetic
   :+ {:arity 2 :kind :infix :wasm (vt3 :f64.add :f32.add :i32.add) :c "+" :wgsl "+"}
   :- {:arity 2 :kind :infix :wasm (vt3 :f64.sub :f32.sub :i32.sub) :c "-" :wgsl "-"}
   :* {:arity 2 :kind :infix :wasm (vt3 :f64.mul :f32.mul :i32.mul) :c "*" :wgsl "*"}
   :div {:arity 2 :kind :infix :wasm (vt3 :f64.div :f32.div :i32.div_s) :c "/" :wgsl "/"}
   ;; comparisons (bool result)
   :lt {:arity 2 :kind :cmp :wasm (vt3 :f64.lt :f32.lt :i32.lt_s) :c "<" :wgsl "<"}
   :gt {:arity 2 :kind :cmp :wasm (vt3 :f64.gt :f32.gt :i32.gt_s) :c ">" :wgsl ">"}
   :le {:arity 2 :kind :cmp :wasm (vt3 :f64.le :f32.le :i32.le_s) :c "<=" :wgsl "<="}
   :ge {:arity 2 :kind :cmp :wasm (vt3 :f64.ge :f32.ge :i32.ge_s) :c ">=" :wgsl ">="}
   :eq {:arity 2 :kind :cmp :wasm (vt3 :f64.eq :f32.eq :i32.eq) :c "==" :wgsl "=="}
   :ne {:arity 2 :kind :cmp :wasm (vt3 :f64.ne :f32.ne :i32.ne) :c "!=" :wgsl "!="}
   ;; integer remainder / modulo / quotient
   :rem  {:arity 2 :kind :special :wasm {:i32 :i32.rem_s} :c "%" :wgsl "%"}
   :mod  {:arity 2 :kind :special :c :floored-mod :wgsl :floored-mod}
   :quot {:arity 2 :kind :infix :wasm {:i32 :i32.div_s} :c "/" :wgsl "/"}
   ;; math — unary
   :sqrt  (math1 :f64.sqrt :f32.sqrt "sqrt")
   :abs   {:arity 1 :kind :fn :wasm (vt3 :f64.abs :f32.abs nil) :c {:fn "fabs" :glsl "abs"} :wgsl {:fn "abs"}}
   :floor (math1 :f64.floor :f32.floor "floor")
   :ceil  {:arity 1 :kind :fn :c {:fn "ceil"} :wgsl {:fn "ceil"}}      ; no wasm opcode (use floor/neg)
   :round {:arity 1 :kind :fn :c {:fn "round"} :wgsl {:fn "round"}}
   :neg   {:arity 1 :kind :fn :wasm (vt3 :f64.neg :f32.neg nil) :c {:prefix "-"} :wgsl {:prefix "-"}}
   ;; math — binary
   :min {:arity 2 :kind :fn :wasm (vt3 :f64.min :f32.min nil) :c {:fn "fmin" :glsl "min"} :wgsl {:fn "min"}}
   :max {:arity 2 :kind :fn :wasm (vt3 :f64.max :f32.max nil) :c {:fn "fmax" :glsl "max"} :wgsl {:fn "max"}}
   ;; transcendentals — wasm has no opcode; sin/cos/tan/exp lower to an inline
   ;; polynomial (:wasm :poly, see backend.wasm.transcendental). log/pow/fma need
   ;; exponent-bit ops the encoder lacks → no :wasm facet (clear error). GPU has all.
   :sin {:arity 1 :kind :fn :wasm :poly :c {:fn "sin"} :wgsl {:fn "sin"}}
   :cos {:arity 1 :kind :fn :wasm :poly :c {:fn "cos"} :wgsl {:fn "cos"}}
   :tan {:arity 1 :kind :fn :wasm :poly :c {:fn "tan"} :wgsl {:fn "tan"}}
   :exp {:arity 1 :kind :fn :wasm :poly :c {:fn "exp"} :wgsl {:fn "exp"}}
   :log {:arity 1 :kind :fn :c {:fn "log"} :wgsl {:fn "log"}}
   :pow {:arity 2 :kind :fn :c {:fn "pow"} :wgsl {:fn "pow"}}
   :fma {:arity 3 :kind :fn :c {:fn "fma"} :wgsl {:fn "fma"}}})

;; ---------------------------------------------------------------------------
;; Normalization: any op form → canonical key
;; ---------------------------------------------------------------------------

;; local-name → canonical key (covers raster.numeric/Math/clojure.core syms and
;; bare names; arithmetic + comparison spellings both routed here).
(def ^:private name->key
  {"+" :+ "-" :- "*" :* "/" :div
   "<" :lt ">" :gt "<=" :le ">=" :ge "==" :eq "=" :eq "not=" :ne "!=" :ne
   "rem" :rem "unchecked-remainder-int" :rem "mod" :mod "quot" :quot
   "sqrt" :sqrt "abs" :abs "floor" :floor "ceil" :ceil "round" :round
   "min" :min "max" :max "sin" :sin "cos" :cos "tan" :tan "exp" :exp "log" :log
   "pow" :pow "fma" :fma})

;; mangled devirtualized prefix → canonical key (e.g. _plus__m_double_double)
(def ^:private mangled-prefix->key
  {"_plus_" :+ "_minus_" :- "_star_" :* "_div_" :div
   "_lt_" :lt "_gt_" :gt "_lteq_" :le "_gteq_" :ge "_eq_" :eq})

(defn canonical
  "Normalize an op form to its canonical key, or nil if not a known intrinsic.
   Accepts a symbol (qualified/bare/Math), a string name, or a keyword."
  [op]
  (cond
    (keyword? op) (when (contains? table op) op)
    (string? op)  (name->key op)
    (symbol? op)  (or (name->key (name op))
                      (when-let [i (clojure.string/index-of (name op) "_m_")]
                        (mangled-prefix->key (subs (name op) 0 i))))
    :else nil))

(defn descriptor [op] (get table (canonical op)))

;; ---------------------------------------------------------------------------
;; wasm accessor — canonical key + element vt → encoder opcode keyword
;; ---------------------------------------------------------------------------
(defn wasm-op
  "Encoder opcode keyword for op at element valtype vt, or nil if the op has no
   wasm lowering for that vt (caller surfaces a clear error)."
  [op vt]
  (let [w (get-in table [(canonical op) :wasm])]
    (when (map? w) (get w vt))))

(defn wasm-poly?
  "True when op lowers to an inline polynomial on wasm (transcendentals with no
   native opcode — see backend.wasm.transcendental)."
  [op]
  (= :poly (get-in table [(canonical op) :wasm])))

(defn kind [op] (:kind (descriptor op)))
(defn arity [op] (:arity (descriptor op)))

;; ---------------------------------------------------------------------------
;; C / OpenCL / GLSL accessor — for the GPU c-emit backend.
;; Resolves a *mangled* devirtualized impl name (prefix before _m_) and returns
;; the c-emit consumption shape {:kind :infix|:fn|:floored-mod :op str}, or nil
;; (→ caller falls back to a generated helper call). glsl? picks GLSL fn names.
;; ---------------------------------------------------------------------------
(defn c-lowering
  [mangled-name glsl?]
  (when-let [i (str/index-of mangled-name "_m_")]
    (when-let [d (descriptor (subs mangled-name 0 i))]
      (let [c (:c d)]
        (cond
          (= :floored-mod c) {:kind :floored-mod}
          (string? c)        {:kind :infix :op c}
          (and (map? c) (:fn c)) {:kind :fn :op (if (and glsl? (:glsl c)) (:glsl c) (:fn c))}
          :else nil)))))
