(ns raster.compiler.passes.scalar.soa-lower
  "Scalar-replacement of value-type aggregates (SROA), driven by the
   backend-agnostic SoA registry (raster.compiler.core.types/soa-registry,
   populated by register-soa! at defvalue expansion).

   This is the shared SoA-lowering for non-JVM backends (Track A wasm now; the
   GPU emitter's emit-time *soa-expansions* path is slated to migrate onto this
   in a later step). It rewrites the IR so value types disappear *before* any
   backend sees them — both backends then emit plain primitive array ops:

     SoA param `os :- CxSoA`          → per-field array params `os_re`, `os_im`
     (aget-cx soa i)                  → exploded {re (aget soa_re i), im (aget soa_im i)}
     (.re <explodable>)               → the re field-expr (projection)
     (->Cx a b)                       → exploded {re a, im b}
     (aset-cx! soa i <explodable>)    → (do (aset soa_re i …) (aset soa_im i …))

   v1 scope: value ops written inline (or already inlined). A value-type *helper
   deftm* (a `.invk` returning a value type whose body is a bare constructor) is
   NOT inlined by the current inliner, so such calls don't explode yet — that's a
   separable follow-on (extend inlinable-body? / inline value-type deftms)."
  (:require [raster.compiler.core.types :as types]))

(defn- field-arr-sym [soa-sym field-name]
  (symbol (str (name soa-sym) "_" (name field-name))))

(defn soa-param-env
  "From ordered param-specs [{:sym :tag}], find SoA-typed params and build
   {param-sym → {:scalar-tag :fields [{:name :element-tag :array-tag}]}}."
  [param-specs]
  (let [rev @types/soa-reverse-registry
        reg @types/soa-registry]
    (into {}
          (keep (fn [{:keys [sym tag]}]
                  (when-let [scalar-tag (get rev tag)]
                    (when-let [info (get reg scalar-tag)]
                      [sym {:scalar-tag scalar-tag :fields (:fields info)}])))
                param-specs))))

(defn expand-params
  "Replace each SoA param with its per-field array params (in field order)."
  [param-specs soa-env]
  (vec (mapcat (fn [{:keys [sym tag] :as p}]
                 (if-let [info (get soa-env sym)]
                   (mapv (fn [{:keys [name array-tag]}]
                           {:sym (field-arr-sym sym name) :tag array-tag})
                         (:fields info))
                   [p]))
               param-specs)))

(defn- constructor->scalar-tag
  "(->Cx …) head → scalar tag 'Cx if registered, else nil."
  [head]
  (let [s (str head)
        local (subs s (inc (.lastIndexOf s "/")))]   ; strip ns qualifier
    (when (.startsWith local "->")
      (let [t (symbol (subs local 2))]
        (when (get @types/soa-registry t) t)))))

(defn- soa-accessor? [prefix head soa-sym soa-env]
  (and (symbol? soa-sym)
       (get soa-env (symbol (name soa-sym)))
       (let [s (str head)
             local (subs s (inc (.lastIndexOf s "/")))]
         (.startsWith local prefix))))

(defn- field-access-head? [head]
  (and (symbol? head)
       (let [s (str head)]
         (and (.startsWith s ".") (> (count s) 1) (not (#{".invk" "."} s))))))

(declare lower)

(defn- explode
  "If form is value-type-valued and explodable, return {field-name → lowered-field-expr}; else nil."
  [soa-env form]
  (when (seq? form)
    (let [head (first form)]
      (cond
        ;; construction (->Cx a b …)
        (constructor->scalar-tag head)
        (let [scalar (constructor->scalar-tag head)
              fields (:fields (get @types/soa-registry scalar))]
          (zipmap (map :name fields) (map #(lower soa-env %) (rest form))))
        ;; SoA element read (aget-cx soa i)
        (soa-accessor? "aget-" head (second form) soa-env)
        (let [soa (symbol (name (second form)))
              i   (lower soa-env (nth form 2))
              info (get soa-env soa)]
          (into {} (map (fn [{nm :name}]
                          [nm (list 'clojure.core/aget (field-arr-sym soa nm) i)])
                        (:fields info))))
        :else nil))))

(defn- lower
  "Rewrite a form, scalar-replacing SoA/value-type access. Returns the rewritten form."
  [soa-env form]
  (cond
    (and (seq? form) (field-access-head? (first form)))
    (let [ex (explode soa-env (second form))
          fname (subs (str (first form)) 1)]            ; ".re" → "re" (field :name is a string)
      (if (and ex (contains? ex fname))
        (get ex fname)
        (apply list (map #(lower soa-env %) form))))   ; not explodable → recurse

    (and (seq? form) (soa-accessor? "aset-" (first form) (second form) soa-env))
    (let [[_ soa-raw i v] form
          soa (symbol (name soa-raw))
          info (get soa-env soa)
          ex (explode soa-env v)
          i' (lower soa-env i)]
      (if ex
        (cons 'do (map (fn [{nm :name}]
                         (list 'clojure.core/aset (field-arr-sym soa nm) i' (get ex nm)))
                       (:fields info)))
        (apply list (map #(lower soa-env %) form))))

    (seq? form) (apply list (map #(lower soa-env %) form))
    (vector? form) (mapv #(lower soa-env %) form)
    :else form))

(defn soa-lower
  "Top-level: expand SoA params + scalar-replace value-type access in the body.
   Returns {:body body' :params param-specs'}. No-op when no SoA params."
  [body param-specs]
  (let [soa-env (soa-param-env param-specs)]
    (if (empty? soa-env)
      {:body body :params param-specs}
      {:body   (lower soa-env body)
       :params (expand-params param-specs soa-env)})))
