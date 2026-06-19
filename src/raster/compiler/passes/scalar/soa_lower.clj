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
  (:require [raster.compiler.core.types :as types]
            [clojure.walk :as walk]))

(defn field-arr-sym [soa-sym field-name]
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
                      [sym {:scalar-tag scalar-tag :soa-tag tag :fields (:fields info)}])))
                param-specs))))

(defn collect-soa-env
  "Body-tag-based SoA detection (the shared replacement for the GPU's
   collect-soa-expansions): scan body for symbols whose :tag is a registered SoA
   type → {soa-sym → {:scalar-tag :soa-tag :fields}}."
  [body]
  (let [rev @types/soa-reverse-registry reg @types/soa-registry acc (atom {})]
    (walk/postwalk
     (fn [f]
       (when (and (symbol? f) (:tag (meta f)))
         (when-let [scalar (get rev (:tag (meta f)))]
           (when-let [info (get reg scalar)]
             (swap! acc assoc (symbol (name f))
                    {:scalar-tag scalar :soa-tag (:tag (meta f)) :fields (:fields info)}))))
       f)
     body)
    @acc))

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

(defn- local-name [head] (let [s (str head)] (subs s (inc (.lastIndexOf s "/")))))

(defn- constructor->scalar-tag
  "(->Cx …) head → scalar tag 'Cx if registered, else nil."
  [head]
  (let [local (local-name head)]
    (when (.startsWith local "->")
      (let [t (symbol (subs local 2))]
        (when (get @types/soa-registry t) t)))))

(defn- field-access-head? [head]
  (and (symbol? head)
       (let [s (str head)]
         (and (.startsWith s ".") (> (count s) 1) (not (#{".invk" "."} s))))))

(defn- aget-head? [head]
  (or (#{'aget 'clojure.core/aget 'raster.arrays/aget} head)
      (.startsWith (local-name head) "aget-")))

(defn- aset-head? [head]
  (or (#{'aset 'clojure.core/aset 'raster.arrays/aset 'raster.arrays/aset!} head)
      (.startsWith (local-name head) "aset-")))

;; ctx: {:soa {soa-sym → soa-info}  :exploded {local-sym → {field-name → expr}}}
(defn- soa-info-for
  "soa-info for a symbol, via param env or its :tag (the GPU idiom: aget on a
   SoA-typed symbol). nil if not a SoA value."
  [ctx sym]
  (when (symbol? sym)
    (or (get (:soa ctx) (symbol (name sym)))
        (when-let [scalar (get @types/soa-reverse-registry (:tag (meta sym)))]
          {:scalar-tag scalar :fields (:fields (get @types/soa-registry scalar))}))))

(declare lower)

(defn- explode
  "If form is value-type-valued and explodable, return {field-name → lowered-field-expr};
   else nil. Handles: exploded local sym, construction, SoA element read (named accessor
   or plain aget on a SoA-typed symbol)."
  [ctx form]
  (cond
    (symbol? form) (get-in ctx [:exploded form])
    (seq? form)
    (let [head (first form)]
      (cond
        (constructor->scalar-tag head)
        (let [fields (:fields (get @types/soa-registry (constructor->scalar-tag head)))]
          (zipmap (map :name fields) (map #(lower ctx %) (rest form))))
        (and (aget-head? head) (soa-info-for ctx (second form)))
        (let [soa  (symbol (name (second form)))
              i    (lower ctx (nth form 2))
              info (soa-info-for ctx (second form))]
          (into {} (map (fn [{nm :name}]
                          [nm (list 'clojure.core/aget (field-arr-sym soa nm) i)])
                        (:fields info))))
        :else nil))
    :else nil))

(defn- lower
  "Scalar-replace SoA/value-type access. ctx threads SoA params + exploded locals."
  [ctx form]
  (cond
    ;; bare exploded local reaching a non-projection position → escape (unsupported v1)
    (and (symbol? form) (contains? (:exploded ctx) form))
    (throw (ex-info (str "value-type local escapes (used non-projectively): " form)
                    {:sym form}))

    ;; (.field x) where x explodes → the field expr
    (and (seq? form) (field-access-head? (first form)))
    (let [ex (explode ctx (second form))
          fname (subs (str (first form)) 1)]
      (if (and ex (contains? ex fname))
        (get ex fname)
        (apply list (map #(lower ctx %) form))))     ; non-value-type field access → recurse

    ;; let* — bind exploded value-locals into ctx and DROP their bindings
    (and (seq? form) (= 'let* (first form)))
    (let [[_ binds & body] form
          [ctx' out] (reduce (fn [[c bs] [s e]]
                               (if-let [ex (explode c e)]
                                 [(assoc-in c [:exploded s] ex) bs]   ; virtual: drop binding
                                 [c (conj bs s (lower c e))]))
                             [ctx []] (partition 2 binds))]
      (apply list 'let* (vec out) (map #(lower ctx' %) body)))

    ;; SoA store: (aset[-k]! soa i v) → per-field stores
    (and (seq? form) (aset-head? (first form)) (soa-info-for ctx (second form)))
    (let [[_ soa-raw i v] form
          soa  (symbol (name soa-raw))
          info (soa-info-for ctx soa-raw)
          ex   (explode ctx v)
          i'   (lower ctx i)]
      (if ex
        (cons 'do (map (fn [{nm :name}]
                         (list 'clojure.core/aset (field-arr-sym soa nm) i' (get ex nm)))
                       (:fields info)))
        (throw (ex-info "aset of a non-explodable value-type" {:value v}))))

    ;; a value-type value reaching a generic (non-consuming) position → escape
    (and (seq? form) (explode ctx form))
    (throw (ex-info "value-type value in a non-consuming position (escapes)" {:form form}))

    (seq? form) (apply list (map #(lower ctx %) form))
    (vector? form) (mapv #(lower ctx %) form)
    :else form))

(defn lower-body
  "Scalar-replace value-type access in body given a SoA env (the shared lowering
   the GPU pass calls after computing its env via collect-soa-env)."
  [soa-env body]
  (lower {:soa soa-env :exploded {}} body))

(defn soa-lower
  "Top-level (wasm): expand SoA params + scalar-replace value-type access.
   Returns {:body body' :params param-specs' :soa-expansion {soa-sym → info}}.
   No-op when no SoA params."
  [body param-specs]
  (let [soa-env (soa-param-env param-specs)]
    (if (empty? soa-env)
      {:body body :params param-specs :soa-expansion {}}
      {:body          (lower-body soa-env body)
       :params        (expand-params param-specs soa-env)
       :soa-expansion soa-env})))
