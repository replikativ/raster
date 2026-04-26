(ns raster.compiler.core.types
  "Foundation type utilities for the raster dispatch system.

  Contains:
  - Primitive/array type tag maps
  - Typed function interface creation (TypedFn, fn-interface registry)
  - Parameter parsing (parse-typed-params, annotation->tag/hint)
  - Compound tag utilities
  - Shape protocol and shape rule registry
  - Tag→Class resolution
  - Name mangling
  - Warning system"
  (:require [clojure.string :as str]))

;; ================================================================
;; Type Utilities
;; ================================================================

(def primitive-info
  "Primitive type tag -> boxed class."
  {'long    Long
   'int     Integer
   'double  Double
   'float   Float
   'boolean Boolean
   'byte    Byte
   'short   Short
   'char    Character})

(def primitive-array-info
  "Primitive array type tag -> array class."
  {'doubles (Class/forName "[D")
   'floats  (Class/forName "[F")
   'longs   (Class/forName "[J")
   'ints    (Class/forName "[I")
   'bytes   (Class/forName "[B")
   'shorts  (Class/forName "[S")
   'chars   (Class/forName "[C")
   'booleans (Class/forName "[Z")})

(def tag->java-class
  "Map type tags to java.lang.Class for gen-interface descriptors."
  {'double  Double/TYPE, 'long Long/TYPE, 'int Integer/TYPE, 'float Float/TYPE,
   'boolean Boolean/TYPE, 'byte Byte/TYPE, 'short Short/TYPE, 'char Character/TYPE,
   'doubles (Class/forName "[D"), 'longs (Class/forName "[J"),
   'ints (Class/forName "[I"), 'floats (Class/forName "[F"),
   'bytes (Class/forName "[B"), 'shorts (Class/forName "[S"),
   'chars (Class/forName "[C"), 'booleans (Class/forName "[Z"),
   'void Void/TYPE, 'Object Object})

;; Registry: {[param-tags ret-tag] -> {:iface-name symbol, :iface-class Class, :param-tags [...], :ret-tag symbol}}
(defonce fn-interface-registry (atom {}))

;; Reverse index: {iface-name-symbol -> registry entry}
(defonce fn-interface-by-name (atom {}))

;; TypedFn marker interface — all ftm objects and fn-algebra results implement this.
(definterface TypedFn)
(def TypedFn-class raster.compiler.core.types.TypedFn)

(defn- make-iface-name
  "Build interface name: raster.fn.IFn__<tags>
   Return type is NOT part of the interface name — dispatch is on
   argument types only (like Julia). The .invk method returns Object;
   callers cast as needed. C2 eliminates box/unbox through escape analysis."
  [param-tags]
  (let [tag-str (str/join "_" (map name param-tags))]
    (symbol (str "raster.fn.IFn__" tag-str))))

(def ^:private gen-iface-tag
  "Tags that gen-interface understands as-is (primitives and primitive arrays)."
  #{'double 'long 'int 'float 'boolean 'byte 'short 'char
    'doubles 'longs 'ints 'floats 'bytes 'shorts 'chars 'booleans
    'void 'Object})

(defn tag->iface-type
  "Convert a tag to a symbol gen-interface understands."
  [tag]
  (if (contains? gen-iface-tag tag) tag 'Object))

(defn ensure-fn-interface!
  "Get or create a typed function interface for a given param type signature.
   Return type is NOT part of the interface — dispatch is on argument types only.
   The .invk method always returns Object; callers cast/unbox as needed.
   Returns {:iface-name symbol, :iface-class Class, :param-tags [...]}.
   Thread-safe, idempotent."
  [param-tags]
  (let [iface-param-tags (mapv tag->iface-type param-tags)
        key iface-param-tags]
    (or (get @fn-interface-registry key)
        (locking fn-interface-registry
          (or (get @fn-interface-registry key)
              (let [iface-name (make-iface-name iface-param-tags)
                    _ (eval `(gen-interface
                              :name ~iface-name
                              :extends [raster.compiler.core.types.TypedFn]
                              :methods [[~'invk ~(vec iface-param-tags) ~'Object]]))
                    iface-class (Class/forName (str iface-name))
                    entry {:iface-name iface-name
                           :iface-class iface-class
                           :param-tags iface-param-tags}]
                (swap! fn-interface-registry assoc key entry)
                (swap! fn-interface-by-name assoc iface-name entry)
                entry))))))

(def boxed->primitive-tag
  "Map boxed class names to primitive dispatch tags."
  {'Long    'long
   'Double  'double
   'Integer 'int
   'Float   'float
   'Boolean 'boolean
   'Byte    'byte
   'Short   'short
   'Character 'char
   'Void   'void})

(def array-elem->tag
  "Map (Array elem) element types to array tag symbols."
  {'double  'doubles
   'float   'floats
   'long    'longs
   'int     'ints
   'byte    'bytes
   'short   'shorts
   'char    'chars
   'boolean 'booleans
   'Object  'objects})

;; ================================================================
;; Compound tag utilities
;; ================================================================

(defn compound-tag?
  "True if tag is a compound parametric tag like (Matrix 32 64)."
  [tag]
  (and (sequential? tag) (symbol? (first tag))))

(defn compound-tag-base
  "Extract base type from a compound tag. (Matrix 32 64) -> Matrix."
  [tag]
  (if (compound-tag? tag) (first tag) tag))

(defn compound-tag-params
  "Extract shape parameters from a compound tag. (Matrix 32 64) -> [32 64]."
  [tag]
  (when (compound-tag? tag) (vec (rest tag))))

;; ================================================================
;; Typed parameter parsing
;; ================================================================

;; SoA registry — forward-declared here because annotation->tag references it.
(defonce soa-registry (atom {}))
(defonce soa-reverse-registry (atom {}))

(defn parse-typed-params
  "Parse a parameter vector with :- Type annotations.
   Returns {:params [names...] :annotations [type-or-nil...]}"
  [param-vec]
  (let [items (vec param-vec)]
    (loop [idx 0
           params []
           anns []]
      (if (>= idx (count items))
        {:params params :annotations anns}
        (let [item (nth items idx)]
          (if (and (< (+ idx 2) (count items))
                   (= :- (nth items (inc idx))))
            (recur (+ idx 3)
                   (conj params item)
                   (conj anns (nth items (+ idx 2))))
            (recur (inc idx)
                   (conj params item)
                   (conj anns nil))))))))

(declare annotation->fn-info)

;; Registry for user-defined array-like types (e.g. GA Multivector).
;; Must be declared before annotation->tag which references it.
;; Maps tag symbol → {:element-cast sym, :compute-cast sym}
(defonce array-like-registry (atom {}))

;; Leaf-semantics wrappers. Param marks a tunable parameter (gradient target,
;; SGD updates it). Frozen marks a leaf that participates structurally but is
;; not a gradient target. Params is the outer marker that signals an arg
;; should be compile-time-flattened (consumed by raster.compiler.core.params-flatten;
;; never reaches the JVM signature). All three are transparent for type-tag/hint
;; computation — they pass through to the inner type's tag.
(def leaf-wrapper-tags '#{Param Frozen Params})

(defn leaf-wrapper?
  "True iff annotation is a leaf-semantics wrapper like (Param T) or (Frozen T)."
  [annotation]
  (and (sequential? annotation)
       (contains? leaf-wrapper-tags (first annotation))))

(defn unwrap-leaf-wrapper
  "Strip Param / Frozen wrappers from annotation, returning the inner type.
  Transparent for tag/hint computation; the wrapper kind is consumed
  separately by the flatten pass and the optimizer."
  [annotation]
  (if (leaf-wrapper? annotation)
    (recur (second annotation))
    annotation))

(defn annotation->tag
  "Convert a typedclojure type annotation to a dispatch tag symbol."
  [annotation param]
  (cond
    (nil? annotation)
    (or (:tag (meta param)) 'Object)

    ;; (Param T) / (Frozen T) — leaf-semantics wrappers, transparent for tags
    (leaf-wrapper? annotation)
    (annotation->tag (second annotation) param)

    ;; (Array T) — check SoA registry first
    (and (sequential? annotation) (= 'Array (first annotation))
         (symbol? (second annotation))
         (get @soa-registry (second annotation)))
    (:soa-type-tag (get @soa-registry (second annotation)))

    ;; (Array double) -> doubles, etc.
    ;; (Array Multivector) or (Array raster.ga.core.Multivector) -> Multivector
    ;; (Array Sym) or any non-primitive element -> objects (Object[])
    (and (sequential? annotation) (= 'Array (first annotation)))
    (or (get array-elem->tag (second annotation))
        ;; Check array-like registry by simple name or FQN
        (let [elem (second annotation)
              simple (when (symbol? elem)
                       (symbol (last (clojure.string/split (str elem) #"\."))))]
          (when (or (contains? @array-like-registry elem)
                    (contains? @array-like-registry simple))
            (or simple elem)))
        'objects)

    ;; (Fn [...]) -> fn interface name
    (and (sequential? annotation) (= 'Fn (first annotation)))
    (:iface-name (annotation->fn-info annotation))

    ;; Compound parametric tag: (Matrix 32 64)
    (and (sequential? annotation)
         (symbol? (first annotation))
         (not (#{'Array 'Fn} (first annotation)))
         (every? #(or (number? %) (keyword? %)) (rest annotation)))
    (apply list annotation)

    ;; Boxed primitives -> primitive tags
    (symbol? annotation)
    (or (get boxed->primitive-tag annotation) annotation)

    :else 'Object))

(defn annotation->fn-info
  "Extract function type info from a (Fn [...]) or (Fn [...] RetType) annotation."
  [annotation]
  (when (and (sequential? annotation) (= 'Fn (first annotation)))
    (let [parts (rest annotation)
          fn-param-anns (first parts)
          fn-ret-type (second parts)
          fn-param-tags (mapv #(annotation->tag % nil) fn-param-anns)
          fn-ret-tag (when fn-ret-type
                       (annotation->tag fn-ret-type nil))]
      (let [info (ensure-fn-interface! fn-param-tags)]
        ;; Preserve ret-tag for callers that need it (e.g. TC annotations)
        (if fn-ret-tag (assoc info :ret-tag fn-ret-tag) info)))))

(defn annotation->hint
  "Convert a typedclojure type annotation to a Clojure type hint symbol."
  [annotation]
  (cond
    (nil? annotation) nil
    (leaf-wrapper? annotation) (annotation->hint (second annotation))
    (and (sequential? annotation) (= 'Array (first annotation)))
    (get array-elem->tag (second annotation))
    (and (sequential? annotation) (= 'Fn (first annotation)))
    (:iface-name (annotation->fn-info annotation))
    (= annotation 'Void) nil
    (contains? boxed->primitive-tag annotation) annotation
    (compound-tag? annotation) (compound-tag-base annotation)
    (symbol? annotation) annotation
    :else nil))

(defn annotation->return-hint
  "Convert a return type annotation to a Clojure return type hint."
  [annotation]
  (cond
    (nil? annotation) nil
    (leaf-wrapper? annotation) (annotation->return-hint (second annotation))
    (and (sequential? annotation) (= 'Array (first annotation)))
    (get array-elem->tag (second annotation))
    (= annotation 'Void) nil
    (= annotation 'Long)   'long
    (= annotation 'Double) 'double
    (compound-tag? annotation) (compound-tag-base annotation)
    (symbol? annotation) annotation
    :else nil))

(defn extract-tags
  "Extract dispatch tags from a parsed parameter vector."
  [params annotations]
  (mapv annotation->tag annotations params))

(defn build-hinted-params
  "Build a parameter vector with Clojure type hints derived from annotations."
  [params annotations]
  (mapv (fn [p ann]
          (if-let [hint (annotation->hint ann)]
            (vary-meta p assoc :tag hint)
            p))
        params annotations))

(defn annotation->record-hint
  "Convert annotation to a defrecord field hint. Like annotation->hint but
  maps Double→double, Long→long etc. for primitive defrecord fields."
  [annotation]
  (cond
    (nil? annotation) nil
    (leaf-wrapper? annotation) (annotation->record-hint (second annotation))
    (and (sequential? annotation) (= 'Array (first annotation)))
    (get array-elem->tag (second annotation))
    (= annotation 'Void) nil
    (contains? boxed->primitive-tag annotation) (get boxed->primitive-tag annotation)
    (symbol? annotation) annotation
    :else nil))

(defn build-record-hinted-params
  "Build a parameter vector with primitive type hints for defrecord fields.
  Maps Double→^double, Long→^long, (Array double)→^doubles etc."
  [params annotations]
  (mapv (fn [p ann]
          (if-let [hint (annotation->record-hint ann)]
            (vary-meta p assoc :tag hint)
            p))
        params annotations))

(defn annotation->prim-hint
  "Convert an annotation to a primitive type hint for deftype method params."
  [annotation]
  (cond
    (nil? annotation) nil
    (leaf-wrapper? annotation) (annotation->prim-hint (second annotation))
    (and (sequential? annotation) (= 'Array (first annotation)))
    (let [tag (get array-elem->tag (second annotation))]
      (when (not= tag 'objects) tag))
    (and (sequential? annotation) (= 'Fn (first annotation))) nil
    (= annotation 'Void) nil
    (= annotation 'Long) 'long
    (= annotation 'Double) 'double
    (= annotation 'Integer) 'int
    (= annotation 'Float) 'float
    (= annotation 'Boolean) 'boolean
    (= annotation 'Byte) 'byte
    (= annotation 'Short) 'short
    (= annotation 'Character) 'char
    :else nil))

(defn build-prim-hinted-params
  "Build a parameter vector with primitive type hints for deftype methods."
  [params annotations]
  (mapv (fn [p ann]
          (if-let [hint (annotation->prim-hint ann)]
            (vary-meta p assoc :tag hint)
            p))
        params annotations))

(defn has-primitive-annotations?
  "Check if any annotations involve typed dispatch — primitive types,
   (Fn [...]) annotations, or concrete reference types (not Object/nil).
   When true, a typed IFn__ interface is generated for devirtualization."
  [annotations ret-type]
  (some (fn [ann]
          (or (and ann (contains? boxed->primitive-tag ann))
              (and ann (sequential? ann) (= 'Fn (first ann)))
              (and ann (symbol? ann)
                   (not= ann 'Object)
                   (not (contains? #{'doubles 'floats 'longs 'ints
                                     'bytes 'shorts 'chars 'booleans 'objects} ann))
                   ;; Concrete reference type: resolves to a class via import or FQN
                   (boolean
                    (try (Class/forName (str ann)) true
                         (catch Exception _
                           (try (let [r (ns-resolve *ns* ann)]
                                  (class? r))
                                (catch Exception _ false))))))))
        (if ret-type (conj (vec annotations) ret-type) annotations)))

;; ================================================================
;; Shape protocol for parametric dispatch
;; ================================================================

(defprotocol Shaped
  "Protocol for types that carry compile-time shape parameters."
  (shape [this] "Return a vector of shape dimensions, e.g. [32 64]."))

;; ================================================================
;; Shape rule registry
;; ================================================================

(defonce ^:private shape-rule-registry (atom {}))

(defn register-shape-rule!
  "Register a shape inference rule for a deftm method name."
  [method-name rule-fn]
  (swap! shape-rule-registry assoc method-name rule-fn)
  nil)

(defn infer-shape-from-rules
  "Apply shape rules to infer the output compound tag for a call."
  [head-sym arg-tags]
  (let [base-name (if (namespace head-sym) (symbol (name head-sym)) head-sym)]
    (when-let [rule (get @shape-rule-registry base-name)]
      (rule arg-tags))))

;; ================================================================
;; Tag→Class resolution
;; ================================================================

;; Global registry: short tag symbol -> Class
(defonce tag-class-registry (atom {}))

(declare resolve-type-ref->class)

(defn tag->check-class
  "Resolve tag to the Class used in instanceof checks."
  [tag]
  (let [base (compound-tag-base tag)]
    (or (get primitive-info base)
        (get primitive-array-info base)
        (when (= base 'Object) Object)
        (when-let [entry (get @fn-interface-by-name base)]
          (:iface-class entry))
        (when (symbol? base)
          ;; Parametric types (Dual__Sym etc.) can't be found by resolve —
          ;; check the registry first for those. For normal types, prefer
          ;; resolve to get correct class hierarchy and specificity.
          (if (and (str/includes? (name base) "__")
                   (get @tag-class-registry base))
            (get @tag-class-registry base)
            (if-let [cls (resolve-type-ref->class base)]
              cls
              (get @tag-class-registry base Object))))
        Object)))

(defn resolve-type-ref->class
  "Normalize a type reference symbol to a Class."
  [tag]
  (if (compound-tag? tag)
    (resolve-type-ref->class (compound-tag-base tag))
    (let [resolved
          (or
           (when (and (namespace tag)
                      (str/includes? (namespace tag) "."))
             (try (Class/forName (str tag))
                  (catch ClassNotFoundException _ nil)))
           (when-let [alias-ns (and (namespace tag)
                                    (get (ns-aliases *ns*) (symbol (namespace tag))))]
             (ns-resolve alias-ns (symbol (name tag))))
           (resolve tag)
           (when (and (nil? (namespace tag))
                      (str/includes? (name tag) "."))
             (try (Class/forName (name tag))
                  (catch ClassNotFoundException _ nil))))]
      (cond
        (class? resolved) resolved
        (var? resolved) (let [v @resolved] (when (class? v) v))
        :else nil))))

;; ================================================================
;; Warning system
;; ================================================================

;; Map abstract class name string -> docstring
(defonce abstract-doc-registry (atom {}))

(defn register-abstract-doc! [class-name doc]
  (swap! abstract-doc-registry assoc class-name doc))

(defn abstract-doc [x]
  (let [k (cond (class? x) (.getName ^Class x) (symbol? x) (str x) :else (str x))]
    (get @abstract-doc-registry k)))

(defn warning-suppressed?
  "Return true when warning category is suppressed by metadata."
  [metadata category]
  (let [direct (get metadata :suppress-warnings)
        nested (get-in metadata [:warn :suppress])
        categories (->> [direct nested]
                        (mapcat (fn [spec]
                                  (cond
                                    (nil? spec) []
                                    (keyword? spec) [spec]
                                    (set? spec) spec
                                    (sequential? spec) spec
                                    :else [])))
                        set)]
    (contains? categories category)))

(defn emit-warning!
  "Emit a Raster warning unless it is suppressed by metadata."
  [metadata category message]
  (when-not (warning-suppressed? metadata category)
    (binding [*out* *err*]
      (println message))))

;; ================================================================
;; Name mangling
;; ================================================================

(def op-name-map
  "Map operator characters to safe identifier names."
  {\+ "plus", \- "minus", \* "star", \/ "div", \< "lt", \> "gt",
   \= "eq", \! "bang", \? "pred", \& "amp", \| "pipe", \~ "tilde"})

(defn- flatten-tag
  "Flatten a tag for mangling."
  [tag]
  (if (compound-tag? tag)
    (str/join "_" (map #(str/replace (str %) "." "_dot_") tag))
    (str/replace (name tag) "." "_dot_")))

(defn mangle
  "Mangle name: foo + [long String] -> foo_m_long_String"
  [base-name tags]
  (let [safe-name (let [n (name base-name)]
                    (if (every? #(contains? op-name-map %) n)
                      (str "_" (str/join "" (map op-name-map n)) "_")
                      n))]
    (symbol (str safe-name "_m_"
                 (str/join "_" (map flatten-tag tags))))))

;; ================================================================
;; Array tags
;; ================================================================

(def primitive-array-tags
  "Set of JVM primitive array dispatch tags."
  #{'doubles 'longs 'ints 'floats 'bytes 'shorts 'chars 'booleans 'objects})

(defn register-array-like!
  "Register a type as array-like for broadcast/reduce!/scan expansion.
  element-cast: the cast function for aset (e.g. 'double)
  compute-cast: the cast function for scalar computation (e.g. 'double)"
  [tag element-cast compute-cast]
  (swap! array-like-registry assoc tag {:element-cast element-cast :compute-cast compute-cast}))

(def array-tags
  "Set of dispatch tags that represent arrays.
  Includes both JVM primitive arrays and registered array-like types."
  primitive-array-tags)

(defn array-tag?
  "True if tag represents an array or array-like type."
  [tag]
  (or (contains? primitive-array-tags tag)
      (contains? @array-like-registry tag)))

(def primitive-array-tag->cast
  "Map primitive array tag to the cast function for aset (store)."
  {'doubles 'double, 'floats 'float,
   'longs 'long, 'ints 'int,
   'bytes 'byte, 'shorts 'short,
   'chars 'char, 'booleans 'boolean})

(def primitive-array-tag->compute-cast
  "Map primitive array tag to the type for scalar computation."
  {'doubles 'double, 'floats 'float,
   'longs 'long, 'ints 'int,
   'bytes 'long, 'shorts 'long,
   'chars 'long, 'booleans nil})

(defn array-tag->cast
  "Get cast function for aset. Checks primitives then array-like registry."
  [tag]
  (or (get primitive-array-tag->cast tag)
      (:element-cast (get @array-like-registry tag))))

(defn array-tag->compute-cast
  "Get cast function for scalar computation. Checks primitives then registry."
  [tag]
  (or (get primitive-array-tag->compute-cast tag)
      (:compute-cast (get @array-like-registry tag))))

;; ================================================================
;; Primitive array element types
;; ================================================================

(def primitive-array-element-types
  "Map from primitive array type tags to their element types."
  {'doubles 'double, 'longs 'long, 'ints 'int, 'floats 'float,
   'bytes 'byte, 'shorts 'short, 'chars 'char, 'booleans 'boolean})

(def primitive->array-tag
  "Maps primitive/boxed type tags to their array type tags."
  {'double 'doubles 'Double 'doubles
   'float  'floats  'Float  'floats
   'long   'longs   'Long   'longs
   'int    'ints    'Integer 'ints})

;; ================================================================
;; Unified Type Metadata Keys
;;
;; Namespaced keywords for type information that flows through the IR
;; as metadata on symbols and forms. Each concern has its own namespace
;; to make producer/consumer relationships clear.
;;
;; Producer → Consumer chains:
;;   :raster.type/tag        walker → all passes, bytecode, GPU
;;   :raster.type/fn-info    walker → walker (fn-args-safe?), bytecode (invokedynamic)
;;   :raster.type/element    walker → walker (parametric dispatch)
;;   :raster.type/source     walker → diagnostics (explain-types)
;;   :raster.buffer/hoistable  buffer-fuse → hoist, mem-merge, resolve-alength
;;   :raster.buffer/write-mode buffer-fuse → hoist, closure (zero-fill)
;;   :raster.effect/effectful  inline, mem-merge → DCE
;;   :raster.op/original       walker → normalize, rewalk, bytecode, GPU
;; ================================================================

;; --- Type keys (set by walker, read by all) ---

(defn sym-type-tag
  "Read the dispatch type tag from a symbol's metadata.
  Prefers :raster.type/tag, falls back to :tag."
  [sym]
  (let [m (meta sym)]
    (or (:raster.type/tag m) (:tag m))))

(defn sym-fn-info
  "Read typed function info from a symbol's metadata.
  Returns {:iface-name sym, :param-tags [...], :ret-tag sym} or nil."
  [sym]
  (:raster.type/fn-info (meta sym)))

(defn sym-element-tag
  "Read the element type tag for an Object[] parameter."
  [sym]
  (:raster.type/element (meta sym)))

(defn fn-type-tag?
  "True if this tag represents a typed function interface (IFn__)."
  [tag]
  (and tag (symbol? tag) (.contains (str tag) "IFn__")))

(defn with-type-info
  "Attach type metadata to a symbol. Preserves existing metadata.
  Sets both :raster.type/tag and :tag (for Clojure compat).
  Optional keys: :fn-info, :element, :source."
  [sym tag & {:keys [fn-info element source]}]
  (cond-> (vary-meta sym assoc
                     :raster.type/tag tag
                     :tag tag)
    fn-info (vary-meta assoc :raster.type/fn-info fn-info)
    element (vary-meta assoc :raster.type/element element)
    source  (vary-meta assoc :raster.type/source source)))

(defn merge-type-info
  "Merge type metadata from one symbol onto another."
  [target-sym source-sym]
  (let [sm (meta source-sym)
        type-keys [:raster.type/tag :raster.type/fn-info :raster.type/element
                   :raster.type/source :tag]]
    (vary-meta target-sym merge (select-keys sm type-keys))))

;; --- Buffer keys (set by buffer-fuse, read by hoist/mem-merge/closure) ---

(defn sym-hoistable?
  "True if symbol is a hoistable buffer (set by buffer-fuse)."
  [sym]
  (boolean (:raster.buffer/hoistable (meta sym))))

(defn sym-write-mode
  "Get the write mode for a hoistable buffer. :overwrite, :accumulate, or :read-only."
  [sym]
  (:raster.buffer/write-mode (meta sym)))

;; --- Effect keys (set by inline/effects, read by DCE) ---

(defn sym-effectful?
  "True if this binding has side effects and must not be removed by DCE."
  [sym]
  (boolean (:raster.effect/effectful (meta sym))))

;; --- Semantic keys (on forms, not symbols) ---

(defn form-original-op
  "Get the original operator symbol from a walked call form."
  [form]
  (:raster.op/original (meta form)))

;; --- Preserved keys for CSE ---

(def ^:const preserved-type-keys
  "Metadata keys that CSE must preserve during normalization.
  These carry type information that downstream passes need."
  [:raster.type/tag :raster.type/fn-info :raster.type/element :raster.type/source])

;; --- JVM numeric widening (used by dispatch and inference) ---

(def numeric-widening
  "JVM numeric widening conversions for dispatch matching."
  {Integer #{Long}
   Short #{Integer Long}
   Byte #{Short Integer Long}})
