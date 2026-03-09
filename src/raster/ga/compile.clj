(ns raster.ga.compile
  "Auto-compile specialized Valhalla value classes for Geometric Algebra.

   Given a metric signature (p,q,r), generates:
   - A defvalue class with named ^double fields (e.g. Multivector_3_0_0)
   - Fully unrolled geometric product, wedge, inner product
   - Involutions (reverse, grade-involution, conjugate)
   - Grade selection, norm, Hodge dual
   - raster.numeric +, -, * dispatch
   - Constructors and display

   Dimension gate: n ≤ 6 (64 fields max).

   Usage:
     (require '[raster.ga.compile :refer [compile-ga!]])
     (def info (compile-ga! (vga 3)))
     ;; => {:class-name ..., :dim 8, :field-names [...], ...}"
  (:refer-clojure :exclude [+ - * /
                            aget aset alength aclone
                            reverse])
  (:require [raster.core :refer [deftm defvalue]]
            [raster.compiler.backend.jvm.valhalla :refer [load-value-class!]]
            [raster.compiler.core.inference :refer [register-field-types!]]
            [raster.types.algebraic-types]
            [raster.numeric :as num]
            [raster.ga.core :as ga :refer [sig-dim algebra-dim
                                           blade-grade blade-product-sign blade-product-result
                                           blade-index index-blade
                                           get-blade-order get-grade-table
                                           collect-contribs sum-expr term-expr]])
  (:import [raster.ga.core Signature]
           [raster.types.algebraic_types GradedAlgebra]))

;; ================================================================
;; Blade naming — bit-pattern to field name
;; ================================================================

(defn blade-bits->field-name
  "Convert blade bit-pattern to a field name string.
   0 → \"s\", 0b001 → \"e1\", 0b011 → \"e12\", 0b111 → \"e123\", etc."
  [^long bits ^long n]
  (if (clojure.core/== bits 0)
    "s"
    (let [sb (StringBuilder. "e")]
      (dotimes [i n]
        (when-not (clojure.core/== (clojure.core/bit-and bits (bit-shift-left 1 i)) 0)
          (.append sb (inc i))))
      (.toString sb))))

;; ================================================================
;; Class naming
;; ================================================================

(defn sig->class-suffix
  "Signature to class name suffix: p_q_r"
  [^Signature sig]
  (str (.p sig) "_" (.q sig) "_" (.r sig)))

(defn sig->class-name
  "Full class name for a compiled GA value class."
  [^Signature sig]
  (str "raster.ga.compile.Multivector_" (sig->class-suffix sig)))

(defn sig->class-sym
  "Simple class symbol (unqualified)."
  [^Signature sig]
  (symbol (str "Multivector_" (sig->class-suffix sig))))

;; ================================================================
;; Field order — grade-then-lex, matching ga.clj blade ordering
;; ================================================================

(defn- ordered-field-specs
  "Generate ordered [{:name \"s\" :tag double :bits 0 :index 0} ...] for signature."
  [sig]
  (let [n (sig-dim sig)
        dim (algebra-dim sig)]
    (mapv (fn [i]
            (let [bits (index-blade i n)]
              {:name (blade-bits->field-name bits n)
               :tag 'double
               :bits bits
               :index i}))
          (range dim))))

;; ================================================================
;; Code generation helpers
;; ================================================================

(defn- ga-eval
  "Eval a form in the raster.ga.compile namespace, regardless of caller's *ns*."
  [form]
  (binding [*ns* (the-ns 'raster.ga.compile)]
    (eval form)))

(defn- field-sym
  "Symbol for accessing a field: prefix_fieldname"
  [prefix field-name]
  (symbol (str prefix "_" field-name)))

(defn- extract-fields-let
  "Generate let bindings that extract all fields from a multivector into locals.
   Returns a vector of [sym expr sym expr ...]"
  [mv-sym prefix field-specs class-sym]
  (vec (mapcat (fn [{:keys [name]}]
                 [(field-sym prefix name)
                  (list (symbol (str "." name))
                        (vary-meta mv-sym assoc :tag class-sym))])
               field-specs)))

;; ================================================================
;; Gen: defvalue class
;; ================================================================

(def ^:private valhalla?*
  "Check if Valhalla value classes are available at runtime."
  (try
    (Class/forName "java.lang.classfile.ClassFile")
    (>= (.feature (Runtime/version)) 27)
    (catch ClassNotFoundException _ false)))

(defn gen-defvalue-class!
  "Generate and load a class for the given GA signature.
   On Valhalla JDK 27+, generates a real value class. Otherwise, generates a
   defrecord via eval as fallback.
   Returns {:class Class, :class-name String, :class-sym Symbol, :fq-sym Symbol, :field-specs [...]}"
  [sig]
  (let [n (sig-dim sig)
        dim (algebra-dim sig)
        class-name (sig->class-name sig)
        class-sym (sig->class-sym sig)
        ;; Fully qualified symbol for use in (new ...) forms
        fq-sym (symbol class-name)
        fspecs (ordered-field-specs sig)
        lvc-specs (mapv #(select-keys % [:name :tag]) fspecs)]
    (if valhalla?*
      ;; Valhalla path: generate value class bytecode
      (let [iface-cd (java.lang.constant.ClassDesc/of "raster.types.algebraic_types" "GradedAlgebra")
            gen-bytes @(resolve 'raster.compiler.backend.jvm.valhalla/generate-value-class-bytes)
            bytes (gen-bytes class-name lvc-specs :interfaces [iface-cd])
            base-loader (clojure.lang.RT/baseLoader)
            cls (if (instance? clojure.lang.DynamicClassLoader base-loader)
                  (.defineClass ^clojure.lang.DynamicClassLoader base-loader class-name bytes nil)
                  (load-value-class! class-name lvc-specs :interfaces [iface-cd]))]
        (.importClass (the-ns 'raster.ga.compile) cls)
        (register-field-types! class-sym
                               (into {} (map (fn [{:keys [name]}] [name 'double]) fspecs)))
        (let [factory-sym (symbol (str "->" class-sym))
              param-syms (mapv (fn [f] (symbol (:name f))) lvc-specs)
              ga-compile-ns (the-ns 'raster.ga.compile)]
          (let [f (ga-eval `(fn ~param-syms
                              (new ~fq-sym ~@(map (fn [f] `(double ~(symbol (:name f)))) lvc-specs))))]
            (intern ga-compile-ns factory-sym f))))
      ;; Fallback: generate defrecord via eval
      (let [field-syms (mapv (fn [f] (with-meta (symbol (:name f)) {:tag 'double})) lvc-specs)
            ga-compile-ns (the-ns 'raster.ga.compile)
            ns-class-prefix (.replace (str (.getName ga-compile-ns)) "-" "_")]
        (binding [*ns* ga-compile-ns]
          (eval `(defrecord ~class-sym ~field-syms)))
        (let [cls (Class/forName (str ns-class-prefix "." class-sym))]
          (.importClass ga-compile-ns cls)
          (register-field-types! class-sym
                                 (into {} (map (fn [{:keys [name]}] [name 'double]) fspecs)))
          (let [factory-sym (symbol (str "->" class-sym))
                param-syms (mapv (fn [f] (symbol (:name f))) lvc-specs)]
            (let [f (ga-eval `(fn ~param-syms
                                (~(symbol (str "raster.ga.compile/->" class-sym))
                                 ~@(map (fn [f] `(double ~(symbol (:name f)))) lvc-specs))))]
              (intern ga-compile-ns factory-sym f))))))
    {:class (Class/forName class-name)
     :class-name class-name
     :class-sym class-sym
     :fq-sym fq-sym
     :field-specs fspecs
     :dim dim
     :n n}))

;; ================================================================
;; Gen: products (geometric, wedge, inner)
;; ================================================================

(defn- gen-product-body
  "Generate the body of a product function as a code form.
   contribs: {result-index -> [[i j sign] ...]}
   field-specs: ordered field specs
   class-sym: symbol for the value class
   Returns a (let [...] (new ...)) form."
  [contribs field-specs class-sym fq-sym]
  (let [a-sym 'a
        b-sym 'b
        a-binds (extract-fields-let a-sym "a" field-specs class-sym)
        b-binds (extract-fields-let b-sym "b" field-specs class-sym)
        ;; For each output field, generate the sum expression
        component-exprs
        (mapv (fn [{:keys [index]}]
                (let [terms (get contribs index)]
                  (if (seq terms)
                    ;; Build sum of sign * a_field[i] * b_field[j]
                    (let [term-forms
                          (map (fn [[i j sign]]
                                 (let [a-f (field-sym "a" (:name (nth field-specs i)))
                                       b-f (field-sym "b" (:name (nth field-specs j)))
                                       prod `(clojure.core/* ~a-f ~b-f)]
                                   (case (long sign)
                                     1  prod
                                     -1 `(clojure.core/- 0.0 ~prod)
                                     `(clojure.core/* ~(double sign) ~prod))))
                               terms)]
                      (reduce (fn [acc t] `(clojure.core/+ ~acc ~t)) term-forms))
                    0.0)))
              field-specs)]
    `(let [~@a-binds ~@b-binds]
       (new ~fq-sym ~@component-exprs))))

(defn gen-product!
  "Generate and eval a deftm for a product operation.
   op-name: symbol for the function name
   sig: GA signature
   info: from gen-defvalue-class!
   grade-filter: nil for geometric, or (fn [ga gb rg] -> bool)"
  [op-name sig info grade-filter]
  (let [{:keys [class-sym fq-sym field-specs]} info
        contribs (collect-contribs sig grade-filter)
        body (gen-product-body contribs field-specs class-sym fq-sym)]
    (ga-eval `(deftm ~op-name [~'a :- ~class-sym, ~'b :- ~class-sym] :- ~class-sym
                ~body))))

(defn gen-geometric-product! [sig info]
  (let [suffix (sig->class-suffix sig)
        op-name (symbol (str "geometric-product-" suffix))]
    (gen-product! op-name sig info nil)
    op-name))

(defn gen-wedge! [sig info]
  (let [suffix (sig->class-suffix sig)
        op-name (symbol (str "wedge-" suffix))]
    (gen-product! op-name sig info
                  (fn [ga gb rg] (clojure.core/== rg (clojure.core/+ ga gb))))
    op-name))

(defn gen-inner! [sig info]
  (let [suffix (sig->class-suffix sig)
        op-name (symbol (str "inner-" suffix))]
    (gen-product! op-name sig info
                  (fn [ga gb rg]
                    (and (clojure.core/> ga 0) (clojure.core/> gb 0)
                         (clojure.core/== rg (Math/abs (clojure.core/- ga gb))))))
    op-name))

;; ================================================================
;; Gen: involutions (reverse, grade-involution, conjugate)
;; ================================================================

(defn- involution-sign
  "Compute the sign for an involution at a given grade."
  [kind ^long g]
  (case kind
    :reverse
    (if (even? (quot (clojure.core/* g (clojure.core/- g 1)) 2)) 1.0 -1.0)
    :grade-inv
    (if (even? g) 1.0 -1.0)
    :conjugate
    (if (even? (quot (clojure.core/* g (clojure.core/+ g 1)) 2)) 1.0 -1.0)))

(defn gen-involution!
  "Generate a deftm for an involution (reverse, grade-involution, conjugate)."
  [kind sig info]
  (let [{:keys [class-sym fq-sym field-specs n]} info
        suffix (sig->class-suffix sig)
        op-name (symbol (str (name kind) "-" suffix))
        grades (get-grade-table n)
        component-exprs
        (mapv (fn [{:keys [index]}]
                (let [g (clojure.core/aget ^ints grades index)
                      s (involution-sign kind g)
                      accessor (symbol (str "." (:name (nth field-specs index))))
                      field-access (list accessor
                                         (vary-meta 'x assoc :tag class-sym))]
                  (cond
                    (clojure.core/== s 1.0)  field-access
                    (clojure.core/== s -1.0) `(clojure.core/- ~field-access)
                    :else `(clojure.core/* ~s ~field-access))))
              field-specs)]
    (ga-eval `(deftm ~op-name [~'x :- ~class-sym] :- ~class-sym
                (new ~fq-sym ~@component-exprs)))
    op-name))

;; ================================================================
;; Gen: grade selection
;; ================================================================

(defn gen-grade-select!
  "Generate a deftm for grade selection."
  [sig info]
  (let [{:keys [class-sym fq-sym field-specs n]} info
        suffix (sig->class-suffix sig)
        op-name (symbol (str "grade-select-" suffix))
        grades (get-grade-table n)
        zero-mv `(new ~fq-sym ~@(repeat (count field-specs) 0.0))
        ;; Build nested if-chain instead of case (avoids deftm macro interference)
        grade-branches
        (reduce (fn [else-form g]
                  (let [component-exprs
                        (mapv (fn [{:keys [index]}]
                                (let [fg (clojure.core/aget ^ints grades index)]
                                  (if (clojure.core/== fg g)
                                    (list (symbol (str "." (:name (nth field-specs index))))
                                          (vary-meta 'x assoc :tag class-sym))
                                    0.0)))
                              field-specs)]
                    `(if (clojure.core/== ~'k ~(long g))
                       (new ~fq-sym ~@component-exprs)
                       ~else-form)))
                zero-mv
                (clojure.core/reverse (range (inc n))))]
    (ga-eval `(deftm ~op-name [~'x :- ~class-sym, ~'k :- ~'Long] :- ~class-sym
                ~grade-branches))
    op-name))

;; ================================================================
;; Gen: norm
;; ================================================================

(defn gen-norm!
  "Generate norm-squared and norm deftm functions."
  [sig info]
  (let [{:keys [class-sym fq-sym field-specs n]} info
        suffix (sig->class-suffix sig)
        ns-sq-name (symbol (str "norm-squared-" suffix))
        n-name (symbol (str "norm-" suffix))
        rev-name (symbol (str "reverse-" suffix))
        gp-name (symbol (str "geometric-product-" suffix))
        ;; Inline: norm-squared = scalar part of x * reverse(x)
        ;; More efficient: directly compute sum of squares with metric signs
        grades (get-grade-table n)
        ;; For norm-squared, we compute x * ~x where ~x is reverse
        ;; The scalar part is sum over all i of sign_i * x_i^2
        ;; where sign_i = reverse_sign(grade(i)) * metric_contribution
        ;; Actually simpler: use the generated functions
        ]
    (ga-eval `(deftm ~ns-sq-name [~'x :- ~class-sym] :- ~'Double
                (let [~'r (~rev-name ~'x)
                      ~'p (~gp-name ~'x ~'r)]
                  (~(symbol (str "." "s"))
                   ~(vary-meta 'p assoc :tag class-sym)))))
    (ga-eval `(deftm ~n-name [~'x :- ~class-sym] :- ~'Double
                (Math/sqrt (Math/abs (~ns-sq-name ~'x)))))
    {:norm-squared ns-sq-name :norm n-name}))

;; ================================================================
;; Gen: Hodge star
;; ================================================================

(defn gen-hodge!
  "Generate Hodge star via pseudoscalar left contraction."
  [sig info]
  (let [{:keys [class-sym fq-sym field-specs n]} info
        suffix (sig->class-suffix sig)
        op-name (symbol (str "hodge-" suffix))
        ;; Precompute the Hodge mapping: for each blade, compute left-contraction with pseudoscalar
        dim (algebra-dim sig)
        all-bits (clojure.core/- dim 1) ;; pseudoscalar bits = all set
        ;; ⋆e_A = sign(A, comp(A)) * e_{comp(A)}
        ;; where sign = blade-product-sign(A, comp(A), sig)
        ;; i.e., the sign from e_A ∧ e_{comp(A)} = sign * I
        component-exprs
        (mapv (fn [out-idx]
                (let [out-bits (index-blade out-idx n)
                      ;; Input blade whose complement is the output blade
                      in-bits (clojure.core/bit-xor out-bits all-bits)
                      in-idx (blade-index in-bits n)
                      ;; Sign from e_in * e_out = sign * I (complement permutation)
                      sign (blade-product-sign in-bits out-bits sig)]
                  (if (not (clojure.core/== sign 0))
                    (let [accessor (symbol (str "." (:name (nth field-specs in-idx))))
                          field-access (list accessor (vary-meta 'x assoc :tag class-sym))]
                      (case (long sign)
                        1  field-access
                        -1 `(clojure.core/- ~field-access)
                        `(clojure.core/* ~(double sign) ~field-access)))
                    0.0)))
              (range dim))]
    (ga-eval `(deftm ~op-name [~'x :- ~class-sym] :- ~class-sym
                (new ~fq-sym ~@component-exprs)))
    op-name))

;; ================================================================
;; Gen: raster.numeric dispatch
;; ================================================================

(defn gen-numeric-ops!
  "Register raster.numeric/+, -, * for the compiled MV type."
  [sig info]
  (let [{:keys [class-sym fq-sym field-specs]} info
        gp-name (symbol (str "geometric-product-" (sig->class-suffix sig)))]
    ;; MV + MV
    (ga-eval `(deftm raster.numeric/+ [~'a :- ~class-sym, ~'b :- ~class-sym] :- ~class-sym
                (new ~fq-sym
                     ~@(map (fn [{:keys [name]}]
                              (let [acc-a (symbol (str "." name))
                                    a-access (list acc-a (vary-meta 'a assoc :tag class-sym))
                                    b-access (list acc-a (vary-meta 'b assoc :tag class-sym))]
                                `(clojure.core/+ ~a-access ~b-access)))
                            field-specs))))
    ;; MV - MV
    (ga-eval `(deftm raster.numeric/- [~'a :- ~class-sym, ~'b :- ~class-sym] :- ~class-sym
                (new ~fq-sym
                     ~@(map (fn [{:keys [name]}]
                              (let [acc (symbol (str "." name))
                                    a-access (list acc (vary-meta 'a assoc :tag class-sym))
                                    b-access (list acc (vary-meta 'b assoc :tag class-sym))]
                                `(clojure.core/- ~a-access ~b-access)))
                            field-specs))))
    ;; Unary -
    (ga-eval `(deftm raster.numeric/- [~'a :- ~class-sym] :- ~class-sym
                (new ~fq-sym
                     ~@(map (fn [{:keys [name]}]
                              `(clojure.core/- (~(symbol (str "." name))
                                                ~(vary-meta 'a assoc :tag class-sym))))
                            field-specs))))
    ;; MV * MV → geometric product
    (ga-eval `(deftm raster.numeric/* [~'a :- ~class-sym, ~'b :- ~class-sym] :- ~class-sym
                (~gp-name ~'a ~'b)))
    ;; Double * MV
    (ga-eval `(deftm raster.numeric/* [~'a :- ~'Double, ~'b :- ~class-sym] :- ~class-sym
                (new ~fq-sym
                     ~@(map (fn [{:keys [name]}]
                              `(clojure.core/* ~'a (~(symbol (str "." name))
                                                    ~(vary-meta 'b assoc :tag class-sym))))
                            field-specs))))
    ;; MV * Double
    (ga-eval `(deftm raster.numeric/* [~'a :- ~class-sym, ~'b :- ~'Double] :- ~class-sym
                (new ~fq-sym
                     ~@(map (fn [{:keys [name]}]
                              `(clojure.core/* (~(symbol (str "." name))
                                                ~(vary-meta 'a assoc :tag class-sym)) ~'b))
                            field-specs))))
    ;; Double + MV (scalar addition)
    (ga-eval `(deftm raster.numeric/+ [~'a :- ~'Double, ~'b :- ~class-sym] :- ~class-sym
                (new ~fq-sym
                     (clojure.core/+ ~'a (~(symbol (str "." (:name (first field-specs))))
                                          ~(vary-meta 'b assoc :tag class-sym)))
                     ~@(map (fn [{:keys [name]}]
                              (list (symbol (str "." name))
                                    (vary-meta 'b assoc :tag class-sym)))
                            (rest field-specs)))))
    ;; MV + Double
    (ga-eval `(deftm raster.numeric/+ [~'a :- ~class-sym, ~'b :- ~'Double] :- ~class-sym
                (new ~fq-sym
                     (clojure.core/+ (~(symbol (str "." (:name (first field-specs))))
                                      ~(vary-meta 'a assoc :tag class-sym)) ~'b)
                     ~@(map (fn [{:keys [name]}]
                              (list (symbol (str "." name))
                                    (vary-meta 'a assoc :tag class-sym)))
                            (rest field-specs)))))))

;; ================================================================
;; Gen: constructors
;; ================================================================

(defn gen-constructors!
  "Generate constructor functions: scalar, per-grade, per-basis-element."
  [sig info]
  (let [{:keys [class-sym fq-sym field-specs n]} info
        suffix (sig->class-suffix sig)
        grades (get-grade-table n)
        prefix (str "mv" (clojure.core/apply str
                                             (interpose "" [(str (.p ^Signature sig))
                                                            (str (.q ^Signature sig))
                                                            (str (.r ^Signature sig))])))]
    ;; Scalar constructor
    (let [sc-name (symbol (str prefix "-scalar"))
          s-sym (gensym "s")]
      (ga-eval `(defn ~sc-name [~(with-meta s-sym {:tag 'double})]
                  (new ~fq-sym
                       ~@(map-indexed (fn [i _]
                                        (if (clojure.core/== i 0) s-sym 0.0))
                                      field-specs)))))
    ;; Per-grade constructors
    (doseq [g (range (inc n))]
      (let [grade-fields (filter #(clojure.core/== (blade-grade (:bits %)) g) field-specs)
            gc-name (symbol (str prefix "-grade" g))
            params (mapv (fn [f] (with-meta (symbol (:name f)) nil))
                         grade-fields)]
        (when (seq grade-fields)
          (let [grade-set (set (map :index grade-fields))]
            (ga-eval `(defn ~gc-name ~params
                        (new ~fq-sym
                             ~@(map (fn [{:keys [index name]}]
                                      (if (grade-set index)
                                        `(double ~(symbol name))
                                        0.0))
                                    field-specs))))))))
    ;; Per-basis-element defs
    (doseq [{:keys [name]} field-specs]
      (let [def-name (symbol (str prefix "-" name))]
        (ga-eval `(def ~def-name
                    (new ~fq-sym
                         ~@(map (fn [f]
                                  (if (clojure.core/= (:name f) name) 1.0 0.0))
                                field-specs))))))))

;; ================================================================
;; Gen: display (print-method)
;; ================================================================

(defn gen-display!
  "Generate print-method for the compiled MV type."
  [sig info]
  (let [{:keys [class-sym fq-sym class field-specs]} info
        suffix (sig->class-suffix sig)]
    (ga-eval
     `(defmethod print-method ~class [~'mv ^java.io.Writer ~'w]
        (let [~'names ~(mapv :name field-specs)
              ~'vals ~(mapv (fn [{:keys [name]}]
                              (list (symbol (str "." name))
                                    (vary-meta 'mv assoc :tag class-sym)))
                            field-specs)
              ~'terms (for [~'i (range ~(count field-specs))
                            :let [~'v (nth ~'vals ~'i)]
                            :when (not (clojure.core/== ~'v 0.0))]
                        (if (clojure.core/== ~'i 0)
                          (str ~'v)
                          (str ~'v "·" (nth ~'names ~'i))))]
          (.write ~'w (str "#ga/mv" ~suffix "["
                           (if (empty? ~'terms) "0"
                               (clojure.core/apply str (interpose " + " ~'terms)))
                           "]")))))))

;; ================================================================
;; Top-level API
;; ================================================================

(defn ops-map
  "Return keyword→fn map for the compiled GA ops."
  [info]
  (into {} (map (fn [[k sym]] [k @(ns-resolve 'raster.ga.compile sym)]) (:ops info))))

(def ^:private compiled-ga-cache (atom {}))

(defn compile-ga!
  "Compile specialized GA value class + ops for signature (p,q,r).
   Returns {:class-name, :class-sym, :dim, :n, :field-specs, :field-names, :ops}.
   Cached per signature. Dimension gate: n ≤ 6."
  [^Signature sig]
  (let [key [(.p sig) (.q sig) (.r sig)]]
    (or (get @compiled-ga-cache key)
        (let [n (sig-dim sig)
              _ (when (clojure.core/> n 6)
                  (throw (ex-info (str "compile-ga! dimension gate: n=" n " > 6. "
                                       "Use raster.ga.core/Multivector for n > 6.")
                                  {:sig sig :n n})))
              info (gen-defvalue-class! sig)
              gp (gen-geometric-product! sig info)
              w  (gen-wedge! sig info)
              ip (gen-inner! sig info)
              rev (gen-involution! :reverse sig info)
              ginv (gen-involution! :grade-inv sig info)
              conj (gen-involution! :conjugate sig info)
              gs (gen-grade-select! sig info)
              norms (gen-norm! sig info)
              hodge (gen-hodge! sig info)
              _ (gen-numeric-ops! sig info)
              _ (gen-constructors! sig info)
              _ (gen-display! sig info)
              result (assoc info
                            :field-names (mapv :name (:field-specs info))
                            :ops {:geometric-product gp
                                  :wedge w
                                  :inner ip
                                  :reverse rev
                                  :grade-involution ginv
                                  :conjugate conj
                                  :grade-select gs
                                  :norm-squared (:norm-squared norms)
                                  :norm (:norm norms)
                                  :hodge hodge})]
          (swap! compiled-ga-cache assoc key result)
          result))))
