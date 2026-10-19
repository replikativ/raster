(ns raster.ad.reverse.normalize
  "ANF normalization for reverse-mode AD.

   Flattens walked S-expressions so that:
   - All call arguments are trivial (symbols or literals)
   - If-expression arms retain their lexical evaluation regions
   - Body is reduced to a single symbol

   This is required before the reverse pass can track operations."
  (:require [raster.compiler.ir.form :as form]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.types :as types]))

(defn trivial-expr?
  "True if expr is simple enough to not need its own binding."
  [x]
  (or (symbol? x) (number? x) (nil? x) (boolean? x) (string? x) (keyword? x)))

(defn closed-call?
  "A call whose original arguments already need no ANF snapshots. This is a
   syntax boundary only, not a purity, derivative-rule or return-carrier proof."
  [expression]
  (and (form/call-form? expression)
       (not (form/introduces-scope? expression))
       (every? trivial-expr? (:arguments (op/call-description expression)))))

(defn- lifted-value-symbol
  "A lifted argument denotes the already-typed expression's value. Preserve
   that declaration and source position, not its callee/effect/ownership facts.
   Missing declarations remain missing; normalization does not infer types."
  [gensym-fn prefix expression]
  (vary-meta (gensym-fn prefix) merge
             (select-keys (meta expression) [:raster.type/tag :line :column :file])))

(defn anf-normalize-expr
  "Flatten nested sub-expressions in a call into separate let bindings.
  Returns [extra-bindings-vec, normalized-expr] where all call arguments
  are trivial (symbols or literals).
  extra-bindings is a flat vector [sym1 init1 sym2 init2 ...].
  gensym-fn: (fn [prefix] -> unique-symbol)"
  [expr gensym-fn]
  (cond
    ;; Trivial expressions need no lifting
    (trivial-expr? expr) [[] expr]

    ;; Function call: lift non-trivial arguments
    (seq? expr)
    (let [head (first expr)
          extras (atom [])]
      (cond
        ;; Quoted syntax is data, including apparent binders and effects.
        (contains? #{'quote 'clojure.core/quote} head)
        [[] expr]

        ;; Object completion commutes with these normal result exits, not
        ;; with arbitrary inputs. Complete each ORIGINAL leaf before ANF can
        ;; introduce a raw temporary. Primitive/checkcast contexts retain
        ;; their source-owner policy and are deliberately opaque here.
        (and (= completion/object-input-contract
                (:contract (completion/canonical-parts expr)))
             (form/normal-exit-region?
              (:operand (completion/canonical-parts expr))))
        (let [{:keys [contract operand]} (completion/canonical-parts expr)]
          (anf-normalize-expr
           (form/map-normal-exits #(completion/complete-value contract %) operand)
           gensym-fn))

        ;; Completion consumes its ORIGINAL operand in an exact JVM context.
        ;; Lifting that operand first can box a primitive result, changing its
        ;; conversion/trap behavior. Snapshot the completed value as a whole;
        ;; its AD producer must own the internal region and representation.
        (completion/canonical-form? expr)
        [[] expr]

        ;; An if is a region boundary, not a call with eagerly evaluated arguments.
        ;; Normalize and snapshot the predicate only; each AD mode transforms the selected
        ;; arm locally. Purity does not license speculation (sqrt/checked reads are partial).
        (= 'if head)
        (let [[_ test then else] expr
              [te tn] (anf-normalize-expr test gensym-fn)
              decision (if (trivial-expr? tn) tn
                           (with-meta (gensym-fn "br") (meta test)))]
          [(cond-> (vec te) (not (trivial-expr? tn)) (into [decision tn]))
           (with-meta (list 'if decision then else) (meta expr))])

        ;; Do expressions: flatten side-effect forms, return last
        (= 'do head)
        (let [sub-exprs (rest expr)]
          (doseq [se (butlast sub-exprs)]
            (let [[sub-extras normalized] (anf-normalize-expr se gensym-fn)
                  s (lifted-value-symbol gensym-fn "_side" normalized)]
              (swap! extras into (concat sub-extras [s normalized]))))
          (let [last-expr (last sub-exprs)
                [le ln] (anf-normalize-expr last-expr gensym-fn)]
            [(vec (concat @extras le)) ln]))

        ;; Scope-introducing forms (let*/loop*/dotimes/fn*/ftm AND every par/*
        ;; SOAC) pass through unchanged — their bodies bind their own locals (loop
        ;; index, accumulator), so lifting a sub-expr OUT of the scope would let a
        ;; scoped var escape. Use the unified scope classifier, not a hardcoded set,
        ;; so par forms are covered (and any new binder is automatically handled).
        (or (form/binding-form? expr) (form/introduces-scope? expr))
        [[] expr]

        ;; Regular call: lift non-trivial args
        :else
        (let [args (rest expr)
              norm-args
              (mapv (fn [arg]
                      (if (trivial-expr? arg)
                        arg
                        (let [[sub-extras normalized] (anf-normalize-expr arg gensym-fn)
                              s (lifted-value-symbol gensym-fn "anf" normalized)]
                          (swap! extras into (concat sub-extras [s normalized]))
                          s)))
                    args)]
          [@extras (with-meta (cons head norm-args) (meta expr))])))

    :else [[] expr]))

(defn- retain-lexical-declarations [expression declarations]
  ;; Nil substitution values contain no free names, so this transports
  ;; declarations without capture-driven renaming or replacing any value.
  ;; The shared scope engine masks declarations under every nested binder.
  (util/subst-syms
   (zipmap (keys declarations) (repeat nil)) expression
   (fn [visible reference]
     (if-not (contains? visible reference)
       reference
       (let [tag (get declarations reference)
             existing (:raster.type/tag (meta reference))]
         (when (and existing (not= tag existing))
           (throw (ex-info "AD normalization encountered conflicting lexical declarations"
                           {:reason :ad-conflicting-lexical-declaration
                            :symbol reference :declaration tag :reference-tag existing})))
         (vary-meta reference assoc :raster.type/tag tag))))))

(defn- binding-declarations [bindings initial-declarations]
  (reduce (fn [declarations [id _]]
            (if-let [tag (:raster.type/tag (meta id))]
              (assoc declarations id tag)
              (dissoc declarations id)))
          initial-declarations (partition 2 bindings)))

(defn anf-normalize-bindings
  "ANF-normalize all bindings: lift nested sub-expressions into flat bindings.
  Input: flat bindings vector [s1 e1 s2 e2 ...].
  Output: flat bindings vector with nested calls flattened."
  ([bindings gensym-fn] (anf-normalize-bindings bindings gensym-fn {}))
  ([bindings gensym-fn initial-declarations]
  (let [result (atom [])
        declarations (atom initial-declarations)]
    (letfn [(retain-value-tag [id expression]
              (let [object-tag (completion/object-value-declaration expression)
                    ;; This exact context preserves the operand's value while
                    ;; boxing its representation. A physical Object hint is
                    ;; not a conflicting logical declaration. Other contexts
                    ;; may convert the value and have no such authority.
                    tag (or (:raster.type/tag (meta expression))
                            object-tag)
                    existing (:raster.type/tag (meta id))]
                (when (and object-tag existing (not= object-tag existing))
                  (throw (ex-info "AD Object snapshot has conflicting logical declarations"
                                  {:reason :ad-conflicting-lexical-declaration
                                   :symbol id :declaration existing :reference-tag object-tag})))
                (if (and tag (nil? (:raster.type/tag (meta id)))
                         (or (nil? (types/sym-type-tag id))
                             (and object-tag (= 'Object (:tag (meta id))))))
                  (vary-meta id assoc :raster.type/tag tag)
                  id)))
            (emit! [sym init]
              ;; Flatten only an initializer's lexical value region, at that
              ;; exact binding position. Do not move call arguments across
              ;; their original typed input conversions or speculate branches.
              (let [init (retain-lexical-declarations init @declarations)
                    sym (retain-value-tag sym init)]
               (if (and (seq? init) (contains? #{'let 'let*} (first init)))
                (let [[_ inner & body] (util/alpha-convert init)
                      _ (doseq [[id value] (partition 2 inner)] (emit! id value))
                      [extras value] (anf-normalize-expr (cons 'do body) gensym-fn)]
                  (doseq [[id value] (partition 2 extras)] (emit! id value))
                  (emit! sym value))
                (let [;; Retain existing call identity, never decode an impl.
                      init (if (and (seq? init) (= '.invk (first init))
                                    (nil? (:raster.op/original (meta init)))
                                    (:raster.op/original (meta sym)))
                             (vary-meta init assoc :raster.op/original
                                        (:raster.op/original (meta sym)))
                             init)
                      [extras normalized] (anf-normalize-expr init gensym-fn)
                      ;; Retain a declaration supplied by composite expansion;
                      ;; this neither infers a type nor proves non-nullness.
                      sym (retain-value-tag sym normalized)]
                  (doseq [[id value] (partition 2 extras)] (emit! id value))
                  (let [normalized (retain-lexical-declarations normalized @declarations)]
                    (swap! result conj sym normalized)
                    ;; Initializers see the previous binding. Publish this
                    ;; occurrence only afterwards, including shadow masking.
                    (swap! declarations
                           #(if-let [tag (:raster.type/tag (meta sym))]
                              (assoc % sym tag) (dissoc % sym))))))))]
      (doseq [[sym init] (partition 2 bindings)] (emit! sym init)))
    @result)))

(defn normalize-for-ad
  "Normalize a let form for uniform AD processing.
  Input: flat bindings vector [s1 e1 s2 e2 ...], body-exprs list, gensym-fn.
  Output: [flat-bindings-vector, body-symbol]

  Guarantees:
  - Body is a single symbol
  - All call arguments are trivial (symbols or literals) — ANF
  - Conditional arms remain lexical regions, never eager sibling bindings
  - Nontrivial branch conditions are snapshotted before the selected arm"
  ([bindings body-exprs gensym-fn] (normalize-for-ad bindings body-exprs gensym-fn {}))
  ([bindings body-exprs gensym-fn initial-declarations]
  (let [anf-bindings (anf-normalize-bindings bindings gensym-fn initial-declarations)
        declarations (binding-declarations anf-bindings initial-declarations)
        body-expr (retain-lexical-declarations
                   (if (= 1 (count body-exprs))
                     (first body-exprs)
                     (cons 'do body-exprs))
                   declarations)
        [extras normalized] (anf-normalize-expr body-expr gensym-fn)
        body-sym (if (symbol? normalized) normalized
                     (with-meta (gensym-fn "body") (meta body-expr)))]
    (let [all-bindings (into (vec anf-bindings)
                             (anf-normalize-bindings
                              (cond-> (vec extras)
                                (not (symbol? normalized)) (into [body-sym normalized]))
                              gensym-fn declarations))]
      [all-bindings
       (retain-lexical-declarations body-sym (binding-declarations all-bindings initial-declarations))]))))
