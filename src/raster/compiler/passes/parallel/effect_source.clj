(ns raster.compiler.passes.parallel.effect-source
  "Shared host continuation construction for validated scheduled effect regions.

   Target projections supply store and loop spelling; this function alone owns ordered effect
   sequencing and the scope of an exported loop result. It neither infers types nor parses source."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]))

(defn storage-cast
  "Generated storage conversion, not a user-written cast. Strict FP32 regions use IEEE rounding
   and overflow, matching KernelBody; legacy flat host regions retain checked casts."
  [strict? tag]
  (if (and strict? (= 'float tag))
    'clojure.core/unchecked-float
    (symbol "clojure.core" (name tag))))

(defn- strip-binder-tags
  [form]
  ;; JVM loop/let initializers already carry their primitive types. Keep Raster's independent
  ;; type facts, but remove source binder hints that the Clojure compiler would reject.
  (walk/postwalk
   (fn [x]
     (if (and (seq? x) (contains? #{'let* 'loop*} (first x)) (vector? (second x)))
       (with-meta
         (list* (first x)
                (vec (map-indexed (fn [ordinal item]
                                    (if (and (even? ordinal) (symbol? item) (:tag (meta item)))
                                      (vary-meta item dissoc :tag)
                                      item))
                                  (second x)))
                (nnext x))
         (meta x))
       x))
   form))

(defn typed-locals
  "Materialize retained local dtypes through explicit generated casts."
  [generated-cast locals body]
  (if (seq locals)
    (let [products
          (vec
           (distinct
            (mapcat (fn [{:keys [init]}]
                      (keep #(when (dialect/product-fold-form? %) %)
                            (tree-seq coll? seq init)))
                    locals)))
          product-bindings (mapv (fn [product] [product (gensym "product_fold__")]) products)
          product-binding-map (into {} product-bindings)
          project-components
          (fn [expression]
            (walk/postwalk
             (fn [form]
               (if (dialect/product-component-form? form)
                 (let [[_ product ordinal] form]
                   (list 'clojure.core/nth (get product-binding-map product)
                         (list 'clojure.core/long ordinal)))
                 form))
             expression))
          ;; Insert each tuple binding immediately before its first projection. Product Fold
          ;; expressions may capture earlier scalar locals (mean/inv-std in layer norm), so
          ;; hoisting every tuple to the front of the let spine would violate lexical order.
          scalar-bindings
          (:bindings
           (reduce
            (fn [{:keys [bindings emitted] :as state} {:keys [id dtype init]}]
              (let [local-products
                    (distinct
                     (keep #(when (dialect/product-fold-form? %) %)
                           (tree-seq coll? seq init)))
                    fresh-products (remove emitted local-products)
                    tag (dtype/scalar-tag-for-dtype dtype)]
                {:bindings
                 (into bindings
                       (concat
                        (mapcat (fn [product]
                                  [(get product-binding-map product)
                                   (projection/product-fold->source product)])
                                fresh-products)
                        [(with-meta id {:raster.type/tag tag})
                         (list (generated-cast tag)
                               (strip-binder-tags (project-components init)))]))
                 :emitted (into emitted fresh-products)}))
            {:bindings [] :emitted #{}}
            locals))]
      (list 'let* (vec scalar-bindings) body))
    body))

(defn counted-loop
  "Spell the validated scheduled counted-loop contract with simultaneous typed carry updates.
   Initializers are already outer-scope operands. Stores precede recurrence evaluation, and a
   zero-trip loop returns the initial tuple. `emit-body` receives the recurrence as its lexical
   tail, so an ordered effect result may be consumed by a carry update without escaping scope."
  [generated-cast {:keys [index lower upper-bound extent locals carries]} emit-body]
  (let [tags (mapv #(generated-cast (dtype/scalar-tag-for-dtype (:dtype %))) carries)
        index-tag (if (seq carries) 'clojure.core/long 'clojure.core/int)
        limit (gensym "effect_carry_limit__")
        initials (mapv (fn [_] (gensym "effect_carry_init__")) carries)
        next-values (mapv (fn [_] (gensym "effect_carry_next__")) carries)
        inclusive? (= :inclusive upper-bound)
        advance (list (if (seq carries) 'clojure.core/unchecked-inc
                          'clojure.core/unchecked-inc-int) index)
        recur-form (if (seq carries)
                     (list 'let* (vec (mapcat (fn [next tag carry]
                                               [next (list tag (strip-binder-tags (:update carry)))])
                                             next-values tags carries))
                           (if inclusive?
                             (list 'if (list 'clojure.core/= index limit)
                                   next-values (list* 'recur advance next-values))
                             (list* 'recur advance next-values)))
                     (if inclusive?
                       (list 'if (list 'clojure.core/= index limit)
                             nil (list 'recur advance))
                       (list 'recur advance)))]
    (list 'let* (into [limit (list index-tag extent)]
                     (mapcat (fn [initial tag carry]
                               [initial (list tag (strip-binder-tags (:init carry)))])
                             initials tags carries))
          (list 'loop* (into [(vary-meta index dissoc :tag) (list index-tag lower)]
                            (mapcat (fn [carry initial]
                                      [(vary-meta (:parameter carry) dissoc :tag) initial])
                                    carries initials))
                (list 'if (list (if inclusive? 'clojure.core/<= 'clojure.core/<)
                                index limit)
                      (typed-locals
                       generated-cast locals
                       (emit-body recur-form))
                      (when (seq carries) (mapv :parameter carries)))))))

(defn ordered-effects
  "Spell scheduled effects around an explicit host `continuation`.

   `emit-store` receives a store descriptor. `emit-loop` receives a loop descriptor and an
   `emit-body` callback; invoking that callback with the loop's recurrence spells the ordered body
   with that recurrence in the lexical scope of every exported body result. `emit-region` receives
   retained locals and its already-spelled effect-only body. All callbacks return forms.

   An exported result encloses only the subsequent continuation, never its initializer or a
   preceding effect. Ordinary region locals remain lexical and do not enclose the outer
   continuation."
  [effects continuation {:keys [emit-store emit-loop emit-region] :as emitters}]
  (if-let [effect (first effects)]
    (let [loop (:loop effect)
          branch (:branch effect)
          results (or (:results branch) (:carries loop))
          form (cond
                 branch
                 (let [emit-arm
                       (fn [{:keys [locals effects yields]}]
                         (emit-region
                          locals
                          (ordered-effects
                           effects
                           (mapv (fn [{:keys [dtype]} value]
                                   (list (storage-cast true (dtype/scalar-tag-for-dtype dtype))
                                         (strip-binder-tags value)))
                                 results yields)
                           emitters)))]
                   (list 'if (:predicate branch)
                         (emit-arm (:then branch)) (emit-arm (:else branch))))
                 (:region effect)
                 (let [{:keys [predicate locals effects]} (:region effect)
                       region (emit-region locals (ordered-effects effects nil emitters))]
                   (if (contains? (:region effect) :predicate)
                     (list 'if predicate region)
                     region))
                 loop (emit-loop loop #(ordered-effects (:effects loop) % emitters))
                 :else (emit-store effect))
          continuation (ordered-effects (next effects) continuation emitters)]
      (cond
        (seq results)
        (let [tuple (gensym "effect_results__")]
          (list 'let* (into [tuple form]
                            (mapcat (fn [ordinal {:keys [result dtype]}]
                                      [(vary-meta result dissoc :tag)
                                       (list (storage-cast true (dtype/scalar-tag-for-dtype dtype))
                                             (list 'clojure.core/nth tuple ordinal))])
                                    (range) results)) continuation))
        (:result effect)
        (list 'let* [(vary-meta (:result effect) dissoc :tag) form] continuation)
        :else (list 'do form continuation)))
    continuation))
