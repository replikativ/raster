(ns raster.compiler.passes.parallel.effect-source
  "Shared host continuation construction for validated scheduled effect regions.

   Target projections supply store and loop spelling; this function alone owns ordered effect
   sequencing and the scope of an exported loop result. It neither infers types nor parses source."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]))

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
    (list 'let*
          (vec (mapcat (fn [{:keys [id dtype init]}]
                         (let [tag (dtype/scalar-tag-for-dtype dtype)]
                           [(with-meta id {:raster.type/tag tag})
                            (list (generated-cast tag) (strip-binder-tags init))]))
                       locals))
          body)
    body))

(defn counted-loop
  "Spell the validated scheduled counted-loop contract, including its optional typed carry.
   Bounds/initializer are evaluated once. Stores precede recurrence evaluation. A zero-trip loop
   returns its initializer. Result scope is supplied separately by `ordered-effects`."
  [generated-cast {:keys [index lower extent locals carry]} ordered-body]
  (let [{:keys [parameter dtype init update]} carry
        tag (when carry (generated-cast (dtype/scalar-tag-for-dtype dtype)))
        index-tag (if carry 'clojure.core/long 'clojure.core/int)
        limit (gensym "effect_carry_limit__")
        initial (gensym "effect_carry_init__")]
    (list 'let* (cond-> [limit (list index-tag extent)]
                  carry (conj initial (list tag (strip-binder-tags init))))
          (list 'loop* (cond-> [(vary-meta index dissoc :tag) (list index-tag lower)]
                         carry (conj (vary-meta parameter dissoc :tag) initial))
                (list 'if (list 'clojure.core/< index limit)
                      (typed-locals
                       generated-cast locals
                       (list 'do ordered-body
                             (list* 'recur
                                    (cond-> [(list (if carry 'clojure.core/unchecked-inc
                                                     'clojure.core/unchecked-inc-int) index)]
                                      carry (conj (list tag (strip-binder-tags update)))))))
                      parameter)))))

(defn ordered-effects
  "Spell scheduled effects as a host continuation. `emit-store` receives a store descriptor;
   `emit-loop` receives a loop descriptor and its already-spelled ordered body. `emit-region`
   receives retained locals and their ordered body. All callbacks return forms.
   A carried result encloses only subsequent effects, never its initializer or preceding stores."
  [effects {:keys [emit-store emit-loop emit-region] :as emitters}]
  (if-let [effect (first effects)]
    (let [loop (:loop effect)
          form (cond
                 (:region effect)
                 (let [{:keys [locals effects]} (:region effect)]
                   (emit-region locals (ordered-effects effects emitters)))
                 loop (emit-loop loop (ordered-effects (:effects loop) emitters))
                 :else (emit-store effect))
          continuation (ordered-effects (next effects) emitters)]
      (if-let [result (get-in loop [:carry :result])]
        (list 'let* [(vary-meta result dissoc :tag) form] continuation)
        (list 'do form continuation)))
    nil))
