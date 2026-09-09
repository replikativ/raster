(ns raster.compiler.passes.parallel.effect-source
  "Shared host continuation construction for validated scheduled effect regions.

   Target projections supply store and loop spelling; this function alone owns ordered effect
   sequencing and the scope of an exported loop result. It neither infers types nor parses source.")

(defn ordered-effects
  "Spell scheduled effects as a host continuation. `emit-store` receives a store descriptor;
   `emit-loop` receives a loop descriptor and its already-spelled ordered body. Both return forms.
   A carried result encloses only subsequent effects, never its initializer or preceding stores."
  [effects {:keys [emit-store emit-loop] :as emitters}]
  (if-let [effect (first effects)]
    (let [loop (:loop effect)
          form (if loop
                 (emit-loop loop (ordered-effects (:effects loop) emitters))
                 (emit-store effect))
          continuation (ordered-effects (next effects) emitters)]
      (if-let [result (get-in loop [:carry :result])]
        (list 'let* [result form] continuation)
        (list 'do form continuation)))
    nil))
