(ns raster.support
  "Shared utility macros for the Raster namespace hierarchy.")

(defmacro import-vars
  "Import public vars from another namespace into this one.
   Preserves metadata from the source var."
  [ns-sym & var-syms]
  `(do ~@(map (fn [sym]
                `(let [src-var# (resolve '~(symbol (str ns-sym) (str sym)))]
                   (intern *ns* (with-meta '~sym (meta src-var#)) @src-var#)))
              var-syms)))
