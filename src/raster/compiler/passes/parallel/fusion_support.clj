(ns raster.compiler.passes.parallel.fusion-support
  "Shared rewrite helpers for parallel fusion passes."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.op-descriptor :as descriptor]))

(defn substitute-aget
  "Replace reads of `(aget target-sym idx)` inside `body` with `replacement`.
	Rewrites the replacement so uses of `source-idx` become `target-idx`."
  [body target-sym source-idx target-idx replacement]
  (walk/postwalk
   (fn [form]
     ;; Match both bare (aget target idx) and devirtualized (.invk aget-impl target idx);
     ;; aget-array-sym unwraps casts + strips ns so it compares against target-sym by name.
     (if (and (descriptor/aget-call? form)
              (let [s (descriptor/aget-array-sym form)]
                (and (symbol? s) (= (name target-sym) (name s)))))
       (walk/postwalk
        (fn [inner]
          (if (= inner source-idx) target-idx inner))
        replacement)
       form))
   body))

(defn rewrite-index-sym
  "Rewrite occurrences of `from-idx` to `to-idx` throughout `body`."
  [body from-idx to-idx]
  (walk/postwalk
   (fn [form]
     (if (= form from-idx) to-idx form))
   body))

(defn emit-side-effect-aset
  "Build an `aset` side effect for a secondary horizontally fused output.
	Rewrites `body` from `from-idx` to `to-idx` and applies `cast-fn` when present."
  [out-sym to-idx from-idx cast-fn body]
  (let [adj-body (rewrite-index-sym body from-idx to-idx)
        cast-body (if cast-fn
                    (list cast-fn adj-body)
                    adj-body)]
    (list 'clojure.core/aset out-sym to-idx cast-body)))