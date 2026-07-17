(ns raster.compiler.passes.parallel.fusion-support
  "Shared rewrite helpers for parallel fusion passes."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.op-descriptor :as descriptor]))

(defn substitute-aget
  "Replace reads of `(aget target-sym idx)` inside `body` with `replacement`.
	Rewrites the replacement so uses of `source-idx` become `target-idx`.

	INDEX-INSENSITIVE (silently-ignored-information family, deferred): this matches an
	aget by ARRAY NAME only and substitutes regardless of the aget's actual index expr —
	it assumes the consumer reads target-sym at the SAME logical position the producer
	wrote (differing only in index-var name, which the source-idx→target-idx rewrite
	handles). A consumer that reads a DIFFERENT element — `(aget tmp (+ j 1))`, a
	transpose, a gather — would be wrongly replaced with the producer's element at the
	consumer's index. Vertical fusion only fires on same-index elementwise producer→
	consumer pairs, so no reachable caller violates this today; a real guard must abort
	the whole fusion (declining to substitute would leave a dangling ref to an eliminated
	buffer), which belongs with the descriptor VALIDATOR (S4). Documented, not yet asserted."
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