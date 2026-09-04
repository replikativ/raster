(ns raster.compiler.passes.region-copy
  "Array region copies spelled as the store loops they are.

   `(acopy! src src-off dst dst-off len)` and the `System/arraycopy` it inlines to are opaque
   effects to every later pass: the loop lifter sees no store, the typed frontend sees an
   untagged binding, and a bias prefill spelled as one copy per row keeps a whole linear layer
   on the host. The copy of `len` elements is, element for element,

     (dotimes [t len] (aset dst (+ dst-off t) (aget src (+ src-off t))))

   whenever the source and the destination are distinct storage. That is what this pass emits,
   in statement positions only: a `do` statement, a `dotimes` body, or an effect binding whose
   value nothing reads. A copy whose value is used keeps its call (it returns the destination),
   and a copy within one array keeps its call too (`System/arraycopy` is a memmove, an
   elementwise forward loop is not).

   The emitted read carries the copied element type as its scalar tag when a fact states one:
   the call's own result tag, the tag of the binder that introduced either array, or the
   deftm parameter tag. Downstream typing then reads a fact rather than inferring one."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

(def ^:private copy-ops
  "Region copy spellings: the deftm and the JVM primitive it inlines to. Arguments are
   `src src-off dst dst-off len` in every spelling."
  '#{raster.arrays/acopy! System/arraycopy java.lang.System/arraycopy})

(defn- array-tag
  "The array type tag a fact states for `array`: the tag on its binder or deftm parameter."
  [environment array]
  (let [tag (get environment array)]
    (when (symbol? tag) tag)))

(defn- copy-call
  "`{:src :src-off :dst :dst-off :len :returns-destination? :tag}` for a region copy call over
   two distinct array symbols, else nil. `environment` maps array symbols to their tags."
  [expression environment]
  (when (and (seq? expression) (contains? copy-ops (descriptor/semantic-op expression)))
    (let [[src src-off dst dst-off len :as arguments] (vec (descriptor/call-args expression))]
      (when (and (= 5 (count arguments)) (symbol? src) (symbol? dst) (not= src dst))
        {:src src :src-off src-off :dst dst :dst-off dst-off :len len
         :returns-destination? (= 'raster.arrays/acopy! (descriptor/semantic-op expression))
         :tag (or (:raster.type/tag (meta expression))
                  (array-tag environment dst)
                  (array-tag environment src))}))))

(defn- offset-index
  "`offset + index`; a zero offset (the walker spells a literal `0` as `(int 0)`) is the bare
   index, which is what the loop lifter's element-wise matchers recognize."
  [offset index]
  (let [offset (descriptor/unwrap-int-cast offset)]
    (if (and (number? offset) (zero? offset))
      index
      (list 'clojure.core/+ offset index))))

(defn- store-loop
  "The counted store loop equal to a copy call."
  [{:keys [src src-off dst dst-off len tag]} index]
  (let [scalar-tag (some-> tag dtype/dtype-for-array-tag dtype/canon dtype/scalar-tag-for-dtype)
        read (cond-> (list 'clojure.core/aget src (offset-index src-off index))
               scalar-tag (with-meta {:raster.type/tag scalar-tag :tag scalar-tag}))]
    (list 'dotimes [index len]
          (list 'clojure.core/aset dst (offset-index dst-off index) read))))

(defn- binder-tag
  [symbol]
  (or (:raster.type/tag (meta symbol)) (:tag (meta symbol))))

(defn expand-region-copies
  "Rewrite every region copy in statement position of `form` into its store loop.
   `param-env` maps deftm parameters to their array tags. Returns `{:form :stats}` with
   `:region-copies-expanded`."
  [form & {:keys [param-env]}]
  (let [counter (atom 0)
        expanded (atom 0)
        fresh (fn [] (symbol (str "rstr_copy_" (swap! counter inc))))
        expand! (fn [call]
                  (swap! expanded inc)
                  (store-loop call (fresh)))]
    (letfn [(statement [expression environment]
              ;; a value nobody reads: a copy here is exactly its loop
              (if-let [call (copy-call expression environment)]
                (expand! call)
                (walk expression environment)))
            (walk [expression environment]
              (cond
                (not (seq? expression)) expression

                (= 'do (first expression))
                (let [statements (butlast (rest expression))
                      value (last expression)]
                  (with-meta (apply list 'do (concat (map #(statement % environment) statements)
                                                     [(walk value environment)]))
                    (meta expression)))

                (= 'dotimes (first expression))
                (let [[_ bindings & body] expression]
                  (with-meta (apply list 'dotimes bindings (map #(statement % environment) body))
                    (meta expression)))

                (form/binding-form? expression)
                (let [[head bindings & body] expression
                      pairs (vec (partition 2 bindings))
                      inits (mapv second pairs)
                      ;; everything evaluated after binding k
                      later (fn [k] (list* 'do (concat (drop (inc k) inits) body)))
                      [environment pairs]
                      (reduce (fn [[environment pairs] [k [symbol init]]]
                                (let [call (copy-call init environment)
                                      init (if (and call
                                                    (or (not (:returns-destination? call))
                                                        (not (contains? (util/free-syms (later k))
                                                                        symbol))))
                                             (expand! call)
                                             (walk init environment))
                                      environment (cond-> environment
                                                    (binder-tag symbol)
                                                    (assoc symbol (binder-tag symbol)))]
                                  [environment (conj pairs [symbol init])]))
                              [environment []]
                              (map-indexed vector pairs))]
                  (with-meta (apply list head (vec (mapcat identity pairs))
                                    (map #(walk % environment) body))
                    (meta expression)))

                :else
                (with-meta (apply list (map #(walk % environment) expression)) (meta expression))))]
      (let [form (walk form (or param-env {}))]
        {:form form :stats {:region-copies-expanded @expanded}}))))
