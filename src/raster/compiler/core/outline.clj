(ns raster.compiler.core.outline
  "Split oversized sequential binding blocks into sibling static methods.

   HotSpot never JIT-compiles a method whose bytecode exceeds 8000 bytes
   (-XX:+DontCompileHugeMethods, the default). A hoisted helper that holds a
   whole loop body, such as the backward sweep of a reduction, or a compute
   orchestrator with a thousand bindings, can exceed that and then runs
   interpreted on every call.

   This namespace cuts large `let`/`let*` blocks into chunks. Each chunk becomes
   a sibling method that receives its free variables as typed parameters and
   writes the values later code still reads (its live-outs) into per-call frame
   arrays: `double[]` for doubles, `long[]` for longs and ints, `Object[]` for
   everything else. The original position calls the chunk and reads the
   live-outs back with `aget`, so evaluation order and every operation stay
   where they were.

   A value's frame follows its physical tag at the binding site. A binding
   tagged Object (for example a helper result whose presence is unproven) is
   stored as Object and never unboxed by the split."
  (:require [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

(def ^:private primitive-frame '{double :double long :long int :long})

(defn- normalize-tag [tag]
  (if (and (sequential? tag) (= 'Array (first tag)))
    (get '{double doubles long longs int ints float floats boolean booleans} (second tag) tag)
    (get '{Double double Long long Integer int Boolean boolean} tag tag)))

(defn- tag-of [env sym]
  (normalize-tag (or (:tag (meta sym)) (get env sym) (:raster.type/tag (meta sym)))))

(defn- free-syms [form]
  (binding [util/*shadowing-locals* (or util/*shadowing-locals* #{})]
    (util/free-syms form)))

(defn- frame-of
  "Frame kind for a live-out with `tag`; nil when the value cannot be carried."
  [tag lenient?]
  (or (primitive-frame tag)
      (when (and lenient? (not (contains? '#{float boolean byte short char} tag))) :object)))

(defn- param-tag
  "Parameter tag for an input with `tag`; nil when unsupported."
  [tag lenient?]
  (cond
    (contains? '#{double long int boolean doubles longs ints floats float booleans objects} tag) tag
    lenient? (or tag 'Object)
    :else nil))

(defn- chunk-pairs
  "Greedily group binding pairs so each group's estimated size stays under budget."
  [pairs size-of budget]
  (reduce (fn [groups pair]
            (let [current (peek groups)
                  size (size-of (second pair))]
              (if (and (seq current) (> (+ (:size current) size) budget))
                (conj groups {:pairs [pair] :size size})
                (conj (pop groups) {:pairs (conj (:pairs current) pair)
                                    :size (+ (:size current) size)}))))
          [{:pairs [] :size 0}] pairs))

(def ^:private frame-tag '{:double doubles :long longs :object objects})

(defn outline-bindings
  "Rewrite the binding pairs of one block whose result is `body` (a vector of
   forms evaluated after the bindings). Returns
   {:pairs new-pairs :helpers [...] :frame {:double n :long n :object n}}
   or nil when some symbol cannot be carried.

   options: :size-of :chunk-budget :new-name :frames {kind sym}, :lenient?
   (accept reference values, for straight-line orchestrators) and :exclude
   (symbols that are methods, not locals, such as sibling helper names)."
  [pairs body env {:keys [size-of chunk-budget new-name frames lenient? exclude]}]
  (let [groups (chunk-pairs pairs size-of chunk-budget)
        outlined (butlast groups)
        tail-pairs (:pairs (last groups))]
    (loop [groups outlined
           env env
           out-pairs []
           helpers []
           frame {:double 0 :long 0 :object 0}
           remaining (vec (mapcat :pairs outlined))]
      (if (empty? groups)
        {:pairs (into out-pairs tail-pairs) :helpers helpers :frame frame}
        (let [{gpairs :pairs} (first groups)
              remaining (subvec remaining (count gpairs))
              defined (mapv first gpairs)
              defined-set (set defined)
              inputs (vec (sort-by str (remove (set exclude)
                                               (free-syms (list 'let* (vec (mapcat identity gpairs)) nil)))))
              used-later (free-syms (list* 'let* (vec (mapcat identity (concat remaining tail-pairs))) body))
              live-outs (filterv #(contains? used-later %) defined)
              input-tags (mapv #(param-tag (tag-of env %) lenient?) inputs)
              env' (into env (map (fn [s] [s (tag-of env s)]) defined))
              kinds (mapv #(frame-of (tag-of env' %) lenient?) live-outs)]
          (when (and (every? some? input-tags)
                     (every? some? kinds)
                     (not-any? defined-set inputs))
            (let [name (new-name)
                  slots (reduce (fn [acc [s k]]
                                  (-> acc
                                      (assoc-in [:slot s] [k (get-in acc [:next k])])
                                      (update-in [:next k] inc)))
                                {:slot {} :next {:double 0 :long 0 :object 0}}
                                (map vector live-outs kinds))
                  store (fn [s]
                          (let [[k i] (get-in slots [:slot s])]
                            (case k
                              :double (list 'clojure.core/aset (frames :double) i s)
                              :long (list 'clojure.core/aset (frames :long) i (list 'clojure.core/long s))
                              :object (list 'clojure.core/aset (frames :object) i s))))
                  read (fn [s]
                         (let [[k i] (get-in slots [:slot s])
                               value (list 'clojure.core/aget (frames k) i)]
                           (if (= 'int (tag-of env' s)) (list 'clojure.core/int value) value)))
                  helper {:name name
                          :params (into (mapv (fn [k] (with-meta (frames k) {:tag (frame-tag k)}))
                                              [:double :long :object])
                                        (map (fn [s t] (with-meta s {:tag t})) inputs input-tags))
                          :return-tag 'double
                          :body (list 'let* (vec (mapcat identity gpairs))
                                      (list* 'do (concat (map store live-outs) [0.0])))}
                  call-sym (with-meta (gensym "outlined_") {:tag 'double})]
              (recur (rest groups) env'
                     (-> out-pairs
                         (conj [call-sym (list* name (frames :double) (frames :long) (frames :object) inputs)])
                         (into (map (fn [s] [s (read s)]) live-outs)))
                     (conj helpers helper)
                     (merge-with max frame (:next slots))
                     remaining))))))))

(defn frame-bindings
  "Binding pairs that allocate the frames used by outlined chunks."
  [frames frame]
  [[(with-meta (frames :double) {:tag 'doubles}) (list 'clojure.core/double-array (max 1 (:double frame)))]
   [(with-meta (frames :long) {:tag 'longs}) (list 'clojure.core/long-array (max 1 (:long frame)))]
   [(with-meta (frames :object) {:tag 'objects}) (list 'clojure.core/object-array (max 1 (:object frame)))]])

(defn new-frames []
  {:double (gensym "outline_dframe_") :long (gensym "outline_lframe_") :object (gensym "outline_oframe_")})

(defn- walk
  "Outline oversized blocks inside loops of `form`, tracking binder tags."
  [form env ctx state]
  (cond
    (not (seq? form)) form

    (form/binding-form? form)
    (let [[head binds & body] form
          pairs (mapv vec (partition 2 binds))
          [pairs env] (reduce (fn [[acc env] [s init]]
                                [(conj acc [s (walk init env ctx state)])
                                 (assoc env s (tag-of env s))])
                              [[] env] pairs)
          body (mapv #(walk % env ctx state) body)
          big? (> ((:size-of ctx) form) (:block-threshold ctx))]
      (if-let [{new-pairs :pairs helpers :helpers frame :frame}
               (when (and (:in-loop? ctx) big? (= 1 (count body)) (every? symbol? (map first pairs)))
                 (outline-bindings pairs body env ctx))]
        (do (swap! state (fn [s] (-> s
                                     (update :helpers into helpers)
                                     (update :frame #(merge-with max % frame)))))
            (list 'let* (vec (mapcat identity new-pairs)) (first body)))
        (apply list head (vec (mapcat identity pairs)) body)))

    (contains? '#{loop loop*} (first form))
    (let [[head binds & body] form
          pairs (partition 2 binds)
          env (into env (map (fn [[s _]] [s (tag-of env s)]) pairs))
          ctx (assoc ctx :in-loop? true)]
      (apply list head
             (vec (mapcat (fn [[s init]] [s (walk init env ctx state)]) pairs))
             (map #(walk % env ctx state) body)))

    :else (apply list (map #(walk % env ctx state) form))))

(defn outline
  "Outline oversized binding blocks inside loops of `form`. Only primitive
   values are carried across chunk boundaries.

   options: :size-of :block-threshold :chunk-budget :new-name :env {sym tag}

   Returns {:form rewritten :helpers [{:name :params :return-tag :body}]}.
   With no helpers, :form is `form` unchanged."
  [form {:keys [env] :as options}]
  (let [frames (new-frames)
        state (atom {:helpers [] :frame {:double 0 :long 0 :object 0}})
        rewritten (walk form (or env {}) (assoc options :frames frames :lenient? false) state)
        {:keys [helpers frame]} @state]
    (if (empty? helpers)
      {:form form :helpers []}
      {:form (list 'let* (vec (mapcat identity (frame-bindings frames frame))) rewritten)
       :helpers helpers})))
