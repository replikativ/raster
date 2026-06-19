(ns raster.compiler.backend.wasm.emit
  "Track A: walk raster's POST-PASS IR (the S-expr compile-aot hands the bytecode
   backend) and emit a WebAssembly module via encoder.clj. Reads the semantic op
   from :raster.op/original metadata (never the mangled impl name), per the
   compiler design rules.

   v1 node set:
     scalar body         — value expr → function returns it
     (loop [v init ...] (if cond <recur-or-effects+recur> <result>))
     let* / do / if / recur / .invk(raster.numeric +,-,*,/)
     clojure.core/{aget,aset,<,<=,>,>=,inc,dec} · long/int/double casts
     param & loop-local syms · numeric literals
   Arrays live in linear memory as (ptr,len); index/count are i32 (v1 — 32-bit
   memory). Float element type from the array param's tag (doubles→f64, floats→f32).
   Every loop returns its non-recur (else) value, so the function result type is
   the deftm's return tag."
  (:require [raster.compiler.backend.wasm.encoder :as e]))

;; --- tag → wasm types ------------------------------------------------------
(def ^:private array-tag->elem
  {'doubles {:vt :f64 :bytes 8 :align 3 :load :f64.load :store :f64.store}
   'floats  {:vt :f32 :bytes 4 :align 2 :load :f32.load :store :f32.store}
   'longs   {:vt :i64 :bytes 8 :align 3 :load :i64.load :store :i64.store}
   'ints    {:vt :i32 :bytes 4 :align 2 :load :i32.load :store :i32.store}})

(defn- array-tag? [tag] (contains? array-tag->elem tag))

(defn- tag->vt
  "Scalar raster/annotation tag → wasm valtype. Arrays are i32 pointers; long/int
   indices are i32 in v1 (32-bit linear memory)."
  [tag]
  (cond
    (#{'double 'Double} tag) :f64
    (#{'float 'Float} tag)   :f32
    (#{'long 'int 'Long 'Integer} tag) :i32
    (array-tag? tag) :i32
    :else :i32))

(defn- arith-op [sem vt]
  (let [f? (#{:f64 :f32} vt)]
    (case sem
      raster.numeric/+ (if f? :f64.add :i32.add)
      raster.numeric/- (if f? :f64.sub :i32.sub)
      raster.numeric/* (if f? :f64.mul :i32.mul)
      raster.numeric// :f64.div
      nil)))

;; ctx: {:env {sym {:idx :vt}}  :elems {sym elem-map}}
(declare emit-val)

(defn- addr
  "byte address of array element: ptr + idx*elem-bytes  (i32)."
  [ctx arr-sym idx-node]
  (let [pidx (get-in ctx [:env arr-sym :idx])
        eb   (get-in ctx [:elems arr-sym :bytes])]
    (-> (e/local-get pidx)
        (into (emit-val ctx idx-node))
        (into (e/i32-const eb)) (into (e/i :i32.mul)) (into (e/i :i32.add)))))

(defn- emit-val
  "Emit a value-producing expression → byte vector."
  [ctx node]
  (cond
    (integer? node) (e/i32-const node)
    (float? node)   (e/f64-const node)
    (symbol? node)  (let [{:keys [idx]} (get-in ctx [:env node])]
                      (when-not idx (throw (ex-info (str "unbound sym " node) {:node node})))
                      (e/local-get idx))
    (seq? node)
    (let [h (first node), A (vec (rest node))]
      (cond
        (#{'long 'int 'double} h) (emit-val ctx (first A)) ; index casts: identity in v1
        (= h 'clojure.core/aget)
        (let [[arr idx] A elem (get-in ctx [:elems arr])]
          (into (addr ctx arr idx) (e/mem-load (:load elem) (:align elem) 0)))
        (= h 'clojure.core/<)  (-> (emit-val ctx (A 0)) (into (emit-val ctx (A 1))) (into (e/i :i32.lt_s)))
        (= h 'clojure.core/<=) (-> (emit-val ctx (A 0)) (into (emit-val ctx (A 1))) (into (e/i :i32.le_s)))
        (= h 'clojure.core/>)  (-> (emit-val ctx (A 0)) (into (emit-val ctx (A 1))) (into (e/i :i32.gt_s)))
        (= h 'clojure.core/>=) (-> (emit-val ctx (A 0)) (into (emit-val ctx (A 1))) (into (e/i :i32.ge_s)))
        (= h 'clojure.core/inc)(-> (emit-val ctx (A 0)) (into (e/i32-const 1)) (into (e/i :i32.add)))
        (= h 'clojure.core/dec)(-> (emit-val ctx (A 0)) (into (e/i32-const 1)) (into (e/i :i32.sub)))
        (= h '.invk)
        (let [sem (:raster.op/original (meta node))
              vt  (case (:raster.type/tag (meta node)) double :f64 float :f32 :i32)
              opk (or (arith-op sem vt) (throw (ex-info (str "unhandled .invk op " sem) {})))
              [_impl o1 o2] A]
          (-> (emit-val ctx o1) (into (emit-val ctx o2)) (into (e/i opk))))
        :else (throw (ex-info (str "unhandled value head " h) {:node node}))))
    :else (throw (ex-info (str "unhandled value node " (pr-str node)) {}))))

(defn- infer-vt
  "Result valtype of an expression, from env + carried :raster.type/tag."
  [ctx node]
  (cond
    (symbol? node)  (get-in ctx [:env node :vt] :i32)
    (float? node)   :f64
    (integer? node) :i32
    (seq? node)
    (case (:raster.type/tag (meta node))
      double :f64, float :f32, long :i32, int :i32
      (cond
        (#{'long 'int 'double} (first node)) (infer-vt ctx (second node))
        (= 'clojure.core/aget (first node))  (get-in ctx [:elems (second node) :vt] :f64)
        :else :i32))
    :else :i32))

(defn- emit-effect
  "Side-effecting statement that leaves nothing on the stack (e.g. aset)."
  [ctx node]
  (let [h (first node), A (vec (rest node))]
    (cond
      (= h 'clojure.core/aset)
      (let [[arr idx v] A elem (get-in ctx [:elems arr])]
        (-> (addr ctx arr idx) (into (emit-val ctx v)) (into (e/mem-store (:store elem) (:align elem) 0))))
      (= h 'do) (vec (mapcat #(emit-effect ctx %) A))
      :else (throw (ex-info (str "unhandled effect head " h) {:node node})))))

(defn- emit-recur
  "(recur v0 v1 ...) → push all new values, set loop locals in REVERSE (so each
   value is read from the OLD locals before any are overwritten), then branch to
   the loop header at depth `loop-depth`."
  [ctx args loop-idxs loop-depth]
  (-> (vec (mapcat #(emit-val ctx %) args))
      (into (vec (mapcat #(e/local-set %) (reverse loop-idxs))))
      (into (e/br loop-depth))))

(defn- emit-then
  "then-branch of the loop's if: optional effects (a `do`) ending in a recur."
  [ctx node loop-idxs loop-depth]
  (cond
    (and (seq? node) (= 'recur (first node)))
    (emit-recur ctx (vec (rest node)) loop-idxs loop-depth)
    (and (seq? node) (= 'do (first node)))
    (let [stmts (vec (rest node))]
      (-> (vec (mapcat #(emit-effect ctx %) (butlast stmts)))
          (into (emit-then ctx (last stmts) loop-idxs loop-depth))))
    :else (throw (ex-info (str "loop then-branch must end in recur, got " (pr-str node)) {}))))

(defn compile-kernel
  "Compile a kernel IR to a wasm module byte[].
   params: ordered [{:sym :tag}].  ir: post-pass S-expr. The result type is
   inferred from the IR (loop else-value / scalar body).
   Returns {:bytes byte[] :name :param-types [vt...] :result-types [vt...]}."
  [{:keys [name params ir mem-pages] :or {mem-pages 256}}]
  (let [param-vts (mapv #(tag->vt (:tag %)) params)
        env0 (into {} (map-indexed (fn [idx {:keys [sym tag]}] [sym {:idx idx :vt (tag->vt tag)}]) params))
        elems (into {} (keep (fn [{:keys [sym tag]}] (when (array-tag? tag) [sym (array-tag->elem tag)])) params))
        base (count params)
        ;; ir = (let* [] body)  (v1: empty let* bindings)
        body (let [[h binds & more] ir]
               (assert (and (= h 'let*) (empty? binds)) "v1 expects (let* [] body)")
               (if (= 1 (count more)) (first more) (cons 'do more)))
        {:keys [locals body-bytes ret-vt]}
        (if (and (seq? body) (= 'loop (first body)))
          ;; ---- loop kernel: returns its if-else value ----
          (let [[_ bindvec ifform] body
                loop-vars (map-indexed (fn [k [sym init]]
                                         {:sym sym :idx (+ base k)
                                          :vt (if (float? init) :f64 :i32) :init init})
                                       (partition 2 bindvec))
                loop-idxs (mapv :idx loop-vars)
                ctx {:env (reduce (fn [e {:keys [sym idx vt]}] (assoc e sym {:idx idx :vt vt}))
                                  env0 loop-vars)
                     :elems elems}
                [_ cnd then els] ifform
                ret-vt (infer-vt ctx els)
                inits (vec (mapcat (fn [{:keys [idx init]}]
                                     (into (emit-val ctx init) (e/local-set idx))) loop-vars))
                ;; depths from inside the if: if=0, loop=1, block=2
                if-bytes (-> (emit-val ctx cnd)
                             (into [(e/op :if) e/empty-block])
                             (into (emit-then ctx then loop-idxs 1))
                             (into [(e/op :else)])
                             (into (emit-val ctx els)) (into (e/br 2))
                             (into [(e/op :end)]))]
            {:ret-vt ret-vt
             :locals (mapv :vt loop-vars)
             :body-bytes (into inits (e/block-t ret-vt (into (e/loop* if-bytes) (e/i :unreachable))))})
          ;; ---- scalar kernel: body value is the result ----
          (let [ctx {:env env0 :elems elems}]
            {:ret-vt (infer-vt ctx body) :locals [] :body-bytes (emit-val ctx body)}))]
    {:name name
     :param-types param-vts
     :result-types [ret-vt]
     :bytes (e/build-module
             {:types [(e/functype param-vts [ret-vt])]
              :funcs [{:name name :type-idx 0
                       :param-types param-vts :result-types [ret-vt]
                       :locals locals :body body-bytes}]
              :memory {:min mem-pages :max mem-pages :export "memory"}})}))
