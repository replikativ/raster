(ns raster.compiler.backend.wasm.emit
  "Track A: walk raster's POST-PASS IR (the S-expr compile-aot hands the bytecode
   backend) and emit a WebAssembly module via encoder.clj. Reads the semantic op
   from :raster.op/original metadata (never the mangled impl name), per the
   compiler design rules.

   v1 scalar node set (saxpy-class):
     let* / loop / recur / if / do
     .invk (raster.numeric +,-,*,/)  ·  clojure.core/{aget,aset,<,<=,>,>=,inc,dec}
     long/int/double casts  ·  param & loop-local syms  ·  numeric literals
   Arrays live in linear memory as (ptr,len); index/count are i32 (v1 — 32-bit
   memory). Float element type from the array param's tag (doubles→f64, floats→f32)."
  (:require [raster.compiler.backend.wasm.encoder :as e]))

;; --- tag → wasm types ------------------------------------------------------
(def ^:private array-tag->elem
  {'doubles {:vt :f64 :bytes 8 :load :f64.load :store :f64.store}
   'floats  {:vt :f32 :bytes 4 :load :f32.load :store :f32.store}
   'longs   {:vt :i64 :bytes 8 :load :i64.load :store :i64.store}
   'ints    {:vt :i32 :bytes 4 :load :i32.load :store :i32.store}})

(defn- array-tag? [tag] (contains? array-tag->elem tag))

(defn- tag->vt
  "Scalar raster tag → wasm valtype. Arrays are i32 pointers. Long/int indices
   are i32 in v1 (32-bit linear memory)."
  [tag]
  (cond
    (= tag 'double) :f64
    (= tag 'float)  :f32
    (#{'long 'int} tag) :i32
    (array-tag? tag) :i32
    :else :i32))

;; --- arithmetic op (read from metadata) → wasm opcode by result type -------
(defn- arith-op [sem vt]
  (let [f? (#{:f64 :f32} vt)]
    (case sem
      raster.numeric/+ (if f? :f64.add :i32.add)
      raster.numeric/- (if f? :f64.sub :i32.sub)
      raster.numeric/* (if f? :f64.mul :i32.mul)
      raster.numeric// :f64.div
      nil)))

;; ctx: {:env {sym {:idx :vt}}  :elems {sym elem-map}  :ret :void|vt}
(declare emit-val emit-stmt)

(defn- addr
  "byte address of array element: ptr + idx*elem-bytes  (i32)."
  [ctx arr-sym idx-node]
  (let [pidx (get-in ctx [:env arr-sym :idx])
        eb   (get-in ctx [:elems arr-sym :bytes])]
    (into (into (e/local-get pidx)
                (emit-val ctx idx-node))
          (into (e/i32-const eb) (concat (e/i (keyword "i32.mul")) (e/i :i32.add))))))

(defn- emit-val
  "Emit a value-producing expression → byte vector."
  [ctx node]
  (cond
    (integer? node) (e/i32-const node)            ; index/count literal (i32 in v1)
    (float? node)   (e/f64-const node)
    (symbol? node)  (let [{:keys [idx]} (get-in ctx [:env node])]
                      (when-not idx (throw (ex-info (str "unbound sym " node) {:node node})))
                      (e/local-get idx))
    (seq? node)
    (let [h (first node), A (vec (rest node))]
      (cond
        (= h 'long)  (emit-val ctx (first A))      ; index cast: identity for i32
        (= h 'int)   (emit-val ctx (first A))
        (= h 'double)(emit-val ctx (first A))
        (= h 'clojure.core/aget)
        (let [[arr idx] A
              elem (get-in ctx [:elems arr])]
          (into (addr ctx arr idx) (e/mem-load (:load elem) 3 0)))
        (= h 'clojure.core/<)  (into (into (emit-val ctx (A 0)) (emit-val ctx (A 1))) (e/i :i32.lt_s))
        (= h 'clojure.core/<=) (into (into (emit-val ctx (A 0)) (emit-val ctx (A 1))) (e/i :i32.le_s))
        (= h 'clojure.core/>)  (into (into (emit-val ctx (A 0)) (emit-val ctx (A 1))) (e/i :i32.gt_s))
        (= h 'clojure.core/inc)(into (into (emit-val ctx (A 0)) (e/i32-const 1)) (e/i :i32.add))
        (= h 'clojure.core/dec)(into (into (emit-val ctx (A 0)) (e/i32-const 1)) (e/i :i32.sub))
        (= h '.invk)
        (let [sem (:raster.op/original (meta node))
              vt  (case (:raster.type/tag (meta node)) double :f64 float :f32 :i32)
              opk (or (arith-op sem vt) (throw (ex-info (str "unhandled .invk op " sem) {})))
              [_impl o1 o2] A]
          (into (into (emit-val ctx o1) (emit-val ctx o2)) (e/i opk)))
        :else (throw (ex-info (str "unhandled value head " h) {:node node}))))
    :else (throw (ex-info (str "unhandled value node " (pr-str node)) {}))))

(defn- emit-stmt
  "Emit a statement (effect / control) → byte vector. loop-label = br depth to
   the enclosing loop's header from the current point."
  [ctx node loop-label]
  (let [h (first node), A (vec (rest node))]
    (cond
      (= h 'do)        (vec (mapcat #(emit-stmt ctx % loop-label) A))
      (= h 'clojure.core/aset)
      (let [[arr idx v] A
            elem (get-in ctx [:elems arr])]
        (into (into (addr ctx arr idx) (emit-val ctx v)) (e/mem-store (:store elem) 3 0)))
      (= h 'recur)                                   ; (recur newval) → set loop var, re-loop
      (into (into (emit-val ctx (first A)) (e/local-set (:loop-idx ctx)))
            (e/br loop-label))
      (= h 'if)
      (let [[cnd then _else] A]
        ;; cond ; if { then } end   (else discarded: loop falls through → exits)
        (into (into (emit-val ctx cnd) [(e/op :if) e/empty-block])
              (into (emit-stmt ctx then 1)            ; depth 1 = the loop (if is 0)
                    [(e/op :end)])))
      :else (throw (ex-info (str "unhandled stmt head " h) {:node node})))))

(defn compile-kernel
  "Compile a saxpy-class kernel IR to a wasm module byte[].
   params: vector of {:sym sym :tag tag} in order.  ir: post-pass S-expr.
   Returns {:bytes byte[] :name name :param-types [vt...]}."
  [{:keys [name params ir mem-pages] :or {mem-pages 256}}]
  (let [param-vts (mapv #(tag->vt (:tag %)) params)
        env (into {} (map-indexed (fn [idx {:keys [sym tag]}] [sym {:idx idx :vt (tag->vt tag)}]) params))
        elems (into {} (keep (fn [{:keys [sym tag]}] (when (array-tag? tag) [sym (array-tag->elem tag)])) params))
        loop-idx (count params)                       ; first local after params = loop var i
        ctx {:env (assoc env nil nil) :elems elems :loop-idx loop-idx}
        ;; unwrap (let* [] (loop [i init] body))
        [_let _binds loop-form] ir
        [_loop bindvec body] loop-form
        i-sym (first bindvec)
        i-init (second bindvec)
        ctx (assoc-in ctx [:env i-sym] {:idx loop-idx :vt :i32})
        body-bytes (into (into (emit-val ctx i-init) (e/local-set loop-idx))
                         (e/loop* (emit-stmt ctx body 0)))
        functype (e/functype param-vts [])]
    {:name name
     :param-types param-vts
     :bytes (e/build-module
             {:types [functype]
              :funcs [{:name name :type-idx 0
                       :param-types param-vts :result-types []
                       :locals [:i32]               ; the loop var i
                       :body body-bytes}]
              :memory {:min mem-pages :max mem-pages :export "memory"}})}))
