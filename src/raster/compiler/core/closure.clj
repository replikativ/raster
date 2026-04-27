(ns raster.compiler.core.closure
  "Terminal closure generation for compiled Raster forms."
  (:require [raster.compiler.core.hoist :as hoist]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.op-descriptor :as op]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.ir.form :as form]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.hoisted :as hoisted]
            [raster.compiler.core.inference :as inf])
  (:import [java.lang.reflect Method]))

;; ================================================================
;; Free variable analysis
;; ================================================================

(def ^:private free-syms
  "Collect free symbol references in a form, excluding locally-bound symbols.
   Delegates to util/free-syms (scoped version)."
  util/free-syms)

;; ================================================================
;; Size-based method partitioning
;; ================================================================

(def ^:private ^:const METHOD_SIZE_BUDGET
  "Target max bytecode size per method in bytes. JVM limit is 64KB,
   but C2 aggressively optimizes methods under ~8KB. We target 25KB
   to stay well within limits while allowing meaningful method bodies."
  25000)

(defn- estimate-bytecode-size
  "Rough estimate of bytecode size for a form. Used to decide when
   to split into a new method. Heuristic — doesn't need to be exact,
   just proportional to actual bytecode."
  [form]
  (cond
    (not (seq? form)) 5
    :else
    (let [head (first form)]
      (cond
        ;; Loops are the biggest contributors (dotimes/loop, not fn*/par)
        (= :scope (:kind (form/form-info form)))
        (+ 40 (reduce + (map estimate-bytecode-size (rest form))))
        ;; Nested let — sum all bindings
        (form/binding-form? form)
        (let [bindings (second form)
              pairs (partition 2 bindings)]
          (+ 10 (reduce + (map (fn [[_ init]] (estimate-bytecode-size init)) pairs))
             (reduce + (map estimate-bytecode-size (drop 2 form)))))
        ;; if/when
        (= 'if head)
        (+ 15 (reduce + (map estimate-bytecode-size (rest form))))
        ;; Function calls
        (symbol? head) (+ 20 (* 5 (count (rest form))))
        ;; Default
        :else (+ 10 (reduce + 0 (map estimate-bytecode-size (rest form))))))))

(defn- partition-bindings-by-size
  "Partition a sequence of [sym expr] pairs into groups where each group's
   total estimated bytecode stays under METHOD_SIZE_BUDGET. Returns a vector
   of vectors, each containing the pairs for one method.

   Bindings that are simple (aliases, literals, small calls) are kept in
   the orchestrator. Only substantial bindings (loops, nested-lets, large
   expressions) are grouped into helper methods."
  [pairs type-env tag-for-param all-compute-params]
  (let [;; Classify each binding
        classified
        (mapv (fn [[sym expr]]
                (let [size (estimate-bytecode-size expr)
                      simple? (or (symbol? expr) (number? expr) (nil? expr)
                                  (and (seq? expr) (< size 50)))]
                  {:sym sym :expr expr :size size :simple? simple?}))
              pairs)]
    ;; Group non-simple bindings into sized chunks
    (loop [remaining classified
           current-group []
           current-size 0
           groups []]
      (if (empty? remaining)
        (if (seq current-group)
          (conj groups current-group)
          groups)
        (let [{:keys [simple? size] :as binding} (first remaining)]
          (if simple?
            ;; Simple bindings stay in orchestrator (pass through)
            (recur (rest remaining)
                   (conj current-group (assoc binding :keep-inline true))
                   current-size
                   groups)
            ;; Non-simple: check if adding it exceeds the budget
            (if (and (seq current-group) (> (+ current-size size) METHOD_SIZE_BUDGET))
              ;; Start a new group
              (recur remaining [] 0 (conj groups current-group))
              ;; Add to current group
              (recur (rest remaining)
                     (conj current-group binding)
                     (+ current-size size)
                     groups))))))))

(defn- expr-returns-void?
  "True if the expression's last evaluated form is void (aset, effect).
   Such expressions produce no value on the JVM stack."
  [expr]
  (cond
    (not (seq? expr)) false
    ;; aset is void
    (op/aset-op? (first expr)) true
    ;; loop/if: check the else/return branch
    (= 'loop (first expr))
    (let [body (last expr)]
      (if (and (seq? body) (= 'if (first body)))
        (expr-returns-void? (last body))  ;; check else branch
        false))
    ;; if: check both branches
    (= 'if (first expr))
    (or (expr-returns-void? (nth expr 2 nil))
        (expr-returns-void? (nth expr 3 nil)))
    ;; let: check last body form
    (form/binding-form? expr)
    (expr-returns-void? (last expr))
    :else false))

(defn- extractable-binding?
  "True if a binding's init expression should be extracted to a separate method.
   Criteria: estimated bytecode size > 50 bytes. Void-returning expressions are
   also extractable — the bytecoder infers the return type automatically."
  [expr]
  (> (estimate-bytecode-size expr) 50))

(defn hoist-and-compile
  "Like hoist-and-eval but compiles the compute body via the bytecode compiler
   instead of Clojure eval. Produces native float/double/long primitive bytecode
   without Clojure's boxing limitations.

   The outer shell (hoisted buffer allocation, lazy init) stays in Clojure eval.
   The inner compute body is split into multiple JVM static methods:
   - One orchestrator method (just calls, no loops)
   - One helper method per loop body (dotimes, loop, nested-let with loops)
   Each helper is small enough for C2 to optimize independently. Same-class
   invokestatic enables JIT inlining of helpers into the orchestrator.

   Returns {:fn compiled-fn :hoisted-count N}."
  ([form all-params fn-params]
   (hoist-and-compile form all-params fn-params {}))
  ([form all-params fn-params opts]
   (let [params-set (set all-params)
         param-env-clean (when-let [pe (:param-env opts)]
                           (into {} (map (fn [[k v]] [(with-meta k nil) v]) pe)))
         dim-env (or (hoist/build-dim-env form params-set) {})
         [_ bindings-vec & body-exprs] form
         pairs (vec (map vec (partition 2 bindings-vec)))
         _ (when (System/getProperty "raster.debug.hoist")
             (let [pred #(or (.startsWith (str %) "K_")
                             (.startsWith (str %) "V_")
                             (.startsWith (str %) "Q_"))
                   tagged-syms (filter pred (map first pairs))]
               (println "  [hoist-input] " (count pairs) "pairs total, K/V/Q-pattern:" (vec tagged-syms))))
         dtype (:dtype opts)
         rewritten-pairs (hoist/rewrite-alloc-exprs pairs dim-env params-set dtype)
         _ (when (System/getProperty "raster.debug.hoist")
             (let [pred #(or (.startsWith (str %) "K_")
                             (.startsWith (str %) "V_")
                             (.startsWith (str %) "Q_"))
                   tagged-syms (filter pred (map first rewritten-pairs))]
               (println "  [post-rewrite-alloc] " (count rewritten-pairs) "pairs total, K/V/Q-pattern:" (vec tagged-syms))))
         safe-pairs (vec (filter #(hoist/hoist-safe-pair? params-set %) rewritten-pairs))
         safe-syms-set (set (map first safe-pairs))
         inner-pairs (vec (remove (fn [[sym _]] (contains? safe-syms-set sym)) rewritten-pairs))
         _ (when (System/getProperty "raster.debug.hoist")
             (let [pred #(or (.startsWith (str %) "K_")
                             (.startsWith (str %) "V_")
                             (.startsWith (str %) "Q_"))
                   inner-tagged (filter pred (map first inner-pairs))
                   safe-tagged (filter pred (map first safe-pairs))]
               (println "  [hoist-split] inner-pairs K/V/Q:" (vec inner-tagged))
               (println "  [hoist-split] safe-pairs K/V/Q:" (vec safe-tagged))))
         safe-buf-syms (mapv first safe-pairs)
         alloc-body (vec (map second safe-pairs))
         ;; In bytecode path, hoisted bufs are direct params — no rebind needed.
         ;; Infer array tag from allocation expression
         safe-alloc-map (into {} safe-pairs)
         infer-alloc-tag (fn [sym]
                           (or (:tag (meta sym))
                               (:raster.type/tag (meta sym))
                               (when-let [alloc (get safe-alloc-map sym)]
                                 (when (seq? alloc)
                                   (let [h (first alloc)]
                                     (or (op/alloc-sym->array-tag h)
                                         (when (and (symbol? h) (namespace h))
                                           (try (:raster.core/return-tag (meta (resolve h)))
                                                (catch Exception _ nil)))))))
                               'doubles))
         ;; Fill hoisted buffers before each call.
         ;; Re-fill hoisted buffers at the start of each call.
         ;; Three modes:
         ;;   (aclone src)           → System/arraycopy from source (copy-init)
         ;;   (double-array n 1.0)   → Arrays.fill(buf, 1.0)     (value-init)
         ;;   (double-array n)       → Arrays.fill(buf, 0.0)     (zero-init)
         zero-fills (vec (mapcat (fn [s]
                                   (when (hoist/needs-zeroing? s inner-pairs)
                                     (let [arr-tag (infer-alloc-tag s)
                                           typed-s (with-meta (gensym (str (name s) "_t_"))
                                                     {:tag arr-tag})
                                           alloc-expr (get safe-alloc-map s)
                                           ;; Detect aclone: (aclone src) or (.invk aclone-impl src)
                                           aclone-src (when (seq? alloc-expr)
                                                        (let [head (first alloc-expr)
                                                              hname (when (symbol? head) (name head))]
                                                          (when (and hname (or (.contains ^String hname "aclone")
                                                                               (= hname "aclone")))
                                                            ;; Source is the second arg (or third for .invk)
                                                            (if (= '.invk head) (nth alloc-expr 2) (second alloc-expr)))))
                                           ;; Only `(*-array n val)` carries a fill value as the 3rd
                                           ;; element. Other 3-element allocations like
                                           ;; `(zeros-like ref n)` have length-not-value as the 3rd
                                           ;; element — treating that as a fill value emits
                                           ;; `Arrays.fill(buf, <length>)` and then JVM overload
                                           ;; resolution picks `Arrays.fill(short[], short)`,
                                           ;; producing a `[D → [S` runtime cast.
                                           array-ctor? (fn [head]
                                                         (when (symbol? head)
                                                           (let [n (name head)]
                                                             (and (.endsWith n "-array")
                                                                  (or (nil? (namespace head))
                                                                      (= "clojure.core" (namespace head)))))))
                                           fill-val (when (and (not aclone-src)
                                                               (seq? alloc-expr) (= 3 (count alloc-expr))
                                                               (array-ctor? (first alloc-expr)))
                                                      (nth alloc-expr 2))
                                           default-fill (case arr-tag
                                                          doubles 0.0
                                                          floats '(float 0.0)
                                                          longs '(long 0)
                                                          ints '(int 0)
                                                          shorts '(short 0)
                                                          bytes '(byte 0)
                                                          chars (char \u0000)
                                                          booleans false
                                                          objects nil
                                                          0.0)]
                                       (if aclone-src
                                         ;; Copy-init: arraycopy from source
                                         [typed-s s
                                          (gensym "_copy_")
                                          (list 'System/arraycopy aclone-src 0 typed-s 0
                                                (list 'clojure.core/alength typed-s))]
                                         ;; Zero/value-init: Arrays.fill
                                         [typed-s s
                                          (gensym "_fill_")
                                          (list 'java.util.Arrays/fill typed-s
                                                (or fill-val default-fill))]))))
                                 safe-buf-syms))
         ;; ================================================================
         ;; Method extraction: split loop-heavy bindings into separate methods
         ;; ================================================================
         ;; Track types: param-env + hoisted buf types + binding types
         type-env (atom (merge (or param-env-clean {})
                               (into {} (map (fn [s] [s (infer-alloc-tag s)]) safe-buf-syms))))
         ;; Track object-array element types for aget inference.
         ;; When we see (object-array [e0 e1 e2]), store {arr-sym -> [type0 type1 type2]}
         ;; so (aget arr-sym 1) can resolve to type1.
         ;; Pre-populate from hoisted bindings (safe-pairs).
         objarray-eltypes (atom (into {}
                                      (keep (fn [[sym expr]]
                                              (when (and (seq? expr)
                                                         (= 'clojure.core/object-array (first expr))
                                                         (vector? (second expr)))
                                                (let [elts (second expr)
                                                      te @type-env
                                                      elt-types (mapv #(when (symbol? %) (get te %)) elts)]
                                                  [sym elt-types])))
                                            safe-pairs)))
         tag-for-param (fn [p]
                         (or (when param-env-clean (get param-env-clean (with-meta p nil)))
                             (when (contains? safe-syms-set p)
                               (infer-alloc-tag p))
                             (:tag (meta p))
                             ;; AD stamps grad syms with :raster.type/tag (e.g. 'doubles).
                             ;; Without reading it, helper params get 'Object, and the
                             ;; aget/aset fallback checkcasts to Object[] — fails at
                             ;; runtime when the captured array is double[].
                             (:raster.type/tag (meta p))
                             (get @type-env p)
                             'Object))
         ;; Build type env from params + hoisted buffers + preceding bindings
         all-compute-params (vec (concat fn-params safe-buf-syms))
         bc-params (mapv (fn [p] (with-meta p {:tag (tag-for-param p)}))
                         all-compute-params)
         element-type (case dtype :float :float :double)
         desugar-invk bytecode/desugar-invk
         ;; Extract method-worthy bindings into separate fn-specs
         helper-specs (atom [])
         helper-counter (atom 0)
         rewritten-pairs
         (vec (map (fn [[sym expr]]
                     (if (extractable-binding? expr)
                       ;; Extract to a separate method
                       (let [method-name (symbol (str "helper_" (swap! helper-counter inc)))
                             free (disj (free-syms expr) '.invk)
                             ;; Order free vars deterministically, params first
                             ordered-free (vec (concat
                                                (filter free all-compute-params)
                                                (sort (remove (set all-compute-params) free))))
                             ;; Infer return type from the binding's tag or the expr
                             ;; For ref types (arrays, objects), use Object to avoid
                             ;; type mismatches with IFn.invoke returning Object.
                             ;; Void-returning bodies are handled by the bytecoder:
                             ;; emit-body returns :void, Object-return coerces to null.
                             raw-return-tag (or (:tag (meta sym))
                                                (:raster.type/tag (meta sym))
                                                (inf/infer-arg-tag expr @type-env)
                                                'Object)
                             return-tag (if (contains? #{'double 'float 'long 'int 'boolean} raw-return-tag)
                                          raw-return-tag
                                          'Object)
                             ;; Build typed params for the helper
                             helper-params (mapv (fn [p]
                                                   (with-meta p {:tag (or (get @type-env p)
                                                                          (tag-for-param p))}))
                                                 ordered-free)
                             ;; The helper body is the extracted expression
                             helper-body (desugar-invk expr)]
                         (swap! helper-specs conj
                                {:name method-name
                                 :params helper-params
                                 :return-tag return-tag
                                 :walked-body [helper-body]
                                 :element-type element-type
                                 :source-binding sym
                                 :source-expr expr})
                         ;; Update type env with the ACTUAL type (not the method
                         ;; return type which is Object for arrays).  Downstream
                         ;; helpers need the real type for their param annotations.
                         (swap! type-env assoc sym raw-return-tag)
                         ;; Replace the binding with a call to the helper
                         [sym (apply list method-name ordered-free)])
                       (do
                         ;; Track object-array element types for aget inference
                         (when (and (seq? expr)
                                    (= 'clojure.core/object-array (first expr))
                                    (vector? (second expr)))
                           (let [elts (second expr)
                                 elt-types (mapv #(when (symbol? %) (get @type-env %)) elts)]
                             (swap! objarray-eltypes assoc sym elt-types)))
                         ;; Update type env for non-extracted bindings.
                         ;; Check symbol aliases, aget on object-arrays, then inference.
                         (when-let [tag (or (:tag (meta sym))
                                            (:raster.type/tag (meta sym))
                                            (when (symbol? expr) (get @type-env expr))
                                            ;; aget on object-array: resolve element type
                                            (when (and (seq? expr)
                                                       (= 'clojure.core/aget (first expr))
                                                       (= 3 (count expr))
                                                       (number? (nth expr 2)))
                                              (let [arr (second expr)
                                                    idx (long (nth expr 2))
                                                    elt-types (get @objarray-eltypes arr)]
                                                (when (and elt-types (< idx (count elt-types)))
                                                  (nth elt-types idx))))
                                            (let [infer inf/infer-arg-tag]
                                              (infer expr @type-env)))]
                           (swap! type-env assoc sym tag))
                         [sym expr])))
                   inner-pairs))
         inner-bindings (vec (mapcat identity rewritten-pairs))
         ;; Compute form: zero-fills + rewritten inner bindings
         compute-form (list* 'let* (vec (concat zero-fills inner-bindings)) body-exprs)
         compute-form (desugar-invk compute-form)
         ;; Re-tag all bindings after extraction — helper calls return Object,
         ;; and the let* handler needs :tag metadata to emit checkcast
         compute-form (let [tag-walk
                            (fn tag-walk [form env]
                              (if (form/binding-form? form)
                                (let [[head binds & body] form
                                      pairs (partition 2 binds)
                                      [new-env new-binds]
                                      (reduce
                                       (fn [[env acc] [sym init]]
                                         (let [tagged-init (tag-walk init env)
                                               tag (or (:tag (meta sym))
                                                       (:raster.type/tag (meta sym))
                                                       (inf/infer-arg-tag tagged-init env))
                                               tagged-sym (if tag (vary-meta sym assoc :tag tag) sym)
                                               new-env (if tag (assoc env sym tag) env)]
                                           [new-env (conj acc tagged-sym tagged-init)]))
                                       [env []] pairs)]
                                  (apply list head (vec new-binds) (map #(tag-walk % new-env) body)))
                                form))]
                        (tag-walk compute-form @type-env))
         _debug-dump (when (System/getProperty "raster.debug.compute-form")
                       (let [src (pr-str compute-form)]
                         (spit "/tmp/compute-form.txt" src)
                         (println "  [compute-form] dumped, len=" (count src))
                         (when-let [[head binds & _body] (when (seq? compute-form) compute-form)]
                           (when (#{'let 'let*} head)
                             (let [pairs (partition 2 binds)
                                   syms (mapv first pairs)]
                               (println "  [compute-form] binding count:" (count syms))
                               (doseq [sym syms
                                       :when (or (.startsWith (str sym) "K_")
                                                 (.startsWith (str sym) "V_")
                                                 (.startsWith (str sym) "Q_"))]
                                 (println "  [compute-form] has binding:" sym)))))))
         ;; Compile via bytecode — static methods + IFn wrapper class
         compile-hoisted! hoisted/compile-hoisted-class!
         class-name (str "raster.compiled.CF_" (System/nanoTime))
         deftm-ns (or (:source-ns opts)
                      (when-let [pe param-env-clean]
                        (some (fn [[k _]] (when (namespace k) (the-ns (symbol (namespace k)))))
                              pe))
                      *ns*)
         ;; All fn-specs: main compute + extracted helpers
         ;; Return tag: from deftm declaration, or default scalar type
         ret-tag (or (:return-tag opts)
                     'Object)
         source-file (or (:source-file opts)
                         (when deftm-ns
                           (some-> deftm-ns str
                                   (.replace "." "/")
                                   (.replace "-" "_")
                                   (str ".clj"))))
         all-fn-specs (into [{:name 'compute
                              :params bc-params
                              :return-tag ret-tag
                              :walked-body [compute-form]
                              :element-type element-type
                              :source-ns deftm-ns
                              :source-file source-file}]
                            (map (fn [spec]
                                   (assoc spec
                                          :element-type element-type
                                          :source-ns deftm-ns
                                          :source-file source-file))
                                 @helper-specs))
         ;; ================================================================
         ;; Type validation: catch Object-typed params before bytecode gen
         ;; ================================================================
         _ (when (System/getProperty "raster.debug")
             (doseq [{:keys [name params]} all-fn-specs]
               (doseq [p params]
                 (when (= 'Object (:tag (meta p)))
                   (println (format "WARN: %s param '%s' typed as Object (may box)" name p))))))
         ;; Validate: warn about Object-typed helper params that should be arrays
         _ (doseq [{:keys [name params]} all-fn-specs]
             (let [obj-params (filterv #(= 'Object (:tag (meta %))) params)
                   ;; Only warn for non-compute methods (helpers with free vars)
                   ;; Compute params are typed by param-env
                   warn? (and (not= name 'compute) (seq obj-params))]
               (when warn?
                 (binding [*out* *err*]
                   (println (format "WARNING: helper '%s' has %d Object-typed params: %s"
                                    name (count obj-params)
                                    (mapv str obj-params)))))))
         ;; Typed interface for zero-boxing invocation
         fn-param-tags (mapv tag-for-param fn-params)
         iface-info (types/ensure-fn-interface! fn-param-tags)
         iface-class (:iface-class iface-info)
         field-tags (mapv infer-alloc-tag safe-buf-syms)
         ;; Compile: generates statics class + wrapper class (extends AFn)
         bc-result (compile-hoisted! class-name all-fn-specs
                                     {:field-tags   field-tags
                                      :fn-param-tags fn-param-tags
                                      :return-tag   ret-tag
                                      :compute-name 'compute
                                      :iface-class  iface-class})
         wrapper-class (:wrapper-class bc-result)
         ;; The wrapper IS the IFn — no Clojure fn wrapper, no eval on hot path.
         ;; Create the wrapper instance with the alloc-fn. On first invoke(),
         ;; the wrapper calls alloc-fn to create buffers, then delegates to invk.
         ;; Strip :tag metadata from params and alloc-body: the alloc-fn is Clojure-
         ;; eval'd (not bytecode-compiled), and type hints on params cause JDK 27
         ;; to select int-returning method overloads that can't be coerced to long.
         strip-tags (fn strip [form]
                      (cond
                        (and (symbol? form) (:tag (meta form)))
                        (with-meta form (dissoc (meta form) :tag))
                        (seq? form) (with-meta (apply list (map strip form)) (meta form))
                        (vector? form) (mapv strip form)
                        :else form))
         clean-params (mapv #(with-meta % nil) all-params)
         clean-alloc-body (strip-tags alloc-body)
         alloc-fn (if (<= (count all-params) 20)
                    (eval (list 'fn clean-params clean-alloc-body))
                    ;; >20 params: use varargs + nth destructuring
                    (let [args-sym (gensym "args_")
                          destructure (vec (mapcat
                                            (fn [i p] [(with-meta p nil) (list 'nth args-sym i)])
                                            (range) all-params))]
                      (eval (list 'fn ['& args-sym]
                                  (list 'let destructure clean-alloc-body)))))
         ctor (first (.getConstructors wrapper-class))
         combined-fn (.newInstance ctor (object-array [alloc-fn]))
         ;; Diagnostics: helper→source mapping + class bytes for inspection
         helper-map (into {} (map (fn [spec]
                                    [(str (:name spec))
                                     {:binding (:source-binding spec)
                                      :expr (:source-expr spec)
                                      :params (:params spec)
                                      :return-tag (:return-tag spec)}])
                                  @helper-specs))]
     {:fn combined-fn
      :hoisted-count (count safe-pairs)
      :statics-bytes (:statics-bytes bc-result)
      :wrapper-bytes (:wrapper-bytes bc-result)
      :helper-map helper-map
      :compute-form compute-form
      :alloc-form (list 'fn clean-params clean-alloc-body)})))

(defn vector-body->typed-array
  "Transform a let* form body from returning a vector [a b c]
  to returning a typed array with direct aset.
  dtype controls the array type (:double, :float, etc.)."
  [form dtype]
  (if (form/binding-form? form)
    (let [[let-sym bindings & body-exprs] form
          last-expr (last body-exprs)]
      (if (vector? last-expr)
        (let [arr-sym (gensym "result_arr_")
              n (count last-expr)
              alloc-sym (device/dtype->array-ctor dtype)
              aset-bindings (vec (mapcat (fn [i sym]
                                           [(gensym "_") (list 'aset arr-sym (int i) sym)])
                                         (range) last-expr))
              new-bindings (vec (concat bindings
                                        [arr-sym (list alloc-sym n)]
                                        aset-bindings))
              new-body (if (= 1 (count body-exprs))
                         [arr-sym]
                         (concat (butlast body-exprs) [arr-sym]))]
          (list* let-sym new-bindings new-body))
        form))
    form))

