(ns raster.compiler.cljs-emit
  "P1/P2 — the build-time half of the cljs↔wasm binding.

   Compiles a set of raster `deftm` kernels to ONE shared-memory wasm module and
   generates a ClojureScript counterpart of the kernel namespace, so a `.cljc`
   game can `(require [some.kernels])` and get the right impl per platform: the JVM
   `deftm`/`defvalue` source (bytecode), or this generated `.cljs` (records + a
   wasm-backed wrapper per kernel).

   Two ABI modes, both emitted as Clojure FORMS and `pr-str`'d (only the ns/defonce
   preamble is a string):

   1. Value/scalar mode (a kernel spec with no `:call`): args are scalars or value
      types (Vec2/3/4). Value-type params are passed field-by-field; a value-type
      return comes back as a wasm multi-value JS array reconstructed into a record.

   2. Array/memory mode (a kernel spec with `:call`): args include `(Array …)` that
      ride the (ptr,len) ABI — written into linear memory at an auto-assigned byte
      offset, passed as that offset, and (for `:inout`) read back after the call.
      Supporting regions are declared once at the top level and LAID OUT AUTOMATICALLY
      (8-aligned), so no caller hand-computes offsets:
        :consts   [{:sym HP :data <ints> :view :i32}]   ; baked into memory at init!
        :scratch  [{:sym POS :view :f64 :bytes 24}]      ; per-call marshal buffers
        :resident [{:sym WBLK :view :i32 :bytes N}]      ; uploaded once (upload fns)
        :uploads  [{:fn \"upload-world!\" :args [b s] :set [[WBLK b] [WSOL s]]}]
      A kernel's `:call` is the wasm export's arg list, each entry one of:
        sym                  ; pass-through cljs fn arg (scalar)
        [:lit v]             ; baked literal scalar
        [:const SYM]         ; const-region byte offset
        [:resident SYM]      ; resident-region byte offset
        [:in SYM arg]        ; marshal array `arg` into scratch SYM (write only)
        [:inout SYM arg]     ; marshal `arg` in, then read it back after the call

   Spec (value mode):  {:var #'ns/move-shape :export \"move_shape\" :fn \"move-shape\"}
   Spec (array mode):  {:var #'ns/k :export \"k\" :fn \"k\" :args [a b] :call […]}"
  (:require [raster.compiler.pipeline :as pl]
            [raster.compiler.core.types :as types]
            [clojure.java.io :as io]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; value/scalar mode (unchanged ABI: pass scalars / value-type fields through)
;; ---------------------------------------------------------------------------

(defn- value-type? [tag] (contains? @types/soa-registry tag))
(defn- field-names [tag] (mapv :name (:fields (get @types/soa-registry tag))))

(defn- kernel-sig
  [{:keys [param-names param-tags return-tag]}]
  {:params (mapv (fn [nm t]
                   (if (value-type? t)
                     {:kind :value :name (name nm) :tag t :fields (field-names t)}
                     {:kind :scalar :name (name nm)}))
                 param-names param-tags)
   :ret (cond (value-type? return-tag) {:kind :value :tag return-tag :fields (field-names return-tag)}
              (nil? return-tag)        {:kind :void}
              :else                    {:kind :scalar})})

(defn- record-form [tag]
  (list 'defrecord (symbol (name tag)) (mapv symbol (field-names tag))))

(defn- value-wrapper-form
  [export fname {:keys [params ret]}]
  (let [arglist   (mapv (comp symbol :name) params)
        call-args (vec (mapcat (fn [{:keys [kind name fields]}]
                                 (if (= :value kind)
                                   (map #(list (keyword (clojure.core/name %)) (symbol name)) fields)
                                   [(symbol name)]))
                               params))
        call (list* (list 'kfn export) call-args)]
    (if (= :value (:kind ret))
      (list 'defn (symbol fname) arglist
            (list 'let ['r call]
                  (list* (symbol (str "->" (name (:tag ret))))
                         (map-indexed (fn [i _] (list 'aget 'r i)) (:fields ret)))))
      (list 'defn (symbol fname) arglist call))))

;; ---------------------------------------------------------------------------
;; array/memory mode: automatic memory layout + (ptr,len) marshaling
;; ---------------------------------------------------------------------------

(def ^:private dtype-bytes {:i8 1 :i32 4 :f32 4 :f64 8})
(def ^:private dtype-view  {:i8 'w/i8-view :i32 'w/i32-view :f32 'w/f32-view :f64 'w/f64-view})

(defn- align8 ^long [^long n] (* 8 (quot (+ n 7) 8)))

(defn- lay-out
  "Assign 8-aligned byte offsets to const+scratch+resident regions, in order.
   Returns {sym {:off :view :bytes :data :kind}}."
  [consts scratch resident]
  (loop [acc {} off 0
         regs (concat (map #(assoc % :kind :const)    consts)
                      (map #(assoc % :kind :scratch)  scratch)
                      (map #(assoc % :kind :resident) resident))]
    (if-let [{:keys [sym view bytes data] :as r} (first regs)]
      (let [nbytes (long (or bytes (* (count data) (dtype-bytes view))))]
        (recur (assoc acc sym {:off off :view view :bytes nbytes :data data :kind (:kind r)})
               (align8 (+ off nbytes))
               (rest regs)))
      acc)))

(defn- elem
  "Element index (for a typed-array view) of region `sym`."
  [layout sym]
  (let [{:keys [off view]} (layout sym)] (quot off (dtype-bytes view))))

(defn- byte-off [layout sym] (:off (layout sym)))

(defn- resolve-call-arg
  "A :call entry → the value passed to the wasm export (byte offsets for memory regions)."
  [layout c]
  (cond
    (symbol? c) c
    (vector? c) (case (first c)
                  :lit      (second c)
                  :const    (byte-off layout (second c))
                  :resident (byte-off layout (second c))
                  (:in :inout) (byte-off layout (nth c 1)))
    :else c))

(defn- view-of [layout sym] (dtype-view (:view (layout sym))))

(defn- array-wrapper-form
  "Generate a cljs fn that marshals array args through scratch and calls the export."
  [export fname args call layout]
  (let [writes    (for [c call :when (and (vector? c) (#{:in :inout} (first c)))
                        :let [[_ sym arg] c]]
                    (list '.set (list (view-of layout sym) 'm) arg (elem layout sym)))
        eargs     (map #(resolve-call-arg layout %) call)
        readbacks (for [c call :when (and (vector? c) (= :inout (first c)))
                        :let [[_ sym arg] c e (elem layout sym)]]
                    (list '.set arg (list '.subarray (list (view-of layout sym) 'm)
                                          e (list '+ e (list 'alength arg)))))]
    (list 'defn (symbol fname) (vec args)
          (concat (list 'let ['m '(deref M)])
                  writes
                  (list (concat (list 'let ['r (list* (list 'w/export 'm export) eargs)])
                                readbacks
                                (list 'r)))))))

(defn- init-form
  "init! that instantiates the module and writes the const tables into memory."
  [consts layout]
  (let [writes (for [{:keys [sym]} consts]
                 (list '.set (list (view-of layout sym) 's) (symbol (str sym)) (elem layout sym)))]
    (list 'defn 'init!
          (list '[src] (list 'init! 'src '(cljs.core/js-obj)))
          (list '[src env]
                (list '.then (list 'w/instantiate! 'src 'env)
                      (list 'fn '[s]
                            (concat (list 'do (list 'reset! 'M 's)) writes (list 's))))))))

(defn- const-form [{:keys [sym data]}]
  (list 'def (with-meta (symbol (str sym)) {:private true}) (tagged-literal 'js (vec data))))

(defn- upload-form
  [{:keys [fn args set] :as _u} layout]
  (list 'defn (symbol fn) (vec args)
        (concat (list 'let ['m '(deref M)])
                (for [[sym arg] set]
                  (list '.set (list (view-of layout sym) 'm) arg (elem layout sym))))))

;; ---------------------------------------------------------------------------

(defn- preamble [ns]
  (str ";; GENERATED by raster.compiler.cljs-emit — do not edit.\n"
       ";; cljs counterpart of the raster kernel ns " ns " (deftm/defvalue → wasm).\n"
       "(ns " ns "\n  (:require [raster.wasm :as w]))\n\n"
       "(defonce M (atom nil))\n"
       "(defn ready? [] (some? @M))\n"))

(defn- default-init-str []
  (str "(defn init!\n"
       "  ([src] (init! src (cljs.core/js-obj)))\n"
       "  ([src env] (.then (w/instantiate! src env) (fn [s] (reset! M s) s))))\n"
       "(defn- kfn [n] (w/export @M n))\n"))

(defn emit!
  "Compile `kernels` to one wasm module + a generated cljs ns (same name as the kernel
   source ns). opts:
     :kernels   [spec…]   value-mode or array-mode kernel specs
     :ns 'a.b   :out-dir … :wasm-name \"kernels\" :mem-pages 256
     :consts :scratch :resident :uploads   (array-mode regions; auto-laid-out)
   Returns {:wasm-file :cljs-file :exports :layout}."
  [{:keys [kernels ns out-dir wasm-name consts scratch resident uploads]
    :or {wasm-name "kernels"}}]
  (let [module      (pl/compile-wasm-module
                     (mapv (fn [{:keys [var export dtype wasm-simd?]}]
                             {:var var :name export :dtype (or dtype :double) :wasm-simd? wasm-simd?})
                           kernels))
        layout      (lay-out consts scratch resident)
        array-mode? (boolean (or consts scratch resident uploads
                                 (some :call kernels)))
        sig-by-name (into {} (map (juxt :name :sig) (:exports module)))
        ;; value-mode kernels (no :call) need sigs + records
        vkernels    (remove :call kernels)
        ksigs       (mapv (fn [{:keys [export] fname :fn}]
                            {:export export :fname fname :sig (kernel-sig (sig-by-name export))})
                          vkernels)
        value-tags  (->> ksigs
                         (mapcat (fn [{:keys [sig]}]
                                   (concat (keep #(when (= :value (:kind %)) (:tag %)) (:params sig))
                                           (when (= :value (:kind (:ret sig))) [(:tag (:ret sig))]))))
                         distinct vec)
        forms       (concat
                     (map record-form value-tags)
                     (when array-mode? (map const-form consts))
                     (when array-mode? [(init-form consts layout)])
                     (when array-mode? (map #(upload-form % layout) uploads))
                     (for [{:keys [export fname sig]} ksigs] (value-wrapper-form export fname sig))
                     (for [{:keys [export args call] fname :fn} kernels :when call]
                       (array-wrapper-form export fname args call layout)))
        src         (str (preamble ns)
                         (if array-mode? "" (default-init-str)) "\n"
                         (str/join "\n" (map pr-str forms)) "\n")
        wasm-file   (io/file out-dir (str wasm-name ".wasm"))
        cljs-file   (io/file out-dir (str (str/replace (str ns) #"[.-]" {"." "/" "-" "_"}) ".cljs"))]
    (io/make-parents wasm-file)
    (with-open [o (io/output-stream wasm-file)] (.write o ^bytes (:bytes module)))
    (io/make-parents cljs-file)
    (spit cljs-file src)
    {:wasm-file (.getPath wasm-file) :cljs-file (.getPath cljs-file)
     :exports (:exports module) :layout layout}))
