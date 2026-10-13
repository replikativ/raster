(ns raster.compiler.core.callee-identity
  "Source-namespace Var identity retained across compiler source rewrites.")

(defn fresh-resolved-callee
  "Resolve `callee` as a Var in the explicit source namespace and return its qualified symbol.
   Lexical names are deliberately unresolvable: global spelling is not evidence once shadowed."
  ([callee source-ns]
   (fresh-resolved-callee callee source-ns #{}))
  ([callee source-ns lexical-symbols]
   (when (and (symbol? callee) source-ns (not (contains? lexical-symbols callee)))
     (let [source-ns (the-ns source-ns)
           v (binding [*ns* source-ns]
               (try (ns-resolve source-ns callee) (catch Exception _ nil)))]
       (when (var? v)
         (symbol (str (ns-name (:ns (meta v)))) (str (:name (meta v)))))))))

(defn retain-source-callee-identity
  "Replace any stale retained assertion with freshly resolved source-namespace Var identity."
  ([callee source-ns]
   (retain-source-callee-identity callee source-ns #{}))
  ([callee source-ns lexical-symbols]
   (if (symbol? callee)
     (let [callee (vary-meta callee dissoc :raster.op/resolved-callee)]
       (if-let [resolved (fresh-resolved-callee callee source-ns lexical-symbols)]
         (vary-meta callee assoc :raster.op/resolved-callee resolved)
         callee))
     callee)))

(defn validated-source-callee
  "Return freshly resolved identity only when it agrees with retained walker evidence.

   Neither a name, retained metadata alone, nor fresh resolution alone is authority. The two
   independently obtained identities must agree, and a lexical binding always masks the Var."
  ([callee source-ns]
   (validated-source-callee callee source-ns #{}))
  ([callee source-ns lexical-symbols]
   (when (symbol? callee)
     (let [retained (:raster.op/resolved-callee (meta callee))
           fresh (try
                   (fresh-resolved-callee callee source-ns lexical-symbols)
                   (catch Exception _ nil))]
       (when (and (symbol? retained) (= retained fresh)) fresh)))))
