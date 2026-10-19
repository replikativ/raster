(ns raster.compiler.backend.jvm.terminal-source
  "Project a closed terminal region to owned JVM calls, without macro/inline expansion.
   Carrier propagation here is physical: ports, literal carriers and resolved Method returns.
   This is backend source projection, not a new function/type registry or interpreter."
  (:require [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.ordered-call :as ordered]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.util :as util])
  (:import [java.lang.reflect Method]))

(defn- fail! [message data]
  (throw (ex-info message (assoc data :reason :unclosed-terminal-source-projection))))

(defonce ^:private intrinsic-source-observations
  ;; Bootstrap references for this backend's intrinsic translations, NOT a
  ;; dispatch/type registry. Conversion semantics remain in op-descriptor.
  ;; Trust premise: standard Clojure definitions at compiler bootstrap. Later
  ;; core redefinition/inline replacement (including namespace reload) must
  ;; refuse; a fresh consumer observation cannot authenticate the replacement.
  (into {}
        (map (fn [operation]
               [operation (entry/emission-definition-snapshot (resolve operation))]))
        (conj (into #{} (filter #(and (= "clojure.core" (namespace %))
                                      (= 'float (descriptor/cast-result-tag %))))
                    (keys descriptor/cast-ops))
              'clojure.core/nil?)))

(defn- intrinsic-current! [v operation]
  (let [original (get intrinsic-source-observations operation)]
    (when-not (and original (identical? v (:var original))
                   (entry/var-value-current? original))
      (throw (ex-info "terminal intrinsic no longer matches its compiler bootstrap definition"
                      {:reason :terminal-intrinsic-definition-mismatch :operation operation})))))

(defn- source-hints!
  [expression object-local? source-ns]
  (let [metadata (meta expression)
        tag (:tag metadata)
        ;; Compiler.IfExpr/LetExpr/BodyExpr derive their carrier from their
        ;; children, not form metadata. Keep that metadata on the projection;
        ;; never use it to manufacture a primitive carrier for the region.
        control? (and (seq? expression) (contains? #{'if 'let* 'do} (first expression)))
        object-tag? (and object-local?
                         (or (identical? Object tag)
                             (and (symbol? tag)
                                  (identical? Object (ns-resolve source-ns tag)))))]
    (when (or (contains? metadata :param-tags)
              (and (contains? metadata :tag) (not (or control? object-tag?))))
      (fail! "terminal JVM source hint needs its own carrier correspondence"
             {:expression expression :source-metadata metadata}))))

(defn project
  "Project under the actual source producer's namespace/policy and definition collector.
   Unsupported macro/inline calls and original class/member syntax refuse. Direct helper calls
   require an unhinted fixed Object ABI; their actual IFn values become private captures.
   Source-to-semantic-program correspondence remains with the enclosing producer."
  [contract source-ns]
  (let [policy (:source-policy contract)
        used (volatile! #{})
        _ (util/postwalk-preserving-meta
           (fn [x] (when (symbol? x) (vswap! used conj x)) x) (:body contract))
        fresh (fn [prefix]
                (loop [x (gensym prefix)]
                  (if (contains? @used x) (recur (gensym prefix))
                      (do (vswap! used conj x) x))))
        captures (volatile! []) methods (volatile! []) references (volatile! {})
        capture! (fn [value]
                   (let [port (fresh "terminal_capture__")]
                     (vswap! captures conj {:port port :value value}) port))
        resolve-var (fn [symbol]
                      (let [v (try (ns-resolve source-ns symbol) (catch IllegalArgumentException _ nil))]
                        (when-not (var? v) (fail! "terminal symbol has no owned Var" {:symbol symbol}))
                        (entry/observe-definition-input! v source-ns symbol) v))]
    (letfn [(resolved [^Method method receiver args]
              (let [normalized (ordered/normalize-resolved-call
                                (ordered/plan-resolved-call method receiver args policy)
                                fresh @used)]
                (vswap! methods conj method)
                (doseq [{:keys [class port]} (:class-ports normalized)]
                  (when-not (contains? @references class)
                    (vswap! captures conj {:port port :value class})
                    (vswap! references assoc class port)))
                {:source (list 'let* (:bindings normalized) (:invocation normalized))
                 :carrier (.getReturnType method)}))
            (static-call [class method carriers args]
              (resolved (.getMethod ^Class class method (into-array Class carriers)) nil args))
            (go [expression env]
              (source-hints! expression
                             (and (symbol? expression) (= Object (:carrier (get env expression))))
                             source-ns)
              (when (seq? expression) (source-hints! (first expression) false source-ns))
              (cond
                (symbol? expression)
                (or (when-let [value (get env expression)]
                      (update value :source with-meta (meta expression)))
                    (let [v (resolve-var expression)]
                      (when (:macro (meta v)) (fail! "terminal macro value is unsupported" {:symbol expression}))
                      {:source (capture! @v) :carrier Object}))
                (or (nil? expression) (string? expression) (keyword? expression))
                {:source expression :carrier Object}
                (instance? Double expression) {:source expression :carrier Double/TYPE}
                (instance? Long expression) {:source expression :carrier Long/TYPE}
                (boolean? expression) {:source expression :carrier Boolean/TYPE}
                (vector? expression) {:source (mapv #(-> (go % env) :source) expression) :carrier Object}
                (completion/canonical-parts expression)
                (let [{:keys [contract operand]} (completion/canonical-parts expression)]
                  (when-not (= completion/object-input-contract contract)
                    (fail! "terminal completion needs a separate owned projection" {}))
                  {:source (completion/complete-value contract (:source (go operand env))) :carrier Object})
                (seq? expression)
                (let [[head & args] expression]
                  (cond
                    (= 'quote head) {:source expression :carrier Object}
                    (= 'if head)
                    (let [[test then else] (mapv #(go % env) args)]
                      (when-not (= 3 (count args)) (fail! "terminal conditional arity" {}))
                      (when (and (not= (:carrier then) (:carrier else))
                                 (.isPrimitive ^Class (:carrier then))
                                 (.isPrimitive ^Class (:carrier else)))
                        (fail! "mixed primitive join needs its actual source carrier" {}))
                      {:source (with-meta (list 'if (:source test) (:source then) (:source else))
                                         (meta expression))
                       :carrier (if (= (:carrier then) (:carrier else)) (:carrier then) Object)})
                    (= 'let* head)
                    (let [[bindings & body] args]
                      (when-not (and (vector? bindings) (even? (count bindings)) (= 1 (count body)))
                        (fail! "terminal binding scope is not closed" {}))
                      (let [{:keys [bindings env]}
                            (reduce (fn [{:keys [bindings env]} [id init]]
                                      (when-not (symbol? id) (fail! "terminal binder is not a symbol" {}))
                                      (let [value (go init env)]
                                        (source-hints! id (= Object (:carrier value)) source-ns)
                                        {:bindings (conj bindings id (:source value))
                                         :env (assoc env id {:source id :carrier (:carrier value)})}))
                                    {:bindings [] :env env} (partition 2 bindings))
                            result (go (first body) env)]
                        (assoc result :source (with-meta (list 'let* bindings (:source result))
                                                        (meta expression)))))
                    (= 'do head)
                    (let [body (mapv #(go % env) args)]
                      {:source (with-meta (list* 'do (map :source body)) (meta expression))
                       :carrier (or (:carrier (last body)) Object)})
                    :else
                    (let [v (when (and (symbol? head) (not (contains? env head))) (resolve-var head))
                          metadata (meta v)
                          operation (when v (symbol (str (ns-name (:ns metadata))) (str (:name metadata))))
                          arguments (mapv #(go % env) args)]
                      (cond
                        (:macro metadata)
                        (fail! "terminal macro has no owned projection" {:head head})
                        (and (= 'float (descriptor/cast-result-tag operation)) (= 1 (count arguments)))
                        (let [{:keys [source carrier]} (first arguments)
                              _ (intrinsic-current! v operation)
                              input (if (contains? #{Double/TYPE Float/TYPE Long/TYPE Integer/TYPE} carrier)
                                      carrier Object)
                              ieee? (or (= :ieee (descriptor/cast-floating-narrowing operation))
                                        (not (false? (:unchecked-math policy))))]
                          (static-call clojure.lang.RT (if ieee? "uncheckedFloatCast" "floatCast") [input] [source]))
                        (and (= 'clojure.core/nil? operation) (= 1 (count arguments)))
                        (do (intrinsic-current! v operation)
                            (static-call java.util.Objects "isNull" [Object] [(:source (first arguments))]))
                        (or (:macro metadata) (:inline metadata) (nil? v))
                        (fail! "terminal call has no owned non-expanding projection" {:head head})
                        :else
                        (let [arity (count arguments)
                              signature (some (fn [signature]
                                                (when (and (vector? signature) (= arity (count signature))
                                                           (every? symbol? signature) (not-any? #{'&} signature)
                                                           (every? #(nil? (:tag (meta %))) signature)) signature))
                                              (:arglists metadata))]
                          (when-not (and signature (nil? (:tag metadata)))
                            (fail! "terminal helper needs its actual Object invocation ABI" {:head head}))
                          (resolved (.getMethod clojure.lang.IFn "invoke" (into-array Class (repeat arity Object)))
                                    (capture! @v) (mapv :source arguments)))))))
                :else (fail! "terminal literal has no owned source projection" {:class (class expression)})))]
      (let [input (fresh "terminal_inputs__")
            projected
            (mapv (fn [index {:keys [symbol dtype]}]
                    (let [read (static-call clojure.lang.RT "nth" [Object Integer/TYPE] [input (int index)])
                          value (static-call clojure.lang.RT (case dtype :double "doubleCast" :long "longCast")
                                             [Object] [(:source read)])]
                      [symbol value])) (range) (:parameters contract))
            env (into {} (map (fn [[id value]] [id {:source (with-meta id nil) :carrier (:carrier value)}])) projected)
            result (go (:body contract) env)]
        {:params [input]
         :source (list 'let* (vec (mapcat (fn [[id value]] [(with-meta id nil) (:source value)]) projected))
                       (:source result))
         :captures @captures :methods @methods :reference-bindings @references}))))
