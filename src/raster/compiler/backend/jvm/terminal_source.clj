(ns raster.compiler.backend.jvm.terminal-source
  "Shared owned JVM source projection for terminal exports and scalar fragments.
   Closed regions become exact calls without macro/inline expansion.
   Carrier propagation here is physical: ports, literal carriers and resolved Method returns.
   This is backend source projection, not a new function/type registry or interpreter."
  (:require [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
            [raster.compiler.core.ordered-call :as ordered]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-arithmetic :as source-arithmetic]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.invocation-plan :as invocation])
  (:import [java.lang.reflect Method]))

(defn- fail! [message data]
  (throw (ex-info message (assoc data :reason :unclosed-terminal-source-projection))))

(defn- intrinsic-current! [v operation]
  (intrinsic-source/assert-original! v operation :terminal-intrinsic-definition-mismatch))

(defn- source-hints!
  [expression local-carrier source-ns]
  (let [metadata (meta expression)
        tag (:tag metadata)
        ;; Compiler.IfExpr/LetExpr/BodyExpr derive their carrier from their
        ;; children, not form metadata. Keep that metadata on the projection;
        ;; never use it to manufacture a primitive carrier for the region.
        control? (and (seq? expression) (contains? #{'if 'let* 'do} (first expression)))
        ;; Completion is reserved IR, not a source Var call. Its existing
        ;; projection owns the physical context and deliberately drops :tag;
        ;; a later logical annotation cannot override that contract.
        completion? (some? (completion/canonical-parts expression))
        object-tag? (and (= Object local-carrier)
                         (or (identical? Object tag)
                             (and (symbol? tag)
                                  (identical? Object (ns-resolve source-ns tag)))))
        primitive-tag? (and local-carrier (.isPrimitive ^Class local-carrier)
                            (= local-carrier (get types/tag->java-class tag)))]
    (when (or (contains? metadata :param-tags)
              (and (contains? metadata :tag) (not (or control? completion? object-tag? primitive-tag?))))
      (fail! "terminal JVM source hint needs its own carrier correspondence"
             {:expression expression :source-metadata metadata}))))

(defn- project*
  ([contract source-ns typed-context?]
   (project* contract source-ns typed-context? nil))
  ([contract source-ns typed-context? observe!]
  (let [policy (:source-policy contract)
        ;; Projection order is not runtime execution order (both conditional
        ;; arms are visited). Tokens retain occurrence/scoping without guessing
        ;; correspondence from equal expressions or generated symbol names.
        active (when observe! (volatile! []))
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
                        (entry/observe-definition-input! v source-ns symbol)
                        (when observe!
                          (observe! {:kind :resolution :occurrence (:occurrence (peek @active))
                                     :symbol symbol :var v}))
                        v))]
    (letfn [(resolved [^Method method receiver args]
              (let [plan (ordered/plan-resolved-call method receiver args policy)
                    normalized (ordered/normalize-resolved-call plan fresh @used)
                    carrier (.getReturnType method)
                    result (cond-> {:source (list 'let* (:bindings normalized) (:invocation normalized))
                                    :carrier carrier}
                             (.isPrimitive carrier)
                             (assoc :boxed-carrier (get types/primitive-info
                                                        (symbol (.getName carrier))) :non-null? true))]
                (when observe!
                  (observe! {:kind :resolved-call :occurrence (:occurrence (peek @active))
                             :phase (if (seq @active) :body :parameter-boundary)
                             :plan plan :normalization normalized
                             :projection (:source result) :result-carrier carrier}))
                (vswap! methods conj method)
                (doseq [{:keys [class port]} (:class-ports normalized)]
                  (when-not (contains? @references class)
                    (vswap! captures conj {:port port :value class})
                    (vswap! references assoc class port)))
                result))
            (static-call [class method carriers args]
              (resolved (.getMethod ^Class class method (into-array Class carriers)) nil args))
            (go* [expression env]
              (when (seq? expression) (source-hints! (first expression) nil source-ns))
              (cond
                (and (:invocation contract)
                     (identical? expression (get contract :invocation-expression (:body contract))))
                (let [bridge (:invocation contract)
                      arguments (mapv #(go % env) (descriptor/call-args expression))]
                  (when-not (and (vector? (:invocation-carriers contract))
                                 (class? (:invocation-payload-class contract))
                                 (= (:invocation-carriers contract) (mapv :carrier arguments)))
                    (fail! "original invocation requires its completed primitive operands" {}))
                  (assoc (resolved (:invocation-method bridge)
                                   (capture! (:implementation bridge)) (mapv :source arguments))
                         ;; This is the actual Object invocation, not its static
                         ;; compute method. The original owner separately proves
                         ;; the conditional normal-return boxing correspondence.
                         :boxed-carrier (:invocation-payload-class contract) :non-null? true))
                (symbol? expression)
                (or (when-let [value (get env expression)]
                      (update value :source with-meta (meta expression)))
                    (let [v (resolve-var expression)]
                      (when (:macro (meta v)) (fail! "terminal macro value is unsupported" {:symbol expression}))
                      {:source (capture! @v) :carrier Object}))
                (or (nil? expression) (string? expression) (keyword? expression))
                {:source expression :carrier Object}
                (instance? Double expression) {:source expression :carrier Double/TYPE :boxed-carrier Double :non-null? true}
                (instance? Long expression) {:source expression :carrier Long/TYPE :boxed-carrier Long :non-null? true}
                (boolean? expression) {:source expression :carrier Boolean/TYPE :boxed-carrier Boolean :non-null? true}
                (vector? expression) {:source (mapv #(-> (go % env) :source) expression) :carrier Object}
                (completion/canonical-parts expression)
                (let [{:keys [contract operand]} (completion/canonical-parts expression)]
                  (when-not (= completion/object-input-contract contract)
                    (fail! "terminal completion needs a separate owned projection" {}))
                  (let [value (go operand env)]
                    (assoc value :source (completion/complete-value contract (:source value)) :carrier Object)))
                (seq? expression)
                (let [[head & args] expression]
                  (cond
                    (= 'quote head) {:source expression :carrier Object}
                    (= 'if head)
                    (let [[test then else] (mapv #(go % env) args)]
                      (when-not (= 3 (count args)) (fail! "terminal conditional arity" {}))
                      (when (and typed-context? (not= (:carrier then) (:carrier else)))
                        (fail! "typed scalar join needs its original physical merge context" {}))
                      (when (and (not= (:carrier then) (:carrier else))
                                 (.isPrimitive ^Class (:carrier then))
                                 (.isPrimitive ^Class (:carrier else)))
                        (fail! "mixed primitive join needs its actual source carrier" {}))
                      (cond-> {:source (with-meta (list 'if (:source test) (:source then) (:source else))
                                         (meta expression))
                               :carrier (if (= (:carrier then) (:carrier else)) (:carrier then) Object)}
                        (and (:boxed-carrier then) (= (:boxed-carrier then) (:boxed-carrier else))
                             (:non-null? then) (:non-null? else))
                        (assoc :boxed-carrier (:boxed-carrier then) :non-null? true)))
                    (= 'let* head)
                    (let [[bindings & body] args]
                      (when-not (and (vector? bindings) (even? (count bindings)) (= 1 (count body)))
                        (fail! "terminal binding scope is not closed" {}))
                      (let [{:keys [bindings env]}
                            (reduce (fn [{:keys [bindings env]} [id init]]
                                      (when-not (symbol? id) (fail! "terminal binder is not a symbol" {}))
                                      (let [value (go init env)]
                                        (source-hints! id (:carrier value) source-ns)
                                        ;; Raster permits an exact primitive declaration on
                                        ;; a primitive initializer; Clojure let* rejects the
                                        ;; redundant JVM hint. Its equality was proved above.
                                        (let [id (if (.isPrimitive ^Class (:carrier value))
                                                   (vary-meta id dissoc :tag) id)]
                                          {:bindings (conj bindings id (:source value))
                                           :env (assoc env id (assoc value :source id))})))
                                    {:bindings [] :env env} (partition 2 bindings))
                            result (go (first body) env)]
                        (assoc result :source (with-meta (list 'let* bindings (:source result))
                                                        (meta expression)))))
                    (= 'do head)
                    (let [body (mapv #(go % env) args)]
                      (assoc (or (last body) {:carrier Object})
                             :source (with-meta (list* 'do (map :source body)) (meta expression))))
                    :else
                    (let [v (when (and (symbol? head) (not (contains? env head))) (resolve-var head))
                          metadata (meta v)
                          operation (when v (symbol (str (ns-name (:ns metadata))) (str (:name metadata))))
                          arguments (mapv #(go % env) args)]
                      (cond
                        (:macro metadata)
                        (fail! "terminal macro has no owned projection" {:head head})
                        (and (contains? (if typed-context? #{'int 'long 'float 'double} #{'float 'double})
                                        (descriptor/cast-result-tag operation))
                             (= 1 (count arguments)))
                        (let [{:keys [source carrier]} (first arguments)
                              target (descriptor/cast-result-tag operation)
                              _ (intrinsic-current! v operation)
                              _ (when (and typed-context? (= 'int target)
                                           (not= :reject (descriptor/cast-integral-narrowing operation)))
                                  (fail! "typed wrapping Int cast needs a separate exact projection"
                                         {:head head :operation operation}))
                              _ (when (and typed-context? (contains? #{'int 'long} target)
                                           (not (contains? #{Integer/TYPE Long/TYPE} carrier)))
                                  (fail! "typed integral cast lacks its original bytecode completion context"
                                         {:head head :carrier carrier}))
                              _ (when (and typed-context? (= 'double target) (= Boolean/TYPE carrier))
                                  (fail! "typed Boolean-to-Double conversion differs from core source"
                                         {:head head :carrier carrier}))
                              input (if (contains? #{Double/TYPE Float/TYPE Long/TYPE Integer/TYPE} carrier)
                                      carrier Object)
                              ieee? (or (= :ieee (descriptor/cast-floating-narrowing operation))
                                        (not (false? (:unchecked-math policy))))]
                          (if (and typed-context? (= 'int target) (= Long/TYPE carrier))
                            ;; Match emit-intrinsic's explicit checked Int(Long)
                            ;; conversion, including its ArithmeticException.
                            ;; Paired actual Clojure/typed oracles qualify the
                            ;; supported dependency's primitive Long overload.
                            (static-call Math "toIntExact" [Long/TYPE] [source])
                            (static-call clojure.lang.RT
                                       (case target
                                         int "intCast"
                                         long "longCast"
                                         double "doubleCast"
                                         float (if ieee? "uncheckedFloatCast" "floatCast"))
                                       [input] [source])))
                        (and (= 'clojure.core/nil? operation) (= 1 (count arguments)))
                        (do (intrinsic-current! v operation)
                            (static-call java.util.Objects "isNull" [Object] [(:source (first arguments))]))
                        (and operation (= "clojure.core" (namespace operation))
                             (source-arithmetic/binary-operator? (name operation)))
                        (let [floating-source? (and (not typed-context?)
                                                    (every? #(contains? #{Float/TYPE Double/TYPE}
                                                                        (:carrier %)) arguments))]
                          (intrinsic-current! v operation)
                          ;; This is physical carrier selection, not a result-tag
                          ;; guess. Other carriers/arity need their own exact source
                          ;; overload/completion projection before admission.
                          (when-not (and (= 2 (count arguments))
                                         (or (every? #(= Double/TYPE (:carrier %)) arguments)
                                             floating-source?
                                             (and typed-context? (false? (:unchecked-math policy))
                                                  (not= "/" (name operation))
                                                  (every? #(= Long/TYPE (:carrier %)) arguments))))
                            (fail! "terminal arithmetic needs its exact primitive source carriers"
                                   {:head head :carriers (mapv :carrier arguments)}))
                          ;; Ordinary Clojure core inlines select Numbers' Double
                          ;; overload for Float operands. Preserve the exported
                          ;; Float checkpoint, then widen exactly at this call.
                          ;; Typed scalar Float arithmetic is a different source
                          ;; context and remains outside this admission rule.
                          (let [arguments (if floating-source?
                                            (mapv #(if (= Float/TYPE (:carrier %))
                                                     (static-call clojure.lang.RT "doubleCast"
                                                                  [Float/TYPE] [(:source %)]) %)
                                                  arguments)
                                            arguments)]
                            (static-call clojure.lang.Numbers
                                         (source-arithmetic/numbers-method-name (name operation))
                                         (mapv :carrier arguments) (mapv :source arguments))))
                        (or (:macro metadata) (:inline metadata) (nil? v))
                        (fail! "terminal call has no owned non-expanding projection" {:head head})
                        typed-context?
                        (fail! "typed scalar helper lacks its original invocation context" {:head head})
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
                :else (fail! "terminal literal has no owned source projection" {:class (class expression)})))
            (checked-go [expression env]
              (let [result (go* expression env)]
                ;; Validate hints against the actual projected Method/local
                ;; carrier, never use the hint to select that carrier.
                (if (and (:invocation-expression contract)
                         (identical? expression (:invocation-expression contract)))
                  (let [tag (:tag (meta expression))
                        payload (get-in contract [:invocation :payload-carrier])]
                    ;; The original owned bytecode observes this exact call as
                    ;; Object despite its retained logical payload annotation.
                    ;; Keep the actual Method result Object; no return cast or
                    ;; materialization completion is being erased here.
                    (when (some (fn [hint]
                                  (and hint (not= payload (get types/tag->java-class hint))))
                                [tag (:raster.type/tag (meta expression))])
                      (fail! "terminal invocation hint differs from its observed payload" {}))
                    (source-hints! (vary-meta expression dissoc :tag) (:carrier result) source-ns))
                  (source-hints! expression (:carrier result) source-ns))
                result))
            (go [expression env]
              (if-not observe!
                ;; Existing projections do not allocate occurrence frames or
                ;; enter a tracing try/finally at every scalar node.
                (checked-go expression env)
                (let [token (Object.)
                      parent (peek @active)
                      child-index (when parent
                                    (let [counter (:children parent) index @counter]
                                      (vswap! counter inc) index))]
                  (observe! {:kind :enter :occurrence token :parent (:occurrence parent)
                             :child-index child-index :expression expression})
                  (vswap! active conj {:occurrence token :children (volatile! 0)})
                  (try
                    (let [result (checked-go expression env)]
                      (observe! {:kind :value :occurrence token :expression expression
                                 :projection (:source result)
                                 :result-facts (select-keys result [:carrier :boxed-carrier :non-null?])})
                      result)
                    (finally (vswap! active pop))))))]
      (let [input (fresh "terminal_inputs__")
            projected
            (mapv (fn [index {:keys [symbol carrier]}]
                    (let [read (static-call clojure.lang.RT "nth" [Object Integer/TYPE] [input (int index)])
                          value (cond
                                  (= Object carrier) read
                                  (= Long/TYPE carrier)
                                  (static-call clojure.lang.RT "longCast" [Object] [(:source read)])
                                  (= Integer/TYPE carrier)
                                  (static-call clojure.lang.RT "intCast" [Object] [(:source read)])
                                  (= Float/TYPE carrier)
                                  ;; The owner checks an exact Float wrapper.
                                  ;; Extract its value without adding a checked
                                  ;; narrowing point (which could reject infinity).
                                  (static-call clojure.lang.RT "uncheckedFloatCast" [Object] [(:source read)])
                                  (= Double/TYPE carrier)
                                  (static-call clojure.lang.RT "doubleCast" [Object] [(:source read)])
                                  :else (fail! "source port has no audited physical projection"
                                               {:symbol symbol :carrier carrier}))]
                      [symbol value])) (range) (:parameters contract))
            env (into {} (map (fn [[id value]] [id (assoc value :source (with-meta id nil))])) projected)
            result (go (:body contract) env)]
        {:params [input]
         :source (list 'let* (vec (mapcat (fn [[id value]] [(with-meta id nil) (:source value)]) projected))
                       (:source result))
         :result-facts (select-keys result [:carrier :boxed-carrier :non-null?])
         :captures @captures :methods @methods :reference-bindings @references})))))

(defn project
  "Project the audited terminal export contract under its original source owner.
   Direct helper calls retain an unhinted Object ABI; private captures are not portable IR."
  [contract source-ns]
  (let [parameters (mapv (fn [{:keys [symbol dtype]}]
                           {:symbol symbol :carrier (case dtype :float Float/TYPE
                                                               :double Double/TYPE :long Long/TYPE)})
                         (invocation/terminal-parameters contract))]
    (project* (assoc contract :parameters parameters) source-ns false)))

(defn project-terminal-invocation
  "Project an owner-authenticated nested terminal invocation using its actual Method.
   The owner must first join the exact source interval and terminal contract.
   This checks physical export/argument/return compatibility, not producer/read
   semantics. It does not add helper resolution to ordinary terminal projection."
  [contract source-ns expression observation]
  (let [bridge (get-in observation [:consumer :computation :bridge])
        call (get-in observation [:consumer :invocation])
        method (:invocation-method bridge)
        producer-id (get-in observation [:producer :binding :binder])
        producer-fact (get-in observation [:producer :completed-value])
        producer-class (or (:builtin-class producer-fact) (:carrier producer-fact))
        dtype (cond (contains? #{Float Float/TYPE} producer-class) :float
                    (contains? #{Double Double/TYPE} producer-class) :double)
        body (:body contract)
        binder (get-in observation [:consumer :binding :binder])
        parameters (invocation/terminal-parameters contract)]
    (when-not (and (= :boxed-primitive-static-return (:kind bridge))
                   (= Object (:result-carrier bridge))
                   (contains? #{Float/TYPE Double/TYPE} (:payload-carrier bridge))
                   (instance? Method method) (= method (:method call))
                   (= Object (.getReturnType ^Method method))
                   (= :ref (get-in observation [:consumer :binding :after :type]))
                   (= Object (get-in observation [:consumer :binding :after :reference-class]))
                   (seq? body) (= 'let* (first body)) (= 3 (count body))
                   (vector? (second body)) (= 2 (count (second body)))
                   (= binder (first (second body)) (nth body 2))
                   (identical? expression (second (second body)))
                   (every? (fn [symbol]
                             (and (not (contains? (meta symbol) :param-tags))
                                  (every? #(or (nil? %)
                                               (= (:payload-carrier bridge) (get types/tag->java-class %)))
                                          [(:tag (meta symbol)) (:raster.type/tag (meta symbol))])))
                           [(first (second body)) (nth body 2)])
                   (.isInstance (.getDeclaringClass ^Method method) (:implementation bridge))
                   (= (:parameter-carriers bridge) (vec (.getParameterTypes ^Method method)))
                   (every? #{Float/TYPE Double/TYPE} (:parameter-carriers bridge))
                   (= 1 (count parameters)) dtype
                   (= producer-id (:symbol (first parameters)))
                   (= dtype (:dtype (first parameters)))
                   (= (:parameter-carriers bridge) (mapv :parameter-carrier (:input-completions call))))
      (fail! "terminal invocation lacks exact original FP export/call correspondence" {}))
    (project*
     ;; The only terminal local is the observed Object result, immediately
     ;; returned unchanged. Elide that alias, not a source call/completion.
     ;; Keeping its logical primitive tag on a Clojure let would introduce an
     ;; unboxing absent from the original emitted Object local.
     (assoc contract :body expression
            :parameters [{:symbol producer-id :carrier (if (= :float dtype) Float/TYPE Double/TYPE)}]
            :invocation bridge :invocation-expression expression
            :invocation-carriers (:parameter-carriers bridge)
            :invocation-payload-class (if (= Float/TYPE (:payload-carrier bridge)) Float Double))
     source-ns false)))

(defn- scalar-contract [expression parameters source-policy]
  (when-not (= {:unchecked-math false} source-policy)
    (fail! "typed scalar source policy has no audited projection" {:source-policy source-policy}))
  (when-not (and (vector? parameters)
                 (every? #(and (symbol? (:symbol %)) (class? (:carrier %))) parameters)
                 (= (count parameters) (count (distinct (map :symbol parameters)))))
    (fail! "scalar source requires independently owned physical ports" {}))
  {:body expression :source-policy source-policy :parameters parameters})

(defn project-scalar
  "Project one closed scalar fragment from actual physical body-entry facts.
   Parameters carry :symbol and actual :carrier Class, never a logical dtype.
   Only checked Long arithmetic, explicit Int/Long conversions over primitive
   integral values, and the existing Double arithmetic/casts have
   source/typed-bytecode correspondence here. Long division, unknown helpers,
   unproved input conversions and buffer-shaped ports refuse. This is source
   projection, not proof that the surrounding typed region's coercions agree."
  [expression parameters source-ns source-policy]
  (project* (scalar-contract expression parameters source-policy) source-ns true))

(defn trace-scalar
  "Construct the existing scalar projection plus its exact constructor trace.
   :typed-scalar uses project-scalar's unchanged restrictions; :clojure-source
   uses the existing ordinary-source Float checkpoint/Numbers widening model.
   The caller must authenticate WHICH context matches its actual emitted body.
   Parameters remain actual carrier requirements supplied by that caller, not
   inferred from dtype tags. No numerical body or resolved Method is invoked.

   This is pending correspondence data, NOT an equivalence/totality/effect proof,
   frontend admission or portable IR. Method input contracts, private captures,
   reference checks and all runtime obligations remain. Enter events are preorder,
   value events postorder, calls/resolution interior, under explicit parent/child
   positions. Parameter-boundary calls are not body operations. This is not
   dynamic execution order. The owner must retain this exact result and its
   enclosing producer history."
  [expression parameters source-ns source-policy context]
  (when-not (and entry/*emission-inputs*
                 (contains? #{:typed-scalar :clojure-source} context)
                 (instance? clojure.lang.Namespace source-ns)
                 (identical? source-ns (find-ns (ns-name source-ns))))
    (fail! "scalar correspondence needs its actual owner, namespace and source context" {}))
  (let [events (volatile! [])
        projection (project* (scalar-contract expression parameters source-policy)
                             source-ns (= context :typed-scalar) #(vswap! events conj %))
        observations @entry/*emission-inputs*]
    (entry/assert-definition-inputs-current! observations)
    (when-not (identical? source-ns (find-ns (ns-name source-ns)))
      (fail! "scalar correspondence namespace changed during projection" {}))
    (assoc projection :correspondence
           {:context context :expression expression :parameters parameters
            :source-ns source-ns :source-policy source-policy
            :events @events :observations observations :admitted? false})))
