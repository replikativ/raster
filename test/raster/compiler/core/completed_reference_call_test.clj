(ns raster.compiler.core.completed-reference-call-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(defonce events (atom []))

(core/deftm delegated-reference [xs :- (Array float) count :- Long] :- (Array float)
  (swap! events conj :delegate)
  (if (< count 0) nil xs))

(defn- receiver [source]
  (ns-resolve (.ns ^clojure.lang.Var source) (symbol (str (:name (meta source)) "-impl"))))

(defn- prepare! [source]
  (core/prepare-deftm-frontend! source)
  (core/ensure-current-owned-bytecode! (receiver source)))

(defn- with-call
  ([helper-body argument run] (with-call helper-body argument identity run))
  ([helper-body argument wrap run]
  ;; Fresh modules avoid renewing an old collector after the real JIT promotion.
  (let [scope (create-ns (gensym "raster.completed-call."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval (list 'raster.core/deftm 'leaf
                    '[xs :- (Array float) unused :- Long] ':- '(Array float) helper-body))
        (eval (list 'raster.core/deftm 'caller
                    '[xs :- (Array float) count :- Long] ':- '(Array float)
                    (wrap (list 'leaf 'xs argument)))))
      (let [leaf (core/resolve-deftm-var (ns-resolve scope 'leaf) {:dtype :float})
            caller (core/resolve-deftm-var (ns-resolve scope 'caller) {:dtype :float})]
        (prepare! leaf)
        (prepare! caller)
        (entry/call-with-definition-inputs
         #(let [frontend (entry/original-source-frontend caller)
                target (symbol (str (ns-name scope)) (str (:name (meta leaf)) "-impl"))
                original (first (filter (fn [node]
                                          (and (seq? node)
                                               (= target (:implementation-op (descriptor/call-description node)))))
                                        (tree-seq coll? seq (:body frontend))))]
            (run {:scope scope :leaf leaf :caller caller :frontend frontend :original original}))))
      (finally (remove-ns (ns-name scope)))))))

(defn- compiled [frontend expression]
  (let [source (:source-entry frontend)
        params (mapv #(with-meta %1 {:tag %2}) (:params source) (:tags source))
        {:keys [methods]}
        (bytecode/compile-specialized-class!
         "raster.compiled.CompletedReferenceCall"
         [{:name 'run :params params :return-tag 'Object
           :walked-body [expression] :source-ns (:source-ns source)}])
        method (get methods 'run)]
    (fn [& args]
      (try (.invoke ^java.lang.reflect.Method method nil (object-array args))
           (catch java.lang.reflect.InvocationTargetException error (throw (.getCause error)))))))

(deftest whole-call-retains-reference-identity-and-unused-positional-completion
  (with-call 'xs 'count
    (fn [{:keys [frontend original]}]
      (let [constructed (with-redefs [core/ensure-bytecode-implementation!
                                     (fn [& _] (throw (AssertionError. "constructor forced JIT")))]
                          (transport/construct-original-reference-call frontend original))
            view (transport/completed-reference-call-view constructed)
            expression (:expression view)
            bindings (second expression)
            original-fn (compiled frontend original)
            imported-fn (compiled frontend expression)]
        (is (some? constructed))
        (is (identical? original (:original view)))
        (is (= 6 (count bindings)) "receiver and BOTH arguments, including unused Long")
        (is (= [:jvm-bytecode-reference-input :jvm-bytecode-reference-input :jvm-bytecode-integral-input]
               (mapv #(get-in (completion/canonical-parts %) [:contract :source-context])
                     (take-nth 2 (rest bindings)))))
        (doseq [value [nil (float-array []) (float-array [1.0 3.0])]
                count [0 17]]
          (is (identical? value (original-fn value count)))
          (is (identical? value (imported-fn value count))))
        (let [before (count (:producer-inputs @entry/*emission-inputs*))]
          (transport/completed-reference-call-view constructed)
          (is (= before (count (:producer-inputs @entry/*emission-inputs*)))))))))

(deftest unused-argument-effects-and-failure-precede-the-whole-helper-body
  (with-call
    '(do (clojure.core/swap! raster.compiler.core.completed-reference-call-test/events
                             clojure.core/conj :body) xs)
    '(do (clojure.core/swap! raster.compiler.core.completed-reference-call-test/events
                             clojure.core/conj :argument)
         (clojure.core/quot 1 count))
    (fn [{:keys [frontend original]}]
      (reset! events [])
      (let [constructed (transport/construct-original-reference-call frontend original)
            expression (:expression (transport/completed-reference-call-view constructed))]
        (is (some? constructed))
        (is (empty? @events) "construction does not run either operand or helper")
        (doseq [f [(compiled frontend original) (compiled frontend expression)]]
          (reset! events [])
          (let [value (float-array [2.0])]
            (is (identical? value (f value 1)))
            (is (= [:argument :body] @events)))
          (reset! events [])
          (is (thrown? ArithmeticException (f nil 0)))
          (is (= [:argument] @events) "unused throwing argument cannot be erased"))))))

(deftest whole-body-retains-delegation-effects-and-nullable-results
  (prepare! (core/resolve-deftm-var #'delegated-reference {:dtype :float}))
  (with-call
    '(raster.compiler.core.completed-reference-call-test/delegated-reference xs unused)
    '(do (clojure.core/swap! raster.compiler.core.completed-reference-call-test/events
                             clojure.core/conj :argument) count)
    (fn [{:keys [frontend original]}]
      (reset! events [])
      (let [constructed (transport/construct-original-reference-call frontend original)]
        (is (some? constructed))
        (is (empty? @events))
        (let [expression (:expression (transport/completed-reference-call-view constructed))]
          (doseq [f [(compiled frontend original) (compiled frontend expression)]
                  count [-1 0 9]]
            (reset! events [])
            (let [value (float-array [1.0 4.0])]
              (is (identical? (when-not (neg? count) value) (f value count))))
            (is (= [:argument :delegate] @events))))))))

(deftest completed-root-normalization-retains-the-original-call-boundary
  (prepare! (core/resolve-deftm-var #'delegated-reference {:dtype :float}))
  (with-call
    '(raster.compiler.core.completed-reference-call-test/delegated-reference xs unused)
    'count
    (fn [{:keys [frontend original]}]
      (let [original-body (:body frontend)
            constructed (transport/construct-original-reference-call frontend original)
            normalized (transport/normalize-frontend-bindings
                        frontend (get-in frontend [:source-entry :params]) constructed)
            root (first (:walked-body normalized))
            origin (transport/normalization-occurrence (:source-normalization normalized) root)]
        (is (some? constructed))
        (is (identical? original-body (:body frontend)) "the frozen frontend is not rewritten")
        (is (identical? original (:original origin)))
        (is (identical? constructed (:completed-call origin)))
        (is (identical? frontend (:frontend origin)))
        (let [protected (last root)]
          (is (completion/bytecode-context-form? protected))
          (with-redefs-fn {#'inline/try-resolve-deftm
                          (fn [& _] (throw (AssertionError. "legacy inliner entered completed body")))}
            #(is (identical? protected (#'inline/inline-invk protected))))
          (is (identical? protected (#'inline/resolve-dispatch-walk protected {})))
          (is (identical? protected (:form (#'inline/rewrite-D-calls-in-form protected))))
          (let [expanded (inline/expand-for-backends root)]
            (is (some (fn [node] (identical? protected node)) (tree-seq coll? seq expanded))
                "backend expansion retains the whole canonical argument/return boundary")))
        (doseq [expression (filter #(and (seq? %) (= '.invk (first %)))
                                  (tree-seq coll? seq root))]
          (is (nil? (transport/normalization-occurrence (:source-normalization normalized) expression))
              "an imported callee occurrence does not acquire caller-source authority"))
        (doseq [f [(compiled frontend original) (compiled frontend root)]
                count [-1 0 5]]
          (reset! events [])
          (let [value (float-array [3.0])]
            (is (identical? (when-not (neg? count) value) (f value count))))
          (is (= [:delegate] @events)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exact unique completed call"
                             (transport/normalize-frontend-bindings
                              frontend (get-in frontend [:source-entry :params])
                              (transport/completed-reference-call-view constructed))))))))

(deftest completed-call-splicing-preserves-branches-rebinding-and-earlier-failure
  (doseq [wrap [(fn [call]
                 (list 'let* '[count (clojure.core/inc count)]
                       (list 'if '(clojure.core/< count 0) 'xs call)))
               (fn [call] (list 'do '(clojure.core/quot 1 count) call))
               ;; Equal source calls remain separate actual occurrences: one
               ;; completed call must not rewrite its sibling by syntax equality.
               (fn [call] (list 'do call call))]]
    (with-call
      '(do (clojure.core/swap! raster.compiler.core.completed-reference-call-test/events
                               clojure.core/conj :body) xs)
      '(do (clojure.core/swap! raster.compiler.core.completed-reference-call-test/events
                               clojure.core/conj :argument)
           (clojure.core/quot 1 count))
      wrap
      (fn [{:keys [frontend original]}]
        (let [constructed (transport/construct-original-reference-call frontend original)
              normalized (transport/normalize-frontend-bindings
                          frontend (get-in frontend [:source-entry :params]) constructed)
              imported (transport/prepare-inline-body
                        frontend (:body frontend) (get-in frontend [:source-entry :params])
                        (get-in frontend [:source-entry :tags]) (get-in frontend [:source-entry :source-ns])
                        constructed)
              root (first (:walked-body normalized))
              root-origin (transport/normalization-occurrence (:source-normalization normalized) root)
              source-fn (compiled frontend (first (:body frontend)))
              constructed-fn (compiled frontend root)
              imported-fn (compiled frontend (first (:walked-body imported)))
              value (float-array [2.0])
              run (fn [f count]
                    (reset! events [])
                    (try [:value (identical? value (f value count)) @events]
                         (catch Exception error [:exception (class error) @events])))]
          (is (some? constructed))
          (is (identical? (first (:body frontend)) (:original root-origin)))
          (is (nil? (:completed-call root-origin)) "rebuilt ancestors carry origin, not call authority")
          (let [import-root (first (:walked-body imported))
                origin (transport/body-import-occurrence
                        (:source-import imported) (:walked-body imported) import-root)
                origins (keep #(transport/body-import-occurrence
                                (:source-import imported) (:walked-body imported) %)
                              (filter seq? (tree-seq coll? seq import-root)))]
            (is (identical? (first (:body frontend)) (:original origin)))
            (is (nil? (:completed-call origin)))
            (is (= 1 (count (filter :completed-call origins)))))
          (doseq [count [-2 -1 0 3]]
            (is (= (run source-fn count) (run constructed-fn count) (run imported-fn count))))
          (is (= 1 (count (filter :completed-call
                                 (keep #(transport/normalization-occurrence
                                         (:source-normalization normalized) %)
                                       (filter seq? (tree-seq coll? seq root))))))))))))

(deftest completed-argument-descendants-keep-only-their-original-caller-origin
  (with-call 'xs '(clojure.core/identity count)
    (fn [{:keys [frontend original]}]
      (let [argument (second (:arguments (descriptor/call-description original)))
            constructed (transport/construct-original-reference-call frontend original)
            normalized (transport/normalize-frontend-bindings
                        frontend (get-in frontend [:source-entry :params]) constructed)
            imported (transport/prepare-inline-body
                      frontend (:body frontend) (get-in frontend [:source-entry :params])
                      (get-in frontend [:source-entry :tags]) (get-in frontend [:source-entry :source-ns])
                      constructed)
            origins (keep #(transport/normalization-occurrence (:source-normalization normalized) %)
                          (filter seq? (tree-seq coll? seq (:walked-body normalized))))
            imported-origins (keep #(transport/body-import-occurrence
                                    (:source-import imported) (:walked-body imported) %)
                                  (filter seq? (tree-seq coll? seq (:walked-body imported))))]
        (is (some? constructed))
        (is (= 1 (count (filter #(identical? argument (:original %)) origins))))
        (is (= 1 (count (filter #(identical? argument (:original %)) imported-origins))))))))

(deftest invocation-input-origin-query-is-positional-and-transactional
  (with-call 'xs '(clojure.core/identity count)
    (fn [{:keys [frontend original]}]
      (let [before @entry/*emission-inputs*
            copied (with-meta (apply list original) (meta original))]
        (doseq [[parent index] [[original -1] [original 2] [original 1.5]
                               [original nil] [original 0] [copied 1]]]
          (is (nil? (entry/frontend-invocation-input-source frontend parent index)))
          (is (identical? before @entry/*emission-inputs*)))
        (let [argument (second (:arguments (descriptor/call-description original)))
              parent (entry/frontend-completed-invocation-context frontend original)
              origin (entry/frontend-invocation-input-source frontend original 1)]
          (is (some? origin))
          (is (identical? argument (:source-input origin)))
          (is (identical? (get-in parent [:inputs 1 :operand]) (:operand origin)))
          (is (not (identical? argument (:operand origin)))
              "the tested join crosses the real producer reconstruction boundary"))))))

(deftest constructor-rejects-copied-occurrences-and-retains-stale-history
  (with-call 'xs 'count
    (fn [{:keys [frontend original leaf]}]
      (let [before @entry/*emission-inputs*]
        (is (nil? (transport/construct-original-reference-call frontend (apply list original))))
        (is (identical? before @entry/*emission-inputs*)))
      (let [constructed (transport/construct-original-reference-call frontend original)
            metadata (meta leaf)]
        (is (some? constructed))
        (try
          (alter-meta! leaf assoc ::changed true)
          (is (thrown? clojure.lang.ExceptionInfo
                       (transport/completed-reference-call-view constructed)))
          (finally (reset-meta! leaf metadata)))))))

(deftest helper-import-rejects-copied-foreign-and-stale-completed-calls
  (with-call 'xs 'count
    (fn [{:keys [frontend original leaf]}]
      (let [constructed (transport/construct-original-reference-call frontend original)
            import! (fn [source cap]
                      (transport/prepare-inline-body
                       source (:body source) (get-in source [:source-entry :params])
                       (get-in source [:source-entry :tags]) (get-in source [:source-entry :source-ns]) cap))
            foreign (entry/original-source-frontend leaf)
            copied (transport/completed-reference-call-view constructed)]
        (doseq [[source cap] [[frontend copied] [foreign constructed]]]
          (let [before @entry/*emission-inputs*]
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exact unique completed call"
                                 (import! source cap)))
            (is (identical? before @entry/*emission-inputs*))))
        (let [metadata (meta leaf) before @entry/*emission-inputs*]
          (try
            (alter-meta! leaf assoc ::changed-import true)
            (is (thrown? clojure.lang.ExceptionInfo (import! frontend constructed)))
            (is (identical? before @entry/*emission-inputs*))
            (finally (reset-meta! leaf metadata))))))))

(deftest constructor-needs-an-explicit-whole-body-dependency-witness
  (with-call 'xs 'count
    (fn [{:keys [frontend original]}]
      (let [observe entry/frontend-reference-return-bridge]
        ;; Only remove/deny facts from an actual observed bridge. This grants no
        ;; positive authority; the real sibling/global collision is qualified at
        ;; its emitter in reference-return-bridge-test.
        (doseq [change [#(update % :body-context dissoc :unretained-sibling-calls?)
                        #(assoc-in % [:body-context :unretained-sibling-calls?] nil)
                        #(assoc-in % [:body-context :unretained-sibling-calls?] true)]]
          (let [before @entry/*emission-inputs*]
            (with-redefs [entry/frontend-reference-return-bridge
                          (fn [callee] (some-> (observe callee) change))]
              (is (nil? (transport/construct-original-reference-call frontend original))))
            (is (identical? before @entry/*emission-inputs*)
                "refused construction publishes no speculative dependencies")))))))

(deftest authentic-method-local-callee-cannot-rebind-to-a-global-during-construction
  ;; The injected sibling's qualified lookup name is also munged into a JVM
  ;; method name; keep this fixture namespace free of illegal method-name dots.
  (let [scope (create-ns (gensym "raster_completed_call_sibling_"))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(defn collision [xs]
                 (swap! raster.compiler.core.completed-reference-call-test/events conj :global)
                 xs))
        (eval '(raster.core/deftm leaf [xs :- (Array float) unused :- Long] :- (Array float)
                 (collision xs)))
        (eval '(raster.core/deftm caller [xs :- (Array float) count :- Long] :- (Array float)
                 (leaf xs count))))
      (let [leaf (core/resolve-deftm-var (ns-resolve scope 'leaf) {:dtype :float})
            caller (core/resolve-deftm-var (ns-resolve scope 'caller) {:dtype :float})
            emit @#'core/emit-owned-bytecode!
            sibling {:name (symbol (str (ns-name scope)) "collision")
                     :params [(with-meta 'xs {:tag 'floats})]
                     :return-tag 'floats :walked-body '[nil] :source-ns scope}]
        ;; The real source registry, ownership snapshots and generated wrapper
        ;; are unchanged. Only the actual emitter receives an owned sibling.
        (with-redefs-fn
          {#'core/emit-owned-bytecode!
           (fn [emitter & args]
             (apply emit
                    (fn [& original-args] (apply emitter (concat original-args [[sibling]])))
                    args))}
          #(prepare! leaf))
        (prepare! caller)
        (reset! events [])
        (is (nil? ((var-get leaf) (float-array [1.0]) 0)))
        (is (empty? @events) "the original wrapper selects its actual same-class sibling")
        (entry/call-with-definition-inputs
         (fn []
           (let [frontend (entry/original-source-frontend caller)
                 callee (entry/original-source-frontend leaf)
                 bridge (entry/frontend-reference-return-bridge callee)
                 before @entry/*emission-inputs*]
             (is (some? bridge))
             (is (true? (get-in bridge [:body-context :unretained-sibling-calls?])))
             (is (nil? (transport/construct-original-reference-call frontend (first (:body frontend)))))
             (is (identical? before @entry/*emission-inputs*))
             (is (empty? @events))))))
      (finally (remove-ns (ns-name scope))))))
