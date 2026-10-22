(ns raster.compiler.core.completed-reference-call-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.core.value-completion :as completion]
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

(defn- with-call [helper-body argument run]
  ;; Fresh modules avoid renewing an old collector after the real JIT promotion.
  (let [scope (create-ns (gensym "raster.completed-call."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval (list 'raster.core/deftm 'leaf
                    '[xs :- (Array float) unused :- Long] ':- '(Array float) helper-body))
        (eval (list 'raster.core/deftm 'caller
                    '[xs :- (Array float) count :- Long] ':- '(Array float)
                    (list 'leaf 'xs argument))))
      (let [leaf (core/resolve-deftm-var (ns-resolve scope 'leaf) {:dtype :float})
            caller (core/resolve-deftm-var (ns-resolve scope 'caller) {:dtype :float})]
        (prepare! leaf)
        (prepare! caller)
        (entry/call-with-definition-inputs
         #(let [frontend (entry/original-source-frontend caller)
                original (first (:body frontend))]
            (run {:scope scope :leaf leaf :caller caller :frontend frontend :original original}))))
      (finally (remove-ns (ns-name scope))))))

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
