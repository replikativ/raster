(ns raster.compiler.backend.jvm.terminal-source-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.backend.jvm.terminal-source :as terminal]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.parallel-program :as program]))

(core/deftm source-definition [value :- Double] :- Object (float value))
(core/deftm primitive-source-definition [value :- Double] :- Float (float value))
(core/deftm object-port-source-definition [value :- Double, coefficient :- Object] :- Object
  (float (* value (double coefficient))))
(def source-effects (atom 0))
(defn observed-helper [value] (swap! source-effects inc) (float value))
(defn observed-identity [value] (swap! source-effects inc) value)

(defn- reason [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- traced-scalar [expression ports context]
  (entry/call-with-definition-inputs
   #(terminal/trace-scalar expression ports
                           (the-ns 'raster.compiler.backend.jvm.terminal-source-owner-test)
                           {:unchecked-math false} context)))

(defn- compile-traced [projection]
  (owner/compile-function (:params projection) (:source projection)
                          (fn [params body] (list 'fn* params body))
                          (merge projection {:source-policy {:unchecked-math false}
                                             :source-ns (the-ns 'raster.compiler.backend.jvm.terminal-source-owner-test)})))

(deftest scalar-trace-retains-exact-occurrences-and-selected-method-plans
  (let [expression '(clojure.core/+ left (clojure.core/* right 2.0))
        ports [{:symbol 'left :carrier Double/TYPE} {:symbol 'right :carrier Double/TYPE}]
        projection (traced-scalar expression ports :typed-scalar)
        correspondence (:correspondence projection)
        events (:events correspondence)
        entered (filterv #(= :enter (:kind %)) events)
        values (filterv #(= :value (:kind %)) events)
        methods (filterv #(and (= :resolved-call (:kind %)) (:occurrence %)) events)
        execute (compile-traced projection)]
    (is (identical? expression (:expression correspondence)))
    (is (identical? ports (:parameters correspondence)))
    (is (false? (:admitted? correspondence)))
    (is (= Double/TYPE (get-in projection [:result-facts :carrier])))
    (is (= ["multiply" "add"] (mapv #(-> % :plan :method .getName) methods)))
    (is (every? #(= [Double/TYPE Double/TYPE] (vec (.getParameterTypes ^java.lang.reflect.Method (get-in % [:plan :method])))) methods))
    (is (= (count entered) (count values)))
    (is (= 1 (count (filter #(nil? (:parent %)) entered))))
    (is (= 1 (count (filter #(identical? expression (:expression %)) values))))
    (is (every? #(= (if (:occurrence %) :body :parameter-boundary) (:phase %))
                (filter #(= :resolved-call (:kind %)) events)))
    (doseq [[_ children] (group-by :parent (filter :parent entered))]
      (is (= (vec (range (count children))) (mapv :child-index children))))
    (doseq [event methods]
      (is (= 1 (count (filter #(and (identical? (:occurrence event) (:occurrence %))
                                    (identical? (:projection event) (:projection %))) values)))))
    (doseq [event entered]
      (let [completed (filterv #(identical? (:occurrence event) (:occurrence %)) values)]
        (is (= 1 (count completed)))
        (is (identical? (:expression event) (:expression (first completed))))
      (when-let [parent (:parent event)]
        (is (= 1 (count (filter #(identical? parent (:occurrence %)) entered)))))))
    (doseq [[left right] [[0.0 -0.0] [1.25 2.0] [Double/MAX_VALUE Double/MAX_VALUE]
                         [Double/NaN 1.0] [Double/NEGATIVE_INFINITY Double/POSITIVE_INFINITY]]]
      (is (= (Double/doubleToRawLongBits (+ left (* right 2.0)))
             (Double/doubleToRawLongBits (execute [left right])))))))

(deftest ordinary-float-source-trace-is-not-typed-float-admission
  (let [expression '(clojure.core/+ left right)
        ports [{:symbol 'left :carrier Float/TYPE} {:symbol 'right :carrier Float/TYPE}]
        projection (traced-scalar expression ports :clojure-source)
        methods (filterv #(and (= :resolved-call (:kind %)) (:occurrence %))
                         (get-in projection [:correspondence :events]))
        execute (compile-traced projection)]
    (is (= Double/TYPE (get-in projection [:result-facts :carrier])))
    (is (= ["doubleCast" "doubleCast" "add"] (mapv #(-> % :plan :method .getName) methods)))
    (is (= [[Float/TYPE] [Float/TYPE] [Double/TYPE Double/TYPE]]
           (mapv #(vec (.getParameterTypes ^java.lang.reflect.Method (get-in % [:plan :method]))) methods)))
    (is (= :unclosed-terminal-source-projection (reason #(traced-scalar expression ports :typed-scalar))))
    (doseq [[left right] [[Float/MAX_VALUE Float/MAX_VALUE] [(float -0.0) (float -0.0)]
                         [Float/POSITIVE_INFINITY Float/NEGATIVE_INFINITY] [(float 0.1) (float 0.2)]]]
      (is (= (Double/doubleToRawLongBits (+ left right))
             (Double/doubleToRawLongBits (execute [left right])))))))

(deftest tracing-does-not-execute-or-certify-reference-conversions-and-helpers
  (let [before @source-effects
        projection (traced-scalar
                    '(raster.compiler.backend.jvm.terminal-source-owner-test/observed-helper value)
                    [{:symbol 'value :carrier Double/TYPE}] :clojure-source)
        methods (filterv #(and (= :resolved-call (:kind %)) (:occurrence %))
                         (get-in projection [:correspondence :events]))]
    (is (= before @source-effects))
    (is (seq (:captures projection)))
    (is (= [Object] (mapv :result-carrier methods)))
    (is (false? (get-in projection [:correspondence :admitted?]))))
  (is (= :unclosed-terminal-source-projection
         (reason #(terminal/trace-scalar 'x [{:symbol 'x :carrier Double/TYPE}]
                                         *ns* {:unchecked-math false} :typed-scalar))))
  (is (= :unclosed-terminal-source-projection
         (reason #(traced-scalar 'x [{:symbol 'x :carrier Double/TYPE}] :unknown-context)))))

(deftest conditional-trace-keeps-branch-positions-even-for-identical-child-objects
  (let [branch '(raster.compiler.backend.jvm.terminal-source-owner-test/observed-helper value)
        expression (list 'if '(clojure.core/nil? missing) branch branch)
        before @source-effects
        projection (traced-scalar expression [{:symbol 'missing :carrier Object}
                                              {:symbol 'value :carrier Double/TYPE}] :clojure-source)
        entered (filterv #(= :enter (:kind %)) (get-in projection [:correspondence :events]))
        root (first entered)
        children (filterv #(identical? (:occurrence root) (:parent %)) entered)
        [then else] (rest children)]
    (is (= before @source-effects))
    (is (= [0 1 2] (mapv :child-index children)))
    (is (identical? branch (:expression then)))
    (is (identical? branch (:expression else)))
    (is (not (identical? (:occurrence then) (:occurrence else))))))

(deftest scalar-trace-fences-a-namespace-change-after-the-final-projection
  (let [source-ns (the-ns 'raster.compiler.backend.jvm.terminal-source-owner-test)
        replacement (create-ns (gensym "raster.test.projection-namespace."))
        project @#'terminal/project*
        find-original find-ns
        changed (atom false)]
    (try
      (is (= :unclosed-terminal-source-projection
             (reason #(entry/call-with-definition-inputs
                       (fn []
                         (with-redefs-fn
                           {#'terminal/project* (fn [& args] (let [result (apply project args)] (reset! changed true) result))
                            #'clojure.core/find-ns (fn [symbol] (if (and @changed (= symbol (ns-name source-ns)))
                                                                 replacement (find-original symbol)))}
                           (fn [] (terminal/trace-scalar 'value [{:symbol 'value :carrier Double/TYPE}]
                                                         source-ns {:unchecked-math false} :typed-scalar))))))))
      (finally (remove-ns (ns-name replacement))))))

(defn- boundary [body]
  ;; A structural region fixture. This alone cannot certify source-to-program
  ;; correspondence, physical exports or completed numerical execution.
  (program/make
   {:dialect :typed-parallel :source (list 'let* '[total (sum x)] body)
    :values {'x (av/tensor {:dtype :double :shape [4]})
             'total (av/tensor {:dtype :double :shape []})}
    :inputs '[x] :outputs '[total]
    :equations [(program/map->ProgramEquation
                 {:id :sum :site [:binding 'total] :source '(sum x)
                  :operands '[x] :results '[total] :operations '[(reduce)]
                  :effects #{} :provenance {} :attributes {}})]}))

(defn- source [] (core/resolve-deftm-var #'source-definition {:dtype :double}))
(defn- contract [p v]
  (invocation/terminal-source-result
   p {:source-policy (:raster.core/source-policy (meta v))
      :source-namespace (ns-name (:ns (meta v)))}))

(defn- compile-owned [p c v]
  (entry/call-with-definition-inputs
   (fn []
     (entry/observe-definition-input! v (:ns (meta v)) (:name (meta v)))
     {:owner (owner/compile-terminal-result p c v) :collector entry/*emission-inputs*})))

(defn- outcome [f x]
  (try (let [value (f x)] [:value (class value) (Float/floatToRawIntBits value)])
       (catch Exception e [:error (class e)])))

(deftest terminal-public-port-never-borrows-a-primitive-carrier-from-logical-dtype
  (object-port-source-definition 1.0 1.5)
  (let [v (core/resolve-deftm-var #'object-port-source-definition {:dtype :double})
        parameters [{:symbol 'value :value (av/tensor {:dtype :double :shape []})}
                    {:symbol 'coefficient :value (av/tensor {:dtype :double :shape []})}]
        p (assoc-in (boundary '(float (* total coefficient)))
                    [:attributes :invocation-plan :parameters] parameters)
        c (invocation/terminal-source-result
           p {:source-policy (:raster.core/source-policy (meta v))
              :source-namespace (ns-name (:ns (meta v))) :public-parameters parameters})]
    ;; The semantic fixture deliberately claims Double. Actual original source
    ;; accepts Object, so this must fail before the generated source is compiled.
    (entry/call-with-definition-inputs
     (fn []
       (entry/observe-definition-input! v (:ns (meta v)) (:name (meta v)))
       (let [frontend (entry/original-source-frontend v)]
         (is (= Object (get-in (entry/frontend-body-entry-carriers frontend v)
                               [:parameter-facts 'coefficient :carrier])))
         (is (= :terminal-source-public-entry
                (reason #(owner/compile-terminal-result p c v frontend)))))))))

(deftest generated-terminal-owner-preserves-checked-float-and-object-completion
  (let [body (list 'if '(clojure.core/nil? total) '(float 0.0)
                   (list 'let* '[projected (clojure.core/float total)]
                         (completion/complete-value completion/object-input-contract 'projected)))
        p (boundary body) v (source) c (contract p v)
        {:keys [owner]} (compile-owned p c v)
        oracle (binding [*unchecked-math* (get-in c [:source-policy :unchecked-math])]
                 (eval '(fn [^double x] (clojure.core/float x))))
        invoke #(owner/invoke-terminal-result owner p c [%])]
    (doseq [value [0.0 -0.0 1.25 Math/PI Double/MAX_VALUE
                   Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN]]
      (is (= (outcome oracle value) (outcome invoke value))))
    (is (= (float 2.0) (invoke 2.0)))
    (is (= (float 3.0) (invoke 3.0)))))

(deftest terminal-double-arithmetic-keeps-the-source-operation-before-narrowing
  (doseq [operator '[clojure.core/+ clojure.core/- clojure.core/* clojure.core//]
          right [0.25 0.0 -0.0]]
    (let [body (list 'clojure.core/float (list operator 'total right))
          p (boundary body) v (source) c (contract p v)
          {:keys [owner]} (compile-owned p c v)
          oracle (eval (list 'fn [(with-meta 'total {:tag 'double})] body))]
      (doseq [value [16777217.0 0.0 -0.0 Double/MAX_VALUE
                     Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN]]
        (is (= (outcome oracle value)
               (outcome #(owner/invoke-terminal-result owner p c [%]) value)))))))

(deftest terminal-float-export-preserves-the-checkpoint-before-core-arithmetic
  ;; Structural source-owner mechanics only; device tests independently owe an
  ;; exact emitted Float ABI and initialized one-element storage.
  (let [v (source)
        samples [(float 0.0) (float -0.0) (float 16777216.0) Float/MAX_VALUE
                 Float/POSITIVE_INFINITY Float/NEGATIVE_INFINITY Float/NaN
                 (Float/intBitsToFloat 2143289361)]
        observe (fn [evaluate]
                  (try
                    (let [value (evaluate)]
                      [(class value) (if (instance? Float value)
                                     (Float/floatToRawIntBits value)
                                     (Double/doubleToRawLongBits value))])
                    (catch Exception error [:error (class error)])))]
    (doseq [body (concat ['total '(clojure.core/double total) '(clojure.core/float total)
                         (list 'clojure.core/float (list 'clojure.core/+ 'total Double/MAX_VALUE))]
                        (for [operation '[clojure.core/+ clojure.core/- clojure.core/* clojure.core//]
                              [left right] [['total 0.25] [0.25 'total] ['total 'total]]]
                          (list operation left right)))]
      (let [p (-> (boundary body)
                  (assoc-in [:values 'x :dtype] :float)
                  (assoc-in [:values 'total :dtype] :float))
            c (contract p v)
            {:keys [owner]} (compile-owned p c v)
            ;; Compile ordinary Clojure source over an ACTUAL primitive Float
            ;; local, independently of Raster's terminal projection.
            oracle (eval (list 'fn '[x]
                               (list 'let* '[total (.floatValue ^Float x)] body)))]
        (doseq [value samples]
          (is (= (observe #(oracle value))
                 (observe #(owner/invoke-terminal-result owner p c [value])))))
        (doseq [wrong [nil 1.0 1]]
          (is (= :terminal-source-arguments
                 (reason #(owner/invoke-terminal-result owner p c [wrong])))))))))

(deftest float-terminal-promotion-does-not-authorize-typed-float-arithmetic
  (is (= :unclosed-terminal-source-projection
         (entry/call-with-definition-inputs
          #(reason (fn []
                     (terminal/project-scalar '(clojure.core/+ value value)
                                              [{:symbol 'value :carrier Float/TYPE}]
                                              *ns* {:unchecked-math false})))))))

(deftest produced-float-support-does-not-invent-a-public-float-entry-carrier
  (is (= :invocation-terminal-public-carrier
         (reason #(invocation/terminal-source-result
                   (boundary '(clojure.core/+ total coefficient))
                   {:source-policy {:unchecked-math false}
                    :source-namespace 'raster.compiler.backend.jvm.terminal-source-owner-test
                    :public-parameters [{:symbol 'coefficient
                                         :value (av/tensor {:dtype :float :shape []})}]})))))

(deftest terminal-arithmetic-never-borrows-a-dtype-tag-or-unowned-definition
  (let [v (source)]
    (doseq [body ['(float (+ total 1.0 2.0))
                 '(float (+ total 1))]]
      (let [p (boundary body) c (contract p v)]
        (is (= :unclosed-terminal-source-projection (reason #(compile-owned p c v))))))
    (let [p (boundary '(float (+ total 0.25))) c (contract p v)
          original (meta #'clojure.core/+)]
      (try
        (alter-meta! #'clojure.core/+ assoc :inline (fn [& _] 0.0))
        (is (= :terminal-intrinsic-definition-mismatch (reason #(compile-owned p c v))))
        (finally (reset-meta! #'clojure.core/+ original))))))

(deftest terminal-executable-needs-preexisting-definition-ownership
  (let [p (boundary '(float total)) v (source) c (contract p v)]
    (is (= :unowned-terminal-source (reason #(owner/compile-terminal-result p c v))))
    (entry/call-with-definition-inputs
     #(is (= :unowned-terminal-source (reason (fn [] (owner/compile-terminal-result p c v))))))))

(deftest terminal-invocation-checks-exact-owner-and-carriers-before-effects
  (let [p (boundary '(raster.compiler.backend.jvm.terminal-source-owner-test/observed-helper total))
        v (source) c (contract p v) {:keys [owner]} (compile-owned p c v)
        before @source-effects]
    (doseq [arguments [[nil] [(float 1.0)] [1] [] [1.0 2.0] (list 1.0)]]
      (is (= :terminal-source-arguments
             (reason #(owner/invoke-terminal-result owner p c arguments)))))
    (is (= :terminal-source-owner-mismatch
           (reason #(owner/invoke-terminal-result owner (assoc p :extra true) c [1.0]))))
    (is (= :terminal-source-owner-mismatch
           (reason #(owner/invoke-terminal-result owner p
                                                  (dissoc (assoc c :extra true) :extra) [1.0]))))
    (is (= before @source-effects))
    (is (= (float 1.0) (owner/invoke-terminal-result owner p c [1.0])))
    (is (= (inc before) @source-effects))))

(deftest terminal-observations-cannot-be-refreshed-after-emission
  (let [p (boundary '(raster.compiler.backend.jvm.terminal-source-owner-test/observed-helper total))
        v (source) c (contract p v) {:keys [owner collector]} (compile-owned p c v)
        original @#'observed-helper before @source-effects]
    (try
      (alter-var-root #'observed-helper (constantly (fn [_] :redefined)))
      ;; A caller mutating the construction collector cannot refresh the
      ;; historical snapshot captured by the already-compiled executable.
      (reset! collector {:definition-proofs {}})
      (is (= :stale-definition-proof-input
             (reason #(owner/invoke-terminal-result owner p c [1.0]))))
      (is (= before @source-effects))
      (finally (alter-var-root #'observed-helper (constantly original))))))

(deftest changed-source-policy-invalidates-the-terminal-owner
  (let [p (boundary '(float total)) v (source) c (contract p v)
        {:keys [owner]} (compile-owned p c v) original (meta v)]
    (try
      (alter-meta! v assoc :raster.core/source-policy {:unchecked-math :warn-on-boxed})
      (is (= :stale-definition-proof-input
             (reason #(owner/invoke-terminal-result owner p c [1.0]))))
      (finally (reset-meta! v original)))))

(deftest source-compilation-uses-the-definitions-policy-not-ambient-state
  (doseq [mode [false true :warn-on-boxed]]
    (let [generic (binding [*ns* (the-ns 'raster.compiler.backend.jvm.terminal-source-owner-test)
                            *unchecked-math* mode]
                    (eval '(core/deftm mode-definition [x :- Double] :- Object (float x))))
          v (core/resolve-deftm-var generic {:dtype :double})
          p (boundary '(clojure.core/float total)) c (contract p v)
          {:keys [owner]} (binding [*unchecked-math* (not mode)] (compile-owned p c v))
          oracle (binding [*unchecked-math* mode]
                   (eval '(fn [^double x] (clojure.core/float x))))]
      (doseq [value [Double/MAX_VALUE Double/POSITIVE_INFINITY Double/NaN -0.0]]
        (is (= (outcome oracle value)
               (outcome #(owner/invoke-terminal-result owner p c [%]) value)))))))

(deftest many-long-ports-retain-integer-precision-without-primitive-ifn-limits
  (let [ids (mapv #(symbol (str "long" %)) (range 25))
        scalar (av/tensor {:dtype :long :shape []})
        p (program/make
           {:dialect :typed-parallel :source (list 'let* (vec (mapcat #(vector % '(count x)) ids)) ids)
            :values (into {'x (av/tensor {:dtype :double :shape [4]})} (map #(vector % scalar)) ids)
            :inputs '[x] :outputs ids
            :equations (mapv (fn [id]
                               (program/map->ProgramEquation
                                {:id id :site [:binding id] :source '(count x)
                                 :operands '[x] :results [id] :operations '[(reduce)]
                                 :effects #{} :provenance {} :attributes {}})) ids)})
        v (source) c (contract p v) {:keys [owner]} (compile-owned p c v)
        values (vec (take 25 (cycle [Long/MIN_VALUE Long/MAX_VALUE 9007199254740993])))]
    (is (= values (owner/invoke-terminal-result owner p c values)))
    (is (every? #(instance? Long %) (owner/invoke-terminal-result owner p c values)))))

(defn invalidating-helper [value]
  (alter-meta! (source) assoc ::changed-during-result true)
  (float value))

(deftest definition-changes-during-observation-prevent-result-publication
  (let [p (boundary '(raster.compiler.backend.jvm.terminal-source-owner-test/invalidating-helper total))
        v (source) c (contract p v) {:keys [owner]} (compile-owned p c v)
        original (meta v)]
    (try
      (is (= :stale-definition-proof-input
             (reason #(owner/invoke-terminal-result owner p c [1.0]))))
      (finally (reset-meta! v original)))))

(def expansion-effects (atom 0))
(defn hidden-helper [x] (float x))
(defmacro hidden-macro [x]
  (swap! expansion-effects inc)
  (list 'raster.compiler.backend.jvm.terminal-source-owner-test/hidden-helper x))
(defn hidden-inline [x] (float x))

(deftest macro-and-inline-introduced-dependencies-are-refused-not-unobserved
  (let [v (source) before @expansion-effects original (meta #'hidden-inline)]
    (try
      (alter-meta! #'hidden-inline assoc :inline
                   (fn [x]
                     (swap! expansion-effects inc)
                     (list 'raster.compiler.backend.jvm.terminal-source-owner-test/hidden-helper x)))
      (doseq [body ['(raster.compiler.backend.jvm.terminal-source-owner-test/hidden-macro total)
                   '(raster.compiler.backend.jvm.terminal-source-owner-test/hidden-inline total)
                   '(Math/exp total)
                   '(.doubleValue total)]]
        (let [p (boundary body) c (contract p v)]
          (is (= :unclosed-terminal-source-projection
                 (reason #(compile-owned p c v))))))
      (is (= before @expansion-effects) "no expansion callable ran during source compilation")
      (finally (reset-meta! #'hidden-inline original)))))

(deftest nested-conversions-use-actual-method-result-carriers
  (let [p (boundary '(clojure.core/float (clojure.core/unchecked-float total)))
        v (source) c (contract p v) {:keys [owner]} (compile-owned p c v)
        oracle (fn [^double x] (float (unchecked-float x)))]
    (doseq [value [Double/MAX_VALUE Double/POSITIVE_INFINITY Double/NaN -0.0]]
      (is (= (outcome oracle value)
             (outcome #(owner/invoke-terminal-result owner p c [%]) value))))))

(deftest altered-intrinsic-inline-metadata-does-not-authenticate-itself
  (doseq [[intrinsic body] [[#'clojure.core/float '(clojure.core/float total)]
                            [#'clojure.core/nil? '(clojure.core/nil? total)]]]
    (let [original (meta intrinsic) v (source) p (boundary body) c (contract p v)]
      (try
        (alter-meta! intrinsic assoc :inline (fn [& _] :changed-inline))
        (is (= :terminal-intrinsic-definition-mismatch (reason #(compile-owned p c v))))
        (finally (reset-meta! intrinsic original))))))

(deftest mixed-primitive-join-is-not-guessed-from-result-dtype
  (let [v (source)]
    (doseq [body ['(clojure.core/float (if (clojure.core/nil? total)
                                         (clojure.core/unchecked-float total) total))
                 '(clojure.core/float (if (clojure.core/nil? total) 9007199254740993 total))]]
      (let [p (boundary body) c (contract p v)]
        (is (= :unclosed-terminal-source-projection (reason #(compile-owned p c v))))))))

(defn primitive-helper [^double value] (float value))

(deftest helper-primitive-abi-needs-its-own-correspondence
  (let [v (source) p (boundary '(raster.compiler.backend.jvm.terminal-source-owner-test/primitive-helper total))
        c (contract p v)]
    (is (= :unclosed-terminal-source-projection (reason #(compile-owned p c v))))))

(deftest reference-joins-retain-nil-and-float-overload-behavior
  (let [v (source)]
    (doseq [selected [true false]]
      (let [body (list 'clojure.core/float
                       (list 'if selected '(clojure.core/unchecked-float total) nil))
            p (boundary body) c (contract p v) {:keys [owner]} (compile-owned p c v)
            oracle (binding [*unchecked-math* (get-in c [:source-policy :unchecked-math])]
                     (eval (list 'fn [(with-meta 'total {:tag 'double})] body)))]
        (doseq [value [Double/MAX_VALUE Double/POSITIVE_INFINITY Double/NaN -0.0]]
          (is (= (outcome oracle value)
                 (outcome #(owner/invoke-terminal-result owner p c [%]) value))))))))

(deftest source-hints-are-not-silently-discarded-at-host-projection
  (let [v (source)
        helper '(raster.compiler.backend.jvm.terminal-source-owner-test/hidden-helper total)
        long-local (with-meta 'local {:tag 'long})]
    (doseq [body [(list 'let* [long-local helper] '(clojure.core/float local))
                 (list 'clojure.core/float (with-meta helper {:tag 'long}))
                 (list 'clojure.core/float (with-meta 'total {:tag 'long}))
                 (list (with-meta 'clojure.core/float {:param-tags ['Object]}) 'total)]]
      (let [p (boundary body) c (contract p v)]
        (is (= :unclosed-terminal-source-projection (reason #(compile-owned p c v))))))))

(deftest retained-control-and-object-local-hints-match-source-carriers
  ;; This is the actual generated return shape of mixed-scale-energy-gradient:
  ;; an advisory Float tag on a conditional whose other arm completes to Object.
  ;; Clojure derives control-form carriers from children, ignoring the form tag.
  (let [v (source)
        object-local (with-meta 'completed {:tag 'Object})
        completed (list 'let* ['projected '(clojure.core/float total)
                              object-local
                              (completion/complete-value completion/object-input-contract 'projected)]
                        object-local)
        terminal (list 'if '(clojure.core/nil? total) '(float 0.0) completed)]
    (doseq [body [(with-meta terminal {:tag 'float})
                 (with-meta (list 'let* [] terminal) {:tag 'long})
                 (with-meta (list 'do terminal) {:tag 'double})]]
      (let [p (boundary body) c (contract p v) {:keys [owner]} (compile-owned p c v)
            ;; Canonical completion is IR, not executable Clojure. A normal
            ;; Object-returning call supplies its independent source context.
            oracle-body (util/postwalk-preserving-meta
                         (fn [x]
                           (if-let [parts (completion/canonical-parts x)]
                             (list 'clojure.core/identity (:operand parts)) x)) body)
            oracle (binding [*unchecked-math* (get-in c [:source-policy :unchecked-math])]
                     (eval (list 'fn [(with-meta 'total {:tag 'double})] oracle-body)))]
        (doseq [value [0.0 -0.0 1.25 Double/MAX_VALUE Double/POSITIVE_INFINITY
                       Double/NEGATIVE_INFINITY Double/NaN]]
          (is (= (outcome oracle value)
                 (outcome #(owner/invoke-terminal-result owner p c [%]) value))))))))

(deftest object-hints-do-not-authorize-a-primitive-initializer
  (let [v (source) id (with-meta 'local {:tag 'Object})
        p (boundary (list 'let* [id '(clojure.core/float total)] id)) c (contract p v)]
    (is (= :unclosed-terminal-source-projection (reason #(compile-owned p c v))))))

(deftest terminal-return-adaptation-is-not-guessed-from-a-declared-tag
  (let [v (core/resolve-deftm-var #'primitive-source-definition {:dtype :double})]
    (doseq [body ['total '(if true nil (float total))
                 '(raster.compiler.backend.jvm.terminal-source-owner-test/observed-helper total)]]
      (let [p (boundary body) c (contract p v)]
        (is (= :terminal-source-return-completion (reason #(compile-owned p c v))))))
    (let [id (with-meta 'projected {:tag 'float})
          p (boundary (list 'let* [id '(float total)] id)) c (contract p v)
          {:keys [owner]} (compile-owned p c v)]
      (is (= [:value Float (Float/floatToRawIntBits (float 1.25))]
             (outcome #(owner/invoke-terminal-result owner p c [%]) 1.25)))
      (is (= [:error IllegalArgumentException]
             (outcome #(owner/invoke-terminal-result owner p c [%]) Double/MAX_VALUE))))))

(deftest unused-terminal-check-retains-effects-and-exception-order
  (let [body '(let* [before (raster.compiler.backend.jvm.terminal-source-owner-test/observed-identity total)
                    unused (clojure.core/float total)
                    after (raster.compiler.backend.jvm.terminal-source-owner-test/observed-identity total)] after)
        v (source) p (boundary body) c (contract p v) {:keys [owner]} (compile-owned p c v)
        before @source-effects]
    (is (thrown? IllegalArgumentException (owner/invoke-terminal-result owner p c [Double/MAX_VALUE])))
    (is (= (inc before) @source-effects) "the earlier effect occurs; the later effect does not")))
