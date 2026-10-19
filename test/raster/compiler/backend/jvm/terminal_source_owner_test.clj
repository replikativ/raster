(ns raster.compiler.backend.jvm.terminal-source-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.parallel-program :as program]))

(core/deftm source-definition [value :- Double] :- Object (float value))
(def source-effects (atom 0))
(defn observed-helper [value] (swap! source-effects inc) (float value))

(defn- reason [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

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
