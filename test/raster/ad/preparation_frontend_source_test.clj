(ns raster.ad.preparation-frontend-source-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.ad.reverse :as reverse]
            [raster.ad.templates :as templates]
            [raster.compiler.ir.form :as form]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.core.method-entry :as entry]))

(deftm squared-loss [x :- Double] :- Double (numeric/* x x))
(def preparation-body-calls (atom 0))
(deftm checked-policy-loss [x :- Double y :- Double] :- Double
  (let [rounded (float y)] (numeric/+ (numeric/* x x) (double rounded))))
(deftm typed-entry-loss [unused :- Double data :- (Array double) x :- Double] :- Double
  (numeric/* x (aget data 0)))
(deftm two-squared-loss [x :- Double y :- Double] :- Double
  (numeric/+ (numeric/* x x) (numeric/* y y)))

(deftm gradient-calling-root [x :- Double] :- Double
  (let [vg ((reverse/value+grad (var squared-loss) :wrt [0]) x)]
    (clojure.core/nth vg 1)))

(deftm same-signature-owner-donor [x :- Double] :- Double (numeric/+ x 1.0))

(defn- result [prepared x]
  (((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared) x))

(deftest preparation-uses-the-frozen-frontend-not-replaceable-walk-caches
  (is (= 9.0 (squared-loss 3.0)))
  (let [source (core/resolve-deftm-var #'squared-loss)
        metadata (meta source)]
    (try
      ;; This is derived cache data, not a source definition or original body.
      ;; It must not replace what the published implementation actually means.
      (alter-meta! source assoc
                   :raster.core/deftm-walked-body '[(clojure.core/* x 7.0)]
                   :raster.core/deftm-walked-body-typed '[(clojure.core/* x 7.0)])
      (is (= [9.0 6.0] (result (reverse/prepare-value+grad #'squared-loss [0]) 3.0)))
      (finally (reset-meta! source metadata)))))

(deftest generated-program-fields-remain-bound-to-their-original-producer
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])]
    (is (identical? prepared (reverse/assert-prepared-program-source! prepared)))
    (doseq [changed [(update prepared :walked-body #(vec (map identity %)))
                     (update prepared :params #(vec (map identity %)))
                     (update prepared :tags #(vec (map identity %)))
                     (update prepared :param-adj-syms #(vec (map identity %)))
                     (update prepared :result-sym #(with-meta % {:raster.type/tag 'float}))
                     (assoc prepared :source-ns (the-ns 'clojure.core))
                     (assoc prepared ::unexpected true)]]
      (is (= :ad-prepared-program-mismatch
             (try (reverse/assert-prepared-program-source! changed) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
    (is (= :ad-prepared-program-owner
           (try (reverse/assert-prepared-program-source!
                 (assoc prepared :raster.ad.reverse/generated-program-owner (Object.))) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(deftest generated-program-retains-the-original-template-epoch
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        revision (templates/registry-revision)]
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (reverse/assert-prepared-program-source! prepared) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
  (let [revision (atom (templates/registry-revision))
        prepare reverse/ad-prepare]
    (binding [entry/*emission-inputs* (atom {:static [] :values []})]
      (with-redefs [templates/registry-revision (fn [] @revision)
                    reverse/ad-prepare (fn [& arguments]
                                         (let [body (apply prepare arguments)]
                                           (swap! revision inc)
                                           body))]
        (is (= :stale-definition-proof-input
               (try (reverse/prepare-value+grad #'squared-loss [0]) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))))

(deftest generation-and-compilation-use-the-definition-policy-not-the-caller-policy
  (let [seen (atom [])
        prepare reverse/ad-prepare
        emit core/emit-owned-bytecode!]
    (binding [*unchecked-math* true]
      (with-redefs [reverse/ad-prepare (fn [& arguments]
                                       (swap! seen conj [:prepare *unchecked-math*])
                                       (apply prepare arguments))
                    core/emit-owned-bytecode!
                    (fn [emitter class-name & arguments]
                      (when (.startsWith ^String class-name "raster.compiled.PreparedGradient")
                        (swap! seen conj [:compile *unchecked-math*]))
                      (apply emit emitter class-name arguments))]
        (let [prepared (reverse/prepare-value+grad #'squared-loss [0])]
          (is (= [9.0 6.0] (result prepared 3.0))))))
    (is (= [[:prepare false] [:compile false]] @seen))))

(deftest prepared-runtime-reuses-its-exact-owned-physical-entry
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        physical (reverse/prepared-bytecode-entry prepared)
        emission (:emission physical)
        compile-runtime (deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad))
        execute (with-redefs [core/emit-owned-bytecode!
                             (fn [& _] (throw (AssertionError. "prepared runtime recompiled its entry")))]
                  (compile-runtime prepared))]
    (is (identical? physical (reverse/prepared-bytecode-entry prepared)))
    (is (identical? (:implementation physical) (:implementation emission)))
    (is (identical? (:walked-body prepared) (:input-body physical)))
    (is (some? (bytecode/retained-input-binding-context
                emission (:walked-body prepared) (first (:walked-body prepared)) 0)))
    (doseq [x [3.0 -2.5 -0.0]]
      (is (= [(* x x) (* 2.0 x)] (execute x))))
    (let [revision (templates/registry-revision)]
      (with-redefs [templates/registry-revision (constantly (inc revision))]
        (is (= :ad-prepared-program-stale-rules
               (try (execute 3.0) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest owned-generated-entry-converts-all-parameters-before-body-evaluation
  (let [prepared (reverse/prepare-value+grad #'typed-entry-loss [2])
        execute ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared)]
    (is (= [6.0 nil nil 2.0] (execute 1.0 (double-array [2.0]) 3.0)))
    ;; Even an unused declared parameter is converted at the typed entry.
    ;; Its null failure precedes the bounds failure in the body.
    (is (thrown? NullPointerException (execute nil (double-array 0) 3.0)))
    (is (thrown? ClassCastException (execute 1.0 (float-array [2.0]) 3.0)))
    (is (thrown? NullPointerException (execute 1.0 (double-array 0) nil)))
    (is (thrown? ArrayIndexOutOfBoundsException (execute 1.0 (double-array 0) 3.0)))))

(deftest legacy-input-does-not-acquire-generated-program-authority
  (binding [entry/*emission-inputs* (atom {:static [] :values []})]
    (with-redefs [entry/original-source-frontend (constantly nil)]
      (let [prepared (reverse/prepare-value+grad #'squared-loss [0])]
        (is (nil? (reverse/assert-prepared-program-source! prepared)))
        (is (nil? (reverse/prepared-bytecode-entry prepared)))))))

(deftest frozen-source-remains-available-when-a-transformed-module-has-stale-ad-rules
  (is (= 6.0 (gradient-calling-root 3.0)))
  (let [source (core/resolve-deftm-var #'gradient-calling-root)
        receiver (ns-resolve (:ns (meta source))
                             (symbol (str (:name (meta source)) "-impl")))
        old (reverse/prepare-value+grad #'squared-loss [0])
        ;; Earlier qualification can legitimately promote a selected helper.
        ;; Start from a current emission so this test isolates rule invalidation,
        ;; rather than observing an already stale helper dependency first.
        _ (core/ensure-current-owned-bytecode! receiver)
        implementation (var-get receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        revision (templates/registry-revision)]
    (is (identical? emission (entry/assert-emission-inputs-current! emission)))
    (is (= :jvm-bytecode (:backend emission)))
    (is (seq (:producer-inputs emission)) "the real emitted root retains its generated AD owner")
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (entry/assert-emission-inputs-current! emission) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= :ad-prepared-program-stale-rules
             (try (reverse/assert-prepared-program-source! old) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (entry/call-with-definition-inputs
       (fn []
         (let [frontend (entry/original-source-frontend source)]
           (is (some? frontend))
           (is (identical? frontend (entry/assert-frontend-source! frontend source)))
           (is (identical? implementation (var-get receiver)))
           (is (= :ad-prepared-program-stale-rules
                  (try (entry/frontend-primitive-body-return-bridge frontend) nil
                       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
      (let [fresh (reverse/prepare-value+grad #'squared-loss [0])]
        (is (identical? fresh (reverse/assert-prepared-program-source! fresh)))
        (is (= [9.0 6.0] (result fresh 3.0)))))))

(deftest rule-renewal-rebuilds-source-without-reviving-old-generated-programs
  (gradient-calling-root 3.0)
  (let [source (core/resolve-deftm-var #'gradient-calling-root)
        receiver (ns-resolve (:ns (meta source))
                             (symbol (str (:name (meta source)) "-impl")))
        emission (core/ensure-current-owned-bytecode! receiver)
        prepared (reverse/prepare-value+grad #'squared-loss [0])
        execute ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared)
        reason (fn [f] (try (f) nil
                            (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        revision (templates/registry-revision)]
    (try
      (with-redefs [templates/registry-revision (constantly (inc revision))]
        (is (= :ad-prepared-program-stale-rules
               (reason #(entry/assert-emission-inputs-current! emission))))
        (is (= :ad-prepared-program-stale-rules (reason #(execute 3.0))))
        (let [fresh (core/ensure-current-owned-bytecode! receiver)]
          (is (not (identical? emission fresh)))
          (is (not (identical? (:implementation emission) (:implementation fresh))))
          (is (identical? fresh (entry/assert-emission-inputs-current! fresh)))
          (is (= 6.0 (gradient-calling-root 3.0)))
          (is (= :ad-prepared-program-stale-rules
                 (reason #(entry/assert-emission-inputs-current! emission))))
          (is (= :ad-prepared-program-stale-rules (reason #(execute 3.0))))))
      (finally
        ;; The temporary epoch also invalidates the replacement on unwind.
        ;; Leave a genuinely current implementation for subsequent REPL tests.
        (let [restored (core/ensure-current-owned-bytecode! receiver)]
          (is (identical? restored (entry/assert-emission-inputs-current! restored))))))))

(deftest source-renewal-cannot-launder-a-captured-stale-gradient
  (let [scope (create-ns (gensym "raster.ad.captured-rule-renewal."))
        captured (reverse/value+grad #'squared-loss :wrt [0])
        revision (templates/registry-revision)]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (intern scope 'captured captured)
        (eval '(raster.core/deftm caller [x :- Double] :- Double
                 (raster.numeric/+
                  (clojure.core/nth
                   ((raster.ad.reverse/value+grad
                     (var raster.ad.preparation-frontend-source-test/squared-loss) :wrt [0]) x) 1)
                  (clojure.core/nth (captured x) 1)))))
      (let [root (ns-resolve scope 'caller)
            source (core/resolve-deftm-var root)
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))]
        (is (= 12.0 (root 3.0)))
        (core/ensure-current-owned-bytecode! receiver)
        (with-redefs [templates/registry-revision (constantly (inc revision))]
          ;; The direct constructor can renew. The dynamic captured call is
          ;; not expanded by AD-only source renewal and must retain its own
          ;; runtime owner guard, rather than becoming a fresh gradient.
          (let [fresh (core/ensure-current-owned-bytecode! receiver)]
            (is (identical? fresh (entry/assert-emission-inputs-current! fresh))))
          (is (= :ad-prepared-program-stale-rules
                 (try (root 3.0) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest frozen-source-rejects-a-same-signature-owner-from-another-implementation
  (is (= 9.0 (squared-loss 3.0)))
  (is (= 4.0 (same-signature-owner-donor 3.0)))
  (let [source (core/resolve-deftm-var #'squared-loss)
        receiver-for (fn [source]
                       (ns-resolve (:ns (meta source))
                                   (symbol (str (:name (meta source)) "-impl"))))
        receiver (receiver-for source)
        donor (receiver-for (core/resolve-deftm-var #'same-signature-owner-donor))
        foreign (get-in (meta donor) [:raster.core/bytecode-emission :frontend-source-owner])
        before (meta receiver)]
    (is (some? foreign))
    (try
      (alter-meta! receiver assoc-in [:raster.core/bytecode-emission :frontend-source-owner] foreign)
      (is (= :retained-frontend-owner-mismatch
             (try (entry/call-with-definition-inputs #(entry/original-source-frontend source)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (finally (reset-meta! receiver before)))
    (is (some? (entry/call-with-definition-inputs #(entry/original-source-frontend source))))))

(deftest inactive-checked-conversion-keeps-its-definition-policy
  (let [compiled (binding [*unchecked-math* true]
                   ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad))
                    (reverse/prepare-value+grad #'checked-policy-loss [0])))]
    (is (= [11.0 6.0 nil] (compiled 3.0 2.0)))
    (is (thrown? IllegalArgumentException (compiled 3.0 Double/MAX_VALUE)))))

(deftest normalized-parameter-selection-retains-padded-gradient-slots
  (doseq [[wrt expected] [[[0] [25.0 6.0 nil]] [[1] [25.0 nil 8.0]]]]
    (let [prepared (reverse/prepare-value+grad #'two-squared-loss wrt)
          compiled ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared)]
      (is (identical? prepared (reverse/assert-prepared-program-source! prepared)))
      (is (= expected (compiled 3.0 4.0))))))

(deftest generated-normalization-keeps-active-values-and-inactive-slot-padding
  (let [normalize (deref (ns-resolve 'raster.ad.reverse 'normalize-generated-preparation))
        program {:params '[x observations] :source-ns *ns*
                 :walked-body '[(let* [primal x derivative 1.0]
                                  [primal derivative nil])]}
        plan {:candidate-params '[x]}
        normalized (normalize program plan)]
    (is (= '[primal derivative nil] (last (first (:walked-body normalized)))))
    (is (= '[derivative] (:param-adj-syms normalized)))
    ;; A direct public-parameter alias remains valid; it is not an absent slot.
    (is (= '[x] (:param-adj-syms
                 (normalize (assoc program :walked-body
                                   '[(let* [primal x] [primal x nil])]) plan))))
    (doseq [body '[(let* [primal x] [primal nil nil])
                  (let* [primal x derivative 1.0] [primal derivative observations])
                  (let* [primal x] [primal missing nil])]]
      (is (nil? (normalize (assoc program :walked-body [body]) plan))))))

(deftest prepared-compilation-retains-history-in-an-enclosing-owner
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        parent (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* parent]
      (is (= [9.0 6.0] (result prepared 3.0))))
    (is (seq (:definition-proofs @parent)))
    (let [owner (:raster.ad.reverse/generated-program-owner prepared)
          producers (:producer-inputs @parent)]
      (is (= 1 (count (filter #(identical? owner %) producers))))
      (is (every? #(satisfies? entry/RetainedProducerInput %) producers)))
    (is (entry/assert-definition-inputs-current! @parent))
    (let [revision (templates/registry-revision) before @parent]
      (with-redefs [templates/registry-revision (constantly (inc revision))]
        (binding [entry/*emission-inputs* parent]
          (is (= :ad-prepared-program-stale-rules
                 (try (reverse/adopt-prepared-program-inputs! prepared) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
      (is (identical? before @parent)))))

(deftest enclosing-owner-retains-the-preparations-template-guard
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        parent (atom {:static [] :values []})
        revision (templates/registry-revision)]
    (binding [entry/*emission-inputs* parent]
      (reverse/adopt-prepared-program-inputs! prepared))
    (let [first-inputs (:producer-inputs @parent)
          owner (:raster.ad.reverse/generated-program-owner prepared)]
      (binding [entry/*emission-inputs* parent]
        (reverse/adopt-prepared-program-inputs! prepared))
      (is (= 1 (count (filter #(identical? owner %) (:producer-inputs @parent)))))
      (is (every? #(satisfies? entry/RetainedProducerInput %) first-inputs))
      (is (= (count first-inputs) (count (:producer-inputs @parent))))
      (is (every? true? (map identical? first-inputs (:producer-inputs @parent)))))
    (is (every? #(not (satisfies? entry/RetainedProducerInput %)) (:values @parent)))
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (entry/assert-definition-inputs-current! @parent) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))

(deftest whole-call-projection-uses-shared-nested-scope-hygiene
  ;; Exercise the structural legacy path without manufacturing an opaque owner.
  ;; Existing public AD tests separately exercise authenticated preparations.
  (let [loss (with-meta 'loss {:raster.type/tag 'double :projection-marker :kept})
        source (list 'let*
                     [loss '(let* [reused x
                                   f (fn* [x] (clojure.core/* x 2.0))
                                   walked (loop* [x 0 total reused]
                                            (if (clojure.core/< x 2)
                                              (recur (clojure.core/inc x)
                                                     (clojure.core/+ total (clojure.core/inc x)))
                                              total))]
                              (clojure.core/+ (f reused) walked))
                      'gradient 'x]
                     [loss 'gradient])
        prepared {:walked-body [source] :params '[x] :source-ns (the-ns 'clojure.core)}
        project #(reverse/project-prepared-value+grad-call prepared '[reused] :value+grad {})
        first-call (project)
        second-call (project)
        source-of #(list 'let* (vec (mapcat identity (:bindings %))) (:elements %))
        binders (fn [expression]
                  (set (mapcat #(mapcat :binders (:scopes (form/scope-info %)))
                               (tree-seq coll? seq expression))))
        first-binders (binders (source-of first-call))]
    (is (= [12.0 3.0] ((eval (list 'fn* '[reused] (source-of first-call))) 3.0)))
    (is (empty? (set/intersection first-binders (binders (source-of second-call)))))
    (is (empty? (set/intersection first-binders '#{loss reused f walked x total gradient})))
    (is (= :kept (:projection-marker (meta (:loss-sym first-call)))))
    (is (nil? (reverse/assert-prepared-program-source! prepared)))
    (is (= 2 (count (:elements first-call))))))

(deftest whole-call-projection-retains-owned-padding-and-rejects-stale-inputs
  (doseq [[wrt expected] [[[] [25.0 nil nil]] [[0] [25.0 6.0 nil]] [[1] [25.0 nil 8.0]]]]
    (let [prepared (reverse/prepare-value+grad #'two-squared-loss wrt)
          projection (reverse/project-prepared-value+grad-call prepared '[a b] :value+grad {})
          ;; A structural compiler projection consumes explicitly typed ports;
          ;; it does not promise erasure of physical contexts into generic fn*.
          {:keys [methods]} (bytecode/compile-specialized-class!
                            "raster.compiled.PreparedProjectionTest"
                            [{:name 'run :params [(with-meta 'a {:tag 'double})
                                                (with-meta 'b {:tag 'double})]
                              :return-tag 'Object :source-ns *ns*
                              :walked-body [(list 'let* (vec (mapcat identity (:bindings projection)))
                                                  (:elements projection))]}])]
      (is (= expected (.invoke ^java.lang.reflect.Method (get methods 'run) nil
                              (object-array [3.0 4.0]))))
      (is (= expected
             (((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared)
              3.0 4.0)))
      (is (= 3 (count (:elements projection))))))
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        revision (templates/registry-revision)]
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (reverse/project-prepared-value+grad-call prepared '[x] :value+grad {}) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
    (is (= :ad-prepared-program-mismatch
           (try (reverse/project-prepared-value+grad-call
                 (assoc prepared :walked-body '[(let* [] [0.0 0.0])]) '[x] :value+grad {}) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(deftest whole-call-projection-rejects-private-or-malformed-source
  (doseq [source '[42 (let* [] [0.0]) (let* [a] [0.0 0.0]) (let* [] [0.0 0.0] 9.0)]]
    (is (= :ad-prepared-call-projection
           (try (reverse/project-prepared-value+grad-call
                 {:walked-body [source] :params '[x] :source-ns (the-ns 'clojure.core)}
                 '[x] :value+grad {}) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
  (is (= :unprojected-jvm-source-owner
         (try (reverse/project-prepared-value+grad-call
               {:source-owner nil :walked-body '[(let* [] [0.0 0.0])]
                :params '[x] :source-ns (the-ns 'clojure.core)}
               '[x] :value+grad {}) nil
              (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))

(deftest whole-call-boundary-adopts-original-history-only-on-success
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        parent (atom {:static [] :values []}) before @parent]
    (binding [entry/*emission-inputs* parent]
      (is (thrown? clojure.lang.ExceptionInfo
                   (reverse/project-prepared-value+grad-call prepared '[] :value+grad {})))
      (is (identical? before @parent))
      (reverse/project-prepared-value+grad-call prepared '[x] :value+grad {})
      (is (seq (:producer-inputs @parent)))
      (is (entry/assert-definition-inputs-current! @parent)))))

(deftest independent-preparation-keeps-parent-history-out-of-child-candidates
  (is (= :without-owner
         (binding [entry/*emission-inputs* nil]
           (entry/call-with-independent-definition-inputs
            (fn []
              (is (nil? entry/*emission-inputs*))
              :without-owner)))))
  (let [scope (create-ns (gensym "raster.independent-preparation."))
        reason (fn [f] (try (f) nil
                            (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm parent-source [x :- Double] :- Double (raster.numeric/* x x)))
        (eval '(raster.core/deftm child-source [x :- Double] :- Double (raster.numeric/+ x 1.0))))
      (let [parent-source (core/resolve-deftm-var (ns-resolve scope 'parent-source))
            child-source (core/resolve-deftm-var (ns-resolve scope 'child-source))
            receiver-for #(ns-resolve scope (symbol (str (:name (meta %)) "-impl")))
            parent-receiver (receiver-for parent-source)
            child-receiver (receiver-for child-source)
            _ (doseq [source [parent-source child-source]] (core/prepare-deftm-frontend! source))
            parent-implementation (var-get parent-receiver)
            child-implementation (var-get child-receiver)
            owner (atom {:static [] :values []})
            candidates (atom [])]
        (binding [entry/*emission-inputs* owner]
          (entry/original-source-frontend parent-source)
          (let [historical @owner]
            (is (= :stale-definition-proof-input
                   (reason #(entry/call-with-independent-definition-inputs
                             (fn []
                               (entry/original-source-frontend child-source)
                               (entry/prepare-pending-definition-inputs!
                                (fn [receiver]
                                  (swap! candidates conj receiver)
                                  (core/ensure-current-owned-bytecode! receiver))))))))
            (is (= [child-receiver] @candidates))
            (is (identical? parent-implementation (var-get parent-receiver)))
            (is (not (identical? child-implementation (var-get child-receiver))))
            (is (identical? historical @owner) "a failed child cannot partially adopt its observations")
            (is (entry/assert-definition-inputs-current! historical))
            (is (= :prepared
                   (entry/call-with-independent-definition-inputs
                    (fn [] (entry/original-source-frontend child-source) :prepared))))
            (is (entry/same-var-snapshot? (get-in historical [:definition-proofs parent-receiver])
                                         (get-in @owner [:definition-proofs parent-receiver])))
            (is (contains? (:definition-proofs @owner) child-receiver)))
          (let [historical @owner metadata (meta parent-source)]
            (try
              (is (= :stale-definition-proof-input
                     (reason #(entry/call-with-independent-definition-inputs
                               (fn []
                                 (entry/original-source-frontend child-source)
                                 (alter-meta! parent-source assoc ::changed-during-child true)
                                 :must-not-publish)))))
              (is (identical? historical @owner))
              (finally (reset-meta! parent-source metadata))))))
      (finally (remove-ns (ns-name scope))))))

(deftest lazy-jit-gradient-root-retains-its-actual-ad-producer
  (let [scope (create-ns (gensym "raster.ad-preparation-jit."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double (raster.numeric/* x x)))
        (eval '(raster.core/deftm gradient [x :- Double] :- Double
                 (let [vg ((raster.ad.reverse/value+grad (var square)) x)]
                   (clojure.core/nth vg 1)))))
      (let [root (ns-resolve scope 'gradient)
            _ (is (= 6.0 (binding [core/*jit-silent-fallback?* false] (root 3.0))))
            source (core/resolve-deftm-var root)
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            emission (:raster.core/bytecode-emission (meta receiver))
            revision (templates/registry-revision)]
        (is (seq (:producer-inputs emission))
            "a real JIT root retains preparation consumed before bytecode emission")
        (let [guard (entry/prepare-emission-input-guard! emission)]
          (with-redefs [templates/registry-revision (constantly (inc revision))]
            (is (= :ad-prepared-program-stale-rules
                   (try (guard) nil
                        (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest cold-preparation-observes-the-producer-without-running-its-numeric-body
  ;; Repeated REPL runs must exercise a genuinely cold source receiver too.
  (binding [*ns* (the-ns 'raster.ad.preparation-frontend-source-test)]
    (eval '(deftm cold-squared-loss [x :- Double] :- Double
             (do (swap! preparation-body-calls inc) (numeric/* x x)))))
  (let [function (ns-resolve 'raster.ad.preparation-frontend-source-test 'cold-squared-loss)
        source (core/resolve-deftm-var function)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        before @preparation-body-calls
        implementation (var-get receiver)
        _ (is (some? (:raster.core/jit-spec (meta receiver))))
        emit core/emit-owned-bytecode!
        attempts (atom [])
        prepared (with-redefs [core/emit-owned-bytecode!
                              (fn [emitter class-name & arguments]
                                (let [result (apply emit emitter class-name arguments)]
                                  (when (.startsWith ^String class-name "raster.compiled.PreparedGradient")
                                    (swap! attempts conj result))
                                  result))]
                   (reverse/prepare-value+grad function [0]))
        physical (reverse/prepared-bytecode-entry prepared)
        emission (:emission physical)]
    (is (= before @preparation-body-calls)
        "materializing an observed dependency must not evaluate the model")
    (is (identical? implementation (var-get receiver))
        "an inlined source producer is not an additional physical invocation")
    (is (map? (:raster.core/jit-spec (meta receiver))))
    (is (identical? (:implementation emission) (:implementation physical)))
    (is (identical? emission ((entry/prepare-emission-input-guard! emission))))
    (is (<= 1 (count @attempts)) "the actual numerical producer was emitted")
    (is (identical? (:emission (last @attempts))
                    (:emission (reverse/prepared-bytecode-entry prepared)))
        "only the newly emitted module belongs to the completed preparation")
    (is (= [16.0 8.0] (result prepared 4.0)))
    (is (= (inc before) @preparation-body-calls)
        "the prepared call executes the original effect exactly once")
    (let [original (meta source)]
      (try
        (alter-meta! source assoc :raster.core/deftm-source-body '[42.0])
        (is (thrown? clojure.lang.ExceptionInfo (result prepared 4.0))
            "retained source history still guards the inlined producer")
        (is (= (inc before) @preparation-body-calls))
        (finally (reset-meta! source original))))))

(deftest enclosing-owner-detects-a-source-change-during-ad-preparation
  (let [source (core/resolve-deftm-var #'squared-loss)
        metadata (meta source)
        prepare reverse/ad-prepare]
    (try
      (binding [entry/*emission-inputs* (atom {:static [] :values []})]
        (with-redefs [reverse/ad-prepare
                      (fn [& arguments]
                        (let [body (apply prepare arguments)]
                          (alter-meta! source assoc ::changed-during-preparation (Object.))
                          body))]
          (is (= :stale-definition-proof-input
                 (try (reverse/prepare-value+grad #'squared-loss [0]) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
      (finally (reset-meta! source metadata)))))
