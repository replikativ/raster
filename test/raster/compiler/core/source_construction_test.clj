(ns raster.compiler.core.source-construction-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-prefix-test :as fixture]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.core.util :as util]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.fixtures.distributed-training :as training]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.pipeline :as pipeline]))

(defn- capture []
  ;; Finish the ordinary numerical JIT before retaining historical inputs.
  (fixture/unused-checked-helper-input 2 3)
  (let [captured (atom nil) construct transport/with-source-construction-history
        target :ocl:source-construction-test
        compilation
        (with-redefs [transport/with-source-construction-history
                      (fn [normalization run]
                        (let [result (construct normalization run)]
                          (reset! captured (assoc result :normalization normalization))
                          result))]
          (equation/compile #'fixture/unused-checked-helper-input {:target target :dtype :double}
                            {:device-id target :device-type :gpu :backend :ocl
                             :subgroup-dialect :opencl-portable :max-workgroup-size 256}))]
    (assoc @captured :compilation compilation)))

(defn- phase-view [{:keys [source-construction normalization form]}]
  (entry/call-with-definition-inputs
   #(transport/source-construction-view source-construction normalization form)))

(defn- builder-view [input expansion]
  (entry/call-with-definition-inputs #(transport/builder-expansion-view input expansion)))

(defn- origin-view [input expression]
  (entry/call-with-definition-inputs #(transport/construction-occurrence input expression)))

(defn- history [normalization run]
  (entry/call-with-definition-inputs
   #(transport/with-source-construction-history normalization run)))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftm recursive-argument-prefix [n :- Long] :- (Array double)
  (let [size (fixture/literal-count (fixture/literal-count n))
        out (double-array size)]
    (par/map! out i size double 1.0)))

(deftm substitution-helper [unused :- Long x :- Long y :- Long] :- Long
  (let [value (+ x y)] value))

(deftm shadowed-substitution-helper [x :- Long] :- Long
  (let [quoted '(x) x (+ x 1)] x))

(deftest nested-substitution-law-is-positional-and-independent-of-the-builder-callback
  (substitution-helper 0 2 3)
  (entry/call-with-definition-inputs
   (fn []
     (let [callee (#'inline/try-resolve-walked-body #'substitution-helper_m_long_long_long)
           argument '(clojure.core/long value)
           arguments ['(clojure.core/int 4294967296) argument argument]
           build @#'inline/build-inline-expansion
           expansion (build callee arguments {} inline/subst-syms nil '(helper))
           check #(inline/nested-expansion-substitution-view callee arguments %)
           slots (:argument-slots expansion)]
       (is (some? (check expansion)))
       (is (= [0 1 2] (mapv :ordinal slots)))
       (is (= 3 (count (:argument-bindings expansion))))
       (is (identical? argument (:argument (nth slots 1))))
       (is (identical? argument (:argument (nth slots 2))))
       (is (not= (:binder (nth slots 1)) (:binder (nth slots 2))))
       (doseq [changed [(assoc expansion :argument-slots (vec (reverse slots)))
                        (assoc expansion :argument-slots (pop slots))
                        (assoc-in expansion [:argument-slots 2 :binder] (:binder (nth slots 1)))
                        (assoc expansion :argument-bindings (vec (reverse (:argument-bindings expansion))))
                        (assoc expansion :argument-bindings (pop (:argument-bindings expansion)))
                        (assoc expansion :argument-bindings [0])
                        (assoc expansion :argument-bindings nil)
                        (assoc expansion :argument-slots [0])
                        (assoc expansion :argument-slots :invalid)
                        (assoc expansion :argument-slots
                               (mapv #(assoc % :binder false) slots) :argument-bindings [])
                        (assoc-in expansion [:argument-slots 0 :binder] 'value)
                        (assoc expansion :body 0)
                        (assoc expansion :body [])
                        (assoc expansion :body (vary-meta (:body expansion) assoc :tag 'float))
                        (assoc expansion :bindings [])]]
         (is (nil? (check changed))))
       (is (nil? (check (build callee arguments {} (fn [_ _] 0) nil '(helper))))
           "a faulty substitution callback cannot validate itself")
       (let [renamed (with-meta (apply list argument) (meta argument))]
         (is (nil? (check (assoc-in expansion [:argument-slots 1 :argument] renamed)))))
       (is (nil? (inline/nested-expansion-substitution-view
                  (assoc callee :walked-body (mapv identity (:walked-body callee)))
                  arguments expansion)))
       (let [arguments [0 'value 'value]
             expansion (build callee arguments {} inline/subst-syms nil '(helper))]
         (is (empty? (:argument-bindings expansion)))
         (is (some? (inline/nested-expansion-substitution-view callee arguments expansion))))))))

(deftest nested-substitution-cannot-drop-an-unused-throwing-slot
  (fixture/literal-count 0)
  (entry/call-with-definition-inputs
   (fn []
     (let [callee (#'inline/try-resolve-walked-body #'fixture/literal-count_m_long)
           arguments ['(clojure.core/int 4294967296)]
           expansion (#'inline/build-inline-expansion callee arguments {} inline/subst-syms nil '(helper))
           forged (-> expansion (assoc :argument-bindings [])
                      (assoc-in [:argument-slots 0 :binder] false))]
       (is (= 7 (:body expansion)))
       (is (some? (inline/nested-expansion-substitution-view callee arguments expansion)))
       (is (nil? (inline/nested-expansion-substitution-view callee arguments forged)))
       (is (nil? (inline/nested-expansion-substitution-view
                  callee arguments (assoc-in forged [:argument-slots 0 :binder] nil))))))))

(deftest nested-substitution-preserves-shadowing-and-quoted-products
  (shadowed-substitution-helper 1)
  (entry/call-with-definition-inputs
   (fn []
     (let [callee (#'inline/try-resolve-walked-body #'shadowed-substitution-helper_m_long)
           build @#'inline/build-inline-expansion
           arguments ['x]
           good (build callee arguments {} inline/subst-syms nil '(helper))
           copied (build callee arguments {}
                         (fn [substitutions body]
                           (walk/postwalk identity (inline/subst-syms substitutions body)))
                         nil '(helper))]
       (is (some? (inline/nested-expansion-substitution-view callee arguments good)))
       (is (= (:body good) (:body copied)))
       (is (nil? (inline/nested-expansion-substitution-view callee arguments copied)))
       (is (nil? (inline/nested-expansion-substitution-view
                  (assoc callee :source-import {}) arguments good)))))))

(deftest substitution-origins-do-not-promote-inserted-or-shared-calls
  (substitution-helper 0 1 2)
  (let [{:keys [source-construction normalization] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        caller (:normalized (:caller-origin (builder-view source-construction (:expansion event))))
        callee (entry/call-with-definition-inputs
                #(#'inline/try-resolve-walked-body #'substitution-helper_m_long_long_long))
        inserted '(clojure.core/long 2)
        candidate (history normalization
                           #(#'inline/build-inline-expansion callee [0 inserted 'n] {} inline/subst-syms nil caller))]
    (entry/call-with-definition-inputs
     (fn []
       (is (some? (inline/nested-builder-substitution-view (:source-construction candidate) (:form candidate))))
       (is (some #(identical? inserted %) (tree-seq coll? seq (:body (:form candidate)))))
       (let [before @entry/*emission-inputs*]
         (is (nil? (inline/nested-substitution-call-origin (:source-construction candidate) (:form candidate) inserted)))
         (is (identical? before @entry/*emission-inputs*)))
       ;; This fixture constructs a local substitution candidate only. It must
       ;; not establish that the different original root selected this callee.
       (is (nil? (inline/nested-original-call-frame (:source-construction candidate) (:form candidate))))))
    (let [candidate (history normalization
                             #(#'inline/build-inline-expansion callee [0 inserted inserted] {}
                                                              inline/subst-syms nil caller))]
      (entry/call-with-definition-inputs
       (fn []
         (is (some? (inline/nested-builder-substitution-view (:source-construction candidate) (:form candidate))))
         (is (= 2 (count (filter #(identical? inserted %) (tree-seq coll? seq (:body (:form candidate)))))))
         (let [before @entry/*emission-inputs*]
           (is (nil? (inline/nested-substitution-call-origin (:source-construction candidate) (:form candidate) inserted)))
           (is (identical? before @entry/*emission-inputs*))))))))

(deftest selected-ancestry-search-refuses-ambiguity-cycles-and-broken-ancestors
  ;; Algorithm controls only: producer authenticity and actual selected Methods
  ;; are exercised separately by real source imports below.
  (let [child (Object.) parent (Object.) other (Object.)
        callers {child (Object.) parent (Object.) other (Object.)}
        candidates [child parent other]
        roots (atom #{parent other})
        links (atom #{[parent child]})
        selected? (atom true)
        calls (atom 0)
        selected {:source-arguments [] :call-context {:inputs [] :input-completions []}}
        run #(entry/call-with-definition-inputs
              (fn [] (inline/nested-original-call-frame ::algorithm child)))]
    (with-redefs [transport/recorded-builder-expansions (fn [_] candidates)
                  inline/nested-builder-substitution-view
                  (fn [_ current]
                    {:builder {:caller (get callers current)} :substitution {:argument-slots []}})
                  transport/construction-occurrence
                  (fn [_ caller] (some #(when (identical? caller (get callers %)) {:original caller}) @roots))
                  transport/original-builder-invocation-view (fn [& _] (when @selected? selected))
                  inline/nested-substitution-call-origin
                  (fn [_ candidate caller]
                    (swap! calls inc)
                    (some (fn [[p c]]
                            (when (and (identical? p candidate) (identical? caller (get callers c)))
                              {:parent {:builder {:expansion p}} :origin {:constructed caller}})) @links))
                  transport/imported-builder-invocation-candidate (fn [& _] selected)]
      (is (some? (:parent-frame (run))))
      (reset! links #{[parent child] [other child]})
      (is (nil? (run)) "two parents cannot be resolved by first-match selection")
      (reset! links #{[parent child]})
      (reset! selected? false)
      (is (nil? (run)) "a correctly linked child cannot repair wrong ancestor selection")
      (reset! selected? true)
      (reset! roots #{})
      (reset! links #{[parent child] [child parent]})
      (reset! calls 0)
      (is (nil? (run)))
      (is (<= @calls 12) "cyclic ancestry terminates within its finite search budget")
      (reset! roots #{child})
      (reset! selected? false)
      (reset! calls 0)
      (is (nil? (run)))
      (is (zero? @calls) "a wrongly selected root cannot be reinterpreted as an import")
      (reset! roots #{parent})
      (reset! links #{[parent child]})
      (with-redefs [transport/original-builder-invocation-view
                    (fn [& _] (throw (ex-info "stale ancestor" {:reason :stale-definition-proof-input})))]
        (is (= :stale-definition-proof-input (reason run)))))))

(defn- prepare-original-training-loss! []
  ;; Preparation compiles the ORIGINAL caller only; it executes no model code.
  ;; A promoted definition discards the complete owner and is recaptured below.
  (core/prepare-deftm-frontend! (core/resolve-deftm-var #'training/local-loss))
  (entry/call-with-definition-inputs
   (fn []
     (let [frontend (entry/original-source-frontend (core/resolve-deftm-var #'training/local-loss))]
       (entry/prepare-owned-definition-inputs! core/ensure-current-owned-bytecode!
                                              [(get-in frontend [:definition :receiver-var])])))))

(deftest completed-training-helper-retains-the-whole-cpu-loss
  (prepare-original-training-loss!)
  (let [{:keys [original completed inputs]}
        (entry/call-with-definition-inputs
         (fn []
           (let [frontend (entry/original-source-frontend (core/resolve-deftm-var #'training/local-loss))
                 source (:source-entry frontend)
                 call (first (filter #(= 'raster.dl.nn/linear-nb (descriptor/semantic-op %))
                                     (tree-seq coll? seq (:body frontend))))
                 constructed (transport/construct-original-reference-call frontend call)
                 _ (is (some? constructed))
                 normalized (transport/normalize-frontend-bindings frontend (:params source) constructed)
                 context (bytecode/retained-body-context (:emission frontend))
                 ^java.lang.reflect.Method method (:method context)
                 name (gensym "completed_training_")
                 compiled (bytecode/compile-specialized-class!
                           "raster.compiled.CompletedTrainingLoss"
                           [{:name name :params (mapv #(with-meta %1 {:tag %2}) (:params source) (:tags source))
                             :return-tag (.getReturnType method)
                             :walked-body (:walked-body normalized) :source-ns (:source-ns source)
                             :raster.compiler.backend.jvm.bytecode/body-input-context (:result-context context)}])
                 ^java.lang.reflect.Method completed (get-in compiled [:methods name])]
             (is (some? context))
             (is (= (vec (.getParameterTypes method)) (vec (.getParameterTypes completed))))
             {:original method :completed completed :inputs @entry/*emission-inputs*})))]
    ;; Numerical execution is OUTSIDE the compiler's retryable proof transaction.
    ;; No numerical array is executed to establish shape or residency facts.
    (doseq [iteration [0 1]
            {:keys [rows x y]} (training/batches [1 3])]
      (let [theta (training/initial-theta)
            _ (when (= iteration 1) (aset-float theta 0 (+ 0.5 (aget theta 0))))
            arguments (object-array [theta x y (long rows) (long 17)])
            expected (/ (reduce +
                                (for [row (range rows)]
                                  (let [prediction (reduce +
                                                           (for [column (range 17)]
                                                             (* (double (aget x (+ (* row 17) column)))
                                                                (double (aget theta column)))))
                                        delta (- prediction (double (aget y row)))]
                                    (* delta delta))))
                        rows)]
        (entry/assert-definition-inputs-current! inputs)
        (let [before (.invoke ^java.lang.reflect.Method original nil arguments)
              after (.invoke ^java.lang.reflect.Method completed nil arguments)]
          (is (= (Double/doubleToRawLongBits before) (Double/doubleToRawLongBits after)))
          (is (< (Math/abs (- expected (double after))) 1.0e-12))
          (entry/assert-definition-inputs-current! inputs))))))

(deftest unchanged-training-imports-obey-the-nested-substitution-law
  (prepare-original-training-loss!)
  (let [captured (atom nil) construct transport/with-source-construction-history
        target :ocl:training-substitution-test]
    (with-redefs [transport/with-source-construction-history
                  (fn [normalization run]
                    (let [result (construct normalization run)]
                      (reset! captured (assoc result :normalization normalization))
                      result))]
      (equation/compile #'training/local-loss {:target target :dtype :float
                                             :schedule {:precision :f32-scalar}}
                        {:device-id target :device-type :gpu :backend :ocl
                         :subgroup-dialect :opencl-portable :max-workgroup-size 256}))
    (entry/call-with-definition-inputs
     (fn []
       (let [events (:builder-expansions (phase-view @captured))
             selected (filterv #(contains? '#{raster.dl.nn/linear-nb raster.dl.loss/mse-loss}
                                           (descriptor/semantic-op (:caller %))) events)]
         (is (= '#{raster.dl.nn/linear-nb raster.dl.loss/mse-loss}
                (set (map #(descriptor/semantic-op (:caller %)) selected))))
         (doseq [event selected]
           (let [construction (:source-construction @captured)
                 checked (inline/nested-builder-substitution-view construction (:expansion event))
                 frame (inline/nested-original-call-frame construction (:expansion event))]
             (is (some? checked) (str (descriptor/semantic-op (:caller event))))
             (is (identical? (:expansion event) (get-in checked [:builder :expansion])))
             (is (nil? (inline/nested-builder-substitution-view construction
                                                              (into {} (:expansion event)))))
             (is (not-any? #(contains? checked %)
                           [:executor :completion :return-carrier :resident-result]))
             (is (some? frame))
             (is (= :unclosed (:return-correspondence frame) (:final-placement frame)))
             (is (every? #(= :unclosed (:constructed-correspondence %)) (:argument-slots frame)))
             (is (= (count (:arguments event)) (count (:argument-slots frame))))
             (is (every? #(contains? (:input-completion %) :adaptation) (:argument-slots frame)))
             (let [callee (get-in frame [:original-invocation :builder :callee-import :frontend])
                   bridge (entry/frontend-reference-return-bridge callee)]
               (if (= 'raster.dl.nn/linear-nb (descriptor/semantic-op (:caller event)))
                 (do
                   (is (= :reference-static-return (:kind bridge)))
                   (is (= "[F" (.getName ^Class (:body-return-carrier bridge))))
                   (is (identical? (:implementation callee) (:implementation bridge))))
                 (is (nil? bridge) "MSE's boxed scalar is not a reference-body return recipe")))
             (when (= 'raster.dl.loss/mse-loss (descriptor/semantic-op (:caller event)))
               (let [slot (first (:argument-slots frame))]
                 (is (= 'raster.dl.nn/linear-nb (descriptor/semantic-op (:source-operand slot))))
                 (is (= :checkcast (get-in slot [:input-completion :adaptation])))
                 (is (.isArray ^Class (get-in slot [:input-completion :parameter-carrier])))
                 (is (nil? (:executor slot)))))))
         (let [parent (first (filter #(= 'raster.dl.nn/linear-nb (descriptor/semantic-op (:caller %))) selected))
               body (:walked-body parent)
               imported (transport/body-import-view (:source-import parent) body)
               originals (tree-seq coll? seq (:body (:frontend imported)))
               calls (filter #(contains? '#{raster.arrays/alloc-like raster.numeric/oftype}
                                        (descriptor/semantic-op %)) (tree-seq coll? seq body))]
           (is (= '#{raster.arrays/alloc-like raster.numeric/oftype}
                  (set (map descriptor/semantic-op calls))))
           (doseq [call calls]
             (let [origin (transport/body-import-occurrence (:source-import parent) body call)]
               (is (some? origin))
               (is (identical? (:frontend imported) (:frontend origin)))
               (is (some #(identical? (:original origin) %) originals))
               (is (identical? call (:constructed origin)))))
           (let [children (filter #(contains? '#{raster.arrays/alloc-like raster.numeric/oftype}
                                              (descriptor/semantic-op (:caller %))) events)]
             (is (= 3 (count children)))
             (doseq [child children]
               (let [caller (:caller child)
                     construction (:source-construction @captured)
                     derived (inline/nested-substitution-call-origin construction (:expansion parent) caller)
                     frame (inline/nested-original-call-frame construction (:expansion child))]
                 (is (some? derived))
                 (is (identical? (:frontend imported) (get-in derived [:origin :frontend])))
                 (is (some #(identical? (get-in derived [:origin :original]) %) originals))
                 (is (identical? (:expansion parent) (get-in derived [:parent :builder :expansion])))
                 (if (= 'raster.arrays/alloc-like (descriptor/semantic-op caller))
                   (do
                     (is (some? frame))
                     (is (identical? (:expansion parent)
                                     (get-in frame [:parent-frame :original-invocation :builder :expansion])))
                     (is (identical? caller (get-in frame [:parent-substitution :constructed])))
                     (is (= :unclosed (:return-correspondence frame) (:final-placement frame)))
                     (let [callee (get-in frame [:original-invocation :builder :callee-import :frontend])
                           bridge (entry/frontend-reference-return-bridge callee)]
                       (is (= :reference-static-return (:kind bridge)))
                       (is (= "[F" (.getName ^Class (:body-return-carrier bridge)))))
                     (is (every? #(= :unclosed (:constructed-correspondence %)) (:argument-slots frame)))
                     (is (= :unclosed (get-in frame [:original-invocation :parent-linkage])))
                     ;; Explicit preparation is separate from the frame query.
                     ;; Reuse the original allocator argument's bytecode region,
                     ;; including the selected Long completion, not an algebraic
                     ;; replacement of batch*out-f with the eventual rows value.
                     (let [original (:original-invocation frame)
                           region (entry/prepare-frontend-invocation-input!
                                   (get-in original [:caller-origin :frontend])
                                   (:original-call original) 1)
                           execute (:invoke region)]
                       (is (= '[batch out-f] (:parameters region)))
                       (is (identical? Long/TYPE
                                       (get-in region [:input-completion :parameter-carrier])))
                       (is (= [0 1 7 6]
                              (mapv (fn [[rows columns]] (execute rows columns))
                                    [[0 1] [1 1] [7 1] [3 2]])))
                       (is (thrown? ArithmeticException (execute Long/MAX_VALUE 2)))
                       (is (= :unclosed (:final-placement frame)))
                       (is (nil? (:executor frame))
                           "an executable argument is not permission to replay its enclosing prefix"))
                     (let [before @entry/*emission-inputs*
                           wrong (first (filter #(= 'raster.numeric/oftype (descriptor/semantic-op %)) calls))]
                       (is (nil? (transport/imported-builder-invocation-candidate
                                  construction (:expansion child) (:expansion parent) wrong)))
                       (is (identical? before @entry/*emission-inputs*)))
                     (with-redefs [transport/original-builder-invocation-view (fn [& _] nil)]
                       (let [before @entry/*emission-inputs*]
                         (is (nil? (inline/nested-original-call-frame construction (:expansion child))))
                         (is (identical? before @entry/*emission-inputs*)))))
                   ;; Bare cast constants still lack their independent source
                   ;; substitution proof. A valid parent cannot conceal that.
                   (is (nil? frame)))
                 (is (nil? (inline/nested-substitution-call-origin construction (into {} (:expansion parent)) caller)))
                 (is (nil? (inline/nested-substitution-call-origin construction (:expansion parent)
                                                                  (with-meta (apply list caller) (meta caller))))))))))))))

(deftest selected-call-frames-do-not-outline-or-execute-original-arguments
  (let [{:keys [source-construction normalization] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        owned (builder-view source-construction (:expansion event))
        imported (:callee-import owned)
        callee {:params (:parameters imported) :tags (:tags imported)
                :walked-body (:walked-body imported) :source-import (:source-import event)}
        caller (:normalized (:caller-origin owned)) args (vec (descriptor/call-args caller))
        candidate (history normalization
                           #(#'inline/build-inline-expansion callee args {} inline/subst-syms nil caller))
        forbidden (fn [& _] (throw (AssertionError. "call-frame query prepared or executed an argument")))]
    (with-redefs [entry/prepare-frontend-invocation-input! forbidden
                  bytecode/compile-invocation-input-region forbidden
                  core/ensure-current-owned-bytecode! forbidden]
      (entry/call-with-definition-inputs
       (fn []
         (let [frame (inline/nested-original-call-frame (:source-construction candidate) (:form candidate))
               slot (first (:argument-slots frame))]
           (is (some? frame))
           (is (= [0] (mapv :ordinal (:argument-slots frame))))
           (is (= Long/TYPE (get-in slot [:input-completion :parameter-carrier])))
           (is (= :coerce (get-in slot [:input-completion :adaptation])))
           (is (nil? (:executor frame)))
           (is (= :unclosed (:constructed-correspondence slot)))))))
    (let [foreign (entry/call-with-definition-inputs
                   #(#'inline/try-resolve-walked-body #'shadowed-substitution-helper_m_long))
          foreign-candidate (history normalization
                                     #(#'inline/build-inline-expansion foreign args {} inline/subst-syms nil caller))]
      (entry/call-with-definition-inputs
       (fn []
         (is (some? (inline/nested-builder-substitution-view
                     (:source-construction foreign-candidate) (:form foreign-candidate))))
         ;; Use a fresh baseline after the successful substitution query.
         (let [before @entry/*emission-inputs*]
           (is (nil? (inline/nested-original-call-frame
                      (:source-construction foreign-candidate) (:form foreign-candidate))))
           (is (identical? before @entry/*emission-inputs*))))))))

(deftest actual-source-phase-connects-elaboration-to-the-shared-builder
  (let [{:keys [source-construction compilation] :as captured} (capture)
        view (phase-view captured)
        event (first (:builder-expansions view))
        owned (builder-view source-construction (:expansion event))
        origin (:caller-origin owned)
        original (second (second (first (:body (:frontend origin)))))]
    (is (seq (:type-elaborations view)))
    (is (= 1 (count (:builder-expansions view))))
    (is (identical? original (:original origin)))
    (is (identical? (:caller event) (:constructed origin)))
    (is (= 1 (count (:construction-edges origin))))
    (is (identical? (:caller event) (:constructed (last (:construction-edges origin)))))
    (is (every? true? (map identical? (:arguments event)
                           (descriptor/call-args (:caller event)))))
    (is (some? (:callee-import owned)))
    (is (identical? (:walked-body event) (get-in owned [:callee-import :walked-body])))
    (is (= 'size (:result-symbol owned)))
    (is (identical? (:expansion event) (:expansion owned)))
    (is (nil? (:source-construction compilation)))
    (is (nil? (get-in compilation [:options :source-construction])))
    (is (nil? (get-in compilation [:semantic :attributes :source-construction])))
    (is (= :unclosed-ordered-source-executor
           (reason #(equation/lower compilation [Long/MAX_VALUE 2])))
        "builder provenance is not formal completion or final interval execution permission")))

(deftest construction-queries-require-exact-phase-and-output-identities
  (let [{:keys [source-construction normalization form] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        caller (:caller event) expansion (:expansion event)
        copied-caller (with-meta (apply list caller) (meta caller))
        copied-form (with-meta (apply list form) (meta form))]
    (is (nil? (transport/source-construction-view source-construction normalization form)))
    (is (nil? (transport/construction-occurrence source-construction caller)))
    (is (nil? (transport/builder-expansion-view source-construction expansion)))
    (entry/call-with-definition-inputs
     (fn []
       (is (nil? (transport/source-construction-view {} normalization form)))
       (is (nil? (transport/source-construction-view source-construction {} form)))
       (is (nil? (transport/source-construction-view source-construction normalization copied-form)))
       (is (nil? (transport/construction-occurrence source-construction copied-caller)))
       (is (nil? (transport/builder-expansion-view source-construction (into {} expansion))))))
    (let [forbidden (fn [& _] (throw (AssertionError. "query reconstructed source")))]
      (with-redefs-fn {#'pipeline/tag-expr-types forbidden
                      #'util/uniquify-rebindings forbidden
                      #'inline/build-inline-expansion forbidden}
        #(do (is (some? (phase-view captured)))
             (is (some? (origin-view source-construction caller)))
             (is (some? (builder-view source-construction expansion))))))
    (is (= {:form :plain} (transport/with-source-construction-history normalization (constantly :plain))))))

(deftest declined-composed-frame-queries-do-not-publish-partial-history
  (let [{:keys [source-construction normalization] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        imported (:callee-import (builder-view source-construction (:expansion event)))
        callee {:params (:parameters imported) :tags (:tags imported)
                :walked-body (:walked-body imported) :source-import (:source-import event)}
        caller (:normalized (:caller-origin (builder-view source-construction (:expansion event))))
        args (vec (descriptor/call-args caller))
        copied-args [(with-meta (apply list (first args)) (meta (first args)))]
        changed-args (history normalization
                              #(#'inline/build-inline-expansion callee copied-args {} inline/subst-syms nil caller))
        faulty-law (history normalization
                            #(#'inline/build-inline-expansion callee args {} (fn [_ _] 8) nil caller))]
    (doseq [query [#(transport/builder-expansion-view (:source-construction changed-args) (:form changed-args))
                   #(inline/nested-builder-substitution-view (:source-construction faulty-law) (:form faulty-law))
                   #(inline/nested-expansion-substitution-view callee args (:form faulty-law))]]
      (entry/call-with-definition-inputs
       (fn []
         (let [before @entry/*emission-inputs*]
           (is (nil? (query)))
           (is (identical? before @entry/*emission-inputs*))))))))

(deftest construction-chains-do-not-grant-type-or-completion-permission
  (let [{:keys [source-construction normalization] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        original (:normalized (:caller-origin (builder-view source-construction (:expansion event))))
        changed (with-meta original (assoc (meta original) :tag 'double))
        result (history normalization
                        #(do (transport/observe-type-elaboration! original changed {'rows 'long})
                             changed))
        origin (origin-view (:source-construction result) changed)]
    (is (some? origin) "the observed origin survives even an incompatible tag change")
    (is (= 'double (:tag (meta (:constructed origin)))))
    (is (not-any? #(contains? origin %) [:completion :result-carrier :identity-cast :executor]))
    (is (= 1 (count (:construction-edges origin))))
    (let [result (history normalization
                          #(do (transport/observe-type-elaboration! original original {}) original))]
      (is (some? (origin-view (:source-construction result) original)))
      (is (empty? (:type-elaborations
                   (entry/call-with-definition-inputs
                    #(transport/source-construction-view (:source-construction result)
                                                         normalization original))))))
    (doseq [mode [:duplicate :fan-out :fan-in :cycle]]
      (let [other (with-meta (apply list original) {:tag 'long})
            result (history normalization
                            #(do
                               (transport/observe-type-elaboration! original changed {})
                               (case mode
                                 :duplicate (transport/observe-type-elaboration! original changed {})
                                 :fan-out (transport/observe-type-elaboration! original other {})
                                 :fan-in (transport/observe-type-elaboration! other changed {})
                                 :cycle (transport/observe-type-elaboration! changed original {}))
                               changed))]
        (is (nil? (origin-view (:source-construction result) changed)) (str mode))))))

(deftest builder-arguments-cannot-be-replaced-by-equal-or-rewritten-inputs
  (let [{:keys [source-construction normalization] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        imported (:callee-import (builder-view source-construction (:expansion event)))
        callee {:params (:parameters imported) :tags (:tags imported)
                :walked-body (:walked-body imported) :source-import (:source-import event)}
        caller (:normalized (:caller-origin (builder-view source-construction (:expansion event))))
        arguments (vec (descriptor/call-args caller))
        build @#'inline/build-inline-expansion]
    (doseq [mode [:original :copied-argument :missing-argument :repeated-caller]]
      (let [actual (case mode
                     :copied-argument [(with-meta (apply list (first arguments)) (meta (first arguments)))]
                     :missing-argument []
                     arguments)
            result (history normalization
                            #(if (= mode :missing-argument)
                               ;; Model an independently transformed builder input; observation is
                               ;; provenance only and the query must reject the lost argument.
                               (let [expansion {:argument-bindings [] :bindings []}]
                                 (transport/observe-builder-expansion! caller callee actual {} 'result expansion)
                                 expansion)
                               (let [expansion (build callee actual {} inline/subst-syms 'result caller)]
                                 (when (= mode :repeated-caller)
                                   (build callee actual {} inline/subst-syms 'other caller))
                                 expansion)))
            view (builder-view (:source-construction result) (:form result))]
        (if (= mode :original)
          (is (some? view))
          (is (nil? view) (str mode)))))))

(deftest construction-history-keeps-original-frontend-and-callee-guards
  (let [{:keys [source-construction] :as captured} (capture)
        event (first (:builder-expansions (phase-view captured)))
        owned (builder-view source-construction (:expansion event))
        caller (get-in owned [:caller-origin :frontend :source-entry :source-var])
        callee (get-in owned [:callee-import :frontend :source-entry :source-var])]
    (doseq [source [caller callee]]
      (let [metadata (meta source)
            body (:raster.core/deftm-source-body metadata)]
        (try
          (alter-meta! source assoc :raster.core/deftm-source-body
                       (with-meta (mapv identity body) (meta body)))
          (is (= :stale-definition-proof-input (reason #(phase-view captured))))
          (is (= :stale-definition-proof-input
                 (reason #(builder-view source-construction (:expansion event)))))
          (finally (reset-meta! source metadata)))))))

(deftest recursive-builder-observes-caller-before-rewriting-arguments
  (recursive-argument-prefix 2)
  (let [normalized
        (entry/call-with-definition-inputs
         (fn []
           (let [imported (#'inline/try-resolve-walked-body #'recursive-argument-prefix_m_long)
                 frontend (:frontend (transport/body-import-view (:source-import imported)
                                                                 (:walked-body imported)))]
             (transport/normalize-frontend-bindings frontend '[n]))))
        caller (second (second (first (:walked-body normalized))))
        nested (first (descriptor/call-args caller))]
    (doseq [rewrite-argument? [false true]]
      (let [actual-caller (if rewrite-argument? caller nested)
            result (history (:source-normalization normalized)
                            #(#'inline/inline-invk actual-caller))
            input (:source-construction result)
            phase (entry/call-with-definition-inputs
                   #(transport/source-construction-view input (:source-normalization normalized)
                                                        (:form result)))
            events (:builder-expansions phase)
            outer (first (filter #(identical? actual-caller (:caller %)) events))
            view (builder-view input (:expansion outer))]
        (is (= 7 (:form result)))
        (is (some? outer))
        (if rewrite-argument?
          (do (is (= 2 (count events)))
              (is (nil? view) "rewritten arguments cannot borrow the original call's occurrences")
              (is (some? (builder-view input (:expansion (first events))))))
          (do (is (= 1 (count events)))
              (is (some? view))))))))
