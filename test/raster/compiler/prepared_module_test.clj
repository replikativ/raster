(ns raster.compiler.prepared-module-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.fixtures.distributed-training :as training]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.prepared-module :as module]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defn- source-frontend []
  (let [root (core/resolve-deftm-var #'training/local-gradient-sum)]
    (core/prepare-deftm-frontend! root)
    (entry/original-source-frontend root)))

(defn- capture [] (module/capture (source-frontend) {:dtype :float}))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(core/deftm ordinary-source [n :- Long] :- Long (clojure.core/+ n 1))

(core/deftm ordinary-multiform-source [out :- (Array long) n :- Long] :- Long
  (clojure.core/aset out 0 n)
  (clojure.core/aset out 1 (clojure.core/inc (clojure.core/aget out 0)))
  (clojure.core/+ (clojure.core/aget out 1) 1))

(core/deftm two-gradient-imports
  [theta :- (Array float) x :- (Array float) y :- (Array float)
   rows :- Long width :- Long] :- (Array float)
  (let [first-vg ((reverse/value+grad (var raster.compiler.fixtures.distributed-training/local-loss)
                                   :wrt [0]) theta x y rows width)
        first-gradient (clojure.core/nth first-vg 1)
        second-vg ((reverse/value+grad (var raster.compiler.fixtures.distributed-training/local-loss)
                                    :wrt [0]) theta x y rows width)
        second-gradient (clojure.core/nth second-vg 1)]
    (raster.dl.nn/residual-add first-gradient second-gradient width)))

(deftest two-late-imports-retain-distinct-occurrence-and-obligation-identities
  (entry/call-with-definition-inputs
   (fn []
     (let [root (core/resolve-deftm-var #'two-gradient-imports)
           _ (core/prepare-deftm-frontend! root)
           phase (module/capture (entry/original-source-frontend root) {:dtype :float})
           view (module/view phase (module/normalized-source phase))
           imports (:imports view)
           obligations (mapcat :obligation-origins (:import-origins view))]
       (is (= 2 (count imports)))
       (is (true? (every? #(identical? (:physical %)
                                      (reverse/prepared-bytecode-entry (:prepared %))) imports)))
       (is (not (identical? (:nonce (first imports)) (:nonce (second imports)))))
       (is (= (count obligations) (count (set (map :id obligations)))))
       (is (= (count (:origins view)) (count (set (map :index (:origins view))))))
       (is (= 2 (count (:unclosed-source-imports @entry/*emission-inputs*))))
       (is (false? (:admitted? view)))))))

(deftest ordinary-multiform-source-keeps-every-statement-in-order
  (entry/call-with-definition-inputs
   (fn []
     (let [root (core/resolve-deftm-var #'ordinary-multiform-source)
           _ (core/prepare-deftm-frontend! root)
           frontend (entry/original-source-frontend root)
           result (module/construct-source frontend {:dtype :double})
           params (get-in frontend [:source-entry :params])
           compiled (binding [*ns* (get-in frontend [:source-entry :source-ns])]
                      (eval (list 'fn params (:frontend-source result))))
           expected (long-array 2)
           actual (long-array 2)]
       (is (nil? (:prepared-phase result)))
       (is (= (ordinary-multiform-source expected 11) (compiled actual 11)))
       (is (= [11 12] (vec actual)))
       (is (= (vec expected) (vec actual)))))))

(deftest ordinary-source-shares-construction-without-an-import-owner
  (entry/call-with-definition-inputs
   (fn []
     (let [root (core/resolve-deftm-var #'ordinary-source)
           _ (core/prepare-deftm-frontend! root)
           frontend (entry/original-source-frontend root)
           events (atom [])
           normalize frontend/normalize-source
           result
           (with-redefs [entry/frontend-body-entry-carriers
                         (fn [& _] (throw (AssertionError. "ordinary source queried AD entry carriers")))
                         frontend/normalize-source
                         (fn [source options observer]
                           (swap! events conj [:normalize source])
                           (normalize source options observer))]
             (module/construct-source
              frontend {:dtype :double}
              {:prepare-source (fn [source] (swap! events conj [:prepare source]) source)
               :represent-source
               (fn [source options]
                 (swap! events conj [:represent source])
                 ;; A real representation step may rebuild an ordinary source.
                 {:body (with-meta (apply list source) (meta source)) :options options})}))]
       (is (= [:prepare :represent :normalize] (mapv first @events)))
       (is (nil? (:prepared-phase result)))
       (is (not (identical? (:represented-source result) (:physical-source result))))
       (is (identical? (:physical-source result) (second (last @events))))
       (is (some? (transport/source-construction-view
                   (:construction result) (:normalization result) (:represented-source result))))
       (is (empty? (:unclosed-source-imports @entry/*emission-inputs*)))))))

(deftest unclosed-import-cannot-borrow-a-different-representation
  (doseq [represent [(fn [source options]
                      {:body (with-meta (apply list source) (meta source)) :options options})
                    (fn [source options] {:body source :options (assoc options :dtype :double)})
                    (fn [source options] {:body source :options (into {} options)})
                    (fn [source options] {:body source :options options :representation {:forged true}})]]
    (entry/call-with-definition-inputs
     (fn []
       (is (= :unclosed-prepared-module-phase
              (reason #(module/construct-source (source-frontend) {:dtype :float}
                                                {:represent-source represent}))))))))

(deftest actual-training-module-retains-complete-pending-construction
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           view (module/view phase source)
           origins (:origins view)
           imports (:imports view)
           {:keys [event call physical]} (first imports)
           grouped (group-by :source-index origins)
           split (first (filter #(some (fn [origin] (= :continuation (:lift-role origin))) (val %)) grouped))
           joined (first (:import-origins view))]
       (is (= 1 (count imports)))
       (is (false? (:admitted? view)))
       (is (false? (:admitted? call)))
       (is (identical? physical (reverse/prepared-bytecode-entry (:prepared event))))
       (is (= (quot (count (second source)) 2) (count origins)))
       (is (= (vec (range (count origins))) (mapv :index origins)))
       (is (= #{:reduction :continuation} (set (map :lift-role (val split)))))
       (is (= 5 (count (:input-obligations call))))
       (is (identical? call (:call joined)))
       (is (every? true?
                    (map (fn [{:keys [call-binding-index id expression]}]
                           (let [[actual-id actual-expression]
                                 (nth (:bindings call) call-binding-index)]
                             (and (identical? id actual-id)
                                  (identical? expression actual-expression))))
                         (:root-bindings call))))
       (is (identical? (:import (first imports)) (:import joined)))
       (is (= (count (:bindings call)) (count (:binding-origins joined))))
       (is (= (+ (count (:input-obligations call))
                 (count (get-in call [:proposal :completion-obligations]))
                 (reduce + 0 (map #(count (get-in % [:argument-construction :arguments]))
                                  (:root-bindings call))))
              (count (:obligation-origins joined))))
       (is (every? true? (map identical?
                              (concat (:input-obligations call) (get-in call [:proposal :completion-obligations]))
                              (map :obligation (:obligation-origins joined)))))
       (is (apply distinct? (map :id (:obligation-origins joined))))
       (is (every? seq (map :binding-origins (:obligation-origins joined))))
       (is (some #(= #{:reduction :continuation} (set (map :lift-role (:origins %))))
                 (:binding-origins joined))
           "one original module binding joins both reduction and continuation, including derived extents")
       (is (every? #(= :pending (:status %)) (:input-obligations call)))
       (is (seq (get-in call [:proposal :completion-obligations])))
       (is (every? #(= :pending (:status %)) (get-in call [:proposal :completion-obligations])))
       (is (some #(and (= :prepared-reverse-ad (first %))
                       (identical? (:occurrence event) (nth % 2 nil)))
                 (:unclosed-source-imports @entry/*emission-inputs*)))
       (is (some? (frontend/coverage-decline source (:options view)))
           "owned construction alone does not admit generated scalar/storage obligations")
       (doseq [{:keys [index binding]} origins]
         (is (every? true? (map identical? binding (take 2 (drop (* 2 index) (second source)))))))))))

(deftest exact-phase-queries-never-renormalize-or-refresh-owner-evidence
  (let [[phase source]
        (entry/call-with-definition-inputs
         (fn [] (let [phase (capture)] [phase (module/normalized-source phase)])))
        copied (with-meta (apply list source) (meta source))]
    (is (nil? (module/normalized-source phase)))
    (is (nil? (module/view phase source)))
    (entry/call-with-definition-inputs
     (fn []
       (is (nil? (module/view {} source)))
       (is (nil? (module/view phase copied)))
       (with-redefs [frontend/normalize-source (fn [& _] (throw (AssertionError. "query normalized again")))]
         (is (some? (module/view phase source))))
       (let [query reverse/prepared-bytecode-entry]
         (with-redefs [reverse/prepared-bytecode-entry #(into {} (query %))]
           (is (= :unclosed-prepared-module-phase (reason #(module/view phase source))))))))))

(deftest phase-rejects-missing-or-tampered-constructor-histories
  (doseq [edit [(fn [event] (when-not (= :normalize-binding-spine (:stage event)) event))
                (fn [event] (if (= :normalize-binding-spine (:stage event))
                              (assoc-in event [:entries 0 :input-index] 99) event))
                (fn [event] (if (= :normalize-binding-spine (:stage event))
                              (assoc-in event [:entries 0 :outputs 0 :index] 99) event))
                (fn [event] (if (= :reduction-split (:stage event))
                              (assoc event :output-binding-index 99) event))]]
    (let [normalize frontend/normalize-source]
      (entry/call-with-definition-inputs
       (fn []
         (with-redefs [frontend/normalize-source
                       (fn [source options observer]
                         (normalize source options #(when-let [event (edit %)] (observer event))))]
           (is (= :unclosed-prepared-module-phase (reason capture))))))))
  (let [query transport/source-construction-view]
    (entry/call-with-definition-inputs
     (fn []
       (with-redefs [transport/source-construction-view
                     (fn [& args] (assoc (apply query args) :prepared-program-cuts []))]
         (is (= :unclosed-prepared-module-phase (reason capture))))))))

(deftest construction-refuses-without-a-definition-owner
  (is (= :unclosed-prepared-module-phase (reason #(module/capture {} {})))))

(deftest preparation-cannot-insert-the-generated-spine-before-the-final-cut
  (entry/call-with-definition-inputs
   (fn []
     (let [source (source-frontend)
           parameters (get-in source [:source-entry :params])
           environment (zipmap parameters (get-in source [:source-entry :tags]))
           project reverse/project-prepared-value+grad-call
           prepare reverse/prepare-value+grad
           preparation-policies (atom [])
           projections (atom 0)
           preparations (atom 0)
           phase
           (with-redefs [reverse/prepare-value+grad
                         (fn [& args]
                           (swap! preparation-policies conj
                                  [(nil? transport/*prepared-program-consumer*)
                                   (false? inline/*defer-prepared-ad-expansion?*)])
                           (apply prepare args))
                         reverse/project-prepared-value+grad-call
                         (fn [& args]
                           (swap! projections inc)
                           (apply project args))]
             (module/capture
              source {:dtype :float}
              (fn [form]
                (swap! preparations inc)
                (is (nil? transport/*prepared-program-consumer*))
                (is inline/*defer-prepared-ad-expansion?*)
                ;; Exercise the real AD recognizer in the earlier phase. It
                ;; must retain the application, not mint an untracked import.
                (let [result (inline/inline-ad-applications form environment)]
                  (is (zero? @projections))
                  result))))
           normalized (module/normalized-source phase)
           view (module/view phase normalized)]
       (is (= 1 @preparations))
       (is (= 1 @projections))
       (is (seq @preparation-policies))
       (is (every? #(= [true true] %) @preparation-policies)
           "nested preparation cannot inherit either outer insertion or deferral policy")
       (is (= 1 (count (:imports view))))
       (is (= 1 (count (:prepared-program-cuts
                        (transport/source-construction-view
                         (:source-construction view) (:normalization view) (:source view))))))
       (is (false? (:admitted? view)))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest real-pre-soa-preparation-retains-the-original-ad-call-occurrence
  (entry/call-with-definition-inputs
   (fn []
     (let [frontend (source-frontend)
           options {:dtype :float :target :opencl
                    :source-ns (get-in frontend [:source-entry :source-ns])
                    :param-env (zipmap (get-in frontend [:source-entry :params])
                                       (get-in frontend [:source-entry :tags]))}
           phase (module/capture
                  frontend options
                  #(pipeline/run-passes % pipeline/gpu-resident-pre-soa-passes options))
           normalized (module/normalized-source phase)
           view (module/view phase normalized)
           construction (transport/source-construction-view
                         (:source-construction view) (:normalization view) (:source view))
           cut (first (:prepared-program-cuts construction))
           origin (get-in cut [:event :caller-origin])]
       (is (= 1 (count (:imports view))))
       (is (= 1 (count (:prepared-program-cuts construction))))
       (is (identical? frontend (:frontend origin)))
       (is (identical? (get-in cut [:event :caller]) (:constructed origin)))
       (is (seq (:construction-edges origin))
           "actual reconstruction edges, not a structural search, join the late call")
       (is (= (count (:origins view)) (quot (count (second normalized)) 2)))
       (is (false? (:admitted? view)))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest ambiguous-rewalk-origins-cannot-mint-a-late-import
  (entry/call-with-definition-inputs
   (fn []
     (let [frontend (source-frontend)
           options {:dtype :float :target :opencl
                    :source-ns (get-in frontend [:source-entry :source-ns])
                    :param-env (zipmap (get-in frontend [:source-entry :params])
                                       (get-in frontend [:source-entry :tags]))}
           observe transport/observe-construction!
           duplicated (atom 0)]
       (with-redefs [transport/observe-construction!
                     (fn [original constructed environment]
                       (observe original constructed environment)
                       (when (and (not (identical? original constructed))
                                  (seq? original) (seq? (first original))
                                  (= 'raster.ad.reverse/value+grad (ffirst original)))
                         (swap! duplicated inc)
                         (observe original constructed environment)))]
         (is (= :unclosed-prepared-program-placement
                (reason #(module/capture
                          frontend options
                          (fn [form]
                            (pipeline/run-passes form pipeline/gpu-resident-pre-soa-passes options)))))))
       (is (pos? @duplicated))
       (is (empty? (:unclosed-source-imports @entry/*emission-inputs*))
           "failed local construction does not publish a pending import to its parent")))))

(deftest construction-retains-compatible-declarations-and-refuses-conflicting-options
  (entry/call-with-definition-inputs
   (fn []
     (let [frontend (source-frontend)
           phase (module/capture frontend {:dtype :float :array-types {'extra :float64 'theta :float32}})
           view (module/view phase (module/normalized-source phase))]
       (is (= :double (get-in view [:options :array-types 'extra])))
       (is (= :float (get-in view [:options :array-types 'theta])))
       (doseq [options [{:array-types {'theta :double}}
                        {:scalar-types {'theta :float}}
                        {:scalar-types {'rows :double}}
                        {:array-types {'extra :not-a-dtype}}
                        {:source-ns (the-ns 'clojure.core)}
                        {:public-parameters '[unrelated]}]]
         (is (= :unclosed-prepared-module-phase (reason #(module/capture frontend options)))))))))

(deftest a-count-preserving-reorder-after-insertion-cannot-borrow-the-cut
  (let [expand inline/inline-ad-applications]
    (entry/call-with-definition-inputs
     (fn []
       (with-redefs [inline/inline-ad-applications
                     (fn [& args]
                       (let [[head bindings & body :as source] (apply expand args)
                             ;; Same number of bindings and same minted symbols,
                             ;; but no longer the actual ordered cut interval.
                             reordered (into (subvec bindings 2 4)
                                             (into (subvec bindings 0 2) (subvec bindings 4)))]
                         (with-meta (list* head reordered body) (meta source))))]
         (is (= :unclosed-prepared-module-phase (reason capture))))))))

(deftest a-generator-layout-change-cannot-reassign-existing-obligations
  (let [construct generated/instantiate-value-source]
    (entry/call-with-definition-inputs
     (fn []
       (with-redefs [generated/instantiate-value-source
                     (fn [& args]
                       (let [call (apply construct args) bindings (:bindings call)]
                         (assoc call :bindings (assoc bindings 0 (nth bindings 1) 1 (nth bindings 0)))))]
         (is (= :unclosed-prepared-module-phase (reason capture))))))))

(deftest root-index-mappings-cannot-redirect-completion-obligations
  (let [construct generated/instantiate-value-source]
    (doseq [alter [(fn [call] (update-in call [:root-bindings 0] dissoc :call-binding-index))
                  (fn [call] (assoc-in call [:root-bindings 0 :call-binding-index] 0))
                  (fn [call] (update-in call [:root-bindings 0 :call-binding-index] inc))
                  (fn [call] (assoc-in call [:root-bindings 0 :root-index] 1))]]
      (entry/call-with-definition-inputs
       (fn []
         (with-redefs [generated/instantiate-value-source
                       (fn [& args] (alter (apply construct args)))]
           (is (= :unclosed-prepared-module-phase (reason capture)))))))))

(deftest allocation-inputs-retain-actual-completion-order-and-pending-obligations
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           view (module/view phase (module/normalized-source phase))
           call (get-in view [:imports 0 :call])
           roots (filter :argument-construction (:root-bindings call))
           obligations (get-in view [:import-origins 0 :obligation-origins])]
       (is (= 3 (count roots)))
       (is (= 6 (count (filter #(= :argument (second (:id %))) obligations))))
       (doseq [root roots
               :let [{:keys [arguments observed]} (:argument-construction root)
                     [reference size] arguments]]
         (is (= [:checkcast :coerce] (mapv #(get-in % [:completion :adaptation]) arguments)))
         (is (= [:ref :ref] (mapv #(get-in % [:completion :emitted-type]) arguments)))
         (is (= [:jvm-bytecode-reference-input :jvm-bytecode-integral-input]
                (mapv #(get-in % [:contract :source-context]) arguments)))
         (is (= [(- (:call-binding-index root) 2) (dec (:call-binding-index root))]
                (mapv :call-binding-index arguments)))
         (is (every? #(= :pending (:status %)) arguments))
         (is (every? true?
                      (map (fn [argument]
                             (every? true? (map identical? (:binding argument)
                                                (nth (:bindings call) (:call-binding-index argument)))))
                           arguments)))
         (is (identical? (:operand size)
                         (:operand (raster.compiler.core.value-completion/canonical-parts
                                    (second (:binding size))))))
         (is (identical? Long/TYPE (get-in size [:completion :parameter-carrier])))
         (is (= :global-typed-indy (:invocation-mode observed))))
       (is (false? (:admitted? view)))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest malformed-argument-manifests-cannot-redirect-completions
  (let [construct generated/instantiate-value-source]
    (doseq [change [(fn [argument] (assoc argument :call-binding-index 0))
                    (fn [argument] (assoc argument :input-index 1))
                    (fn [argument] (assoc argument :template-source-path [1 1]))
                    (fn [argument] (assoc argument :status :complete))
                    (fn [argument] (assoc-in argument [:contract :expected-input-carrier] :long))]]
      (entry/call-with-definition-inputs
       (fn []
         (with-redefs [generated/instantiate-value-source
                       (fn [& args]
                         (let [call (apply construct args)
                               root-index (first (keep-indexed
                                                  #(when (:argument-construction %2) %1)
                                                  (:root-bindings call)))]
                           (update-in call [:root-bindings root-index :argument-construction :arguments 0]
                                      change)))]
           (is (= :unclosed-prepared-module-phase (reason capture)))))))))

(deftest allocation-staging-requires-the-original-invocation-observation
  (entry/call-with-definition-inputs
   (fn []
     (with-redefs [bytecode/retained-input-completed-invocation-context (fn [& _] nil)]
       (is (= :generated-body-unsupported (reason capture)))))))

(deftest allocation-proposal-cannot-replace-the-prepared-physical-entry
  (let [construct generated/instantiate-value-source]
    (entry/call-with-definition-inputs
     (fn []
       (with-redefs [generated/instantiate-value-source
                     (fn [& args]
                       (update-in (apply construct args) [:proposal :physical] #(into {} %)))]
         (is (= :unclosed-prepared-module-phase (reason capture))))))))
