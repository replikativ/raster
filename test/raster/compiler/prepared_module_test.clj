(ns raster.compiler.prepared-module-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.fixtures.distributed-training :as training]
            [raster.compiler.equation-first :as equation-first]
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

(core/deftm hidden-gradient
  [theta :- (Array float) x :- (Array float) y :- (Array float)
   rows :- Long width :- Long] :- (Array float)
  (let [vg ((reverse/value+grad (var raster.compiler.fixtures.distributed-training/local-loss)
                              :wrt [0]) theta x y rows width)]
    (clojure.core/nth vg 1)))

(core/deftm authored-with-hidden-gradient
  [theta :- (Array float) x :- (Array float) y :- (Array float)
   rows :- Long width :- Long] :- (Array float)
  (let [vg ((reverse/value+grad (var raster.compiler.fixtures.distributed-training/local-loss)
                              :wrt [0]) theta x y rows width)
        first-gradient (clojure.core/nth vg 1)
        second-gradient (hidden-gradient theta x y rows width)]
    (raster.dl.nn/residual-add first-gradient second-gradient width)))

(deftest public-compilation-reaches-the-owned-import-boundary
  (doseq [[f imports] [[#'training/local-gradient-sum 1] [#'two-gradient-imports 2]]]
    (let [construct module/construct-source
          analyze module/analyze-source-components
          calls (atom {:construction 0 :analysis 0})
          failure
          (with-redefs [module/construct-source
                        (fn [& args]
                          (let [result (apply construct args)]
                            ;; Cold definition loading may retry the complete
                            ;; owner transaction. Count only completed handoffs.
                            (swap! calls update :construction inc)
                            result))
                        module/analyze-source-components
                        (fn [& args]
                          (swap! calls update :analysis inc)
                          (swap! calls assoc :markers
                                 (count (:unclosed-source-imports @entry/*emission-inputs*)))
                          (apply analyze args))
                        pipeline/normalized-semantic-program-attempt
                        (fn [& _]
                          (throw (ex-info "pending imports entered ordinary admission" {})))]
            (try
              (equation-first/compile
               f {:dtype :float :target :ocl:prepared-production-boundary
                  :schedule {:precision :f32-scalar}}
               {:device-id :ocl:prepared-production-boundary :device-type :gpu :backend :ocl
                :subgroup-dialect :opencl-portable :max-workgroup-size 256})
              nil
              (catch clojure.lang.ExceptionInfo error (ex-data error))))]
      (is (= {:reason :unclosed-prepared-module-phase :compiler :equation-first
              :pending-inputs (* 6 imports) :allocations (* 3 imports) :fallback :none}
             (select-keys failure [:reason :compiler :pending-inputs :allocations :fallback])))
      (is (= {:construction 1 :analysis 1 :markers imports} @calls)))))

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
       (let [inputs (keep #(module/input-occurrence phase (:normalized view) (:binding %)) (:origins view))]
         (is (= 12 (count inputs)))
         (is (= #{(:import (first imports)) (:import (second imports))} (set (map :import inputs)))))
       (let [components (module/analyze-source-components phase (:normalized view))
             pending (:pending-inputs components)]
         (is (= 6 (count (:allocations components))))
         (is (= 12 (count pending) (count (set (map #(get-in % [:obligation :id]) pending)))))
         (is (nil? (#'frontend/continue-source-components* components))))
       (is (false? (:admitted? view)))))))

(deftest authored-layout-preserves-every-cut-and-uses-emitted-output-carriers
  (doseq [[source expected-imports] [[#'training/local-gradient-sum 1] [#'two-gradient-imports 2]]]
    (entry/call-with-definition-inputs
     (fn []
       (let [root (core/resolve-deftm-var source)
             _ (core/prepare-deftm-frontend! root)
             phase (module/capture (entry/original-source-frontend root) {:dtype :float})
             normalized (module/normalized-source phase)
             view (module/view phase normalized)
             layout (module/authored-layout phase normalized)
             original-pairs (partition 2 (second (:source view)))
             rebuilt-pairs (mapcat :bindings layout)
             imports (filter #(= :import (:kind %)) layout)
             authored (filter #(= :authored (:kind %)) layout)]
         (is (= (inc (* 2 expected-imports)) (count layout)))
         (is (= expected-imports (count imports)))
         (is (= (count original-pairs) (count rebuilt-pairs)))
         (is (every? true? (map (fn [a b] (every? true? (map identical? a b))) original-pairs rebuilt-pairs)))
         (is (every? #(= (:live-outs %) (mapv first (:bindings %))) authored))
         (is (= (vec (drop 2 (:source view))) (:tail (last authored))))
         (doseq [[stage import] (map vector imports (:imports view))]
           (is (identical? import (:import stage)))
           (is (identical? (:bindings (:call import)) (:bindings stage)))
           (is (= '[double floats] (mapv :tag (remove nil? (:outputs stage)))))
           (let [array-output (second (:outputs stage))]
             (is (identical? (class (float-array 0))
                            (get-in array-output [:binding-context :after :reference-class])))
             (is (not-any? #(contains? array-output %) [:presence :shape :capacity :admitted?]))))
         (is (nil? (module/authored-layout phase (apply list normalized))))
         (is (= expected-imports (count (:unclosed-source-imports @entry/*emission-inputs*)))))))))

(deftest authored-layout-refuses-damaged-intervals-and-output-carriers
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           normalized (module/normalized-source phase)
           {:keys [source imports]} (module/view phase normalized)
           import (first imports)
           layout #'module/authored-source-layout
           query bytecode/retained-input-binding-context]
       (doseq [changed [[import import]
                        [(update-in import [:call :bindings] subvec 1)]
                        [(assoc-in import [:placement :binding-site :prefix] [['wrong 1]])]
                        [(update-in import [:call :outputs] (comp vec reverse))]]]
         (is (= :unclosed-prepared-module-phase (reason #(layout source changed)))))
       (doseq [change [(fn [record] (assoc-in record [:after :reference-class] Object))
                       (fn [record] (assoc-in record [:after :verified] false))]]
         (with-redefs [bytecode/retained-input-binding-context
                       (fn [& args]
                         (let [record (apply query args)]
                           (if (identical? (class (float-array 0)) (get-in record [:after :reference-class]))
                             (change record) record)))]
           (is (= :unclosed-prepared-module-phase
                  (reason #(module/authored-layout phase normalized))))))
       (doseq [change [(constantly nil)
                       #(assoc % :binder 'unrelated)
                       #(assoc-in % [:after :reference-class] String)]]
         (with-redefs [bytecode/retained-input-binding-context (fn [& args] (change (apply query args)))]
           (is (= :unclosed-prepared-module-phase
                  (reason #(module/authored-layout phase normalized))))))
       (let [scalar (first (get-in import [:call :outputs]))
             scalar-root (first (filter #(= scalar (:id %)) (get-in import [:call :root-bindings])))]
         (doseq [edit [#(assoc-in % [:call :proposal :normal-return-facts (:original-id scalar-root) :carrier] Double/TYPE)
                       #(assoc-in % [:call :proposal :normal-return-facts (:original-id scalar-root) :builtin-class] String)]]
           (is (= :unclosed-prepared-module-phase
                  (reason #(layout source [(edit import)]))))))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest actual-authored-training-consumer-prepares-without-rewriting-its-import
  (entry/call-with-definition-inputs
   (fn []
     (let [original-frontend (source-frontend)
           phase (module/capture original-frontend {:dtype :float :segmented-plans? true})
           normalized (module/normalized-source phase)
           view (module/view phase normalized)
           layout (module/authored-layout phase normalized)
           imported (first (filter #(= :import (:kind %)) layout))
           authored (last layout)
           tags (into {} (keep #(when % [(:output %) (:tag %)])) (:outputs imported))
           environment (merge (zipmap (get-in original-frontend [:source-entry :params])
                                      (get-in original-frontend [:source-entry :tags])) tags)
           source (list* 'let* (vec (mapcat identity (:bindings authored))) (:tail authored))
           prepared (pipeline/run-passes
                     source pipeline/gpu-resident-pre-soa-passes
                     (assoc (:options view) :target-device :ocl:authored-training-consumer
                                            :param-env environment))
           census (last @pipeline/fixpoint-census)]
       (is (= :final (:label census)))
       (is (zero? (:non-exempt-untagged census)))
       (is (zero? (:undevirtualized census)))
       (is (some #(and (seq? %) (= '.invk (first %))
                       (= 'raster.numeric/_star__m_float_float-impl (second %)))
                 (tree-seq coll? seq prepared)))
       (is (identical? (get-in imported [:import :call :bindings]) (:bindings imported)))
       (is (identical? normalized (module/normalized-source phase)))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest authored-preparation-assembles-one-and-two-imports-without-rewriting-them
  (doseq [[source expected-imports] [[#'training/local-gradient-sum 1] [#'two-gradient-imports 2]]]
    (entry/call-with-definition-inputs
     (fn []
       (let [root (core/resolve-deftm-var source)
             _ (core/prepare-deftm-frontend! root)
             frontend (entry/original-source-frontend root)
             phase (module/capture frontend {:dtype :float})
             normalized (module/normalized-source phase)
             view (module/view phase normalized)
             layout (module/authored-layout phase normalized)
             tags (zipmap (get-in frontend [:source-entry :params]) (get-in frontend [:source-entry :tags]))
             imported-ids (into #{} (mapcat #(map first (:bindings %)))
                                (filter #(= :import (:kind %)) layout))
             calls (atom [])
             result (#'module/prepare-authored-layout
                     layout tags (assoc (:options view) :target-device :ocl:authored-assembly)
                     (fn [source options]
                       (is (not-any? imported-ids (take-nth 2 (second source))))
                       (swap! calls conj {:source source :environment (:param-env options)})
                       (pipeline/run-passes source pipeline/gpu-resident-pre-soa-passes options)))
             result-pairs (vec (partition 2 (second (:source result))))]
         (is (= expected-imports (count @calls) (count (:imports result))))
         (doseq [[original assembled] (map vector (:imports view) (:imports result))]
           (let [start (get-in assembled [:assembly :start])
                 bindings (get-in original [:call :bindings])
                 actual (subvec result-pairs start (+ start (count bindings)))]
             (is (identical? (:call original) (:call assembled)))
             (is (every? true? (map (fn [a b] (every? true? (map identical? a b))) bindings actual)))
             (is (= start (count (get-in assembled [:assembly :prefix]))))))
         (when (= 2 expected-imports)
           (let [later-outputs (remove nil? (get-in view [:imports 1 :call :outputs]))]
             (is (not-any? (set later-outputs) (keys (:environment (first @calls)))))))
         (is (= expected-imports (count (:unclosed-source-imports @entry/*emission-inputs*))))
         (is (identical? normalized (module/normalized-source phase))))))))

(deftest authored-preparation-refuses-lost-exports-and-unavailable-values
  (let [a (with-meta 'a {:raster.type/tag 'long})
        layout [{:kind :authored :start 0 :end 1 :bindings [[a 1]] :live-outs [a]}
                {:kind :authored :start 1 :end 1 :bindings [] :live-outs [] :tail [a]}]
        prepare #'module/prepare-authored-layout]
    (doseq [change [(fn [_ _] '(let* [] []))
                    (fn [_ _] '(let* [] [a]))
                    (fn [_ _] '(let* [a not-yet-produced] [a]))]]
      (is (= :unclosed-prepared-module-phase (reason #(prepare layout {} {} change)))))
    (is (= :unclosed-prepared-module-phase
           (reason #(prepare (pop layout) {} {} (fn [source _] source)))))
    (doseq [[entry change]
            [[{'x 'long} (fn [_ _] '(let* [x 99 a 1] [a]))]
             [{} (fn [_ _] '(let* [a 1 a 2] [a]))]]]
      (is (= :unclosed-prepared-module-phase (reason #(prepare layout entry {} change)))))
    (let [later (assoc-in layout [1 :bindings] [['later 2]])]
      (is (= :unclosed-prepared-module-phase
             (reason #(prepare later {} {} (fn [_ _] '(let* [later 99 a 1] [a])))))))
    (let [seen (atom [])
          result (prepare layout {} {:scalar-types {'future :long}
                                     :values {'future :unavailable 'a :not-yet-produced}}
                          (fn [source options] (swap! seen conj options) source))]
      (is (= 'long (get-in (second @seen) [:param-env 'a])))
      (is (every? #(not (contains? (:scalar-types %) 'future)) @seen))
      (is (every? #(not (contains? (:values %) 'future)) @seen))
      (is (every? #(not (contains? (:values %) 'a)) @seen))
      (is (= 1 (quot (count (second (:source result))) 2))))))

(deftest authored-assembly-composes-construction-and-normalization-history
  (doseq [[source expected-imports] [[#'training/local-gradient-sum 1] [#'two-gradient-imports 2]]]
    (entry/call-with-definition-inputs
     (fn []
       (let [root (core/resolve-deftm-var source)
             _ (core/prepare-deftm-frontend! root)
             frontend (entry/original-source-frontend root)
             normalizations (atom 0)
             normalize frontend/normalize-source
             result (with-redefs [frontend/normalize-source
                                  (fn [& args] (swap! normalizations inc) (apply normalize args))]
                      (module/construct-source
                       frontend {:dtype :float :target-device :ocl:assembled-source :segmented-plans? true}
                       {:prepare-source (fn [_] (throw (AssertionError. "unresolved source entered whole preparation")))
                        :canonicalize-ordinary-source
                        (fn [& _] (throw (AssertionError. "unclosed import entered ordinary canonicalization")))
                        :prepare-authored-segment
                        (fn [source options]
                          (pipeline/run-passes source pipeline/gpu-resident-pre-soa-passes options))}))
             phase (:prepared-phase result)
             normalized (:frontend-source result)
             view (module/view phase normalized)
             assembly (:authored-assembly view)
             construction (transport/source-construction-view (:construction result) (:normalization result)
                                                              (:represented-source result))
             components (module/analyze-source-components phase normalized)]
         (is (= 1 @normalizations))
         (is (some? phase))
         (is (identical? (:constructed assembly) (:represented-source result)))
         (is (not (identical? (:original assembly) (:constructed assembly))))
         (is (= expected-imports (count (:prepared-program-cuts construction))))
         (is (= (* 3 expected-imports) (count (:allocations components))))
         (is (= (* 6 expected-imports) (count (:pending-inputs components))))
         (is (= expected-imports (count (:unclosed-source-imports @entry/*emission-inputs*))))
         (is (false? (:admitted? view)))
         (is (nil? (#'frontend/continue-source-components* components)))
         (is (= (inc (* 2 expected-imports)) (count (module/authored-layout phase normalized)))))))))

(deftest authored-assembly-tracks-shifted-cuts-and-refuses-damaged-history
  (entry/call-with-definition-inputs
   (fn []
     (let [root (core/resolve-deftm-var #'two-gradient-imports)
           _ (core/prepare-deftm-frontend! root)
           frontend (entry/original-source-frontend root)
           options {:dtype :float :target-device :ocl:shifted-assembly :segmented-plans? true}
           prepare (fn [source options]
                     (let [prepared (pipeline/run-passes source pipeline/gpu-resident-pre-soa-passes options)
                           fresh (with-meta (gensym "assembly_local_") {:raster.type/tag 'long})]
                       (list* (first prepared) (into (vec (second prepared)) [fresh 0]) (drop 2 prepared))))
           result (module/construct-source frontend options {:prepare-authored-segment prepare})
           phase (:prepared-phase result)
           normalized (:frontend-source result)
           view (module/view phase normalized)
           second-import (second (:imports view))
           insertion (:assembly second-import)
           components (module/analyze-source-components phase normalized)]
       (is (= (inc (:original-start insertion)) (:start insertion)))
       (is (= 6 (count (:allocations components))))
       (is (= 12 (count (:pending-inputs components))))
       (doseq [change [#(update % :start inc)
                       #(update % :original-start inc)
                       #(dissoc % :original-source)
                       #(update % :constructed-source (fn [source] (apply list source)))]]
         (is (= :unclosed-prepared-module-phase
                (reason #(#'module/import-insertion-site
                          (:source view) (update second-import :assembly change))))))
       (is (= 2 (count (:unclosed-source-imports @entry/*emission-inputs*)))))))
  (entry/call-with-definition-inputs
   (fn []
     (with-redefs [transport/observe-construction! (fn [& _] nil)]
       (is (= :unclosed-prepared-module-phase
              (reason #(module/construct-source
                        (source-frontend) {:dtype :float}
                        {:prepare-authored-segment (fn [source _] source)}))))))))

(deftest modular-construction-keeps-ordinary-preparation-once
  (entry/call-with-definition-inputs
   (fn []
     (let [root (core/resolve-deftm-var #'ordinary-source)
           _ (core/prepare-deftm-frontend! root)
           calls (atom 0)
           result (module/construct-source
                   (entry/original-source-frontend root) {:dtype :double}
                   {:prepare-source (fn [source] (swap! calls inc) source)
                    :prepare-authored-segment (fn [& _] (throw (AssertionError. "ordinary source prepared as import")))})]
       (is (= 1 @calls))
       (is (nil? (:prepared-phase result)))
       (is (empty? (:unclosed-source-imports @entry/*emission-inputs*)))))))

(deftest authored-preparation-cannot-expand-an-undiscovered-import
  (entry/call-with-definition-inputs
   (fn []
     (let [root (core/resolve-deftm-var #'authored-with-hidden-gradient)
           _ (core/prepare-deftm-frontend! root)
           failure (try
                     (module/construct-source
                      (entry/original-source-frontend root) {:dtype :float :target-device :ocl:undiscovered-import}
                      {:prepare-authored-segment
                       (fn [source options]
                         (is (true? inline/*defer-prepared-ad-expansion?*))
                         (is (nil? transport/*prepared-program-consumer*))
                         (pipeline/run-passes source pipeline/gpu-resident-pre-soa-passes options))})
                     nil
                     (catch clojure.lang.ExceptionInfo error
                       (if (contains? #{:stale-definition-proof-input :unstable-definition-proof-input}
                                      (:reason (ex-data error)))
                         (throw error)
                         (ex-data error))))]
       (is (some? failure))
       (is (or (seq (:non-exempt-untagged failure)) (seq (:undevirtualized failure))))
       (is (empty? (:unclosed-source-imports @entry/*emission-inputs*)))))))

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
               :canonicalize-ordinary-source
               (fn [source _] (swap! events conj [:canonicalize source]) source)
               :represent-source
               (fn [source options]
                 (swap! events conj [:represent source])
                 ;; A real representation step may rebuild an ordinary source.
                 {:body (with-meta (apply list source) (meta source)) :options options})}))]
       (is (= [:prepare :represent :canonicalize :normalize] (mapv first @events)))
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

(deftest staged-input-query-follows-physical-roots-without-preparing-code
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           view (module/view phase source)
           joined (first (:import-origins view))
           pending (filterv #(= :argument (second (:id %))) (:obligation-origins joined))
           inputs
           (with-redefs [frontend/normalize-source
                         (fn [& _] (throw (AssertionError. "query normalized source")))
                         bytecode/compile-input-invocation-region
                         (fn [& _] (throw (AssertionError. "query prepared an executor")))]
             (mapv #(module/input-occurrence phase source
                                             (get-in % [:binding-origins 0 :origins 0 :binding])) pending))]
       (is (= 6 (count inputs)))
       (is (every? some? inputs))
       (is (= [0 1 0 1 0 1] (mapv #(get-in % [:argument :input-index]) inputs)))
       (doseq [[expected actual] (map vector pending inputs)]
         (is (identical? expected (:obligation actual)))
         (is (identical? phase (:phase actual)))
         (is (identical? (:import joined) (:import actual)))
         (is (identical? (get-in view [:imports 0 :physical]) (:physical actual)))
         (is (identical? (:obligation expected) (:argument actual)))
         (is (= (count (get-in actual [:argument :input :ports])) (count (:ports actual))))
         (doseq [{:keys [kind parameter physical-port root binding-origin origin binding]} (:ports actual)]
           (is (= :root kind) "actual size inputs include completed locals, not merely formals")
           (is (= parameter (:original-id root)))
           (is (= (:call-binding-index root) (:call-index binding-origin)))
           (is (identical? physical-port (get-in actual [:argument :input :ports parameter])))
           (is (identical? origin (first (:origins binding-origin))))
           (is (identical? binding (:binding origin)))
           (is (< (:index origin) (get-in actual [:origin :index])))))
       (is (every? #(= :long (get-in % [:physical-port :type]))
                   (mapcat :ports (filter #(= 1 (get-in % [:argument :input-index])) inputs))))
       (is (false? (:admitted? (module/view phase source))))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest staged-input-query-refuses-copied-and-unrelated-occurrences
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           view (module/view phase source)
           pending (first (filter #(= :argument (second (:id %)))
                                  (get-in view [:import-origins 0 :obligation-origins])))
           [id expression :as binding] (get-in pending [:binding-origins 0 :origins 0 :binding])
           other-phase (capture)
           copied-source (with-meta (apply list source) (meta source))
           copied-expression (with-meta (apply list expression) (meta expression))]
       (is (some? (module/input-occurrence phase source binding)))
       ;; Pair containers are recreated by source-descriptions; their members
       ;; must be the exact retained binder and initializer objects.
       (is (some? (module/input-occurrence phase source [id expression])))
       (is (nil? (module/input-occurrence phase source [id copied-expression])))
       (is (nil? (module/input-occurrence phase copied-source binding)))
       (is (nil? (module/input-occurrence other-phase source binding)))
       (is (nil? (module/input-occurrence other-phase (module/normalized-source other-phase) binding)))
       (is (nil? (module/input-occurrence phase source (get-in view [:origins 0 :binding]))))
       (is (nil? (module/input-occurrence {} source binding)))))))

(deftest staged-port-origin-requires-each-order-coordinate-and-physical-local-fact
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           view (module/view phase source)
           joined (first (:import-origins view))
           pending (first (filter #(and (= :argument (second (:id %))) (= 1 (last (:id %))))
                                  (:obligation-origins joined)))
           current (get-in pending [:binding-origins 0])
           final (get-in current [:origins 0])
           occurrence (module/input-occurrence phase source (:binding final))
           port (first (:ports occurrence))
           parameter (:parameter port)
           physical-port (:physical-port port)
           map-port (fn [joined current final]
                      (#'module/input-port-origin joined current final parameter physical-port))
           root-index (get-in port [:root :root-index])
           call-index (get-in port [:binding-origin :call-index])]
       (is (some? (map-port joined current final)))
       (is (nil? (map-port joined (assoc current :call-index call-index) final)))
       (is (nil? (map-port joined (assoc current :source-index (get-in port [:binding-origin :source-index])) final)))
       (is (nil? (map-port joined current (assoc final :index (get-in port [:origin :index])))))
       (is (nil? (map-port (assoc-in joined [:binding-origins call-index :source-index] (:source-index current))
                           current final)))
       (doseq [change [(fn [record] (assoc record :binder 'foreign))
                       (fn [record] (update-in record [:after :slot] inc))
                       (fn [record] (update record :source-input #(with-meta (apply list %) (meta %))))]]
         (let [query bytecode/retained-input-binding-context]
           (with-redefs [bytecode/retained-input-binding-context
                         (fn [& arguments] (change (apply query arguments)))]
             (is (nil? (map-port joined current final))))))
       (is (nil? (map-port (update-in joined [:call :root-bindings root-index :root-index] inc) current final)))
       (is (nil? (map-port (update-in joined [:call :root-bindings root-index :original-expression]
                                     #(with-meta (apply list %) (meta %))) current final)))
       ;; The same mapping also supports genuine formal verifier locals, but
       ;; only through their actual completed wrapper slot and body-entry fact.
       (let [slot (nth (get-in joined [:call :argument-plan :slots]) 3)
             {:keys [emission input-body]} (:physical occurrence)
             body (bytecode/retained-input-body-context emission input-body)
             formal (:formal slot)
             physical (get-in body [:locals formal])
             mapped (#'module/input-port-origin joined current final formal physical)]
         (is (= :formal (:kind mapped)))
         (is (identical? slot (:slot mapped)))
         (is (nil? (#'module/input-port-origin joined current final formal (update physical :slot inc))))
         (let [query bytecode/retained-input-body-context]
           (with-redefs [bytecode/retained-input-body-context
                         (fn [& arguments] (update-in (apply query arguments) [:locals formal :slot] inc))]
             (is (nil? (#'module/input-port-origin joined current final formal physical))))))))))

(deftest real-prepared-allocations-compose-without-fabricating-exemplar-values
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           view (module/view phase source)
           joined (first (:import-origins view))
           roots (filterv :argument-construction (get-in joined [:call :root-bindings]))]
       (is (= 3 (count roots)))
       (doseq [root roots]
         (let [inputs (mapv (fn [argument]
                              (module/input-occurrence
                               phase source (get-in joined [:binding-origins (:call-binding-index argument)
                                                           :origins 0 :binding])))
                            (get-in root [:argument-construction :arguments]))
               facts (into {} (map (fn [input]
                                    [(first (:binding input))
                                     {:carrier (get-in input [:argument :completion :parameter-carrier])}])) inputs)
               [_ expression] (get-in joined [:binding-origins (:call-binding-index root) :origins 0 :binding])
               projection (descriptor/direct-allocation-projection expression {} facts)
               completed-size (first (:binding (second inputs)))]
           (is (every? some? inputs))
           (is (= (list 'clojure.core/float-array completed-size) (:constructor projection)))
           (is (= 'raster.arrays/zeros-like
                  (get-in projection [:helper :definition :delegation :operation])))))
       (is (false? (:admitted? (module/view phase source))))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest staged-storage-analysis-retains-source-objects-and-pending-obligations
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           view (module/view phase source)
           analyze @#'frontend/source-storage-analysis
           describe @#'frontend/source-descriptions
           analyses (atom 0)
           descriptions (atom 0)
           components
           (with-redefs-fn
             {#'frontend/source-storage-analysis (fn [& args] (swap! analyses inc) (apply analyze args))
              #'frontend/source-descriptions (fn [& args] (swap! descriptions inc) (apply describe args))}
             #(module/analyze-source-components phase source))
           pending (:pending-inputs components)
           originals (filter #(= :argument (second (:id %)))
                             (get-in view [:import-origins 0 :obligation-origins]))]
       (is (= 1 @analyses @descriptions))
       (is (identical? source (:source components)))
       (is (= 3 (count (:allocations components))))
       (is (= #{:float} (set (map :dtype (:allocations components)))))
       (is (every? #(= [(:extent %)] (get-in components [:storage-values (:destination %) :shape]))
                   (:allocations components)))
       (is (= 6 (count pending) (count (:carrier-facts components))))
       (is (every? true? (map identical? originals (map :obligation pending))))
       (is (every? #(identical? phase (:phase %)) pending))
       (is (every? (fn [input]
                     (let [[id expression] (:binding input)
                           description (nth (:descriptions components) (get-in input [:origin :index]))]
                       (and (identical? id (first (:source-binding description)))
                            (identical? expression (second (:source-binding description)))
                            (not (contains? (:storage-values components) id)))))
                   (filter #(= 0 (get-in % [:argument :input-index])) pending))
           "no reference AbstractValue or capacity is fabricated from a carrier")
       (is (nil? (#'frontend/continue-source-components* components)))
       ;; Pin the barrier independently of today's unsupported C expressions.
       (let [options {:dtype :double :segmented-plans? true :scalar-types {'n :long}
                      :array-types {'x :double} :public-parameters '[x n]}
             simple (frontend/normalize-source
                     '(let* [out (raster.par/pmap i n double (clojure.core/aget x i))] out) options)
             ready (frontend/analyze-source-components simple options {})]
         (is (some? (#'frontend/continue-source-components* ready)))
         (is (nil? (#'frontend/continue-source-components* (assoc ready :pending-inputs pending)))))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest prepared-scalar-input-regions-retain-original-ports-and-pending-obligations
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           components (module/analyze-source-components phase source)
           inputs (:pending-inputs components)
           scalars (filter #(= 1 (get-in % [:argument :input-index])) inputs)
           references (filter #(= 0 (get-in % [:argument :input-index])) inputs)
           observations @entry/*emission-inputs*]
       (is (= 3 (count scalars) (count references)))
       (doseq [input scalars]
         (let [region (module/prepare-scalar-input-region phase source (:binding input))
               ports (:ports input)]
           (is (some? region))
           (is (identical? (:argument input) (get-in region [:occurrence :argument])))
           (is (= (mapv :parameter ports) (:parameters region)))
           (is (= [Long/TYPE Long/TYPE]
                  (vec (.getParameterTypes ^java.lang.reflect.Method (:method region)))))
           (is (= (:contract (:argument input))
                  (assoc (:input-contract region) :expected-input-carrier
                         (get-in region [:input-completion :emitted-type]))))
           (doseq [[a b] [[2 17] [0 17] [-1 17]]]
             (is (= (* a b) (apply (:invoke region) [a b]))))))
       (doseq [input references]
         (is (nil? (module/prepare-scalar-input-region phase source (:binding input)))))
       (is (nil? (module/prepare-scalar-input-region
                  phase (apply list source) (:binding (first scalars)))))
       (is (= (:unclosed-source-imports observations)
              (:unclosed-source-imports @entry/*emission-inputs*)))
       (is (nil? (#'frontend/continue-source-components* components)))))))

(deftest scalar-input-region-mismatches-and-stale-guards-cannot-execute
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           inputs (:pending-inputs (module/analyze-source-components phase source))
           scalar (first (filter #(= 1 (get-in % [:argument :input-index])) inputs))
           reference (first (filter #(= 0 (get-in % [:argument :input-index])) inputs))
           original (module/prepare-scalar-input-region phase source (:binding scalar))
           parameter (first (:parameters original))
           calls (atom 0)]
       (doseq [change [(constantly nil)
                       #(update % :parameters (comp vec reverse))
                       #(update-in % [:physical-ports parameter :slot] inc)
                       #(assoc-in % [:input-contract :target-carrier] Double/TYPE)
                       #(assoc-in % [:input-completion :adaptation] :identity)
                       #(assoc-in % [:input-completion :emitted-type] :long)]]
         (with-redefs [bytecode/compile-input-invocation-region (fn [& _] (change original))]
           (is (= :unclosed-prepared-module-phase
                  (reason #(module/prepare-scalar-input-region phase source (:binding scalar)))))))
       (with-redefs [bytecode/compile-input-invocation-region
                     (fn [& _] (swap! calls inc) (throw (ex-info "must abstain" {})))]
         (is (nil? (module/prepare-scalar-input-region phase source (:binding reference))))
         (is (nil? (module/prepare-scalar-input-region phase (apply list source) (:binding scalar))))
         (is (nil? (module/prepare-scalar-input-region phase source ['foreign 1])))
         (is (zero? @calls)))
       (let [prepared (with-redefs [bytecode/compile-input-invocation-region
                                   (fn [& _] (assoc original :invoke (fn [& _] (swap! calls inc))))]
                        (module/prepare-scalar-input-region phase source (:binding scalar)))
             source-var (core/resolve-deftm-var #'training/local-gradient-sum)]
         (apply (:invoke prepared) [2 17])
         (is (= 1 @calls))
         (reset! calls 0)
         (with-redefs-fn {source-var (fn [& _] (throw (AssertionError. "must not execute replacement")))}
           #(is (thrown? clojure.lang.ExceptionInfo (apply (:invoke prepared) [2 17]))))
         (is (zero? @calls)))
       (let [broken? (atom false)
             prepared (with-redefs [bytecode/compile-input-invocation-region
                                   (fn [& _]
                                     (assoc original
                                            :guard #(when @broken?
                                                      (throw (ex-info "stale region" {:reason :test-stale-region})))
                                            :invoke (fn [& _] (swap! calls inc))))]
                        (module/prepare-scalar-input-region phase source (:binding scalar)))]
         (reset! broken? true)
         (is (= :test-stale-region (reason #(apply (:invoke prepared) [2 17]))))
         (is (zero? @calls)))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest staged-storage-analysis-rejects-detached-or-reordered-source-data
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           options (:options (module/view phase source))
           components (module/analyze-source-components phase source)
           inputs (into {} (map (juxt #(get-in % [:origin :index]) identity)) (:pending-inputs components))
           index (apply min (keys inputs))
           input (get inputs index)
           copy (fn [form] (with-meta (apply list form) (meta form)))]
       (is (nil? (module/analyze-source-components phase (copy source))))
       (doseq [change [(fn [o] (update-in o [:binding 1] copy))
                       (fn [o] (update-in o [:origin :index] inc))
                       (fn [o] (assoc-in o [:argument :contract :target-carrier] Object))
                       (fn [o] (assoc-in o [:argument :completion :parameter-carrier] Object))
                       (fn [o] (assoc-in o [:ports 0 :origin :index] index))
                       (fn [o] (update-in o [:ports 0 :binding 0]
                                         #(with-meta % (assoc (meta %) :detached true))))]]
         (is (= :source-input-correspondence
                (reason #(frontend/analyze-source-components source options (update inputs index change))))))
       (is (= :source-input-correspondence
              (reason #(frontend/analyze-source-components source options (assoc inputs 9999 input)))))
       (doseq [subset [{} (dissoc inputs index) {index input}]]
         (let [partial (frontend/analyze-source-components source options subset)]
           (is (= (count subset) (count (:pending-inputs partial))))
           (is (nil? (#'frontend/continue-source-components* partial)))))
       ;; Accounting IDs are not authority even when changed in internal data.
       (let [forged (frontend/analyze-source-components
                     source options (assoc-in inputs [index :obligation :id] [:forged]))]
         (is (nil? (#'frontend/continue-source-components* forged))))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest staged-reference-aliases-preserve-the-entire-prior-value
  (entry/call-with-definition-inputs
   (fn []
     (let [x (av/tensor {:dtype :float :shape ['rows 'width]
                         :logical-layout {:kind :dense} :ownership :borrowed})
           phase (module/capture (source-frontend) {:dtype :float :segmented-plans? true :values {'x x}})
           source (module/normalized-source phase)
           components (module/analyze-source-components phase source)
           input (first (filter #(= 0 (get-in % [:argument :input-index])) (:pending-inputs components)))
           prior (get-in input [:ports 0 :binding 0])
           result (first (:binding input))]
       (is (identical? x (get-in components [:storage-values prior])))
       (is (identical? x (get-in components [:storage-values result])))
       (is (nil? (#'frontend/continue-source-components* components)))
       (is (= 1 (count (:unclosed-source-imports @entry/*emission-inputs*))))))))

(deftest staged-input-ports-execute-the-original-size-region
  (entry/call-with-definition-inputs
   (fn []
     (let [phase (capture)
           source (module/normalized-source phase)
           view (module/view phase source)
           pending (filter #(and (= :argument (second (:id %))) (= 1 (last (:id %))))
                           (get-in view [:import-origins 0 :obligation-origins]))
           parameters (get-in view [:imports 0 :event :arguments])
           source-ns (get-in view [:imports 0 :physical :emission :source-ns])]
       (is (= 3 (count pending)))
       (doseq [obligation pending]
         (let [occurrence (module/input-occurrence phase source
                                                   (get-in obligation [:binding-origins 0 :origins 0 :binding]))
               {:keys [physical construction ports origin]} occurrence
               region (bytecode/compile-input-invocation-region
                       (:emission physical) (:input-body physical) (:source-call construction) 1)
               positions (zipmap (map :parameter ports) (range))
               indices (mapv positions (:parameters region))
               ;; Evaluate the real normalized prefix with JVM semantics, then
               ;; route its completed host values by origin, not operand names.
               prefix (subvec (second source) 0 (* 2 (inc (:index origin))))
               tags (get-in view [:imports 0 :prepared :tags])
               interface (types/ensure-fn-interface! tags)
               read-values
               (:implementation
                (core/emit-owned-bytecode!
                 bytecode/compile-typed-impl! "raster.compiled.ModuleInputPrefixTest"
                 parameters tags 'Object
                 [(list 'let* prefix [(mapv (comp first :binding) ports)
                                      (first (:binding occurrence))])]
                 source-ns (:iface-name interface) 'module-input-prefix-test))]
           (is (every? some? indices))
           (is (every? #(= Long/TYPE %) (vec (.getParameterTypes ^java.lang.reflect.Method (:method region)))))
           (is (= Long/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
           (doseq [rows [1 3]]
             (let [[inputs expected] (read-values (float-array [0.2 0.4])
                                                  (float-array (repeat (* rows 2) 0.5))
                                                  (float-array (repeat rows 0.1)) rows 2)]
               (is (= expected (apply (:invoke region) (mapv inputs indices))))))))
       (is (false? (:admitted? (module/view phase source))))))))

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
