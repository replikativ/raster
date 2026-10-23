(ns raster.compiler.prepared-module-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.ad.reverse :as reverse]
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
       (is (identical? (:import (first imports)) (:import joined)))
       (is (= (count (:bindings call)) (count (:binding-origins joined))))
       (is (= (+ (count (:input-obligations call)) (count (get-in call [:proposal :completion-obligations])))
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
           projections (atom 0)
           preparations (atom 0)
           phase
           (with-redefs [reverse/project-prepared-value+grad-call
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
