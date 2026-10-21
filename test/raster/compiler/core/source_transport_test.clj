(ns raster.compiler.core.source-transport-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.scalar.inline :as inline]))

(deftm scoped-helper [n :- Long] :- Long
  (let [x n]
    (let [x (inc x)]
      (+ x n))))

(def unrelated-input 7)

(deftm loop-helper [n :- Long] :- Long
  (loop [i 0 total 0]
    (if (< i n) (recur (inc i) (+ total i)) total)))

(deftm normalization-shadowed [n :- Long] :- Long
  (let [n (inc n)]
    (let [n (inc n)] n)))

(defn- imported-helper []
  (entry/call-with-definition-inputs
   #(#'inline/try-resolve-walked-body #'scoped-helper_m_long)))

(defn- view [imported]
  (entry/call-with-definition-inputs
   #(transport/body-import-view (:source-import imported) (:walked-body imported))))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- original-frontend [source]
  (:frontend
   (view (entry/call-with-definition-inputs #(#'inline/try-resolve-walked-body source)))))

(defn- normalize-original [frontend]
  (entry/call-with-definition-inputs
   #(transport/normalize-frontend-bindings frontend (get-in frontend [:source-entry :params]))))

(defn- normalized-view [normalized constructed]
  (entry/call-with-definition-inputs
   #(transport/normalization-occurrence (:source-normalization normalized) constructed)))

(deftest source-normalization-keeps-actual-occurrences-and-lexical-renamings
  (let [frontend (original-frontend #'scoped-helper_m_long)
        normalized (normalize-original frontend)
        root (first (:walked-body normalized))
        original (first (:body frontend))
        root-view (normalized-view normalized root)
        inner (nth root 2) renamed (first (second inner))
        inner-body (nth inner 2)
        inner-view (normalized-view normalized inner-body)
        evaluate (fn [expression] (eval (list 'fn '[n] expression)))]
    (is (identical? frontend (:frontend root-view)))
    (is (identical? original (:original root-view)))
    (is (identical? root (:constructed root-view)))
    (is (= 0 (:root root-view)))
    (is (= {} (:renaming root-view)))
    (is (= {'x renamed} (:renaming inner-view)))
    (is (contains? (:bound inner-view) 'n))
    (doseq [n [-7 0 1 13]]
      (is (= (scoped-helper n) ((evaluate original) n) ((evaluate root) n))))
    (with-redefs [util/uniquify-rebindings
                  (fn [& _] (throw (AssertionError. "query renormalized source")))]
      (is (some? (normalized-view normalized root))))))

(deftest normalization-cannot-authorize-copied-foreign-or-ownerless-occurrences
  (let [frontend (original-frontend #'scoped-helper_m_long)
        normalized (normalize-original frontend) input (:source-normalization normalized)
        root (first (:walked-body normalized))
        copied (with-meta (apply list root) (meta root))]
    (is (nil? (transport/normalization-occurrence input root)))
    (is (nil? (normalized-view normalized copied)))
    (is (nil? (normalized-view {:source-normalization {}} root)))
    (is (nil? (normalized-view normalized (first (:body frontend)))))
    (is (= :source-normalization-input-mismatch
           (reason #(entry/call-with-definition-inputs
                     (fn [] (transport/normalize-frontend-bindings frontend '[other]))))))))

(deftest normalization-rejects-ambiguous-duplicate-reconstruction-events
  (doseq [distinct-output? [false true]]
   (let [frontend (original-frontend #'scoped-helper_m_long)
        uniquify util/uniquify-rebindings
        normalized
        (with-redefs [util/uniquify-rebindings
                      (fn [parameters expression observe!]
                        (uniquify parameters expression
                                  (fn [original constructed renaming bound]
                                    (observe! original constructed renaming bound)
                                    (observe! original
                                              (if (and distinct-output? (seq? constructed))
                                                (with-meta (apply list constructed) (meta constructed))
                                                constructed)
                                              renaming bound))))]
          (normalize-original frontend))]
    (is (nil? (normalized-view normalized (first (:walked-body normalized))))))))

(deftest normalization-reuses-the-shared-freshness-and-history-guards
  (let [frontend (original-frontend #'normalization-shadowed_m_long)]
    (doseq [allocate [(constantly 'n)
                      (fn [original] (with-meta 'reused (meta original)))
                      (fn [_] (with-meta (gensym "retagged") {:tag 'double}))]]
      (binding [util/*fresh-binder* allocate]
        (is (= :nonfresh-source-import-binder (reason #(normalize-original frontend)))))))
  (let [frontend (original-frontend #'scoped-helper_m_long)
        normalized (normalize-original frontend)
        root (first (:walked-body normalized))
        owner (atom {:static [] :values []})
        source (get-in frontend [:source-entry :source-var]) metadata (meta source)
        raw (:raster.core/deftm-source-body metadata)]
    (binding [entry/*emission-inputs* owner]
      (is (some? (transport/normalization-occurrence (:source-normalization normalized) root))))
    (try
      (alter-meta! source assoc :raster.core/deftm-source-body (with-meta (mapv identity raw) (meta raw)))
      (is (= metadata (meta source)))
      (is (= :stale-definition-proof-input
             (reason #(entry/assert-definition-inputs-current! @owner))))
      (finally (reset-meta! source metadata)))))

(deftest actual-inliner-keeps-the-original-body-and-ordered-hygiene-recipe
  (let [imported (imported-helper)
        {:keys [frontend parameters effective-body hygienic-body walked-body renamings]} (view imported)
        original (first (:body frontend))
        evaluate (fn [body] (eval (list 'fn parameters (first body))))]
    (is (some? (:source-import imported)))
    (is (identical? (:body frontend) effective-body))
    (is (identical? (:walked-body imported) walked-body))
    (is (seq renamings))
    (is (= (count renamings) (count (distinct (map second renamings)))))
    (is (every? #(not= 'n (second %)) renamings))
    (is (every? #(identical? (meta (first %)) (meta (second %))) renamings))
    (doseq [n [0 1 3 -7 1000]]
      (is (= (scoped-helper n) ((evaluate [original]) n)
             ((evaluate hygienic-body) n) ((evaluate walked-body) n))))))

(deftest copied-or-ownerless-body-import-queries-cannot-reconstruct-a-witness
  (let [imported (imported-helper) input (:source-import imported) body (:walked-body imported)]
    (is (nil? (transport/body-import-view input body)))
    (entry/call-with-definition-inputs
     (fn []
       (doseq [[candidate output] [[nil body] [{:source-import input} body]
                                  [input (mapv identity body)]]]
         (is (nil? (transport/body-import-view candidate output))))
       (is (empty? (:definition-proofs @entry/*emission-inputs*)))))))

(deftest import-queries-check-history-without-transforming-source-again
  (let [imported (imported-helper)
        forbidden (fn [& _] (throw (AssertionError. "query reran source transformation")))]
    (with-redefs [util/alpha-convert forbidden inference/qualify-body-symbols forbidden]
      (is (some? (view imported))))
    (let [metadata (meta #'scoped-helper_m_long)]
      (try
        (alter-meta! #'scoped-helper_m_long assoc ::changed true)
        (is (= :stale-definition-proof-input (reason #(view imported))))
        (finally (reset-meta! #'scoped-helper_m_long metadata))))))

(deftest body-import-history-excludes-unrelated-parent-dependencies
  (let [parent (atom {:static [] :values []})
        imported (binding [entry/*emission-inputs* parent]
                   (entry/observe-definition-input! #'unrelated-input
                                                    (:ns (meta #'unrelated-input)) 'unrelated-input)
                   (#'inline/try-resolve-walked-body #'scoped-helper_m_long))
        metadata (meta #'unrelated-input)]
    (try
      (alter-meta! #'unrelated-input assoc ::changed true)
      (is (= :stale-definition-proof-input
             (reason #(entry/assert-definition-inputs-current! @parent))))
      (binding [entry/*emission-inputs* (atom {:static [] :values []})]
        (is (some? (transport/body-import-view (:source-import imported) (:walked-body imported))))
        (is (not (contains? (:definition-proofs @entry/*emission-inputs*) #'unrelated-input))))
      (finally (reset-meta! #'unrelated-input metadata)))))

(deftest owned-body-import-rejects-nonfresh-and-retagged-binders
  (doseq [allocate [(constantly 'n)
                    (fn [original] (with-meta 'reused (meta original)))
                    (fn [_] (with-meta (gensym "changed") {:tag 'double}))]]
    (binding [util/*fresh-binder* allocate]
      (is (= :nonfresh-source-import-binder (reason imported-helper))))))

(deftest legacy-body-transformation-does-not-acquire-original-source-ownership
  (let [body '[(let* [x n] x)]
        imported (entry/call-with-definition-inputs
                  #(transport/prepare-inline-body nil body '[n] '[long] *ns*))]
    (is (nil? (:source-import imported)))
    (is (= 4 ((eval (list 'fn '[n] (first (:walked-body imported)))) 4)))))

(deftest loop-carries-use-the-same-scope-aware-body-import
  (let [imported (entry/call-with-definition-inputs
                  #(#'inline/try-resolve-walked-body #'loop-helper_m_long))
        {:keys [parameters walked-body renamings]} (view imported)
        transformed (eval (list 'fn parameters (first walked-body)))]
    (is (= 2 (count renamings)))
    (doseq [n [0 1 3 7 20]]
      (is (= (quot (* n (dec n)) 2) (loop-helper n) (transformed n))))))

(deftest an-import-cannot-use-a-copied-body-or-different-declarations
  (let [imported (imported-helper)
        {:keys [frontend parameters]} (view imported)
        body (:body frontend) tags (get-in frontend [:source-entry :tags])
        source-ns (get-in frontend [:source-entry :source-ns])]
    (doseq [[input p t ns] [[(mapv identity body) parameters tags source-ns]
                           [body '[foreign] tags source-ns]
                           [body parameters '[double] source-ns]
                           [body parameters tags (the-ns 'clojure.core)]]]
      (is (= :source-import-input-mismatch
             (reason #(entry/call-with-definition-inputs
                       (fn [] (transport/prepare-inline-body frontend input p t ns)))))))))

(deftest a-competing-parent-update-cannot-be-replaced-by-import-history
  (let [parent (atom {:static [] :values []}) allocate util/*fresh-binder*
        touched? (atom false)]
    (binding [entry/*emission-inputs* parent
              util/*fresh-binder* (fn [source]
                                   (when (compare-and-set! touched? false true)
                                     (swap! parent assoc ::competing true))
                                   (allocate source))]
      (is (= :unstable-definition-proof-input
             (reason #(#'inline/try-resolve-walked-body #'scoped-helper_m_long))))
      (is (::competing @parent)))))

(deftest formal-class-history-survives-body-import-into-a-fresh-owner
  (let [name (gensym "raster.imported.class.") source (create-ns name)]
    (try
      (binding [*ns* source]
        (clojure.core/refer 'clojure.core)
        (.importClass ^clojure.lang.Namespace source java.util.Date)
        (eval '(raster.core/deftm helper [unused :- Date] :- Long (let [one 1] one))))
      (let [imported (entry/call-with-definition-inputs
                      #(#'inline/try-resolve-walked-body (ns-resolve source 'helper_m_Date)))]
        (is (= '[java.util.Date] (:tags (view imported))))
        (ns-unmap source 'Date)
        (.importClass ^clojure.lang.Namespace source java.sql.Date)
        (is (= :stale-source-resolution (reason #(view imported)))))
      (finally (remove-ns name)))))

(deftest legacy-parameter-spellings-keep-the-existing-symbol-normalization
  (let [source (the-ns 'raster.compiler.core.source-transport-test)
        imported (transport/prepare-inline-body nil '[n] [:n] '[long] source)]
    (is (= '[n] (:walked-body imported)))
    (is (nil? (:source-import imported)))))

(deftest downstream-owner-retains-exact-raw-source-identity-after-a-query
  (let [imported (imported-helper)
        owner (atom {:static [] :values []})
        description (binding [entry/*emission-inputs* owner]
                      (transport/body-import-view (:source-import imported) (:walked-body imported)))
        source (get-in description [:frontend :source-entry :source-var])
        metadata (meta source)
        raw (:raster.core/deftm-source-body metadata)
        copied (with-meta (mapv identity raw) (meta raw))]
    (try
      (is (not (identical? raw copied)))
      (alter-meta! source assoc :raster.core/deftm-source-body copied)
      (is (= metadata (meta source)) "ordinary metadata equality cannot fence this copy")
      (is (every? entry/var-snapshot-current? (vals (:definition-proofs @owner))))
      ;; No further token query: the enclosing publication/reuse fence itself
      ;; must retain the original source identity after successful adoption.
      (is (= :stale-definition-proof-input
             (reason #(entry/assert-definition-inputs-current! @owner))))
      (finally (reset-meta! source metadata)))))
