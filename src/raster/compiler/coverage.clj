(ns raster.compiler.coverage
  "Typed-route coverage of the compiler over a corpus of `deftm` functions.

   The compiler's coverage claim is measured, not asserted: every source `deftm` in the corpus
   namespaces is compiled through the diagnostic pipeline for a GPU target and the normalized
   report records which route it took (`:typed-soac`, `:compatibility`, `:scalar`), every
   route decline with its reason and operation, and any hard error. A committed baseline turns
   this into a ratchet (`ratchet-violations`): a function that took the typed route may not fall
   back, and a function that compiled may not start failing.

   This namespace only enumerates vars and calls `raster.compiler.pipeline/compile-report`; it
   never inspects source forms or infers operations."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.pprint :as pp]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.report :as report]))

(def default-namespaces
  "Corpus namespaces: the deep-learning substrate, the ODE/PDE surface and the NN primitives."
  '[raster.dl.nn raster.dl.attention raster.dl.loss raster.dl.array-ops raster.dl.optim
    raster.dl.tensor raster.dl.einsum raster.dl.diffusion raster.dl.train raster.nn
    raster.ode.pde])

(def default-baseline-path "test/resources/coverage/gpu-corpus.edn")

(defn corpus-vars
  "Source `deftm` vars (generic functions, not their generated specializations) in `namespaces`."
  [namespaces]
  (vec
   (for [ns-sym namespaces
         :let [_ (require ns-sym)]
         v (sort-by (comp str :name meta) (vals (ns-publics ns-sym)))
         :when (:raster.core/generic-function (meta v))]
     v)))

(defn- var-symbol [v]
  (symbol (str (ns-name (:ns (meta v)))) (str (:name (meta v)))))

(defn- decline-facts [declines]
  (->> declines
       (mapcat (fn [decline]
                 (if (seq (:bindings decline))
                   (map (fn [binding]
                          {:reason (:reason binding)
                           :operation (:operation binding)
                           :kind (:kind binding)})
                        (:bindings decline))
                   [(select-keys decline [:reason :operation :kind])])))
       (remove empty?)
       distinct
       vec))

(defn- error-reason [throwable]
  (or (:reason (ex-data throwable))
      (some-> (.getMessage throwable) (subs 0 (min 120 (count (.getMessage throwable)))))))

(defn- effect-order-facts
  "How the typed program's effect maps iterate: `{:independent n :sequential m}`. A store the
   index algebra proves injective keeps its map `:independent`; an unproven one serializes the
   whole launch, so a move from `:independent` to `:sequential` is a performance regression the
   ratchet must see even though the route is unchanged."
  [pipeline]
  (let [program (:soac-fused pipeline)]
    (when (and (map? program) (= :typed-soac (:dialect program)))
      (frequencies
       (for [equation (:equations program)
             typed-equation (dialect/equations (:algorithm equation))
             :let [{:keys [kind attributes]} (dialect/operation-parts typed-equation)]
             :when (= 'effect-map kind)]
         (:iteration-order attributes))))))

(defn report-var
  "Compile one source deftm for `target-device` at `dtype` and reduce its report to route facts."
  [v {:keys [target-device dtype] :or {dtype :float}}]
  (let [row {:var (var-symbol v)}]
    (try
      (let [pipeline (pipeline/show-pipeline v :target-device target-device :dtype dtype)
            report (report/from-pipeline pipeline)
            orders (effect-order-facts pipeline)]
        (cond-> (assoc row
                       :route (get-in report [:route :source-dialect])
                       :typed-validated (boolean (get-in report [:route :typed-validated]))
                       :declines (decline-facts (get-in report [:route :declines]))
                       ;; TypedSOAC frontend coverage does not imply KernelBody emission.
                       ;; Retain the existing normalized emitter evidence from this SAME compile.
                       :emission (:emission report)
                       :emission-declines (count (get-in report [:emission :declines])))
          (seq orders) (assoc :effect-orders orders)))
      (catch Throwable t
        (assoc row :route :error :error (error-reason t))))))

(defn corpus-report
  "Route facts for every corpus var, with a summary by route."
  ([opts] (corpus-report default-namespaces opts))
  ([namespaces opts]
   (let [rows (mapv #(report-var % opts) (corpus-vars namespaces))]
     {:target-device (:target-device opts)
      :dtype (:dtype opts :float)
      :summary (merge {:total (count rows)}
                      (frequencies (map :route rows)))
      ;; Counts artifacts, including dispatch alternatives, not runtime launches or latency.
      :emission-summary
      {:artifact-routes (reduce #(merge-with + %1 (get-in %2 [:emission :routes] {})) {} rows)
       :programs-with-declines (count (filter #(seq (get-in % [:emission :declines])) rows))}
      :vars (vec (sort-by (comp str :var) rows))})))

(def route-rank
  "Routes ordered from best to worst; a move down this order is a regression."
  {:typed-soac 0 :compatibility 1 :scalar 2 :error 3})

(defn ratchet-violations
  "Ways in which `report` regressed against `baseline`.

   A var may not move to a worse route (typed → compatibility → scalar → error) and a validated
   typed program may not lose its validation. New and removed vars are not violations, so the
   corpus can grow and shrink without editing the baseline."
  [baseline report]
  (let [before (into {} (map (juxt :var identity)) (:vars baseline))]
    (vec
     (for [row (:vars report)
           :let [old (get before (:var row))]
           :when old
           violation (cond-> []
                       (> (get route-rank (:route row) 3) (get route-rank (:route old) 3))
                       (conj {:var (:var row) :violation :route-downgraded
                              :before (:route old) :after (:route row)
                              :declines (:declines row) :error (:error row)})

                       (and (:typed-validated old) (not (:typed-validated row)))
                       (conj {:var (:var row) :violation :lost-typed-validation
                              :route (:route row)})

                       ;; an effect map that iterated independently may not start serializing
                       ;; or disappear from the typed program
                       (or (> (get-in row [:effect-orders :sequential] 0)
                              (get-in old [:effect-orders :sequential] 0))
                           (< (get-in row [:effect-orders :independent] 0)
                              (get-in old [:effect-orders :independent] 0)))
                       (conj {:var (:var row) :violation :effect-map-serialized
                              :before (:effect-orders old) :after (:effect-orders row)}))]
       violation))))

(defn read-baseline [path]
  (with-open [reader (java.io.PushbackReader. (io/reader path))]
    (edn/read reader)))

(defn baseline-facts
  "The existing device-independent ratchet excludes target-specific emission diagnostics."
  [report]
  (-> report
      (dissoc :emission-summary)
      (update :vars #(mapv (fn [row] (dissoc row :emission :emission-declines)) %))))

(defn write-baseline!
  "Write `report` as the committed baseline. Emission facts are excluded: they depend on the
   device's tuned leaves, while route and error facts are device-independent."
  [path report]
  (io/make-parents path)
  (with-open [writer (io/writer path)]
    (binding [*print-length* nil *print-level* nil]
      (pp/pprint (baseline-facts report) writer)))
  path)

(defn -main
  "`update` rewrites the baseline for the OpenCL device the runtime selects; otherwise prints the
   summary and the ratchet violations against the committed baseline."
  [& [command]]
  (let [report (corpus-report {:target-device :ocl:0 :dtype :float})]
    (println "coverage summary:" (pr-str (:summary report)))
    (println "emitted artifacts (not executed launches):" (pr-str (:emission-summary report)))
    (if (= "update" command)
      (println "baseline written:" (write-baseline! default-baseline-path report))
      (let [violations (ratchet-violations (read-baseline default-baseline-path) report)]
        (doseq [v violations] (println "  violation:" (pr-str v)))
        (when (seq violations) (System/exit 1))))
    (shutdown-agents)))
