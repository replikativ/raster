(ns raster.acceptance.finetune-chain
  "Opt-in test support accepting unchanged external Gemma declarations through public composition.
   Run in an isolated JVM: selected trusted source forms are evaluated, not copied or rewritten."
  (:refer-clojure :exclude [run!])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.java.shell :as shell]
            [raster.ad.reverse :as reverse]
            [raster.dl.loss :as loss]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]))

(def source-revision "9e9ba5d62f3822f056e01c37231d7eaa7c84947c")

(def ^:private source-specs
  [{:path "src/finetune/train.clj" :namespace 'finetune.train
    :definitions '#{lora-lin gblock gblock-vjp-loss gblock-vjp-step! gblock-fwd!
                    real-cfg layer-theta weight-keys adapter-keys hf-layer-names hf-layer-weights
                    init-adapters rand-weights scalars fwd-args bwd-args}}
   {:path "test/finetune/gemma_resident_train_test.clj"
    :namespace 'finetune.gemma-resident-train-test
    :definitions '#{ref-loss2 layer-arrs adapter-pos fvec worst-rel recovered-grads chain-cfg}}])

(defn- checked-source [root {:keys [path]}]
  (let [{:keys [exit out err]} (shell/sh "git" "--no-replace-objects" "-C" (str root) "show"
                                      (str source-revision ":" path))
        file (io/file root path)]
    (when-not (and (zero? exit) (.isFile file))
      (throw (ex-info "external acceptance requires the pinned source files"
                      {:reason :external-training-source :path path :revision source-revision
                       :git-error err})))
    (let [actual (slurp file :encoding "UTF-8")]
      (when-not (= out actual)
        (throw (ex-info "external source differs from the pinned Git object"
                        {:reason :external-training-source-drift :path path
                         :revision source-revision})))
      actual)))

(defn- selected-forms [text {:keys [namespace definitions]}]
  (binding [*read-eval* false]
   (with-open [reader (clojure.lang.LineNumberingPushbackReader. (java.io.StringReader. text))]
    (let [forms (loop [result []]
                  (let [form (read {:eof ::eof} reader)]
                    (if (= ::eof form) result (recur (conj result form)))))
          ns-form (first forms)
          selected (filterv #(contains? definitions (second %)) (rest forms))]
      (when-not (and (= 'ns (first ns-form)) (= namespace (second ns-form))
                     (= definitions (set (map second selected)))
                     (= (count definitions) (count selected)))
        (throw (ex-info "external declaration set is incomplete or ambiguous"
                        {:reason :external-training-declarations :namespace namespace})))
      (into [ns-form] selected)))))

(defn load-sources!
  "Check both source files before evaluating anything. Reject existing external namespaces.
   The obsolete session adapter and test orchestration are not loaded; model/oracle bodies stay exact."
  [root]
  (let [sources (mapv #(selected-forms (checked-source root %) %) source-specs)]
    (doseq [{:keys [namespace]} source-specs]
      (when (find-ns namespace)
        (throw (ex-info "external acceptance must run in an isolated namespace environment"
                        {:reason :external-training-namespace :namespace namespace}))))
    (doseq [[spec forms] (map vector source-specs sources)]
      (binding [*ns* *ns* *file* (str (io/file root (:path spec)))]
        (let [ns-form (first forms)
              oracle? (= 'finetune.gemma-resident-train-test (second ns-form))]
          ;; Only the loader dependency changes: use the already evaluated numerical namespace
          ;; rather than requiring the old package adapter. The selected definition forms are exact.
          (eval (if oracle?
                  (apply list (map (fn [part]
                                    (if (and (seq? part) (= :require (first part)))
                                      (apply list :require
                                             (remove #(= 'finetune.train (first %)) (rest part)))
                                      part)) ns-form))
                  ns-form))
          (when oracle? (alias 'ft 'finetune.train))
          (doseq [form (rest forms)] (eval form)))))
    (let [resolve! (fn [namespace name]
                     (or (ns-resolve namespace name)
                         (throw (ex-info "checked external declaration is absent"
                                         {:namespace namespace :name name}))))]
      {:train (into {} (for [name (:definitions (first source-specs))]
                         [name (resolve! 'finetune.train name)]))
       :oracle (into {} (for [name (:definitions (second source-specs))]
                          [name (resolve! 'finetune.gemma-resident-train-test name)]))})))

(defn- reference-args [oracle cfg weights adapters input target]
  (-> [input]
      (into ((oracle 'layer-arrs) (weights 0) (adapters 0)))
      (into ((oracle 'layer-arrs) (weights 1) (adapters 1)))
      (conj target)
      (into (map cfg [:seq :d :nq :nkv :hd :dff :r :eps]))))

(defn- checked-error [oracle actual expected]
  (when-not (= (count actual) (count expected))
    (throw (ex-info "external oracle array lengths differ"
                    {:reason :external-training-shape
                     :actual (count actual) :expected (count expected)})))
  ((oracle 'worst-rel) actual expected))

(defn- difference-summary
  "Diagnostic only: never substitutes for the pinned oracle's componentwise error.
   Finite-input subtraction may overflow; retain that infinite error rather than hide it."
  [actual expected]
  (when-not (= (count actual) (count expected))
    (throw (ex-info "diagnostic array lengths differ"
                    {:reason :external-training-shape
                     :actual (count actual) :expected (count expected)})))
  (reduce (fn [summary [index a b]]
            (let [a (double a) b (double b)
                  finite? (and (Double/isFinite a) (Double/isFinite b))
                  absolute (Math/abs (- a b))]
              (if finite?
                (cond-> (update summary :max-absolute-error-of-finite-inputs max absolute)
                  (or (nil? (:worst-finite-input-coordinate summary))
                      (> absolute (get-in summary [:worst-finite-input-coordinate
                                                   :absolute-error])))
                  (assoc :worst-finite-input-coordinate
                         {:index index :actual a :expected b :absolute-error absolute}))
                (-> summary
                    (update :nonfinite-coordinate-count inc)
                    (update :first-nonfinite-coordinate
                            #(or % {:index index :actual a :expected b}))))))
          {:length (count actual) :max-absolute-error-of-finite-inputs 0.0
           :worst-finite-input-coordinate nil :nonfinite-coordinate-count 0
           :first-nonfinite-coordinate nil}
          (map vector (range) actual expected)))

(defn- prepare-chain [train target-device cfg weights adapters input target]
  (let [n (* (:seq cfg) (:d cfg))
        options {:compiler :equation-first :target target-device :dtype :float :inline? true}
        adapter-symbols (mapv (comp symbol name) @(train 'adapter-keys))
        constant-symbols (conj (mapv (comp symbol name) @(train 'weight-keys)) 'zero)
        layers (mapv (fn [layer]
                       (let [theta ((train 'layer-theta) cfg layer)
                             zero (float-array n)
                             x (if (zero? layer) input (float-array n))]
                         {:forward
                          (compiled/lower (train 'gblock-fwd!)
                                          ((train 'fwd-args) cfg theta (weights layer)
                                           (adapters layer) x zero (float-array n))
                                          (assoc options :constants constant-symbols :outputs '[yo]))
                          :backward
                          (compiled/lower (train 'gblock-vjp-step!)
                                          ((train 'bwd-args) cfg theta (weights layer)
                                           (adapters layer) x (float-array n) zero (float-array n) 1.0)
                                          (assoc options :constants constant-symbols
                                                 :donate adapter-symbols :outputs '[dxo]))}))
                     [0 1])
        seed (compiled/lower #'loss/mse-grad [(float-array n) target (/ 2.0 n) (long n)] options)
        components [{:id :fwd0 :program (:forward (layers 0))}
                    {:id :fwd1 :program (:forward (layers 1))}
                    {:id :seed :program seed}
                    {:id :bwd1 :program (:backward (layers 1))}
                    {:id :bwd0 :program (:backward (layers 0))}]
        prepared
        (compiled/compose
         {:id :external-two-layer-training :components components
          :connections [{:from [:fwd0 :yo] :to [:fwd1 :x]}
                        {:from [:fwd0 :yo] :to [:bwd1 :x]}
                        {:from [:fwd1 :yo] :to [:seed :pred]}
                        {:from [:seed :result] :to [:bwd1 :g]}
                        {:from [:bwd1 :dxo] :to [:bwd0 :g]}]
          :shares (into [[[:fwd0 :x] [:bwd0 :x]]]
                        (for [layer [0 1] sym constant-symbols]
                          [[(keyword (str "fwd" layer)) (keyword (name sym))]
                           [(keyword (str "bwd" layer)) (keyword (name sym))]]))
          :mutable-shares
          (vec (for [layer [0 1] sym adapter-symbols]
                 {:owner [(keyword (str "bwd" layer)) (keyword (name sym))]
                  :borrowers [[(keyword (str "fwd" layer)) (keyword (name sym))]]
                  :output [(keyword (str "bwd" layer)) (keyword (str (name sym) "'"))]}))
          :outputs
          (into [{:key :prediction :from [:fwd1 :yo]} {:key :dx0 :from [:bwd0 :dxo]}]
                (for [layer [0 1] sym adapter-symbols]
                  {:key [layer (keyword (name sym))]
                   :from [(keyword (str "bwd" layer)) (keyword (str (name sym) "'"))]}))})]
    (when-not (and (every? #(zero? (get-in (compiled/plan (:program %))
                                         [:attributes :driver-allocations])) components)
                   (= 28 (count (:donated prepared)))
                   (not-any? (set [[:fwd1 :x] [:bwd1 :x] [:seed :pred] [:bwd1 :g] [:bwd0 :g]])
                             (map :key (:in-tree prepared))))
      (throw (ex-info "external training composition retains an unexpected boundary"
                      {:reason :external-training-composition})))
    prepared))

(defn run-loaded!
  "Run two full resident updates against the unchanged CPU monolithic AD oracle.
   Downloads are oracle reads only. Binding metadata is not a replay or timing witness.
   Every native artifact closes, including on mismatch or failed inspection."
  [{:keys [train oracle]} target-device]
  (let [cfg @(oracle 'chain-cfg)
        n (* (:seq cfg) (:d cfg))
        weights (mapv #((train 'rand-weights) cfg %) [11 12])
        adapters (mapv #((train 'init-adapters) cfg % 0.02) [21 22])
        input ((oracle 'fvec) n 31 0.5) target ((oracle 'fvec) n 32 0.5)
        started (System/nanoTime)
        prepared (prepare-chain train target-device cfg weights adapters input target)
        preparation-ns (- (System/nanoTime) started)
        started (System/nanoTime)
        live (compiled/instantiate! prepared)
        binding-ns (- (System/nanoTime) started)]
    (try
      (let [bound-schedules (compiled/execution-info live)]
       {:revision source-revision :target target-device :config cfg
       :preparation-ns preparation-ns :binding-ns binding-ns
       :bound-schedules bound-schedules
       :numerical-calls (mapv #(count (filter :graph (get-in % [:call :steps])))
                             (:instances (compiled/plan prepared)))
       :replays
       (loop [iteration 0 current adapters previous nil results []]
         (if (= iteration 2)
           results
           (let [arguments (into (reference-args oracle cfg weights current input target)
                                 (map #((train 'layer-theta) cfg %) [0 1]))
                 vg (apply (reverse/value+grad (oracle 'ref-loss2)) arguments)
                 outputs (live {})
                 predicted-loss (double (loss/mse-loss (value/->host (:prediction outputs)) target n))
                 reference-loss (double (first vg))
                 loss-error (Math/abs (- predicted-loss reference-loss))
                 dx-error (checked-error oracle (value/->host (:dx0 outputs)) (nth vg 1))
                 updated (mapv (fn [layer]
                                 (into {} (for [key @(train 'adapter-keys)]
                                            [key (value/->host (get outputs [layer key]))]))) [0 1])
                 recovered (mapv (fn [layer]
                                   (doseq [key @(train 'adapter-keys)]
                                     (when-not (= (count ((current layer) key))
                                                  (count ((updated layer) key)))
                                       (throw (ex-info "resident adapter shape changed"
                                                       {:reason :external-training-shape
                                                        :layer layer :adapter key}))))
                                   ((oracle 'recovered-grads) (current layer) (updated layer)))
                                 [0 1])
                 errors (vec (for [layer [0 1] key @(train 'adapter-keys)]
                               (let [gradient ((recovered layer) key)
                                     expected (nth vg (+ 2 (* layer 27)
                                                         (get @(oracle 'adapter-pos) key)))]
                                 (checked-error oracle gradient expected))))]
             (when-not (and (Double/isFinite predicted-loss)
                            (< loss-error (+ 1.0e-6 (* 1.0e-4 (Math/abs reference-loss))))
                            (< dx-error 2.0e-2) (= 28 (count errors))
                            (every? #(< % 2.0e-2) errors)
                            (or (nil? previous) (not (value/live? (:prediction previous)))))
               (throw (ex-info "external chain differs from the monolithic CPU AD oracle"
                               {:reason :external-training-parity :iteration iteration
                                :predicted-loss predicted-loss :reference-loss reference-loss
                                :loss-error loss-error :dx-error dx-error :adapter-errors errors
                                :bound-schedules bound-schedules
                                :dx-diagnostics
                                (difference-summary (value/->host (:dx0 outputs)) (nth vg 1))
                                :adapter-diagnostics
                                (vec (for [layer [0 1]
                                           [index key] (map-indexed vector @(train 'adapter-keys))]
                                       (assoc (difference-summary
                                               ((recovered layer) key)
                                               (nth vg (+ 2 (* layer 27)
                                                          (get @(oracle 'adapter-pos) key))))
                                              :layer layer :adapter key
                                              :oracle-relative-error
                                              (errors (+ (* layer 14) index)))))})))
             (recur (inc iteration) updated outputs
                    (conj results {:loss predicted-loss :reference-loss reference-loss
                                   :loss-error loss-error :dx-error dx-error
                                   :adapter-count (count errors) :adapter-max-error (apply max errors)})))))})
      (finally (compiled/close! live)))))

(defn run! [{:keys [source-root targets] :or {targets [:ocl:0 :ze:0]}}]
  (when-not (and (string? source-root) (seq source-root) (vector? targets) (seq targets)
                 (every? #{:ocl:0 :ze:0} targets) (= (count targets) (count (distinct targets))))
    (throw (ex-info "provide a pinned source checkout and unique local native targets"
                    {:reason :external-training-options})))
  (let [loaded (load-sources! source-root)]
    (mapv #(run-loaded! loaded %) targets)))

(defn -main [options]
  (prn (run! (edn/read-string options))))
