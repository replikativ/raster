(ns raster.compiler.ir.ordered-invocation-realization-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.backend.gpu.parallel-program-c-family :as program-c-family]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-source :as invocation-source]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.passes.parallel.structured-control-route :as route]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attention]
            [raster.par :as par]))

(deftm source-sized-output
  [x :- (Array double), first-output :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! first-output i 1 double (+ (aget x i) 1.0))
        out (double-array n)]
    (par/map! out i n double 1.0)
    out))

(deftm shape-before-first-stage
  [x :- (Array double), out :- (Array double)] :- (Array double)
  (let [cnt (alength x)
        written (par/map! out i cnt double (+ (aget x i) 1.0))]
    written))

(deftm ordered-prefix-buffer-return
  [out :- (Array double), n :- Long] :- Object
  (let [written (par/map! out i 2 double (+ 1.0 (double i)))
        late (double-array n)
        late-written (par/map! late j n double 0.0)]
    [written late-written]))

(deftm ordered-unused-constructor-return
  [out :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! out i 2 double (+ 1.0 (double i)))
        unused (double-array n)]
    written))

(deftm ordered-prefix-zero-buffer-return
  [early :- (Array double), n :- Long] :- Object
  (let [first-write (par/map! early i 2 double (+ 1.0 (double i)))
        out (double-array 8)
        written (par/map! out i 2 double (+ 1.0 (double i)))
        late (double-array n)
        late-written (par/map! late j n double 0.0)]
    [written late-written]))

(deftm ordered-dynamic-prefix-return
  [out :- (Array double), prefix :- Long, n :- Long] :- Object
  (let [written (par/map! out i prefix double (double i))
        late (double-array n)
        late-written (par/map! late j n double 0.0)]
    [written late-written]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest retained-materialized-aliases-do-not-create-or-strengthen-storage
  (doseq [source [(double-array [1 2 3]) nil]
          initialization [:unspecified :zero]]
    (let [abstract (av/tensor {:dtype :double :shape [3]})
          token [:test :shared-storage]
          buffer (materialization/->MaterializedBuffer token abstract [3] source initialization)
          owned {:id token :compiler-values #{'original} :abstract abstract
                 :shape [3] :source source :initialization :unspecified}
          realized {:buffers {'original token 'unrelated :other}
                    :storage {token owned} :acquired-storage #{token}}
          materialized {:program-buffers {'alias buffer}}
          retain (fn [state prefix values]
                   (#'invocation-link/retain-materialized-storage-aliases state prefix values))
          result (retain realized materialized {'alias abstract})]
      (is (= (assoc (:buffers realized) 'alias token) (:buffers result)))
      (is (= #{'original 'alias} (get-in result [:storage token :compiler-values])))
      (is (= owned (assoc (get-in result [:storage token]) :compiler-values #{'original})))
      (is (identical? source (get-in result [:storage token :source])))
      (is (= (:acquired-storage realized) (:acquired-storage result)))
      (is (= result (retain result materialized {'alias abstract})) "retention is idempotent")
      (is (= realized (retain realized materialized {})) "non-program names gain no binding")
      (let [unowned (assoc realized :storage {})]
        (is (= unowned (retain unowned materialized {'alias abstract}))
            "materialization alone does not acquire or initialize storage"))
      (doseq [[bad-state bad-prefix bad-values]
              [[(assoc-in realized [:buffers 'alias] :different-storage) materialized {'alias abstract}]
               [(assoc-in realized [:storage token :id] :different-storage) materialized {'alias abstract}]
               [realized (assoc-in materialized [:program-buffers 'alias :source] (double-array [1 2 3]))
                {'alias abstract}]
               [realized (assoc-in materialized [:program-buffers 'alias :shape] [2]) {'alias abstract}]
               [realized materialized {'alias (av/tensor {:dtype :float :shape [3]})}]
               [(assoc-in realized [:storage token :abstract] (av/tensor {:dtype :float :shape [3]}))
                materialized {'alias abstract}]]]
        (is (= :invocation-ordered-input-storage
               (reason #(retain bad-state bad-prefix bad-values)))
            "tokens cannot override source identity, views, representation, or an existing binding")))))

(deftest ordered-stages-retain-materialized-input-aliases-with-one-acquisition
  ;; Use the real frontend, source ordering, materializer and LinkPlan constructors.
  ;; Only synchronous device completion is simulated: this is a storage/lifecycle
  ;; test, not a numerical execution oracle.
  (let [alias (with-meta 'alias {:tag 'java.lang.Object :raster.type/tag 'doubles})
        identity (fn [source]
                   (vary-meta (completion/complete-value completion/object-input-contract source)
                              assoc :raster.effect/preserve-evaluation true))
        source (list 'let*
                     [alias (identity 'xs)
                      'first-write '(raster.par/map! early i 1 double
                                      (+ (clojure.core/aget alias i) 1.0))
                      'second-write '(raster.par/map! out i n double
                                       (+ (clojure.core/aget xs i) 2.0))]
                     'second-write)
        arrays {'xs :double 'early :double 'out :double}
        values (into {} (map (fn [id] [id (av/tensor {:dtype :double :shape [2]})]))
                     (keys arrays))
        options {:dtype :double :target-dialect :opencl-portable
                 :preserve-declared-array-storage? true :values values
                 :public-parameters '[xs early out n] :active-params '[xs early out n]
                 :array-types arrays :scalar-types {'n :long}}
        numerical (:program (typed-route/attempt
                             source :double arrays
                             {:scalar-types (:scalar-types options) :values values
                              :lexical-inputs (:public-parameters options)
                              :preserve-declared-array-storage? true}))
        semantic (route/promote-program numerical options)
        emitted (:program (program-c-family/emit-program
                           (route/schedule-program semantic options) options))
        initial (invocation-link/begin-ordered-realization
                 {:semantic semantic :emitted emitted :target :ocl:0}
                 [(double-array [3 4]) (double-array 2) (double-array 2) 2] nil)
        input-token (:id (get-in (materialization/prefix-bindings (:materialization initial))
                                [:program-buffers 'xs]))
        acquisitions (atom {})
        stages (atom [])
        completed
        (loop [state initial]
          (let [position (:source-cursor state)
                entry (nth (:source-order state) position nil)]
            (if-not entry
              state
              (if-let [span (get (:equation-spans state) position)]
                (let [ready (invocation-link/realize-ready-equation state (:equation span) nil)
                      next-state (:state ready)]
                  (swap! stages conj ready)
                  (doseq [token (keys (:new-storage ready))]
                    (swap! acquisitions update token (fnil inc 0)))
                  (recur (-> next-state
                             (update :acquired-storage into (keys (:new-storage ready)))
                             (assoc :pending nil :source-cursor
                                    (invocation-link/ordered-stage-source-end state next-state)))))
                (recur
                 (cond
                   (= :storage-alias (get-in state [:source-realizations position :kind]))
                   (invocation-link/advance-source-alias state nil)
                   (:invocation-step entry)
                   (invocation-link/advance-host-source
                    state scalar/evaluate-invocation-step scalar/evaluate-host-equation nil)
                   :else (update state :source-cursor inc)))))))
        final (invocation-link/finish-ordered-realization completed nil)
        first-stage (first @stages)
        second-stage (second @stages)]
    (is (= 2 (count @stages)) "distinct map domains retain two source stages")
    (is (= input-token (get-in first-stage [:state :realized :buffers 'alias])))
    (is (= input-token (get-in first-stage [:state :realized :buffers 'xs]))
        "the first fragment must retain the original name even though only its alias is read")
    (is (= input-token (get-in second-stage [:state :realized :buffers 'xs])))
    (is (= 1 (get @acquisitions input-token)) "the second name does not acquire/upload the input again")
    (is (not (contains? (:new-storage second-stage) input-token)))
    (is (some? (:plan final)))))

(deftest compound-source-spans-cannot-skip-observable-work-or-imports
  (let [semantic {:equations [{:id :host :attributes {:host-only true}}
                              {:id :producer} {:id :consumer}]}
        emitted {:equations [{:id :host :attributes {:host-only true}}
                             {:id :fused :attributes {:emitted-source-equations [:producer :consumer]}}]}
        order [{:position 0 :storage :scratch} {:position 1 :equation :host}
               {:position 2 :equation :producer} {:position 3 :equation :consumer}]
        spans (fn [program plan entries]
                (#'invocation-link/ordered-equation-spans semantic program plan entries))]
    (is (= {2 {:equation :fused :end 4}} (spans emitted {:steps []} order)))
    (is (= {2 {:equation :fused :end 4}}
           (#'invocation-link/ordered-equation-spans
            (assoc-in semantic [:equations 0 :attributes] {}) emitted {:steps []} order))
        "host execution classification comes from checked emission, even before semantic scheduling marks it")
    (doseq [action [{:invocation-step :step} {:storage :buffer} {:realization :allocation}
                    {:program-result-import {:equation :producer :result :value}}
                    {:equation :host}]]
      (is (= :invocation-ordered-compound-span
             (reason #(spans emitted {:steps []}
                             (vec (concat (take 3 order) [action] (drop 3 order))))))))
    (doseq [ids [[:consumer :producer] [:producer :missing] [:producer :producer]]]
      (is (= :invocation-ordered-compound-span
             (reason #(spans (assoc-in emitted [:equations 1 :attributes :emitted-source-equations] ids)
                             {:steps []} order)))))
    (is (= :invocation-ordered-compound-coverage
           (reason #(spans (assoc emitted :equations []) {:steps []} order))))
    (doseq [id [:producer :consumer]]
      (is (= :invocation-ordered-compound-span
             (reason #(spans emitted
                             {:steps [(invocation/map->ProgramResultImport {:equation id})]}
                             order)))
          "a constituent result import requires its own exact producer correspondence"))))

(defn- ordered-result-import-fixture []
  (let [snapshot (completion/complete-value completion/object-input-contract 'total)
        cast (completion/complete-value
              (completion/source-floating-cast-contract :float {:unchecked-math false}) 'saved)
        source (list 'let* ['total '(raster.par/reduce acc 0.0 i 3 (+ acc (clojure.core/aget xs i)))
                           'saved snapshot 'rounded cast
                           (with-meta 'out {:tag 'floats :raster.type/tag 'floats}) '(clojure.core/float-array 1)
                           'written '(raster.par/map-void! i 1 (clojure.core/aset out i rounded))] 'out)
        options {:dtype :double :target-dialect :opencl-portable :resident-reductions? true
                 :preserve-declared-array-storage? true
                 :values {'xs (av/tensor {:dtype :double :shape [3]})}
                 :public-parameters '[xs] :active-params '[xs] :array-types {'xs :double} :scalar-types {}}
        numerical (:program (typed-route/attempt source :double {'xs :double} options))
        semantic (route/promote-program numerical options)
        emitted (:program (program-c-family/emit-program (route/schedule-program semantic options) options))
        initial (invocation-link/begin-ordered-realization
                 {:semantic semantic :emitted emitted :target :ocl:0} [(double-array [1 2 3])] nil)
        position (first (keep-indexed #(when (:program-result-import %2) %1) (:source-order initial)))
        entry (nth (:source-order initial) position)
        step (first (filter #(= (:invocation-step entry) (:id %))
                            (get-in initial [:materialization :plan :steps])))
        token [:test-result-storage (:result step)]
        state (-> initial (assoc :source-cursor position)
                  (assoc-in [:realized :buffers (:result step)] token))]
    {:initial initial :state state :step step :entry entry :token token}))

(deftest ordered-program-result-import-is-a-pure-exact-cursor-transition
  (let [{:keys [initial state step entry token]} (ordered-result-import-fixture)
        callbacks (atom 0)
        forbidden (fn [& _] (swap! callbacks inc) (throw (AssertionError. "pure import cannot execute resources")))
        hostile (proxy [Object] []
                  (equals [_] (throw (AssertionError. "payload equality")))
                  (hashCode [] (throw (AssertionError. "payload hash")))
                  (toString [] (throw (AssertionError. "payload print"))))]
    (is (identical? initial (invocation-link/validate-ordered-source-actions! initial)))
    (with-redefs [materialization/materialize-next forbidden]
      (let [prepared (invocation-link/prepare-program-result-import state nil)]
        (is (identical? state (:state prepared)))
        (is (identical? step (:step prepared)))
        (is (= entry (:entry prepared)))
        (is (= (:equation step) (get-in prepared [:equation :id])))
        (is (= (:result step) (:result prepared)))
        (is (= token (:storage prepared)))
        (is (empty? (:acquired-storage (:state prepared))))
        (is (nil? (:pending (:state prepared)))))
      (doseq [payload [nil hostile]]
        (let [completed (invocation-link/complete-program-result-import state step payload nil)
              wrapper (get-in completed [:materialization :values (:id step)])]
          (is (identical? payload (:source wrapper)))
          (is (= (:id step) (:id wrapper)))
          (is (= (inc (:source-cursor state)) (:source-cursor completed)))
          (is (= 1 (get-in completed [:materialization :next-step])))
          (is (identical? (:realized state) (:realized completed)))
          (is (identical? (:acquired-storage state) (:acquired-storage completed)))
          (is (some? (reason #(invocation-link/complete-program-result-import completed step payload nil)))))))
    (is (zero? @callbacks))
    (is (= :invocation-ordered-import-step
           (reason #(invocation-link/complete-program-result-import state (into {} step) hostile nil))))
    (is (= :invocation-ordered-next-host
           (reason #(invocation-link/advance-host-source state forbidden forbidden nil))))
    (doseq [[changed caller-options expected]
            [[state {:math :changed} :invocation-ordered-request]
             [(assoc state :pending {:equation :prior}) nil :invocation-ordered-pending]
             [(assoc state :source-cursor 0) nil :invocation-ordered-next-import]
             [(update-in state [:realized :buffers] dissoc (:result step)) nil :invocation-ordered-import-storage]
             [(update-in state [:source-order (:source-cursor state) :program-result-import]
                         assoc :result 'wrong) nil :invocation-ordered-import-source]
             [(update state :source-order #(vec (reverse %))) nil :invocation-ordered-import-source]]]
      (is (= expected (reason #(invocation-link/prepare-program-result-import changed caller-options)))))))

(deftest ordered-preparation-requirement-is-static-and-owner-checked
  (let [late (equation-first/compile #'fixture/write-before-shape {:target :ocl:0 :dtype :double})
        simple (equation-first/compile #'shape-before-first-stage {:target :ocl:0 :dtype :double})
        proof (emitted-program/validate-with-physical-results! (:emitted late))
        other-proof (emitted-program/validate-with-physical-results! (:emitted simple))
        forbidden (fn [& _] (throw (ex-info "static ordering query evaluated source" {})))]
    (with-redefs [scalar/evaluate-invocation-step forbidden
                  scalar/evaluate-host-equation forbidden
                  materialization/begin-materialization forbidden
                  materialization/materialize-next forbidden]
      (let [report (invocation-link/ordered-preparation-requirement late)]
        (is (true? (:required? report)))
        (is (some #(= :after-numerical (:reason %)) (:barriers report)))
        (is (every? #(and (integer? (:position %)) (integer? (:preceding-position %)))
                    (:barriers report)))
        (is (string? (fingerprint/fingerprint report)) "diagnostics contain only portable data")
        (is (= report (invocation-link/ordered-preparation-requirement late proof nil))))
      (is (= {:required? false :barriers []}
             (invocation-link/ordered-preparation-requirement simple)))
      (is (= :emitted-parallel-program-retained-validation
             (reason #(invocation-link/ordered-preparation-requirement late other-proof nil))))
      (is (= :invocation-ordered-program
             (reason #(invocation-link/ordered-preparation-requirement
                       (assoc late :emitted (:emitted simple)))))))))

(deftest ordering-query-keeps-source-free-origin-separate-from-source-execution
  (let [tensor (av/tensor {:dtype :double :shape [3]})
        typed (soac/make
               (soac/default-program-facts
                {:values {'input tensor 'output tensor} :inputs '[input]
                 :equations {:map (soac/default-equation-facts)}})
               [(list '= :map '[output]
                      (list 'map {:index 'index :extent 3} '[input] []
                            (soac/lambda-form '[element] '[(* element 2.0)])))] '[output])
        semantic (route/promote-source-free-algorithm typed)
        options {:dtype :double :target-dialect :opencl-portable}
        emitted (:program (program-c-family/emit-program
                           (route/schedule-program semantic options) options))
        compilation {:semantic semantic :emitted emitted :target :ocl:0}
        forged (assoc-in compilation [:semantic :provenance :pass] :forged)]
    (is (invocation/source-free-program? semantic))
    (is (= {:required? false :barriers []}
           (invocation-link/ordered-preparation-requirement compilation)))
    (is (= :invocation-program-mismatch
           (reason #(invocation-link/ordered-preparation-requirement forged))))
    (is (= :invocation-source-order-boundary
           (reason #(invocation-link/begin-ordered-realization compilation [(double-array 3)] nil))))))

(deftest source-completion-prefix-requires-the-owned-executor-before-any-kernel
  (doseq [[expression required?] [[(completion/complete-value
                                   (completion/bytecode-boxed-parameter-contract Double/TYPE) 'x) true]
                                ['(clojure.core/+ x 0.0) false]]]
    (let [values (zipmap '[x y] (repeat (av/tensor {:dtype :double :shape []})))
          plan (invocation/from-prefix
                {:id :completion-prefix :parameters '[x]
                 :parameter-values (select-keys values '[x])
                 :bindings [['y expression]] :binding-values (select-keys values '[y])
                 :program-values values :program-inputs '[y] :program-outputs '[y]})
          program (parallel-program/make
                   {:dialect :opencl-parallel :values values :inputs '[y] :outputs '[y]
                    :source (list 'let* ['y expression] 'y)
                    :attributes {:invocation-plan plan}})
          forbidden (fn [& _] (throw (AssertionError. "ordering query must not execute source")))
          report (with-redefs [scalar/evaluate-invocation-step forbidden
                               materialization/materialize forbidden]
                   (invocation-link/ordered-preparation-requirement
                    {:semantic program :emitted program :target :ocl:0}))]
      (is (= required? (:required? report)))
      (is (= (if required? [:source-completion] []) (mapv :reason (:barriers report)))))))

(deftest invocation-prefix-cannot-cross-an-earlier-host-equation
  (let [values {'n (av/tensor {:dtype :long :shape []})
                'xs (av/tensor {:dtype :double :shape '[(extent xs)]})
                'checked (av/tensor {:dtype :int :shape []})
                'length (av/tensor {:dtype :int :shape []})}
        expression '(clojure.core/int n)
        algorithm (soac/make
                   (soac/default-program-facts
                    {:values (select-keys values '[n checked]) :inputs '[n]
                     :equations {:check (soac/default-equation-facts)}})
                   [(list '= :check '[checked]
                          (list 'scalar {:dtypes [:int]} '[n]
                                (soac/lambda-form '[n] [expression])))] '[checked])
        equation (parallel-program/map->ProgramEquation
                  {:id :check :site [:binding 'checked] :source expression
                   :operands '[n] :results '[checked] :algorithm algorithm
                   :operations [] :effects #{} :provenance {} :attributes {:host-only true}})
        plan (invocation/from-prefix
              {:id :host-before-prefix :parameters '[n xs]
               :parameter-values (select-keys values '[n xs])
               :bindings [['length '(clojure.core/alength xs)]]
               :binding-values (select-keys values '[length]) :program-values values
               :program-inputs '[n xs] :program-outputs '[checked]})
        program (parallel-program/make
                 {:dialect :opencl-parallel :values values :inputs '[n xs] :outputs '[checked]
                  :equations [equation]
                  :source '(let* [checked (clojure.core/int n)
                                  length (clojure.core/alength xs)] checked)
                  :attributes {:invocation-plan plan}})]
    (is (= {:required? true
            :barriers [{:reason :prefix-after-host :position 1 :preceding-position 0
                        :action :invocation-step}]}
           (invocation-link/ordered-preparation-requirement
            {:semantic program :emitted program :target :ocl:0})))))

(deftest ordered-value-aliases-preserve-the-existing-buffer-without-execution
  (let [saved (with-meta 'saved {:tag 'java.lang.Object :raster.type/tag 'doubles})
        chained (with-meta 'chained {:tag 'java.lang.Object :raster.type/tag 'doubles})
        identity (fn [source]
                   (vary-meta (completion/complete-value completion/object-input-contract source)
                              assoc :raster.effect/preserve-evaluation true))
        source (list 'let* [saved (identity 'x) chained (identity saved)
                            'y '(raster.par/map! out i n double (clojure.core/aget x i))] 'y)
        options {:dtype :double :public-parameters '[x out n]
                 :array-types {'x :double 'out :double} :scalar-types {'n :long}}
        promote (fn [source]
                  (route/promote-program
                   (:program (typed-route/attempt
                              source :double (:array-types options)
                              {:scalar-types (:scalar-types options)
                               :lexical-inputs (:public-parameters options)})) options))
        semantic (promote source)
        emitted (:program (program-c-family/emit-program
                           (route/schedule-program semantic options)
                           (assoc options :target-dialect :opencl-portable)))
        compilation {:semantic semantic :emitted emitted :target :ocl:0}
        input (double-array [1 2 3])
        initial (invocation-link/begin-ordered-realization compilation [input (double-array 3) 3] nil)
        forbidden (fn [& _] (throw (ex-info "alias evaluated source or acquired storage" {})))
        first-alias (first (get-in initial [:materialization :plan :steps]))
        second-alias (second (get-in initial [:materialization :plan :steps]))
        advanced (invocation-link/advance-host-source initial forbidden forbidden nil)
        finished (invocation-link/advance-host-source advanced forbidden forbidden nil)
        buffer (fn [state symbol]
                 (let [prefix (:materialization state)]
                   (get-in prefix [:values (get (:lexical prefix) symbol)])))]
    (is (identical? initial (invocation-link/validate-ordered-source-actions! initial)))
    (is (every? invocation/value-alias? [first-alias second-alias]))
    (is (not-any? #{saved chained} (:inputs semantic)) "both preserved aliases are numerically dead")
    (doseq [entry (take 2 (:source-order initial))]
      (is (some? (:invocation-step entry)))
      (is (nil? (:storage entry)))
      (is (nil? (:equation entry))))
    (is (= 1 (:source-cursor advanced)))
    (is (= 2 (:source-cursor finished)))
    (is (= 2 (get-in finished [:materialization :next-step])))
    (is (identical? (buffer initial 'x) (buffer advanced saved)))
    (is (identical? (buffer initial 'x) (buffer finished chained)))
    (is (identical? input (:source (buffer finished chained))))
    (is (not-any? #{saved chained}
                  (keys (:program-buffers (materialization/prefix-bindings (:materialization finished)))))
        "executing dead source identities does not manufacture numerical inputs")
    (is (= (:realized initial) (:realized finished)))
    (is (empty? (:acquired-storage finished)))
    (is (empty? (:host-scalars finished)))
    (is (empty? (:host-steps finished)))
    (is (= :invocation-ordered-next-host
           (reason #(invocation-link/advance-host-source finished forbidden forbidden nil)))
        "the alias executor cannot advance the following numerical source action")
    (is (= :invocation-ordered-pending
           (reason #(invocation-link/advance-host-source
                     (assoc initial :pending {:equation :earlier}) forbidden forbidden nil))))
    (let [changed (assoc-in compilation
                            [:semantic :attributes :invocation-plan :steps 0 :source-expression]
                            (identity 'out))]
      (is (= :invocation-value-alias-source
             (reason #(invocation-link/begin-ordered-realization changed [input (double-array 3) 3] nil)))))
    (is (= :host-abstract-value-reference-alias
           (reason #(promote (list 'let* [chained (identity saved) saved (identity 'x)
                                          'y '(raster.par/map! out i n double (clojure.core/aget x i))]
                                  'y)))))))

(deftest early-equation-binds-without-evaluating-the-late-checked-shape
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        invocation (get-in compilation [:semantic :attributes :invocation-plan])
        source (double-array [7.0])
        destination (double-array 1)
        state (materialization/begin-materialization invocation
                                                     [source destination Long/MAX_VALUE])
        available (materialization/prefix-bindings state)
        buffers (into {} (map (fn [[id buffer]] [id (:id buffer)]))
                      (:program-buffers available))
        scalars (:program-scalars available)
        program (:emitted compilation)
        proof (emitted-program/validate-with-physical-results! program)
        early (first (:equations program))
        constructors [#(program-call/make-equation-call program %1 %2 %3 {} nil)
                      #(program-call/make-equation-call program %1 %2 %3 {} proof nil)]
        {:keys [call]} (program-call/make-equation-call program (:id early) buffers scalars {} nil)]
    (is (= 0 (:next-step state)))
    (is (= #{'x 'first-output} (set (keys (:program-buffers available)))))
    (is (= #{'n} (set (keys scalars))))
    (is (identical? source (get-in available [:program-buffers 'x :source])))
    (is (identical? destination (get-in available [:program-buffers 'first-output :source])))
    (is (not (materialization/materialized-invocation? available)))
    (is (= :invocation-materialization-incomplete
           (reason #(materialization/finish-materialization state))))
    (is (program-call/emitted-equation-call? call))
    (is (= (:id early) (get-in call [:equation :id])))
    (is (= (get buffers 'first-output) (get-in call [:outputs 'written])))
    (is (not (contains? (:buffers call) 'out)))
    (doseq [bindings [buffers (into {} (map (fn [[id _]] [id [:different-runtime-binding id]])) buffers)]]
      (is (= ((first constructors) (:id early) bindings scalars)
             ((second constructors) (:id early) bindings scalars))
          "static reuse does not retain runtime buffer bindings"))
    (doseq [construct constructors
            id (concat (:results early) ['first-output])]
      (is (= :emitted-program-device-result-scalar
             (reason #(construct (:id early) buffers
                                 (assoc scalars id {:type :double :value 0.0}))))))
    (doseq [construct constructors
            equation-id [(:id (second (:equations program))) :absent]]
      (is (= :emitted-program-ready-equation
             (reason #(construct equation-id buffers scalars)))))
    (doseq [construct constructors]
      (is (= :emitted-program-result-buffer
             (reason #(construct (:id (last (:equations program))) buffers scalars)))))))

(deftest ordered-stages-reuse-only-their-exact-static-validation
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        validate emitted-program/validate-with-physical-results!
        checks (atom 0)]
    (with-redefs [emitted-program/validate-with-physical-results!
                  (fn [& args] (swap! checks inc) (apply validate args))]
      (let [state (invocation-link/begin-ordered-realization
                   compilation [(double-array [7.0]) (double-array 1) Long/MAX_VALUE] nil)
            equation-id (:id (first (:equations (:program state))))
            ready (invocation-link/realize-ready-equation state equation-id nil)]
        (is (= 2 @checks) "parent and fresh fragment each validate once, not once per consumer")
        (let [fragment-validation (:fragment-validation ready)
              repeated (invocation-link/realize-ready-equation
                        state equation-id fragment-validation nil)
              fragment (:program fragment-validation)]
          (is (identical? fragment-validation (:fragment-validation repeated)))
          (is (= 2 @checks)
              "an exact owner-local fragment proof avoids only repeated fragment validation")
          (is (= 0 (get-in repeated [:state :source-cursor])))
          (is (= equation-id (get-in repeated [:state :pending :equation])))
          (is (empty? (get-in repeated [:state :acquired-storage]))
              "reusing immutable proof still grants neither acquisition nor readiness")
          (let [foreign-parent (with-meta (:program state) {:copy true})]
            (is (= foreign-parent (:program state)))
            (is (= :emitted-parallel-program-fragment-validation
                   (reason #(emitted-program/checked-equation-fragment-validation!
                             foreign-parent equation-id fragment-validation nil)))))
          (is (= :emitted-parallel-program-fragment-validation
                 (reason #(emitted-program/checked-equation-fragment-validation!
                           (:program state) :wrong-equation fragment-validation nil))))
          (is (= :emitted-parallel-program-fragment-validation
                 (reason #(emitted-program/checked-equation-fragment-validation!
                           nil nil (:retained-validation state) nil))))
          (is (= :emitted-parallel-program-retained-validation
                 (reason #(invocation-link/realize-ready-equation
                           state equation-id (assoc fragment-validation :copy true) nil))))
          (is (= :emitted-parallel-program-retained-validation
                 (reason #(invocation-link/realize-ready-equation
                           state equation-id
                           (assoc fragment-validation :program
                                  (with-meta fragment {:copy true})) nil))))
          (let [options {:scalar-math {:overrides {[:tanh :float]
                                                   :f64-target-library-rte-f32}}}]
            (is (= :emitted-parallel-program-retained-validation
                   (reason #(invocation-link/realize-ready-equation
                             (assoc state :caller-options options) equation-id
                             fragment-validation options)))))
          (is (= 2 @checks) "foreign or copied evidence is refused without revalidation"))
        (let [other-state (invocation-link/begin-ordered-realization
                           compilation [(double-array [8.0]) (double-array 1) Long/MAX_VALUE] nil)
              other-ready (invocation-link/realize-ready-equation
                           other-state equation-id nil)]
          (is (= 4 @checks) "a separate ordered owner independently validates its fragment")
          (is (not (identical? (:fragment-validation ready)
                               (:fragment-validation other-ready)))))
        (is (identical? (:program state) (get-in state [:retained-validation :program])))
        (is (identical? (:projections state) (get-in state [:retained-validation :projections])))
        (is (= 0 (get-in ready [:state :source-cursor])))
        (is (= equation-id (get-in ready [:state :pending :equation])))
        (is (empty? (get-in ready [:state :acquired-storage]))
            "static reuse grants neither completion nor acquired storage")
        (is (= :invocation-ordered-projections
               (reason #(invocation-link/realize-ready-equation
                         (assoc state :projections (java.util.IdentityHashMap.)) equation-id nil))))
        (doseq [changed [(assoc state :program (with-meta (:program state) {:copy true}))
                         (update state :retained-validation assoc :copy true)]]
          (is (= :emitted-parallel-program-retained-validation
                 (reason #(invocation-link/realize-ready-equation changed equation-id nil)))))
        (let [options {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}]
          (is (= :emitted-parallel-program-retained-validation
                 (reason #(invocation-link/realize-ready-equation
                           (assoc state :caller-options options) equation-id options)))))
        (is (= 4 @checks) "forged reports are refused without whole-program reconstruction")))))

(deftest ready-storage-is-conditional-not-completed-invocation-storage
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        state (invocation-link/begin-ordered-realization
               compilation [(double-array [7.0]) (double-array 1) Long/MAX_VALUE] nil)
        equation-id (:id (first (:equations (:program state))))
        ready (invocation-link/realize-ready-equation state equation-id nil)]
    (is (= (inc (:source-cursor state))
           (invocation-link/ordered-stage-source-end state (:state ready))))
    (is (= :invocation-ordered-stage-source-span
           (reason #(invocation-link/ordered-stage-source-end
                     state (update-in (:state ready) [:pending :source-end] inc)))))
    (is (= :invocation-ordered-next-equation
           (reason #(invocation-link/realize-ready-equation
                     (update-in state [:equation-spans (:source-cursor state) :end] inc)
                     equation-id nil))))
    (is (= 1 (count (get-in ready [:stage-plan :instances]))))
    (is (= 0 (get-in ready [:state :source-cursor])))
    (is (= equation-id (get-in ready [:state :pending :equation])))
    (is (seq (get-in ready [:entry-contract :reads])))
    (is (seq (get-in ready [:entry-contract :writes])))
    (is (not (contains? (get-in ready [:state :realized :buffers]) 'out)))
    (is (not (contains? (get-in ready [:state :realized :buffers]) 'square)))
    (is (= :invocation-ordered-pending
           (reason #(invocation-link/realize-ready-equation (:state ready) equation-id nil))))
    (is (= :invocation-ordered-next-equation
           (reason #(invocation-link/realize-ready-equation
                     state (:id (last (:equations (:program state)))) nil))))
    (let [derived-only (assoc state :realized (get-in ready [:state :realized]))
          repeated (invocation-link/realize-ready-equation derived-only equation-id nil)
          acquired (assoc derived-only :acquired-storage (set (keys (:new-storage ready))))
          borrowed (invocation-link/realize-ready-equation acquired equation-id nil)]
      (is (= (set (keys (:new-storage ready))) (set (keys (:new-storage repeated)))))
      (is (every? :source (vals (get-in repeated [:stage-plan :nodes]))))
      (is (empty? (:new-storage borrowed)))
      (is (not-any? :source (vals (get-in borrowed [:stage-plan :nodes]))))
      (is (seq (get-in borrowed [:entry-contract :requires]))))))

(deftest ready-graph-retains-a-logical-prefix-without-initializing-its-tail
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        state (invocation-link/begin-ordered-realization
               compilation [(double-array [7.0]) (double-array [10 11 12 13]) Long/MAX_VALUE] nil)
        equation-id (:id (first (:equations (:program state))))
        ready (invocation-link/realize-ready-equation state equation-id nil)
        call (:call ready)
        [result physical] (first (:result-views (first (:steps call))))
        prefix (get (:buffers call) result)
        base (get (:buffers call) physical)
        plan (:stage-plan ready)
        fresh (-> plan (assoc-in [:nodes base :source] nil)
                  (assoc-in [:nodes base :role] :internal))
        evidence (:effect-evidence (link-plan/validate-with-effect-evidence! fresh))
        contract (first (link-plan/relative-entry-contracts!
                         fresh evidence [[(:id (first (:instances fresh)))]]))]
    (is (= 'first-output physical))
    (is (not= prefix base))
    (is (= [1] (get-in plan [:nodes prefix :view :shape])))
    (is (= [4] (get-in plan [:nodes base :view :shape])))
    (is (= (get-in plan [:nodes prefix :view :allocation])
           (get-in plan [:nodes base :view :allocation])))
    (is (nil? (get-in plan [:nodes prefix :source])))
    (is (contains? (:full-produces contract) prefix))
    (is (not (contains? (:full-produces contract) base)))
    (is (not (contains? (:requires contract) base)))
    (is (contains? (:writes contract) base))
    (let [acquired (-> state
                       (assoc :realized (get-in ready [:state :realized])
                              :acquired-storage
                              (into #{} (map #(get-in % [:view :allocation :id]))
                                    (vals (:nodes plan)))))
          repeated (invocation-link/realize-ready-equation acquired equation-id nil)]
      (is (not-any? :source (vals (get-in repeated [:stage-plan :nodes])))
          "acquired roots cannot re-upload an earlier host initializer")
      (is (contains? (get-in repeated [:entry-contract :full-produces]) prefix))
      (is (not (contains? (get-in repeated [:entry-contract :full-produces]) base))))
    (is (= :link-unproduced-output
           (reason #(link-plan/validate! (assoc fresh :outputs [base])))))
    (is (nil? (reason #(link-plan/validate! (assoc plan :outputs [base]))))
        "an initialized backing tail can be returned, without whole-write credit")
    (is (= :program-link-result-view
           (reason #(link-plan/validate! (assoc-in fresh [:nodes prefix :view :byte-offset] 8)))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (link-plan/validate!
                  (assoc-in fresh [:nodes prefix :view :shape] [5]))))
    (is (= 0 (get-in ready [:state :source-cursor])))
    (is (empty? (get-in ready [:state :acquired-storage])))
    (is (= :invocation-ordered-pending
           (reason #(invocation-link/realize-ready-equation (:state ready) equation-id nil))))
    (is (= :invocation-ordered-incomplete
           (reason #(invocation-link/finish-ordered-realization (:state ready) nil))))))

(deftest an-empty-ordered-prefix-does-not-initialize-the-nonempty-backing-buffer
  (let [compilation (equation-first/compile
                     #'ordered-dynamic-prefix-return
                     {:target :ocl:0 :dtype :double
                      :values {'out (av/tensor {:dtype :double :shape [8]})}})]
    (doseq [count [0 2]]
      (let [state (invocation-link/begin-ordered-realization
                   compilation [(double-array 8) count 1] nil)
            ready (invocation-link/realize-ready-equation
                   state (:id (first (:equations (:program state)))) nil)
            call (:call ready)
            [result physical] (first (:result-views (first (:steps call))))
            prefix (get (:buffers call) result) base (get (:buffers call) physical)
            fresh (-> (:stage-plan ready) (assoc-in [:nodes base :source] nil)
                      (assoc-in [:nodes base :role] :internal))
            evidence (:effect-evidence (link-plan/validate-with-effect-evidence! fresh))
            contract (first (link-plan/relative-entry-contracts!
                             fresh evidence [[(:id (first (:instances fresh)))]]))]
        (is (= [count] (get-in fresh [:nodes prefix :view :shape])))
        (is (= [8] (get-in fresh [:nodes base :view :shape])))
        (is (not (contains? (:full-produces contract) base)))
        (is (= :link-unproduced-output
               (reason #(link-plan/validate! (assoc fresh :outputs [base])))))))))

(deftest initial-shape-definition-does-not-require-acquired-buffer-storage
  (let [compilation (equation-first/compile #'shape-before-first-stage
                                           {:target :ocl:0 :dtype :double})
        initial (invocation-link/begin-ordered-realization
                 compilation [(double-array [1 2 3]) (double-array 3)] nil)
        entry (first (:source-order initial))
        advanced (invocation-link/advance-host-source initial nil nil nil)]
    (is (= 'cnt (:binding entry)))
    (is (some? (:invocation-step entry)))
    (is (nil? (:equation entry)) "this public shape is represented only by an invocation step")
    (is (= 1 (:source-cursor advanced)))
    (is (= 3 (get-in (materialization/prefix-bindings (:materialization advanced))
                     [:program-scalars 'cnt :value])))
    (is (= (:realized initial) (:realized advanced)) "shape evaluation acquires no storage")
    (is (empty? (:acquired-storage advanced)))
    (let [id 'x
          conflicting (assoc-in initial [:realized :buffers id] :different-root)]
      (is (= :invocation-ordered-host-buffer
             (reason #(invocation-link/advance-host-source conflicting nil nil nil)))))))

(deftest shared-initial-shape-definition-binds-unacquired-public-storage
  ;; Legal shared representation built through the same constructors and source attestation.
  ;; The natural public fixture above legitimately canonicalizes this to an invocation step only.
  (let [values {'xs (av/tensor {:dtype :double :shape '[(extent xs)]})
                'cnt (av/tensor {:dtype :long :shape []})}
        expression '(clojure.core/long (clojure.core/alength xs))
        algorithm (soac/make
                   (soac/default-program-facts
                    {:values values :inputs '[xs]
                     :equations {:length (soac/default-equation-facts)}})
                   [(list '= :length '[cnt]
                          (list 'scalar {:dtypes [:long]} '[xs]
                                (soac/lambda-form '[xs] [expression])))] '[cnt])
        equation (parallel-program/map->ProgramEquation
                  {:id :length :site [:binding 'cnt] :source expression
                   :operands '[xs] :results '[cnt] :algorithm algorithm
                   :operations [] :effects #{} :provenance {} :attributes {:host-only true}})
        plan (invocation/from-prefix
              {:id :shared-shape :parameters '[xs] :parameter-values (select-keys values '[xs])
               :bindings [['cnt expression]] :binding-values (select-keys values '[cnt])
               :program-values values :program-inputs '[xs] :program-outputs '[cnt]})
        program (parallel-program/make
                 {:dialect :opencl-parallel :values values :inputs '[xs] :outputs '[cnt]
                  :equations [equation] :source (list 'let* ['cnt expression] 'cnt)
                  :attributes {:invocation-plan plan}})
        initial (invocation-link/begin-ordered-realization
                 {:semantic program :emitted program :target :ocl:0} [(double-array 3)] nil)
        entry (first (:source-order initial))
        advanced (invocation-link/advance-host-source initial nil nil nil)
        host-step (get-in advanced [:host-steps :length])]
    (is (some? (:invocation-step entry)))
    (is (= :length (:equation entry)))
    (is (= 1 (:source-cursor advanced)))
    (is (= 1 (get-in advanced [:materialization :next-step])))
    (is (= {'cnt {:type :long :value 3}} (:results host-step)))
    (is (= (get-in initial [:materialization :values
                            (get-in initial [:materialization :lexical 'xs]) :id])
           (get-in host-step [:operands 'xs])))
    (is (= (:realized initial) (:realized advanced)))
    (is (empty? (:acquired-storage advanced)))
    (let [host-only-plan (invocation/from-prefix
                          {:id :host-only-shape :parameters '[xs]
                           :parameter-values (select-keys values '[xs])
                           :bindings [] :binding-values {} :program-values values
                           :program-inputs '[xs] :program-outputs '[cnt]})
          host-only-program (assoc-in program [:attributes :invocation-plan] host-only-plan)
          start (invocation-link/begin-ordered-realization
                 {:semantic host-only-program :emitted host-only-program :target :ocl:0}
                 [(double-array 3)] nil)
          calls (atom 0)
          context-vars [#'link-plan/*validated-program-instances*
                        #'link-plan/*retained-program-validations*
                        #'link-plan/*caller-options*
                        #'program-call/*validated-boundary-projections*
                        #'program-call/*validated-projection-policy*]
          end (with-bindings (zipmap context-vars (repeat ::outer-proof-context))
                (invocation-link/advance-host-source
               start nil
               (fn [equation context]
                 (swap! calls inc)
                 (is (every? nil? (map deref context-vars)))
                 (is (every? nil? @(future (mapv deref context-vars)))
                     "host callbacks and their futures inherit no compiler proof context")
                 (is (= {'xs [3]} (:buffer-shapes context)))
                 (scalar/evaluate-host-equation *ns* equation context)) nil))]
      (is (= 1 @calls))
      (is (= {'cnt {:type :long :value 3}} (:host-scalars end)))
      (is (zero? (get-in end [:materialization :next-step])))
      (is (empty? (:acquired-storage end))))))

(deftest shared-host-definition-is-evaluated-once-at-its-source-position
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})]
    (doseq [n [3 Long/MAX_VALUE]]
      (let [initial (invocation-link/begin-ordered-realization
                     compilation [(double-array [7.0]) (double-array 1) n] nil)
            at-host (assoc initial :source-cursor 1)
            calls (atom 0)
            evaluate (fn [step operands]
                       (swap! calls inc)
                       (scalar/evaluate-invocation-step (:source-ns compilation) step operands))
            duplicate (fn [& _] (throw (ex-info "shared host definition evaluated twice" {})))]
        (is (= :invocation-ordered-next-host
               (reason #(invocation-link/advance-host-source initial evaluate duplicate nil))))
        (is (zero? @calls))
        (is (= :invocation-ordered-pending
               (reason #(invocation-link/advance-host-source
                         (assoc at-host :pending {:equation :earlier}) evaluate duplicate nil))))
        (is (zero? @calls))
        (if (= n Long/MAX_VALUE)
          (is (thrown? ArithmeticException
                       (invocation-link/advance-host-source at-host evaluate duplicate nil)))
          (let [advanced (invocation-link/advance-host-source at-host evaluate duplicate nil)
                equation (:equation (nth (:source-order initial) 1))
                result (get-in advanced [:host-steps equation :results 'square])]
            (is (= 2 (:source-cursor advanced)))
            (is (= 1 (get-in advanced [:materialization :next-step])))
            (is (= {:type :long :value 9} result))
            (is (= result (get-in advanced [:host-scalars 'square])))
            (is (= (:realized initial) (:realized advanced)))
            (is (empty? (:acquired-storage advanced)))))
        (is (= 1 @calls))
        (is (= 0 (get-in initial [:materialization :next-step])))
        (is (= 1 (:source-cursor at-host)))))))

(deftest source-allocation-retains-its-own-conversion-and-initialization-boundary
  (let [compilation (equation-first/compile #'source-sized-output {:target :ocl:0 :dtype :double})
        calls (atom 0)
        evaluate (fn [expression scalars]
                   (swap! calls inc)
                   (scalar/evaluate-array-size (:source-ns compilation)
                                                (update-vals scalars :value) expression))]
    (doseq [n [-1 2147483648 4294967296 0 3]]
      (let [initial (invocation-link/begin-ordered-realization
                     compilation [(double-array [7.0]) (double-array 1) n] nil)
            position (first (keep-indexed (fn [i entry] (when (= 'out (:storage entry)) i))
                                           (:source-order initial)))
            state (assoc initial :source-cursor position)]
        (is (= 1 position) "allocation belongs after the early write")
        (reset! calls 0)
        (cond
          (contains? #{-1 2147483648} n)
          (is (thrown? NegativeArraySizeException
                       (invocation-link/prepare-source-allocation state evaluate nil)))

          (= 4294967296 n)
          (is (= :invocation-ordered-allocation-shape
                 (reason #(invocation-link/prepare-source-allocation state evaluate nil))))

          :else
          (let [prepared (invocation-link/prepare-source-allocation state evaluate nil)]
            (is (= n (:elements prepared)))
            (is (= :zero (:initialization prepared)))
            (is (= position (get-in prepared [:state :source-cursor])))
            (is (= :source-allocation (get-in prepared [:state :pending :kind])))
            (is (empty? (get-in prepared [:state :acquired-storage])))))
        (is (= 1 @calls))
        (is (empty? (get-in state [:realized :storage])))))))

(deftest allocation-only-actions-retain-helper-definition-before-size-evaluation
  ;; This is a constructor-ownership test, not an OpenCL trapping-arithmetic test. Use a fixed
  ;; compile-only dialect which supports the source's checked math; :ocl:0 otherwise changes
  ;; dialect with the host vendor. Real ASR OpenCL/Level Zero execution has device parity tests.
  (let [compilation (equation-first/compile #'attention/batched-causal-attn-weights
                                          {:target :cuda:allocation-owner-compile-only :dtype :float
                                           :schedule {:precision :f32-scalar}})
        initial (invocation-link/begin-ordered-realization
                 compilation [(float-array 6) (float-array 6) 1 2 3] nil)
        position (first (keep-indexed (fn [i entry] (when (= 'W (:storage entry)) i))
                                     (:source-order initial)))
        entry (nth (:source-order initial) position)
        claim (get-in initial [:source-allocations 'W])
        projection (invocation-source/allocation-source
                    (:source-expression claim) (get-in initial [:program :values]))
        step-index (first (keep-indexed
                           (fn [i step] (when (= (:invocation-step entry) (:id step)) i))
                           (get-in initial [:materialization :plan :steps])))
        calls (atom 0)
        evaluate (fn [& _] (swap! calls inc) (throw (AssertionError. "evaluated unowned size")))]
    (is (some? step-index) "the returned whole buffer now retains its invocation allocation owner")
    (is (some? (get-in claim [:source-projection :helper])))
    (is (fingerprint/equivalent?
         (invocation-source/allocation-definition-projection projection)
         (:source-projection claim)))
    ;; These are ownership preflight checks only, not simulated acquisition of
    ;; earlier storage. They must refuse before asking for any count or resource.
    (doseq [[changed expected]
            [[(update-in initial [:source-allocations 'W] dissoc :source-projection)
              :invocation-ordered-allocation-argument]
             [(assoc-in initial [:source-allocations 'W :source-projection :constructor]
                        '(clojure.core/double-array 1))
              :invocation-ordered-allocation-definition]
             [(assoc-in initial [:source-allocations 'W :source-projection :helper :definition]
                        {:parameters [] :parameter-tags [] :constructor '(clojure.core/float-array 0)})
              :invocation-ordered-allocation-definition]
             [(-> initial
                  (update-in [:source-allocations 'W] dissoc :source-projection)
                  (update-in [:materialization :plan :steps step-index] dissoc :source-projection))
              :invocation-ordered-allocation-definition]]]
      (is (= expected
             (reason #(invocation-link/prepare-source-allocation
                       (assoc changed :source-cursor position) evaluate nil)))))
    (is (zero? @calls))
    (is (empty? (:acquired-storage initial)))))

(deftest delegated-unspecified-allocation-prepares-only-its-proved-zero-realization
  (let [compilation (equation-first/compile #'fixture/helper-source-unspecified-observed
                                          {:target :ocl:0 :dtype :double})
        initial (invocation-link/begin-ordered-realization
                 compilation [(double-array [7 8]) (double-array 1) 2] nil)
        position (first (keep-indexed #(when (= 'out (:storage %2)) %1) (:source-order initial)))
        step-index (first (keep-indexed #(when (= 'out (:symbol %2)) %1)
                                      (get-in initial [:materialization :plan :steps])))
        state (assoc initial :source-cursor position)
        claim (get-in state [:source-allocations 'out])
        calls (atom 0)
        evaluate (fn [expression scalars]
                   (swap! calls inc)
                   (scalar/evaluate-array-size (:source-ns compilation)
                                               (update-vals scalars :value) expression))
        prepared (invocation-link/prepare-source-allocation state evaluate nil)
        token (:storage prepared)]
    (is (= 1 position) "the allocation still follows the earlier numerical write")
    (is (some? step-index))
    (is (= :unspecified (:initialization claim)))
    (is (= :unspecified (get-in state [:materialization :plan :steps step-index :initialization])))
    (is (= {:public-initialization :unspecified :realized-initialization :zero}
           (select-keys (:source-projection claim) [:public-initialization :realized-initialization])))
    (is (= :unspecified (:public-initialization prepared)))
    (is (= :zero (:initialization prepared)) "the existing runtime must execute the actual zero action")
    (is (= 2 (:elements prepared)))
    (is (= claim (get-in prepared [:state :source-allocations 'out])))
    (is (= :unspecified
           (get-in prepared [:state :materialization :plan :steps step-index :initialization])))
    (is (= position (get-in prepared [:state :source-cursor])))
    (is (= {:kind :source-allocation :storage token :source-cursor position}
           (get-in prepared [:state :pending])))
    (is (empty? (get-in prepared [:state :acquired-storage]))
        "preparing a zero request does not acquire or initialize its storage")
    (is (= 1 @calls))
    (doseq [changed [(-> state
                         (update-in [:source-allocations 'out] dissoc :source-projection)
                         (update-in [:materialization :plan :steps step-index] dissoc :source-projection))
                     (assoc-in state [:source-allocations 'out :source-projection
                                      :realized-initialization] :unspecified)
                     (assoc-in state [:materialization :plan :steps step-index :source-projection
                                      :public-initialization] :zero)]]
      (is (= :invocation-ordered-allocation-definition
             (reason #(invocation-link/prepare-source-allocation
                       changed (fn [& _] (swap! calls inc)
                                 (throw (AssertionError. "unowned allocation evaluated a count"))) nil)))))
    (is (= 1 @calls) "a public unspecified label cannot supply a missing zero-constructor proof")
    (is (thrown? IllegalStateException
                 (invocation-link/prepare-source-allocation
                  state (fn [& _] (throw (IllegalStateException. "count failed"))) nil)))
    (is (empty? (:acquired-storage state)))
    (is (empty? (get-in state [:realized :storage])))))

(deftest literal-helper-source-allocation-prepares-without-replaying-the-helper
  (let [compilation (equation-first/compile #'fixture/literal-helper-source-zero-observed
                                          {:target :ocl:0 :dtype :float})
        initial (invocation-link/begin-ordered-realization
                  compilation [(float-array [7.0]) (float-array 1)] nil)
        position (first (keep-indexed (fn [i entry] (when (= 'out (:storage entry)) i))
                                     (:source-order initial)))
        step-index (first (keep-indexed (fn [i step] (when (= 'out (:symbol step)) i))
                                       (get-in initial [:materialization :plan :steps])))
        calls (atom [])
        state (assoc initial :source-cursor position)
        prepared (invocation-link/prepare-source-allocation
                   state
                   (fn [expression scalars]
                     (swap! calls conj expression)
                     (scalar/evaluate-array-size (:source-ns compilation)
                                                 (update-vals scalars :value) expression)) nil)]
    (is (= 1 position))
    (is (= 1 (count @calls)) "only the projected constructor extent is evaluated")
    (is (= 1 (:elements prepared)))
    (is (= :zero (:initialization prepared)))
    (is (= position (get-in prepared [:state :source-cursor])))
    (is (empty? (get-in prepared [:state :acquired-storage])))
    ;; Check both retained producers before evaluating a size or acquiring a
    ;; buffer. This real literal helper is independently executable; placing
    ;; these controls behind a larger unsupported program would not test them.
    (is (some? step-index))
    (doseq [changed [(assoc-in state [:source-allocations 'out :source-projection
                                     :public-initialization] :unspecified)
                     (assoc-in state [:materialization :plan :steps step-index
                                      :source-projection :realized-initialization] :unspecified)]]
      (is (= :invocation-ordered-allocation-definition
             (reason #(invocation-link/prepare-source-allocation
                       changed
                       (fn [& _] (swap! calls conj :unowned-evaluation)
                         (throw (AssertionError. "evaluated changed allocation proof"))) nil)))))
    (is (= 1 (count @calls)) "changed definition facets refuse before evaluating a size")))
