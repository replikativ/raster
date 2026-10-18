(ns raster.compiler.ir.ordered-invocation-realization-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.core :refer [deftm]]
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

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

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
        early (first (:equations program))
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
    (doseq [id (concat (:results early) ['first-output])]
      (is (= :emitted-program-device-result-scalar
             (reason #(program-call/make-equation-call
                       program (:id early) buffers
                       (assoc scalars id {:type :double :value 0.0}) {} nil)))))
    (doseq [equation-id [(:id (second (:equations program))) :absent]]
      (is (= :emitted-program-ready-equation
             (reason #(program-call/make-equation-call program equation-id buffers scalars {} nil)))))
    (is (= :emitted-program-result-buffer
           (reason #(program-call/make-equation-call
                     program (:id (last (:equations program))) buffers scalars {} nil))))))

(deftest ready-storage-is-conditional-not-completed-invocation-storage
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        state (invocation-link/begin-ordered-realization
               compilation [(double-array [7.0]) (double-array 1) Long/MAX_VALUE] nil)
        equation-id (:id (first (:equations (:program state))))
        ready (invocation-link/realize-ready-equation state equation-id nil)]
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

(deftest ready-graph-cannot-drop-a-logical-prefix-result-contract
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        state (invocation-link/begin-ordered-realization
               compilation [(double-array [7.0]) (double-array 4) Long/MAX_VALUE] nil)]
    (is (= :invocation-ordered-result-views
           (reason #(invocation-link/realize-ready-equation
                     state (:id (first (:equations (:program state)))) nil))))))

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
          end (invocation-link/advance-host-source
               start nil
               (fn [equation context]
                 (swap! calls inc)
                 (is (= {'xs [3]} (:buffer-shapes context)))
                 (scalar/evaluate-host-equation *ns* equation context)) nil)]
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
