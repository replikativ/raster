(ns raster.compiler.passes.parallel.typed-soac-resident-test
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.kernel-call :as kernel-call]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.numeric :as numeric]))

(defn- fused [source]
  (first (fusion/fusion-fixpoint
          (frontend/form->program source
                                  {:dtype :double :array-types {'x :double 'out :double}
                                   :scalar-types {'scale :double}}))))

(defn- public-reused-reduction-program []
  (first
   (fusion/fusion-fixpoint
    (frontend/form->program
     '(let* [left (raster.par/reduce sum (float 0.0) i n
                     (let* [v (clojure.core/aget x i)] (+ sum (* v v))))
             right (raster.par/reduce sum (float 0.0) i n
                      (let* [v (clojure.core/aget y i)] (+ sum (* v v))))
             ^double norm (raster.numeric/sqrt (+ (double left) (double right)))
             effect (raster.par/map-void! i n
                      (clojure.core/aset out i (float norm)))]
        (double norm))
     {:dtype :float :array-types {'x :float 'y :float 'out :float}
      :scalar-types {'n :long}}))))

(defn- public-root-proof-context [program]
  (let [facts (dialect/facts program)
        infos (mapv #'resident/operation-info (dialect/equations program))
        info-by-id (into {} (map (juxt :id identity)) infos)
        position-by-id (into {} (map-indexed (fn [i info] [(:id info) i])) infos)
        producers (group-by first
                            (mapcat (fn [info]
                                      (map (fn [result] [result info]) (:results info)))
                                    infos))
        output 'norm
        producer (second (first (get producers output)))
        sites (get (group-by first (#'resident/use-sites (dialect/equations program))) output)]
    {:facts facts :info-by-id info-by-id :position-by-id position-by-id
     :producers producers :outputs (set (dialect/outputs program)) :output output
     :producer producer :sites sites}))

(defn- public-root-proof?
  [{:keys [facts info-by-id position-by-id producers outputs output producer sites]
    :as context}]
  (#'resident/public-reused-reduction-root?
   facts info-by-id position-by-id producers outputs
   (set (get-in facts [:attributes :host-read-values])) producer output sites))

(deftest direct-public-reduction-reused-by-a-device-capture-becomes-resident
  (let [program (public-reused-reduction-program)
        root (:output (public-root-proof-context program))
        outputs-before (dialect/outputs program)
        [result stats] (resident/realize program)
        consumer-equation (some #(when (some #{root} (:captures (fusion/equation-info %))) %)
                                (dialect/equations result))
        consumer (fusion/equation-info consumer-equation)
        capture-index (.indexOf ^java.util.List (:captures consumer) root)
        capture-parameter (nth (:capture-parameters
                                (dialect/parameter-layout consumer-equation))
                               capture-index)]
    (is (= 2 (:resident-reductions stats))
        "the internal left norm and the directly returned/reused final norm are resident")
    (is (= [root] (get-in (dialect/facts result)
                          [:attributes :public-resident-reduction-outputs])))
    (is (= outputs-before (dialect/outputs result))
        "the complete logical public output boundary is unchanged")
    (is (resident/resident-scalar-value? (get-in (dialect/facts result) [:values root])))
    (is (some #{(list 'clojure.core/aget capture-parameter 0)}
              (tree-seq coll? seq (:body-results consumer))))
    (is (= result (dialect/validate! result)))))

(deftest public-output-only-and-mixed-use-reduction-roots-stay-scalar
  (let [context (public-root-proof-context (public-reused-reduction-program))
        producer-position (get (:position-by-id context) (get-in context [:producer :id]))
        later-id (some (fn [[id position]] (when (< producer-position position) id))
                       (:position-by-id context))]
    (is (public-root-proof? context))
    (is (not (public-root-proof? (assoc context :sites []))) "output-only is unchanged")
    (is (not (public-root-proof? (assoc context :outputs #{'public-alias})))
        "an indirect public alias does not authorize residency of its producer root")
    (doseq [role [:array :extent :contract-capture :multi-region-capture
                  :result-transform-operand :result-transform-scalar]]
      (is (not (public-root-proof?
                (update context :sites conj [(:output context)
                                             {:equation later-id :role role}])))
          (name role)))))

(deftest an-indirect-public-alias-does-not-activate-public-root-residency
  (let [program (fused
                 '(let* [total (raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
                         result (raster.par/map-void! j n
                                  (aset out j (* (aget x j) total)))
                         ^double returned (double total)]
                    returned))
        [result stats] (resident/realize program)]
    (is (= '[returned] (dialect/outputs program))
        "the post-effect scalar, not its reduction dependency, is the direct public output")
    (is (zero? (:resident-reductions stats)))
    (is (= :plain (get-in (dialect/facts result) [:values 'total :representation :kind])))
    (is (nil? (get-in (dialect/facts result)
                      [:attributes :public-resident-reduction-outputs])))
    (is (= result (dialect/validate! result)))))

(deftest public-residency-requires-pristine-producer-and-value-facets
  (let [context (public-root-proof-context (public-reused-reduction-program))
        root (:output context)
        equation-id (get-in context [:producer :id])]
    (doseq [[label update-facts]
            [[:host-read #(assoc-in % [:attributes :host-read-values] [root])]
             [:effect #(assoc-in % [:equations equation-id :effects] #{:memory/read})]
             [:alias #(assoc-in % [:equations equation-id :aliases] {root 'other})]
             [:storage #(assoc-in % [:equations equation-id :attributes :result-storage]
                                  [{:destination 'scratch :access :write :host-return :buffer}])]
             [:representation #(assoc-in % [:values root :representation] {:kind :external})]
             [:logical-layout #(assoc-in % [:values root :logical-layout] :row-major)]
             [:memory-space #(assoc-in % [:values root :memory-space] :host)]
             [:placement #(assoc-in % [:values root :placement] :external)]
             [:sharding #(assoc-in % [:values root :sharding] {:axis 0})]
             [:ownership #(assoc-in % [:values root :ownership] :owned)]
             [:value-effects #(assoc-in % [:values root :effects] #{:memory/read})]
             [:attributes #(assoc-in % [:values root :attributes] {:external true})]
             [:unknown-dtype #(assoc-in % [:values root :dtype] :unknown)]]]
      (is (not (public-root-proof? (update context :facts update-facts))) (name label)))))

(deftest resident-realization-preserves-unrelated-fold-map-regions
  (let [program (frontend/form->program
                 '(let* [total (raster.par/reduce acc 0.0 i 4 (+ acc (aget x i)))
                         folded (raster.par/segmented-fold-map!
                                 [out] [[row 1]] col 4
                                 [[sum 0.0 :double 4 (+ sum (aget x col))]]
                                 [(+ sum (aget x col))])
                         result (raster.par/map-void! j 4 (aset out j (* (aget x j) total)))]
                    result)
                 {:dtype :double :array-types {'x :double 'out :double}})
        fold-equation (fn [p]
                        (first (filter #(= :segmented-fold-map
                                           (:kind (fusion/equation-info %)))
                                       (dialect/equations p))))
        [realized stats] (resident/realize program)]
    (is (= 1 (:resident-reductions stats)))
    (is (some? (fold-equation program)))
    (is (= (fold-equation program) (fold-equation realized))
        "unrelated fold regions and their map lambda survive without reconstruction")
    (is (= realized (dialect/validate! realized)))))

(deftest resident-realization-never-replaces-a-multi-region-extent-with-a-buffer
  (doseq [role [:segment :fold]]
    (let [segment-extent (if (= :segment role) 'total 1)
          fold-extent (if (= :fold role) 'total 4)
          source `(let* [~'total (raster.par/reduce ~'acc (long 0) ~'i 4
                                  (clojure.core/unchecked-add ~'acc (clojure.core/aget ~'x ~'i)))
                         ~'result (raster.par/segmented-fold-map!
                                   [~'out] [[~'row ~segment-extent]] ~'col 4
                                   [[~'sum 0 :long ~fold-extent
                                     (clojure.core/unchecked-add ~'sum
                                                                (clojure.core/aget ~'x ~'col))]]
                                   [~'sum])]
                    ~'result)
          program (frontend/form->program source {:dtype :long
                                                  :array-types {'x :long 'out :long}})
          [realized stats] (resident/realize program)]
      (is (= 0 (:resident-reductions stats)))
      (is (= program realized) "every segment/fold extent keeps its scalar representation"))))

(deftest resident-uniform-load-is-a-stable-device-capture
  (let [source '(let* [^double alpha (clojure.core/aget params 0)
                       result (raster.par/map-void! i n
                                (clojure.core/aset out i (+ alpha (clojure.core/aget out i))))]
                  result)
        program (frontend/form->program source
                                        {:dtype :double
                                         :array-types {'params :double 'out :double}
                                         :scalar-types {'n :long}})
        [result stats] (resident/inline-uniform-input-loads program)
        infos (mapv fusion/equation-info (dialect/equations result))
        effect (first (filter #(contains? #{:map :effect-map} (:kind %)) infos))]
    (is (= 1 (:resident-uniform-input-loads stats)))
    (is (not-any? #(some #{'alpha} (:results %)) infos))
    (is (some #{'params} (:captures effect)))
    (is (some #{'params} (get-in effect [:attributes :attributes :stable-array-captures])))
    (is (not (some #{'alpha} (:captures effect))))
    (is (= result (dialect/validate! result)))))

(deftest resident-uniform-load-never-moves-a-launch-extent
  (let [source '(let* [^long n (clojure.core/aget dims 0)
                       result (raster.par/map-void! i n
                                (clojure.core/aset out i (double i)))]
                  result)
        program (frontend/form->program source
                                        {:dtype :double
                                         :array-types {'dims :long 'out :double}})
        [result stats] (resident/inline-uniform-input-loads program)]
    (is (zero? (:resident-uniform-input-loads stats)))
    (is (some (fn [equation] (some #{'n} (nth equation 2))) (dialect/equations result)))
    (is (= result (dialect/validate! result)))))

(deftest resident-uniform-load-never-crosses-a-write-to-its-source
  (let [source '(let* [^double alpha (clojure.core/aget params 0)
                       result (raster.par/map-void! i n
                                (clojure.core/aset params i (+ alpha (double i))))]
                  result)
        program (frontend/form->program source
                                        {:dtype :double
                                         :array-types {'params :double}
                                         :scalar-types {'n :long}})
        [result stats] (resident/inline-uniform-input-loads program)]
    (is (zero? (:resident-uniform-input-loads stats)))
    (is (some (fn [equation] (some #{'alpha} (nth equation 2))) (dialect/equations result)))
    (is (= result (dialect/validate! result)))))

(defn- uniform-clock-program
  ([] (uniform-clock-program {}))
  ([{:keys [n] :or {n 'n}}]
   (first
    (fusion/fusion-fixpoint
     (frontend/form->program
      `(let* [~'unrelated (clojure.core/long 7)
              ~'clock-value (clojure.core/long (clojure.core/aget ~'time-buffer 0))
              ~'step (clojure.core/long
                      (clojure.core/unchecked-add (clojure.core/long ~'clock-value)
                                                  (clojure.core/long 3)))
              ~'scaled (clojure.core/long
                        (clojure.core/unchecked-multiply (clojure.core/long ~'step)
                                                        (clojure.core/long 2)))
              ~'effect-a (raster.par/map-void! ~'i ~n
                          (clojure.core/aset ~'out-a ~'i
                            (clojure.core/unchecked-add
                             (clojure.core/aget ~'out-a ~'i) ~'step)))
              ~'effect-b (raster.par/map-void! ~'i ~n
                          (clojure.core/aset ~'out-b ~'i
                            (clojure.core/unchecked-add
                             (clojure.core/aget ~'out-b ~'i) ~'scaled)))]
         [~'effect-a ~'effect-b ~'unrelated])
      {:dtype :long
       :array-types {'time-buffer :long 'out-a :long 'out-b :long}
       :scalar-types (if (symbol? n) {n :long} {})
       :values {'time-buffer (av/tensor {:dtype :long
                                        :shape [(list 'extent 'time-buffer)]})}})))))

(defn- canonical-double-pow-call [base exponent]
  (with-meta (list '.invk 'raster.numeric/pow_m_double_double-impl base exponent)
    {:raster.op/original 'raster.numeric/pow
     :raster.op/resolved-callee 'raster.numeric/pow
     :raster.type/tag 'double :tag 'double}))

(defn- ordered-island-program
  ([] (ordered-island-program {}))
  ([{:keys [n] :or {n 'n}}]
   (first
    (fusion/fusion-fixpoint
     (frontend/form->program
      `(let* [~'clock-value (clojure.core/long (clojure.core/aget ~'time-buffer 0))
              ~'gap (raster.par/map-void! ~'i ~n
                      (clojure.core/aset ~'scratch ~'i
                        (clojure.core/float
                         (clojure.core/+ (clojure.core/aget ~'scratch ~'i) 1.0))))
              ~(with-meta 'beta-checkpoint {:tag 'double}) (clojure.core/double ~'beta)
              ~(with-meta 'powered {:tag 'double})
              (clojure.core/double
               ~(canonical-double-pow-call
                 '(clojure.core/double beta-checkpoint)
                 '(clojure.core/double clock-value)))
              ~(with-meta 'checkpoint {:tag 'float}) (clojure.core/float ~'powered)
              ~'effect-a (raster.par/map-void! ~'i ~n
                          (clojure.core/aset ~'out-a ~'i
                            (clojure.core/float
                             (clojure.core/+ (clojure.core/aget ~'out-a ~'i) ~'powered))))
              ~'effect-b (raster.par/map-void! ~'i ~n
                          (clojure.core/aset ~'out-b ~'i
                            (clojure.core/float
                             (clojure.core/- (clojure.core/aget ~'out-b ~'i)
                                             ~'checkpoint))))]
         [~'gap ~'effect-a ~'effect-b])
      {:dtype :float
       :array-types {'time-buffer :long 'scratch :float 'out-a :float 'out-b :float}
       :scalar-types (cond-> {'beta :double} (symbol? n) (assoc n :long))
       :values {'time-buffer (av/tensor {:dtype :long
                                        :shape [(list 'extent 'time-buffer)]})}})))))

(defn- generated-stage [program pass]
  (some (fn [equation]
          (when (= pass (get-in (dialect/facts program)
                                [:equations (second equation) :provenance :pass]))
            equation))
        (dialect/equations program)))

(defn- singleton-map-equation [program]
  (some (fn [equation]
          (let [{:keys [kind attributes]} (fusion/equation-info equation)]
            (when (and (= :map kind) (= 1 (:extent attributes))) equation)))
        (dialect/equations program)))

(deftest uniform-root-peels-only-a-proved-core-identity-cast
  (let [program (uniform-clock-program)
        facts (dialect/facts program)
        info (some (fn [equation]
                     (let [info (fusion/equation-info equation)]
                       (when (some #{'clock-value} (:results info)) info)))
                   (dialect/equations program))
        input-values (set (:inputs facts))
        root (fn [candidate]
               (#'resident/exact-uniform-root facts candidate input-values
                                                #{'out-a 'out-b} [] #{} #{}))
        expression (first (:body-results info))
        read (first (descriptor/call-args expression))]
    (is (some? (root info)) "the retained `(long (aget input 0))` fixture is admitted")
    (is (nil? (root (assoc info :body-results [(list 'clojure.core/double read)])))
        "a widening cast is not an identity read")
    (is (nil? (root (assoc info :body-results [(list 'long read)])))
        "cast spelling without canonical retained identity is insufficient")
    (is (nil? (root (assoc info :body-results
                           [(with-meta expression {:raster.op/lexical-callee true})]))))
    (is (nil? (root (assoc info :body-results
                           [(apply list (with-meta (first expression)
                                          {:raster.op/lexical-callee true})
                                   (rest expression))]))))))

(deftest uniform-root-write-audit-follows-the-complete-alias-component
  (let [program (uniform-clock-program)
        facts (dialect/facts program)
        info (some (fn [equation]
                     (let [info (fusion/equation-info equation)]
                       (when (some #{'clock-value} (:results info)) info)))
                   (dialect/equations program))]
    (is (nil? (#'resident/exact-uniform-root
               facts info (set (:inputs facts)) #{'out-a}
               [['time-buffer 'intermediate] ['intermediate 'out-a]] #{} #{})))
    (is (some? (#'resident/exact-uniform-root
                facts info (set (:inputs facts)) #{'out-a}
                [['time-buffer 'intermediate] ['unrelated 'out-a]] #{} #{})))))

(deftest multi-use-uniform-scalar-closure-is-evaluated-once-in-a-singleton-map
  (doseq [n ['n 0]]
    (let [program (uniform-clock-program {:n n})
          [result stats] (resident/inline-uniform-input-loads program)
          equation (singleton-map-equation result)
          info (fusion/equation-info equation)
          facts (dialect/facts result)
          frontier (vec (:results info))
          equation-position (into {} (map-indexed (fn [index equation]
                                                    [(second equation) index]))
                                  (dialect/equations result))
          terminal-infos (filter #(contains? #{:map :effect-map} (:kind %))
                                 (map fusion/equation-info (dialect/equations result)))]
      (is (= 1 (:resident-uniform-input-closures stats)))
      (is (= 3 (:resident-uniform-input-scalars stats)))
      (is (zero? (:resident-uniform-input-loads stats)))
      (is (= ['time-buffer] (:captures info)))
      (is (= ['time-buffer]
             (get-in info [:attributes :attributes :stable-array-captures])))
      (is (= 3 (count (:locals info))))
      (is (some #{'clojure.core/unchecked-add}
                (tree-seq coll? seq (mapv :init (:locals info)))))
      (is (some #{'clojure.core/unchecked-multiply}
                (tree-seq coll? seq (mapv :init (:locals info)))))
      (is (= (set frontier)
             (set (get-in facts [:attributes :resident-uniform-input-scalar-closures
                                 0 :frontier]))))
      (doseq [value frontier]
        (is (= [1] (get-in facts [:values value :shape])))
        (is (= :long (get-in facts [:values value :dtype]))))
      (doseq [old '[clock-value step scaled]]
        (is (nil? (get-in facts [:values old]))))
      (is (= :long (get-in facts [:values 'unrelated :dtype]))
          "an unrelated atomic scalar is retained and cannot suppress the valid candidate")
      (is (every? (fn [terminal]
                    (or (= (:id info) (:id terminal))
                        (some (set frontier) (:captures terminal))))
                  terminal-infos))
      (is (every? #(< (equation-position (:id info)) (equation-position (:id %)))
                  (remove #(= (:id info) (:id %)) terminal-infos))
          "the singleton producer remains before every consumer when terminal extents are zero")
      (is (= result (dialect/validate! result))))))

(deftest ordered-uniform-scalar-islands-preserve-gaps-and-stage-each-island-once
  (doseq [n ['n 0]]
    (let [program (ordered-island-program {:n n})
          original-facts (dialect/facts program)
          original-position (into {} (map-indexed (fn [i equation] [(second equation) i]))
                                  (dialect/equations program))
          original-info (mapv fusion/equation-info (dialect/equations program))
          original-root (some #(when (some #{'clock-value} (:results %)) %) original-info)
          original-gap (some #(when (= 'gap (get-in original-facts
                                                     [:equations (:id %) :attributes
                                                      :host-binding])) %)
                             original-info)
          original-beta (some #(when (some #{'beta-checkpoint} (:results %)) %) original-info)
          original-island (filterv #(some #{'powered 'checkpoint} (:results %)) original-info)
          [result stats] (resident/inline-uniform-input-loads program)
          facts (dialect/facts result)
          root-equation (generated-stage result :ordered-uniform-scalar-root)
          island-equation (generated-stage result :ordered-uniform-scalar-island)
          root-info (fusion/equation-info root-equation)
          island-info (fusion/equation-info island-equation)
          root-buffer (first (:results root-info))
          root-capture-index (.indexOf ^java.util.List (:captures island-info) root-buffer)
          root-parameter (nth (:parameters island-info) root-capture-index)
          frontier (vec (:results island-info))
          positions (into {} (map-indexed (fn [i equation] [(second equation) i]))
                          (dialect/equations result))
          certificate (get-in facts [:attributes :resident-uniform-input-scalar-closures 0])
          root-constituents (get-in facts [:equations (:id root-info)
                                           :attributes :fusion/constituents])
          island-constituents (get-in facts [:equations (:id island-info)
                                             :attributes :fusion/constituents])]
      (is (= 1 (:resident-uniform-input-closures stats)))
      (is (= 1 (:resident-uniform-input-scalar-islands stats)))
      (is (= 3 (:resident-uniform-input-scalars stats)))
      (is (= ['time-buffer] (:captures root-info)))
      (is (= 1 (count (:locals root-info))))
      (is (= 2 (count (:locals island-info))))
      (is (some #{'beta-checkpoint} (:captures island-info))
          "the late map captures, but does not duplicate, the certified host scalar prefix")
      (is (some #{(list 'clojure.core/aget root-parameter 0)}
                (tree-seq coll? seq (mapv :init (:locals island-info))))
          "the late stage explicitly reads the earlier private singleton buffer")
      (is (= #{(:id original-root)} (set (keys root-constituents))))
      (is (= (set (map :id original-island)) (set (keys island-constituents))))
      (is (< (positions (:id root-info)) (positions (:id original-gap))
             (positions (:id island-info)))
          "the unrelated effect gap remains between its two original scalar islands")
      (is (= (original-position (:id original-gap)) (positions (:id original-gap)))
          "the first stage replaces one equation and does not move the gap")
      (is (= (original-position (:id original-beta)) (positions (:id original-beta))))
      (is (= :double (get-in facts [:values 'beta-checkpoint :dtype])))
      (is (= [root-buffer] (:root-frontier certificate)))
      (is (= (set frontier) (set (:frontier certificate))))
      (is (= [:root :island] (mapv :kind (:stages certificate))))
      (doseq [value (conj frontier root-buffer)]
        (is (= [1] (get-in facts [:values value :shape]))))
      (doseq [old '[clock-value powered checkpoint]]
        (is (nil? (get-in facts [:values old]))))
      (is (some #(= 'gap (get-in facts [:equations (second %) :attributes :host-binding]))
                (dialect/equations result)))
      (is (= result (dialect/validate! result))))))

(deftest ordered-uniform-island-lowering-keeps-both-singleton-buffer-contracts
  (let [[program _] (resident/inline-uniform-input-loads (ordered-island-program {:n 0}))
        equations [(generated-stage program :ordered-uniform-scalar-root)
                   (generated-stage program :ordered-uniform-scalar-island)]
        facts (dialect/facts program)]
    (doseq [equation equations]
      (let [info (fusion/equation-info equation)
            equation-id (:id info)
            ids (concat (:captures info) (:results info))
            values (select-keys (:values facts) ids)
            stage (dialect/make
                   (-> facts
                       (assoc :values values :inputs (vec (:captures info))
                              :equations {equation-id (get-in facts [:equations equation-id])}
                              :effects #{})
                       (assoc :attributes {}))
                   [equation] (:results info))
            operation (first (soac-lower/lower-typed-map stage :ocl:0 :dtype :float))
            array-types (into {} (keep (fn [[id value]]
                                         (when (seq (:shape value)) [id (:dtype value)]))) values)
            scalar-types (into {} (keep (fn [[id value]]
                                          (when (empty? (:shape value)) [id (:dtype value)]))) values)
            artifact (segop-opencl/generate-scheduled-segmap-kernel
                      operation :array-types array-types :scalar-types scalar-types)
            kernel (get-in artifact [:attributes :kernel-body])
            parameters (into {} (map (juxt :id identity)) (:parameters kernel))
            stable (set (get-in info [:attributes :attributes :stable-array-captures]))]
        (is (= 1 (get-in info [:attributes :extent])))
        (is (seq stable))
        (doseq [capture stable]
          (is (= [1] (:shape (parameters capture)))
              "each staged buffer read carries the exact one-element ABI footprint"))))))

(deftest singleton-map-lowering-derives-the-existing-one-element-capacity-contract
  (let [program (uniform-clock-program {:n 0})
        [result _] (resident/inline-uniform-input-loads program)
        equation (singleton-map-equation result)
        equation-id (second equation)
        results (vec (nth equation 2))
        captures (:captures (fusion/equation-info equation))
        facts (dialect/facts result)
        values (select-keys (:values facts) (concat captures results))
        map-facts (-> facts
                      (assoc :values values :inputs (vec captures)
                             :equations {equation-id (get-in facts [:equations equation-id])}
                             :effects #{})
                      (assoc :attributes {}))
        map-program (dialect/make map-facts [equation] results)
        operation (first (soac-lower/lower-typed-map map-program :ocl:0 :dtype :long))
        array-types (into {} (map (fn [[id value]] [id (:dtype value)])) values)
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                  operation :array-types array-types :scalar-types {})
        kernel (get-in artifact [:attributes :kernel-body])
        parameters (into {} (map (juxt :id identity)) (:parameters kernel))
        operations (tree-seq #(or (map? %) (sequential? %))
                             #(if (map? %) (vals %) (seq %)) (:operations kernel))
        arguments (mapv (fn [slot argument]
                          (cond
                            (= 'time-buffer argument) (long-array 0)
                            (= :scalar (:kind slot)) {:type (:kernel-dtype slot) :value 1}
                            :else (long-array 1)))
                        (:abi artifact) (:arguments artifact))]
    (is (= [1] (:shape (parameters 'time-buffer)))
        "the emitted n=0 producer's read footprint owns the exact preflight capacity")
    (is (some #(and (= :wrap (get-in % [:expression :options :overflow]))
                    (contains? #{:+ :*} (get-in % [:expression :op])))
              operations)
        "unchecked Long arithmetic retains explicit wrap semantics in KernelBody")
    (try
      (kernel-call/validate-retained-input-capacities!
       artifact arguments
       (fn [value _]
         (when value (try (java.lang.reflect.Array/getLength value)
                          (catch IllegalArgumentException _ nil)))))
      (is false "a zero-capacity public input must fail before the singleton kernel launches")
      (catch clojure.lang.ExceptionInfo failure
        (is (= :kernel-body-buffer-capacity (:reason (ex-data failure))))
        (is (= 1 (:required-elements (ex-data failure))))
        (is (= 0 (:buffer-elements (ex-data failure))))))))

(deftest singleton-map-retains-each-scalar-checkpoint-dtype-and-explicit-conversion
  (let [program
        (first
         (fusion/fusion-fixpoint
          (frontend/form->program
           '(let* [^float checkpoint (clojure.core/float
                                      (clojure.core/aget params 0))
                   ^double widened (clojure.core/double checkpoint)
                   ^float narrowed (clojure.core/float widened)
                   effect-a (raster.par/map-void! i n
                              (clojure.core/aset out-a i
                                (clojure.core/float (+ (clojure.core/aget out-a i)
                                                       widened))))
                   effect-b (raster.par/map-void! i n
                              (clojure.core/aset out-b i
                                (clojure.core/float (- (clojure.core/aget out-b i)
                                                       narrowed))))]
              [effect-a effect-b])
           {:dtype :float :array-types {'params :float 'out-a :float 'out-b :float}
            :scalar-types {'n :long}
            :values {'params (av/tensor {:dtype :float
                                         :shape [(list 'extent 'params)]})}})))
        [result stats] (resident/inline-uniform-input-loads program)
        info (fusion/equation-info (singleton-map-equation result))
        locals (:locals info)
        frontiers (:results info)
        equation (singleton-map-equation result)
        equation-id (second equation)
        facts (dialect/facts result)
        values (select-keys (:values facts) (concat (:captures info) frontiers))
        map-program (dialect/make
                     (-> facts
                         (assoc :values values :inputs (vec (:captures info))
                                :equations {equation-id (get-in facts [:equations equation-id])}
                                :effects #{})
                         (assoc :attributes {}))
                     [equation] frontiers)
        operation (first (soac-lower/lower-typed-map map-program :ocl:0 :dtype :float))
        kernel (:kernel-body
                (segmap-body/lower
                 operation
                 {:array-types (into {} (map (fn [[id value]] [id (:dtype value)])) values)
                  :scalar-types {}}))
        operations (tree-seq #(or (map? %) (sequential? %))
                             #(if (map? %) (vals %) (seq %)) (:operations kernel))
        computes (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %) operations)
        compute-types (mapv #(get-in % [:result :type]) computes)
        cast-result-types (into #{}
                                (keep (fn [operation]
                                        (when (= :cast (get-in operation [:expression :op]))
                                          (get-in operation [:expression :result-type]))))
                                computes)]
    (is (= 1 (:resident-uniform-input-closures stats)))
    (is (= [:float :double :float] (mapv :dtype locals)))
    (is (some #{'clojure.core/double}
              (tree-seq coll? seq (mapv :init locals))))
    (is (some #{'clojure.core/float}
              (tree-seq coll? seq (mapv :init locals))))
    (is (contains? (set compute-types) :double))
    (is (contains? (set compute-types) :float))
    (is (= #{:float :double} cast-result-types)
        "KernelBody retains both explicit widening and narrowing checkpoints after alpha-renaming")
    (is (= #{:double :float}
           (set (map #(get-in facts [:values % :dtype]) frontiers))))
    (is (= result (dialect/validate! result)))))

(deftest uniform-scalar-closure-refuses-conflicting-boundaries
  (let [program (uniform-clock-program)
        facts (dialect/facts program)
        root-equation (some #(when (some #{'clock-value} (nth % 2)) %) (dialect/equations program))
        root-id (second root-equation)
        candidates
        [[:host-read (assoc-in facts [:attributes :host-read-values] ['step])]
         [:root-effect (assoc-in facts [:equations root-id :effects] #{:memory/read})]
         [:root-alias (assoc-in facts [:equations root-id :aliases] {'clock-value 'other})]
         [:input-memory (assoc-in facts [:values 'time-buffer :memory-space] :host)]
         [:input-ownership (assoc-in facts [:values 'time-buffer :ownership] :owned)]
         [:scalar-attributes (assoc-in facts [:values 'step :attributes] {:opaque true})]]]
    (doseq [[label candidate-facts] candidates]
      (let [candidate-facts
            (assoc candidate-facts :effects
                   (reduce set/union #{} (map :effects (vals (:equations candidate-facts)))))
            candidate (dialect/make candidate-facts (dialect/equations program)
                                    (dialect/outputs program))
            [result stats] (#'resident/materialize-uniform-scalar-closure candidate)]
        (is (zero? (:resident-uniform-input-closures stats)) (name label))
        (is (= candidate result) (name label))))))

(deftest scalar-closure-requires-canonical-pristine-scalar-facts
  (let [program (uniform-clock-program)
        facts (dialect/facts program)
        info (some (fn [equation]
                     (let [info (fusion/equation-info equation)]
                       (when (some #{'step} (:results info)) info)))
                   (dialect/equations program))]
    (is (#'resident/pristine-scalar-equation? facts info false))
    (let [expression (#'resident/typed-scalar-effect-expression facts info)]
      (is (= #{:checked-source-cast} (:flags (effects/descriptor expression)))
          "the effect authority remains conservative about the outer compound cast")
      (is (#'resident/typed-scalar-expression-total? facts info expression)
          "strict source-typed lowering independently proves that conversion total")
      (is (#'resident/removable-typed-scalar-expression? facts info expression)))
    (let [parameter (first (:parameters info))
          result (first (:results info))
          forged-facts (assoc-in facts [:values result :dtype] :double)
          forged-info (assoc-in info [:attributes :dtypes] [:double])
          narrowing (list 'clojure.core/int parameter)
          narrowing-facts (assoc-in facts [:values result :dtype] :int)
          narrowing-info (-> info
                             (assoc :body-results [narrowing])
                             (assoc-in [:attributes :dtypes] [:int]))
          checked-arithmetic (list 'clojure.core/+ parameter (list 'clojure.core/long 1))
          integral-division (list 'clojure.core/quot parameter parameter)]
      (is (not (#'resident/typed-scalar-expression-total?
                forged-facts forged-info (first (:body-results info))))
          "a forged result dtype cannot coerce a differently typed compound body")
      (is (not (#'resident/typed-scalar-expression-total?
                narrowing-facts narrowing-info narrowing))
          "unknown-range checked Long-to-Int narrowing remains trapping")
      (is (not (#'resident/typed-scalar-expression-total?
                facts (assoc info :body-results [checked-arithmetic]) checked-arithmetic))
          "checked integral arithmetic anywhere in the body remains trapping")
      (is (not (#'resident/removable-typed-scalar-expression?
                facts (assoc info :body-results [checked-arithmetic]) checked-arithmetic))
          "empty effect flags cannot bypass the strict trapping-operation audit")
      (is (not (#'resident/pristine-scalar-equation?
                facts (assoc info :body-results [integral-division]) false))
          "typed integral division with an unproved nonzero divisor is not total"))
    (is (not (#'resident/pristine-scalar-equation?
              facts (update info :attributes assoc :opaque true) false)))
    (is (not (#'resident/pristine-scalar-equation?
              facts (assoc-in info [:attributes :attributes] {:opaque true}) false)))
    (is (not (#'resident/pristine-scalar-equation?
              (assoc-in facts [:equations (:id info) :attributes :opaque] true)
              info false)))
    (is (not (#'resident/pristine-scalar-equation?
              (assoc-in facts [:equations (:id info) :attributes :result-storage]
                        [{:destination 'scratch :access :write :host-return :buffer}])
              info false))
        "the local guard refuses a forged scalar storage certificate before reconstruction")
    (is (not (#'resident/pristine-scalar-equation?
              facts (assoc-in info [:attributes :dtypes] [:f64]) false)))))

(deftest uniform-scalar-closure-does-not-import-another-equation-produced-scalar
  (let [program
        (first
         (fusion/fusion-fixpoint
          (frontend/form->program
           '(let* [prior (clojure.core/long
                          (clojure.core/unchecked-add (clojure.core/long bias) 1))
                   clock-value (clojure.core/long (clojure.core/aget time-buffer 0))
                   step (clojure.core/long
                         (clojure.core/unchecked-add (clojure.core/long clock-value) prior))
                   effect-a (raster.par/map-void! i n
                              (clojure.core/aset out-a i step))
                   effect-b (raster.par/map-void! i n
                              (clojure.core/aset out-b i step))]
              [effect-a effect-b])
           {:dtype :long
            :array-types {'time-buffer :long 'out-a :long 'out-b :long}
            :scalar-types {'n :long 'bias :long}
            :values {'time-buffer (av/tensor
                                   {:dtype :long :shape '[(extent time-buffer)]})}})))
        [result stats] (resident/inline-uniform-input-loads program)]
    (is (zero? (:resident-uniform-input-closures stats)))
    (is (some (fn [equation] (some #{'clock-value} (nth equation 2)))
              (dialect/equations result)))
    (is (= result (dialect/validate! result)))))

(deftest uniform-scalar-closure-refuses-to-coordinate-two-independent-roots
  (let [program
        (first
         (fusion/fusion-fixpoint
          (frontend/form->program
           '(let* [left (clojure.core/long (clojure.core/aget first-clock 0))
                   right (clojure.core/long (clojure.core/aget second-clock 0))
                   effect-a (raster.par/map-void! i n
                              (clojure.core/aset out-a i
                                (clojure.core/unchecked-add left right)))
                   effect-b (raster.par/map-void! i n
                              (clojure.core/aset out-b i
                                (clojure.core/unchecked-add left right)))]
              [effect-a effect-b])
           {:dtype :long
            :array-types {'first-clock :long 'second-clock :long
                          'out-a :long 'out-b :long}
            :scalar-types {'n :long}
            :values {'first-clock (av/tensor
                                   {:dtype :long :shape '[(extent first-clock)]})
                     'second-clock (av/tensor
                                    {:dtype :long :shape '[(extent second-clock)]})}})))
        [result stats] (resident/inline-uniform-input-loads program)]
    (is (zero? (:resident-uniform-input-closures stats)))
    (is (every? #(some (set (nth % 2)) '[left right])
                (take 2 (dialect/equations result))))))

(deftest ordered-island-refuses-an-asymmetric-second-exact-root
  (let [program
        (first
         (fusion/fusion-fixpoint
          (frontend/form->program
           '(let* [left (clojure.core/long (clojure.core/aget first-clock 0))
                   limit (clojure.core/long (clojure.core/aget second-clock 0))
                   gap (raster.par/map-void! i limit
                         (clojure.core/aset scratch i
                           (clojure.core/float
                            (clojure.core/+ (clojure.core/aget scratch i) 1.0))))
                   step (clojure.core/double
                         (clojure.core/+ (clojure.core/double left)
                                         (clojure.core/double bias)))
                   effect-a (raster.par/map-void! i n
                              (clojure.core/aset out-a i
                                (clojure.core/float
                                 (clojure.core/+ (clojure.core/aget out-a i) step))))
                   effect-b (raster.par/map-void! i n
                              (clojure.core/aset out-b i
                                (clojure.core/float
                                 (clojure.core/+ (clojure.core/aget out-b i) step))))]
              [gap effect-a effect-b])
           {:dtype :float
            :array-types {'first-clock :long 'second-clock :long
                          'scratch :float 'out-a :float 'out-b :float}
            :scalar-types {'n :long 'bias :double}
            :values {'first-clock (av/tensor
                                   {:dtype :long :shape '[(extent first-clock)]})
                     'second-clock (av/tensor
                                    {:dtype :long :shape '[(extent second-clock)]})}})))
        _fixture (is (some? program) "asymmetric-root fixture must enter TypedSOAC")
        [result stats] (when program (resident/inline-uniform-input-loads program))]
    (is (zero? (:resident-uniform-input-closures stats)))
    (is (= program result)
        "an unsupported second exact root participates in multiplicity and blocks the transaction")))

(deftest ordered-island-refuses-public-host-and-device-produced-scalar-dependencies
  (let [program (ordered-island-program)
        facts (dialect/facts program)
        equations (dialect/equations program)
        candidates
        [[:public (dialect/make facts equations ['powered])]
         [:host-read
          (dialect/make (assoc-in facts [:attributes :host-read-values] ['powered])
                        equations (dialect/outputs program))]]]
    (doseq [[label candidate] candidates]
      (let [[result stats] (#'resident/materialize-uniform-scalar-closure candidate)]
        (is (zero? (:resident-uniform-input-closures stats)) (name label))
        (is (= candidate result) (name label))))
    (let [device-prefix
          (first
           (fusion/fusion-fixpoint
            (frontend/form->program
             '(let* [clock-value (clojure.core/long
                                  (clojure.core/aget time-buffer 0))
                     prior (raster.par/reduce acc (clojure.core/float 0.0) i n
                             (clojure.core/+ (clojure.core/float acc)
                                             (clojure.core/aget x i)))
                     gap (raster.par/map-void! i n
                           (clojure.core/aset scratch i
                             (clojure.core/float
                              (clojure.core/+ (clojure.core/aget scratch i) 1.0))))
                     ^double step (clojure.core/double
                                   (clojure.core/+ (clojure.core/double clock-value)
                                                   (clojure.core/double prior)))
                     effect-a (raster.par/map-void! i n
                                (clojure.core/aset out-a i
                                  (clojure.core/float
                                   (clojure.core/+ (clojure.core/aget out-a i) step))))
                     effect-b (raster.par/map-void! i n
                                (clojure.core/aset out-b i
                                  (clojure.core/float
                                   (clojure.core/+ (clojure.core/aget out-b i) step))))]
                [gap effect-a effect-b])
             {:dtype :float
              :array-types {'time-buffer :long 'x :float 'scratch :float
                            'out-a :float 'out-b :float}
              :scalar-types {'n :long}
              :values {'time-buffer (av/tensor
                                     {:dtype :long :shape '[(extent time-buffer)]})}})))
          _fixture (is (some? device-prefix) "device-prefix fixture must enter TypedSOAC")
          [result stats] (when device-prefix
                           (resident/inline-uniform-input-loads device-prefix))]
      (is (zero? (:resident-uniform-input-closures stats)))
      (is (= device-prefix result)
          "a scalar-looking capture backed by a reduction is not a retained host prefix"))))

(deftest ordered-island-refuses-two-dependent-scalar-islands
  (let [program
        (first
         (fusion/fusion-fixpoint
          (frontend/form->program
           '(let* [clock-value (clojure.core/long (clojure.core/aget time-buffer 0))
                   gap-a (raster.par/map-void! i n
                           (clojure.core/aset scratch i
                             (clojure.core/float
                              (clojure.core/+ (clojure.core/aget scratch i) 1.0))))
                   first-step (clojure.core/double
                               (clojure.core/+ (clojure.core/double clock-value)
                                               (clojure.core/double bias)))
                   effect-a (raster.par/map-void! i n
                              (clojure.core/aset out-a i
                                (clojure.core/float
                                 (clojure.core/+ (clojure.core/aget out-a i) first-step))))
                   gap-b (raster.par/map-void! i n
                           (clojure.core/aset scratch i
                             (clojure.core/float
                              (clojure.core/+ (clojure.core/aget scratch i) 1.0))))
                   second-step (clojure.core/double
                                (clojure.core/* (clojure.core/double clock-value)
                                                (clojure.core/double bias)))
                   effect-b (raster.par/map-void! i n
                              (clojure.core/aset out-b i
                                (clojure.core/float
                                 (clojure.core/+ (clojure.core/aget out-b i) second-step))))]
              [gap-a effect-a gap-b effect-b])
           {:dtype :float
            :array-types {'time-buffer :long 'scratch :float 'out-a :float 'out-b :float}
            :scalar-types {'n :long 'bias :double}
            :values {'time-buffer (av/tensor
                                   {:dtype :long :shape '[(extent time-buffer)]})}})))
        _fixture (is (some? program) "multi-island fixture must enter TypedSOAC")
        [result stats] (when program (resident/inline-uniform-input-loads program))]
    (is (zero? (:resident-uniform-input-closures stats)))
    (is (= program result)
        "dependent scalar blocks separated by device equations are not merged or reordered")))

(deftest singleton-stage-private-ids-share-one-collision-proof
  (let [reserved '#{uniform-index__uniform_scalar_buffer_0
                    clock-value__uniform_scalar_buffer_0}
        [index reserved] (#'resident/allocate-private-id reserved 'uniform-index 0)
        [frontier _] (#'resident/allocate-private-id reserved 'clock-value 0)]
    (is (= 'uniform-index__uniform_scalar_buffer_1 index))
    (is (= 'clock-value__uniform_scalar_buffer_1 frontier))
    (is (not= index frontier))))

(deftest singleton-stage-capture-descriptors-are-complete-and-type-preserving
  (let [program (ordered-island-program)
        facts (dialect/facts program)
        infos (mapv fusion/equation-info (dialect/equations program))
        root-info (some #(when (some #{'clock-value} (:results %)) %) infos)
        powered-info (some #(when (some #{'powered} (:results %)) %) infos)
        root-stage {:infos [root-info] :frontier ['clock-value]
                    :equations #{(:id root-info)} :pass :test}
        island-stage {:infos [powered-info] :frontier ['powered]
                      :equations #{(:id powered-info)} :pass :test}
        ordinary {:logical 'beta-checkpoint :value 'beta-checkpoint :load? false}
        load {:logical 'clock-value :value 'clock-buffer :load? true}
        long-buffer {'clock-buffer (av/tensor {:dtype :long :shape [1]})}
        build-root (fn [descriptors]
                     (#'resident/singleton-scalar-stage
                      facts (assoc root-stage :capture-descriptors descriptors
                                   :stable-captures #{'time-buffer}) #{}))
        build-island
        (fn [facts descriptor stable values]
          (#'resident/singleton-scalar-stage
           facts (assoc island-stage :capture-descriptors [ordinary descriptor]
                        :stable-captures stable :capture-values values) #{}))]
    (is (thrown? clojure.lang.ExceptionInfo (build-root [])))
    (is (thrown? clojure.lang.ExceptionInfo
                 (build-root [{:logical 'time-buffer :value 'time-buffer :load? false}
                              {:logical 'time-buffer :value 'time-buffer :load? false}])))
    (is (thrown? clojure.lang.ExceptionInfo
                 (build-root [{:logical 'time-buffer :value 'other :load? false}])))
    (is (thrown? clojure.lang.ExceptionInfo
                 (build-island facts load #{} long-buffer)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (build-island facts load #{'clock-buffer}
                               {'clock-buffer (av/tensor {:dtype :float :shape [1]})})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (build-island (assoc-in facts [:values 'clock-value :shape] [1])
                               load #{'clock-buffer} long-buffer)))))

(deftest result-transform-integral-casts-state-their-rounding
  (doseq [source [:int :long] target [:float :double]]
    (let [region (body/->ScalarRegion ['acc 'width] (list (symbol (name target)) 'width)
                                      [] target)
          result (scalar-region/lower
                  region {:accumulator 'acc :accumulator-dtype target :store-dtype target
                          :parameters {'width {:id 'width :kind :scalar :role :parameter
                                               :dtype source}}})
          cast (:expression (first (:operations result)))]
      (is (= :cast (:op cast)))
      (is (= {:rounding (if (and (= :int source) (= :double target))
                         :exact :nearest-even)
              :overflow :exact}
             (:options cast))
          "all Int32 values are exactly representable in Double; other pairs require rounding"))))

(deftest resident-reduction-preserves-host-scalar-captures
  (let [program (fused
                 '(let* [^long width (alength x)
                         total (raster.par/reduce acc 0.0 i width (+ acc (aget x i)))
                         result (raster.par/map-void! j width
                                  (aset out j (* (aget x j) total)))]
                    result))
        scalar-info (fn [p]
                      (first (filter #(= :scalar (:kind %))
                                     (map fusion/equation-info (dialect/equations p)))))
        before (scalar-info program)
        [result stats] (resident/realize program)
        after (scalar-info result)]
    (is (some? before))
    (is (= 1 (:resident-reductions stats)))
    (is (= (:captures before) (:captures after)))
    (is (= (:parameters before) (:parameters after)))
    (is (= (:body-results before) (:body-results after)))
    (is (= result (dialect/validate! result)))))

(deftest resident-reduction-rewrites-an-effect-map-without-touching-host-scalar-equations
  (let [program (fused
                 '(let* [^long width (alength x)
                         total (raster.par/reduce acc 0.0 i width (+ acc (aget x i)))
                         result (raster.par/map-void! j width
                                  (if (> total 0.0)
                                    (let* [^double scaled (* total 2.0)]
                                      (aset x j (* (aget x j) scaled)))
                                    nil))]
                    result))
        scalar-before (first (filter #(= :scalar (:kind %))
                                     (map fusion/equation-info
                                          (dialect/equations program))))
        [result stats] (resident/realize program)
        infos (mapv fusion/equation-info (dialect/equations result))
        scalar-after (first (filter #(= :scalar (:kind %)) infos))
        effect (first (filter #(= :effect-map (:kind %)) infos))]
    (is (= 1 (:resident-reductions stats)))
    (is (= scalar-before scalar-after))
    (is (= '[x] (:destinations effect)))
    (is (= (inc (count (:captures effect))) (count (:parameters effect))))
    (is (resident/resident-scalar-value? (get-in (dialect/facts result)
                                                  [:values 'total])))
    (is (= result (dialect/validate! result)))))

(deftest resident-reduction-preserves-the-independent-transform-boundary
  (let [program (fused
                 '(let* [total (raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
                         ^double scaled (* ^double total ^double scale)
                         result (raster.par/map-void! j n
                                  (aset out j (* (aget x j) scaled)))]
                    result))
        transform-before (get-in (fusion/equation-info (first (dialect/equations program)))
                                 [:attributes :result-transform])
        [result stats] (resident/realize program)
        reduction (fusion/equation-info (first (dialect/equations result)))]
    (is (some? transform-before))
    (is (= '[scale] (mapv :value (:scalars transform-before))))
    (is (= 1 (:resident-reductions stats)))
    (is (some #{'scale} (:captures reduction)))
    (is (= transform-before (get-in reduction [:attributes :result-transform])))
    (is (resident/resident-scalar-value? (get-in (dialect/facts result) [:values 'scaled])))
    (is (= result (dialect/validate! result)))))

(defn- two-reduction-transform-program []
  (fused
   '(let* [left (raster.par/reduce a 0.0 i n (+ a (aget x i)))
           right (raster.par/reduce b 0.0 j n (+ b (* (aget x j) 2.0)))
           ^double combined (* ^double right ^double left)
           result (raster.par/map-void! k n
                    (aset out k (* (aget x k) combined)))]
      result)))

(defn- transform-scalar-values [program]
  (into #{} (mapcat #(map :value (get-in (fusion/equation-info %)
                                         [:attributes :result-transform :scalars])))
        (dialect/equations program)))

(deftest audited-transform-scalars-become-explicit-resident-singleton-loads
  (let [program (two-reduction-transform-program)
        transform-inputs (into #{} (mapcat #(map :value (get-in (fusion/equation-info %)
                                                                [:attributes :result-transform :scalars])))
                               (dialect/equations program))
        [result stats] (resident/realize program)
        result-infos (mapv fusion/equation-info (dialect/equations result))
        transform-operands
        (into {} (mapcat #(map (juxt :value identity)
                               (get-in % [:attributes :result-transform :operands])))
              result-infos)
        transform-scalars
        (into #{} (mapcat #(map :value (get-in % [:attributes :result-transform :scalars])))
              result-infos)]
    (is (seq transform-inputs))
    (doseq [value transform-inputs]
      (is (resident/resident-scalar-value? (get-in (dialect/facts result) [:values value])))
      (is (= {:groups [[]]} (:map (get transform-operands value))))
      (is (not (contains? transform-scalars value)))
      (let [transform (some #(when (some #{value}
                                         (map :value (get-in % [:attributes :result-transform
                                                               :operands])))
                               (get-in % [:attributes :result-transform]))
                            result-infos)
            operand (some #(when (= value (:value %)) %)
                          (:operands transform))
            expression (first (:body-results (dialect/lambda-parts (:lambda transform))))]
        (is (some #{(list 'clojure.core/aget (:parameter operand) 0)}
                  (tree-seq coll? seq expression))))
      (is (some (fn [info]
                  (some #{value}
                        (get-in info [:attributes :attributes :stable-array-captures])))
                result-infos)))
    (is (pos? (:resident-reductions stats)))
    (is (= result (dialect/validate! result)))))

(deftest transform-load-realization-is-blocked-by-public-or-host-escape
  (let [program (two-reduction-transform-program)
        value (first (transform-scalar-values program))
        facts (dialect/facts program)
        equations (dialect/equations program)
        outputs (dialect/outputs program)
        public (dialect/make facts equations (vec (distinct (conj outputs value))))
        host (dialect/make (assoc-in facts [:attributes :host-read-values] [value])
                           equations outputs)]
    (doseq [[label candidate] [[:public public] [:host host]]]
      (let [[result _] (resident/realize candidate)
            transform-values (transform-scalar-values result)]
        (is (contains? transform-values value) (name label))
        (is (not (resident/resident-scalar-value?
                  (get-in (dialect/facts result) [:values value])))
            (name label))
        (is (= result (dialect/validate! result)))))))

(deftest transform-load-realization-preserves-incompatible-value-facets
  (let [program (two-reduction-transform-program)
        value (first (transform-scalar-values program))]
    (doseq [[facet facet-value]
            [[:memory-space :host]
             [:ownership :owned]
             [:attributes {:external-contract true}]]]
      (let [facts (assoc-in (dialect/facts program) [:values value facet] facet-value)
            candidate (dialect/make facts (dialect/equations program)
                                    (dialect/outputs program))
            [result _] (resident/realize candidate)]
        (is (contains? (transform-scalar-values result) value) (name facet))
        (is (= facet-value (get-in (dialect/facts result) [:values value facet]))
            (name facet))
        (is (not (resident/resident-scalar-value?
                  (get-in (dialect/facts result) [:values value])))
            (name facet))
        (is (= result (dialect/validate! result)))))))

(deftest transform-load-realization-rejects-a-mixed-extent-use
  (let [program
        (first
         (fusion/fusion-fixpoint
          (frontend/form->program
           '(let* [left (raster.par/reduce a (long 0) i n
                          (clojure.core/unchecked-add a (aget x i)))
                   right (raster.par/reduce b (long 0) j n
                           (clojure.core/unchecked-add b (aget x j)))
                   ^long combined (clojure.core/unchecked-add right left)
                   result (raster.par/map-void! k left
                            (aset out k (clojure.core/unchecked-add (aget x k) combined)))]
              result)
           {:dtype :long :array-types {'x :long 'out :long}
            :scalar-types {'n :long}})))
        value (first (transform-scalar-values program))
        [result _] (resident/realize program)]
    (is (some? value))
    (is (contains? (transform-scalar-values result) value))
    (is (not (resident/resident-scalar-value?
              (get-in (dialect/facts result) [:values value]))))
    (is (= result (dialect/validate! result)))))
