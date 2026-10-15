(ns raster.ad.normalized-map-chain-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.numeric]
            [raster.par]))

(def ^:private square-chain-source
  '(let* [squared (raster.par/pmap i n float
                                   (raster.numeric/* (clojure.core/aget x i)
                                                     (clojure.core/aget x i)))
          fourth (raster.par/pmap j n float
                                  (raster.numeric/* (clojure.core/aget squared j)
                                                    (clojure.core/aget squared j)))]
         fourth))

(def ^:private rounding-chain-source
  '(let* [squared (raster.par/pmap i n float
                                   (raster.numeric/* (clojure.core/aget x i)
                                                     (clojure.core/aget x i)))
          shifted (raster.par/pmap j n float
                                   (raster.numeric/- (clojure.core/aget squared j)
                                                     (clojure.core/float 16785408.0)))]
         shifted))

(def ^:private three-map-source
  '(let* [squared (raster.par/pmap i n float
                                   (raster.numeric/* (clojure.core/aget x i)
                                                     (clojure.core/aget x i)))
          fourth (raster.par/pmap j n float
                                  (raster.numeric/* (clojure.core/aget squared j)
                                                    (clojure.core/aget squared j)))
          eighth (raster.par/pmap k n float
                                  (raster.numeric/* (clojure.core/aget fourth k)
                                                    (clojure.core/aget fourth k)))]
         eighth))

(def ^:private diamond-source
  '(let* [squared (raster.par/pmap i n float
                                   (raster.numeric/* (clojure.core/aget x i)
                                                     (clojure.core/aget x i)))
          shifted (raster.par/pmap j n float
                                   (raster.numeric/+ (clojure.core/aget squared j)
                                                     (clojure.core/float 1.0)))
          fourth (raster.par/pmap k n float
                                  (raster.numeric/* (clojure.core/aget squared k)
                                                    (clojure.core/aget squared k)))
          joined (raster.par/pmap l n float
                                  (raster.numeric/+ (clojure.core/aget shifted l)
                                                    (clojure.core/aget fourth l)))]
         joined))

(defn- typed-source [source]
  (frontend/form->program source {:dtype :float :array-types {'x :float}
                                  :scalar-types {'n :long}}))

(defn- build [source]
  (normalized/linearize-map-chain-program
   source (the-ns 'raster.ad.normalized-map-chain-test)))

(defn- reason [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- recertify [bundle]
  (assoc bundle :certificate (#'normalized/certificate bundle)))

(defn- evaluate-lane [bundle phase operands]
  (let [program (get-in bundle [:map-programs phase])
        equation (first (soac/equations program))
        {:keys [lambda]} (soac/operation-parts equation)
        dtypes (mapv #(get-in (soac/facts program) [:values % :dtype]) (nth equation 2))]
    (mapv :value
          (scalar/evaluate-region (:source-ns bundle) lambda
                                  (mapv #(hash-map :type :float :value %) operands)
                                  dtypes))))

(defn- lane-run [bundle x dx seed]
  (let [[primal & residuals] (evaluate-lane bundle :primal [(float x)])
        [tangent] (evaluate-lane bundle :linear (into [(float dx)] residuals))
        [gradient] (evaluate-lane bundle :transpose (into [(float seed)] residuals))]
    {:primal primal :residuals residuals :tangent tangent :gradient gradient}))

(defn- fmul [left right] (float (* (float left) (float right))))
(defn- fadd [left right] (float (+ (float left) (float right))))
(defn- fbits [x] (Float/floatToRawIntBits (float x)))

(deftest one-node-lifecycle-allocates-once-and-replays-retained-identities
  (let [source (frontend/form->program
                '(let* [paired (raster.par/pmap i n float
                                                (raster.numeric/*
                                                 (clojure.core/aget x i)
                                                 (clojure.core/aget q i)))
                        output (raster.par/pmap j n float
                                                (raster.numeric/*
                                                 (clojure.core/aget paired j)
                                                 (clojure.core/aget r j)))]
                       output)
                {:dtype :float :array-types {'x :float 'q :float 'r :float}
                 :scalar-types {'n :long}})
        bundle (normalized/linearize-map-chain-program
                source (the-ns 'raster.ad.normalized-map-chain-test)
                {:active-input 'r})
        ordered-bundle (normalized/linearize-map-chain-program
                        source (the-ns 'raster.ad.normalized-map-chain-test)
                        {:active-inputs '[r q x]})
        phases (select-keys (:map-programs bundle) [:primal :linear :transpose])
        reject-allocation (fn [& _]
                            (throw (ex-info "retained replay allocated an identity" {})))
        rebuilt (with-redefs-fn
                  {#'normalized/allocate-map-stage-interface reject-allocation
                   #'normalized/node-equation-ids reject-allocation}
                  #(#'normalized/reconstructed-map-proof bundle))
        node-path [:map-programs :proof :nodes]]
    (is (= [false true]
           (mapv #(contains? % :scalar-derivation) (get-in bundle node-path))))
    (is (every? (fn [phase]
                  (fingerprint/equivalent? (get phases phase) (get rebuilt phase)))
                [:primal :linear :transpose]))
    (let [nodes (get-in ordered-bundle node-path)
          residuals (mapv #(vec (vals (get-in % [:stage-interface :residual-arrays]))) nodes)
          expected-outputs (into [(first (soac/outputs source))] (mapcat identity residuals))]
      (is (< 1 (count (first residuals))))
      (is (= expected-outputs
             (soac/outputs (get-in ordered-bundle [:map-programs :primal])))))
    (is (= :ad-normalized-map-node-derivation
           (reason (fn []
                     (#'normalized/reconstructed-map-proof
                      (assoc-in bundle (conj node-path 1 :scalar-derivation) nil))))))
    (doseq [changed [(assoc-in bundle (conj node-path 0 :equation-ids :linear)
                               'forged-inactive-linear)
                     (update-in bundle (conj node-path 1 :equation-ids) dissoc :linear)
                     (assoc-in bundle (conj node-path 1 :equation-ids :extra)
                               'forged-active-extra)]]
      (is (= :ad-normalized-map-node-interface
             (reason (fn [] (#'normalized/reconstructed-map-proof changed))))))))
(defn- square-push [x dx] (fadd (fmul dx x) (fmul dx x)))

(defn- alter-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

(deftest equation-dataflow-retains-ordered-occurrences-and-separate-controls
  (let [source (frontend/form->program
                '(let* [left (raster.par/pmap i n float
                                               (raster.numeric/* (clojure.core/aget x i)
                                                                 (clojure.core/aget x i)))
                        right (raster.par/pmap j n float
                                                (raster.numeric/+ (clojure.core/aget x j)
                                                                  (clojure.core/float 1.0)))]
                       right)
                {:dtype :float :array-types {'x :float} :scalar-types {'n :long}})
        {:keys [nodes producers consumers controls]}
        (#'normalized/equation-dataflow source)
        equation-ids (mapv (comp second :equation) nodes)]
    (is (= equation-ids (mapv :id nodes)))
    (is (= [0 1] (mapv :consumer-index (get consumers 'x)))
        "repeated data uses across consumers remain ordered occurrences")
    (is (= [0 1] (mapv :consumer-index (get controls 'n)))
        "extent references are ordered separately from executable operands")
    (is (nil? (get consumers 'n)))
    (is (= {:producer-index 1 :producer-id (second equation-ids) :position 0}
           (get producers 'right)))
    (is (every? true?
                (map #(fingerprint/equivalent? (:equation %)
                                               (first (soac/equations (:subprogram %))))
                     nodes)))))

(deftest equation-dataflow-batches-validation-and-defers-malformed-ssa-to-the-dialect
  (let [source (typed-source square-chain-source)
        original soac/validate!
        calls (atom 0)]
    (with-redefs [soac/validate! (fn [candidate]
                                  (when (identical? candidate source) (swap! calls inc))
                                  (original candidate))]
      (#'normalized/equation-dataflow source))
    (is (= 1 @calls)))
  (let [source (typed-source square-chain-source)
        equations (vec (soac/equations source))
        first-id (second (first equations))
        duplicate-id (apply list (assoc (vec (second equations)) 1 first-id))
        malformed (list 'soac-program (soac/facts source)
                        [(first equations) duplicate-id] (soac/outputs source))
        failure (try (#'normalized/equation-dataflow malformed) nil
                     (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= :typed-soac-equation-ids (:reason failure)))))

(deftest equation-dataflow-does-not-weaken-distinct-outer-port-invariants
  (let [source (typed-source square-chain-source)
        equations (vec (soac/equations source))
        equation (first equations)
        operation (vec (nth equation 3))
        ;; Repeated scalar-lambda use (x*x) belongs to scalar AD. Repeating one outer array
        ;; port is malformed TypedSOAC and must be rejected before the dataflow projection.
        duplicate-operation (apply list (assoc operation 2 '[x x]))
        duplicate-equation (apply list (assoc (vec equation) 3 duplicate-operation))
        malformed (list 'soac-program (soac/facts source)
                        (assoc equations 0 duplicate-equation) (soac/outputs source))
        failure (try (#'normalized/equation-dataflow malformed) nil
                     (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= :typed-soac-operands (:reason failure)))))

(deftest exact-two-map-source-owns-one-reconstructed-stage
  (let [source (typed-source square-chain-source)
        bundle (build source)
        programs (:map-programs bundle)
        residual-ids (vals (get-in programs [:interface :residual-arrays]))]
    (is (identical? source (:source programs)))
    (is (= '[map map] (mapv soac/operation-kind (soac/equations (:source programs)))))
    (is (= '[map] (mapv soac/operation-kind (soac/equations (:primal programs)))))
    (is (identical? bundle (normalized/validate! bundle)))
    (is (<= 2 (count residual-ids)) "both nonlinear stages retain their needed coefficients")
    (is (= (count residual-ids) (count (distinct residual-ids))))
    (doseq [id residual-ids]
      (is (= :owned (get-in (soac/facts (:primal programs)) [:values id :ownership])))
      (is (= '[n] (get-in (soac/facts (:primal programs)) [:values id :shape]))))
    (let [owner (canonical/adapt-map bundle)]
      (is (identical? source (:source owner)))
      (is (= '[map] (mapv soac/operation-kind
                           (soac/equations (:execution-source owner)))))
      (is (identical? owner (canonical/validate! owner))))))

(deftest exact-three-map-source-retains-lineage-and-one-execution-stage
  (let [source (typed-source three-map-source)
        bundle (build source)
        programs (:map-programs bundle)
        owner (canonical/adapt-map bundle)
        source-equations (mapv second (soac/equations source))]
    (is (identical? source (:source programs)))
    (is (= '[map map map] (mapv soac/operation-kind (soac/equations (:source programs)))))
    (is (= '[map] (mapv soac/operation-kind (soac/equations (:primal programs)))))
    (is (= '[map] (mapv soac/operation-kind (soac/equations (:execution-source owner)))))
    (is (every? #(= source-equations (get-in % [:origin :source-equations]))
                (vals (:residual-bindings owner))))
    (is (identical? bundle (normalized/validate! bundle)))
    (is (identical? owner (canonical/validate! owner)))))

(deftest closed-diamond-differentiates-one-exact-fused-execution-map
  (let [source (typed-source diamond-source)
        bundle (normalized/linearize-map-program
                source (the-ns 'raster.ad.normalized-map-chain-test))
        stage (normalized/execution-map-source bundle)
        equation (first (soac/equations stage))
        {:keys [lambda]} (soac/operation-parts equation)
        {:keys [parameters locals body-results]} (soac/lambda-parts lambda)
        element (first parameters)
        fused-body (walk/postwalk-replace
                    {element 'element}
                    (soac/scalar-converts->source (first body-results)))
        expected-body
        '(clojure.core/float
          (raster.numeric/+
           (clojure.core/float
            (raster.numeric/+ (clojure.core/float
                               (raster.numeric/* element element))
                              (clojure.core/float 1.0)))
           (clojure.core/float
            (raster.numeric/* (clojure.core/float
                               (raster.numeric/* element element))
                              (clojure.core/float
                               (raster.numeric/* element element))))))
        source-ids (mapv second (soac/equations source))
        stage-id (second equation)
        constituents (get-in (soac/facts stage)
                             [:equations stage-id :attributes :fusion/constituents])]
    (is (identical? source (get-in bundle [:map-programs :source])))
    (is (= 4 (count (soac/equations source))))
    (is (= '[map] (mapv soac/operation-kind (soac/equations stage))))
    (is (empty? locals))
    (is (= expected-body fused-body)
        "the execution model duplicates the shared square in the exact fused operation order")
    (is (= (set source-ids) (set (keys constituents))))
    (is (identical? bundle (normalized/validate! bundle)))
    (doseq [[x dx seed] [[2.0 0.25 -0.5] [-3.0 0.75 1.25]
                          [4097.0 0.25 -0.5]]]
      (let [{:keys [primal tangent gradient]} (lane-run bundle x dx seed)
            x (float x) dx (float dx) seed (float seed)
            square-left (fmul x x)
            shifted (fadd square-left (float 1.0))
            square-right-left (fmul x x)
            square-right-right (fmul x x)
            fourth (fmul square-right-left square-right-right)
            expected-primal (fadd shifted fourth)
            shifted-tangent (square-push x dx)
            right-left-tangent (square-push x dx)
            right-right-tangent (square-push x dx)
            fourth-tangent (fadd (fmul right-left-tangent square-right-right)
                                 (fmul right-right-tangent square-right-left))
            expected-tangent (fadd shifted-tangent fourth-tangent)
            ;; Transposition visits the duplicated right branch first. Its four equal
            ;; contributions precede the two contributions from the shifted branch.
            t (fmul (fmul seed square-right-left) x)
            u (fmul seed x)
            expected-gradient (fadd (fadd (fadd (fadd (fadd t t) t) t) u) u)]
        (is (= (fbits expected-primal) (fbits primal)))
        (is (= (fbits expected-tangent) (fbits tangent)))
        (is (= (fbits expected-gradient) (fbits gradient)))))
    (let [x (float 0.1)
          seed (float 0.1)
          {:keys [gradient]} (lane-run bundle x 0.0 seed)
          a (fmul x x)
          t (fmul (fmul seed a) x)
          u (fmul seed x)
          fused-gradient (fadd (fadd (fadd (fadd (fadd t t) t) t) u) u)
          source-a-cotangent (fadd seed (fadd (fmul seed a) (fmul seed a)))
          source-node-gradient (fadd (fmul source-a-cotangent x)
                                     (fmul source-a-cotangent x))]
      (is (= (fbits fused-gradient) (fbits gradient)))
      (is (not= (fbits fused-gradient) (fbits source-node-gradient))
          "the admitted contract is the fused model, not rounded source-node transpose"))))

(deftest retained-public-input-order-is-not-a-type-erasing-canonicalization
  (let [source (typed-source square-chain-source)
        reordered (soac/make (update (soac/facts source) :inputs #(vec (reverse %)))
                             (soac/equations source) (soac/outputs source))
        bundle (build reordered)
        x (with-meta 'x {:raster.type/tag 'float})]
    (is (identical? reordered (get-in bundle [:map-programs :source])))
    (is (= (vec (reverse (:inputs (soac/facts source))))
           (:inputs (soac/facts (get-in bundle [:map-programs :source])))))
    (is (#'normalized/exact-semantic-members? '[n x] '[x n]))
    (is (not (#'normalized/exact-semantic-members? '[n x] (list x 'n))))))

(deftest composed-float-lane-retains-primal-and-adjoint-checkpoints
  (let [bundle (build (typed-source square-chain-source))]
    (doseq [[x dx seed] [[2.0 0.25 -0.5] [-3.0 0.75 1.25] [0.5 -2.0 3.0]]]
      (let [{:keys [primal tangent gradient]} (lane-run bundle x dx seed)
            x (float x) dx (float dx) seed (float seed)
            squared (float (* x x))
            expected-primal (float (* squared squared))
            producer-tangent (float (* (float (* (float 2.0) x)) dx))
            expected-tangent (float (* (float (* (float 2.0) squared)) producer-tangent))
            consumer-pull (float (* (float (* (float 2.0) squared)) seed))
            expected-gradient (float (* (float (* (float 2.0) x)) consumer-pull))]
        (is (= expected-primal primal))
        (is (= expected-tangent tangent))
        (is (= expected-gradient gradient))
        (is (< (Math/abs (- (double (* tangent seed))
                            (double (* dx gradient))))
               2.0e-4)))))
  (let [bundle (build (typed-source rounding-chain-source))
        {:keys [primal]} (lane-run bundle 4097.0 1.0 1.0)]
    (is (zero? (float primal))
        "the producer rounds 16785409 to Float before the consumer subtracts"))
  (let [bundle (build (typed-source square-chain-source))
        x (float 4097.0)
        dx (float 0.25)
        seed (float -0.5)
        {:keys [tangent gradient]} (lane-run bundle x dx seed)
        squared (float (* x x))
        producer-tangent (float (* (float (* (float 2.0) x)) dx))
        expected-tangent (float (* (float (* (float 2.0) squared)) producer-tangent))
        consumer-pull (float (* (float (* (float 2.0) squared)) seed))
        expected-gradient (float (* (float (* (float 2.0) x)) consumer-pull))]
    (is (= expected-tangent tangent)
        "the JVP retains the producer and consumer Float checkpoints")
    (is (= expected-gradient gradient)
        "the VJP retains the consumer and producer Float checkpoints")))

(deftest activity-may-select-an-external-array-introduced-by-the-consumer
  (let [source
        (frontend/form->program
         '(let* [squared (raster.par/pmap i n float
                                          (raster.numeric/* (clojure.core/aget x i)
                                                            (clojure.core/aget x i)))
                 scaled (raster.par/pmap j n float
                                         (raster.numeric/* (clojure.core/aget squared j)
                                                           (clojure.core/aget q j)))]
                scaled)
         {:dtype :float :array-types {'x :float 'q :float} :scalar-types {'n :long}})
        bundle (normalized/linearize-map-chain-program
                source (the-ns 'raster.ad.normalized-map-chain-test)
                {:active-input 'q})]
    (is (= ['q] (get-in bundle [:map-programs :interface :active-inputs])))
    (is (every? (set (:inputs (soac/facts (get-in bundle [:map-programs :source]))))
                '[x q]))
    (is (identical? bundle (normalized/validate! bundle)))))

(deftest nodewise-map-composition-does-not-turn-a-scalar-capture-into-an-array-gradient
  (let [source
        (frontend/form->program
         '(let* [paired (raster.par/pmap i n float
                                         (raster.numeric/* (clojure.core/aget x i)
                                                           (clojure.core/aget q i)))
                 result (raster.par/pmap j n float
                                         (raster.numeric/*
                                          (raster.numeric/* (clojure.core/aget paired j)
                                                            (clojure.core/aget r j))
                                          scale))]
                result)
         {:dtype :float :array-types {'x :float 'q :float 'r :float}
          :scalar-types {'n :long 'scale :float}})]
    (is (= :ad-normalized-map-scalar-requires-sum
           (reason #(normalized/linearize-map-chain-program
                     source (the-ns 'raster.ad.normalized-map-chain-test)
                     {:active-input 'scale}))))))

(deftest zero-extent-retains-the-same-composition-evidence
  (let [source
        (frontend/form->program
         '(let* [squared (raster.par/pmap i 0 float
                                          (raster.numeric/* (clojure.core/aget x i)
                                                            (clojure.core/aget x i)))
                 fourth (raster.par/pmap j 0 float
                                         (raster.numeric/* (clojure.core/aget squared j)
                                                           (clojure.core/aget squared j)))]
                fourth)
         {:dtype :float :array-types {'x :float}})
        bundle (build source)]
    (is (identical? source (get-in bundle [:map-programs :source])))
    (is (= 0 (get-in (soac/operation-parts
                      (first (soac/equations (get-in bundle [:map-programs :primal]))))
                     [:attributes :extent])))
    (is (identical? bundle (normalized/validate! bundle)))))

(deftest chain-validation-rebuilds-source-fusion-and-generated-programs
  (let [bundle (build (typed-source square-chain-source))
        source (get-in bundle [:map-programs :source])
        residual (first (vals (get-in bundle [:map-programs :interface :residual-arrays])))
        changed-source (alter-facts source #(assoc-in % [:values 'squared :dtype] :double))
        changed-program (update-in bundle [:map-programs :linear]
                                   alter-facts
                                   #(assoc-in % [:values residual :ownership] :borrowed))]
    (is (some? (reason #(normalized/validate! (recertify
                                               (assoc-in bundle [:map-programs :source]
                                                         changed-source))))))
    (is (= :ad-normalized-map-programs
           (reason #(normalized/validate! (recertify changed-program)))))
    (is (fingerprint/equivalent? source (get-in bundle [:map-programs :source])))))

(deftest nonchains-and-unsafe-composition-boundaries-refuse
  (let [one-map (typed-source
                 '(let* [only (raster.par/pmap i n float
                                                (raster.numeric/* (clojure.core/aget x i)
                                                                  (clojure.core/aget x i)))]
                        only))
        wrong-extent
        (frontend/form->program
         '(let* [squared (raster.par/pmap i n float
                                          (raster.numeric/* (clojure.core/aget x i)
                                                            (clojure.core/aget x i)))
                 shifted (raster.par/pmap j m float
                                          (raster.numeric/+ (clojure.core/aget squared j)
                                                            (clojure.core/float 1.0)))]
                shifted)
         {:dtype :float :array-types {'x :float} :scalar-types {'n :long 'm :long}})
        chain (typed-source square-chain-source)
        public-intermediate (soac/make (soac/facts chain) (soac/equations chain)
                                       '[squared fourth])
        long-chain (typed-source three-map-source)
        public-long-intermediate (soac/make (soac/facts long-chain)
                                            (soac/equations long-chain)
                                            '[squared eighth])
        horizontal
        (frontend/form->program
         '(let* [left (raster.par/pmap i n float
                                       (raster.numeric/* (clojure.core/aget x i)
                                                         (clojure.core/aget x i)))
                 right (raster.par/pmap j n float
                                        (raster.numeric/* (clojure.core/aget q j)
                                                          (clojure.core/aget q j)))]
                right)
         {:dtype :float :array-types {'x :float 'q :float} :scalar-types {'n :long}})
        dag
        (frontend/form->program
         '(let* [a (raster.par/pmap i n float
                                    (raster.numeric/* (clojure.core/aget x i)
                                                      (clojure.core/aget x i)))
                 b (raster.par/pmap j n float
                                    (raster.numeric/* (clojure.core/aget a j)
                                                      (clojure.core/aget a j)))
                 c (raster.par/pmap k n float
                                    (raster.numeric/* (clojure.core/aget a k)
                                                      (clojure.core/aget b k)))]
                c)
         {:dtype :float :array-types {'x :float} :scalar-types {'n :long}})
        skipped
        (frontend/form->program
         '(let* [a (raster.par/pmap i n float
                                    (raster.numeric/* (clojure.core/aget x i)
                                                      (clojure.core/aget x i)))
                 b (raster.par/pmap j n float
                                    (raster.numeric/* (clojure.core/aget q j)
                                                      (clojure.core/aget q j)))
                 c (raster.par/pmap k n float
                                    (raster.numeric/* (clojure.core/aget a k)
                                                      (clojure.core/aget b k)))]
                c)
         {:dtype :float :array-types {'x :float 'q :float} :scalar-types {'n :long}})
        conversion
        (frontend/form->program
         '(let* [converted (raster.par/pmap i n float
                                            (clojure.core/float (clojure.core/aget x i)))
                 copied (raster.par/pmap j n float (clojure.core/aget converted j))]
                copied)
         {:dtype :float :array-types {'x :long} :scalar-types {'n :long}})
        three-fused-arrays
        (frontend/form->program
         '(let* [squared (raster.par/pmap i n float
                                          (raster.numeric/* (clojure.core/aget x i)
                                                            (clojure.core/aget x i)))
                 combined (raster.par/pmap j n float
                                           (raster.numeric/*
                                            (raster.numeric/*
                                             (clojure.core/aget squared j)
                                             (clojure.core/aget q j))
                                            (clojure.core/aget r j)))]
                combined)
         {:dtype :float :array-types {'x :float 'q :float 'r :float}
          :scalar-types {'n :long}})
        three-boundary-spine
        (frontend/form->program
         '(let* [paired (raster.par/pmap i n float
                                         (raster.numeric/* (clojure.core/aget x i)
                                                           (clojure.core/aget q i)))
                 combined (raster.par/pmap j n float
                                           (raster.numeric/*
                                            (clojure.core/aget paired j)
                                            (clojure.core/aget r j)))]
                combined)
         {:dtype :float :array-types {'x :float 'q :float 'r :float}
          :scalar-types {'n :long}})]
    (is (= :ad-normalized-map-chain-boundary
           (reason #(build one-map))))
    (is (some? (reason #(build wrong-extent))))
    (is (some? (reason #(build public-intermediate))))
    (is (some? (reason #(build public-long-intermediate))))
    (let [bundle (build dag)]
      (is (identical? bundle (normalized/validate! bundle))))
    (let [bundle (normalized/linearize-map-chain-program
                  skipped (the-ns 'raster.ad.normalized-map-chain-test)
                  {:active-inputs ['q 'x]})]
      (is (= ['q 'x] (get-in bundle [:map-programs :interface :active-inputs])))
      (is (identical? bundle (normalized/validate! bundle))))
    (let [bundle (normalized/linearize-map-chain-program
                  three-boundary-spine (the-ns 'raster.ad.normalized-map-chain-test)
                  {:active-input 'x})
          nodes (get-in bundle [:map-programs :proof :nodes])
          residual (first (vals (get-in nodes [0 :stage-interface :residual-arrays])))
          aliased (assoc-in bundle
                            [:map-programs :proof :nodes 1 :stage-interface :tangent-inputs 0]
                            residual)]
      (is (= 2 (count nodes)))
      (is (= (mapv second (soac/equations three-boundary-spine))
             (mapv (comp first :source-equations) nodes)))
      (is (every? :scalar-derivation nodes))
      (is (identical? bundle (normalized/validate! bundle)))
      (is (= :ad-normalized-map-node-proof
             (reason #(normalized/validate!
                       (recertify (update bundle :map-programs dissoc :proof))))))
      (is (= :ad-normalized-map-node-interface
             (reason #(normalized/validate! (recertify aliased))))))
    (doseq [[label source] [[:horizontal horizontal]
                            [:active-conversion conversion]
                            [:three-operand-node three-fused-arrays]]]
      (is (some? (reason #(build source))) (name label)))))

(deftest effects-aliases-and-physical-storage-cannot-enter-chain-evidence
  (let [source (typed-source square-chain-source)
        [producer] (soac/equations source)
        producer-id (second producer)
        reject (fn [change]
                 (reason
                  #(let [facts (change (soac/facts source))
                         changed (soac/make facts (soac/equations source) (soac/outputs source))]
                     (build changed))))]
    (is (some? (reject #(-> %
                            (assoc :effects #{:memory/write})
                            (assoc-in [:equations producer-id :effects] #{:memory/write})))))
    (is (some? (reject #(assoc-in % [:equations producer-id :aliases]
                                  {'squared 'x}))))
    (is (some? (reject #(-> %
                            (assoc-in [:equations producer-id :aliases] {'squared 'x})
                            (assoc-in [:equations producer-id :attributes :result-storage]
                                      [{:destination 'x :access :write
                                        :host-return :buffer}])))))))
