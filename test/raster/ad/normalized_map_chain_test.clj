(ns raster.ad.normalized-map-chain-test
  (:require [clojure.test :refer [deftest is]]
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

(defn- alter-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

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
    (is (= :ad-program-bundle-composition
           (reason #(canonical/adapt-map bundle)))
        "runtime projection refuses before assigning single-equation residual origins")))

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
    (is (= 'q (get-in bundle [:map-programs :interface :active-input])))
    (is (every? (set (:inputs (soac/facts (get-in bundle [:map-programs :source]))))
                '[x q]))
    (is (identical? bundle (normalized/validate! bundle)))))

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
        three-maps
        (frontend/form->program
         '(let* [a (raster.par/pmap i n float (clojure.core/aget x i))
                 b (raster.par/pmap j n float (clojure.core/aget a j))
                 c (raster.par/pmap k n float (clojure.core/aget b k))]
                c)
         {:dtype :float :array-types {'x :float} :scalar-types {'n :long}})
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
          :scalar-types {'n :long}})]
    (is (= :ad-normalized-map-chain-boundary
           (reason #(build one-map))))
    (is (some? (reason #(build wrong-extent))))
    (is (some? (reason #(build public-intermediate))))
    (doseq [[label source] [[:horizontal horizontal] [:extra-equation three-maps]
                            [:active-conversion conversion]
                            [:three-fused-arrays three-fused-arrays]]]
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
