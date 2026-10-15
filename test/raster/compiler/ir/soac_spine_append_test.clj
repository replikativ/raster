(ns raster.compiler.ir.soac-spine-append-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-sum-test :as sums]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.numeric]
            [raster.par]))

(defn- parts []
  (soac/equation-subprograms (#'sums/source :double)))

(defn- facts-change [p f]
  (soac/make (f (soac/facts p)) (soac/equations p) (soac/outputs p)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- scoped-product-primal []
  (let [body '(raster.numeric/+ acc
                (raster.numeric/* (clojure.core/aget lhs (+ (* i 2) l))
                                  (clojure.core/aget rhs (+ (* j 2) l))))
        attributes {:segment-axes '[[i 2] [j 3]] :index 'l :extent 2
                    :result-map {:groups '[[[i 2] [j 3]]]}
                    :accumulators '[acc] :identities [0.0] :dtypes [:double]
                    :algebra [(scan/certify-reassociation {:acc 'acc :init 0.0 :lambda body} :double)]
                    :source-arithmetic (numerics/blas-source-arithmetic :double)
                    :attributes {:stable-array-captures '[A B] :pointwise-storage-inputs []}}
        product (list '= 'product '[C]
                      (list 'segmented-reduce attributes [] '[A B]
                            (soac/lambda-form '[acc lhs rhs] [] [body])))
        snapshot (list '= 'snapshot '[B-copy]
                       (list 'map {:index 'p :extent 6} '[B] [] (soac/lambda-form '[x] [] '[x])))
        value (fn [n ownership] (av/tensor {:dtype :double :shape [n] :ownership ownership}))]
    (soac/make (soac/default-program-facts
                {:values {'A (value 4 nil) 'B (value 6 nil) 'C (value 6 :owned) 'B-copy (value 6 :owned)}
                 :inputs '[A B] :equations {'product (soac/default-equation-facts)
                                           'snapshot (soac/default-equation-facts)}})
               [product snapshot] '[C B-copy])))

(defn- run-host [program environment]
  (let [{:keys [parameters body inputs]} (route/materialize-host-program program)]
    (apply (eval (list 'fn parameters body)) (mapv environment inputs))))

(deftest scoped-product-and-retained-map-append-in-both-execution-orders
  (let [product (scoped-product-primal)
        [reduction] (soac/equations product)
        owned (av/tensor {:dtype :double :shape [6] :ownership :owned})
        activation (list '= 'activation '[Y]
                         (list 'map {:index 'p :extent 6} '[C] []
                               (soac/lambda-form '[x] [] '[(raster.numeric/* x x)])))
        map-source (soac/make (soac/default-program-facts
                               {:values {'C owned 'Y owned} :inputs '[C]
                                :attributes {:source-arithmetic numerics/retained-source-arithmetic}
                                :equations {'activation (soac/default-equation-facts)}})
                              [activation] '[Y])
        combined (soac/append-pure-spines product map-source {'C 'C} '[Y B-copy])
        a [1.0 2.0 3.0 4.0] b [5.0 6.0 7.0 8.0 9.0 10.0]
        c (vec (for [i (range 2) j (range 3)]
                 (reduce + (for [l (range 2)] (* (nth a (+ (* i 2) l)) (nth b (+ (* j 2) l)))))))
        [y copied] (run-host combined {'A (double-array a) 'B (double-array b)})
        policies (fn [program]
                   (mapv #(numerics/effective-source-arithmetic
                           (get-in (soac/operation-parts %) [:attributes :source-arithmetic])
                           (get-in (soac/facts program) [:attributes :source-arithmetic])) (soac/equations program)))
        retained numerics/retained-source-arithmetic blas (numerics/blas-source-arithmetic :double)]
    (is (= [blas retained retained] (policies combined)))
    (is (= {} (:attributes (soac/facts combined))))
    (is (= (mapv #(* % %) c) (vec y)))
    (is (= b (vec copied)))
    (let [[local-product] (soac/equation-subprograms product)
          local-product (facts-change local-product #(assoc-in % [:attributes :source-arithmetic] blas))
          [equation] (soac/equations local-product)
          operation (nth equation 3)
          without-local (apply list (assoc (vec operation) 1 (dissoc (second operation) :source-arithmetic)))
          fallback-only (soac/make (soac/facts local-product)
                                   [(apply list (assoc (vec equation) 3 without-local))] '[C])]
      (is (= '[C] (:outputs (route/materialize-host-program local-product))))
      (is (= :typed-soac-host-fresh-segmented-reduce-boundary
             (reason #(route/materialize-host-program fallback-only)))
          "descriptive fallback resolution does not supply the host route's explicit local contract"))
    ;; Reverse-mode operation order: a retained unary cotangent map precedes
    ;; the segmented fiber. The mathematical reference is authored independently.
    (let [body '(raster.numeric/+ acc
                  (raster.numeric/* (clojure.core/aget seed (+ (* i 3) j))
                                    (clojure.core/aget rhs (+ (* j 2) l))))
          attributes (-> (second (nth reduction 3))
                         (assoc :segment-axes '[[i 2] [l 2]] :index 'j :extent 3
                                :result-map {:groups '[[[i 2] [l 2]]]}
                                :attributes {:stable-array-captures '[S B] :pointwise-storage-inputs []}
                                :algebra [(scan/certify-reassociation {:acc 'acc :init 0.0 :lambda body} :double)]))
          transpose (list '= 'fiber '[G]
                          (list 'segmented-reduce attributes [] '[S B]
                                (soac/lambda-form '[acc seed rhs] [] [body])))
          fiber (soac/make (soac/default-program-facts
                            {:values {'S owned 'B (av/tensor {:dtype :double :shape [6]})
                                      'G (av/tensor {:dtype :double :shape [4] :ownership :owned})}
                             :inputs '[S B] :equations {'fiber (soac/default-equation-facts)}}) [transpose] '[G])
          seed-map (list '= 'seed-map '[S]
                         (list 'map {:index 'p :extent 6} '[DY C] []
                               (soac/lambda-form '[dy coefficient] []
                                                 '[(raster.numeric/+ (raster.numeric/* dy coefficient)
                                                                    (raster.numeric/* dy coefficient))])))
          seed-source (soac/make (soac/default-program-facts
                                  {:values {'DY (assoc owned :ownership nil) 'C (assoc owned :ownership nil) 'S owned}
                                   :inputs '[DY C] :attributes {:source-arithmetic retained}
                                   :equations {'seed-map (soac/default-equation-facts)}}) [seed-map] '[S])
          reverse (soac/append-pure-spines seed-source fiber {'S 'S} '[G])
          dy [0.5 -1.0 2.0 0.25 -0.5 1.5]
          seed (mapv #(+ (* %1 %2) (* %1 %2)) dy c)
          expected (vec (for [i (range 2) l (range 2)]
                          (reduce + (for [j (range 3)]
                                      (* (nth seed (+ (* i 3) j)) (nth b (+ (* j 2) l)))))))
          [gradient] (run-host reverse {'DY (double-array dy) 'C (double-array c) 'B (double-array b)})]
      (is (= [retained blas] (policies reverse)))
      (is (= {} (:attributes (soac/facts reverse))))
      (is (= expected (vec gradient))))))

(deftest explicit-spine-preserves-facts-and-output-order
  (let [[p c] (parts)
        p (facts-change p #(-> % (assoc :provenance {:producer :kept})
                               (assoc-in [:equations 'map-0 :provenance] {:map :kept})))
        c (facts-change c #(-> % (assoc :provenance {:consumer :kept})
                               (assoc-in [:equations 'sum-1 :provenance] {:sum :kept})))
        combined (soac/append-pure-spines p c {'mapped 'mapped} '[loss mapped])]
    (is (= '[xs weights size] (:inputs (soac/facts combined))))
    (is (= '[loss mapped] (soac/outputs combined)))
    (is (= (concat (soac/equations p) (soac/equations c)) (soac/equations combined)))
    (is (= {:producer :kept :consumer :kept} (:provenance (soac/facts combined))))
    (doseq [part [p c] [id facts] (:equations (soac/facts part))]
      (is (fingerprint/equivalent? facts (get-in (soac/facts combined) [:equations id]))))
    (is (identical? combined (soac/validate! combined)))))

(deftest only-absent-and-retained-fallbacks-join-with-unchanged-operation-meaning
  (let [[p c] (parts)
        retained numerics/retained-source-arithmetic
        fallback (fn [program policy]
                   (facts-change program #(update % :attributes
                                                   (fn [attrs]
                                                     (if policy (assoc attrs :source-arithmetic policy)
                                                         (dissoc attrs :source-arithmetic))))))
        policies (fn [program]
                   (mapv #(numerics/effective-source-arithmetic
                           (get-in (soac/operation-parts %) [:attributes :source-arithmetic])
                           (get-in (soac/facts program) [:attributes :source-arithmetic]))
                         (soac/equations program)))]
    (doseq [[left right expected] [[nil nil nil] [retained retained retained]
                                 [nil retained nil] [retained nil nil]]]
      (let [producer (fallback p left) consumer (fallback c right)
            combined (soac/append-pure-spines producer consumer {'mapped 'mapped} '[loss mapped])]
        (is (= expected (get-in (soac/facts combined) [:attributes :source-arithmetic])))
        (is (= (into (policies producer) (policies consumer)) (policies combined)))
        (is (fingerprint/equivalent? (vec (concat (soac/equations producer) (soac/equations consumer)))
                                    (soac/equations combined)))))
    (doseq [policy [nil retained]]
      (let [producer (facts-change (fallback p policy) #(assoc-in % [:attributes :opaque] :producer))
            consumer (facts-change (fallback c policy) #(assoc-in % [:attributes :opaque] :consumer))]
        (is (= :typed-soac-composition-numerics
               (reason #(soac/append-pure-spines producer consumer {'mapped 'mapped} '[loss]))))))
    (is (= :typed-soac-composition-numerics
           (reason #(soac/append-pure-spines (fallback p nil)
                                           (fallback c (numerics/blas-source-arithmetic :double))
                                           {'mapped 'mapped} '[loss]))))))

(deftest accumulated-pure-spine-accepts-one-more-exact-consumer
  (let [source
        (frontend/form->program
         '(let* [a (raster.par/pmap i n float
                                     (raster.numeric/* (clojure.core/aget x i)
                                                       (clojure.core/aget x i)))
                 b (raster.par/pmap j n float
                                     (raster.numeric/+ (clojure.core/aget a j)
                                                       (clojure.core/float 1.0)))
                 c (raster.par/pmap k n float
                                     (raster.numeric/* (clojure.core/aget b k)
                                                       (clojure.core/float 2.0)))]
                c)
         {:dtype :float :array-types {'x :float} :scalar-types {'n :long}})
        [a b c] (soac/equation-subprograms source)
        ab (soac/append-pure-spines a b {'a 'a} '[b])
        abc (soac/append-pure-spines ab c {'b 'b} '[c])
        rebuilt (soac/make (assoc (soac/facts abc)
                                  :inputs (:inputs (soac/facts source)))
                           (soac/equations abc) (soac/outputs abc))]
    (is (= 2 (count (soac/equations ab))))
    (is (= (soac/equations source) (soac/equations abc)))
    (is (= (soac/outputs source) (soac/outputs abc)))
    (is (#'normalized/exact-semantic-members? (:inputs (soac/facts source))
                                               (:inputs (soac/facts abc))))
    (is (fingerprint/equivalent? source rebuilt))
    (is (identical? abc (soac/validate! abc)))))

(deftest conflicting-shared-value-facts-never-prefer-either-side
  (let [[p c] (parts)]
    (doseq [[left right] [[(av/tensor {:dtype :double :shape [1]})
                          (av/tensor {:dtype :float :shape [1]})]
                         [(av/tensor {:dtype :double :shape [1]})
                          (av/tensor {:dtype :double :shape [2]})]
                         [(av/tensor {:dtype :double :shape [1] :ownership :owned})
                          (av/tensor {:dtype :double :shape [1]})]]]
      (let [p (facts-change p #(assoc-in % [:values 'retained-fact] left))
            c (facts-change c #(assoc-in % [:values 'retained-fact] right))]
        (doseq [[a b boundary] [[p c {'mapped 'mapped}] [c p {}]]]
          (is (= :typed-soac-composition-facts
                 (reason #(soac/append-pure-spines a b boundary '[loss mapped])))))))))

(deftest non-identity-boundary-remaps-consumer-towards-producer
  (let [[p original] (parts)
        p (facts-change p #(assoc % :extension-contract {:opaque :same}))
        original (facts-change original #(assoc % :extension-contract {:opaque :same}))
        consumer (soac/remap-values original {'mapped 'consumer-input})
        combined (soac/append-pure-spines p consumer {'consumer-input 'mapped} '[loss mapped])]
    (is (not (contains? (:values (soac/facts combined)) 'consumer-input)))
    (is (= '[xs weights size] (:inputs (soac/facts combined))))
    (is (= '[loss mapped] (soac/outputs combined)))
    (is (= {:opaque :same} (:extension-contract (soac/facts combined))))
    (is (= (concat (soac/equations p) (soac/equations original)) (soac/equations combined)))
    (is (fingerprint/equivalent? (:equations (soac/facts original))
                                (select-keys (:equations (soac/facts combined)) ['sum-1])))))

(deftest boundaries-identities-and-numerical-contracts-refuse
  (let [[p c] (parts)]
    (is (= :typed-soac-composition-boundary
           (reason #(soac/append-pure-spines p c {} '[loss]))))
    (is (= :typed-soac-composition-boundary
           (reason #(soac/append-pure-spines
                     (soac/make (soac/facts p) (soac/equations p) []) c {} '[loss])))
        "private producer results cannot become an implicit boundary")
    (is (= :typed-soac-composition-boundary
           (reason #(soac/append-pure-spines p c {'size 'mapped} '[loss]))))
    (is (= :typed-soac-composition-numerics
           (reason #(soac/append-pure-spines p
                     (facts-change c (fn [f] (assoc-in f [:attributes :source-arithmetic] :different)))
                     {'mapped 'mapped} '[loss]))))
    (is (= :typed-soac-composition-numerics
           (reason #(soac/append-pure-spines p
                     (facts-change c (fn [f] (assoc f :extension-contract :consumer-only)))
                     {'mapped 'mapped} '[loss])))
        "an unknown consumer program contract is not silently discarded")
    (let [e (first (soac/equations c))
          duplicate (soac/make (-> (soac/facts c)
                                  (assoc :equations {'map-0 (get-in (soac/facts c) [:equations 'sum-1])}))
                              [(apply list (assoc (vec e) 1 'map-0))] (soac/outputs c))]
      (is (= :typed-soac-equation-ids
             (reason #(soac/append-pure-spines p duplicate {'mapped 'mapped} '[loss])))))
    (let [duplicate (soac/remap-values c {'mapped 'temporary-input 'loss 'mapped})]
      (is (= :typed-soac-remap-collision
             (reason #(soac/append-pure-spines p duplicate {'temporary-input 'mapped} '[mapped])))))
    (is (= :typed-soac-output-boundary
           (reason #(soac/append-pure-spines p c {'mapped 'mapped} '[xs]))))))

(deftest effects-aliases-and-provenance-conflicts-are-not-erased
  (let [[p c] (parts)]
    (doseq [changed [(facts-change c #(-> % (assoc :effects #{:memory/read})
                                         (assoc-in [:equations 'sum-1 :effects] #{:memory/read})))
                     (facts-change c #(assoc-in % [:equations 'sum-1 :aliases] {'loss 'mapped}))]]
      (is (= :typed-soac-composition-boundary
             (reason #(soac/append-pure-spines p changed {'mapped 'mapped} '[loss])))))
    (is (= :typed-soac-composition-facts
           (reason #(soac/append-pure-spines
                     (facts-change p (fn [f] (assoc f :provenance {:shared :producer})))
                     (facts-change c (fn [f] (assoc f :provenance {:shared :consumer})))
                     {'mapped 'mapped} '[loss]))))))

(deftest ad-owned-boundary-is-an-explicit-derivation-not-a-merge-preference
  (doseq [dt [:float :double] ownership [nil :owned]]
    (let [source (facts-change (#'sums/source dt)
                               #(assoc-in % [:values 'mapped :ownership] ownership))
          bundle (#'sums/build source 'xs)
          sum-programs (:sum-programs bundle)
          tangent (get-in bundle [:map-programs :interface :tangent-output])]
      (is (identical? source (:source sum-programs)))
      (is (= ownership (get-in (soac/facts source) [:values 'mapped :ownership])))
      (is (= :owned (get-in (soac/facts (:primal sum-programs)) [:values 'mapped :ownership])))
      (is (= :owned (get-in (soac/facts (:linear sum-programs)) [:values tangent :ownership])))))
  (let [source (#'sums/source :double) bundle (#'sums/build source 'xs)
        producer (get-in bundle [:map-programs :primal])
        reduction (second (soac/equations source))]
    (doseq [ownership [:borrowed :external]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (#'sums/build (facts-change source #(assoc-in % [:values 'mapped :ownership] ownership)) 'xs))))
    (doseq [changed [(facts-change producer #(assoc-in % [:values 'mapped :ownership] nil))
                     (facts-change producer #(assoc-in % [:values 'mapped :attributes] {:different true}))]]
      (is (= :ad-normalized-sum-boundary
             (reason #(#'normalized/append-ordered-sum changed reduction source 'loss)))))))
