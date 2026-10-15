(ns raster.compiler.passes.parallel.host-operation-admission-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-element-test :as maps]
            [raster.ad.normalized-map-sum-test :as sums]
            [raster.ad.normalized-indexed-map-test :as indexed]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.par]
            [raster.numeric]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- changed [program f]
  (soac/make (f (soac/facts program)) (soac/equations program) (soac/outputs program)))

(defn- changed-operation-attributes [program f]
  (let [[equation] (soac/equations program)
        operation (nth equation 3)
        operation (apply list (assoc (vec operation) 1 (f (second operation))))
        equation (apply list (assoc (vec equation) 3 operation))]
    (soac/make (soac/facts program) [equation] (soac/outputs program))))

(defn- changed-segmented-body [program f]
  (let [[equation] (soac/equations program)
        operation (nth equation 3)
        attributes (second operation)
        arrays (nth operation 2)
        captures (nth operation 3)
        {:keys [parameters locals body-results]} (soac/lambda-parts (nth operation 4))
        body (f (first body-results))
        dtype (first (:dtypes attributes))
        algebra (scan/certify-reassociation
                 {:acc (first (:accumulators attributes))
                  :init (first (:identities attributes)) :lambda body} dtype)
        attributes (assoc attributes :algebra [algebra])
        operation (list 'segmented-reduce attributes arrays captures
                        (soac/lambda-form parameters locals [body]))
        equation (apply list (assoc (vec equation) 3 operation))]
    (soac/make (soac/facts program) [equation] (soac/outputs program))))

(defn- map-equation [id results index extent arrays parameters bodies]
  (list '= id (vec results)
        (list 'map {:index index :extent extent} (vec arrays) []
              (soac/lambda-form (vec parameters) [] (vec bodies)))))

(defn- graph-program [dt extent equations outputs inputs]
  (let [array-ids (set (concat (mapcat #(nth % 2) equations)
                               (mapcat (comp :arrays soac/operation-parts) equations)
                               outputs))
        values (into (if (symbol? extent)
                       {extent (av/tensor {:dtype :long :shape []})} {})
                     (map #(vector % (av/tensor {:dtype dt :shape [extent]})) array-ids))]
    (soac/make
     (soac/default-program-facts
      {:values values :inputs inputs
       :equations (into {} (map #(vector (second %) (soac/default-equation-facts)) equations))})
     equations outputs)))

(defn- chain-graph [dt extent]
  (graph-program
   dt extent
   [(map-equation 'mul '[a] 'i extent '[x q] '[xe qe]
                  '[(raster.numeric/* xe qe)])
    (map-equation 'add '[b] 'j extent '[a r] '[ae re]
                  '[(raster.numeric/+ ae re)])]
   '[b a] (if (symbol? extent) [extent 'x 'q 'r] '[x q r])))

(defn- diamond-graph [dt extent]
  (graph-program
   dt extent
   [(map-equation 'square '[a] 'i extent '[x] '[xe]
                  '[(raster.numeric/* xe xe)])
    (map-equation 'shift '[b] 'j extent '[a] '[ae]
                  '[(raster.numeric/+ ae (clojure.core/float 1.0))])
    (map-equation 'fourth '[c] 'k extent '[a] '[ae]
                  '[(raster.numeric/* ae ae)])
    (map-equation 'join '[y] 'l extent '[b c] '[be ce]
                  '[(raster.numeric/+ be ce)])]
   '[y a c] (if (symbol? extent) [extent 'x] '[x])))

(defn- tuple-graph [dt extent]
  (graph-program
   dt extent
   [(map-equation 'pair '[a shifted] 'i extent '[x] '[xe]
                  '[(raster.numeric/* xe xe)
                    (raster.numeric/+ xe xe)])
    (map-equation 'join '[y] 'j extent '[a shifted] '[ae se]
                  '[(raster.numeric/+ ae se)])]
   '[y a shifted] (if (symbol? extent) [extent 'x] '[x])))

(defn- run-host [program arguments]
  (let [{:keys [parameters body]} (route/materialize-host-program program)]
    (apply (eval (list 'fn parameters body)) arguments)))

(defn- segmented-program
  ([dt] (segmented-program dt nil))
  ([dt destination]
   (let [init (if (= :float dt) '(clojure.core/float 0.0) 0.0)
         body '(raster.numeric/+ acc
                  (raster.numeric/*
                   (clojure.core/aget lhs (clojure.core/+ (clojure.core/* i 2) l))
                   (clojure.core/aget rhs (clojure.core/+ (clojure.core/* j 2) l))))
         algebra (scan/certify-reassociation {:acc 'acc :init init :lambda body} dt)
         arithmetic (numerics/blas-source-arithmetic dt)
         equation
         (list '= 'dense-product '[C]
               (list 'segmented-reduce
                     {:segment-axes '[[i 2] [j 3]] :index 'l :extent 2
                      :accumulators '[acc] :identities [init] :dtypes [dt]
                      :algebra [algebra] :source-arithmetic arithmetic
                      :attributes {:stable-array-captures '[A B]
                                   :pointwise-storage-inputs []}}
                     [] '[A B] (soac/lambda-form '[acc lhs rhs] [] [body])))
         value (fn [shape ownership]
                 (av/tensor {:dtype dt :shape shape :ownership ownership}))
         equation-facts
         (cond-> (soac/default-equation-facts)
           destination
           (assoc :effects #{:memory/write}
                  :aliases {'C destination}
                  :attributes {:result-storage
                               [{:destination destination :access :write
                                 :host-return :buffer}]}))
         facts (soac/default-program-facts
                {:values (cond-> {'A (value [2 2] nil) 'B (value [3 2] nil)
                                  'C (value [2 3] :owned)}
                           destination (assoc destination (value [2 3] :owned)))
                 :inputs '[A B]
                 :effects (if destination #{:memory/write} #{})
                 :aliases (if destination {'C destination} {})
                 :attributes {:source-arithmetic arithmetic}
                 :equations {'dense-product equation-facts}})]
     (soac/make facts [equation] '[C]))))

(defn- run-projection [projection arguments]
  (apply (eval (list 'fn (:parameters projection) (:body projection))) arguments))

(defn- fbits [^floats values]
  (mapv #(Float/floatToRawIntBits (aget values %)) (range (alength values))))

(defn- independent-graph [left-dtype right-dtype left-extent right-extent]
  (let [left (graph-program
              left-dtype left-extent
              [(map-equation 'left '[a] 'i left-extent '[x] '[xe]
                             '[(raster.numeric/+ xe xe)])]
              '[a] (if (symbol? left-extent) [left-extent 'x] '[x]))
        right (graph-program
               right-dtype right-extent
               [(map-equation 'right '[b] 'j right-extent '[q] '[qe]
                              '[(raster.numeric/+ qe qe)])]
               '[b] (if (symbol? right-extent) [right-extent 'q] '[q]))]
    (soac/append-pure-spines left right {} '[a b])))

(deftest admission-preserves-existing-logical-and-physical-projections
  (doseq [dt [:float :double]]
    (let [map-source (#'maps/map-program dt)
          sum-source (#'sums/source dt)
          indexed-bundles (for [it [:int :long]]
                            (normalized/linearize-indexed-map-program
                             (#'indexed/indexed-map dt it 'n 'm) *ns*))
          programs (concat [[map-source route/materialize-host-map]
                            [sum-source route/materialize-host-map-reduce]]
                           (for [bundle indexed-bundles
                                 phase [:primal :linear :transpose]]
                             [(get-in bundle [:gather-programs phase]) route/materialize-host-indexed-map]))]
      ;; Source generation stays in its existing owner. Compare the exact checked roles
      ;; supplied to it, avoiding string/gensym equality or a second execution engine.
      (with-redefs-fn {#'route/host-projection
                      (fn [program physical outputs]
                        {:program program :physical-inputs physical :output-values outputs})}
        (fn []
          (doseq [[program specialized] programs]
            (is (= (specialized program) (route/materialize-host-program program)))
            (is (identical? program (:program (route/materialize-host-program program))))))))))

(deftest specialization-boundaries-and-topology-remain-closed
  (let [pure (#'maps/map-program :double)
        gather (#'indexed/indexed-map :double :long 'n 'm)
        reduction (second (soac/equation-subprograms (#'sums/source :double)))]
    (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map gather))))
    (is (= :typed-soac-host-program-topology (reason #(route/materialize-host-program reduction))))
    (doseq [program [(changed pure #(assoc-in % [:attributes :unknown-host-contract] true))
                     (changed pure #(assoc-in % [:values 'x :memory-space] :device))]]
      (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-program program))))
      (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map program)))))
    (let [bad-gather (changed gather #(assoc-in % [:values 'out :ownership] :borrowed))]
      (is (= :typed-soac-host-indexed-boundary (reason #(route/materialize-host-program bad-gather))))
      (is (= :typed-soac-host-indexed-boundary (reason #(route/materialize-host-indexed-map bad-gather)))))))

(deftest generic-and-specialized-admission-validate-the-program-once
  (let [program (#'maps/map-program :double)
        validations (atom 0) validate soac/validate!]
    (with-redefs-fn {#'soac/validate! (fn [p] (swap! validations inc) (validate p))
                    #'route/host-projection (fn [_ physical outputs] [physical outputs])}
      (fn []
        (route/materialize-host-program program)
        (is (= 1 @validations))
        (route/materialize-host-map program)
        (is (= 2 @validations))))))

(deftest fresh-segmented-reduction-matches-the-existing-stored-host-realization
  (let [fresh (segmented-program :float)
        stored (segmented-program :float 'destination)
        a (float-array [4097.0 -1.25 0.03125 8193.0])
        b (float-array [0.1 -0.3 1025.0 0.0009765625 -7.0 11.0])
        a-before (fbits a)
        b-before (fbits b)
        fresh-projection (route/materialize-host-program fresh)
        swapped-projection (route/materialize-host-program
                            (changed fresh #(assoc % :inputs '[B A])))
        stored-projection (#'route/host-projection stored '[destination] '[C])
        [first-result] (run-projection fresh-projection [a b])
        [second-result] (run-projection fresh-projection [a b])
        [swapped-result] (run-projection swapped-projection [b a])
        destination (float-array 6)
        [stored-result] (run-projection stored-projection [a b destination])
        [_ bindings returned] (:body fresh-projection)
        [allocated allocation effect call] bindings]
    (is (= (fbits stored-result) (fbits first-result) (fbits second-result)
           (fbits swapped-result)))
    (is (every? #(Float/isFinite %) first-result))
    (is (not (identical? first-result second-result)))
    (is (not (identical? first-result a)))
    (is (not (identical? first-result b)))
    (is (identical? destination stored-result))
    (is (= a-before (fbits a)))
    (is (= b-before (fbits b)))
    (is (= 2 (count (:parameters fresh-projection))))
    (is (= 3 (count (:parameters stored-projection))))
    ;; Fresh realization allocates before invoking the unchanged contraction form and returns
    ;; the allocated logical result. Successful execution above checks the generated scope.
    (is (= 4 (count bindings)))
    (is (= allocated (get (:value-symbols fresh-projection) 'C)))
    (is (= allocated (second call)))
    (is (= [allocated] returned))
    (is (not= allocated effect))
    (is (not (some #{allocated effect} (:parameters fresh-projection))))
    (is (seq allocation))))

(deftest fresh-double-segmented-reduction-has-the-declared-product
  (let [[result] (run-host (segmented-program :double)
                           [(double-array [1.0 2.0 3.0 4.0])
                            (double-array [5.0 6.0 7.0 8.0 9.0 10.0])])]
    (is (= [17.0 23.0 29.0 39.0 53.0 67.0] (vec result)))))

(deftest fresh-segmented-reduction-boundary-is-closed
  (let [program (segmented-program :float)
        arithmetic (numerics/blas-source-arithmetic :float)
        eq-id (second (first (soac/equations program)))
        refuse (fn [candidate]
                 (= :typed-soac-host-fresh-segmented-reduce-boundary
                    (reason #(route/materialize-host-program candidate))))]
    (is (refuse (changed program #(assoc-in % [:values 'C :ownership] nil))))
    ;; Read minima stay small when the summand broadcasts over the reduction axis.
    ;; The host macro still narrows its loop bound to int, independently of storage.
    (let [broadcast (changed-segmented-body
                     program
                     #(walk/postwalk-replace
                       {'(clojure.core/+ (clojure.core/* i 2) l) 'i
                        '(clojure.core/+ (clojure.core/* j 2) l) 'j}
                       %))]
      (is (some? (route/materialize-host-program broadcast)))
      (is (refuse (changed-operation-attributes broadcast #(assoc % :extent 2147483648)))))
    (is (refuse (changed program #(assoc-in % [:values 'A :memory-space] :device))))
    (is (refuse (changed program #(assoc-in % [:values 'A :dtype] :double))))
    (is (refuse (changed program #(assoc-in % [:values 'A :shape] [50000 50000]))))
    (is (refuse (changed program #(assoc-in % [:values 'A :shape] [1]))))
    (is (refuse (changed program #(assoc-in % [:values 'B :shape] [1]))))
    (is (refuse (changed program #(update-in % [:values 'A] assoc :unknown-facet true))))
    (is (refuse (changed program #(update-in % [:values 'C] assoc :unknown-facet true))))
    (is (refuse (changed program #(assoc-in % [:attributes :unknown] true))))
    (is (some? (reason (fn [] (route/materialize-host-program
                                (changed program #(assoc % :inputs '[A B C])))))))
    (is (refuse (changed program #(assoc-in % [:values 'unused]
                                             (av/tensor {:dtype :float :shape [1]})))))
    (is (refuse (changed program #(assoc-in % [:equations eq-id :attributes]
                                             {:unexpected true}))))
    (is (refuse (changed program #(assoc-in % [:attributes :source-arithmetic]
                                             numerics/retained-source-arithmetic))))
    (is (refuse
         (soac/make (soac/facts program)
                    (walk/postwalk-replace {'i 'float} (soac/equations program))
                    (soac/outputs program))))
    (is (refuse
         (changed-operation-attributes
          (changed-segmented-body program #(walk/postwalk-replace {'i 0} %))
          #(assoc % :segment-axes '[[qualified/i 2] [j 3]]))))
    (is (refuse
         (changed-segmented-body
          program
          #(walk/postwalk-replace
            {'(clojure.core/+ (clojure.core/* j 2) l)
             '(clojure.core/rem (clojure.core/+ (clojure.core/* j 2) l) 6)}
            %))))
    (is (refuse (changed-operation-attributes program #(dissoc % :source-arithmetic))))
    (is (= :typed-soac-syntax
           (reason (fn []
                     (changed-operation-attributes
                      program #(assoc % :source-arithmetic
                                      (numerics/blas-source-arithmetic :double)))))))
    (is (refuse (changed-operation-attributes program #(assoc % :result-transform nil))))
    (is (some? (reason (fn [] (route/materialize-host-program
                                (changed-operation-attributes
                                 program #(assoc % :stages [])))))))
    (is (some? (reason (fn [] (route/materialize-host-program
                                (changed-operation-attributes
                                 program #(assoc % :decode {})))))))
    (is (refuse (segmented-program :float 'destination)))
    (let [[equation] (soac/equations program)
          operation (nth equation 3)
          attributes (assoc (second operation) :segment-axes '[[i 50000] [j 50000]])
          operation (apply list (assoc (vec operation) 1 attributes))
          equation (apply list (assoc (vec equation) 3 operation))
          facts (assoc-in (soac/facts program) [:values 'C :shape] [50000 50000])
          oversized (soac/make facts [equation] (soac/outputs program))]
      (is (refuse oversized)))
    (is (= :typed-soac-host-map-boundary
           (reason #(route/materialize-host-map program))))
    (is (= arithmetic
           (get-in (soac/operation-parts (first (soac/equations program)))
                   [:attributes :source-arithmetic])))))

(deftest map-graph-validates-the-original-through-public-and-batch-boundaries
  (let [program (diamond-graph :float 'n)
        validations (atom 0) validate soac/validate!]
    (with-redefs-fn {#'soac/validate! (fn [candidate]
                                       (when (identical? candidate program)
                                         (swap! validations inc))
                                       (validate candidate))
                    #'route/host-projection (fn [_ physical outputs] [physical outputs])}
      (fn []
        (route/materialize-host-program program)
        (is (= 2 @validations)
            "public admission and the validating batch projector each check the original")))))

(deftest ordered-map-graphs-materialize-all-declared-results
  (let [[b a] (run-host (chain-graph :double 'n)
                        [(long 3) (double-array [1.0 2.0 -3.0])
                         (double-array [4.0 -0.5 2.0])
                         (double-array [0.25 8.0 -1.0])])]
    (is (= [4.25 7.0 -7.0] (vec b)))
    (is (= [4.0 -1.0 -6.0] (vec a))))
  (let [[y a c] (run-host (diamond-graph :float 'n)
                          [(long 3) (float-array [1.0 2.0 -3.0])])]
    (is (= (mapv float [3.0 21.0 91.0]) (vec y)))
    (is (= (mapv float [1.0 4.0 9.0]) (vec a)))
    (is (= (mapv float [1.0 16.0 81.0]) (vec c))))
  (let [[y a shifted] (run-host (tuple-graph :double 'n)
                                [(long 2) (double-array [2.0 -3.0])])]
    (is (= [8.0 3.0] (vec y)))
    (is (= [4.0 9.0] (vec a)))
    (is (= [4.0 -6.0] (vec shifted))))
  (is (= [[] [] []]
         (mapv vec (run-host (diamond-graph :float 0) [(float-array 0)])))))

(deftest ordered-map-graph-retains-float-checkpoints
  (let [program
        (graph-program
         :float 'n
         [(map-equation 'square '[a] 'i 'n '[x] '[xe]
                        '[(raster.numeric/* xe xe)])
          (map-equation 'shift '[y] 'j 'n '[a] '[ae]
                        '[(raster.numeric/- ae (clojure.core/float 16785408.0))])]
         '[y a] '[n x])
        [y a] (run-host program [(long 1) (float-array [4097.0])])]
    (is (= [(float 16785408.0)] (vec a)))
    (is (= [(float 0.0)] (vec y)))))

(deftest map-graph-boundary-refuses-unsafe-or-open-programs
  (let [program (diamond-graph :float 'n)
        equations (vec (soac/equations program))
        facts (soac/facts program)
        dead-equation (map-equation 'dead '[dead] 'd 'n '[x] '[xe]
                                    '[(raster.numeric/+ xe xe)])
        dead (soac/make
              (-> facts
                  (assoc-in [:values 'dead] (av/tensor {:dtype :float :shape '[n]}))
                  (assoc-in [:equations 'dead] (soac/default-equation-facts)))
              (conj equations dead-equation) (soac/outputs program))
        captured-square
        (list '= 'square '[a]
              (list 'map {:index 'i :extent 'n} '[x] '[scale]
                    (soac/lambda-form '[xe scale-element] []
                                      '[(raster.numeric/* xe xe)])))
        mixed-capture
        (soac/make
         (-> facts
             (assoc :inputs '[n x scale])
             (assoc-in [:values 'scale] (av/tensor {:dtype :double :shape []})))
         (assoc equations 0 captured-square) (soac/outputs program))]
    (is (= :typed-soac-host-map-graph-boundary
           (reason #(route/materialize-host-program dead))))
    (is (= :typed-soac-host-map-graph-boundary
           (reason #(route/materialize-host-program
                     (independent-graph :float :float 'n 'm)))))
    (is (= :typed-soac-host-map-graph-boundary
           (reason #(route/materialize-host-program
                     (independent-graph :float :double 'n 'n)))))
    (is (= :typed-soac-host-map-graph-boundary
           (reason #(route/materialize-host-program mixed-capture))))
    (doseq [change [#(assoc-in % [:attributes :unknown-host-contract] true)
                    #(assoc-in % [:attributes :host-read-values] ['x])
                    #(assoc-in % [:values 'x :memory-space] :device)
                    #(assoc-in % [:values 'a :dtype] :double)
                    #(-> %
                         (assoc :effects #{:memory/write})
                         (assoc-in [:equations 'square :effects] #{:memory/write}))
                    #(-> %
                         (assoc :aliases {'a 'x})
                         (assoc-in [:equations 'square :aliases] {'a 'x}))
                    #(-> %
                         (assoc :aliases {'a 'x})
                         (assoc-in [:equations 'square :aliases] {'a 'x})
                         (assoc-in [:equations 'square :attributes :result-storage]
                                   [{:destination 'x :access :write
                                     :host-return :buffer}]))]]
      (is (some? (reason #(route/materialize-host-program (changed program change))))))
    (let [first-id (second (first equations))
          duplicate (apply list (assoc (vec (second equations)) 1 first-id))
          malformed (list 'soac-program facts (assoc equations 1 duplicate)
                          (soac/outputs program))]
      (is (= :typed-soac-equation-ids
             (reason #(route/materialize-host-program malformed)))))))
