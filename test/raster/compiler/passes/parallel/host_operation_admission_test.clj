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

(defn- append-ordered-sum [program dt input extent]
  (let [zero (if (= dt :float) (float 0.0) 0.0)
        step '(raster.numeric/+ total element)
        equation (list '= 'ordered-loss '[loss]
                       (list 'reduce
                             {:index 'sum-index :extent extent :accumulators '[total]
                              :identities [zero] :dtypes [dt]
                              :algebra [(scan/certify-reassociation
                                         {:acc 'total :init zero :lambda step} dt)]}
                             [input] [] (soac/lambda-form '[total element] [step])))
        facts (-> (soac/facts program)
                  (assoc-in [:values 'loss] (av/tensor {:dtype dt :shape [] :ownership :owned}))
                  (assoc-in [:equations 'ordered-loss] (soac/default-equation-facts)))]
    (soac/make facts (conj (soac/equations program) equation) '[loss])))

(defn- owned-chain-sum [dt extent]
  (append-ordered-sum
   (changed (chain-graph dt extent)
            #(-> % (assoc-in [:values 'a :ownership] :owned)
                   (assoc-in [:values 'b :ownership] :owned))) dt 'b extent))

(deftest ordered-scalar-reductions-use-existing-host-realization
  (doseq [dt [:float :double]]
    (let [array (if (= dt :float) float-array double-array)
          cast (if (= dt :float) float double)
          sum #(reduce (fn [acc x] (cast (+ acc x))) (cast 0.0) %)
          chain (owned-chain-sum dt 3)
          ;; Project the actual reduction, retaining its own ordered input boundary.
          standalone (last (soac/equation-subprograms chain))
          x (array [1.0e20 -1.0e20 3.0])
          q (array [1.0 1.0 1.0]) r (array [0.0 0.0 0.0])]
      (is (= [(sum (vec x))] (run-host standalone [x])))
      (is (= [3.0] (run-host standalone [x])))
      (is (= [3.0] (run-host chain [x q r])))
      (is (= [3.0] (run-host (changed chain #(assoc-in % [:attributes :source-arithmetic]
                                                       numerics/retained-source-arithmetic))
                             [x q r])))
      (is (= [12.0] (run-host chain [(array [1.0 2.0 3.0]) q (array [2.0 2.0 2.0])]))))
    ;; The older two-equation boundary still accepts dynamic and zero extents.
    (let [source (#'sums/source dt)
          standalone (second (soac/equation-subprograms source))
          array (if (= dt :float) float-array double-array)]
      (is (= [0.0] (run-host standalone [(array []) 0])))
      (is (= [14.0] (run-host source [(array [1.0 2.0 3.0]) (array [1.0 1.0 1.0]) 3]))))))

(deftest ordered-scalar-reduction-graph-boundary-remains-closed
  (let [program (owned-chain-sum :double 3)
        standalone (last (soac/equation-subprograms program))]
    (is (= :typed-soac-host-map-boundary
           (reason #(route/materialize-host-program
                     (changed program (fn [f] (assoc-in f [:values 'b :ownership] :borrowed)))))))
    (doseq [bad [(owned-chain-sum :double 'n)
                 (owned-chain-sum :double 0)
                 ;; All operations must contribute to an explicitly requested output.
                 (soac/make (soac/facts program) (soac/equations program) '[a])]]
      (is (= :typed-soac-host-pure-graph-boundary
             (reason #(route/materialize-host-program bad)))))
    (doseq [bad [(changed standalone #(assoc-in % [:attributes :unknown-policy] true))
                 (changed-operation-attributes standalone
                                               #(assoc % :source-arithmetic
                                                       (numerics/blas-source-arithmetic :double)))]]
      (is (= :typed-soac-host-reduce-boundary
             (reason #(route/materialize-host-program bad)))))
    (is (= :typed-soac-host-map-binder-collision
           (reason #(route/materialize-host-program
                     (changed-operation-attributes standalone (fn [a] (assoc a :index 'double)))))))))

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

(defn- regrouped-segmented-program [dt]
  (let [source (segmented-program dt)
        [equation] (soac/equations source)
        operation (nth equation 3)
        attributes (assoc (second operation)
                          :result-map {:groups '[[[i 2] [j 3]]]})
        operation (apply list (assoc (vec operation) 1 attributes))
        equation (apply list (assoc (vec equation) 3 operation))
        facts (assoc-in (soac/facts source) [:values 'C :shape] [6])]
    (soac/make facts [equation] '[C])))

(defn- segmented-snapshot-program
  ([dt] (segmented-snapshot-program dt 'B '[A B]))
  ([dt copied inputs]
   (let [source (segmented-program dt)
         [reduction] (soac/equations source)
         extent (get {'A 4 'B 6} copied)
         copy (map-equation 'snapshot '[W-snapshot] 'snapshot-index extent [copied]
                            '[snapshot-element] '[snapshot-element])
         value (fn [shape ownership]
                 (av/tensor {:dtype dt :shape shape :ownership ownership}))
         facts (-> (soac/facts source)
                   (assoc :inputs inputs :attributes {}
                          :equations {'dense-product (soac/default-equation-facts)
                                      'snapshot (soac/default-equation-facts)})
                   (assoc-in [:values 'A] (value [4] nil))
                   (assoc-in [:values 'B] (value [6] nil))
                   (assoc-in [:values 'W-snapshot] (value [extent] :owned)))]
     (soac/make facts [reduction copy] '[C W-snapshot]))))

(defn- run-projection [projection arguments]
  (apply (eval (list 'fn (:parameters projection) (:body projection))) arguments))

(defn- fbits [^floats values]
  (mapv #(Float/floatToRawIntBits (aget values %)) (range (alength values))))

(defn- dbits [^doubles values]
  (mapv #(Double/doubleToRawLongBits (aget values %)) (range (alength values))))

(defn- map-product-graph [dt inputs]
  (let [product (regrouped-segmented-program dt)
        [reduction] (soac/equations product)
        before (map-equation 'before '[A] 'p 4 '[X] '[xe] '[(raster.numeric/* xe xe)])
        value (fn [extent ownership]
                (av/tensor {:dtype dt :shape [extent] :ownership ownership}))
        facts (-> (soac/facts product)
                  (assoc :inputs inputs :attributes {})
                  (assoc-in [:values 'X] (value 4 nil))
                  (assoc-in [:values 'A] (value 4 :owned))
                  (assoc-in [:values 'B] (value 6 nil))
                  (assoc-in [:equations 'before] (soac/default-equation-facts)))]
    (soac/make facts [before reduction] '[C A])))

;; Independent row/column arithmetic, not another invocation of the materializer.
;; Float explicitly rounds each product and each fold addition, then the store.
(defn- product-reference [dt a b]
  (let [cast (if (= :float dt) float double)]
    (vec (for [i (range 2) j (range 3)]
           (cast (reduce (fn [acc l]
                           (cast (+ acc (cast (* (nth a (+ (* i 2) l))
                                                (nth b (+ (* j 2) l)))))))
                         (cast 0.0) (range 2)))))))

(deftest product-map-ordered-loss-retains-independent-arithmetic
  (doseq [dt [:float :double]]
    (let [product (regrouped-segmented-program dt)
          activation (map-equation 'activation '[squared] 'p 6 '[C] '[c]
                                   '[(raster.numeric/* c c)])
          facts (-> (soac/facts product)
                    (assoc :attributes {})
                    (assoc-in [:values 'squared]
                              (av/tensor {:dtype dt :shape [6] :ownership :owned}))
                    (assoc-in [:equations 'activation] (soac/default-equation-facts)))
          program (append-ordered-sum
                   (soac/make facts (conj (soac/equations product) activation) '[squared])
                   dt 'squared 6)
          array (if (= dt :float) float-array double-array)
          cast (if (= dt :float) float double)]
      (doseq [[as bs] [[[-2.0 0.5 3.0 -4.0] [1.25 -0.5 -3.0 2.0 0.125 4.5]]
                      [[1.0 2.0 3.0 4.0] [-1.0 2.0 0.5 3.0 -2.0 1.0]]]]
        (let [a (array as) b (array bs)
              cs (product-reference dt (vec a) (vec b))
              expected (reduce (fn [acc c] (cast (+ acc (cast (* c c)))))
                               (cast 0.0) cs)]
          (is (= [expected] (run-host program [a b])))
          (is (= (mapv cast as) (vec a)))
          (is (= (mapv cast bs) (vec b))))))))

(deftest product-without-map-does-not-widen-the-mixed-graph-boundary
  (let [product (regrouped-segmented-program :double)
        program (append-ordered-sum
                 (changed product #(assoc % :attributes {})) :double 'C 6)]
    (is (= :typed-soac-host-pure-graph-boundary
           (reason #(route/materialize-host-program program))))))

(deftest ordered-map-product-graphs-retain-independent-host-results-and-owned-edges
  (let [base (map-product-graph :double '[B X])
        [before reduction] (soac/equations base)
        ;; Source order deliberately differs from both lexical equation-ID order
        ;; and declared result order; neither may decide which results allocate.
        second-reduction (apply list (assoc (vec reduction) 1 'a-product 2 '[D]))
        facts (-> (soac/facts base)
                  (assoc-in [:values 'D] (get-in (soac/facts base) [:values 'C]))
                  (assoc-in [:equations 'a-product] (soac/default-equation-facts)))
        program (soac/make facts [before reduction second-reduction] '[D C A])
        admission (#'route/admit-host-ordered-pure-program! program)
        {:keys [parameters body outputs]} (route/materialize-host-program program)
        executable (eval (list 'fn parameters body))
        x (double-array [-2.0 0.5 3.0 -4.0])
        b (double-array [1.25 -0.5 -3.0 2.0 0.125 4.5])
        expected-a (mapv #(* % %) (vec x))
        expected-c (product-reference :double expected-a (vec b))
        [d c a] (executable b x)
        [again-d again-c again-a] (executable b x)]
    (is (= '[dense-product a-product] (:fresh-segmented-reductions admission)))
    (is (= '[D C A] (:output-values admission) outputs))
    (is (= expected-c (vec d) (vec c) (vec again-d) (vec again-c)))
    (is (= expected-a (vec a) (vec again-a)))
    (is (not (identical? c d)))
    (is (every? true? (map #(not (identical? %1 %2)) [d c a] [again-d again-c again-a]))))
  (doseq [dt [:float :double] inputs ['[X B] '[B X]]]
    (let [program (map-product-graph dt inputs)
          {:keys [parameters body] :as projection} (route/materialize-host-program program)
          executable (eval (list 'fn parameters body))
          array (if (= :float dt) float-array double-array)
          cast (if (= :float dt) float double)
          bits (if (= :float dt) fbits dbits)
          observations
          (for [[xs bs] [[[4097.0 -1.25 0.33333334 -0.0009765625]
                          [0.33333334 -0.3 1025.0 0.0009765625 -7.0 11.0]]
                         [[-2.0 0.5 3.0 -4.0] [1.25 -0.5 -3.0 2.0 0.125 4.5]]]]
            (let [x (array xs) b (array bs)
                  x-before (bits x) b-before (bits b)
                  environment {'X x 'B b}
                  arguments (mapv environment (:inputs projection))
                  expected-a (mapv #(cast (* (cast %) (cast %))) xs)
                  expected-c (product-reference dt expected-a (mapv cast bs))
                  [c a] (apply executable arguments)
                  [again-c again-a] (apply executable arguments)]
              (is (= inputs (:inputs projection)))
              (is (= (bits (array expected-c)) (bits c) (bits again-c)))
              (is (= (bits (array expected-a)) (bits a) (bits again-a)))
              (is (= x-before (bits x)))
              (is (= b-before (bits b)))
              (is (not (identical? x a)))
              (is (not (identical? c again-c)))
              (is (not (identical? a again-a)))
              (if (= dt :float) (aset-float a 0 (float 99.0)) (aset-double a 0 99.0))
              (let [[fresh-c fresh-a] (apply executable arguments)]
                (is (= (bits (array expected-c)) (bits fresh-c)))
                (is (= (bits (array expected-a)) (bits fresh-a))))
              {:c c :a a}))]
      ;; Realize both changed invocations of this one compiled projection.
      (is (= 2 (count observations)))
      (let [[first-run second-run] (vec observations)]
        (is (not= (bits (:c first-run)) (bits (:c second-run))))))))

(deftest ordered-map-product-graphs-refuse-unsafe-edges-and-open-policies
  (let [program (map-product-graph :float '[X B])
        equations (vec (soac/equations program))
        facts (soac/facts program)
        dead (map-equation 'dead '[D] 'd 4 '[X] '[xe] '[xe])]
    (doseq [[label expected build]
            [[:unowned-produced-edge :typed-soac-host-pure-graph-boundary
              #(changed program (fn [f] (assoc-in f [:values 'A :ownership] nil)))]
             [:undersized-product-operand :typed-soac-host-fresh-segmented-reduce-boundary
              #(let [before (map-equation 'before '[A] 'p 3 '[X] '[xe] '[(raster.numeric/* xe xe)])
                     f (-> facts (assoc-in [:values 'A :shape] [3])
                           (assoc-in [:values 'X :shape] [3]))]
                 (soac/make f [before (second equations)] '[C A]))]
             [:dead-equation :typed-soac-host-pure-graph-boundary
              #(soac/make (-> facts
                               (assoc-in [:values 'D] (av/tensor {:dtype :float :shape [4] :ownership :owned}))
                               (assoc-in [:equations 'dead] (soac/default-equation-facts)))
                          (conj equations dead) '[C A])]
             [:unclaimed-value :typed-soac-host-pure-graph-boundary
              #(changed program (fn [f] (assoc-in f [:values 'ghost] (av/tensor {:dtype :float :shape [4]}))))]
             [:external-caller-output :typed-soac-output-boundary
              #(soac/make facts equations '[C A X])]
             [:effect :typed-soac-host-map-boundary
              #(changed program (fn [f] (-> f (assoc :effects #{:memory/write})
                                            (assoc-in [:equations 'before :effects] #{:memory/write}))))]
             [:alias :typed-soac-host-map-boundary
              #(changed program (fn [f] (-> f (assoc :aliases {'A 'X})
                                            (assoc-in [:equations 'before :aliases] {'A 'X}))))]
             [:global-source-policy :typed-soac-host-fresh-segmented-reduce-boundary
              #(changed program (fn [f] (assoc f :attributes {:source-arithmetic numerics/retained-source-arithmetic})))]]]
      (is (= expected (reason #(route/materialize-host-program (build)))) (name label)))))

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
        gather (#'indexed/indexed-map :double :long 'n 'm)]
    (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map gather))))
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

(deftest segmented-reduction-primal-owns-its-exact-operand-snapshot
  (doseq [dt [:float :double]]
    (let [standalone (segmented-program dt)
          mixed (segmented-snapshot-program dt)
          swapped (segmented-snapshot-program dt 'B '[B A])
          a ((if (= :float dt) float-array double-array) [4097.0 -1.25 0.03125 8193.0])
          b ((if (= :float dt) float-array double-array)
             [0.1 -0.3 1025.0 0.0009765625 -7.0 11.0])
          bits (if (= :float dt) fbits dbits)
          [expected] (run-host standalone [a b])
          [actual snapshot] (run-host mixed [a b])
          [swapped-actual swapped-snapshot] (run-host swapped [b a])
          [again again-snapshot] (run-host mixed [a b])]
      (is (= (bits expected) (bits actual) (bits swapped-actual) (bits again)))
      (is (= (bits b) (bits snapshot) (bits swapped-snapshot) (bits again-snapshot)))
      (is (not (identical? b snapshot)))
      (is (not (identical? snapshot again-snapshot)))
      (is (not (identical? actual again)))))
  (let [a (float-array [1.0 2.0 3.0 4.0])
        b (float-array [5.0 6.0 7.0 8.0 9.0 10.0])
        [_ a-snapshot] (run-host (segmented-snapshot-program :float 'A '[A B]) [a b])]
    ;; Contract output has six elements; this snapshot has the copied operand's four.
    (is (= 4 (alength ^floats a-snapshot)))
    (is (= (fbits a) (fbits a-snapshot))))
  (let [a (float-array [1.0 2.0 3.0 4.0])
        b (float-array [5.0 6.0 7.0 8.0 9.0 10.0])
        [_ snapshot] (run-host (segmented-snapshot-program :float) [a b])
        before (fbits snapshot)]
    (aset-float b 0 -999.0)
    (is (= before (fbits snapshot)))))

(deftest segmented-reduction-graphs-admit-pure-maps-and-refuse-unsafe-contracts
  (let [program (segmented-snapshot-program :float)
        [reduction copy] (soac/equations program)
        refuse? (fn [candidate]
                  (some? (reason #(route/materialize-host-program candidate))))
        copy-with (fn [arrays body]
                    (map-equation 'snapshot '[W-snapshot] 'snapshot-index 6 arrays
                                  '[snapshot-element] [body]))]
    ;; These are general pure functional graphs, not AD snapshot certificates.
    ;; Both a nonidentity map and an independent earlier map now have ordinary
    ;; source-ordered host meanings; neither grants ownership to caller storage.
    (let [a (float-array [4097.0 -1.25 0.33333334 2.0])
          b (float-array [0.33333334 -0.3 1025.0 0.0009765625 -7.0 11.0])
          expected-c (float-array (product-reference :float (vec a) (vec b)))]
      (doseq [[equations expected-map]
              [[[reduction (copy-with '[B] '(raster.numeric/+ snapshot-element (clojure.core/float 1.0)))]
                (mapv #(float (+ % (float 1.0))) (vec b))]
               [[copy reduction] (vec b)]]]
        (let [[c mapped] (run-host (soac/make (soac/facts program) equations (soac/outputs program)) [a b])]
          (is (= (fbits expected-c) (fbits c)))
          (is (= (fbits (float-array expected-map)) (fbits mapped)))
          (is (not (identical? b mapped))))))
    ;; A produced contraction result is not an external operand snapshot. This malformed shape
    ;; may be rejected by the dialect even before the narrower topology boundary.
    (is (some? (reason (fn []
                         (soac/make (soac/facts program)
                                    [reduction (copy-with '[C] 'snapshot-element)]
                                    (soac/outputs program))))))
    (is (= :typed-soac-input-boundary
           (reason (fn []
                     (route/materialize-host-program
                      (changed program
                               #(-> %
                                    (assoc :inputs '[A B unused])
                                    (assoc-in [:values 'unused]
                                              (av/tensor {:dtype :float :shape [1]})))))))))
    (doseq [policy [(numerics/blas-source-arithmetic :float)
                    numerics/retained-source-arithmetic]]
      (is (refuse? (changed program #(assoc % :attributes {:source-arithmetic policy})))))
    (is (some? (reason (fn []
                         (changed program #(assoc % :effects #{:memory/write}))))))
    (is (= :typed-soac-host-fresh-segmented-reduce-boundary
           (reason (fn []
                     (route/materialize-host-program
                      (changed program #(assoc % :aliases {'W-snapshot 'B})))))))
    (let [destination (av/tensor {:dtype :float :shape [6] :ownership :owned})
          storage [{:destination 'snapshot-storage :access :write :host-return :buffer}]
          facts (-> (soac/facts program)
                    (assoc :effects #{:memory/write}
                           :aliases {'W-snapshot 'snapshot-storage})
                    (assoc-in [:values 'snapshot-storage] destination)
                    (assoc-in [:equations 'snapshot :effects] #{:memory/write})
                    (assoc-in [:equations 'snapshot :aliases]
                              {'W-snapshot 'snapshot-storage})
                    (assoc-in [:equations 'snapshot :attributes :result-storage] storage))
          stored (soac/make facts (soac/equations program) (soac/outputs program))]
      (is (refuse? stored)))))

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

(deftest fresh-segmented-reduction-consumes-a-regrouped-result-layout
  (doseq [dt [:float :double]]
    (let [canonical (segmented-program dt)
          regrouped (regrouped-segmented-program dt)
          array (if (= :float dt) float-array double-array)
          bits (if (= :float dt) fbits dbits)
          a (array [0.1 -0.3 1025.0 0.0009765625])
          b (array [0.1 -0.3 1025.0 0.0009765625 -7.0 11.0])
          a-before (bits a)
          b-before (bits b)
          [expected] (run-host canonical [a b])
          [actual] (run-host regrouped [a b])
          [again] (run-host regrouped [a b])]
      (is (= (bits expected) (bits actual) (bits again)))
      (is (= a-before (bits a)))
      (is (= b-before (bits b)))
      (is (not (identical? actual again)))
      (is (= 6 (java.lang.reflect.Array/getLength actual)))))
  (let [source (segmented-program :float)
        regrouped (regrouped-segmented-program :float)]
    (is (nil? (get-in (soac/operation-parts (first (soac/equations source)))
                      [:attributes :result-map])))
    (is (= {:groups '[[[i 2] [j 3]]]}
           (get-in (soac/operation-parts (first (soac/equations regrouped)))
                   [:attributes :result-map])))))

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
