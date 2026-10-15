(ns raster.compiler.passes.parallel.host-operation-admission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-element-test :as maps]
            [raster.ad.normalized-map-sum-test :as sums]
            [raster.ad.normalized-indexed-map-test :as indexed]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.numeric]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- changed [program f]
  (soac/make (f (soac/facts program)) (soac/equations program) (soac/outputs program)))

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
