(ns raster.compiler.passes.parallel.source-free-invocation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-map-element-test :as map-fixture]
            [raster.ad.normalized-map-sum-test :as sum-fixture]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.structured-control-route :as route]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- failure-data [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (ex-data error))))

(defn- change-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

(defn- direct-flat-plan [id parameters values]
  (invocation/from-prefix
   {:id id :parameters parameters :parameter-values (select-keys values parameters)
    :bindings [] :binding-values {} :program-values values
    :program-inputs parameters :program-outputs []}))

(defn- pure-map
  ([] (pure-map {}))
  ([{:keys [shape output-shape] :or {shape '[n]}}]
   (let [array (av/tensor {:dtype :double :shape shape})
         output-array (av/tensor {:dtype :double :shape (or output-shape shape)})
         n (av/tensor {:dtype :long :shape []})
         equation (list '= 'map-0 '[y]
                        (list 'map {:index 'i :extent 'n} '[x] []
                              (soac/lambda-form
                               '[element] '[(raster.numeric/* element element)])))]
     (soac/make
      (soac/default-program-facts
       {:values {'x array 'n n 'y output-array}
        :inputs '[x n]
        :equations {'map-0 (soac/default-equation-facts)}})
      [equation] '[y]))))

(deftest source-free-promotion-retains-the-exact-zero-prefix-interface
  (let [typed (pure-map)
        promoted (route/promote-typed-algorithm typed)
        plan (get-in promoted [:attributes :invocation-plan])
        values (:values (soac/facts typed))]
    (is (= :typed-parallel (:dialect promoted)))
    (is (nil? (:source promoted)))
    (is (= '[x n] (:inputs promoted) (mapv :symbol (:parameters plan))))
    (is (= '[y] (:outputs promoted) (:program-outputs plan)))
    (is (empty? (:steps plan)))
    (is (empty? (:storage-bindings plan)))
    (is (= '[x n] (mapv :program-value (:bindings plan))))
    (is (= (select-keys values '[x n])
           (into {} (map (juxt :symbol :value)) (:parameters plan))))
    (is (every? true?
                (map #(identical? (get values (:symbol %)) (:value %)) (:parameters plan))))
    (is (= values (:values promoted)))
    (is (= plan (invocation/validate-against! plan promoted)))
    (is (= :typed-soac (get-in plan [:attributes :source-dialect])))
    (is (= :explicit-typed-algorithm (get-in promoted [:provenance :front-end])))))

(deftest source-free-promotion-materializes-through-the-existing-flat-boundary
  (let [promoted (route/promote-typed-algorithm (pure-map))
        plan (get-in promoted [:attributes :invocation-plan])
        input (double-array [1.0 2.0 3.0])
        materialized (materialization/materialize plan [input 3] nil)
        x-id (:invocation-value (first (:bindings plan)))
        n-id (:invocation-value (second (:bindings plan)))]
    (is (= [3] (get-in materialized [:program-buffers 'x :shape])))
    (is (identical? input (get-in materialized [:program-buffers 'x :source])))
    (is (= {:type :long :value 3} (get-in materialized [:program-scalars 'n])))
    (is (= (get-in materialized [:values x-id])
           (get-in materialized [:program-buffers 'x])))
    (is (= (get-in materialized [:values n-id])
           (get-in materialized [:program-scalars 'n])))
    (is (= :invocation-materialization-arity
           (reason #(materialization/materialize plan [input] nil))))
    (is (= :invocation-materialization-shape-value
           (reason #(materialization/materialize plan [input 3.0] nil))))
    (is (= '[x n] (mapv :symbol (:parameters plan)))
        "buffer-before-dimension order is retained and materializes from the complete scalar set"))
  (testing "the existing materializer executes its canonical value and self-extent forms"
    (let [n (av/tensor {:dtype :long :shape []})
          by-value (av/tensor {:dtype :double :shape '[(value n)]})
          self-sized (av/tensor {:dtype :double :shape '[(extent self-sized)]})
          value-plan (direct-flat-plan :value-dimension '[buffer n]
                                       {'buffer by-value 'n n})
          self-plan (direct-flat-plan :self-extent '[self-sized]
                                      {'self-sized self-sized})]
      (is (= [3] (get-in (materialization/materialize
                           value-plan [(double-array 3) 3] nil)
                         [:program-buffers 'buffer :shape])))
      (is (= [4] (get-in (materialization/materialize
                           self-plan [(double-array 4)] nil)
                         [:program-buffers 'self-sized :shape]))))))

(deftest source-free-promotion-refuses-unsupported-physical-boundaries
  (testing "rank, missing/private and compound dimensions remain outside the flat boundary"
    (is (some? (reason #(route/promote-typed-algorithm (pure-map {:shape '[2 n]})))))
    (doseq [shape '[[private-size] [(value private-size)] [(extent private-buffer)]
                    [(clojure.core/+ n 1)]]]
      (let [typed (change-facts
                   (pure-map)
                   #(assoc-in % [:values 'unsupported]
                              (av/tensor {:dtype :double :shape shape})))]
        (is (= :structured-control-source-free-dimension
               (reason #(route/promote-typed-algorithm typed)))))))
  (testing "present private and wrongly typed public dimensions remain unavailable"
    (let [base (pure-map)
          private-scalar (av/tensor {:dtype :long :shape []})
          private-shaped (change-facts
                          base
                          #(-> %
                               (assoc-in [:values 'private-size] private-scalar)
                               (assoc-in [:values 'private-buffer]
                                         (av/tensor {:dtype :double :shape '[private-size]}))))]
      (is (= :structured-control-source-free-dimension
             (reason #(route/promote-typed-algorithm private-shaped))))
      (is (= :structured-control-source-free-dimension
             (reason #(#'route/source-free-dimension!
                        {'scale (av/tensor {:dtype :double :shape []})}
                        #{} false 'buffer 'scale))))))
  (testing "an input may inspect only its own extent; produced values may use a public buffer"
    (let [base (pure-map)
          other (av/tensor {:dtype :double :shape '[(extent x)]})
          produced (change-facts base #(assoc-in % [:values 'scratch] other))]
      (is (= :structured-control-source-free-dimension
             (reason #(#'route/source-free-dimension!
                        {'x (av/tensor {:dtype :double :shape '[(extent x)]})
                         'other other}
                        #{'x 'other} true 'other '(extent x)))))
      (is (= '(value n)
             (#'route/source-free-dimension!
              {'n (av/tensor {:dtype :long :shape []})} #{} false 'scratch '(value n))))
      (is (some? (route/promote-typed-algorithm produced)))))
  (testing "effects, aliases, storage and native initialization never become invocation authority"
    (let [base (pure-map)]
      (is (= :structured-control-source-free-effects
             (reason #(route/promote-typed-algorithm
                       (change-facts base (fn [facts] (assoc facts :aliases {'y 'x})))))))
      (is (= :structured-control-source-free-effects
             (reason #(route/promote-typed-algorithm
                       (change-facts
                        base
                        (fn [facts]
                          (assoc-in facts [:attributes :native-initialization-providers]
                                    ['x])))))))
      (is (some? (reason #(route/promote-typed-algorithm
                           (change-facts
                            base
                            (fn [facts]
                              (-> facts
                                  (assoc-in [:equations 'map-0 :aliases] {'y 'x})
                                  (assoc-in [:equations 'map-0 :attributes :result-storage]
                                            [{:destination 'x :access :read-write
                                              :host-return :buffer}]))))))))
      (is (some? (reason #(route/promote-typed-algorithm
                           (change-facts
                            base
                            (fn [facts]
                              (-> facts
                                  (assoc :effects #{:memory/write})
                                  (assoc-in [:equations 'map-0 :effects]
                                            #{:memory/write})))))))))))

(deftest source-free-promotion-does-not-relax-the-source-route
  (let [typed (assoc (route/promote-typed-algorithm (pure-map))
                     :source '(let* [x (double-array 2)] x))]
    (is (some? (reason #(route/promote-program typed {}))))))

(defn- canonical-programs []
  (let [map-bundle (normalized/linearize-map-program
                    (#'map-fixture/map-program :double {})
                    (the-ns 'raster.compiler.passes.parallel.source-free-invocation-test))
        sum-bundle (#'sum-fixture/build (#'sum-fixture/source :double) 'xs)]
    (for [[family bundle facet] [[:map map-bundle :map-programs]
                                 [:sum sum-bundle :sum-programs]]
          phase [:primal :linear :transpose]]
      {:family family :phase phase :program (get-in bundle [facet phase])})))

(deftest canonical-derivatives-use-the-common-source-free-schedule-and-emitters
  (doseq [{:keys [family phase] typed :program} (canonical-programs)]
    (let [promoted (route/promote-typed-algorithm typed)
          scheduled (route/schedule-program promoted {:target-device :ocl:0 :dtype :double})]
      (is (= :typed-parallel (:dialect promoted)))
      (is (= :scheduled-parallel (:dialect scheduled)))
      (doseq [target [:opencl-intel :opencl-portable :cuda :hip]]
        (if (= :linear phase)
          ;; Canonical linear programs currently retain reverse/grad-acc. Promotion and scheduling
          ;; are valid; the common ScalarIR-to-KernelBody construction does not yet own that
          ;; operation, independently of the selected C-family target.
          (let [failure (failure-data
                         #(c-family/emit-program
                           scheduled {:target-dialect target :dtype :double}))]
            (is (= (if (contains? #{:cuda :hip} target)
                     :kernel-graph-target-lowering-missing
                     :segmap-kernel-body-declined)
                   (:reason failure))
                (str family " " target " linear emission reason"))
            (when (contains? #{:cuda :hip} target)
              (is (= :none (:fallback failure))))
            (is (= :scalar-expression
                   (:missing-rule (or (:kernel-body-decline failure) failure)))
                (str family " " target " missing shared scalar rule"))
            (is (= 'raster.ad.reverse/grad-acc
                   (first (:expression (or (:kernel-body-decline failure) failure))))
                (str family " " target " retained canonical operation")))
          (let [{:keys [program kernels]}
                (c-family/emit-program scheduled {:target-dialect target :dtype :double})]
            (is (seq kernels))
            (is (some? program))))))))

