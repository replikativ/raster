(ns raster.compiler.passes.parallel.carried-effect-loop-fixture
  "Scheduled and canonical region fixtures; analyzed-source admission is not claimed."
  (:require [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.segop :as segop]))

(defn scheduled-loop [trips]
  (segop/map->SegMap
   {:id :carried-effect-loop :space (segop/make-seg-space 'i 'rows)
    :dtype :float :inputs #{'x} :outputs #{'words 'totals} :scalars #{'rows}
    :write-conflict :ordered-effects :write-conflicts {'words :unique 'totals :unique}
    :scalar-region
    {:locals [{:id 'seed :dtype :float :init 0.25}]
     :iteration-order :independent
     :effects [{:loop {:index 'k :lower 0 :extent trips
                      :locals [{:id 'loaded :dtype :float :init '(aget x (+ (* i 8) k))}]
                      :effects [{:destination 'words :dtype :float :conflict :unique
                                 :destination-index '(+ (* i 8) k) :predicate true :value 'loaded}]
                      :carry {:parameter 'acc :result 'sum :dtype :float
                              :init 'seed :update (with-meta '(+ acc loaded) {:raster.type/tag 'double})}}}
               {:destination 'totals :dtype :float :conflict :unique
                :destination-index 'i :predicate true :value 'sum}]}}))

(defn scheduled-normalization [trips]
  (let [base (update (scheduled-loop trips) :inputs conj 'words)
        total (get-in base [:scalar-region :effects 1])]
    (assoc-in base [:scalar-region :effects 1]
              {:region
               {:locals [{:id 'inverse :dtype :float
                          :init (with-meta '(/ 1.0 sum) {:raster.type/tag 'double})}]
                :effects [{:loop
                           {:index 'j :lower 0 :extent trips :locals []
                            :effects [{:destination 'words :dtype :float :conflict :unique
                                       :destination-index '(+ (* i 8) j) :predicate true
                                       :value (with-meta '(* (aget words (+ (* i 8) j)) inverse)
                                                {:raster.type/tag 'double})}]}}
                          (assoc total :value 'inverse)]}})))

(defn artifact [operation target & {:keys [scalar-types] :or {scalar-types {'rows :long 'seed :float}}}]
  (emit/generate-scheduled-segmap-kernel
   operation :dtype :float :target-dialect target
   :array-types {'x :float 'words :float 'totals :float} :scalar-types scalar-types))

(defn typed-program [trips]
  (let [words-result [:carry 0] totals-result [:carry 1]
        tensor #(av/tensor {:dtype :float :shape [%]})
        equation
        (list '= :carry [words-result totals-result]
              (list 'effect-map
                    {:index 'i :extent 'rows :dtypes [:float :float]
                     :iteration-order :independent :attributes {:stable-array-captures ['x]}}
                    [] '[x seed rows] '[words totals]
                    (dialect/effect-lambda-form
                     '[source initial row-count packed sums]
                     [(list 'effect-loop {:index 'k :lower 0
                                          :carry {:parameter 'acc :result 'sum :dtype :float}}
                            trips 'initial
                            (list 'lambda '[k acc]
                                  (list 'effect-region
                                        [(dialect/local-value 'loaded :float '(aget source (+ (* i 8) k)))]
                                        [(list 'effect 'packed :unique '(+ (* i 8) k) true 'loaded)]
                                        (with-meta '(+ acc loaded) {:raster.type/tag 'double}))))
                      (list 'effect 'sums :unique 'i true 'sum)])))
        equation-facts (assoc (dialect/default-equation-facts)
                              :effects #{:memory/read :memory/write}
                              :aliases {words-result 'words totals-result 'totals}
                              :attributes {:result-storage
                                           [{:destination 'words :access :write :host-return :effect}
                                            {:destination 'totals :access :write :host-return :effect}]})]
    (dialect/make
     (dialect/default-program-facts
      {:values {'x (tensor 16) 'words (tensor 16) 'totals (tensor 2)
                'rows (av/tensor {:dtype :long :shape []})
                'seed (av/tensor {:dtype :float :shape []})
                words-result (tensor 16) totals-result (tensor 2)}
       :inputs '[x seed rows] :equations {:carry equation-facts}
       :effects #{:memory/read :memory/write}})
     [equation] [])))
