(ns raster.compiler.passes.parallel.structured-control-lower-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]))

(defn- loop-program
  ([] (loop-program false))
  ([chained?]
   (let [extent (av/tensor {:dtype :long :shape []})
         scalar (av/tensor {:dtype :float :shape []})
         tensor (av/tensor {:dtype :float :shape '[n]})
         inner-tensor (av/tensor {:dtype :float :shape '[n-in]})
         first-result (if chained? 'u-temporary 'u-next)
         equation
         (list '= 'advance '[u-next]
               (list 'map {:index 'i :extent 'n-in}
                     '[u-in] '[alpha-in iteration]
                     (soac/lambda-form
                      '[u-value alpha-value iteration-value]
                      '[(+ u-value alpha-value (* 0.0 iteration-value))])))
         first-equation (assoc (vec equation) 1 'advance-first 2 [first-result])
         first-equation (apply list first-equation)
         second-equation
         (list '= 'advance-second '[u-next]
               (list 'map {:index 'i :extent 'n-in}
                     '[u-temporary] '[]
                     (soac/lambda-form '[temporary-value] '[(* 2.0 temporary-value)])))
         equations (if chained? [first-equation second-equation] [equation])
         equation-facts (into {}
                              (map (fn [equation]
                                     [(second equation) (soac/default-equation-facts)]))
                              equations)
         body (soac/make
               (soac/default-program-facts
                {:values {'iteration extent 'n-in extent 'alpha-in scalar
                          'u-in inner-tensor 'u-temporary inner-tensor
                          'u-next inner-tensor}
                 :inputs '[iteration n-in alpha-in u-in]
                 :equations equation-facts})
               equations '[u-next])]
     (control/make
      {:id 'time-loop :effects #{} :provenance {:source :test}
       :attributes {:association :sequential}}
      '[iteration steps]
      [{:outer 'n :parameter 'n-in}
       {:outer 'alpha :parameter 'alpha-in}]
      [{:initial 'u0 :parameter 'u-in :result 'u-next :output 'u-final}]
      body
      {'steps extent 'n extent 'alpha scalar 'u0 tensor 'u-final tensor}))))

(deftest structured-control-takes-the-shared-soac-schedule-vertical
  (let [scheduled (lower/schedule (loop-program) {:target-device :cpu:0 :dtype :float})
        body (:body scheduled)
        graph (:graph scheduled)]
    (is (lower/scheduled-loop? scheduled))
    (is (= :segop (:dialect body)))
    (is (parallel-program/parallel-program? body))
    (is (= 1 (count (:nodes graph))))
    (is (= '[u-in] (mapv :id (:inputs graph))))
    (is (= '[u-next] (mapv :id (:outputs graph))))
    (is (= {:kind :host-repetition :association :sequential}
           (:strategy scheduled)))
    (is (= :typed-soac (get-in scheduled [:attributes :body-dialect])))
    (is (= scheduled (lower/validate! scheduled)))))

(deftest structured-loop-body-dataflow-becomes-one-verified-iteration-graph
  (let [scheduled (lower/schedule (loop-program true) {:target-device :cpu:0 :dtype :float})
        graph (:graph scheduled)
        [first-node second-node] (:nodes graph)]
    (is (= 2 (count (:nodes graph))))
    (is (= '[u-temporary] (mapv :id (:temporaries graph))))
    (is (= [(:id first-node)] (:dependencies second-node)))
    (is (= '[u-in] (mapv :id (:inputs graph))))
    (is (= '[u-next] (mapv :id (:outputs graph))))))
