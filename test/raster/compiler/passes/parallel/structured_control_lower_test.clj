(ns raster.compiler.passes.parallel.structured-control-lower-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-loop-call :as loop-call]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]))

(defn- loop-program
  ([] (loop-program false))
  ([chained?]
   (let [extent (av/tensor {:dtype :int :shape []})
         trip-index (av/tensor {:dtype :long :shape []})
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
                {:values {'iteration trip-index 'n-in extent 'alpha-in scalar
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
      {'steps trip-index 'n extent 'alpha scalar 'u0 tensor 'u-final tensor}))))

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
    (is (= scheduled (lower/validate! scheduled)))
    (testing "the scheduled graph is re-derived from exact SegOp dataflow"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"storage, uses, or dependencies differ"
           (lower/validate!
            (assoc-in scheduled [:graph :inputs 0 :elements] 'different-extent)))))))

(deftest structured-loop-body-dataflow-becomes-one-verified-iteration-graph
  (let [scheduled (lower/schedule (loop-program true) {:target-device :cpu:0 :dtype :float})
        graph (:graph scheduled)
        [first-node second-node] (:nodes graph)]
    (is (= 2 (count (:nodes graph))))
    (is (= '[u-temporary] (mapv :id (:temporaries graph))))
    (is (= [(:id first-node)] (:dependencies second-node)))
    (is (= '[u-in] (mapv :id (:inputs graph))))
    (is (= '[u-next] (mapv :id (:outputs graph))))
    (is (= '[iteration n-in alpha-in] (mapv :id (:scalars graph)))
        "public scalars retain the typed body input order")
    (let [emitted (opencl/generate-kernel-graph
                   graph :scalar-types {'alpha-in :float 'iteration :long})]
      (is (emitted-loop/emitted-loop? (emitted-loop/make scheduled emitted)))
      (is (every? artifact/kernel-artifact? (map :operation (:nodes emitted))))
      (is (= '[u-in u-next iteration n-in alpha-in] (:arguments emitted)))
      (is (= :opencl-c (get-in emitted [:provenance :target-dialect])))
      (is (= (mapv :operation (:nodes graph))
             (mapv #(get-in % [:operation :provenance :scheduled-operation])
                   (:nodes emitted))))
      (is (every? #(re-find #"__kernel void graph_segmap" (:source %))
                  (map :operation (:nodes emitted))))
      (is (some #(re-find #"long iteration" (:source %))
                (map :operation (:nodes emitted))))
      (let [call (loop-call/make
                  scheduled emitted
                  {'u0 :initial-buffer 'u-final :output-buffer}
                  {'steps {:type :long :value 3}
                   'n {:type :int :value 64}
                   'alpha {:type :float :value 0.25}}
                  {'u-final :scratch-buffer})]
        (is (= {'u-final :output-buffer} (:outputs call)))
        (is (= {'u-in :initial-buffer 'u-next :output-buffer}
               (:buffers (loop-call/iteration-binding call 0))))
        (is (= {'u-in :output-buffer 'u-next :scratch-buffer}
               (:buffers (loop-call/iteration-binding call 1))))
        (is (= {'u-in :scratch-buffer 'u-next :output-buffer}
               (:buffers (loop-call/iteration-binding call 2))))
        (is (= {:type :long :value 2}
               (get-in (loop-call/iteration-binding call 2)
                       [:scalar-values 'iteration])))
        (testing "a copied SegOp ID cannot hide a changed operation certificate"
          (let [tampered (assoc-in emitted
                                   [:nodes 0 :operation :provenance
                                    :scheduled-operation :grid :block-size]
                                   128)]
            (is (thrown-with-msg?
                 clojure.lang.ExceptionInfo #"operation certificate"
                 (loop-call/validate! (assoc call :graph tampered))))))
        (testing "target emission cannot change buffers or graph effects"
          (is (thrown-with-msg?
               clojure.lang.ExceptionInfo #"scheduled loop dataflow"
               (loop-call/validate!
                (assoc call :graph
                       (assoc-in emitted [:inputs 0 :elements] 'different-extent)))))
          (is (thrown-with-msg?
               clojure.lang.ExceptionInfo #"scheduled loop dataflow"
               (loop-call/validate!
                (assoc call :graph (assoc emitted :effects {:semantic #{:io}}))))))))))

(deftest zero-trip-loop-returns-the-initial-logical-value-without-scratch
  (let [scheduled (lower/schedule (loop-program) {:target-device :cpu:0 :dtype :float})
        emitted (opencl/generate-kernel-graph
                 (:graph scheduled) :scalar-types {'alpha-in :float 'iteration :long})
        call (loop-call/make
              scheduled emitted
              {'u0 :initial-buffer 'u-final :unused-output-buffer}
              {'steps {:type :long :value 0}
               'n {:type :int :value 64}
               'alpha {:type :float :value 0.25}}
              {})]
    (is (= 0 (:trip-count call)))
    (is (= {'u-final :initial-buffer} (:outputs call)))))

(deftest dotimes-derived-loop-clamps-a-negative-runtime-bound-to-zero
  (let [program (loop-program)
        program (control/make
                 (assoc-in (control/facts program)
                           [:attributes :trip-count-semantics]
                           :clamp-nonnegative)
                 (control/loop-index program)
                 (control/invariants program)
                 (control/carried program)
                 (control/body program)
                 (control/outer-values program))
        scheduled (lower/schedule program {:target-device :cpu:0 :dtype :float})
        emitted (opencl/generate-kernel-graph
                 (:graph scheduled) :scalar-types {'alpha-in :float 'iteration :long})
        call (loop-call/make
              scheduled emitted
              {'u0 :initial-buffer 'u-final :unused-output-buffer}
              {'steps {:type :long :value -3}
               'n {:type :int :value 64}
               'alpha {:type :float :value 0.25}}
              {})]
    (is (zero? (:trip-count call)))
    (is (= {'u-final :initial-buffer} (:outputs call)))))
