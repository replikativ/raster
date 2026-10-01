(ns raster.compiler.public-tree-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(deftm project-tree!
  [model :- (HMap {:buffers (HVec [(Array float) (Array int)]) :offset Float})
   out :- (Array float) n :- Long] :- (Array float)
  (let [positions (nth (:buffers model) 0)
        labels (nth (:buffers model) 1)
        offset (:offset model)]
    (par/map-void! i n
                   (aset out i (+ offset (aget positions i) (float (aget labels i))))))
  out)

(deftm anonymous-collision!
  [{:keys [values]} :- (HMap {:values (Array float)})
   arg0 :- Long out :- (Array float)] :- (Array float)
  (par/map-void! i arg0 (aset out i (aget values i)))
  out)

(deftest anonymous-labels-are-stable-and-capture-free
  (let [metadata (meta #'anonymous-collision!)]
    (is (= '[arg0_ arg0 out] (:raster.params/public-args metadata)))
    (is (not= (first (:raster.params/public-args metadata))
              (first (:raster.params/original-args metadata))))
    (is (= [1.0 2.0]
           (vec (anonymous-collision! {:values (float-array [1 2])} 2 (float-array 2)))))))

(defn- input [offset]
  {:buffers [(float-array (range offset (+ offset 17))) (int-array (map #(mod % 3) (range 17)))]
   :offset (float 2)})

(defn- oracle [model]
  (vec (project-tree! model (float-array 17) 17)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest nested-tree-retains-declared-paths-and-types
  (let [c (equation/compile #'project-tree! {:target :cuda:0 :dtype :float})
        p (get-in c [:semantic :attributes :invocation-plan])
        projection (get-in p [:attributes :parameter-projection])
        model (input 0) out (float-array 17)
        args (materialization/parameter-arguments p [model out 17])]
    (is (= '[model out n] (:public-parameters projection)))
    (is (= [[:buffers 0] [:buffers 1] [:offset] nil nil]
           (mapv :path (:physical-parameters projection))))
    (is (identical? (first (:buffers model)) (first args)))
    (is (identical? (second (:buffers model)) (second args)))
    (is (= (float 2) (nth args 2)))
    (is (= :none (get-in c [:stats :fallback])))
    (doseq [bad [(dissoc model :offset) (assoc model :extra 0)
                 (assoc model :buffers [(first (:buffers model))])]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (materialization/parameter-arguments p [bad out 17]))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (materialization/parameter-arguments
                  p [(assoc model :buffers (vec (repeat 2 (first (:buffers model))))) out 17])))
    (doseq [bad [(assoc-in p [:attributes :parameter-projection :physical-parameters 0 :path] [:offset])
                 (assoc-in p [:attributes :parameter-projection :trees 'model]
                           '(HMap {:buffers (HVec [(Array double) (Array int)]) :offset Float}))]]
      (is (= :invocation-parameter-projection (reason #(invocation/validate! bad)))))
    (let [identity #(#'compiled/source-specialization-identity #'project-tree! :float)
          before (get-in (identity) [:semantic :parameter-projection])
          original-meta (meta #'project-tree!)]
      (try
        (alter-meta! #'project-tree! assoc :raster.params/public-args '[other out n])
        (is (not= before (get-in (identity) [:semantic :parameter-projection]))
            "the logical contract participates in the existing compiler cache identity")
        (finally (reset-meta! #'project-tree! original-meta))))))

(defn- run-device [target]
  (let [model (input 0)
        prepared (compiled/lower #'project-tree! [model (float-array 17) 17]
                                 {:target target :dtype :float :compiler :equation-first})
        program (compiled/instantiate! prepared)]
    (try
      (is (= #{[:model :buffers 0] [:model :buffers 1] :out}
             (set (map :key (:in-tree prepared)))))
      (is (= 0 (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
      (is (= (oracle model) (vec (value/->host (:result (program {}))))))
      (let [changed (input 17) result (:result (program {:model changed}))]
        (is (= (oracle changed) (vec (value/->host result))))
        (is (= :compiled-aggregate-scalar-change
               (reason #(program {:model (assoc changed :offset (float 3))}))))
        (is (= (oracle changed) (vec (value/->host result)))))
      (let [constant (compiled/instantiate!
                      (compiled/lower #'project-tree! [model (float-array 17) 17]
                                      {:target target :dtype :float :compiler :equation-first
                                       :constants '[model]}))]
        (try
          (is (= :compiled-aggregate-input-role (reason #(constant {:model (input 17)}))))
          (finally (compiled/close! constant))))
      (is (= :compiled-aggregate-output
             (reason #(compiled/lower #'project-tree! [model (float-array 17) 17]
                                      {:target target :dtype :float :compiler :equation-first
                                       :donate '[model]}))))
      (finally (compiled/close! program)))))

(deftest nested-tree-on-opencl
  (if @opencl/opencl-available? (run-device :ocl:0)
      (opencl/opencl-skip! "nested-tree-opencl")))

(deftest nested-tree-on-level-zero
  (if @ze/gpu-available? (run-device :ze:0)
      (ze/gpu-skip! "nested-tree-level-zero")))
