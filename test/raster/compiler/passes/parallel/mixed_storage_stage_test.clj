(ns raster.compiler.passes.parallel.mixed-storage-stage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.compiler.backend.gpu.staged-contraction-fixtures :as packed]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.staged-scalar-body :as staged]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]))

(deftest raw-mixed-storage-arithmetic-needs-retained-width
  (let [source (-> (packed/packed-facts 1 1 1 4)
                   (assoc-in [:stages 1 :dtype] :float)
                   (assoc-in [:opts :stages 1 :dtype] :float))
        failure (try (staged/analyze! source) nil
                     (catch clojure.lang.ExceptionInfo e (ex-data e)))]
    (is (= :scalar-source-type (:missing-rule failure)))
    (let [typed (update source :body vary-meta assoc :raster.type/tag 'long)
          operations (tree-seq coll? seq (:operations (:body (staged/lower typed))))
          products (filter #(and (map? %) (= :* (:op %))) operations)]
      (is (= {:long 1 :float 2} (frequencies (map :result-type products)))
          "one Long byte product precedes the two Float scale products"))))

(deftest homogeneous-contractions-keep-existing-schedule-selection
  (with-redefs [facts/single-axis-accumulator-stage
                (fn [& _] (throw (ex-info "homogeneous contraction diverted to scalar stage" {})))]
    (is (some? (frontend/form->program
                '(let* [effect (raster.par/contract out [[i 2] [j 3]] [[k 4]]
                                 (raster.numeric/* (raster.arrays/aget a (+ (* i 4) k))
                                                   (raster.arrays/aget b (+ (* k 3) j))))]
                   out)
                {:dtype :float :array-types '{a :float b :float out :float}})))))

(deftest byte-storage-does-not-change-float-accumulation
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "mixed-storage canonical accumulator")
    (let [a (byte-array (concat (repeat 1024 -128) [1 1]))
          b (aclone a)
          terms (map #(clojure.core/* (long %1) (long %2)) a b)
          expected (reduce #(float (+ %1 (float %2))) (float 0) terms)
          widened (float (reduce + 0 terms))
          out (float-array [Float/NaN])
          compiled (equation-first/compile #'fixtures/byte-products-float-accumulation!
                                           {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compiled [a b out])
          output-id (some (fn [[id node]] (when (identical? out (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (is (= 16777216.0 expected))
        (is (= 16777218.0 widened))
        (is (= :none (get-in compiled [:stats :fallback])))
        (is (= {:kernel-body 1} (get-in compiled [:stats :emission :emission-routes])))
        (is (not-any? #(re-find #"rstr_dp4a" (:source %)) (:kernels compiled)))
        (link/run! executable)
        (is (= [expected] (vec (link/download executable output-id))))
        (finally (link/close! executable))))))
