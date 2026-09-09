(ns raster.compiler.ir.explicit-accumulator-stage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]))

(defn- source [opts]
  (facts/from-components
   {:out 'out :free-axes '[[i 2]] :contract-axes '[[t 3]]
    :body '(raster.numeric/* (raster.arrays/aget a t) (raster.arrays/aget b t))
    :dtype :float :opts opts}))

(deftest explicit-precision-comes-from-the-canonical-reduction
  (let [original (source {:acc-dtype :int})
        normalized (facts/explicit-accumulator-stage original)]
    (is (= [{:axis 't :extent 3 :dtype :int :init 0}] (:stages normalized)))
    (is (= :float (:out-dtype normalized)))
    (is (= (select-keys (facts/scalar-reduction-view original) [:dtype :combine :neutral])
           (select-keys (facts/scalar-reduction-view normalized) [:dtype :combine :neutral])))
    (is (identical? normalized (facts/explicit-accumulator-stage normalized)))))

(deftest unrelated-reduction-policies-are-not-replaced
  (doseq [original [(source {})
                    (source {:acc-dtype :int :combine '* :init 1})
                    (source {:stages [{:axis 't :extent 3 :dtype :float :init 0.0}]})
                    (facts/from-components
                     {:out 'out :free-axes [] :contract-axes '[[t 3]]
                      :body '(raster.arrays/aget a t) :dtype :float :opts {:acc-dtype :double}})
                    (assoc (source {:acc-dtype :int}) :contract-axes '[[x 2] [t 3]])]]
    (is (identical? original (facts/explicit-accumulator-stage original))))
  (is (thrown? clojure.lang.ExceptionInfo (source {:acc-dtype :int :init 2})))
  (let [byte-storage (assoc (source {}) :dtype :byte)]
    (is (identical? byte-storage (facts/explicit-accumulator-stage byte-storage))
        "byte storage without an accumulator declaration does not synthesize an Int stage")))

(deftest explicit-accumulator-types-the-epilogue-before-output-conversion
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "explicit accumulator epilogue precision")
    (let [a (float-array [16777216.0 1.0 0.0]) out (float-array [Float/NaN])
          expected (float (- (reduce + 0.0 (map double a)) 16777216.0))
          compiled (equation-first/compile #'fixtures/explicit-double-accumulator-epilogue!
                                           {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compiled [a out])
          output-id (some (fn [[id node]] (when (identical? out (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (is (= 1.0 expected))
        (is (= :none (get-in compiled [:stats :fallback])))
        (link/run! executable)
        (is (= [expected] (vec (link/download executable output-id))))
        (finally (link/close! executable))))))
