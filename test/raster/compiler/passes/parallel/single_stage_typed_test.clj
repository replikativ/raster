(ns raster.compiler.passes.parallel.single-stage-typed-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]))

(defn- single-stage-form [width stages]
  (list 'let*
        ['effect (list 'raster.par/contract 'out [['i 2]] [['t width]]
                       '(raster.numeric/* (raster.arrays/aget a t) (raster.arrays/aget b t))
                       :out-dtype :float :stages stages)]
        'out))

(deftest single-stage-admission-retains-shape-and-overflow-proofs
  (let [stage (fn [width] [{:axis 't :extent width :dtype :int :init 0}])
        options {:dtype :float :array-types '{a :byte b :byte out :float}}
        compile #(frontend/form->program % options)
        program (compile (single-stage-form 3 (stage 3)))]
    (is (some? (soac/validate! program)))
    (is (nil? (compile (single-stage-form 131072 (stage 131072))))
        "an unproved integral prefix is a capability decline, not malformed syntax")
    (doseq [form [(single-stage-form 3 (stage 2))
                  (single-stage-form 3 [{:axis 't :extent 3 :dtype :int :init 1}])]]
      (is (thrown? clojure.lang.ExceptionInfo (compile form))))))

(deftest decoded-single-stage-preserves-integer-accumulation-and-float-storage
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "single-stage typed byte contraction")
    (doseq [[function width decode]
            [[#'fixtures/decoded-byte-single-stage! 3 #(- (long %) 7)]
             [#'fixtures/packed-byte-single-stage! 4 long]]]
     (let [a (byte-array (take (* 2 width) [-128 127 0 11 -13 5 127 -128]))
          b (byte-array (take (* 2 width) [127 -128 9 0 11 -4 -128 127]))
          expected (mapv (fn [i]
                           (float (reduce + 0
                                          (for [t (range width) :let [k (+ (* i width) t)]]
                                            (* (decode (aget a k)) (long (aget b k)))))))
                         (range 2))
          out (float-array [Float/NaN Float/NaN])
          compiled (equation-first/compile function
                                           {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compiled [a b out])
          output-id (some (fn [[id node]] (when (identical? out (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (is (= :none (get-in compiled [:stats :fallback])))
        (is (= {:kernel-body 1} (get-in compiled [:stats :emission :emission-routes])))
        (is (= (= width 4)
               (boolean (some #(re-find #"rstr_dp4a" (:source %)) (:kernels compiled))))
            "only the proved exact-product case uses the packed fragment")
        (link/run! executable)
        (is (= expected (vec (link/download executable output-id))))
        (finally (link/close! executable)))))))
