(ns raster.compiler.passes.parallel.product-reduction-body-device-test
  "Differential execution of the candidate and retained product schedule on OpenCL."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.segop-opencl :as reference]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.passes.parallel.product-reduction-body :as product]
            [raster.compiler.passes.parallel.product-reduction-body-test :as fixtures]
            [raster.gpu.device-probe :as probe]))

(defn- argmax [xs]
  (first
   (reduce (fn [[best-index best] [i x]]
             (if (or (and (Float/isNaN x) (not (Float/isNaN best)))
                     (and (not (Float/isNaN best))
                          (or (> x best) (and (== x best) (< i best-index)))))
               [i x] [best-index best]))
           [Integer/MAX_VALUE Float/NEGATIVE_INFINITY]
           (map-indexed vector xs))))

(deftest candidate-product-tree-agrees-with-reference-and-host
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "candidate product KernelBody numerical comparison")
    (let [runtime (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve runtime 'register-kernel!)
          buffer-of-array (ns-resolve runtime 'buffer-of-array)
          bind-call (ns-resolve runtime 'bind-kernel-call)
          execute! (ns-resolve runtime 'launch-registered-bound!)
          read! (ns-resolve runtime 'buffer->array)
          free! (ns-resolve runtime 'free-buffer!)
          segred (fixtures/argmax-segred)
          old (reference/generate-product-reduction-kernel segred
                :scalar-types (:scalar-types fixtures/options)
                :array-types (:array-types fixtures/options))
          retained-options (dissoc fixtures/options :element-binding-types :combine-binding-types)
          new (target/emit-artifact "candidate_product_argmax"
                                   (product/schedule (fixtures/typed-argmax-segred) retained-options)
                                   :opencl-portable)
          local (target/emit-artifact "candidate_product_local_address"
                                     (product/schedule (fixtures/typed-local-address-segred)
                                                       retained-options) :opencl-portable)]
      (doseq [artifact [old new local]] (register! (:kernel-name artifact) artifact))
      (doseq [[nrows width] (cons [0 1] (map #(vector 5 %) [0 1 7 32 33 65]))]
        (let [rows (vec (take nrows [(vec (repeat width (float 2)))
                    (mapv #(float (mod % 7)) (range width))
                    (mapv #(if (odd? %) Float/NaN (float %)) (range width))
                    (vec (repeat width Float/NEGATIVE_INFINITY))
                    (vec (repeat width Float/POSITIVE_INFINITY))]))
              input (buffer-of-array (float-array (if (or (zero? width) (zero? nrows))
                                                   [0] (mapcat identity rows))) :float)]
          (try
            (let [results
                  (mapv (fn [artifact]
                          (let [output (buffer-of-array
                                        (int-array (repeat (max 1 nrows) -77)) :int)
                                bindings {'values input 'indices output
                                          'nrows {:type :int :value (count rows)}
                                          'width {:type :int :value width}}]
                            (try
                              (if (zero? nrows)
                                ;; Both routes reject zero-group launches. The planner must elide
                                ;; a zero-row operation rather than submit an invalid device call.
                                (is (thrown-with-msg? clojure.lang.ExceptionInfo
                                                     #"group-count dimension must be a positive integer"
                                                     (call/make artifact
                                                                (mapv bindings (:arguments artifact)))))
                                (execute! (bind-call (call/make artifact
                                                               (mapv bindings (:arguments artifact))))))
                              (vec (read! output))
                              (finally (free! output))))) [old new local])]
              (is (apply = (cons (if (zero? nrows) [-77] (mapv argmax rows)) results))
                  (str "rows " nrows ", width " width)))
            (finally (free! input))))))))
