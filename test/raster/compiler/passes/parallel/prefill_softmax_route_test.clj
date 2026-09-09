(ns raster.compiler.passes.parallel.prefill-softmax-route-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.dl.attention :as attention]
            [raster.gpu.device-probe :as probe]))

(def ^:private compiled
  (delay (pipeline/show-pipeline #'attention/attn-prefill-softmax!
                                 :target-device :ocl:0 :dtype :float)))

(deftest prefill-softmax-enters-the-typed-ordered-region-vertical
  (let [p @compiled
        kernel (first (:kernels p))
        equations (mapcat #(dialect/equations (:algorithm %))
                          (get-in p [:soac-fused :equations]))]
    (is (= :typed-soac (get-in p [:soac-fused :dialect])))
    (is (= 1 (count (:kernels p))))
    (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
    (is (= '[sc nrows _n_bound] (mapv :name (:abi kernel))))
    (is (= [:float :long :long] (mapv :dtype (:abi kernel))))
    (is (= [:sequential]
           (mapv #(get-in (dialect/operation-parts %) [:attributes :iteration-order])
                 (filter #(= 'effect-map (dialect/operation-kind %)) equations))))
    (is (= [1] (get-in kernel [:launch :workgroup-size]))
        "source admission must not silently assert the stronger cross-loop ownership proof")))

(deftest prefill-softmax-keeps-row-values-and-output-tails-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "typed prefill softmax")
    (let [runtime (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve runtime 'register-kernel!)
          buffer-of-array (ns-resolve runtime 'buffer-of-array)
          bind-call (ns-resolve runtime 'bind-kernel-call)
          launch! (ns-resolve runtime 'launch-registered-bound!)
          read! (ns-resolve runtime 'buffer->array)
          free! (ns-resolve runtime 'free-buffer!)
          kernel (first (:kernels @compiled))]
      (register! (:kernel-name kernel) kernel)
      (doseq [width [0 1 3 8]]
        (let [heads 2 rows (* width heads) n (* rows width)
              input (vec (mapcat (fn [row]
                                  (map #(float (+ (* (- row 3) 1000) (* 0.5 %))) (range width)))
                                (range rows)))
              expected (vec (mapcat (fn [row]
                                     (let [m (apply max row)
                                           values (map #(Math/exp (- (double %) m)) row)
                                           total (reduce + values)]
                                       (map #(/ % total) values)))
                                   (if (pos? width) (partition width input) [])))
              scores (buffer-of-array (float-array (concat input [-77 -77])) :float)]
          (try
            (launch! (bind-call (call/make kernel [scores {:type :long :value width}
                                                   {:type :long :value rows}])))
            (let [actual (vec (read! scores))]
              (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) 1.0e-6)
                                     expected (take n actual))))
              (is (= [-77.0 -77.0] (subvec actual n))))
            (finally (free! scores))))))))
