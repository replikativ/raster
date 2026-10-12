(ns raster.compiler.passes.parallel.typed-scalar-packet-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as graph]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.compiler.passes.parallel.typed-scalar-packet-test :as packet]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl-probe]
            [raster.dl.gpu-grad-parity :as level-zero]))

(defn- check-scalar! [target source & [options]]
  (let [runtime (case target :ocl:0 'raster.gpu.ocl-runtime :ze:0 'raster.gpu.ze-runtime)
        register! (requiring-resolve (symbol (str runtime) "register-kernel-dispatch!"))
        scheduled (:form (lower/segop-lower-pass source
                                                (assoc (merge packet/options options) :target-device target)))
        emitted (opencl/opencl-pass scheduled :device-id target :dtype :float
                                   :compile-spirv? (= :ze:0 target) :min-elements 0)
        execute (eval (list 'fn '[xs] (:form emitted)))]
    (doseq [dispatch (:dispatches emitted)] (register! dispatch))
    (is (seq (:dispatches emitted)))
    (is (zero? (get-in emitted [:stats :fallback] 0)))
    (doseq [values [[-1 2 3.5] [7 -4 0.5] [] (repeat 1025 1)]]
      (let [input (float-array values)
            before (vec input)
            expected (Math/sqrt (reduce + 0.0 (map #(* (double %) (double %)) values)))]
        (is (< (Math/abs (- expected (double (execute input)))) 1.0e-6))
        (is (= before (vec input)))))))

(def ^:private known-shape-source
  '(let* [n (clojure.core/long (clojure.core/alength xs))
          sum (raster.par/reduce acc (float 0.0) i n
                                (+ acc (* (float (aget xs i)) (float (aget xs i)))))]
     sum))

(defn- check-backing-length! [target]
  (let [options (assoc packet/options :target-device target
                       :values {'xs (av/tensor {:dtype :float :shape [32]})})
        typed (frontend/form->program known-shape-source options)
        scheduled (:form (lower/segop-lower-pass (route/program-envelope typed) options))
        equation (last (:equations scheduled))
        algorithm (:algorithm equation)
        {:keys [body graph]} (graph/make-for-equation scheduled equation)
        emitted (emit/generate-kernel-graph graph
                                            :target-device target
                                            :scalar-types {'n :long}
                                            :scheduled-equation-algorithm algorithm
                                            :scheduled-equation-body body)]
    (is (= [32] (get-in scheduled [:values 'xs :shape])))
    (is (= 'n (:elements (first (:inputs emitted)))))
    (gpu/with-gpu-session [session target]
      (doseq [values [[-1 2 3.5] [7 -4 0.5] [] (repeat 1025 1)]]
        (let [input (float-array values)
              input-key (keyword (str (gensym "input-")))
              output-key (keyword (str (gensym "output-")))
              n (count values)
              expected (reduce + 0.0 (map #(* (double %) (double %)) values))]
          (gpu/alloc! session {input-key [:float n input] output-key [:float 1 nil]})
          (is (thrown? clojure.lang.ExceptionInfo
                       (gpu/bind-kernel-graph! session :oversized-backing-bound emitted
                                              {'xs input-key 'sum output-key}
                                              {'n {:type :long :value (inc n)}})))
          (when (pos? n)
            (let [short-view (gpu/buffer-view session input-key {:shape [(dec n)]})]
              (is (thrown? clojure.lang.ExceptionInfo
                           (gpu/bind-kernel-graph! session :short-backing-view emitted
                                                  {'xs short-view 'sum output-key}
                                                  {'n {:type :long :value n}})))))
          (let [handle (gpu/bind-kernel-graph! session :backing-length emitted
                                             {'xs input-key 'sum output-key}
                                             {'n {:type :long :value n}})]
            (try
              (gpu/run-kernel-graph! session handle)
              (is (< (Math/abs (- expected (double (first (gpu/download session output-key))))) 1.0e-6))
              (is (= (vec input) (vec (gpu/download session input-key))))
              (finally (gpu/release-kernel-graph! session handle)))))))))

(deftest opencl-host-scalar-consumer-uses-imported-reduction-graph
  (if @opencl-probe/opencl-available?
    (do (doseq [source [packet/source packet/nested-source]] (check-scalar! :ocl:0 source))
        (check-scalar! :ocl:0 packet/causal-source
                       {:values {'xs (av/tensor {:dtype :float :shape [32]})}})
        (check-backing-length! :ocl:0))
    (opencl-probe/opencl-skip! "imported scalar reduction result identity")))

(deftest level-zero-host-scalar-consumer-uses-imported-reduction-graph
  (if @level-zero/gpu-available?
    (do (doseq [source [packet/source packet/nested-source]] (check-scalar! :ze:0 source))
        (check-scalar! :ze:0 packet/causal-source
                       {:values {'xs (av/tensor {:dtype :float :shape [32]})}})
        (check-backing-length! :ze:0))
    (level-zero/gpu-skip! "imported scalar reduction result identity")))
