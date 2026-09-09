(ns raster.compiler.backend.gpu.java-round-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.device-probe :as probe]))

(deftest java-round-expands-before-target-emission
  (doseq [[input output] [[:float :int] [:double :long]]]
    (let [body (fixtures/java-round-body input output 1)
          computes (filter :expression (:operations body))]
      (is (= [:floor :- :ge :+ :select :cast] (mapv #(get-in % [:expression :op]) computes)))
      (is (= {:rounding :toward-zero :overflow :saturate}
             (get-in (last computes) [:expression :options])))
      (is (= output (get-in (last computes) [:result :type])))
      (doseq [target [:opencl-portable :cuda :hip]]
        (let [source (emit/emit-scalar-kernel "java_round_test" body {:target-dialect target})]
          (is (not (str/includes? source "= round(")))
          (when (and (= output :long) (not= target :opencl-portable))
            (is (str/includes? source "(-9223372036854775807LL - 1LL)")))
          (is (str/includes? source (if (= target :opencl-portable) "_sat_rtz(" "isnan("))))))))

(defn- samples [type]
  (let [fp (if (= type :float) unchecked-float double)
        down (if (= type :float) #(Math/nextDown (float %)) #(Math/nextDown (double %)))
        up (if (= type :float) #(Math/nextUp (float %)) #(Math/nextUp (double %)))
        limit (if (= type :float) 2147483648.0 9223372036854775808.0)
        random (java.util.Random. 9281)]
    (mapv fp (concat [Double/NaN Double/NEGATIVE_INFINITY Double/POSITIVE_INFINITY
                      -0.0 0.0 Double/MIN_VALUE (- Double/MIN_VALUE)
                      Float/MIN_VALUE (- Float/MIN_VALUE)]
                    (mapcat (fn [x] [(down x) x (up x)])
                            [-2.5 -1.5 -0.5 0.5 1.5 2.5 (- limit) limit
                             (if (= type :float) 8388608.0 4503599627370496.0)])
                    (repeatedly 128 #(if (= type :float)
                                       (Float/intBitsToFloat (.nextInt random))
                                       (Double/longBitsToDouble (.nextLong random))))))))

(deftest java-round-matches-jvm-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "typed Java round overloads and numerical edge cases")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve ocl 'register-kernel!)
          buffer-of-array (ns-resolve ocl 'buffer-of-array)
          make-buffer (ns-resolve ocl 'make-buffer)
          bind-call (ns-resolve ocl 'bind-kernel-call)
          launch! (ns-resolve ocl 'launch-registered-bound!)
          read! (ns-resolve ocl 'buffer->array)
          free! (ns-resolve ocl 'free-buffer!)]
      (doseq [[input output] [[:float :int] [:double :long]]]
        (let [values (samples input)
              expected (mapv (if (= input :float) #(Math/round (unchecked-float %)) #(Math/round (double %))) values)
              body (fixtures/java-round-body input output (count values))
              name (str "java_round_" (name input))
              compiled (artifact/make
                        {:kernel-name name :target :opencl-c
                         :source (emit/emit-scalar-kernel name body {:target-dialect :opencl-portable})
                         :abi (body-abi/project-contracts
                               [(abi/slot 'x :input input :c-name "rstr_x" :role :input)
                                (abi/slot 'y :output output :c-name "rstr_y" :role :result)] body)
                         :arguments '[x y] :launch (:launch body) :effects {:kind :map}
                         :provenance {:kernel-body (:id body)}})
              x (buffer-of-array ((if (= input :float) float-array double-array) values) input)
              y (make-buffer (count values) output)]
          (try
            (register! name compiled)
            (launch! (bind-call (call/make compiled [x y])))
            (is (= expected (vec (read! y))) (str input " Java round boundary oracle"))
            (finally (free! y) (free! x))))))))
