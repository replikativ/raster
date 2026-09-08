(ns raster.compiler.backend.gpu.kernel-body-shift-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.device-probe :as probe]))

(deftest word-shifts-have-explicit-width-and-sign-spelling
  (doseq [type [:int :long] target [:opencl-portable :cuda :hip]]
    (let [source (emit/emit-scalar-kernel "word_shifts" (fixtures/word-shifts-body type 1)
                                          {:target-dialect target})]
      (is (str/includes? source (str " & " (if (= :int type) 31 63))))
      (is (str/includes? source " < 0) ? ~(~"))
      (is (str/includes? source (case target
                                 :opencl-portable (if (= :int type) "uint" "ulong")
                                 (if (= :int type) "unsigned int" "unsigned long long"))))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not defined for its operand dtype"
                        (fixtures/word-shifts-body :byte 1)))
  (doseq [op [:shl :shr :ushr]]
    (is (not (intrinsics/accepts-scalar-dtype? op :byte)))))

(defn- reference [type x n]
  (let [n (bit-and n (if (= :int type) 31 63))
        narrow (if (= :int type) unchecked-int identity)
        unsigned-x (if (= :int type) (bit-and x 0xffffffff) x)]
    (mapv narrow [(bit-shift-left x n)
                  (bit-shift-right x n)
                  (unsigned-bit-shift-right unsigned-x n)])))

(deftest word-shifts-match-width-specific-reference-on-opencl
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "typed Int/Long shifts")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          resolve-op #(ns-resolve ocl %)
          upload (resolve-op 'buffer-of-array)
          download (resolve-op 'buffer->array)
          free! (resolve-op 'free-buffer!)]
      (doseq [type [:int :long]]
        (let [minimum (if (= :int type) Integer/MIN_VALUE Long/MIN_VALUE)
              maximum (if (= :int type) Integer/MAX_VALUE Long/MAX_VALUE)
              pairs (vec (for [x [minimum -255 -1 0 1 255 maximum]
                               n [minimum maximum -1 0 1 24 31 32 33 63 64 65]] [x n]))
              width (count pairs)
              array-of (if (= :int type) int-array long-array)
              kernel-body (fixtures/word-shifts-body type width)
              kernel-name (str "word_shifts_" (name type))
              compiled (artifact/make
                        {:kernel-name kernel-name :target :opencl-c
                         :source (emit/emit-scalar-kernel kernel-name kernel-body
                                                         {:target-dialect :opencl-portable
                                                          :parameter-names {'out "rstr_output"}})
                         :abi [(abi/slot 'x :input type :c-name "rstr_x" :role :input)
                               (abi/slot 'counts :input type :c-name "rstr_counts" :role :input)
                               (abi/slot 'out :output type :c-name "rstr_output" :role :result)]
                         :arguments '[x counts out] :launch (:launch kernel-body)
                         :effects {:kind :word-shifts} :provenance {:dialect :test}})
              x (upload (array-of (map first pairs)) type)
              counts (upload (array-of (map second pairs)) type)
              out (upload (array-of (repeat (* 3 width) -555)) type)]
          (try
            ((resolve-op 'register-kernel!) kernel-name compiled)
            ((resolve-op 'launch-registered-bound!)
             ((resolve-op 'bind-kernel-call) (call/make compiled [x counts out])))
            (is (= (vec (mapcat (fn [[x n]] (reference type x n)) pairs))
                   (vec (download out))) (name type))
            (finally (free! out) (free! counts) (free! x))))))))
