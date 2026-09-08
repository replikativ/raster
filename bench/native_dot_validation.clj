(ns native-dot-validation
  "Explicit device validation; run on an OpenCL device with packed integer dot support.
   No silent capability skip and no dependency on this device in the ordinary CI suite."
  (:refer-clojure :exclude [run!])
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.ocl-runtime :as ocl])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(defn reference [a b acc]
  (let [lane (fn [word shift]
               (let [x (bit-and 255 (unsigned-bit-shift-right (long word) shift))]
                 (if (< x 128) x (- x 256))))]
    (unchecked-int
     (reduce + (long acc) (for [shift [0 8 16 24]]
                           (* (lane a shift) (lane b shift)))))))

(defn cases []
  (vec (for [a [0 -1 0x7f7f7f7f (unchecked-int 0x80808080)
                 (unchecked-int 0x80ff017f) 0x7f010080]
             b [0 -1 0x7f7f7f7f (unchecked-int 0x80808080)
                 (unchecked-int 0xff017f80)]
             acc [0 -1 1 Integer/MIN_VALUE Integer/MAX_VALUE]]
         [a b acc])))

(defn run-c!
  "Hardware-free portable-helper oracle under C undefined-behavior sanitization.
   Requires a C compiler with UBSan; compiler absence/failure is an error, not a skip."
  ([] (run-c! "cc"))
  ([compiler]
   (let [directory (Files/createTempDirectory "raster-dot-ubsan-" (make-array FileAttribute 0))
         binary (.resolve directory "dot-validation")
         source (str "#include <limits.h>\n_Static_assert(INT_MAX == 2147483647, \"32 bit int required\");\nstatic "
                     (:c-helper-src (intrinsics/descriptor 'dp4a))
                     "\nint main(void) {\nvolatile int cases[][4] = {\n"
                     (str/join ",\n" (for [[a b acc] (cases)]
                                          (str "{" a "," b "," acc "," (reference a b acc) "}")))
                     "};\nfor (unsigned i=0; i<sizeof(cases)/sizeof(cases[0]); ++i) {\n"
                     "if (rstr_dp4a(cases[i][0],cases[i][1],cases[i][2]) != cases[i][3]) return 1;\n"
                     "}\nreturn 0; }\n")]
     (try
       (let [compiled (shell/sh compiler "-x" "c" "-std=c11" "-O2"
                                "-fsanitize=undefined" "-fno-sanitize-recover=undefined"
                                "-o" (str binary) "-" :in source)]
         (when-not (zero? (:exit compiled))
           (throw (ex-info "portable dot sanitizer compilation failed" compiled))))
       (let [result (shell/sh (str binary))]
         (when-not (zero? (:exit result))
           (throw (ex-info "portable dot sanitizer validation failed" result)))
         {:implementation :portable :cases (count (cases)) :passed? true :sanitizer :undefined})
       (finally
         (Files/deleteIfExists binary)
         (Files/deleteIfExists directory))))))

(defn run!
  ([] (run! :native))
  ([implementation]
  (when-not (contains? #{:native :portable} implementation)
    (throw (ex-info "unknown dot validation implementation" {:implementation implementation})))
  (let [kernel-body
        (body/make
         {:id :native-dot-validation
          :parameters [(body/->KernelParameter 'a :scalar :int [] nil nil :a)
                       (body/->KernelParameter 'b :scalar :int [] nil nil :b)
                       (body/->KernelParameter 'acc :scalar :int [] nil nil :acc)
                       (body/->KernelParameter 'out :output :int [1] :global
                                               (layout/row-major [1] :int) :result)]
          :operations [(body/->ScalarCompute
                        (body/value 'dot :int)
                        (body/scalar-expression :dp4a :int ['a 'b 'acc]))
                       (body/->ScalarStore 'out [0] 'dot nil)]
          :launch (launch/spec {:workgroup-size [1] :group-count [1]})
          :provenance {:dialect :validation} :attributes {}})
        module (emit/emit-scalar-module
                "native_dot_edges" kernel-body
                {:target-dialect :opencl-portable
                 :parameter-names {'out "rstr_output"}
                 :target-features (when (= :native implementation)
                                    {:intrinsic-implementations {:dp4a :opencl-packed-dot}})})
        compiled (artifact/make
                  {:kernel-name "native_dot_edges" :target :opencl-c :source (:source module)
                   :abi [(abi/slot 'a :scalar :int :c-name "rstr_a")
                         (abi/slot 'b :scalar :int :c-name "rstr_b")
                         (abi/slot 'acc :scalar :int :c-name "rstr_acc")
                         (abi/slot 'out :output :int :c-name "rstr_output")]
                   :arguments '[a b acc out] :launch (:launch kernel-body)
                   :effects {:kind :native-dot-validation}
                   :attributes {:compilation (:compilation module)}})
        output (ocl/buffer-of-array (int-array [42]) :int)]
    (try
      (ocl/register-kernel! "native_dot_edges" compiled)
      (doseq [[a b acc :as inputs] (cases)]
        (let [expected (reference a b acc)
              _ (ocl/array->buffer! output (int-array [(bit-xor expected 1)]))
              bound (ocl/bind-kernel-call
                     (call/make compiled [{:type :int :value a} {:type :int :value b}
                                          {:type :int :value acc} output]))]
          (try
            (ocl/launch-registered-bound! bound)
            (let [actual (first (ocl/buffer->array output))]
              (when-not (= expected actual)
                (throw (ex-info "integer dot differs from wrapping reference"
                                {:implementation implementation
                                 :inputs inputs :expected expected :actual actual}))))
            (finally (ocl/destroy-prepared! bound)))))
      {:implementation implementation :cases (count (cases)) :passed? true :device (ocl/selected-device-info)
       :compilation (:compilation module)}
      (finally (ocl/free-buffer! output))))))
