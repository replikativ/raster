(ns raster.gpu.compatibility-scan-packet-device-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.segop-lower-pass :as lower]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]))

(defn- scheduled-graph [mode dtype]
  (let [original soac/par-form->soac
        source (list 'let* ['before '(identity :before)
                           'result (list (if (= mode :inclusive) 'raster.par/scan
                                             'raster.par/scan-exclusive)
                                         'out 'acc 0.0 'i 'n
                                         (if (= dtype :float) 'float 'double)
                                         '(+ acc (clojure.core/aget input i)))] 'result)
        packet (with-redefs [soac/par-form->soac
                             (fn [& arguments]
                               (when (contains? '#{raster.par/scan raster.par/scan-exclusive}
                                                (first (second arguments)))
                                 (throw (AssertionError. "native scan used legacy inference")))
                               (apply original arguments))]
                 (:form (lower/segop-lower-pass source
                                               {:dtype dtype :array-types {'input dtype 'out dtype}
                                                :scalar-types {'n :long}})))
        scan (last (:equations packet))]
    (is (= 'scan (raster.compiler.ir.soac-dialect/operation-kind
                  (first (raster.compiler.ir.soac-dialect/equations (:algorithm scan))))))
    (emit/generate-scan-kernel-graph (get-in scan [:attributes :kernel-graph]))))

(defn- check-scans! [device]
  (gpu/with-gpu-session [session device]
    (doseq [dtype [:float :double] mode [:inclusive :exclusive]]
      (let [graph (scheduled-graph mode dtype)
            prefix [dtype mode]
            input-key [prefix :input]
            output-key [prefix :output]
            short-key [prefix :short]
            array (if (= dtype :float) float-array double-array)]
        (gpu/alloc! session {input-key [dtype 1025 nil]
                             output-key [dtype 1026 nil]
                             short-key [dtype 1 nil]})
        (doseq [n [0 1 17 1025]]
          (testing [device dtype mode n]
            (try
              (let [handle (gpu/bind-kernel-graph! session [prefix n] graph
                                                  {'input input-key 'out output-key}
                                                  {'n {:type :long :value n}})]
                (try
                  (doseq [replay [0 1]]
                    (let [xs (mapv #(if (zero? replay) 1.0 (* 0.25 (- (mod % 11) 5)))
                                   (range 1025))
                          input (array xs)
                          sums (reductions + 0.0 (take n xs))
                          expected (if (= mode :inclusive) (rest sums) sums)]
                      (gpu/upload! session input-key input)
                      (gpu/upload! session output-key (array (repeat 1026 -999.0)))
                      (gpu/run-kernel-graph! session handle)
                      (let [actual (vec (gpu/download session output-key))
                            written (count expected)]
                        (is (= (vec expected) (mapv double (take written actual))))
                        (is (every? #{-999.0} (map double (drop written actual)))
                            "scan does not write beyond its logical result")
                        (is (= xs (mapv double (gpu/download session input-key)))
                            "inputs remain immutable across replay"))))
                  (finally (gpu/release-kernel-graph! session handle))))
              (catch clojure.lang.ExceptionInfo failure
                (is false (str "valid scan shape refused: " (ex-data failure)))))))
        (doseq [[bindings buffer required]
                [[{'input short-key 'out output-key} 'input 1025]
                 [{'input input-key 'out short-key} 'out
                  (if (= mode :inclusive) 1025 1026)]]]
          (let [retained-keys [:buffers :kernel-graphs :allocations :buffer-owners
                               :prepared :graphs :events]
                before (select-keys @session retained-keys)
                failure (try
                            (let [handle (gpu/bind-kernel-graph! session [prefix :undersized] graph
                                                                bindings {'n {:type :long :value 1025}})]
                              (gpu/release-kernel-graph! session handle)
                              nil)
                            (catch clojure.lang.ExceptionInfo failure failure))]
            (is (= "kernel graph buffer extent exceeds its resident view"
                   (some-> failure .getMessage)))
            (is (= {:graph-buffer buffer :elements required :view-elements 1}
                   (some-> failure ex-data))
                "undersized source or destination must fail before replay")
            (is (= before (select-keys @session retained-keys))
                "rejected binding must not leave private allocations or graph handles")))))))

(deftest complete-scan-packets-execute-on-opencl
  (if @opencl/opencl-available?
    (check-scans! :ocl:0)
    (opencl/opencl-skip! "complete compatibility scan packets")))

(deftest complete-scan-packets-execute-on-level-zero
  (if @ze/gpu-available?
    (check-scans! :ze:0)
    (ze/gpu-skip! "complete compatibility scan packets")))
