(ns raster.compiler.backend.gpu.segmented-contract-device-test
  "W1 step 2c: ON-DEVICE numeric validation of the naive segmented-contraction emitter.
   Emits with LITERAL dims (no int scalar params → matches invoke-registered-kernel's arg
   order: inputs, output, trailing count), compiles SPIR-V, launches on the Arc, and
   compares to an independent CPU reference. Gated on a real GPU (skips cleanly otherwise).
   This is the golden correctness gate for the emit path — contention-insensitive."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.backend.gpu.segop-opencl :as sco]
            [raster.gpu.runtime-backend :as backend]
            [raster.gpu.device-probe :as opencl]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- ref-matmul [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (loop [l 0 acc 0.0]
                (if (< l k)
                  (recur (inc l) (+ acc (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))))
                  acc)))))
    C))

(defn- approx= [a b] (< (Math/abs (- (double a) (double b))) 1.0e-9))

(defn- repeated-dense-map-form [same-coordinate?]
  (list 'raster.par/contract 'C '[[i 2] [j 5]] []
        (list 'clojure.core/+ '(clojure.core/aget A i)
              (list 'clojure.core/aget 'A (if same-coordinate? 'i 'j)))))

(defn- check-repeated-dense-map! [device]
  (let [runtime (backend/runtime-namespace device)
        runtime-var #(requiring-resolve (symbol (str runtime) %))
        arena ((runtime-var "make-kernel-arena!"))]
    (try
      (with-bindings {(runtime-var "*current-arena*") arena}
        (doseq [[same-coordinate? transform-kind output-dtype]
                [[false nil :double] [true nil :double]
                 [false :add-constant :double] [false :float-checkpoint :double]
                 [false :float-output :float]]]
          (let [transform (case transform-kind
                            :add-constant {:acc 'acc :expr '(clojure.core/+ acc 3.0)
                                           :operands [] :scalars [] :dtype :double}
                            :float-checkpoint {:acc 'acc
                                               :expr '(clojure.core/double (clojure.core/float acc))
                                               :operands [] :scalars [] :dtype :double}
                            :float-output {:acc 'acc
                                           :expr '(clojure.core/float (clojure.core/+ acc 3.0))
                                           :operands [] :scalars [] :dtype :float}
                            nil)
                form (cond-> (repeated-dense-map-form same-coordinate?)
                       transform (concat [:epilogue transform])
                       (= :float output-dtype) (concat [:out-dtype :float]))
                form (apply list form)
                ;; An accepted epilogue must never escape to a source-shaped leaf.
                routed (with-redefs [sco/generate-segmap-nd-kernel
                                     (fn [& _] (throw (ex-info "source fallback reentered" {})))]
                         (route/route-contraction form :dtype :double))
                artifact (:artifact routed)
                required (if same-coordinate? 2 5)
                allocate-output #(if (= :float output-dtype) (float-array %) (double-array %))
                result-slot (first (filter #(= :result (:role %)) (:abi artifact)))]
            (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
            (is (= output-dtype (:out-dtype routed)))
            (is (= output-dtype (:dtype result-slot)))
            (is (= output-dtype (:kernel-dtype result-slot)))
            (is (= output-dtype (:dtype (first (filter #(= 'C (:id %))
                                                       (:parameters (:kernel-body routed)))))))
            (is (= [required]
                   (:shape (first (filter #(= 'A (:id %))
                                          (:parameters (:kernel-body routed)))))))
            ((runtime-var "register-kernel!") (:kernel-name artifact) artifact)
            ;; The ten-element output leaves a partial workgroup. Reuse the same registered
            ;; artifact with changed inputs; no shape inflation or handwritten native binder.
            (doseq [values (map #(take required %)
                                (if (contains? #{:float-checkpoint :float-output} transform-kind)
                                  [[16777217.0 0.0 -16777217.0 1.0 -1.0]
                                   [-16777217.0 0.0 16777217.0 -1.0 1.0]]
                                  [[1.25 -3.5 7.0 8.0 0.125]
                                   [-8.0 2.0 0.5 -1.0 16.0]]))]
              (let [input (double-array values)
                    before (vec input)
                    output (allocate-output 10)
                    expected (vec (for [i (range 2) j (range 5)]
                                    (let [element (+ (nth values i)
                                                     (nth values (if same-coordinate? i j)))]
                                      (case transform-kind
                                        :add-constant (+ element 3.0)
                                        :float-checkpoint (double (float element))
                                        :float-output (float (+ element 3.0))
                                        element))))]
                ((runtime-var "invoke-registered-map-void-kernel")
                 (:kernel-name artifact) [input output] [] 10)
                (is (= expected (vec output)))
                (is (= (class (allocate-output 0)) (class output)))
                (when (= :float-checkpoint transform-kind)
                  (is (= (if (pos? (first values)) 16777216.0 -16777216.0)
                         (aget ^doubles output 1))
                      "the large completed result must pass through an observable Float checkpoint"))
                (is (= before (vec input)))))
            ;; Required capacity must be rejected before staging/native contact, not padded
            ;; into a different mathematical operation or left to an out-of-bounds device read.
            (try
              ((runtime-var "invoke-registered-map-void-kernel")
               (:kernel-name artifact)
               [(double-array (dec required)) (allocate-output 10)] [] 10)
              (is false "known undersized input must fail at retained capacity admission")
              (catch clojure.lang.ExceptionInfo error
                (is (= :kernel-body-buffer-capacity (:reason (ex-data error))))
                (is (= required (:required-elements (ex-data error))))
                (is (= (dec required) (:buffer-elements (ex-data error)))))))))
      (finally ((runtime-var "close-kernel-arena!") arena)))))

(deftest repeated-dense-map-reads-match-independent-coordinates-on-both-backends
  (if @opencl/opencl-available?
    (check-repeated-dense-map! :ocl:0)
    (opencl/opencl-skip! "zero-contract repeated dense reads"))
  (if @gpu?
    (check-repeated-dense-map! :ze:0)
    (println "[skip] zero-contract repeated dense reads: no Level Zero device available")))

(deftest segmented-contraction-matches-cpu-on-device
  (if-not @gpu?
    (println "[skip] segmented-contraction-device: no GPU device available")
    (let [ze (find-ns 'raster.gpu.ze-runtime)
          register! (ns-resolve ze 'register-kernel!)
          invoke!   (ns-resolve ze 'invoke-registered-kernel)
          m 3 k 4 n 2
          A (double-array (map double (range (* m k))))
          B (double-array (map #(* 0.5 (double %)) (range (* k n))))
          C (double-array (* m n))
          ;; LITERAL dims → the emitter inlines 3/2/4, no int scalar params.
          form (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
                     (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
                           (list 'aget 'B (list '+ (list '* 'l n) 'j))))
          sr  (cl/contract-form->segred form)
          {:keys [kernel-name scalar-params] :as kernel}
          (sco/generate-segmented-reduce-kernel sr 'C)]
      (testing "literal-dim contraction has no scalar params (dims inlined)"
        (is (empty? scalar-params)))
      ;; Register the emitter-owned ABI as well as its source. The runtime deliberately refuses
      ;; a hand-reconstructed map signature because dropping one slot here is a silent miscompile.
      (register! kernel-name (assoc kernel :workgroup-size 256))
      (invoke! kernel-name [A B] C [] (* m n))     ; args: A, B, out, int _nseg=(m*n)
      (testing "GPU segmented-contraction result == CPU reference"
        (let [ref (ref-matmul A B m k n)]
          (is (every? true? (map approx= (vec C) (vec ref)))
              (str "GPU " (vec C) " vs CPU " (vec ref))))))))
