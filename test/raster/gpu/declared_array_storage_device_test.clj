(ns raster.gpu.declared-array-storage-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.compiler.compatibility-map-packet-test :as map-packet]
            [raster.compiler.backend.gpu.segmap-retirement-fixture :as retirement]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.runtime-backend :as backend]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.value :as value]))

(defn- run-retired-segmap-invariants [target]
  (let [runtime (backend/runtime-namespace target)
        resolve-runtime #(requiring-resolve (symbol (str runtime) %))
        arena ((resolve-runtime "make-kernel-arena!"))]
    (try
      (with-bindings {(resolve-runtime "*current-arena*") arena}
        (doseq [kind [:secondary-output :integer-offset]]
          (let [{:keys [source emitted]} (retirement/emit kind target true)
                original (eval (list 'fn '[a b d n offset] source))
                native (eval (list 'fn '[a b d n offset] (:form emitted)))]
            (is (= 1 (count (:kernels emitted))))
            (is (zero? (get-in emitted [:stats :fallback] 0)))
            (#'pipeline/register-gpu-kernels! (:kernels emitted) target)
            (#'pipeline/register-gpu-dispatches! (:dispatches emitted) target)
            (doseq [n [0 1 3] seed [1.0 -2.0]]
              (let [a (float-array (map #(+ seed %) (range 4)))
                    b (float-array [2.0 -4.0 8.0])
                    d (float-array [0.5 0.25 -0.5])
                    expected (if (= kind :secondary-output)
                               [(mapv #(float (* %1 %2)) (take n d) (take n b))
                                (mapv #(float (* %1 %2)) (take n d) (take n a))]
                               (mapv float (take n (drop 1 a))))
                    source-result (original a b d (long n) (int 1))
                    actual (native a b d (long n) (int 1))
                    arrays #(if (= kind :secondary-output) % [%])]
                (is (= (if (= kind :secondary-output) (mapv vec source-result) (vec source-result))
                       expected
                       (if (= kind :secondary-output) (mapv vec actual) (vec actual)))
                    (str target " " kind " n=" n " seed=" seed))
                (is (every? #(= (class (float-array 0)) (class %)) (arrays actual)))
                (doseq [index (range (count (arrays actual)))]
                  (is (not (identical? a (nth (arrays actual) index)))))
                (when (= kind :secondary-output)
                  (is (not (identical? (first actual) (second actual)))))
                (is (= (mapv float (map #(+ seed %) (range 4))) (vec a))
                    "input captures remain unchanged"))))))
      (finally ((resolve-runtime "close-kernel-arena!") arena)))))

(deftest retired-source-map-invariants-match-both-local-backends
  (if @opencl/opencl-available?
    (run-retired-segmap-invariants :ocl:0)
    (opencl/opencl-skip! "canonical secondary-output and integer-capture maps on OpenCL"))
  (if @gp/gpu-available?
    (run-retired-segmap-invariants :ze:0)
    (gp/gpu-skip! "canonical secondary-output and integer-capture maps on Level Zero")))

(defn- run-compatibility-map-storage-case [target]
  ;; Execute the supplied compatibility packet, not a fresh whole-program typed route.
  ;; This is the same arena-scoped staged-form boundary as integral-reduction-device-test.
  (let [runtime (backend/runtime-namespace target)
        resolve-runtime #(requiring-resolve (symbol (str runtime) %))
        arena ((resolve-runtime "make-kernel-arena!"))]
    (try
      (let [facts ((resolve-runtime "execution-device-info"))]
        (when-not (contains? (:storage-types facts) :double)
          (throw (ex-info "compatibility map storage test requires selected-device Double support"
                          {:reason :declared-scalar-test-capability :target target :facts facts}))))
      (with-bindings {(resolve-runtime "*current-arena*") arena}
        (doseq [[kind source] map-packet/float-map-sources]
          (let [options (assoc map-packet/options :target-device target)
                packet (:form (segop-lower/segop-lower-pass source options))
                emitted (opencl-pass/opencl-pass
                         packet :device-id target :dtype :double :min-elements 0
                         :array-types (:array-types options)
                         :scalar-types (:scalar-types options))
                native (eval (list 'fn ['a 'out 'n] (:form emitted)))
                original (eval (list 'fn ['a 'out 'n] source))
                buffer-slots (for [artifact (:kernels emitted)
                                   slot (:abi artifact)
                                   :when (not= :scalar (:kind slot))] slot)]
            (is (= :float (get-in packet [:values [:binding 'left] :dtype])))
            (when (= kind :legacy-consumer)
              (is (nil? (:algorithm (last (:equations packet))))
                  "map! remains an honest legacy consumer"))
            (is (some #(= :float (:dtype %)) buffer-slots)
                "the physical ABI contains Float storage under the Double default")
            (when (not= kind :float-result)
              (is (some #(= :double (:dtype %)) buffer-slots))
              (is (some #(and (= :input (:kind %)) (= :float (:dtype %)))
                        (:abi (last (:kernels emitted))))
                  "the downstream consumer's native pointer ABI reads the retained Float array"))
            (is (zero? (get-in emitted [:stats :fallback] 0)))
            ;; Maps retain direct artifact invocation markers beside any graph dispatches.
            ;; Match the production pass-backend admission order for this mixed result.
            (#'pipeline/register-gpu-kernels! (:kernels emitted) target)
            (#'pipeline/register-gpu-dispatches! (:dispatches emitted) target)
            (doseq [n [0 1 3]
                    values [[1.00000001 -3.5 16777217.0]
                            [-1.00000001 16777219.0 0.125]]]
              ;; Nonempty caller storage also exercises zero active extent and unwritten tails.
              (let [input (double-array values)
                    out (double-array (repeat (max 1 n) -77.0))
                    source-out (aclone out)
                    expected (mapv #(double (float %)) (take n values))
                    source-result (original input source-out (long n))
                    actual (native input out (long n))]
                (is (= (class source-result) (class actual)) (str target " " kind " n=" n))
                (is (= (class (if (= kind :float-result) (float-array 0) (double-array 0)))
                       (class actual)))
                (is (= (vec source-result) (vec actual)))
                (is (= expected (vec (take n actual)))
                    "independent explicit Float round-trip oracle")
                (when (= kind :legacy-consumer)
                  (is (identical? out actual))
                  (when (zero? n) (is (= [-77.0] (vec actual))))))))))
      (finally ((resolve-runtime "close-kernel-arena!") arena)))))

(deftest compatibility-map-packets-retain-float-storage-on-both-backends
  (if @opencl/opencl-fp64-available?
    (run-compatibility-map-storage-case :ocl:0)
    (opencl/opencl-skip! "Double-to-Float compatibility map storage on OpenCL"))
  (if @gp/gpu-available?
    (run-compatibility-map-storage-case :ze:0)
    (gp/gpu-skip! "Double-to-Float compatibility map storage on Level Zero")))

(defn- run-case [target]
  (doseq [compiler [nil :equation-first]]
    (let [artifact (compiled/compile
                    #'storage/mixed-storage!
                    [(float-array 4) (double-array 4) (double-array 4) 4]
                    (cond-> (merge storage/policy
                                   {:target target :dtype :double :outputs '[out]})
                      compiler (assoc :compiler compiler)))]
      (try
        (doseq [weights [[1.25 -2.5 0.125 8.0] [16.0 32.0 -4.0 0.5]]]
          (let [x (float-array weights)
                state (double-array [0.0000000001 1.0 4.0 -2.0])
                expected (double-array 4)]
            (storage/mixed-storage! x state expected 4)
            (is (= (vec expected)
                   (vec (value/->host (:out (artifact {:weights x :state state}))))))))
        (finally (compiled/close! artifact))))))

(defn- run-fold-case [target]
  (doseq [compiler [nil :equation-first]]
    (let [initial (float-array [16777216.0 1.0 1.0 8.0 0.25 0.5 19.0])
          expected (aclone initial)
          artifact (compiled/compile
                    #'storage/mixed-fold-storage! [initial 2 3]
                    (cond-> (merge storage/policy
                                   {:target target :dtype :double :donate '[storage]})
                      compiler (assoc :compiler compiler)))]
      (try
        (dotimes [_ 2]
          (storage/mixed-fold-storage! expected 2 3)
          (is (= (vec expected)
                 (vec (value/->host (:storage' (artifact {})))))
              "Double recurrence precedes each Float store, preserving source order and the tail"))
        (finally (compiled/close! artifact))))))

(defn- run-scale-case [target]
  (let [probe (requiring-resolve
               (case target :ze:0 'raster.gpu.ze-runtime/execution-device-info
                            :ocl:0 'raster.gpu.ocl-runtime/execution-device-info))
        facts (probe)]
    (when-not (contains? (:storage-types facts) :double)
      (throw (ex-info "declared Double scalar test requires affirmative selected-device support"
                      {:reason :declared-scalar-test-capability :target target :facts facts}))))
  (doseq [dtype [:float :double]
          compiler [nil :equation-first]]
   (let [alpha (+ 1.0 (Math/scalb 1.0 (int -24)))
        x (float-array [1.5 -1.5 0.0])
        expected (mapv #(float (* alpha (double %))) x)
        artifact (compiled/compile
                  #'storage/mixed-scale [alpha x]
                  (cond-> (merge storage/policy {:target target :dtype dtype})
                    compiler (assoc :compiler compiler)))
        output-keys (mapv :key (:out-tree artifact))]
    (try
      (is (= 1 (count output-keys)))
      (when compiler (is (= [:result] output-keys)))
      (is (= expected (vec (storage/mixed-scale alpha x))))
      (dotimes [_ 2]
        (let [outputs (artifact {})
              result (value/->host (get outputs (first output-keys)))]
          (is (= (set output-keys) (set (keys outputs))))
          (is (= (class x) (class result)))
          (is (= expected (vec result)))))
      (finally (compiled/close! artifact))))))

(deftest mixed-scale-public-jvm-device-storage-boundary
  (if @gp/gpu-available?
    (run-scale-case :ze:0)
    (gp/gpu-skip! "mixed coefficient scale on Level Zero"))
  (if @opencl/opencl-available?
    (run-scale-case :ocl:0)
    (opencl/opencl-skip! "mixed coefficient scale on OpenCL")))

(defn- run-completed-conversion-case [target]
  (doseq [n [0 3 2049]]
    (let [x (double-array (take n (cycle [16777216.0 1.0 1.0])))
          expected (storage/double-reduction-float-result x)
          artifact (compiled/compile
                    #'storage/double-reduction-float-result [x]
                    (merge storage/policy
                           {:compiler :equation-first :target target :dtype :double}))]
      (try
        (dotimes [_ 2]
          (let [outputs (artifact {})
                result (value/->host (:result outputs))]
            (is (= #{:result} (set (keys outputs))))
            (is (= (class (float-array 0)) (class result)))
            (is (= [expected] (vec result)))))
        (finally (compiled/close! artifact))))))

(deftest completed-reduction-conversion-matches-jvm-on-both-backends
  (if @gp/gpu-available?
    (run-completed-conversion-case :ze:0)
    (gp/gpu-skip! "completed reduction conversion on Level Zero"))
  (if @opencl/opencl-available?
    (run-completed-conversion-case :ocl:0)
    (opencl/opencl-skip! "completed reduction conversion on OpenCL")))

(defn- run-scalar-gradient-case [target]
  (doseq [n [0 3 2049]]
    (let [x (double-array (take n (cycle [1.5 -2.0 0.25])))
          coefficient (float 2.0)
          expected (storage/mixed-scale-energy-gradient coefficient x)
          artifact (compiled/compile
                    #'storage/mixed-scale-energy-gradient [coefficient x]
                    (merge storage/policy
                           {:compiler :equation-first :target target :dtype :double}))]
      (try
        (dotimes [_ 2]
          ;; Select by the public result's tangent representation, never generated ABI
          ;; names. Other returned arrays are Double intermediates, not scalar gradients.
          (let [results (map value/->host (vals (artifact {})))
                gradients (filter #(instance? (class (float-array 0)) %) results)]
            (is (= 1 (count gradients)))
            (is (= [expected] (vec (first gradients))))))
        (finally (compiled/close! artifact))))))

(deftest projected-scalar-gradient-matches-jvm-on-both-backends
  (if @gp/gpu-available?
    (run-scalar-gradient-case :ze:0)
    (gp/gpu-skip! "projected scalar gradient on Level Zero"))
  (if @opencl/opencl-available?
    (run-scalar-gradient-case :ocl:0)
    (opencl/opencl-skip! "projected scalar gradient on OpenCL")))

(deftest level-zero-mixed-storage-public-compilers-match-jvm
  (if @gp/gpu-available?
    (do (run-case :ze:0) (run-fold-case :ze:0))
    (gp/gpu-skip! "mixed array storage on Level Zero")))

(deftest opencl-mixed-storage-public-compilers-match-jvm
  (if @opencl/opencl-available?
    (do (run-case :ocl:0) (run-fold-case :ocl:0))
    (opencl/opencl-skip! "mixed array storage on OpenCL")))
