(ns raster.compiler.passes.parallel.typed-contraction-matrix-device-test
  "The optimized matrix route is exercised from the typed contraction equation through the
   backend-neutral KernelExecutable binder.  This is the device guard that permits the old
   Level Zero GEMM-specific binding surface to disappear without losing its numerical oracle."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.target :as gpu-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.fixtures.contractions :as contractions]
            [raster.compiler.fixtures.attention-projection :as attention-projection]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.passes.parallel.contract-route :as contract-route]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.dl.nn :as nn]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.value :as value]
            [raster.perf.production-canary :as canary]))

(defn- sequential-fp32-projection
  "Independent ordered oracle: two-round multiply/add or single-round Math/fma."
  ([layout left right m k n]
   (sequential-fp32-projection layout left right m k n :separate))
  ([layout ^floats left ^floats right m k n multiply-add]
  (float-array
    (for [i (range m) j (range n)]
      (loop [p 0 acc (float 0.0)]
        (if (= p k)
          acc
          (let [li (if (= layout :tn) (+ (* p m) i) (+ (* i k) p))
                ri (if (= layout :nt) (+ (* j k) p) (+ (* p n) j))
                term (float (* (double (aget left li)) (double (aget right ri))))]
            (recur (inc p)
                   (if (= :fused multiply-add)
                     (Math/fma (aget left li) (aget right ri) (float acc))
                     (float (+ (double acc) (double term))))))))))))

(deftest public-register-product-realizations-match-local-rounding-oracles
  (doseq [[device available? skip!] [[:ze:0 gpu-probe/gpu-available? gpu-probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "explicit FP32 FMA projection on " device))
      (doseq [layout [:nn :nt] policy [:decomposed :fused :decomposed]]
        (let [m 2 n 3 k 2
              a (float-array [-1.0 1.0000001192092896 -1.0 1.0000001192092896])
              b (float-array (if (= layout :nn)
                               [1.0 1.0 1.0 0.9999998807907104 0.9999998807907104 0.9999998807907104]
                               [1.0 0.9999998807907104 1.0 0.9999998807907104 1.0 0.9999998807907104]))
              source (if (= layout :nn) #'contractions/projected-nn #'contractions/projected-nt)
              expected (vec (sequential-fp32-projection layout a b m k n policy))
              opposite (vec (sequential-fp32-projection layout a b m k n
                             (if (= policy :fused) :decomposed :fused)))
              prepared (compiled/lower source [a b m k n]
                         {:compiler :equation-first :target device :dtype :float
                          :schedule {:typed-contraction {:strategy :register-tiled
                                                         :multiply-add policy}}})
              live (compiled/instantiate! prepared)]
          (try
            (is (not= expected opposite) "the oracle distinguishes one-round FMA from two rounds")
            (doseq [changed? [false true]]
              (let [next-a (if changed?
                             (float-array [-1.0 1.000000238418579 -1.0 1.000000238418579])
                             a)
                    reference (sequential-fp32-projection layout next-a b m k n policy)
                    actual (value/->host (:result (live (if changed? {:a next-a} {}))))
                    bits #(mapv (fn [v] (Float/floatToRawIntBits (float v))) %)]
                (is (= (bits reference) (bits actual))
                    (str device " " layout " " policy " changed=" changed?))))
            (finally (compiled/close! live))))))))

(deftest public-portable-blas-projections-retain-sequential-fp32-evaluation
  (doseq [[device available? skip!] [[:ze:0 gpu-probe/gpu-available? gpu-probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "same-operand FP32 projection on " device))
      (doseq [[layout source] [[:nn #'contractions/projected-nn]
                              [:nt #'contractions/projected-nt]
                              [:tn #'contractions/projected-tn]]
              [input-kind k] [[:smooth 3] [:smooth 17] [:smooth 640] [:cancellation 3]]]
        (let [m 2 n 3
              left (float-array
                     (if (= input-kind :cancellation)
                       (map #(nth [1.0e8 1.0 -1.0e8]
                                  (if (= layout :tn) (quot % m) (mod % k)))
                            (range (* m k)))
                       (map #(Math/sin (* 0.73 %)) (range (* m k)))))
              right (float-array
                      (if (= input-kind :cancellation)
                        (repeat (* k n) 1.0)
                        (map #(Math/cos (* 0.51 %)) (range (* k n)))))
              expected (sequential-fp32-projection layout left right m k n)
              prepared (compiled/lower source [left right m k n]
                         {:compiler :equation-first :target device :dtype :float
                          :schedule {:precision :f32-scalar}})
              live (compiled/instantiate! prepared)]
          (try
            (let [actual (value/->host (:result (live {})))]
              (is (instance? (Class/forName "[F") actual))
              (is (java.util.Arrays/equals ^floats expected ^floats actual)
                  (str device " " layout " " input-kind " " [m k n]
                       "; CPU BLAS association is deliberately not this oracle")))
            (finally (compiled/close! live))))))))

(deftest fixed-register-tile-replays-through-the-public-equation-api
  (doseq [[device available? skip!] [[:ze:0 gpu-probe/gpu-available? gpu-probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "equation-first fixed register tile on " device))
      (let [left (float-array (map #(/ (- % 7) 4.0) (range 15)))
            right (float-array (map #(/ (- % 10) 8.0) (range 21)))
            expected (vec (contractions/fixed-matmul left right))
            prepared (compiled/lower
                      #'contractions/fixed-matmul [left right]
                      {:compiler :equation-first :target device :dtype :float
                       :schedule {:typed-contraction {:strategy :register-tiled}}})
            live (compiled/instantiate! prepared)]
        (try
          (is (= [:fixed]
                 (mapv #(get-in % [:executable :selection]) (compiled/execution-info live))))
          (dotimes [_ 2]
            (is (= expected (vec (value/->host (:result (live {}))))) (str device)))
          (finally (compiled/close! live)))))))

(deftest certified-contraction-dispatch-replays-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 gpu-probe/gpu-available? gpu-probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "certified contraction dispatch on " device))
      (let [m 65 n 67 k 17
            left (float-array (map #(/ (- (mod % 17) 8) 4.0) (range (* m k))))
            right (float-array (map #(/ (- (mod % 13) 6) 8.0) (range (* k n))))
            arguments [left right m n k]
            expected (vec (apply contractions/dynamic-matmul arguments))
            prepared (compiled/lower
                      #'contractions/dynamic-matmul arguments
                      {:compiler :equation-first :target device :dtype :float
                       :schedule {:typed-contraction {:strategy :dispatch-register-tiled}}})
            live (compiled/instantiate! prepared)]
        (try
          (dotimes [_ 2]
            (is (= expected (vec (value/->host (:result (live {})))))
                (str device " certified ragged register dispatch")))
          (finally (compiled/close! live)))))))

(deftest public-mixed-attention-and-projection-matches-the-jvm-on-local-backends
  (let [source-args (attention-projection/arguments)
        expected (vec (apply attention-projection/attention-projection source-args))
        device-args (attention-projection/fp32-arguments source-args)]
    (doseq [[device available? skip!] [[:ze:0 gpu-probe/gpu-available? gpu-probe/gpu-skip!]
                                      [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
      (if-not @available?
        (skip! (str "mixed attention/projection oracle on " device))
        (let [descriptor (hardware/descriptor-for device)
              subgroup? (and (= :gpu (:device-type descriptor))
                             (gpu-target/intel-opencl-subgroup-dialect? descriptor)
                             (integer? (hardware/preferred-subgroup-size descriptor)))]
          ;; Always exercise the complete reference graph, including on CI's subgroup-less
          ;; PoCL device. Add reassociated dispatch only where the current target admits it.
          (doseq [strategy (cond-> [:reference] subgroup? (conj :dispatch-reassociated))]
            (let [prepared (compiled/lower
                            #'attention-projection/attention-projection device-args
                            {:compiler :equation-first :target device :dtype :float
                             :schedule {:typed-contraction {:strategy :dispatch-register-tiled}
                                        :segmented-weighted-reduction {:strategy strategy}}})
                  live (compiled/instantiate! prepared)]
              (try
                (dotimes [_ 2]
                  (let [actual (vec (value/->host (:result (live {}))))]
                    (is (= (count expected) (count actual)))
                    (is (every? true? (map #(<= (Math/abs (- (double %1) (double %2))) 1.0e-6)
                                          expected actual))
                        (str device " " strategy " Double-source oracle versus explicit FP32 execution"))
                    (is (= (vec (repeat 7 0.0)) (subvec actual 7 14))
                        "the empty destination row remains initialized through projection")))
                (finally (compiled/close! live))))))))))

(deftest dynamic-register-tiles-replay-with-tail-shapes-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 gpu-probe/gpu-available? gpu-probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "equation-first dynamic register tile on " device))
      (doseq [[batch in-f out-f] [[1 3 2] [3 5 7] [65 17 67]]]
        (let [left (float-array (map #(/ (- (mod % 17) 8) 4.0)
                                     (range (* batch in-f))))
              right (float-array (map #(/ (- (mod % 13) 6) 8.0)
                                      (range (* out-f in-f))))
              arguments [left right batch out-f in-f]
              expected (vec (apply contractions/dynamic-matmul arguments))
              prepared (compiled/lower
                        #'contractions/dynamic-matmul arguments
                        {:compiler :equation-first :target device :dtype :float
                         :schedule {:typed-contraction {:strategy :register-tiled}}})
              live (compiled/instantiate! prepared)]
          (try
            ;; Dyadic values keep these sums exact in FP32, even with target FMA.
            (dotimes [_ 2]
              (is (= expected (vec (value/->host (:result (live {})))))
                  (str device " " [batch in-f out-f])))
            (finally (compiled/close! live))))))))

(deftest shared-transposed-weights-use-the-generated-register-schedule
  (doseq [[device available? skip!] [[:ze:0 gpu-probe/gpu-available? gpu-probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "equation-first generated linear-nb on " device))
      (doseq [[batch in-f out-f] [[1 3 2] [3 5 7] [65 17 67]]]
        (let [x (float-array (map #(/ (- (mod % 17) 8) 4.0) (range (* batch in-f))))
              w (float-array (map #(/ (- (mod % 13) 6) 8.0) (range (* out-f in-f))))
              arguments [x w batch in-f out-f]
              expected (vec (apply nn/linear-nb arguments))
              prepared (compiled/lower
                        #'nn/linear-nb arguments
                        {:compiler :equation-first :target device :dtype :float
                         :constants '[W]
                         :schedule {:typed-contraction {:strategy :register-tiled}}})
              live (compiled/instantiate! prepared)]
          (try
            (dotimes [_ 2]
              (is (= expected (vec (value/->host (:result (live {})))))
                  (str device " shared NT " [batch in-f out-f])))
            (finally (compiled/close! live))))))))

(def ^:private source
  '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                      (* (clojure.core/aget A (+ (* i k) l))
                                         (clojure.core/aget B (+ (* l n) j))))]
     step))

(def ^:private batched-source
  '(let* [step (raster.par/contract
                C [[b batch] [i m] [j n]] [[l k]]
                (* (clojure.core/aget A (+ (* (+ (* b m) i) k) l))
                   (clojure.core/aget B (+ (* l n) j))))]
     step))

(def ^:private both-batched-source
  '(let* [step (raster.par/contract
                C [[b batch] [i m] [j n]] [[l k]]
                (* (clojure.core/aget A (+ (* (+ (* b m) i) k) l))
                   (clojure.core/aget B (+ (* (+ (* b k) l) n) j))))]
     step))

(defn- typed-dispatch
  ([device-id] (typed-dispatch device-id :int))
  ([device-id scalar-dtype]
  (let [{:keys [form]}
        (pipeline/schedule-parallel-form
         source {:target-device device-id
                 :dtype :float
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m scalar-dtype 'n scalar-dtype 'k scalar-dtype}})
        equation (first (:equations form))]
    (contract-route/route-typed-contraction-dispatch
     (:algorithm equation) (first (:operations equation))
     :dtype :float
     :desc (hardware/descriptor-for device-id)
     :precision :mixed-f16-f32))))

(defn- typed-batched-dispatch
  [device-id scalar-dtype both-batched?]
  (let [{:keys [form]}
        (pipeline/schedule-parallel-form
         (if both-batched? both-batched-source batched-source)
                        {:target-device device-id
                         :dtype :float
                         :array-types {'A :float 'B :float 'C :float}
                         :scalar-types {'batch scalar-dtype 'm scalar-dtype
                                        'n scalar-dtype 'k scalar-dtype}})
        equation (first (:equations form))]
    (contract-route/route-typed-contraction-dispatch
     (:algorithm equation) (first (:operations equation))
     :dtype :float
     :desc (hardware/descriptor-for device-id)
     :precision :mixed-f16-f32)))

(defn- typed-epilogue-dispatch
  [device-id scalar-dtype]
  (let [transform {:acc 'acc
                   :expr '(raster.numeric/*
                           (raster.numeric/+ acc (clojure.core/aget bias j)) scale)
                   :operands [{:sym 'bias :dtype :float
                               :map {:groups [[['j 'n]]]}}]
                   :scalars [{:sym 'scale :dtype :float}]
                   :dtype :float}
        contract (apply list
                        (concat
                         '(raster.par/contract C [[i m] [j n]] [[l k]]
                                               (* (clojure.core/aget A (+ (* i k) l))
                                                  (clojure.core/aget B (+ (* l n) j))))
                         [:epilogue transform]))
        {:keys [form]}
        (pipeline/schedule-parallel-form
         (list 'let* ['step contract] 'step)
         {:target-device device-id
          :dtype :float
          :array-types {'A :float 'B :float 'C :float 'bias :float}
          :scalar-types {'m scalar-dtype 'n scalar-dtype 'k scalar-dtype 'scale :float}})
        equation (first (:equations form))]
    (contract-route/route-typed-contraction-dispatch
     (:algorithm equation) (first (:operations equation))
     :dtype :float
     :desc (hardware/descriptor-for device-id)
     :precision :mixed-f16-f32)))

(defn- input-array
  [n seed]
  (let [result (float-array n)
        random (java.util.Random. (long seed))]
    (dotimes [index n]
      (aset result index (float (* 0.05 (.nextGaussian random)))))
    result))

(defn- f16
  ^double [value]
  (double (Float/float16ToFloat (Float/floatToFloat16 (float value)))))

(defn- reference
  [^floats a ^floats b m n k]
  (let [result (float-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset result (+ (* i n) j)
              (float
               (loop [l 0
                      sum 0.0]
                 (if (< l k)
                   (recur (inc l)
                          (+ sum (* (f16 (aget a (+ (* i k) l)))
                                    (f16 (aget b (+ (* l n) j))))))
                   sum))))))
    result))

(defn- batched-reference
  [^floats a ^floats b batch m n k both-batched?]
  (let [result (float-array (* batch m n))]
    (dotimes [batch-index batch]
      (dotimes [i m]
        (dotimes [j n]
          (aset result (+ (* (+ (* batch-index m) i) n) j)
                (float
                 (loop [l 0
                        sum 0.0]
                   (if (< l k)
                     (recur (inc l)
                            (+ sum
                               (* (f16 (aget a (+ (* (+ (* batch-index m) i) k) l)))
                                  (f16 (aget b (+ (if both-batched? (* batch-index k n) 0)
                                                  (* l n) j))))))
                     sum)))))))
    result))

(defn- relative-l1
  [^floats actual ^floats expected]
  (loop [index 0
         difference 0.0
         scale 0.0]
    (if (< index (alength actual))
      (recur (inc index)
             (+ difference
                (Math/abs (- (double (aget actual index))
                             (double (aget expected index)))))
             (+ scale (Math/abs (double (aget expected index)))))
      (/ difference (max scale 1.0e-30)))))

(defn- run-contraction
  ([device-id scheduled m n k] (run-contraction device-id scheduled m n k :int))
  ([device-id scheduled m n k scalar-dtype]
   (run-contraction device-id scheduled m n k scalar-dtype :nn))
  ([device-id scheduled m n k scalar-dtype variant]
  (let [a (input-array (* m k) 17)
        b (input-array (* k n) 29)
        transpose (fn [^floats values rows columns]
                    (float-array (for [column (range columns) row (range rows)]
                                   (aget values (+ (* row columns) column)))))
        stored-a (if (contains? #{:tn :tt} variant) (transpose a m k) a)
        stored-b (if (contains? #{:nt :tt} variant) (transpose b k n) b)
        runtime-arguments
        [:a :b :c
         {:type scalar-dtype :value m}
         {:type scalar-dtype :value n}
         {:type scalar-dtype :value k}]
        selected (if (dispatch/kernel-dispatch? scheduled)
                   (dispatch/select-alternative scheduled runtime-arguments)
                   scheduled)]
    (gpu/with-gpu-session [session device-id]
      (gpu/alloc! session {:a [:float (* m k) stored-a]
                           :b [:float (* k n) stored-b]
                           :c [:float (* m n) nil]})
      (let [handle (gpu/bind-kernel-executable!
                    session [:typed-contraction m n k] selected runtime-arguments)]
        (try
          (gpu/run-kernel-graph! session handle)
          {:strategy (executable/strategy selected)
           :actual (gpu/download session :c)
           :expected (reference a b m n k)}
          (finally
            (gpu/release-kernel-graph! session handle))))))))

(deftest public-composed-matrix-selects-one-tile-local-input-kernel
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "public composed GEMM input fusion")
    (let [shape [13 32 32]
          [^floats a b :as arrays] (canary/gemm-arguments shape)
          ordinary (canary/prepare-gemm :ze:0 arrays shape
                                       {:variant :relu-prebound :gemm-precision :mixed-f16-f32
                                        :constants ['B]})
          choice (get-in ordinary [:descriptor :steps 0 :dispatch])
          ;; Exercise the public recompilation schedule seam, without manufacturing a measured
          ;; winner or replacing a Prepared descriptor after its certificate was constructed.
          selected (compiled/lower
                    #'canary/gemm-relu-prebound! (into arrays shape)
                    {:target :ze:0 :dtype :float :gemm-precision :mixed-f16-f32
                     :constants ['B] :on-non-resident :throw
                     :schedule {:typed-contraction
                                {:measured-selectors
                                 {(:id choice) {:kind :fixed-strategy
                                                :strategy :xmx-direct-tile-inputs}}}}})]
      (is (= 1 (count (get-in selected [:descriptor :steps]))))
      (is (empty? (get-in selected [:descriptor :allocs])))
      (is (some #{:xmx-direct-tile-inputs}
                (map executable/strategy (:alternatives choice))))
      (let [live (compiled/instantiate! selected {:profile? true})]
        (try
          (let [bound (mapv :executable (compiled/execution-info live))]
            (is (= [:xmx-direct-tile-inputs] (mapv :strategy bound)))
            (is (= [:mixed-f16-f32] (mapv :precision bound)))
            (is (= [[{:strategy :xmx-direct-tile-inputs :reasons []}]]
                   (mapv :admission bound))))
          (dotimes [replay 2]
            (when (pos? replay)
              (dotimes [i (alength a)] (aset a i (- (aget a i)))))
            (let [result (compiled/profile live)
                  actual (vec (first (vals (:result result))))
                  expected (mapv #(max (float 0.0) %)
                                 (canary/gemm-reference a b shape))
                  profile (:profile result)]
              (is (= expected actual) (str "public input-fused replay " replay))
              (is (= 1 (count profile)) "conversion and matrix execution share one kernel")
              (is (= [:xmx-direct-tile-inputs :contract]
                     (take-last 2 (:phase (first profile)))))))
          (finally (compiled/close! live)))))))

(deftest long-graph-interface-executes-layout-matrix-and-combine
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "Long matrix graph interface")
    (let [device-id :ze:0
          interface (update (executable/common-view
                             (dispatch/default-alternative (typed-dispatch device-id))) :abi
                            #(mapv (fn [slot] (if (= :scalar (:kind slot))
                                               (assoc slot :dtype :long :kernel-dtype :long)
                                               slot)) %))]
      ;; This exercises the graph constructor and common binder; it deliberately does not
      ;; bypass the still-closed Long admission gate in the public typed contraction route.
      (doseq [variant [:nn :nt :tn :tt]
              :let [{:keys [alternatives]}
                    (gemm/emit-matrix-alternatives
                     {:id [:long-device-matrix variant] :a 'A :b 'B :c 'C :m 'm :n 'n :k 'k
                      :variant variant :tile (hardware/derive-gemm-tile {})
                      :fill-workgroups 32 :split-factors [2] :external-interface interface})]
              strategy [:xmx-direct :xmx-split-k-2]
              :let [graph (some #(when (= strategy (executable/strategy %)) %) alternatives)
                    {:keys [actual expected]} (run-contraction device-id graph 8 32 64 :long variant)]]
        (is (some? graph))
        (is (< (relative-l1 actual expected) 1.0e-3))))))

(deftest ordinary-typed-contraction-executes-the-matrix-schedule
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "typed contraction matrix KernelExecutable")
    (let [scheduled (typed-dispatch :ze:0)]
      (testing "an aligned dynamic contraction selects and executes the direct matrix graph"
        (let [{:keys [strategy actual expected]}
              (run-contraction :ze:0 scheduled 16 32 32)]
          (is (= :xmx-direct strategy))
          (is (< (relative-l1 actual expected) 1.0e-3))))
      (testing "a low-output-occupancy contraction executes the graph-private split-K schedule"
        (let [{:keys [strategy actual expected]}
              (run-contraction :ze:0 scheduled 13 32 8192)]
          (is (= :xmx-split-k strategy))
          (is (< (relative-l1 actual expected) 1.0e-3)))))))

(deftest public-long-contraction-executes-guarded-matrix-schedules
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "public Long typed contraction")
    (let [scheduled (typed-dispatch :ze:0 :long)]
      (doseq [[m n k expected-strategy] [[16 32 32 :xmx-direct] [13 32 8192 :xmx-split-k]]
              :let [{:keys [strategy actual expected]}
                    (run-contraction :ze:0 scheduled m n k :long)]]
        (is (= expected-strategy strategy))
        (is (< (relative-l1 actual expected) 1.0e-3))))))

(deftest batched-typed-contraction-executes-with-shared-weights
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "batched typed contraction matrix KernelExecutable")
    (doseq [scalar-dtype [:int :long]
            both-batched? [false true]]
     (let [device-id :ze:0
          batch 2
          m 8
          n 32
          k 32
          a (input-array (* batch m k) 41)
          b-elements (* (if both-batched? batch 1) k n)
          b (input-array b-elements 43)
          scheduled (typed-batched-dispatch device-id scalar-dtype both-batched?)
          runtime-arguments
          [:a :b :c
           {:type scalar-dtype :value batch}
           {:type scalar-dtype :value m}
           {:type scalar-dtype :value n}
           {:type scalar-dtype :value k}]
          selected (dispatch/select-alternative scheduled runtime-arguments)]
      (is (= :xmx-batched (executable/strategy selected)))
      (gpu/with-gpu-session [session device-id]
        (gpu/alloc! session {:a [:float (* batch m k) a]
                             :b [:float b-elements b]
                             :c [:float (* batch m n) nil]})
        (let [handle (gpu/bind-kernel-executable!
                      session :typed-batched-contraction selected runtime-arguments)]
          (try
            (gpu/run-kernel-graph! session handle)
            (is (< (relative-l1 (gpu/download session :c)
                                (batched-reference a b batch m n k both-batched?))
                   1.0e-3))
            (finally
              (gpu/release-kernel-graph! session handle)))))))))

(deftest typed-result-transform-executes-inside-the-matrix-store
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "typed matrix result transform")
    (doseq [scalar-dtype [:int :long]]
     (let [device-id :ze:0
          m 8
          n 32
          k 32
          scale 0.5
          a (input-array (* m k) 47)
          b (input-array (* k n) 53)
          bias (input-array n 59)
          base (reference a b m n k)
          expected (float-array (* m n))
          _ (dotimes [index (* m n)]
              (aset expected index
                    (float (* scale
                              (+ (double (aget base index))
                                 (double (aget bias (mod index n))))))))
          scheduled (typed-epilogue-dispatch device-id scalar-dtype)
          runtime-arguments
          [:a :b :c :bias
           {:type :float :value scale}
           {:type scalar-dtype :value m}
           {:type scalar-dtype :value n}
           {:type scalar-dtype :value k}]
          selected (dispatch/select-alternative scheduled runtime-arguments)]
      (is (= :xmx-direct (executable/strategy selected)))
      (gpu/with-gpu-session [session device-id]
        (gpu/alloc! session {:a [:float (* m k) a]
                             :b [:float (* k n) b]
                             :bias [:float n bias]
                             :c [:float (* m n) nil]})
        (let [handle (gpu/bind-kernel-executable!
                      session :typed-result-transform selected runtime-arguments)]
          (try
            (gpu/run-kernel-graph! session handle)
            (is (< (relative-l1 (gpu/download session :c) expected) 1.0e-3))
            (finally
              (gpu/release-kernel-graph! session handle)))))))))
