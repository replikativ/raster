(ns raster.compiler.passes.parallel.typed-contraction-matrix-device-test
  "The optimized matrix route is exercised from the typed contraction equation through the
   backend-neutral KernelExecutable binder.  This is the device guard that permits the old
   Level Zero GEMM-specific binding surface to disappear without losing its numerical oracle."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.passes.parallel.contract-route :as contract-route]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.perf.production-canary :as canary]))

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

(deftest public-composed-matrix-selects-input-fusion-and-hoists-constant-weights
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
                                                :strategy :xmx-direct-lhs-tile-cast}}}}})]
      (is (= 1 (count (get-in selected [:descriptor :steps]))))
      (is (empty? (get-in selected [:descriptor :allocs])))
      (is (some #{:xmx-direct-lhs-tile-cast}
                (map executable/strategy (:alternatives choice))))
      (let [live (compiled/instantiate! selected {:profile? true})]
        (try
          (dotimes [replay 2]
            (when (pos? replay)
              (dotimes [i (alength a)] (aset a i (- (aget a i)))))
            (let [result (compiled/profile live)
                  actual (vec (first (vals (:result result))))
                  expected (mapv #(max (float 0.0) %)
                                 (canary/gemm-reference a b shape))
                  profile (:profile result)]
              (is (= expected actual) (str "public input-fused replay " replay))
              (is (= 1 (count profile)) "weight conversion stays in the one-time prologue")
              (is (= [:xmx-direct-lhs-tile-cast :contract]
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
