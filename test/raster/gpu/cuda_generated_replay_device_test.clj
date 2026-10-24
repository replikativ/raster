(ns raster.gpu.cuda-generated-replay-device-test
  "Generated float arithmetic and changed-view replay through public Prepared/Compiled.
   CUDA admission is test-local; native calls and NVRTC compilation are real delegates."
  (:require [clojure.string :as string]
            [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.cuda-nvrtc :as nvrtc]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.link :as link]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-backend :as backend]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.value :as value]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.math :as math]
            [raster.numeric :as numeric]
            [raster.par :as par]
            [raster.runtime.hardware :as hardware]))

(deftm arithmetic!
  [a :- (Array float) b :- (Array float) c :- (Array float)
   separate :- (Array float) fused :- (Array float) n :- Long] :- Void
  (par/map-void! i n
    (do
      (arrays/aset separate i
                   (numeric/+ (numeric/* (arrays/aget a i) (arrays/aget b i))
                              (arrays/aget c i)))
      (arrays/aset fused i
                   (math/fma (arrays/aget a i) (arrays/aget b i) (arrays/aget c i))))))

(defn- float-bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- inputs [n shift]
  (let [a (float (+ 1.0 (Math/scalb 1.0 (int -13))))
        b (float (- 1.0 (Math/scalb 1.0 (int -13))))
        cases [[a b -1.0] [b a -1.0] [(- a) b 1.0] [1.0 1.0 1.0]]]
    (into {} (map-indexed
              (fn [column key]
                [key (float-array (map #(nth (nth cases (mod (+ % shift) 4)) column)
                                      (range n)))]) [:a :b :c]))))

(defn- independent-bits [{:keys [a b c]} n]
  ;; Products/sums of these selected dyadics are exact in binary64.
  ;; This deliberately uses neither Raster arithmetic nor Math/fma.
  (let [rows (map vector (take n a) (take n b) (take n c))]
    {:separate (float-bits (map (fn [[x y z]]
                                 (float (+ (double (float (* (double x) (double y))))
                                           (double z)))) rows))
     :fused (float-bits (map (fn [[x y z]]
                              (float (+ (* (double x) (double y)) (double z)))) rows))}))

(defn- lower [n]
  (let [{:keys [a b c]} (inputs n 0)
        guarded #(float-array (concat (repeat n 123.0) [456.0 -789.0]))]
    (compiled/lower #'arithmetic! [a b c (guarded) (guarded) (long n)]
                    {:compiler :equation-first :target :cuda:0 :dtype :float
                     :schedule {:precision :f32-scalar}
                     :outputs '[separate fused]
                     :roles {'separate :state 'fused :state}
                     :on-non-resident :throw})))

(defn- with-target [f]
  ;; Synthetic catalogue input for device-free compiler lowering only. The native
  ;; runtime and NVRTC select architecture from the initialized physical device.
  (hardware-fixture/isolated
    #(do (hardware/register-target-device! :cuda:0 {:name "NVIDIA A100"}) (f))))

(defn- with-cuda-admission [f]
  (let [original-type backend/backend-type original-descriptor backend/descriptor
        descriptor {:namespace 'raster.gpu.cuda-runtime :kernel-body-c-dialect :cuda
                    :memory-space :device :coherence :explicit-transfer :owned-slice? true}]
    (with-redefs [backend/backend-type
                  (fn [device] (if (= :cuda:0 device) :cuda (original-type device)))
                  backend/descriptor
                  (fn [device] (if (= :cuda:0 device) descriptor (original-descriptor device)))]
      (f))))

(defn- preserving-primary [body retire!]
  (let [primary (volatile! nil)]
    (try (body)
         (catch Throwable failure (vreset! primary failure) (throw failure))
         (finally
           (try (retire!)
                (catch Throwable failure
                  (if-let [original @primary]
                    (when-not (identical? original failure) (.addSuppressed original failure))
                    (throw failure))))))))

(deftest dyadic-oracle-distinguishes-separate-and-explicit-fused-rounding
  (let [bits (independent-bits (inputs 4 0) 4)]
    (is (= [0 0 0 1073741824] (:separate bits)))
    (is (= [(unchecked-int 0xb2800000) (unchecked-int 0xb2800000)
            847249408 1073741824] (:fused bits)))
    (is (not= (:separate bits) (:fused bits)))
    (is (not= (independent-bits (inputs 257 0) 257)
              (independent-bits (inputs 257 1) 257)))
    (is (not= (independent-bits (inputs 257 1) 257)
              (independent-bits (inputs 257 2) 257)))))

(deftest public-lowering-is-allocation-free-without-runtime-admission
  (let [calls (atom [])]
    (binding [driver/*invoke* (fn [operation args]
                               (swap! calls conj [operation args])
                               (throw (ex-info "pure lowering attempted driver access" {:operation operation})))]
      (with-target
        #(let [prepared (lower 257) plan (compiled/plan prepared)]
           (is (compiled/prepared? prepared))
           (is (zero? (get-in plan [:attributes :driver-allocations])))
           (is (= :state (get-in plan [:attributes :public-buffer-roles 'separate])))
           (is (= :state (get-in plan [:attributes :public-buffer-roles 'fused])))
           (is (= #{:separate :fused} (set (map :key (:out-tree prepared)))))))
      ;; Compile-only CUDA is valid; session creation remains refused outside the adapter.
      (let [error (try (gpu/make-session :cuda:0) nil
                       (catch clojure.lang.ExceptionInfo failure failure))]
        (is (some? error))
        (is (= true (:compile-only? (ex-data error)))))
      (is (empty? @calls)))))

(defn- prove-outputs! [program outputs expected n phase]
  (let [session (get-in program [:executable :session])]
    (doseq [{:keys [key node]} (:out-tree program)]
      (let [resident (link/node-view (:executable program) node)
            buffer (gpu/buffer session (:key resident))
            actual (float-bits (value/->host (get outputs key)))
            physical (float-bits (gpu/download session (:key resident)))
            expected-full (into (get expected key) (float-bits [456.0 -789.0]))]
        (is (= (* 4 (+ n 2)) (:byte-size buffer)) "whole output backing includes guards")
        (is (= (get expected key) (vec (take n actual))))
        (is (= expected-full physical) "masked tail and complete output backing")
        (prn {:event :cuda-generated-output-witness :phase phase :output key
              :elements n :expected-bits expected-full :actual-bits physical})))))

(defn- retire-fixture-root! [ops returned]
  (when-let [entry (::root/entry @runtime/state)]
    (let [leases (count @(:leases entry)) ledger (:prepared-execution @runtime/state)
          pending (count @(:pending ledger)) debts (count @(:debts ledger))]
      (when-not (and (zero? leases) (zero? pending) (zero? debts) (empty? @runtime/kernel-registry))
        (throw (ex-info "Generated fixture still owns children or completion debt"
                        {:leases leases :pending pending :debts debts})))
      ;; Test-owned canonical root retirement, not public shutdown qualification.
      (cleanup/release! (::cleanup/owner entry))
      (when (seq (cleanup/pending (::cleanup/owner entry)))
        (throw (ex-info "Generated fixture root teardown remains incomplete" {})))
      (doseq [[acquire release] [[:allocate :free] [:host-allocate :host-free]
                                [:module-load :module-unload]]]
        (is (= (count (filter #(= acquire %) @returned))
               (count (filter #(= release %) @returned)))))
      (is (= 1 (count (filter #(= :destroy-runtime %) @returned))))
      (is (= 1 (count (filter #(= :destroy-runtime (first %)) @ops))))
      (reset! runtime/state {})
      (is (empty? @runtime/state))
      (prn {:event :cuda-generated-retired :leases leases :pending pending :debts debts
            :kernel-registry-empty true :destroy-runtime-returned 1
            :production-shutdown-qualified false}))))

(defn- replay-proof! [ops]
  (let [n 257 prepared (lower n) program (compiled/instantiate! prepared) last-output (atom nil)]
    (preserving-primary
      (fn []
        (doseq [phase [0 1]]
          (let [host (if (zero? phase) (zipmap [:a :b :c] (take 3 (:args prepared)))
                         (inputs n phase)) original (into {} (map (fn [[key data]] [key (float-bits data)]) host))
                output (program (if (zero? phase) {} host))]
            (when @last-output (is (every? #(not (value/live? %)) (vals @last-output))))
            (reset! last-output output)
            (prove-outputs! program output (independent-bits host n) n phase)
            (let [actual (into {} (map (fn [[key data]] [key (float-bits data)]) host))]
              (is (= original actual))
              (prn {:event :cuda-generated-host-source-witness :phase phase
                    :expected-bits original :actual-bits actual}))))
        (let [source-session (gpu/make-session :cuda:0) views (atom {})]
          (preserving-primary
            (fn []
              (let [host (inputs n 2)
                    backing (into {} (map (fn [[key data]]
                                            [key (float-array (concat [999.0 -999.0 321.0]
                                                                      data [-321.0 654.0 -654.0]))]) host))]
                (doseq [[key data] backing]
                  (gpu/alloc! source-session {key [:float (+ n 6) data]})
                  (let [resident (gpu/buffer-view source-session key {:byte-offset 12 :shape [n]})]
                    (swap! views assoc key (value/wrap-external-view
                                            (gpu/buffer source-session key) :cuda:0 (:view resident)))))
                (let [before (count @ops)
                      output (with-redefs [value/->host
                                           (fn [_] (throw (ex-info "resident replay materialized an input" {})))]
                               (program @views))
                      copies (filter #(= :copy (first %)) (drop before @ops))]
                  (is (= 3 (count copies)) "each external input refresh delegates real D2D")
                  (is (= (set (map (fn [[key _]]
                                    (+ 12 (:device-pointer (gpu/buffer source-session key)))) backing))
                         (set (map #(nth (second %) 1) copies))))
                  (is (every? #(= (* 4 n) (nth (second %) 2)) copies))
                  (is (every? #(not (value/live? %)) (vals @last-output)))
                  (reset! last-output output)
                  (prove-outputs! program output (independent-bits host n) n 2)
                  (doseq [[key data] backing]
                    (let [expected (float-bits data) actual (float-bits (gpu/download source-session key))]
                      (is (= expected actual) "complete external source backing remains unchanged")
                      (is (value/live? (get @views key)))
                      (prn {:event :cuda-generated-source-witness :phase 2 :input key
                            :source-element 3 :elements n :expected-bits expected :actual-bits actual}))))))
            (fn [] (preserving-primary
                     (fn [] (doseq [view (vals @views)] (value/free! view)))
                     (fn [] (gpu/close-session! source-session)))))))
      #(compiled/close! program))
    (is (every? #(not (value/live? %)) (vals @last-output)))
    (is (= 0 (root/lease-count runtime/state)))))

(deftest native-generated-map-changed-host-and-external-offset-view-replay
  (let [required? (some #(contains? #{"1" "true" "yes" "on"}
                                    (some-> (System/getenv %) string/lower-case))
                        ["RASTER_REQUIRE_CUDA_DEVICE" "RASTER_REQUIRE_NVRTC"])
        device (try (probe/probe) (catch Throwable error {:status :probe-error :error error}))
        ready? (and (= :ready (:status device)) (nvrtc/available?))]
    (if-not ready?
      (if required?
        (throw (ex-info "Required generated replay needs both a ready CUDA device and NVRTC"
                        {:probe device :nvrtc-available? (nvrtc/available?) :required true} (:error device)))
        (do (println "[CUDA GENERATED SKIP]" {:status (:status device) :nvrtc-available? (nvrtc/available?)})
            (is true "cuda-generated-device-skip-marker")))
      (with-target
        (fn [] (with-cuda-admission
           (fn []
             (with-redefs [runtime/state (atom {}) runtime/kernel-registry (atom {})]
               (let [ops (atom []) returned (atom []) captures (atom []) delegate nvrtc/compile-artifact-ptx]
                 (binding [driver/*invoke*
                           (fn [operation args]
                             (swap! ops conj [operation args])
                             (let [result (binding [driver/*invoke* nil] (driver/invoke! operation args))]
                               (swap! returned conj operation) result))]
                   (with-redefs [nvrtc/compile-artifact-ptx
                                 (fn [artifact architecture & options]
                                   (let [result (apply delegate artifact architecture options)
                                         ptx (String. (:ptx result) java.nio.charset.StandardCharsets/UTF_8)
                                         capture {:artifact artifact :architecture architecture
                                                  :runtime-device (runtime/execution-device-info)
                                                  :compilation (assoc result :ptx ptx)}]
                                     (swap! captures conj capture)
                                     (prn {:event :cuda-generated-nvrtc-witness :capture capture})
                                     result))]
                     (preserving-primary #(replay-proof! ops) #(retire-fixture-root! ops returned)))
                   (is (seq @captures) "actual generated artifacts compiled through NVRTC")
                   (let [ptx (string/join "\n" (map #(get-in % [:compilation :ptx]) @captures))]
                     (is (re-find #"\bmul\.rn\.f32\b" ptx))
                     (is (re-find #"\badd\.rn\.f32\b" ptx))
                     (is (re-find #"\bfma\.rn\.f32\b" ptx)))
                   (doseq [{:keys [artifact architecture runtime-device compilation]} @captures]
                     (let [physical (first (filter #(zero? (:ordinal %)) (:devices device)))
                           [major minor] (:compute-capability runtime-device)
                           expected-architecture (str "compute_" major minor)]
                       (is (= :ready (:status device)))
                       (is (some? physical) "ready probe includes ordinal zero")
                       (is (= 0 (:device-ordinal runtime-device)))
                       (is (= (:name physical) (:name runtime-device)))
                       (is (= (:compute-capability physical) (:compute-capability runtime-device)))
                       (is (= (:driver-api-version device) (:driver-api-version runtime-device)))
                       (is (= expected-architecture architecture))
                       (is (= expected-architecture (:virtual-architecture compilation)))
                       (is (= [(str "--gpu-architecture=" expected-architecture)]
                              (filterv #(string/starts-with? % "--gpu-architecture=")
                                       (:compiler-options compilation))))
                       (prn {:event :cuda-generated-device-witness :probe device
                             :ordinal-zero physical :runtime-device runtime-device
                             :architecture architecture :compiler-options (:compiler-options compilation)
                             :lowering-model {:name "NVIDIA A100" :synthetic true
                                              :purpose :compile-only-catalogue}}))
                     (is (seq (:source artifact)))
                     (is (seq (:abi artifact)))
                     (is (some? (:launch artifact)))
                     (is (some #{"--fmad=false"} (:compiler-options compilation))))))))))))))
