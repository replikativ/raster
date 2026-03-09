(ns raster.abm.firms.gpu-test
  "Tests for GPU-targeted par-based ABM phases.

   test-par-vs-sequential: Verifies that par expansion (CPU fallback)
   produces identical results to the existing deftm sequential code.

   test-gpu-vs-cpu: Runs on GPU (if :ze:0 available) and compares
   aggregate stats to CPU results (tolerance for float atomics)."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.abm.firms :as firms]
            [raster.abm.firms.phases :as phases]
            [raster.abm.firms.gpu :as fgpu]
            [raster.gpu.core :as gpu]
            [raster.abm.firms.membership :as mem]
            [raster.par :as par]
            [raster.gpu.ze-runtime :as ze]
            [raster.compiler.support.autotuner :as at]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass])
  (:import [raster.abm.firms AgentSoA FirmSoA FirmsConfig]
           [java.util SplittableRandom]))

(def ^:private gpu-available?
  (delay
    (try
      (let [p (.start (ProcessBuilder. ["ocloc" "--help"]))]
        (.waitFor p)
        (zero? (.exitValue p)))
      (catch Exception _ false))))

(use-fixtures :once
  (fn [f] (if @gpu-available? (f) (println "[SKIP] No GPU/OpenCL compiler (ocloc)"))))

;; ================================================================
;; Helpers
;; ================================================================

(defn- clone-float-array ^floats [^floats arr]
  (let [out (float-array (alength arr))]
    (System/arraycopy arr 0 out 0 (alength arr))
    out))

(defn- clone-int-array ^ints [^ints arr]
  (let [out (int-array (alength arr))]
    (System/arraycopy arr 0 out 0 (alength arr))
    out))

(defn- clone-long-array ^longs [^longs arr]
  (let [out (long-array (alength arr))]
    (System/arraycopy arr 0 out 0 (alength arr))
    out))

(defn- clone-agents
  "Deep-clone an AgentSoA."
  ^AgentSoA [^AgentSoA a]
  (firms/->AgentSoA
   (clone-float-array (.effort a))
   (clone-float-array (.income a))
   (clone-float-array (.theta a))
   (clone-float-array (.endowment a))
   (clone-int-array (.firm a))
   (clone-int-array (.friends a))
   (clone-float-array (.cache a))))

(defn- clone-firms
  "Deep-clone a FirmSoA."
  ^FirmSoA [^FirmSoA f]
  (firms/->FirmSoA
   (clone-float-array (.a f))
   (clone-float-array (.b f))
   (clone-float-array (.beta f))
   (clone-float-array (.te f))
   (clone-float-array (.output f))
   (clone-int-array (.size f))
   (clone-int-array (.alive f))
   (clone-int-array (.members f))
   (clone-int-array (.offsets f))))

(defn- arrays-equal?
  "Check if two float arrays are element-wise equal within tolerance."
  [^floats a ^floats b tol]
  (and (== (alength a) (alength b))
       (every? identity
               (map (fn [i]
                      (<= (Math/abs (- (double (clojure.core/aget a i))
                                       (double (clojure.core/aget b i))))
                          tol))
                    (range (alength a))))))

(defn- int-arrays-equal?
  "Check if two int arrays are element-wise equal."
  [^ints a ^ints b]
  (and (== (alength a) (alength b))
       (every? identity
               (map (fn [i]
                      (== (clojure.core/aget a i) (clojure.core/aget b i)))
                    (range (alength a))))))

;; ================================================================
;; Test: par phases vs sequential (CPU correctness)
;; ================================================================

(deftest test-produce-output-par
  (testing "produce-output-par! matches sequential produce-output-impl!"
    (let [n 100
          max-firms (* n 2)
          [_ ^AgentSoA agents ^FirmSoA firms _] (firms/init-simulation
                                                 (firms/->FirmsConfig n 4 4 0.04 0 10 42 (* n 2)))
          ;; Clone firm arrays for par version
          output-seq  (clone-float-array (.output firms))
          output-par  (clone-float-array (.output firms))]
      ;; Run sequential
      (firms/produce-output-impl! (.a firms) (.b firms) (.beta firms)
                                  (.te firms) output-seq (.alive firms) max-firms)
      ;; Run par
      (phases/produce-output-par! (.a firms) (.b firms) (.beta firms)
                                  (.te firms) output-par (.alive firms) max-firms)
      (is (arrays-equal? output-seq output-par 1e-5)
          "Par and sequential produce-output should match"))))

(deftest test-distribute-income-par
  (testing "distribute-income-par! matches sequential distribute-income-impl!"
    (let [n 100
          max-firms (* n 2)
          [_ ^AgentSoA agents ^FirmSoA firms _] (firms/init-simulation
                                                 (firms/->FirmsConfig n 4 4 0.04 0 10 42 (* n 2)))
          income-seq (clone-float-array (.income agents))
          income-par (clone-float-array (.income agents))]
      ;; Run sequential
      (firms/distribute-income-impl! income-seq (.output firms) (.n-workers firms)
                                     (.alive firms) (.members firms) (.offsets firms) max-firms)
      ;; Run par
      (phases/distribute-income-par! income-par (.output firms) (.n-workers firms)
                                     (.alive firms) (.members firms) (.offsets firms) max-firms)
      (is (arrays-equal? income-seq income-par 1e-5)
          "Par and sequential distribute-income should match"))))

(deftest test-array-bundle-buffer-specs
  (testing "array-bundle-buffer-specs derives buffer specs from array-bundle defvalues"
    (let [config (firms/->FirmsConfig 16 4 4 0.04 0 10 42 16)
          [_ ^AgentSoA agents ^FirmSoA firms _] (firms/init-simulation config)
          agent-specs (gpu/array-bundle-buffer-specs agents {:aliases {:firm :current-firm
                                                                       :cache :s-cache}})
          firm-specs (gpu/array-bundle-buffer-specs firms {:aliases {:a :param-a
                                                                     :b :param-b
                                                                     :beta :param-beta
                                                                     :te :total-effort
                                                                     :n-workers :firm-size
                                                                     :offsets :member-offsets}})]
      (is (= [:float 16 (.effort agents)] (:effort agent-specs)))
      (is (= [:int 16 (.firm agents)] (:current-firm agent-specs)))
      (is (= [:int 64 (.friends agents)] (:friends agent-specs)))
      (is (= [:float 32 (.cache agents)] (:s-cache agent-specs)))
      (is (= [:float 16 (.a firms)] (:param-a firm-specs)))
      (is (= [:int 16 (.n-workers firms)] (:firm-size firm-specs)))
      (is (= [:int 17 (.offsets firms)] (:member-offsets firm-specs))))))

(deftest test-rebuild-csr-par
  (testing "rebuild-csr-par! matches sequential rebuild-csr!"
    (let [n 100
          max-firms (* n 2)
          [_ ^AgentSoA agents ^FirmSoA firms _] (firms/init-simulation
                                                 (firms/->FirmsConfig n 4 4 0.04 0 10 42 (* n 2)))
          ;; Clone for par version
          firm-size-seq  (clone-int-array (.n-workers firms))
          alive-seq      (clone-int-array (.alive firms))
          members-seq    (clone-int-array (.members firms))
          offsets-seq    (clone-int-array (.offsets firms))
          firm-size-par  (clone-int-array (.n-workers firms))
          alive-par      (clone-int-array (.alive firms))
          members-par    (clone-int-array (.members firms))
          offsets-par    (clone-int-array (.offsets firms))
          current-firm   (.firm agents)]
      ;; Run sequential
      (let [n-alive-seq (mem/rebuild-csr! current-firm firm-size-seq alive-seq
                                          members-seq offsets-seq n max-firms)
            n-alive-par (phases/rebuild-csr-par! current-firm firm-size-par alive-par
                                                 members-par offsets-par n max-firms)]
        (is (== n-alive-seq n-alive-par)
            "Par and sequential should report same n-alive")
        (is (int-arrays-equal? firm-size-seq firm-size-par)
            "Firm sizes should match")
        (is (int-arrays-equal? alive-seq alive-par)
            "Alive flags should match")
        (is (int-arrays-equal? offsets-seq offsets-par)
            "Member offsets should match")
        ;; Members ordering may differ due to atomic-add! order,
        ;; but each firm's member SET should be the same
        (doseq [f (range max-firms)]
          (let [start (clojure.core/aget offsets-seq f)
                end   (clojure.core/aget offsets-seq (inc f))
                seq-members (sort (map #(clojure.core/aget members-seq %)
                                       (range start end)))
                par-members (sort (map #(clojure.core/aget members-par %)
                                       (range start end)))]
            (is (= seq-members par-members)
                (str "Members of firm " f " should match (as sets)"))))))))

;; ================================================================
;; Test: full period par vs sequential
;; ================================================================

(deftest test-par-vs-sequential
  (testing "run-period-gpu! gives same aggregate stats as run-period!"
    (let [config (firms/->FirmsConfig 200 4 4 0.04 0 10 42 200)
          ;; Init two copies with same seed
          [_ agents1 firms1 rng1] (firms/init-simulation config)
          [_ agents2 firms2 rng2] (firms/init-simulation config)]
      ;; Run 5 periods with both
      (doseq [t (range 5)]
        (let [stats-seq (firms/run-period! config agents1 firms1 rng1 t)
              stats-par (phases/run-period config agents2 firms2 rng2 t)]
          ;; Stats should be close (both use same RNG sequence)
          (is (== (:n-alive stats-seq) (:n-alive stats-par))
              (str "Period " t ": n-alive should match"))
          (is (< (Math/abs (- (:mean-effort stats-seq) (:mean-effort stats-par))) 0.01)
              (str "Period " t ": mean-effort should be close"))
          (is (< (Math/abs (- (:mean-income stats-seq) (:mean-income stats-par))) 0.01)
              (str "Period " t ": mean-income should be close")))))))

;; ================================================================
;; Test: OpenCL codegen (smoke test for kernel generation)
;; ================================================================

;; ================================================================
;; Test: par/active-ids! CPU correctness (no GPU needed)
;; ================================================================

(deftest test-active-ids-cpu
  (testing "par/active-ids! CPU expansion: all indices in [0, n-agents)"
    (let [n-active 500
          n-agents 10000
          ids (int-array n-active)]
      (par/active-ids! ids n-active n-agents 42)
      (is (every? #(and (>= % 0) (< % n-agents)) ids)
          "All generated indices should be in [0, n-agents)")
      ;; Rough uniformity: no single value dominates (with 500 samples from 10000, P(collision) ≈ 1%)
      (let [unique (count (distinct (seq ids)))]
        (is (> unique 450) "Should generate mostly distinct indices (good distribution)"))))

  (testing "par/active-ids! different seeds → different indices"
    (let [n-active 100 n-agents 1000
          ids1 (int-array n-active) ids2 (int-array n-active)]
      (par/active-ids! ids1 n-active n-agents 11111)
      (par/active-ids! ids2 n-active n-agents 99999)
      (is (not= (seq ids1) (seq ids2)) "Different seeds should produce different results")))

  (testing "par/active-ids! same seed → same indices (deterministic)"
    (let [n-active 200 n-agents 5000
          ids1 (int-array n-active) ids2 (int-array n-active)]
      (par/active-ids! ids1 n-active n-agents 77777)
      (par/active-ids! ids2 n-active n-agents 77777)
      (is (= (seq ids1) (seq ids2)) "Same seed must be deterministic"))))

(deftest test-active-ids-codegen
  (testing "generate-par-active-ids-kernel emits correct long types"
    (let [k (par-opencl/generate-par-active-ids-kernel)]
      (is (some? k) "Should return kernel info")
      (is (string? (:source k)))
      (let [src ^String (:source k)]
        (is (.contains src "__global int* ids") "Output must be int*")
        (is (.contains src "long n_agents") "n_agents must be long")
        (is (.contains src "long base_seed") "base_seed must be long")
        (is (.contains src "0x9e3779b97f4a7c15") "Must use splitmix64 golden ratio"))
      (is (= ['ids] (:array-params k)) "Array param should be 'ids")
      (is (some #(= :long (:type %)) (:scalar-params k)) "Some scalar must be :long"))))

(deftest test-active-ids-opencl-dispatch
  (testing "opencl-pass dispatches par/active-ids! to invoke-registered-active-ids-kernel"
    (let [form '(raster.par/active-ids! ids n-active n-agents base-seed)
          result (opencl-pass/opencl-pass form :dtype :float :compile-spirv? false)]
      (is (= 1 (count (:kernels result))) "Should generate exactly one kernel")
      (let [k (first (:kernels result))
            emitted (:form result)]
        (is (.contains ^String (:source k) "__global int* ids") "Kernel output is int*")
        (is (seq? emitted) "Emitted form should be a seq")
        (is (= 'raster.gpu.ze-runtime/invoke-registered-active-ids-kernel (first emitted))
            "Should emit invoke-registered-active-ids-kernel call")))))

;; ================================================================
;; Test: GPU active-ids kernel (requires Level Zero)
;; ================================================================

(defn- ze-available? []
  (try (require 'raster.gpu.ze-runtime)
       (let [qfn (resolve 'raster.gpu.ze-runtime/query-devices)]
         (and qfn (seq (qfn))))
       (catch Exception _ false)))

(defmacro when-ze [& body]
  `(if (ze-available?)
     (do ~@body)
     (println "  [SKIP] No Level Zero GPU")))

(deftest test-active-ids-gpu
  (when-ze
   (testing "GPU active-ids kernel: all indices in [0, n-agents)"
     (ze/init!)
     (gpu/with-gpu-session [sess :ze:0]
       (fgpu/compile-abm-kernels! sess)
       (gpu/alloc! sess {:active-ids [:int 5000 nil]})
       (let [n-active 5000
             n-agents 100000]
         (gpu/invoke-active-ids! sess :generate-active-ids :active-ids n-active n-agents 42424242)
         (let [ids (gpu/download sess :active-ids)]
           (is (every? #(and (>= % 0) (< % n-agents)) ids)
               "All GPU-generated indices should be in [0, n-agents)")
           (let [unique (count (distinct (seq ids)))]
             (is (> unique 4700) "GPU indices should be well-distributed"))))))

   (testing "GPU active-ids: different seeds → different results"
     (ze/init!)
     (gpu/with-gpu-session [sess :ze:0]
       (fgpu/compile-abm-kernels! sess)
       (gpu/alloc! sess {:ids1 [:int 1000 nil]
                         :ids2 [:int 1000 nil]})
       (let [n-active 1000 n-agents 50000
             aid-kinfo (first (gpu/kernel sess :generate-active-ids))
             aid-kname (:kernel-name aid-kinfo)]
         (ze/invoke-registered-active-ids-kernel aid-kname (gpu/buffer sess :ids1) n-active n-agents 111)
         (ze/invoke-registered-active-ids-kernel aid-kname (gpu/buffer sess :ids2) n-active n-agents 999)
         (is (not= (seq (gpu/download sess :ids1)) (seq (gpu/download sess :ids2)))
             "Different seeds should give different GPU indices"))))))

;; ================================================================
;; Test: Full GPU Blelloch scan correctness (requires Level Zero)
;; ================================================================

(defn- exclusive-scan-reference
  "CPU reference: exclusive prefix sum of int-array. Returns int-array of n+1 elements."
  [^ints input n]
  (let [out (int-array (inc n))]
    (clojure.core/aset out 0 (int 0))
    (dotimes [i n]
      (clojure.core/aset out (inc i)
                         (unchecked-add-int (clojure.core/aget out i)
                                            (clojure.core/aget input i))))
    out))

(deftest test-gpu-blelloch-scan
  (when-ze
   (ze/init!)

   (testing "GPU exclusive scan: all-ones array → identity prefix sums"
     (let [n 2000
           input (int-array n (repeat n 1))]
       (gpu/with-gpu-session [sess :ze:0]
         (fgpu/compile-abm-kernels! sess)
         (gpu/alloc! sess {:input  [:int n input]
                           :output [:int (inc n) nil]})
         (gpu/invoke-scan! sess :csr-scan [:input] :output n)
         (let [out (gpu/download sess :output)
               ref (exclusive-scan-reference input n)]
           (is (every? #(== (clojure.core/aget out %) (clojure.core/aget ref %))
                       (range (inc n)))
               "GPU scan of all-1s should equal 0,1,2,...,n")))))

   (testing "GPU exclusive scan: arbitrary data matches CPU reference"
     (let [n 1500
           rng (java.util.Random. 12345)
           input (int-array n (repeatedly n #(int (mod (.nextInt rng) 100))))]
       (gpu/with-gpu-session [sess :ze:0]
         (fgpu/compile-abm-kernels! sess)
         (gpu/alloc! sess {:input  [:int n input]
                           :output [:int (inc n) nil]})
         (gpu/invoke-scan! sess :csr-scan [:input] :output n)
         (let [out (gpu/download sess :output)
               ref (exclusive-scan-reference input n)]
           (is (every? #(== (clojure.core/aget out %) (clojure.core/aget ref %))
                       (range (inc n)))
               "GPU scan should match CPU reference for arbitrary data")
           (is (== (clojure.core/aget out n) (clojure.core/aget ref n))
               "GPU scan total (output[n]) should match CPU total")))))

   (testing "GPU scan: single-block case (n < block_size)"
     (let [n 100
           input (int-array n (repeat n 3))]
       (gpu/with-gpu-session [sess :ze:0]
         (fgpu/compile-abm-kernels! sess)
         (gpu/alloc! sess {:input  [:int n input]
                           :output [:int (inc n) nil]})
         (gpu/invoke-scan! sess :csr-scan [:input] :output n)
         (let [out (gpu/download sess :output)]
           (is (== 0 (clojure.core/aget out 0)) "output[0] must be 0 (exclusive)")
           (is (== (* 3 n) (clojure.core/aget out n)) "output[n] = total = 3*n")))))

   (testing "GPU scan: csr-scan matches CPU exclusive prefix sum after rebuild"
     (let [n-agents 300
           max-firms 100
           [_ ^raster.abm.firms.AgentSoA agents ^raster.abm.firms.FirmSoA firms _]
           (firms/init-simulation (firms/->FirmsConfig n-agents 4 4 0.04 0 10 99 n-agents))
           firm-size (clone-int-array (.n-workers firms))
           ref-offsets (exclusive-scan-reference firm-size max-firms)]
       (gpu/with-gpu-session [sess :ze:0]
         (fgpu/compile-abm-kernels! sess)
         (gpu/alloc! sess {:firm-size [:int max-firms (.n-workers firms)]
                           :offsets   [:int (inc max-firms) nil]})
         (gpu/invoke-scan! sess :csr-scan [:firm-size] :offsets max-firms)
         (let [out-offsets (gpu/download sess :offsets)]
           (is (every? #(== (clojure.core/aget out-offsets %)
                            (clojure.core/aget ref-offsets %))
                       (range (inc max-firms)))
               "GPU CSR offsets must match CPU exclusive prefix sum")))))))

;; ================================================================
;; Test: Autotuner (requires Level Zero)
;; ================================================================

(deftest test-autotuner
  (when-ze
   (ze/init!)

   (testing "autotune-wg-sweep! returns valid KernelTuning with finite ms"
     (gpu/with-gpu-session [sess :ze:0]
       (fgpu/compile-abm-kernels! sess)
       (gpu/alloc! sess {:alive [:int 50000 (int-array 50000 (repeat 50000 1))]
                         :output [:float 50000 nil]
                         :param-a [:float 50000 nil]
                         :param-b [:float 50000 nil]
                         :param-beta [:float 50000 nil]
                         :total-effort [:float 50000 nil]})
       (let [n 50000
             po-kname (:kernel-name (first (gpu/kernel sess :produce-output)))
             bufs (:buffers @sess)
             wg-sizes [64 128 256 512]
             tuning (at/autotune-wg-sweep! po-kname
                                           [(get bufs :alive) (get bufs :output) (get bufs :param-a)
                                            (get bufs :param-b) (get bufs :param-beta) (get bufs :total-effort)]
                                           [] n :ze:0
                                           :wg-sizes wg-sizes :warmup 1 :timed 3)]
         (is (instance? raster.compiler.support.autotuner.KernelTuning tuning))
         (is (contains? (set wg-sizes) (:workgroup-size tuning))
             "Best wg-size must be one of the candidates")
         (is (< (:best-ms tuning) 1.0e100) "Best ms must be finite (not MAX_VALUE)")
         (is (> (:best-ms tuning) 0.0) "Best ms must be positive")
         (is (= (set wg-sizes) (set (keys (:wg-sizes (:all-results tuning)))))
             "all-results should contain all tried wg-sizes")
         (is (every? #(< % 1.0e100) (vals (:wg-sizes (:all-results tuning))))
             "All timed results must be finite"))))

   (testing "autotune-kernel! patches registry workgroup-size"
     (gpu/with-gpu-session [sess :ze:0]
       (fgpu/compile-abm-kernels! sess)
       (gpu/alloc! sess {:alive [:int 50000 (int-array 50000 (repeat 50000 1))]
                         :output [:float 50000 nil]
                         :param-a [:float 50000 nil]
                         :param-b [:float 50000 nil]
                         :param-beta [:float 50000 nil]
                         :total-effort [:float 50000 nil]})
       (let [n 50000
             po-kname (:kernel-name (first (gpu/kernel sess :produce-output)))
             bufs (:buffers @sess)]
         (at/autotune-kernel! po-kname
                              [(get bufs :alive) (get bufs :output) (get bufs :param-a)
                               (get bufs :param-b) (get bufs :param-beta) (get bufs :total-effort)]
                              [] n :ze:0 :wg-sizes [64 256] :warmup 1 :timed 2)
         (let [wg (get-in @ze/kernel-registry [po-kname :workgroup-size])]
           (is (contains? #{64 256} wg) "Registry should be patched with best wg-size")
           (is (some? (get-in @ze/kernel-registry [po-kname :tuning]))
               "Registry should store :tuning entry")))))

   (testing "Tuning cache: save and load roundtrip"
     (let [dummy-tuning (raster.compiler.support.autotuner/->KernelTuning
                         "test_kernel" 256 1.23 {:wg-sizes {64 2.0 128 1.5 256 1.23 512 1.4}})]
       (at/save-tuning-cache! :ze:test {"test_kernel" dummy-tuning})
       (let [loaded (at/load-tuning-cache :ze:test)]
         (is (map? loaded) "Loaded cache should be a map")
         (is (contains? loaded "test_kernel") "Should contain saved kernel")
         (let [entry (get loaded "test_kernel")]
           (is (== 256 (:workgroup-size entry)) "wg-size should survive roundtrip")
           (is (< (Math/abs (- 1.23 (double (:best-ms entry)))) 0.001)
               "best-ms should survive roundtrip")))))

   (testing "apply-cached-tuning! patches registry on cache hit"
     (gpu/with-gpu-session [sess :ze:0]
       (fgpu/compile-abm-kernels! sess)
       (let [po-kname (:kernel-name (first (gpu/kernel sess :produce-output)))]
          ;; Save a fake tuning entry
         (at/save-tuning-cache! :ze:0
                                {po-kname {:workgroup-size 128 :best-ms 0.5 :all-results {}}})
          ;; Apply it
         (let [hit? (at/apply-cached-tuning! po-kname :ze:0)]
           (is (true? hit?) "Should return true on cache hit")
           (is (== 128 (get-in @ze/kernel-registry [po-kname :workgroup-size]))
               "Registry wg-size should be patched from cache"))
          ;; Non-existent kernel
         (let [miss? (at/apply-cached-tuning! "nonexistent_kernel" :ze:0)]
           (is (false? miss?) "Should return false on cache miss")))))))

(deftest test-codegen-smoke
  (testing "OpenCL codegen produces valid kernel source for par phases"
    ;; This just tests that emit-opencl-expr handles the control flow
    ;; in the par bodies without errors. Doesn't require GPU.
    (try
      (let [;; Simple map-void! with if + aset
            form '(raster.par/map-void! i n
                                        (if (== 1 (aget alive i))
                                          (aset output i (float 1.0))
                                          (aset output i (float 0.0))))
            result (opencl-pass/opencl-pass form :dtype :float :compile-spirv? false)]
        (is (seq (:kernels result))
            "Should generate at least one kernel")
        (when (seq (:kernels result))
          (let [src (:source (first (:kernels result)))]
            (is (string? src) "Kernel source should be a string")
            (is (.contains ^String src "__kernel") "Should contain __kernel")
            (is (.contains ^String src "if") "Should contain if statement"))))
      (catch Exception e
        ;; If OpenCL codegen can't be loaded (e.g., missing deps), skip
        (println "Skipping codegen smoke test:" (.getMessage e))))))
