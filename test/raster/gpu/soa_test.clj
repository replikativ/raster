(ns raster.gpu.soa-test
  "Hardware-free typed SoA emission and CPU display tests. Production resident composites
  are exercised on both device backends by public-aggregate-test."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [clojure.string :as str]
            [raster.core :refer [defvalue deftm]]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.runtime.display :as display])
  (:import [java.lang.foreign MemorySegment ValueLayout]))


;; ================================================================
;; Test types
;; ================================================================

(defvalue TestParticle [x :- Float, y :- Float, vx :- Float, vy :- Float])

(deftest legacy-ze-raw-composite-constructors-are-retired
  (require 'raster.gpu.ze-runtime)
  (doseq [name '[GpuSoA ->GpuSoA map->GpuSoA gpu-soa? gpu-array gpu-array-device
                 n-elements copy-to-gpu! copy-from-gpu!]]
    (is (nil? (ns-resolve 'raster.gpu.ze-runtime name))
        "SoA leaves use canonical session buffers, not a raw backend composite allocator")))

;; ================================================================
;; Phase 1: SoA OpenCL kernel generation
;; ================================================================

(defn- tag-sym
  "Add :tag metadata to a symbol."
  [sym tag]
  (with-meta sym {:tag tag}))

(defn- tag-body
  "Walk body and add :tag metadata to symbols matching tag-map."
  [body tag-map]
  (walk/postwalk
   (fn [f]
     (if (and (symbol? f) (contains? tag-map f))
       (tag-sym f (get tag-map f))
       f))
   body))

(deftest soa-kernel-no-struct-typedef-test
  (testing "SoA kernel is fully scalar-replaced — no struct typedef, no struct ops"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(aset particles i (aget particles i)))
                {'particles 'TestParticleSoA})
          result (opencl-pass/opencl-pass body :dtype :float)
          source (:source (first (:kernels result)))]
      (is (some? source))
      ;; SROA pass eliminates the value type before the C emitter sees it
      (is (not (str/includes? source "typedef struct")))
      (is (not (str/includes? source "TestParticle")))
      ;; aget->aset roundtrip lowers to per-field array copies
      (is (re-find #"particles_x\[.*\] = rstr_map_load_" source))
      (is (re-find #"particles_vy\[.*\] = rstr_map_load_" source)))))

(deftest soa-kernel-flat-params-test
  (testing "SoA arrays decompose into flat __global pointers"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(let* [p (aget particles i)]
                             (aset particles i p)))
                {'particles 'TestParticleSoA})
          result (opencl-pass/opencl-pass body :dtype :float)
          source (:source (first (:kernels result)))]
      (is (str/includes? source "__global float* particles_x"))
      (is (str/includes? source "__global float* particles_y"))
      (is (str/includes? source "__global float* particles_vx"))
      (is (str/includes? source "__global float* particles_vy"))
      ;; Should NOT have a single particles param
      (is (not (re-find #"__global float\* particles[^_]" source))))
    (testing "physical field slots retain one logical resident composite binding"
      (let [body (tag-body
                  (list 'raster.par/map-void! 'i 'n
                        '(let* [p (aget particles i)]
                               (aset particles i p)))
                  {'particles 'TestParticleSoA})
            kernel (first (:kernels (opencl-pass/opencl-pass body :dtype :float)))
            abi (:abi kernel)]
        (is (= '#{particles_x particles_y particles_vx particles_vy}
               (set (map :name (kabi/pointer-slots abi)))))
        (is (= '[particles] (kabi/pointer-binding-names abi)))
        (is (= :kernel-body (kart/emission-route kernel)))))))

(deftest soa-kernel-aget-field-projects-test
  (testing "SoA aget + field projection scalar-replaces to the per-field array read"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(let* [p (aget particles i)]
                             (aset out i (.x p))))
                {'particles 'TestParticleSoA 'out 'floats})
          result (opencl-pass/opencl-pass body :dtype :float)
          source (:source (first (:kernels result)))]
      ;; (.x (aget particles i)) → particles_x[idx], no struct literal
      (is (str/includes? source "particles_x["))
      (is (not (str/includes? source "(TestParticle)"))))))

(deftest soa-kernel-aset-fieldwise-test
  (testing "SoA aset decomposes into field-by-field writes (no temp struct)"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(aset particles i (->TestParticle 1.0 2.0 3.0 4.0)))
                {'particles 'TestParticleSoA})
          result (opencl-pass/opencl-pass body :dtype :float)
          source (:source (first (:kernels result)))]
      (is (not (str/includes? source "_soa_tmp")))
      (is (re-find #"particles_x\[.*\] = 1\.0f;" source))
      (is (re-find #"particles_vy\[.*\] = 4\.0f;" source)))))

(deftest soa-kernel-constructor-scalar-replaced-test
  (testing "->Type construction in aset scalar-replaces to per-field stores"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(aset particles i (->TestParticle 0.0 0.0 1.0 1.0)))
                {'particles 'TestParticleSoA})
          result (opencl-pass/opencl-pass body :dtype :float)
          source (:source (first (:kernels result)))]
      ;; No struct constructor of any form survives the SROA pass
      (is (not (str/includes? source "TestParticle")))
      (is (re-find #"particles_x\[.*\] = 0\.0f;" source))
      (is (re-find #"particles_vx\[.*\] = 1\.0f;" source)))))

(deftest soa-kernel-field-access-test
  (testing ".field on a value-type local projects to the per-field array (no struct access)"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(let* [p (aget particles i)]
                             (aset out i (.x p))))
                {'particles 'TestParticleSoA 'out 'floats})
          result (opencl-pass/opencl-pass body :dtype :float)
          source (:source (first (:kernels result)))]
      ;; .x of the SoA-bound local resolves to the flat field array read
      (is (str/includes? source "particles_x["))
      (is (not (str/includes? source ").x"))))))

(deftest typed-soa-kernel-needs-no-emitter-expansion-metadata
  (testing "shared scalar replacement leaves no backend-local SoA expansion contract"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(aset particles i (aget particles i)))
                {'particles 'TestParticleSoA})
          result (opencl-pass/opencl-pass body :dtype :float)
          kernel (first (:kernels result))]
      (is (= :kernel-body (kart/emission-route kernel)))
      (is (nil? (kart/attribute kernel :soa-expansions))))))

;; ================================================================
;; Phase 3: canonical frontend/backend boundary
;; ================================================================

(deftm test-scale [^double x ^double factor] :- Double
  (* x factor))

(deftest scheduled-backend-does-not-inline-deftm-calls-test
  (testing "raw deftm calls must be normalized by the shared frontend before scheduling"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      (list 'aset 'out 'i
                            (list 'raster.gpu.soa-test/test-scale '(aget arr i) 2.5)))
                {'out 'doubles 'arr 'doubles})
          error (try
                  (opencl-pass/opencl-pass body :dtype :double)
                  nil
                  (catch clojure.lang.ExceptionInfo e e))]
      (is (= :scalar-expression (:missing-rule (ex-data error))))
      (is (= 'raster.gpu.soa-test/test-scale
             (first (:expression (ex-data error))))))))

;; ================================================================
;; Phase 4: Display module
;; ================================================================

(deftest render-buffer-test
  (testing "render-buffer creates correct structure"
    (let [buf (display/render-buffer 320 240)]
      (is (= 320 (:width buf)))
      (is (= 240 (:height buf)))
      (is (= (* 320 240) (alength ^ints (:pixels buf))))
      (is (some? (:seg buf))))))

(deftest pack-rgba-test
  (testing "pack-rgba returns correct ARGB values"
    ;; All zeros
    (is (= 0 (display/pack-rgba 0.0 0.0 0.0 0.0)))
    ;; Full red with full alpha = 0xFFFF0000
    (is (= 0xFFFF0000 (display/pack-rgb 1.0 0.0 0.0)))
    ;; Full green = 0xFF00FF00
    (is (= 0xFF00FF00 (display/pack-rgb 0.0 1.0 0.0)))
    ;; Full blue = 0xFF0000FF
    (is (= 0xFF0000FF (display/pack-rgb 0.0 0.0 1.0)))
    ;; Clamps to [0,1]
    (is (= (display/pack-rgb 1.0 1.0 1.0) (display/pack-rgba 2.0 2.0 2.0 2.0)))))

(deftest sync-from-gpu-test
  (testing "sync-from-gpu! copies seg to pixels for gpu-backed buffers"
    ;; For CPU render-buffer, seg wraps the same int[] → writes are immediate.
    ;; sync-from-gpu! is designed for render-buffer-gpu where seg is separate.
    ;; Test that sync copies from a separate MemorySegment into pixels.
    (let [w 4 h 4 n (* w h)
          ;; Simulate a GPU-backed buffer: separate seg and pixels
          pixels  (int-array n)
          seg     (MemorySegment/ofArray (int-array n))
          buf     (display/->RenderBuffer w h pixels seg)]
      ;; Write to seg (separate from pixels)
      (.set ^MemorySegment seg ValueLayout/JAVA_INT (long 0) (int 42))
      ;; Before sync, pixels should be 0
      (is (= 0 (aget ^ints pixels 0)))
      ;; After sync, pixels should reflect seg
      (display/sync-from-gpu! buf)
      (is (= 42 (aget ^ints pixels 0))))))
