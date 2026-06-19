(ns raster.gpu.soa-test
  "Tests for GPU SoA features: GpuSoA type, OpenCL SoA kernel generation,
  deftm inlining, display module, and copy round-trips.

  GPU-dependent tests are gated behind when-ze."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [clojure.string :as str]
            [raster.core :refer [defvalue deftm]]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.runtime.display :as display])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

;; ================================================================
;; GPU availability check (mirrors ze_runtime_test)
;; ================================================================

(defn- ze-available? []
  (try
    (require 'raster.gpu.ze-runtime)
    (let [query-fn (resolve 'raster.gpu.ze-runtime/query-devices)]
      (and query-fn (seq (query-fn))))
    (catch Exception _ false)))

(defmacro when-ze [& body]
  `(if (ze-available?)
     (do ~@body)
     (println "  [SKIP] No Level Zero GPU available")))

;; ================================================================
;; Test types
;; ================================================================

(defvalue TestParticle [x :- Float, y :- Float, vx :- Float, vy :- Float])
(defvalue TestVec2 [a :- Double, b :- Double])

;; ================================================================
;; Phase 2: GpuSoA allocation and copy
;; ================================================================

(deftest gpu-soa-allocation-test
  (when-ze
   (testing "gpu-array creates GpuSoA with correct fields"
     (require 'raster.gpu.ze-runtime)
     (let [gpu-array   (resolve 'raster.gpu.ze-runtime/gpu-array)
           gpu-soa?    (resolve 'raster.gpu.ze-runtime/gpu-soa?)
           n-elements  (resolve 'raster.gpu.ze-runtime/n-elements)
           g           (gpu-array 'TestParticle 256)]
       (is (gpu-soa? g))
       (is (= 256 (n-elements g)))
       (is (= 'TestParticle (:scalar-tag g)))
       (is (= 'TestParticleSoA (:soa-tag g)))
       (is (= 4 (count (:field-segs g))))
       (is (= ["x" "y" "vx" "vy"] (mapv :name (:field-segs g))))
       (is (every? #(= :float (:dtype %)) (:field-segs g)))))))

(deftest gpu-soa-device-allocation-test
  (when-ze
   (testing "gpu-array-device creates device-only GpuSoA"
     (require 'raster.gpu.ze-runtime)
     (let [gpu-array-device (resolve 'raster.gpu.ze-runtime/gpu-array-device)
           g (gpu-array-device 'TestParticle 128)]
       (is ((resolve 'raster.gpu.ze-runtime/gpu-soa?) g))
       (is (= 128 ((resolve 'raster.gpu.ze-runtime/n-elements) g)))))))

(deftest gpu-soa-double-fields-test
  (when-ze
   (testing "gpu-array works with double fields"
     (require 'raster.gpu.ze-runtime)
     (let [g ((resolve 'raster.gpu.ze-runtime/gpu-array) 'TestVec2 64)]
       (is (= 'TestVec2 (:scalar-tag g)))
       (is (= ["a" "b"] (mapv :name (:field-segs g))))
       (is (every? #(= :double (:dtype %)) (:field-segs g)))))))

(deftest gpu-soa-copy-roundtrip-test
  (when-ze
   (testing "copy-to-gpu! and copy-from-gpu! preserve data"
     (require 'raster.gpu.ze-runtime)
     (let [gpu-array    (resolve 'raster.gpu.ze-runtime/gpu-array)
           copy-to-gpu  (resolve 'raster.gpu.ze-runtime/copy-to-gpu!)
           copy-from-gpu (resolve 'raster.gpu.ze-runtime/copy-from-gpu!)
           n  16
            ;; Create and fill a JVM SoA
           soa1 (make-test-particle-soa n)
           _    (do (aset (.x soa1) 0 (float 1.5))
                    (aset (.x soa1) 5 (float -3.0))
                    (aset (.y soa1) 0 (float 10.0))
                    (aset (.vx soa1) 3 (float 42.0))
                    (aset (.vy soa1) 15 (float 99.0)))
            ;; Round-trip through GPU
           g    (gpu-array 'TestParticle n)
           _    (copy-to-gpu g soa1)
           soa2 (make-test-particle-soa n)
           _    (copy-from-gpu g soa2)]
       (is (= 1.5  (aget (.x soa2) 0)))
       (is (= -3.0 (aget (.x soa2) 5)))
       (is (= 10.0 (aget (.y soa2) 0)))
       (is (= 42.0 (aget (.vx soa2) 3)))
       (is (= 99.0 (aget (.vy soa2) 15)))
        ;; Untouched elements should be 0.0
       (is (= 0.0 (aget (.x soa2) 1)))))))

(deftest gpu-soa-unknown-type-test
  (testing "gpu-array throws for unregistered types"
    (when-ze
     (require 'raster.gpu.ze-runtime)
     (is (thrown-with-msg? clojure.lang.ExceptionInfo #"No SoA registered"
                           ((resolve 'raster.gpu.ze-runtime/gpu-array) 'NonExistentType 10))))))

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
      (is (str/includes? source "particles_x[idx] = particles_x[idx];"))
      (is (str/includes? source "particles_vy[idx] = particles_vy[idx];")))))

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
      (is (not (re-find #"__global float\* particles[^_]" source))))))

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
      (is (str/includes? source "particles_x[idx]"))
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
      (is (str/includes? source "particles_x[idx] = 1.0"))
      (is (str/includes? source "particles_vy[idx] = 4.0")))))

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
      (is (str/includes? source "particles_x[idx] = 0.0"))
      (is (str/includes? source "particles_vx[idx] = 1.0")))))

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
      (is (str/includes? source "particles_x[idx]"))
      (is (not (str/includes? source ").x"))))))

(deftest soa-expansions-in-kernel-result-test
  (testing "Kernel result includes soa-expansions map"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      '(aset particles i (aget particles i)))
                {'particles 'TestParticleSoA})
          result (opencl-pass/opencl-pass body :dtype :float)
          kernel (first (:kernels result))]
      (is (some? (:soa-expansions kernel)))
      (is (contains? (:soa-expansions kernel) 'particles)))))

;; ================================================================
;; Phase 3: deftm inlining in OpenCL kernels
;; ================================================================

(deftm test-scale [^double x ^double factor] :- Double
  (* x factor))

(deftm test-add3 [^double a ^double b ^double c] :- Double
  (+ a (+ b c)))

(deftest deftm-inline-basic-test
  (testing "deftm call gets inlined into OpenCL kernel"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      (list 'aset 'out 'i
                            (list 'raster.gpu.soa-test/test-scale '(aget arr i) 2.5)))
                {'out 'doubles 'arr 'doubles})
          result (opencl-pass/opencl-pass body :dtype :double)
          source (:source (first (:kernels result)))]
      ;; Should inline the body, not emit a function call
      (is (not (str/includes? source "test_scale(")))
      ;; Should have the multiplication
      (is (str/includes? source "*")))))

(deftest deftm-inline-3arg-test
  (testing "3-arg deftm inlining"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      (list 'aset 'out 'i
                            (list 'raster.gpu.soa-test/test-add3
                                  '(aget a i) '(aget b i) '(aget c i))))
                {'out 'doubles 'a 'doubles 'b 'doubles 'c 'doubles})
          result (opencl-pass/opencl-pass body :dtype :double)
          source (:source (first (:kernels result)))]
      ;; Should inline as addition, not function call
      (is (not (str/includes? source "test_add3(")))
      (is (str/includes? source "+")))))

(deftest deftm-inline-no-spurious-scalars-test
  (testing "Inlined deftm doesn't leave function name as scalar param"
    (let [body (tag-body
                (list 'raster.par/map-void! 'i 'n
                      (list 'aset 'out 'i
                            (list 'raster.gpu.soa-test/test-scale '(aget arr i) 2.5)))
                {'out 'doubles 'arr 'doubles})
          result (opencl-pass/opencl-pass body :dtype :double)
          source (:source (first (:kernels result)))]
      ;; Should NOT have test_scale as a kernel parameter
      (is (not (str/includes? source "double test_scale"))))))

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
