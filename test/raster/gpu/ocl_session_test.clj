(ns raster.gpu.ocl-session-test
  "Linked resident execution on OpenCL: compile → certify → instantiate → replay → download,
  on whatever OpenCL device is available. With
  RASTER_OCL_DEVICE_TYPE=cpu this runs on POCL/Intel-CPU — the no-GPU
  vendor-portability oracle (CUDA/HIP Phase A). Skips cleanly without OpenCL.

  Portability lesson encoded here: kernels must not read from :output-only
  buffers (uninitialized device memory — Intel's driver zeroes allocations,
  POCL's malloc does not)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.pipeline :as pl]
            [raster.compiler.equation-first :as equation-first]
            [raster.arrays :as ra]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as ops]
            [raster.dl.nn :as nn]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as device-probe]
            [raster.gpu.descriptor-fixture :as fixture]))

(deftest opencl-max-work-group-size-selector
  (require 'raster.gpu.ocl-runtime)
  (is (= 0x1004
         (var-get
          (ns-resolve 'raster.gpu.ocl-runtime 'CL_DEVICE_MAX_WORK_GROUP_SIZE)))
      "CL_DEVICE_MAX_WORK_GROUP_SIZE must not regress to the dimensions selector 0x1003"))

(deftm ocl-session-ax! (All [T] [x :- (Array T) y :- (Array T) a :- T n :- Long] :- Void
                            (raster.par/map-void! i n (ra/aset y i (* a (ra/aget x i))))))

(deftm ocl-session-map
  [x :- (Array float) y :- (Array float) a :- Float n :- Long] :- (Array float)
  (raster.par/map! y i n float (* a (ra/aget x i))))

(deftm ocl-session-inout
  [state :- (Array float) a :- Float n :- Long] :- (Array float)
  (raster.par/map! state i n float (* a (ra/aget state i))))

(deftm ocl-session-negate!
  [x :- (Array float) out :- (Array float) n :- Long] :- Void
  (raster.par/map-void! i n (ra/aset out i (- (ra/aget x i)))))

(deftm ocl-session-long-state
  [out :- (Array long) state :- Long n :- Long] :- (Array long)
  (raster.par/map! out i n long (unchecked-add state (long i))))

(deftm ocl-session-checked-count
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/map! out i (int n) float (ra/aget x i)))

(deftm ocl-session-checked-rng
  [seeds :- (Array long) n :- Long seed :- Long] :- (Array long)
  (raster.par/rng-fill! seeds n (int seed)))

(deftest public-rng-capture-checks-precede-allocation
  (let [descriptor (pl/compile-gpu-program #'ocl-session-checked-rng :ocl:0 :dtype :float)
        step (first (:steps descriptor))
        scalars (filterv #(= :scalar (:kind %)) (:argument-specs step))]
    (is (= [:long :long :int] (mapv :dtype (:abi (:artifact step)))))
    (is (= [-1 3] (mapv #((:value-fn %) [nil 3 -1]) scalars)))
    (doseq [value [(inc (long Integer/MAX_VALUE)) (dec (long Integer/MIN_VALUE))]
            position [1 2]]
      (is (thrown? ArithmeticException
                   ((:value-fn (first scalars)) (assoc [nil 3 0] position value)))))
    (is (thrown? NullPointerException
                 ((:value-fn (first scalars)) [nil nil (inc (long Integer/MAX_VALUE))]))
        "the count conversion still precedes the seed conversion")))

(deftest ocl-checked-rng-captures-match-source-values
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "checked RNG captures")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-checked-rng :ocl:0 :dtype :float)
          s (gpu/make-session :ocl:0)]
      (try
        (doseq [seed [Integer/MIN_VALUE Integer/MAX_VALUE]]
          (let [expected (long-array 17)
                arguments [(long-array 17) 17 seed]
                program (fixture/instantiate! s descriptor arguments)]
            (try
              (raster.par/rng-fill! expected 17 (int seed))
              (is (= (vec expected) (vec (get (fixture/run! program arguments) 'seeds))))
              (finally (fixture/close! program)))))
        (finally (gpu/close-session! s))))))

(deftm ocl-session-fused-buffer-maps
  [x :- (Array float) left :- (Array float) right :- (Array float) n :- Long] :- Object
  (let [a (raster.par/map! left i n float (+ (ra/aget x i) 1.0))
        b (raster.par/map! right j n float (* (ra/aget x j) 2.0))]
    [a b]))

(deftest public-fused-buffer-maps-remain-a-flat-resident-program
  (let [descriptor (pl/compile-gpu-program #'ocl-session-fused-buffer-maps :ocl:0 :dtype :float)]
    (is (= [:map-void] (mapv :convention (:steps descriptor))))
    (is (= '[left right] (:result-sym descriptor)))))

(deftest public-checked-count-refuses-overflow-before-allocation
  (let [descriptor (pl/compile-gpu-program #'ocl-session-checked-count :ocl:0 :dtype :float)
        bound (last (:argument-specs (first (:steps descriptor))))]
    (is (= :int (:type bound)))
    (is (= 3 ((:value-fn bound) [nil nil 3])))
    (doseq [n [(inc (long Integer/MAX_VALUE)) (dec (long Integer/MIN_VALUE))]]
      (is (thrown? ArithmeticException ((:value-fn bound) [nil nil n]))))))

(deftest ocl-public-checked-count-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "public checked-count map")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-checked-count :ocl:0 :dtype :float)
          s (gpu/make-session :ocl:0)
          arguments [(float-array [3 5 7]) (float-array 3) (long 3)]]
      (try
        (let [program (fixture/instantiate! s descriptor arguments)]
          (try
            (is (= [3.0 5.0 7.0] (vec (get (fixture/run! program arguments) 'out))))
            (finally (fixture/close! program))))
        (finally (gpu/close-session! s))))))

(deftest public-compiler-entry-points-preserve-declared-long-scalars
  (let [emitted (equation-first/compile #'ocl-session-long-state
                                       {:target :ocl:0 :dtype :float})
        resident (pl/compile-gpu-program #'ocl-session-long-state :ocl:0 :dtype :float)
        scalar-dtypes #(mapv :dtype (filter (fn [slot] (= :scalar (:kind slot))) (:abi %)))]
    (is (= [[:long :long]] (mapv scalar-dtypes (:kernels emitted))))
    (is (= {:kernel-body 1}
           (get-in (pl/compile-report #'ocl-session-long-state
                                      :target-device :ocl:0 :dtype :float)
                   [:emission :routes])))
    (is (= [:map] (mapv :convention (:steps resident))))))

(deftest ocl-session-long-scalars-survive-session-and-resident-binding
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "declared Long scalar state")
    (let [s (gpu/make-session :ocl:0)]
      (try
        (let [artifacts (gpu/compile! s :long-state #'ocl-session-long-state
                                      {:dtype :float :min-elements 0})
              invoke! (requiring-resolve 'raster.gpu.ocl-runtime/invoke-registered-kernel)
              descriptor (pl/compile-gpu-program #'ocl-session-long-state :ocl:0 :dtype :float)
              out (long-array 3)]
          (is (= [[:long :long :long]] (mapv #(mapv :dtype (:abi %)) artifacts)))
          (gpu/alloc! s {:state [:long 3 nil]})
          (doseq [state [4294967297 Long/MIN_VALUE Long/MAX_VALUE]]
            (let [expected (mapv #(unchecked-add (long state) (long %)) (range 3))]
              (invoke! (:kernel-name (first artifacts)) [] (gpu/buffer s :state)
                       [{:type :long :value state}] 3)
              (is (= expected (vec (gpu/download s :state))))
              (let [program (fixture/instantiate! s descriptor [out state 3])]
                (try
                  (is (= expected (vec (get (fixture/run! program [out state 3]) 'out))))
                  (finally (fixture/close! program)))))))
        (finally (gpu/close-session! s))))))

(deftest scale-clamp-exp-uses-the-common-scalar-body
  (let [report (pl/compile-report #'ops/scale-clamp-exp
                                  :target-device :ocl:0 :dtype :double)]
    (is (= :typed-soac (get-in report [:route :source-dialect])))
    (is (= {:kernel-body 1} (get-in report [:emission :routes])))
    (is (empty? (get-in report [:emission :declines])))))

(deftest sum-kv-heads-keeps-an-ordered-nonzero-origin-body
  (let [report (pl/compile-report #'ops/sum-kv-heads :target-device :ocl:0 :dtype :float)]
    (is (= {:kernel-body 1} (get-in report [:emission :routes])))
    (is (empty? (get-in report [:emission :declines])))))

(deftest ocl-ordered-head-fan-in-preserves-initial-load-and-association
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "ordered head fan-in")
    (let [descriptor (pl/compile-gpu-program #'ops/sum-kv-heads :ocl:0 :dtype :float)
          session (gpu/make-session :ocl:0)]
      (try
        (doseq [group [1 3 5]]
          (let [n-kv 2 slab 3
                input (float-array
                       (for [head (range n-kv) r (range group) col (range slab)]
                         (if (= group 1) -0.0
                             (case r 0 1.0e20 1 -1.0e20 (+ 3.0 head col)))))
                expected (float-array
                          (for [head (range n-kv) col (range slab)]
                            (reduce (fn [acc r]
                                      (float (+ acc (aget input (+ (* (+ (* head group) r) slab) col)))))
                                    (aget input (+ (* head group slab) col))
                                    (range 1 group))))
                arguments [input (long n-kv) (long group) (long slab)]
                program (fixture/instantiate! session descriptor arguments)]
            (try
              (let [result (get (fixture/run! program arguments) (:result-sym descriptor))]
                (is (= (mapv #(Float/floatToRawIntBits %) expected)
                       (mapv #(Float/floatToRawIntBits %) result))))
              (finally (fixture/close! program)))))
        (finally (gpu/close-session! session))))))

(deftest ocl-unary-negation-preserves-ieee-signs
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "generated scalar negation")
    (let [x (float-array [0.0 -0.0 3.5 -3.5 Float/POSITIVE_INFINITY
                          Float/NEGATIVE_INFINITY Float/NaN])
          out (float-array (alength x))
          descriptor (pl/compile-gpu-program #'ocl-session-negate! :ocl:0 :dtype :float)
          session (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! session descriptor [x out (long (alength x))])
              result (get (fixture/run! program [x out (long (alength x))]) 'out)]
          (is (every? #(= :kernel-body (get-in % [:artifact :attributes :emission-route]))
                      (:steps descriptor)))
          (is (= (mapv #(bit-xor Integer/MIN_VALUE (Float/floatToRawIntBits %)) (take 6 x))
                 (mapv #(Float/floatToRawIntBits %) (take 6 result))))
          (is (Float/isNaN (aget ^floats result 6))))
        (finally (gpu/close-session! session))))))

(deftm ocl-session-map2
  [x :- (Array float) y :- (Array float)
   a :- (Array float) b :- (Array float) n :- Long] :- Void
  (raster.par/map2! a b i n float
                    (+ (ra/aget x i) 1.0)
                    (* (ra/aget y i) 2.0)))

(deftm ocl-session-effect-mixed
  [x :- (Array float) q :- (Array byte)
   y :- (Array float) labels :- (Array int) n :- Long] :- Void
  (raster.par/map-void!
   i n
   (do (ra/aset y i (float (* (ra/aget x i) 2.0)))
       (ra/aset labels i (int (+ (int (ra/aget q i)) 7))))))

(deftm ocl-session-effect-shared-local
  [x :- (Array float) a :- (Array float) b :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   i n
   (let [shifted (float (+ (ra/aget x i) 1.0))
         squared (float (* shifted shifted))]
     (ra/aset a i shifted)
     (ra/aset b i squared))))

(deftm ocl-session-contract
  [A :- (Array float) B :- (Array float)] :- (Array float)
  (let [C (ra/alloc-like A 64)]
    (raster.par/contract C [[i 8] [j 8]] [[l 8]]
                         (clojure.core/*
                          (ra/aget A (clojure.core/+ (clojure.core/* i 8) l))
                          (ra/aget B (clojure.core/+ (clojure.core/* l 8) j))))
    C))

(deftm ocl-session-reduce
  [x :- (Array float) out :- (Array float) scale :- Double n :- Long] :- Void
  (let [s (raster.par/reduce acc 0.0 i n
                             (clojure.core/+ acc (clojure.core/* scale (ra/aget x i))))]
    (raster.par/map-void! j n
                          (ra/aset out j (clojure.core/* (ra/aget x j) s)))))

(deftm ocl-session-scan
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/scan out acc 0.0 i n float
                   (clojure.core/+ acc (ra/aget x i))))

(deftm ocl-session-exclusive-scan
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/scan-exclusive out acc 0.0 i n float
                             (clojure.core/+ acc (ra/aget x i))))

(deftm ocl-session-stencil
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/stencil!
   out [x] 1 :dirichlet float i n
   (clojure.core/+ (ra/aget x (clojure.core/dec i))
                   (ra/aget x (clojure.core/inc i)))))

(deftest ocl-map-void-mixed-storage-abi-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "mixed-storage map-void")
    (let [n 64
          q (byte-array (map #(byte (- (mod % 31) 15)) (range n)))
          out (int-array n)
          kernel (par-opencl/generate-par-map-void-kernel
                  '(raster.par/map-void! i n
                                         (aset out i (+ (int (aget q i)) limit)))
                  :dtype :float
                  :array-types {'out :int 'q :byte}
                  :scalar-types {'limit :int})
          register! (resolve 'raster.gpu.ocl-runtime/register-kernel!)
          invoke! (resolve 'raster.gpu.ocl-runtime/invoke-registered-map-void-kernel)]
      (register! (:kernel-name kernel) kernel)
      (invoke! (:kernel-name kernel) [out q] [{:type :int :value 7}] n)
      (is (every? true? (map-indexed (fn [i v] (= v (+ 7 (aget q i)))) out))))))

(deftest ocl-nested-effect-map-mixed-storage-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "nested typed effect map")
    (let [emitted (opencl-pass/opencl-pass
                   '(if enabled
                      (raster.par/map-void! i (int n)
                        (aset out i (+ (int (aget q i)) limit))) nil)
                   :device-id :ocl:0 :min-elements 0 :dtype :float
                   :array-types {'out :int 'q :byte}
                   :scalar-types {'enabled :boolean 'n :long 'limit :int})
          register! (resolve 'raster.gpu.ocl-runtime/register-kernel!)
          execute (eval (list 'fn '[enabled n q out limit]
                              (walk/postwalk #(if (symbol? %) (vary-meta % dissoc :tag) %)
                                             (:form emitted))))
          q (byte-array [-128 -3 0 127])
          out (int-array [99 99 99 99])]
      (is (= [:kernel-body] (mapv #(get-in % [:attributes :emission-route]) (:kernels emitted))))
      (doseq [kernel (:kernels emitted)] (register! (:kernel-name kernel) kernel))
      (is (nil? (execute false nil nil nil nil)))
      (is (nil? (execute true 4 q out 7)))
      (is (= [-121 4 7 134] (vec out))))))

(deftest ocl-raw-effect-map-retains-pre-store-snapshots
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "typed effect lexical snapshots")
    (let [emitted (opencl-pass/opencl-pass
                   '(raster.par/map-void! i n
                      (let* [^float next-state (+ (aget state i) (aget grad i))]
                        (aset state i (float next-state))
                        (aset param i (float (- (aget param i) next-state)))))
                   :device-id :ocl:0 :min-elements 0 :dtype :float
                   :array-types {'state :float 'grad :float 'param :float}
                   :scalar-types {'n :int})
          register! (resolve 'raster.gpu.ocl-runtime/register-kernel!)
          execute (eval (list 'fn '[state grad param n]
                              (walk/postwalk #(if (symbol? %) (vary-meta % dissoc :tag) %)
                                             (:form emitted))))
          state (float-array [1 2 3])
          grad (float-array [4 5 6])
          param (float-array [20 30 40])]
      (is (= [:kernel-body] (mapv #(get-in % [:attributes :emission-route]) (:kernels emitted))))
      (doseq [kernel (:kernels emitted)] (register! (:kernel-name kernel) kernel))
      (is (nil? (execute state grad param 3)))
      (is (= [5.0 7.0 9.0] (vec state)))
      (is (= [15.0 23.0 31.0] (vec param))))))

(deftest ocl-resident-session-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident session")
    (let [p (pl/compile-gpu-program #'ocl-session-ax! :ocl:0 :dtype :float)
          n 4096
          x (float-array (map float (range n)))
          y (float-array n)
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s p [x y (float 2.0) n]
                                            {'x :input 'y :output})]
          (testing "replayed program computes correctly"
            (let [r (fixture/run! program [x y (float 2.0) n])
                  ^floats yg (get r 'y)]
              (is (every? (fn [i] (< (Math/abs (- (aget yg (int i)) (* 2.0 i))) 1e-3))
                          (range n)))))
          (testing "replay is stable across repeated runs"
            (let [r2 (fixture/run! program [x y (float 2.0) n])
                  ^floats yg (get r2 'y)]
              (is (< (Math/abs (- (aget yg 100) 200.0)) 1e-3)))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-typed-effect-tuple-map-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed effect tuple map")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-map2 :ocl:0 :dtype :float)
          n 1024
          x (float-array (map float (range n)))
          y (float-array (map #(float (+ 10 %)) (range n)))
          a (float-array n)
          b (float-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= [:map-void] (mapv :convention (:steps descriptor)))
            "host control retains the nil-returning effect convention")
        (is (= :kernel-body (get-in descriptor [:steps 0 :artifact :provenance :dialect]))
            "kernel generation consumes the scheduled TypedSOAC SegMap through KernelBody")
        (is (= :segmap (get-in descriptor [:steps 0 :artifact :provenance :source-dialect])))
        (is (= [:input :input :output :output :scalar]
               (mapv :kind (get-in descriptor [:steps 0 :abi]))))
        (let [program (fixture/instantiate! session descriptor [x y a b n]
                                            {'x :input 'y :input
                                             'a :output 'b :output})
              result (fixture/run! program [x y a b n])
              ^floats actual-a (get result 'a)
              ^floats actual-b (get result 'b)]
          (is (every? (fn [i] (= (float (inc i)) (aget actual-a i)))
                      [0 1 255 256 1023]))
          (is (every? (fn [i] (= (float (* 2.0 (+ 10 i))) (aget actual-b i)))
                      [0 1 255 256 1023])))
        (finally (gpu/close-session! session))))))

(deftest ocl-resident-typed-effect-tuple-map-preserves-mixed-storage
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed mixed effect tuple map")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-effect-mixed
                                             :ocl:0 :dtype :float)
          n 257
          x (float-array (map #(float (/ % 8.0)) (range n)))
          q (byte-array (map #(byte (- (mod % 17) 8)) (range n)))
          y (float-array n)
          labels (int-array n)
          session (gpu/make-session :ocl:0)]
      (try
        ;; mixed byte/int/float storage lowers through the verified KernelBody: the int8 store
        ;; is a stated narrowing cast, not a source spelling the SegMap generator reparses
        (is (= :kernel-body (get-in descriptor [:steps 0 :artifact :provenance :dialect])))
        (is (= [:byte :float :int :float :long]
               (mapv :dtype (get-in descriptor [:steps 0 :abi]))))
        (let [program (fixture/instantiate! session descriptor [x q y labels n]
                                            {'x :input 'q :input
                                             'y :output 'labels :output})
              result (fixture/run! program [x q y labels n])
              ^floats actual-y (get result 'y)
              ^ints actual-labels (get result 'labels)]
          (is (= (float (* 2.0 (aget x 256))) (aget actual-y 256)))
          (is (= (+ 7 (int (aget q 256))) (aget actual-labels 256))))
        (finally (gpu/close-session! session))))))

(deftest ocl-resident-typed-region-ssa-shares-tuple-work
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed scalar-region SSA")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-effect-shared-local
                                             :ocl:0 :dtype :float)
          source (get-in descriptor [:steps 0 :artifact :source])
          n 513
          x (float-array (map float (range n)))
          a (float-array n)
          b (float-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= :kernel-body (get-in descriptor [:steps 0 :artifact :provenance :dialect])))
        (is (= 1 (count (re-seq #"x\[" source)))
            "the live kernel loads the shared tuple producer's input once")
        (let [program (fixture/instantiate! session descriptor [x a b n]
                                            {'x :input 'a :output 'b :output})
              result (fixture/run! program [x a b n])
              ^floats actual-a (get result 'a)
              ^floats actual-b (get result 'b)]
          (is (= 513.0 (double (aget actual-a 512))))
          (is (= (* 513.0 513.0) (double (aget actual-b 512)))))
        (finally (gpu/close-session! session))))))

(deftest ocl-resident-typed-scan-runs-through-the-compiled-graph-step
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed scan graph")
    (let [n 1025
          descriptor (pl/compile-gpu-program #'ocl-session-scan :ocl:0 :dtype :float)
          x (float-array (repeat n 1.0))
          out (float-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= [:executable] (mapv :convention (:steps descriptor))))
        (let [program (fixture/instantiate! session descriptor [x out n]
                                            {'x :input 'out :output})
              result (fixture/run! program [x out n])
              ^floats scanned (get result 'out)]
          (is (= (float n) (aget scanned (dec n))))
          (is (every? (fn [i] (= (float (inc i)) (aget scanned i)))
                      [0 1 255 256 511 512 1024])))
        (finally (gpu/close-session! session))))))

(deftest ocl-staged-typed-scan-runs-the-same-compiled-graph
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "staged typed scan graph")
    (let [n 1025
          execute (pl/compile-aot #'ocl-session-scan
                                  :target-device :ocl:0 :dtype :float)
          x (float-array (repeat n 1.0))
          out (float-array n)
          result (execute x out n)]
      (is (identical? out result))
      (is (= (float n) (aget out (dec n))))
      (is (every? (fn [i] (= (float (inc i)) (aget out i)))
                  [0 1 255 256 511 512 1024])))))

(deftest ocl-resident-typed-stencil-runs-through-portable-kernel-body
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed stencil KernelBody")
    (let [n 513
          descriptor (pl/compile-gpu-program #'ocl-session-stencil :ocl:0 :dtype :float)
          x (float-array (map float (range n)))
          out (float-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= :portable-segstencil
               (get-in descriptor [:steps 0 :artifact :attributes
                                   :kernel-body :attributes :kind])))
        (let [program (fixture/instantiate! session descriptor [x out n]
                                            {'x :input 'out :output})
              ^floats result (get (fixture/run! program [x out n]) 'out)]
          (is (= 0.0 (double (aget result 0))))
          (is (= 0.0 (double (aget result (dec n)))))
          (is (every? (fn [i] (= (double (* 2 i)) (double (aget result i))))
                      [1 2 255 256 511])))
        (finally (gpu/close-session! session))))))

(deftest ocl-resident-and-staged-exclusive-scan-use-the-same-typed-graph
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident and staged typed exclusive scan graph")
    (let [n 1025
          x (float-array (repeat n 1.0))
          descriptor (pl/compile-gpu-program #'ocl-session-exclusive-scan
                                             :ocl:0 :dtype :float)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= :exclusive
               (get-in descriptor [:steps 0 :artifact :attributes :scan-mode])))
        (let [out (float-array (inc n))
              program (fixture/instantiate! session descriptor [x out n]
                                            {'x :input 'out :output})
              ^floats scanned (get (fixture/run! program [x out n]) 'out)]
          (is (every? (fn [i] (= (float i) (aget scanned i)))
                      [0 1 255 256 257 512 1024 1025])))
        (let [execute (pl/compile-aot #'ocl-session-exclusive-scan
                                      :target-device :ocl:0 :dtype :float)
              out (float-array (inc n))
              result (execute x out n)]
          (is (identical? out result))
          (is (every? (fn [i] (= (float i) (aget out i)))
                      [0 1 255 256 257 512 1024 1025])))
        (finally (gpu/close-session! session))))))

(deftest ocl-resident-indexed-row-reduction-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "indexed row reduction")
    (let [nrows 3
          width 513
          values (float-array (* nrows width) (float -10.0))
          indices (int-array nrows)
          descriptor (pl/compile-gpu-program #'ops/argmax-rows! :ocl:0 :dtype :float)
          s (gpu/make-session :ocl:0)]
      (aset values 3 (float 9.0))
      (aset values 256 (float 9.0))
      (aset values 512 (float 9.0))
      (aset values (+ width 255) Float/NaN)
      (aset values (+ width 257) Float/NaN)
      (aset values (+ width 300) (float 100.0))
      (aset values (+ (* 2 width) 411) (float 8.0))
      (try
        (let [program (fixture/instantiate! s descriptor [values indices nrows width])
              result (get (fixture/run! program [values indices nrows width]) 'indices)]
          (is (= [3 255 411] (vec result))))
        (let [execute (pl/compile-aot #'ops/argmax-rows! :target-device :ocl:0 :dtype :float)
              staged-indices (int-array nrows)]
          (is (nil? (execute values staged-indices nrows width))
              "the effect-only product returns nil, not its output buffer")
          (is (nil? (apply execute [values staged-indices nrows width]))
              "the hoisted IFn applyTo bridge also boxes void as nil")
          (is (= [3 255 411] (vec staged-indices))))
        (finally (gpu/close-session! s))))))

(deftest ocl-fused-maps-retain-each-observable-buffer-alias
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "fused map buffer identities")
    (let [emitted (opencl-pass/opencl-pass
                   '(let* [a (raster.par/map! left i n float (+ (aget x i) 1.0))
                            b (raster.par/map! right j n float (* (aget x j) 2.0))]
                           [a b])
                   :device-id :ocl:0 :min-elements 0 :dtype :float
                   :array-types {'x :float 'left :float 'right :float}
                   :scalar-types {'n :long})
          execute (eval (list 'fn '[x left right n] (:form emitted)))
          register! (requiring-resolve 'raster.gpu.ocl-runtime/register-kernel!)
          x (float-array [-1 0 2])
          left (float-array 3) right (float-array 3)]
      (is (= 1 (count (:kernels emitted))))
      (doseq [artifact (:kernels emitted)] (register! (:kernel-name artifact) artifact))
      (let [result (execute x left right 3)]
        (is (identical? left (first result)))
        (is (identical? right (second result))))
      (is (= [0.0 1.0 3.0] (vec left)))
      (is (= [-2.0 0.0 4.0] (vec right))))))

(deftest ocl-public-fused-buffer-maps-replay-both-results
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident fused buffer maps")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-fused-buffer-maps :ocl:0 :dtype :float)
          s (gpu/make-session :ocl:0)
          args [(float-array [-1 0 2]) (float-array 3) (float-array 3) 3]]
      (try
        (let [program (fixture/instantiate! s descriptor args)]
          (try
            (let [result (fixture/run! program args)]
              (is (= [0.0 1.0 3.0] (vec (get result 'left))))
              (is (= [-2.0 0.0 4.0] (vec (get result 'right)))))
            (finally (fixture/close! program))))
        (finally (gpu/close-session! s))))))

(deftest ocl-strided-transfers-retain-host-control-and-accumulation
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "typed strided transfers in host control")
    (let [register! (requiring-resolve 'raster.gpu.ocl-runtime/register-kernel!)]
      (doseq [op '[raster.par/gather raster.par/scatter!]]
        (let [emitted (opencl-pass/opencl-pass
                       (list 'if 'enabled (list op 'out 'src 'indices 'n 'stride) nil)
                       :device-id :ocl:0 :min-elements 0 :dtype :float
                       :array-types {'out :float 'src :float 'indices :int}
                       :scalar-types {'n :long 'stride :long 'enabled :boolean})
              execute (eval (list 'fn '[out src indices n stride enabled] (:form emitted)))
              source (float-array [1 2 3 4 5 6])
              indices (int-array [2 0 2])
              output (float-array [10 20 30 40 50 60])]
          (doseq [artifact (:kernels emitted)]
            (register! (:kernel-name artifact) artifact))
          ;; nil dimensions would fail extent evaluation: the inactive branch must not touch it.
          (is (nil? (execute output source indices nil nil false)))
          (is (= [10.0 20.0 30.0 40.0 50.0 60.0] (vec output)))
          (is (identical? output (execute output source indices 3 2 true)))
          (is (= (if (= op 'raster.par/gather)
                   [5.0 6.0 1.0 2.0 5.0 6.0]
                   [13.0 24.0 30.0 40.0 56.0 68.0])
                 (vec output))
              "scatter adds colliding rows to existing storage; gather overwrites"))))))

(deftest ocl-resident-row-gather-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "row gather")
    (let [nrows 3
          width 4
          table (float-array (map float (range 32)))
          indices (int-array [5 1 5])
          rows (float-array (* nrows width))
          descriptor (pl/compile-gpu-program #'ops/gather-rows! :ocl:0 :dtype :float)
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [table indices rows nrows width])
              result (get (fixture/run! program [table indices rows nrows width]) 'out)]
          (is (= [20.0 21.0 22.0 23.0
                  4.0 5.0 6.0 7.0
                  20.0 21.0 22.0 23.0]
                 (vec result))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-segmap-artifact-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident SegMap artifact")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-map :ocl:0 :dtype :float)
          n 4096
          x (float-array (map float (range n)))
          y (float-array n)
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [x y (float 2.0) n])
              result (get (fixture/run! program [x y (float 2.0) n]) 'y)]
          (is (= [:map] (mapv :convention (:steps descriptor))))
          (is (every? (fn [i] (< (Math/abs (- (aget ^floats result (int i)) (* 2.0 i)))
                                 1e-3))
                      (range n))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-typed-inout-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed inout SegMap")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-inout :ocl:0 :dtype :float)
          n 4096
          state (float-array (map #(float (inc %)) (range n)))
          s (gpu/make-session :ocl:0)]
      (try
        (is (= [:inout :scalar :scalar]
               (mapv :kind (:argument-specs (first (:steps descriptor))))))
        (let [program (fixture/instantiate! s descriptor [state (float 2.0) n]
                                            {'state :state})]
          (fixture/run! program [state (float 2.0) n])
          (let [^floats once (fixture/download program 'state)]
            (is (= (float (* 2.0 513.0)) (aget once 512))))
          (fixture/run! program [state (float 2.0) n])
          (let [^floats twice (fixture/download program 'state)]
            (is (= (float (* 4.0 513.0)) (aget twice 512))
                "resident state is read and written through one ABI pointer on replay")))
        (finally (gpu/close-session! s))))))

(deftest ocl-staged-typed-inout-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "staged typed inout SegMap")
    (let [n 1024
          execute (pl/compile-aot #'ocl-session-inout
                                  :target-device :ocl:0 :dtype :float)
          state (float-array (map #(float (inc %)) (range n)))
          result (execute state (float 3.0) n)]
      (is (identical? state result))
      (is (= (float (* 3.0 513.0)) (aget state 512))))))

(deftest ocl-resident-contraction-binds-public-long-dimensions
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "public Long contraction dimensions")
    (let [descriptor (pl/compile-gpu-program #'nn/linear-nb :ocl:0 :dtype :float
                                            :gemm-precision :f32-scalar)
          x (float-array [1 2 3 4 5 6])
          weights (float-array [1 0 0 0 1 1])
          arguments [x weights (long 2) (long 3) (long 2)]
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor arguments)]
          (try
            (is (= [1.0 5.0 4.0 11.0]
                   (vec (get (fixture/run! program arguments) (:result-sym descriptor)))))
            (finally (fixture/close! program))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-contraction-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident contraction")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-contract :ocl:0 :dtype :float)
          A (float-array (map float (range 64)))
          B (float-array (repeat 64 (float 1.0)))
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [A B])
              result (get (fixture/run! program [A B]) 'C)]
          (is (= [:executable] (mapv :convention (:steps descriptor))))
          (is (= (vec (mapcat #(repeat 8 (float %))
                              (map (fn [row] (reduce + (range (* row 8) (* (inc row) 8))))
                                   (range 8))))
                 (vec result))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-reduction-ordered-abi-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident reduction")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-reduce :ocl:0 :dtype :float)
          n 1024
          scale 0.75
          x (float-array (map #(float (/ (inc %) 1024.0)) (range n)))
          out (float-array n)
          expected-sum (reduce + 0.0 (map #(* scale (double %)) (seq x)))
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [x out scale n])
              result (get (fixture/run! program [x out scale n]) 'out)
              red-step (first (filter #(= :executable (:convention %)) (:steps descriptor)))]
          (is (= 2 (count (get-in red-step [:artifact :nodes]))))
          (is (= [:input :output :scalar :scalar]
                 (mapv :kind (:argument-specs red-step))))
          (is (< (Math/abs (- (double (aget ^floats result 511))
                              (* (double (aget x 511)) expected-sum)))
                 1e-3)))
        (finally (gpu/close-session! s))))))
