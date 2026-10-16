(ns raster.gpu.float-semantics-device-test
  "GPU float arithmetic has the deftm's JVM semantics.

  Division and square root are correctly rounded single precision
  (ze-runtime/spirv-build-flags), and a product and a sum emitted as separate
  statements are rounded separately rather than contracted into an FMA."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.arrays :as ra]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as ops]
            [raster.dl.nn :as nn]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as ocl]
            [raster.gpu.value :as value]
            [raster.math :as math]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm float-ops!
  [a :- (Array float), b :- (Array float), c :- (Array float),
   q :- (Array float), r :- (Array float), f :- (Array float), n :- Long] :- Void
  (par/map-void! i n
                 (let [x (ra/aget a i) y (ra/aget b i) z (ra/aget c i)
                       p (* x y)]
                   (ra/aset q i (/ x y))
                   (ra/aset r i (math/sqrt z))
                   (ra/aset f i (+ p z)))))

(defn- bits [x] (Float/floatToRawIntBits (float x)))

(defn- array-bits [xs]
  (if (= Float/TYPE (.getComponentType (class xs)))
    (mapv #(Float/floatToRawIntBits (float %)) xs)
    (mapv #(Double/doubleToRawLongBits (double %)) xs)))

(defn- backward-reference [dy x result scale bound]
  (let [n (alength dy)
        float-storage? (= Float/TYPE (.getComponentType (class dy)))
        out (if float-storage?
              (float-array n)
              (double-array n))]
    (dotimes [i n]
      (let [raw (* (ra/aget x i) scale)
            mask (if (and (>= raw (- bound)) (<= raw bound)) 1.0 0.0)
            dy-result (* (ra/aget dy i) (ra/aget result i))
            dy-result (if float-storage? (unchecked-float dy-result) dy-result)]
        (ra/aset out i (* dy-result (* scale mask)))))
    out))

(defn- check-scale-clamp-exp-backward! [target make-array dtype]
  (let [scale 1.0
        bound 2.0
        cases [[[-2.0 -1.0 0.0 1.0 2.0] [1.0 -2.0 3.0 -4.0 5.0]
                [0.5 1.5 2.5 3.5 4.5]]
               [[-3.0 -0.5 -0.0 0.5 3.0] [-5.0 4.0 -3.0 2.0 -1.0]
                [4.5 3.5 2.5 1.5 0.5]]]
        [initial-x initial-dy initial-result] (mapv make-array (first cases))
        executable (compiled/instantiate!
                    (compiled/lower #'ops/scale-clamp-exp-backward
                                    [initial-dy initial-x initial-result scale bound (long 5)]
                                    {:compiler :equation-first :target target :dtype dtype}))]
    (try
      (doseq [[x-values dy-values result-values] cases]
        (let [x (make-array x-values)
              dy (make-array dy-values)
              result (make-array result-values)
              before (mapv array-bits [dy x result])
              expected (backward-reference dy x result scale bound)
              actual (value/->host
                      (:result (compiled/invoke-compiled
                                executable {:dy dy :x x :result result})))]
          (is (= (class expected) (class actual)) (str target " " dtype))
          (is (= (array-bits expected) (array-bits actual)) (str target " " dtype))
          (is (= before (mapv array-bits [dy x result])) (str target " inputs"))))
      (finally (compiled/close! executable)))))

(deftest scale-clamp-exp-backward-runs-on-both-device-routes
  (doseq [[target available? skip!]
          [[:ocl:0 ocl/opencl-available? ocl/opencl-skip!]
           [:ze:0 gp/gpu-available? gp/gpu-skip!]]]
    (if-not @available?
      (skip! (str "scale-clamp-exp backward on " target))
      (let [caps (when (= target :ze:0)
                   ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities)))
            fp64? (if (= target :ocl:0)
                    @ocl/opencl-fp64-available?
                    (:fp64? caps))]
        ;; Both array specializations retain Double scale/bound controls and arithmetic.
        (if fp64?
          (do
            (check-scale-clamp-exp-backward! target float-array :float)
            (check-scale-clamp-exp-backward! target double-array :double)
            nil)
          (if (= target :ocl:0)
            (ocl/opencl-skip! "scale-clamp-exp backward Double controls" :fp64)
            (gp/gpu-capability-skip! "scale-clamp-exp backward Double controls"
                                     :fp64? caps)))))))

(defn- check-dot-rows-dh! [target make-array dtype]
  (let [rows 17 dim 257
        cases [[(mapv #(* 0.125 (- % 8)) (range rows))
                (mapv #(* 0.03125 (- (mod % 17) 8)) (range dim))]
               [(mapv #(* 0.0625 (- 10 %)) (range rows))
                (mapv #(* 0.015625 (- (mod % 23) 11)) (range dim))]]
        [initial-dy initial-w] (mapv make-array (first cases))
        executable (compiled/instantiate!
                    (compiled/lower #'ops/dot-rows-dh [initial-dy initial-w rows dim]
                                    {:compiler :equation-first :target target :dtype dtype}))]
    (try
      (doseq [[dy-values w-values] cases]
        (let [dy (make-array dy-values) w (make-array w-values)
              before (mapv array-bits [dy w])
              ;; Exact dyadic products are an independent oracle for both storage types.
              expected (make-array (for [v (range rows) d (range dim)]
                                     (* (nth dy-values v) (nth w-values d))))
              actual (value/->host
                      (:result (compiled/invoke-compiled executable {:dy dy :W w})))]
          (is (= (class expected) (class actual)) (str target " " dtype))
          (is (= (array-bits expected) (array-bits actual)) (str target " " dtype " all 4369 products"))
          (is (= before (mapv array-bits [dy w])) "changed inputs remain immutable")))
      (finally (compiled/close! executable)))))

(deftest dot-rows-dh-replays-changed-inputs-on-both-device-routes
  (doseq [[target available? skip!] [[:ocl:0 ocl/opencl-available? ocl/opencl-skip!]
                                    [:ze:0 gp/gpu-available? gp/gpu-skip!]]]
    (if-not @available?
      (skip! (str "dot-rows-dh on " target))
      (do
        (check-dot-rows-dh! target float-array :float)
        (let [caps (when (= target :ze:0)
                     ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities)))
              fp64? (if (= target :ocl:0) @ocl/opencl-fp64-available? (:fp64? caps))]
          (if fp64?
            (check-dot-rows-dh! target double-array :double)
            (if (= target :ocl:0)
              (ocl/opencl-skip! "dot-rows-dh Double" :fp64)
              (gp/gpu-capability-skip! "dot-rows-dh Double" :fp64? caps))))))))

(defn- check-gelu-rational-derivatives! [target make-array dtype]
  (doseq [[v with-seed?] [[#'nn/gelu-backward true] [#'nn/gelu-second-deriv false]]]
    (let [cases [[-6.0 -5.0 -3.0 -1.0 -0.0 0.5 3.0 6.0]
                 [6.0 5.0 1.0 0.5 0.0 -0.5 -3.0 -6.0]]
          n (count (first cases))
          seed (make-array [1.0 -2.0 0.0 0.5 -1.0 2.0 -0.5 1.0])
          args (fn [x] (if with-seed? [seed x n] [x n]))
          executable (compiled/instantiate!
                       (compiled/lower v (args (make-array (first cases)))
                                       {:compiler :equation-first :target target :dtype dtype}))]
      (try
        (doseq [xs cases]
          (let [x (make-array xs)
                before (mapv array-bits [seed x])
                expected (apply @v (args x))
                actual (value/->host
                        (:result (compiled/invoke-compiled executable
                                                          (cond-> {:x x} with-seed? (assoc :dy seed)))))]
            (is (= (class expected) (class actual)) (str target " " dtype " " (:name (meta v))))
            (is (every? true? (map #(< (Math/abs (- (double %1) (double %2)))
                                       (if (= dtype :float) 1e-6 1e-12)) expected actual))
                (str target " " dtype " " (:name (meta v))))
            (is (= before (mapv array-bits [seed x])) "replayed inputs remain immutable")))
        (finally (compiled/close! executable))))))

(deftest gelu-rational-derivatives-replay-on-both-device-routes
  (doseq [[target available? skip!] [[:ocl:0 ocl/opencl-available? ocl/opencl-skip!]
                                    [:ze:0 gp/gpu-available? gp/gpu-skip!]]]
    (if-not @available?
      (skip! (str "rational GELU derivatives on " target))
      (let [caps (when (= target :ze:0)
                   ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities)))
            fp64? (if (= target :ocl:0) @ocl/opencl-fp64-available? (:fp64? caps))]
        (check-gelu-rational-derivatives! target float-array :float)
        (if fp64?
          (check-gelu-rational-derivatives! target double-array :double)
          (if (= target :ocl:0)
            (ocl/opencl-skip! "rational GELU Double" :fp64)
            (gp/gpu-capability-skip! "rational GELU Double" :fp64? caps)))))))

(deftm mixed-norm-tail!
  [dy :- (Array float), x :- (Array float), weight :- (Array float),
   inv :- (Array float), dot :- (Array float), out :- (Array float),
   features :- Long, count :- Long, gain :- Double] :- (Array float)
  (par/map-void! t count
    (let [row (quot t features) column (rem t features)
          gi (double (n/+ gain (double (ra/aget weight column))))]
      (ra/aset out t
               (float (n/- (n/* (double (ra/aget inv row))
                                (n/* gi (double (ra/aget dy t))))
                            (double (n/* (ra/aget dot row) (ra/aget x t))))))))
  out)

(deftest float-storage-retains-double-arithmetic-and-float-product-rounding
  ;; Same arithmetic boundary as chunked norm backward: Float dot*x is rounded
  ;; BEFORE widening into a Double subtraction, whose result is stored as Float.
  ;; Prematurely narrowing the other product, or widening dot*x, both erase e².
  (let [e (float (/ 1.0 8388608.0)) a (float (+ 1.0 e)) b (float (- 1.0 e))
        magnitude (float (* (double e) (double e)))
        expected [magnitude (- magnitude) (- magnitude) magnitude]
        arguments [(float-array [1.0 1.0 1.0 1.0]) (float-array [a b a b])
                   (float-array [e (- e)]) (float-array [a b]) (float-array [a b])
                   (float-array 4) 2 4 1.0]
        cpu (apply mixed-norm-tail! arguments)]
    (is (= (mapv bits expected) (mapv bits cpu)) "independent exact dyadic oracle versus JVM")
    (is (every? #(not (zero? %)) expected) "the fixture distinguishes collapsed Float arithmetic")
    (doseq [[target available? skip!] [[:ocl:0 ocl/opencl-available? ocl/opencl-skip!]
                                      [:ze:0 gp/gpu-available? gp/gpu-skip!]]]
      (if-not @available?
        (skip! (str "mixed norm precision on " target))
        (let [live (compiled/compile #'mixed-norm-tail! (assoc arguments 5 (float-array 4))
                                     {:compiler :equation-first :dtype :float :target target})]
          (try
            (dotimes [_ 2]
              (let [outputs (live {})]
                (is (= 1 (count outputs)))
                (is (= (mapv bits expected)
                       (mapv bits (value/->host (first (vals outputs))))) (str target))))
            (finally (compiled/close! live))))))))

(deftest division-and-sqrt-are-correctly-rounded
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "float semantics")
    (let [n 65536
          r (java.util.Random. 5)
          rnd #(let [a (float-array n)]
                 (dotimes [i n]
                   (aset a i (float (* (.nextGaussian r) (Math/pow 10 (- (.nextInt r 8) 4))))))
                 a)
          a (rnd) b (rnd)
          ;; positive, so it can also feed the square root
          c (float-array (map #(Math/abs %) (rnd)))
          sess (gpu/make-session :ze:0)]
      (try
        (gpu/compile! sess :ops #'float-ops!)
        (gpu/alloc! sess {:a [:float n a] :b [:float n b] :c [:float n c]
                          :q [:float n nil] :r [:float n nil] :f [:float n nil]})
        (gpu/prepare! sess :ops {"a" :a "b" :b "c" :c "q" :q "r" :r "f" :f} [] n
                      {:kernel-phase :ops})
        (gpu/invoke-bound! sess :ops)
        (gpu/sync! sess)
        (let [^floats q (gpu/download sess :q)
              ^floats sq (gpu/download sess :r)
              ^floats f (gpu/download sess :f)
              wrong (fn [expected actual]
                      (count (remove true? (map #(= (bits (expected %)) (bits (aget ^floats actual %)))
                                                (range n)))))]
          (testing "division"
            (is (zero? (wrong #(/ (double (aget a %)) (double (aget b %))) q))))
          (testing "square root"
            (is (zero? (wrong #(Math/sqrt (double (aget c %))) sq))))
          (testing "a product bound before a sum is rounded before the sum"
            (is (zero? (wrong #(+ (double (float (* (double (aget a %)) (double (aget b %)))))
                                  (double (aget c %)))
                              f)))))
        (finally (gpu/close-session! sess))))))
