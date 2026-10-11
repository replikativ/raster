(ns raster.gpu.float-semantics-device-test
  "GPU float arithmetic has the deftm's JVM semantics.

  Division and square root are correctly rounded single precision
  (ze-runtime/spirv-build-flags), and a product and a sum emitted as separate
  statements are rounded separately rather than contracted into an FMA."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.arrays :as ra]
            [raster.core :refer [deftm]]
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
