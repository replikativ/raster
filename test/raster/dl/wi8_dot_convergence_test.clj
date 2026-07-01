(ns raster.dl.wi8-dot-convergence-test
  "Cross-backend convergence: the int8-dot (the quantized inner product) lowers to
   every backend's int8-MAC schedule — JVM bytecode, CPU-C (clang auto-vec), and
   wasm-SIMD128 (i32x4.dot_i16x8_s) — all bit-identical (int32 accumulation is exact
   everywhere, cf. torch._int_mm). One algorithm, N hardware schedules.

   CONVENTION NOTE: the CPU-C backend emits `void` kernels that return through an
   output buffer; the JVM and wasm backends return the scalar. So the *same*
   computation is expressed two ways here — `wi8-dot-out` (writes out[0], for C/JVM)
   and `wi8-dot` (scalar return, for wasm/JVM). Unifying the top-level convention so
   a single deftm serves all three (wasm's output-buffer path currently mis-types the
   nested-dot-in-aset form) is a tracked follow-up. GPU dp4a needs a device (covered
   by the qlinear GPU tests)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.compiler.pipeline :as pl]
            [clojure.string :as str])
  (:import [com.dylibso.chicory.wasm Parser]))

;; output-buffer form (C void kernel / JVM): int8 dot → int32 → out[0]
(deftm wi8-dot-out [x :- (Array byte), y :- (Array byte), n :- Long, out :- (Array int)] :- nil
  (ra/aset out 0
    (int (loop [i 0 acc 0]
           (if (clojure.core/< (long i) (long n))
             (recur (clojure.core/inc (long i))
                    (clojure.core/+ (long acc)
                                    (clojure.core/* (long (ra/aget x i)) (long (ra/aget y i)))))
             acc)))))

;; scalar-return form (JVM / wasm): the loop's acc IS the result
(deftm wi8-dot [x :- (Array byte), y :- (Array byte), n :- Long] :- Long
  (loop [i 0 acc 0]
    (if (clojure.core/< (long i) (long n))
      (recur (clojure.core/inc (long i))
             (clojure.core/+ (long acc)
                             (clojure.core/* (long (ra/aget x i)) (long (ra/aget y i)))))
      acc)))

(defn- clang-available? []
  (try (let [p (-> (ProcessBuilder. ^java.util.List [(or (System/getenv "RASTER_CC") "clang") "--version"])
                   (.redirectErrorStream true) (.start))]
         (.waitFor p) (zero? (.exitValue p)))
       (catch Exception _ false)))

(defn- node-available? []
  (try (let [p (-> (ProcessBuilder. ^java.util.List ["node" "--version"]) (.redirectErrorStream true) (.start))]
         (.waitFor p) (zero? (.exitValue p)))
       (catch Exception _ false)))

(defn- gen [n f] (byte-array (map #(unchecked-byte (f %)) (range n))))
(defn- ref-dot [x y n] (int (reduce + (map (fn [a b] (* (long a) (long b))) x y))))

(deftest jvm-schedule
  (testing "JVM bytecode lowers both forms correctly"
    (let [f-out (pl/compile-aot #'wi8-dot-out)
          f-scl (pl/compile-aot #'wi8-dot)]
      (doseq [n [8 40 100]]
        (let [x (gen n #(- (* 3 %) 5)) y (gen n #(- 7 (* 2 %))) o (int-array 1)]
          (f-out x y n o)
          (is (= (ref-dot x y n) (aget o 0)) (str "jvm out n=" n))
          (is (= (ref-dot x y n) (int (f-scl x y n))) (str "jvm scalar n=" n)))))))

(deftest cpu-c-schedule
  (testing "CPU-C (clang) lowers the int8 dot from the same source"
    (if-not (clang-available?)
      (println "SKIP cpu-c-schedule: no C compiler")
      (let [cf (pl/compile-aot #'wi8-dot-out :target :c)]
        (doseq [n [8 40 100]]
          (let [x (gen n #(- (* 3 %) 5)) y (gen n #(- 7 (* 2 %))) o (int-array 1)]
            (cf x y n o)
            (is (= (ref-dot x y n) (aget o 0)) (str "cpu-c n=" n))))))))

(deftest wasm-simd-schedule
  (testing "wasm-SIMD128 lowers the int8 dot to i32x4.dot_i16x8_s"
    (let [m  (pl/compile-wasm #'wi8-dot :name "wi8" :wasm-simd? true)
          bs (:bytes m)]
      (is (some? (Parser/parse bs)) "valid v128 module")
      (is (some (fn [i] (and (= (bit-and (nth bs i) 0xff) 0xfd)
                             (= (bit-and (nth bs (inc i)) 0xff) 0xba)))
                (range (dec (count bs))))
          "emits i32x4.dot_i16x8_s (not a scalar fallback)")
      (when (node-available?)
        (doseq [n [15 16 40 100]]
          (let [xs (mapv #(int (- (* 3 %) 5)) (range n))
                ys (mapv #(int (- 7 (* 2 %))) (range n))
                dir (java.nio.file.Files/createTempDirectory "wi8" (make-array java.nio.file.attribute.FileAttribute 0))
                wp (str dir "/k.wasm") jp (str dir "/r.mjs")]
            (with-open [o (java.io.FileOutputStream. wp)] (.write o bs))
            (spit jp (str "import fs from 'fs';const b=fs.readFileSync('" wp "');"
                          "const {instance}=await WebAssembly.instantiate(b,{});"
                          "const m=new Int8Array(instance.exports.memory.buffer);"
                          "const xs=[" (str/join "," xs) "],ys=[" (str/join "," ys) "];"
                          "for(let i=0;i<" n ";i++){m[i]=xs[i];m[" n "+i]=ys[i];}"
                          "console.log(instance.exports.wi8(0," n "," n "));"))
            (let [p (-> (ProcessBuilder. ^java.util.List ["node" jp]) (.redirectErrorStream true) (.start))
                  out (str/trim (slurp (.getInputStream p)))]
              (.waitFor p)
              (is (= (ref-dot (gen n #(- (* 3 %) 5)) (gen n #(- 7 (* 2 %))) n)
                     (Integer/parseInt out))
                  (str "wasm-v8 n=" n)))))))))
