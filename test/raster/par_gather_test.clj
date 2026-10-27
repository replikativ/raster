(ns raster.par-gather-test
  "Tests for the par/gather combinator: scalar expansion + SIMD vgather emit.
   gather: out[i] = src[index[i]]."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.par :as par]
            [raster.core :refer [deftm]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.par :as ir-par]
            [raster.compiler.backend.jvm.par-simd :as psimd]
            [raster.compiler.backend.jvm.segop-simd :as segop-simd]))

(defn- reference-gather [^doubles src ^ints idx n]
  (let [out (double-array n)]
    (dotimes [i n] (aset out i (aget src (aget idx i))))
    out))

;; ---- scalar (macro) path ----------------------------------------------------

(deftest scalar-gather-flat
  (testing "(par/gather out src index n) = out[i]=src[index[i]]"
    (let [src (double-array [10.0 20.0 30.0 40.0 50.0])
          idx (int-array [4 0 2 0 3 1])
          out (double-array 6)]
      (par/gather out src idx 6)
      (is (= [50.0 10.0 30.0 10.0 40.0 20.0] (vec out))))))

(deftest scalar-gather-strided
  (testing "strided gather copies stride-wide rows"
    (let [src (double-array [1.0 2.0 3.0 4.0 5.0 6.0])  ; 3 rows x 2
          idx (int-array [2 0])
          out (double-array 4)]
      (par/gather out src idx 2 2)
      (is (= [5.0 6.0 1.0 2.0] (vec out))))))

;; ---- SIMD emit --------------------------------------------------------------

(deftest simd-pass-emits-vgather
  (testing "simd-pass lowers flat par/gather to a hardware vgather (fromArray indexMap + intoArray)"
    (let [simd (:form (psimd/simd-pass '(raster.par/gather out src idx n)
                                      :array-types {'out :double 'src :double 'idx :int}))
          s    (pr-str simd)]
      (is (clojure.string/includes? s "fromArray"))
      (is (clojure.string/includes? s "intoArray")))))

;; ---- end-to-end through the JIT (walker → simd-pass → bytecode) --------------

(deftm gather-kernel [out :- (Array double), src :- (Array double),
                      idx :- (Array int), n :- Long] :- (Array double)
  (raster.par/gather out src idx n))

(deftm gather-long-kernel [out :- (Array double) src :- (Array double)
                          idx :- (Array long) n :- Long] :- (Array double)
  (raster.par/gather out src idx n))

(deftm indexed-int-map-kernel [out :- (Array double) src :- (Array double)
                             idx :- (Array int) n :- Long] :- (Array double)
  (par/map! out i n double (clojure.core/aget src (clojure.core/aget idx i))))

(deftm indexed-long-map-kernel [out :- (Array double) src :- (Array double)
                              idx :- (Array long) n :- Long] :- (Array double)
  (par/map! out i n double (clojure.core/aget src (clojure.core/aget idx i))))

(deftm indexed-narrowing-map-kernel [out :- (Array double) src :- (Array double)
                                   idx :- (Array int) n :- Long] :- (Array double)
  (par/map! out i n double
    (clojure.core/float (clojure.core/aget src (clojure.core/aget idx i)))))

(deftest hardware-gather-requires-exact-backing-storage
  (let [types {'out :double 'src :double 'idx :int}]
    (is (some? (segop-simd/compile-par-gather 'out 'src 'idx 'n :double types)))
    (is (nil? (segop-simd/compile-par-gather 'out 'src 'idx 'n :double)))
    (doseq [unsupported [(assoc types 'idx :long) (dissoc types 'idx)
                         (assoc types 'src :float) (assoc types 'out :float)]]
      (is (nil? (segop-simd/compile-par-gather 'out 'src 'idx 'n :double unsupported))))
    (doseq [facts [nil (assoc types 'idx :long)]]
      (let [compiled (psimd/simd-pass '(raster.par/gather out src idx n)
                                     :array-types facts :min-elements 0)]
        (is (nil? (get-in compiled [:stats :simd-gathers])))
        (is (not (clojure.string/includes? (pr-str (:form compiled)) "fromArray")))))))

(deftest gather-storage-projection-does-not-treat-logical-int-as-physical-int-array
  ;; This pure projection consumes an already validated owner's AVs; no test
  ;; module or execution certificate is manufactured here.
  (let [plain {:kind :tensor :dtype :int :shape ['n] :representation {:kind :plain}}
        project #(get (#'psimd/gather-array-storage-types {:values {'idx %}} {'idx :int}) 'idx)]
    (is (= :int (project plain)))
    (doseq [value [(assoc plain :representation nil)
                   (assoc plain :representation {:kind :packed :storage-dtype :long})
                   (assoc plain :representation {:kind :plain :storage-dtype :long})
                   (assoc plain :logical-layout {:storage-dtype :long})
                   (assoc plain :attributes {:storage-dtype :long})
                   (assoc plain :memory-space :device)
                   (assoc plain :shape [])]]
      (is (nil? (project value))))))

(deftest public-jvm-gather-keeps-long-indices-off-the-int-array-vector-overload
  ;; Larger than TWO preferred vectors on every machine, not just the scalar
  ;; tail on a particular CI/local VectorSpecies width.
  (let [lanes (.length jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
        original segop-simd/compile-par-gather]
    (doseq [[source index-array hardware? complete]
            [[#'gather-kernel int-array true identity]
             [#'gather-long-kernel long-array false identity]
             [#'indexed-int-map-kernel int-array true identity]
             [#'indexed-long-map-kernel long-array false identity]
             [#'indexed-narrowing-map-kernel int-array false #(double (float %))]]]
      (let [selected (atom [])]
        (with-redefs [segop-simd/compile-par-gather
                      (fn [& arguments]
                        (let [result (apply original arguments)]
                          (when result (swap! selected conj arguments))
                          result))]
          (let [scalar (pipeline/compile-aot source :simd? false :dtype :double)
                vectorized (pipeline/compile-aot source :simd? true :dtype :double)]
            (is (= hardware? (boolean (seq @selected))) (str source))
            (doseq [n [(* 2 lanes) (+ (* 2 lanes) 3) 0]
                    generation [0 1]]
              (let [size (+ (max n 1) 13)
                    src (double-array (map #(+ (nth [0.1 16777217.0 -1.00000008 1.0e-12] (mod % 4))
                                              (* generation 100.5)) (range size)))
                    indices (mapv #(mod (+ (* 7 %) generation) size) (range n))
                    index-map (index-array indices)
                    original-source (vec src)
                    original-indices (vec index-map)
                    expected (mapv #(complete (aget src %)) indices)
                    scalar-out (double-array n)
                    vector-out (double-array n)]
                (is (identical? scalar-out (scalar scalar-out src index-map n)))
                (is (identical? vector-out (vectorized vector-out src index-map n)))
                (is (= expected (vec scalar-out) (vec vector-out)) (str source " n=" n))
                (is (= original-source (vec src)))
                (is (= original-indices (vec index-map)))))))))))

(deftest simd-gather-end-to-end
  (testing "JIT-compiled par/gather with SIMD matches the reference, incl. scalar tail"
    (doseq [n [40 37 8 7 1 0]]   ; multiples of lanes and not, plus tiny/empty
      (let [src (double-array (map #(* 1.5 (inc %)) (range (max 1 n))))
            idx (int-array (shuffle (range (max 1 n))))
            out (double-array (max 1 n))
            expect (reference-gather src idx n)]
        (binding [raster.core/*jit-simd?* true]
          (gather-kernel out src idx n))
        (is (java.util.Arrays/equals ^doubles (double-array (take n out))
                                     ^doubles expect)
            (str "n=" n))))))
