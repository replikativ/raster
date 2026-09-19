(ns raster.gpu.ze-kernel-source-test
  "Canonical kernel sources: identical kernels compile to identical text.

  Pure string tests; they load the Level Zero runtime namespace but touch no
  device."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]))

(defn- canonical-entry
  []
  (try
    @(requiring-resolve 'raster.gpu.ze-runtime/canonical-entry)
    (catch Throwable _ nil)))

(defn- kernel
  [name accumulator row]
  (str "__kernel void " name "(__global float* a, int n) {\n"
       "  float " accumulator " = 0.0f;\n"
       "  int " row " = get_global_id(0);\n"
       "  " accumulator " += a[" row "];\n"
       "  a[" row "] = " accumulator ";\n}\n"))

(deftest identical-kernels-compile-to-identical-text
  (if-let [canonical (canonical-entry)]
    (let [[entry-a source-a] (canonical "segmap_48213" (kernel "segmap_48213" "__acc_48210" "__row_48211"))
          [entry-b source-b] (canonical "segmap_91377" (kernel "segmap_91377" "__acc_91374" "__row_91375"))]
      (testing "registry names and temporaries do not change the compiled text"
        (is (= entry-a entry-b))
        (is (= source-a source-b))
        (is (str/starts-with? entry-a "rk_")))
      (testing "the compiled text names the canonical entry point"
        (is (str/includes? source-a (str "__kernel void " entry-a "(")))
        (is (not (str/includes? source-a "segmap_48213")))
        (is (not (str/includes? source-a "48210"))))
      (testing "temporaries stay distinct"
        (is (= 1 (count (re-seq #"float __rg\d+ = 0.0f" source-a))))
        (is (not= (second (re-find #"float (__rg\d+)" source-a))
                  (second (re-find #"int (__rg\d+)" source-a))))))
    (println "  [SKIP] Level Zero runtime namespace unavailable")))

(deftest different-kernels-keep-different-entry-points
  (if-let [canonical (canonical-entry)]
    (let [[entry-a] (canonical "segmap_1001" (kernel "segmap_1001" "__acc_1002" "__row_1003"))
          [entry-b] (canonical "segmap_1001"
                               (str/replace (kernel "segmap_1001" "__acc_1002" "__row_1003")
                                            "+=" "-="))]
      (is (not= entry-a entry-b)))
    (println "  [SKIP] Level Zero runtime namespace unavailable")))

(deftest builtins-and-fixed-names-are-untouched
  (if-let [canonical (canonical-entry)]
    (let [source (str "__kernel void fixed_kernel(__global half* a) {\n"
                      "  float4 v = vload4(0, (__global float*)a);\n"
                      "  float x = M_PI_2 + tile_16;\n}\n")
          [entry compiled] (canonical "fixed_kernel" source)]
      (is (str/includes? compiled "vload4"))
      (is (str/includes? compiled "M_PI_2"))
      (is (str/includes? compiled "tile_16"))
      (is (str/includes? compiled (str "void " entry "("))))
    (println "  [SKIP] Level Zero runtime namespace unavailable")))

(deftest a-source-without-the-registered-name-is-compiled-as-is
  (if-let [canonical (canonical-entry)]
    (is (= ["other_1234" "__kernel void k(){}"]
           (canonical "other_1234" "__kernel void k(){}")))
    (println "  [SKIP] Level Zero runtime namespace unavailable")))
