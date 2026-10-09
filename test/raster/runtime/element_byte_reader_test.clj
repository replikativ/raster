(ns raster.runtime.element-byte-reader-test
  (:require [clojure.test :refer [deftest is]]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang.foreign MemorySegment]
           [java.util Arrays]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest unaligned-windows-preserve-exact-bytes-with-bounded-element-transfers
  ;; Include non-power-of-two width: the adapter works on byte extents, not a dtype registry.
  (doseq [width [1 2 3 4 8 16]]
    (let [extent (* width 17000)
          bytes (byte-array (map unchecked-byte (range extent)))
          source (MemorySegment/ofArray bytes)
          copied (byte-array extent)
          target (MemorySegment/ofArray copied)
          reads (atom [])
          reader (content/element-byte-reader
                  extent width
                  (fn [start elements destination]
                    (swap! reads conj [start elements (.byteSize ^MemorySegment destination)])
                    (MemorySegment/copy source (* start width) destination 0 (* elements width))))]
      (loop [offset 0 windows (cycle [1 7 65531 3 11])]
        (when (< offset extent)
          (let [n (min (first windows) (- extent offset))]
            (is (= n (reader offset (.asSlice target offset n))))
            (recur (+ offset n) (next windows)))))
      (is (Arrays/equals bytes copied))
      (is (every? (fn [[start elements n]]
                    (and (= n (* width elements))
                         (<= n (+ 65536 (* 2 (dec width))))
                         (<= (+ start elements) 17000))) @reads))
      (is (= (content/content-address-of source)
             (content/content-address-from-reader extent reader))))))

(deftest invalid-extents-and-windows-never-contact-the-source
  (let [calls (atom 0) download! (fn [& _] (swap! calls inc))]
    (doseq [[extent width f] [[-1 1 download!] [1.5 1 download!]
                             [(inc (bigint Long/MAX_VALUE)) 1 download!]
                             [8 0 download!] [8 -1 download!] [8 1.5 download!]
                             [8 65537 download!] [7 4 download!] [8 4 nil]]]
      (is (= :numerical-content-element-reader
             (reason #(content/element-byte-reader extent width f)))))
    (let [reader (content/element-byte-reader 8 4 download!)
          one (MemorySegment/ofArray (byte-array 1))
          too-large (MemorySegment/ofArray (byte-array 65537))]
      (doseq [[offset destination] [[-1 one] [0.5 one] [9 one] [8 one]
                                    [0 nil] [0 too-large]]]
        (is (= :numerical-content-element-read-range
               (reason #(reader offset destination)))))
      (is (= 0 (reader 8 MemorySegment/NULL))))
    (let [reader (content/element-byte-reader Long/MAX_VALUE 1 download!)]
      (is (= :numerical-content-element-read-range
             (reason #(reader (dec Long/MAX_VALUE)
                              (MemorySegment/ofArray (byte-array 2)))))))
    (is (zero? @calls))))

(deftest source-failure-does-not-copy-stale-staging-bytes
  (let [failure (ex-info "download failed" {})
        bytes (byte-array [1 2 3 4])
        destination (MemorySegment/ofArray bytes)
        reader (content/element-byte-reader 4 4 (fn [& _] (throw failure)))]
    (is (identical? failure (try (reader 1 (.asSlice destination 0 1))
                                (catch Throwable e e))))
    (is (= [1 2 3 4] (vec bytes)))))
