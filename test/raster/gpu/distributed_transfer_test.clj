(ns raster.gpu.distributed-transfer-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.buffer-view :as view]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as distributed])
  (:import [java.lang.foreign MemorySegment]))

(deftest synchronous-host-staging-is-bounded-and-preserves-tail-offsets
  (let [source (float-array (range 9)) target (float-array 9)
        source-segment (MemorySegment/ofArray source)
        target-segment (MemorySegment/ofArray target)
        make-view (fn [id device]
                    (view/view (view/allocation {:id id :device device :byte-size 36 :memory-space :device})
                               {:dtype :float :shape [9]}))
        src (make-view :source :gpu-0) dst (make-view :target :gpu-1)
        chunks (atom []) segments (atom [])]
    (with-redefs [gpu/buffer-view (fn [_ _ opts] opts)
                  gpu/download-range!
                  (fn [_ _ ^MemorySegment host {:keys [src-element elements]}]
                    (swap! segments conj host)
                    (swap! chunks conj [:read src-element elements (.byteSize host)])
                    (.copyFrom (.asSlice host 0 (* elements 4))
                               (.asSlice source-segment (* src-element 4) (* elements 4))))
                  gpu/upload-range!
                  (fn [_ _ ^MemorySegment host {:keys [dst-element elements]}]
                    (swap! chunks conj [:write dst-element elements])
                    (.copyFrom (.asSlice target-segment (* dst-element 4) (* elements 4))
                               (.asSlice host 0 (* elements 4))))]
      (#'distributed/transfer-host-staged! {:gpu-0 :source-session :gpu-1 :target-session} src dst 16))
    (is (= (vec source) (vec target)))
    (is (= [[:read 0 4 16] [:write 0 4] [:read 4 4 16] [:write 4 4]
            [:read 8 1 16] [:write 8 1]] @chunks))
    (is (every? #(not (.isAlive (.scope ^MemorySegment %))) @segments))
    (reset! segments [])
    (with-redefs [gpu/buffer-view (fn [_ _ opts] opts)
                  gpu/download-range! (fn [_ _ host _] (swap! segments conj host))
                  gpu/upload-range! (fn [& _] (throw (ex-info "upload failed" {})))]
      (is (thrown? clojure.lang.ExceptionInfo
                   (#'distributed/transfer-host-staged! {:gpu-0 :source :gpu-1 :target} src dst 16))))
    (is (every? #(not (.isAlive (.scope ^MemorySegment %))) @segments))
    (is (= 1 (count @segments)) "an upload failure stops the transfer rather than publishing later chunks")))
