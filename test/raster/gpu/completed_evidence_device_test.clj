(ns raster.gpu.completed-evidence-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.compiler.build-manifest :as build]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.numeric :as numeric]
            [raster.par :as par]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang.foreign MemorySegment]))

(deftm witnessed-scale [x :- (Array float) n :- Long] :- (Array float)
  (let [out (float-array n)]
    (par/map! out i n float (numeric/* (float 2.0) (arrays/aget x i)))))

(deftm witnessed-state! [state :- (Array float) n :- Long] :- (Array float)
  (par/map-void! i n
    (arrays/aset state i (numeric/+ (arrays/aget state i) (float 1.0))))
  state)

(defn- test-build []
  ;; Synthetic packaged evidence tests the boundary, not release-build provenance.
  (build/manifest-identity
   {:schema-version build/schema-version :library 'raster/raster
    :version "resident-evidence-test-only" :revision "synthetic-resident-evidence"
    :runtime {:java-version (System/getProperty "java.version") :clojure-version (clojure-version)}
    :source-namespaces '[raster.arrays raster.numeric raster.par]
    :dependencies {}}))

(defn- address [^floats xs] (content/content-address-of (MemorySegment/ofArray xs)))
(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- input-node [c] (:node (first (filter #(= :input (:role %)) (:in-tree c)))))
(defn- input-key [c] (:key (first (filter #(= :input (:role %)) (:in-tree c)))))

(defn- run-input-case! [target]
  (with-redefs [build/current-identity (constantly (test-build))]
    (let [defaults (float-array [1 2 3 4])
          prepared (compiled/lower #'witnessed-scale [defaults 4]
                                   {:target target :compiler :equation-first :dtype :float})]
      (aset defaults 0 (float 111))
      (let [c (compiled/instantiate! prepared)
            x (float-array [3 4 5 6])
            expected (float-array [6 8 10 12])]
        (try
          (aset defaults 0 (float 222))
          (with-open [receipt (compiled/invoke-with-evidence c {(input-key c) (MemorySegment/ofArray x)})]
            (let [data @receipt output (first (vals (compiled/completed-output-values receipt)))]
              (is (compiled/completed-evidence? receipt))
              (is (true? (:attests-resident-bytes? data)))
              (is (= (address x) (get-in data [:inputs (input-node c) :content])))
              (is (= :device-native (get-in data [:inputs (input-node c) :representation])))
              (is (not (contains? (get-in data [:inputs (input-node c)]) :byte-order)))
              (is (= (address expected) (:content (first (vals (:outputs data))))))
              (is (= (vec expected) (vec (value/->host output))))
              (is (nil? (:parent data)))
              (is (= :link-output-lease-active
                     (reason #(compiled/invoke-compiled c {(input-key c) defaults}))))
              (is (= :compiled-completed-evidence-owner (reason #(deref (assoc receipt :data {})))))))
          (with-open [receipt (compiled/invoke-with-evidence c {(input-key c) defaults})]
            (is (not= (address x) (get-in @receipt [:inputs (input-node c) :content]))))
          (let [owner-state @(:execution-state (:executable c))
                epoch (:value-epoch owner-state)
                writes (atom 0)]
            (with-redefs [link/execution-order (constantly {:record-time-prologue [:unsupported]})
                          gpu/upload-range! (fn [& _] (swap! writes inc))]
              (is (= :compiled-evidence-record-time-prologue
                     (reason #(compiled/invoke-with-evidence c {(input-key c) x})))))
            (swap! (:session (:executable c)) assoc :events {:in-flight :opaque-test-event})
            (try
              (with-redefs [gpu/upload-range! (fn [& _] (swap! writes inc))]
                (is (= :compiled-evidence-unready
                       (reason #(compiled/invoke-with-evidence c {(input-key c) x})))))
              (finally (swap! (:session (:executable c)) assoc :events {})))
            (is (zero? @writes))
            (is (= epoch (:value-epoch @(:execution-state (:executable c)))))
            (is (identical? (:completed-evidence owner-state)
                            (:completed-evidence @(:execution-state (:executable c)))))
            (is (some? (reason #(compiled/invoke-with-evidence c {(input-key c) (float-array 3)}))))
            (is (= epoch (:value-epoch @(:execution-state (:executable c)))))
            (is (identical? (:completed-evidence owner-state)
                            (:completed-evidence @(:execution-state (:executable c))))))
          (let [foreign (compiled/instantiate! prepared)]
            (try
              (let [source (first (vals (compiled/invoke-compiled foreign {(input-key foreign) x})))]
                (with-open [receipt (compiled/invoke-with-evidence c {(input-key c) source})]
                  (is (= (address expected) (get-in @receipt [:inputs (input-node c) :content])))
                  (compiled/invoke-compiled foreign {(input-key foreign) (float-array 4)})
                  (is (= [12.0 16.0 20.0 24.0]
                         (vec (value/->host (first (vals (compiled/completed-output-values receipt)))))))))
              (finally (compiled/close! foreign))))
          (let [failure (ex-info "input upload failed" {})]
            (with-redefs [gpu/upload-range! (fn [& _] (throw failure))]
              (is (identical? failure (try (compiled/invoke-with-evidence c {(input-key c) x})
                                          (catch Throwable error error)))))
            (is (nil? (:completed-evidence @(:execution-state (:executable c)))))
            (is (seq @(:tainted-inputs (:executable c))))
            (with-open [fresh (compiled/invoke-with-evidence c {(input-key c) x})]
              (is (= (address x) (get-in @fresh [:inputs (input-node c) :content])))
              (is (empty? @(:tainted-inputs (:executable c))))))
          (let [bits (float-array [(Float/intBitsToFloat 0x7fc00011) (float 0.0) (float -0.0) 1.0])
                first-input (with-open [receipt (compiled/invoke-with-evidence c {(input-key c) bits})]
                              (let [actual (get-in @receipt [:inputs (input-node c) :content])]
                                (is (= (address bits) actual)) actual))]
            (aset bits 0 (Float/intBitsToFloat 0x7fc00013))
            (with-open [receipt (compiled/invoke-with-evidence c {(input-key c) bits})]
              (is (= (address bits) (get-in @receipt [:inputs (input-node c) :content])))
              (is (not= first-input (get-in @receipt [:inputs (input-node c) :content])))))
          (finally (compiled/close! c)))))))

(defn- run-ownership-and-range-case! [target]
  (with-redefs [build/current-identity (constantly (test-build))]
    (let [n 40000 x (float-array n) expected (float-array n)]
      (aset x (dec n) (float 3.0))
      (aset expected (dec n) (float 6.0))
      (let [prepared (compiled/lower #'witnessed-scale [x n]
                                    {:target target :compiler :equation-first :dtype :float})
            c (compiled/instantiate! prepared)]
        (try
          (with-open [receipt (compiled/invoke-with-evidence c {})]
            (is (= (address x) (get-in @receipt [:inputs (input-node c) :content])))
            (is (= (address expected) (:content (first (vals (:outputs @receipt)))))))
          (finally (compiled/close! c)))
        (let [session (gpu/make-session target)
              attached (compiled/instantiate! prepared {:session session})]
          (try
            (is (= :compiled-evidence-ownership
                   (reason #(compiled/invoke-with-evidence attached {}))))
            (is (zero? @(:completed-replays (:executable attached))))
            (finally (compiled/close! attached) (gpu/close-session! session))))))))

(defn- run-state-case! [target]
  (with-redefs [build/current-identity (constantly (test-build))]
    (let [c (compiled/compile #'witnessed-state! [(float-array [1 2 3 4]) 4]
                              {:target target :compiler :equation-first :dtype :float :donate '[state]})]
      (try
        (let [first-receipt (compiled/invoke-with-evidence c {})
              first-data @first-receipt]
          (is (seq (:post-state first-data)))
          (is (nil? (:parent first-data)))
          (.close ^java.io.Closeable first-receipt)
          (is (= first-data @first-receipt) "closed evidence still describes its historical bytes")
          (is (= :link-output-lease-released
                 (reason #(compiled/completed-output-values first-receipt))))
          (with-open [second (compiled/invoke-with-evidence c {})]
            (is (= (:fingerprint first-data) (:parent @second)))
            (is (= (:post-state first-data) (select-keys (:inputs @second) (keys (:post-state first-data))))))
          (compiled/invoke-compiled c {})
          (with-open [third (compiled/invoke-with-evidence c {})]
            (is (nil? (:parent @third)) "an unwitnessed replay breaks adjacency"))
          (let [failure (ex-info "injected output readback failure" {})
                reads (atom 0)
                download gpu/download-range!
                before @(:completed-replays (:executable c))]
            (with-redefs [gpu/download-range!
                          (fn [& args]
                            (if (= 2 (swap! reads inc)) (throw failure) (apply download args)))]
              (is (identical? failure (try (compiled/invoke-with-evidence c {})
                                          (catch Throwable error error)))))
            (is (= (inc before) @(:completed-replays (:executable c))))
            (is (zero? @(:output-leases (:executable c))))
            (is (nil? (:completed-evidence @(:execution-state (:executable c)))))
            (with-open [fresh (compiled/invoke-with-evidence c {})]
              (is (nil? (:parent @fresh)) "failed evidence cannot be a hidden parent"))))
        (finally (compiled/close! c))))))

(deftest completed-resident-bytes-and-state-chain-match-real-opencl
  (if @opencl/opencl-available?
    (do (run-input-case! :ocl:0) (run-state-case! :ocl:0) (run-ownership-and-range-case! :ocl:0))
    (opencl/opencl-skip! "completed resident byte producer evidence")))

(deftest completed-resident-bytes-and-state-chain-match-real-level-zero
  (if @ze/gpu-available?
    (do (run-input-case! :ze:0) (run-state-case! :ze:0) (run-ownership-and-range-case! :ze:0))
    (ze/gpu-skip! "completed resident byte producer evidence")))
