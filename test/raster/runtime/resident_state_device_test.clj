(ns raster.runtime.resident-state-device-test
  "Actual generated PDE capture and restored continuation; the bounded test store is not a
   production persistence implementation, and the packaged build fixture is explicitly synthetic."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-state :as state]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.completed-evidence-device-test :as oracle]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.ode.pde :as pde]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state :as resident]
            [raster.runtime.resident-state-test :as unit])
  (:import [java.lang.foreign MemorySegment]
           [java.nio ByteOrder]))

(defn- near? [expected actual]
  (and (= (count expected) (count actual))
       (every? true? (map #(< (Math/abs (- %1 %2)) 1.0e-11) expected actual))))

(defn- run-heat-capture! [target]
  (with-redefs [build/current-identity #'oracle/test-build]
    (let [nx 4 ny 5 n (* nx ny) alpha 0.2 dt 0.001
          initial (double-array (map #(+ 2.0 (Math/sin (double %))) (range n)))
          first-host (double-array n) second-host (double-array n)
          out (double-array n)
          prepared (compiled/lower #'pde/periodic-heat-step-2d!
                                   [out initial nx ny alpha dt (double (* nx nx)) (double (* ny ny))]
                                   {:target target :compiler :equation-first :dtype :double
                                    :inline? true :outputs '[out]})
          node (first (:outputs (compiled/plan prepared)))
          input-key (:key (first (filter #(and (= :input (:role %)) (= 'u (:sym %)))
                                         (:in-tree prepared))))
          {:keys [provider blobs events]} (#'unit/provider (fn [& _]))]
      (is (some? input-key))
      (pde/periodic-heat-step-2d! first-host initial nx ny alpha dt
                                  (double (* nx nx)) (double (* ny ny)))
      (pde/periodic-heat-step-2d! second-host first-host nx ny alpha dt
                                  (double (* nx nx)) (double (* ny ny)))
      (let [captured
            (let [c (compiled/instantiate! prepared)]
              (try
                (let [fact (compiled/measure-storage-representation! c :double)
                      result
                      (with-open [receipt (compiled/invoke-with-evidence c {})]
                        (let [result (resident/capture!
                                      receipt {:double fact} provider :local
                                      {:id :heat/step-1 :parents [] :logical-coordinate {:step 1 :time dt}
                                       :fields [{:id :temperature :source :outputs :node node
                                                 :value (av/tensor {:dtype :double :shape [n]})
                                                 :coordinate-space {:grid-shape [nx ny] :boundary :periodic}}]
                                       :numerical-contract {:mode :ieee-fp64 :determinism :toleranced
                                                            :compatibility-id "periodic-heat-capture-v1"}})]
                          (is (= (:fingerprint @receipt)
                                 (get-in result [:state :manifest :provenance :completed-fingerprint])))
                          (is (= (:bound-schedules @receipt)
                                 (get-in result [:state :manifest :provenance :bound-schedules])))
                          (is (= 1 @(:output-leases (:executable c))))
                          (is (near? (vec first-host) (vec (link/download (:executable c) node))))
                          (assoc result ::first-values (vec (link/download (:executable c) node)))))]
                  (compiled/invoke-compiled c {input-key (double-array (::first-values result))})
                  (assoc result ::uninterrupted (vec (link/download (:executable c) node))))
                (finally (compiled/close! c))))
            certificate (:state captured)
            chunk (get-in certificate [:manifest :fields 0 :chunks 0])
            bytes (get @blobs (:content chunk))
            segment (MemorySegment/ofArray bytes)
            host-order (if (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
                         :little-endian :big-endian)
            restored (double-array n)]
        ;; The producer session is closed before stored bytes are read or a fresh owner is opened.
        (is (= certificate (state/verify! certificate)))
        (is (= (:content chunk) (content/content-address-of segment)))
        (is (empty? @events))
        (is (= :published (:publication (content/finalize-state-availability!
                                         provider certificate :durable (fn [_] :published)))))
        (with-open [lease (content/open-local-content! provider (:content chunk) {:tier :local})]
          (content/decode-raw-array-chunk! chunk lease :double
                                           (MemorySegment/ofArray restored) host-order))
        (is (near? (vec first-host) (vec restored)))
        (is (= (::first-values captured) (vec restored)))
        (let [fresh (compiled/instantiate! prepared)]
          (try
            (compiled/invoke-compiled fresh {input-key restored})
            (is (near? (vec second-host) (vec (link/download (:executable fresh) node))))
            (is (= (::uninterrupted captured) (vec (link/download (:executable fresh) node))))
            (finally (compiled/close! fresh))))
        (is (empty? @events))))))

(deftest generated-heat-capture-and-continuation-on-opencl
  (if @opencl/opencl-available?
    (run-heat-capture! :ocl:0)
    (opencl/opencl-skip! "completed generated heat capture and continuation")))

(deftest generated-heat-capture-and-continuation-on-level-zero
  (if @ze/gpu-available?
    (run-heat-capture! :ze:0)
    (ze/gpu-skip! "completed generated heat capture and continuation")))
