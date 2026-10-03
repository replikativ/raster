(ns raster.gpu.storage-evidence-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.completed-evidence-device-test :as oracle]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.value :as value]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- run-owner-case! [target]
  ;; Reuse the existing completed-byte oracle's labeled synthetic packaged build fixture;
  ;; this exercises real compiler/runtime ownership, not release-build certification.
  (with-redefs [build/current-identity #'oracle/test-build]
    (let [prepared (compiled/lower #'oracle/witnessed-scale [(float-array [1 2 3 4]) 4]
                                   {:target target :compiler :equation-first :dtype :float})
          c (compiled/instantiate! prepared)
          foreign (compiled/instantiate! prepared)]
      (try
        (let [fact (compiled/measure-storage-representation! c :float)
              foreign-fact (compiled/measure-storage-representation! foreign :float)
              session (:session (:executable c))
              order (:byte-order @fact)]
          (is (= (probe/expected-bytes :float order) (:observed-bytes @fact)))
          (is (= (gpu/execution-device-info session) (:device @fact)))
          (is (seq (get-in @fact [:device :driver])))
          (is (zero? @(:completed-replays (:executable c))))
          (is (nil? @(:live-outputs c)))
          (is (empty? (:events @session)))
          (is (not-any? #(and (keyword? %) (.startsWith (name %) "storage-probe-"))
                        (concat (keys (:buffers @session)) (keys (:kernel-graphs @session)))))
          (is (= :compiled-execution-identity-owner
                 (reason #(compiled/measure-storage-representation! (assoc c :target target) :float))))
          (is (= :compiled-evidence-unbound
                 (reason #(compiled/measure-storage-representation! prepared :float))))
          (with-open [receipt (compiled/invoke-with-evidence c {})]
            (let [description (compiled/completed-storage-description receipt {:float fact})]
              (is (= (:fingerprint @receipt) (:completed-fingerprint description)))
              (is (= (get-in @receipt [:outputs])
                     (update-vals (:outputs description) #(dissoc % :storage))))
              (is (every? #(= {:format :raw-array :byte-order order} (:storage %))
                          (vals (:outputs description))))
              (is (= [2.0 4.0 6.0 8.0]
                     (vec (value/->host (first (vals (compiled/completed-output-values receipt))))))))
            (is (= :compiled-representation-mismatch
                   (reason #(compiled/completed-storage-description receipt {:float foreign-fact}))))
            (is (= :compiled-representation-mismatch
                   (reason #(compiled/completed-storage-description receipt {:float (assoc fact :data @fact)}))))
            (let [epoch (:value-epoch @(:execution-state (:executable c)))]
              (is (= :link-output-lease-active
                     (reason #(compiled/measure-storage-representation! c :float))))
              (is (= epoch (:value-epoch @(:execution-state (:executable c)))))))
          ;; Measurement retires prior unleased output wrappers but cannot claim a plan replay.
          (let [output (first (vals (compiled/invoke-compiled c {})))
                completed @(:completed-replays (:executable c))
                registry-entry (requiring-resolve
                                 (symbol (str (case target :ze:0 'raster.gpu.ze-runtime
                                                          :ocl:0 'raster.gpu.ocl-runtime))
                                         "kernel-registry-entry"))
                cached-kernel (:kernel-handle (registry-entry "rstr_storage_probe_float"))]
            (compiled/measure-storage-representation! c :float)
            (is (some? cached-kernel))
            (is (identical? cached-kernel (:kernel-handle (registry-entry "rstr_storage_probe_float")))
                "repeated probe registration retains the module's cached kernel")
            (is (= completed @(:completed-replays (:executable c))))
            (is (nil? @(:live-outputs c)))
            (is (some? (error-of #(value/->host output)))))
          (let [receipt (compiled/invoke-with-evidence c {})]
            (.close ^java.io.Closeable receipt)
            (is (some? (reason #(compiled/completed-storage-description receipt {:float fact})))))
          (compiled/close! c)
          (is (some? (error-of #(compiled/measure-storage-representation! c :float)))))
        (finally (compiled/close! c) (compiled/close! foreign))))))

(deftest owner-bound-storage-evidence-on-opencl
  (if @opencl/opencl-available?
    (run-owner-case! :ocl:0)
    (opencl/opencl-skip! "owner-bound storage representation evidence")))

(deftest owner-bound-storage-evidence-on-level-zero
  (if @ze/gpu-available?
    (run-owner-case! :ze:0)
    (ze/gpu-skip! "owner-bound storage representation evidence")))
