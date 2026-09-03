(ns raster.gpu.device-probe
  "Structured test gates for the OpenCL runtime.

   A missing device is a legitimate, visible skip on generic CI.  A namespace load failure is a
   test failure.  Driver/query failures are visible skips unless RASTER_EXPECT_OPENCL declares
   that this lane must execute OpenCL.  Capability decisions always inspect the exact device bound
   by the singleton runtime, never some other enumerated device."
  (:require [clojure.string :as str]
            [clojure.test :refer [is]]))

(defn- truthy-environment?
  [name]
  (contains? #{"1" "true" "yes" "on"}
             (some-> (System/getenv name) str/lower-case)))

(def ^:dynamic *expect-opencl?*
  (truthy-environment? "RASTER_EXPECT_OPENCL"))

(defn probe-opencl-with
  "Classify an OpenCL runtime using injectable operations.  Public for the gate's unit tests."
  [{:keys [load! query-devices init! selected-device-info]}]
  (let [loaded (try
                 (load!)
                 {:ok true}
                 (catch Throwable error
                   {:ok false :error error}))]
    (if-not (:ok loaded)
      {:backend :opencl :status :load-failed :error (:error loaded)}
      (try
        (let [devices (query-devices)]
          (if-not (seq devices)
            {:backend :opencl :status :no-device}
            (do
              (init!)
              (let [device (selected-device-info)]
                (when-not (map? device)
                  (throw (ex-info "OpenCL runtime did not identify its selected device"
                                  {:selected device})))
                {:backend :opencl
                 :status :available
                 :n-devices (count devices)
                 :device device}))))
        (catch Throwable error
          {:backend :opencl :status :probe-error :error error})))))

(defn- runtime-api
  []
  (require 'raster.gpu.ocl-runtime)
  (let [api {:load! (constantly true)
             :query-devices (resolve 'raster.gpu.ocl-runtime/query-devices)
             :init! (resolve 'raster.gpu.ocl-runtime/init!)
             :selected-device-info (resolve 'raster.gpu.ocl-runtime/selected-device-info)}]
    (doseq [[operation f] (dissoc api :load!)]
      (when-not (ifn? f)
        (throw (ex-info "OpenCL runtime API is incomplete" {:operation operation}))))
    api))

(def opencl-status
  (delay
    (let [loaded (try
                   {:ok true :api (runtime-api)}
                   (catch Throwable error
                     {:ok false :error error}))]
      (if (:ok loaded)
        (probe-opencl-with (:api loaded))
        {:backend :opencl :status :load-failed :error (:error loaded)}))))

(defn- extension-set
  [device]
  (into #{} (remove str/blank?) (str/split (or (:extensions device) "") #"\s+")))

(defn capability-supported?
  [device capability]
  (case capability
    :fp16 (contains? (extension-set device) "cl_khr_fp16")
    ;; Tuned dispatch alternatives (subgroup score reuse, matrix leaves) are emitted for GPU
    ;; descriptors only; a CPU OpenCL device such as PoCL executes the portable kernels.
    :gpu-device (= :gpu (:type device))
    (throw (ex-info "Unknown OpenCL test capability" {:capability capability}))))

(defn opencl-status-for
  [capability]
  (let [status @opencl-status]
    (if (and (= :available (:status status))
             capability
             (not (capability-supported? (:device status) capability)))
      (assoc status :status :missing-capability :capability capability)
      status)))

(def opencl-available?
  (delay (= :available (:status @opencl-status))))

(def opencl-fp16-available?
  (delay (= :available (:status (opencl-status-for :fp16)))))

(def opencl-gpu-available?
  (delay (= :available (:status (opencl-status-for :gpu-device)))))

(defonce ^:private opencl-skip-log (atom {}))

(defonce ^:private opencl-summary-hook
  (delay
    (.addShutdownHook
     (Runtime/getRuntime)
     (Thread.
      (fn []
        (let [status @opencl-status
              skips @opencl-skip-log
              total (reduce + 0 (vals skips))]
          (println
           (format "  [OPENCL SUITE] probe=%s%s | %d test(s) took the skip path%s"
                   (name (:status status))
                   (if-let [device (get-in status [:device :name])]
                     (str " (" device ")")
                     "")
                   total
                   (if (pos? total) (str " " (pr-str skips)) "")))))))
    true))

(defn opencl-skip!
  "Record one visible OpenCL skip marker, or fail when the runtime is broken/required.

   `capability` is nil, :fp16 or :gpu-device.  Missing optional capabilities remain honest skips;
   RASTER_EXPECT_OPENCL requires a usable device but does not imply every optional extension."
  ([test-label] (opencl-skip! test-label nil))
  ([test-label capability]
   @opencl-summary-hook
   (let [{:keys [status error device] :as result} (opencl-status-for capability)]
     (swap! opencl-skip-log update (or capability status) (fnil inc 0))
     (case status
       :load-failed
       (is false
           (str "[OPENCL LOAD FAILED] " test-label " — " (some-> error .getMessage)
                ". This is a runtime breakage, not an absent device."))

       :probe-error
       (if *expect-opencl?*
         (is false
             (str "[OPENCL PROBE FAILED] " test-label " — " (some-> error .getMessage)
                  ". RASTER_EXPECT_OPENCL requires this lane to execute OpenCL."))
         (do
           (println (str "  [OPENCL SKIP/WARN] " test-label " — "
                         (some-> error .getMessage)))
           (is true "opencl-skip-marker")))

       :no-device
       (if *expect-opencl?*
         (is false
             (str "[OPENCL DEVICE MISSING] " test-label
                  " — RASTER_EXPECT_OPENCL requires an OpenCL device."))
         (do
           (println (str "  [OPENCL SKIP] " test-label " — no requested OpenCL device"))
           (is true "opencl-skip-marker")))

       :missing-capability
       (do
         (println (str "  [OPENCL SKIP] " test-label " — selected device " (:name device)
                       " lacks " (name capability)))
         (is true "opencl-skip-marker"))

       :available
       (throw
        (IllegalStateException.
         (str "opencl-skip! called for " test-label " while OpenCL is available")))

       (throw (ex-info "Unknown OpenCL probe status" {:result result}))))))
