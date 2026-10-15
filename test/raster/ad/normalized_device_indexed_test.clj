(ns raster.ad.normalized-device-indexed-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-device :as device]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.kernel-body]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.link-module :as link-module]))

(defn- reason [invoke]
  (try (invoke) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- indexed-source
  ([index-dtype] (indexed-source index-dtype 3 2))
  ([index-dtype extent data-extent]
   (let [dimensions (filter symbol? [extent data-extent])
         values (merge {'indices (av/tensor {:dtype index-dtype :shape [extent]})
                        'data (av/tensor {:dtype :float :shape [data-extent]})
                        'gathered (av/tensor {:dtype :float :shape [extent]
                                             :ownership :owned})}
                       (zipmap dimensions (repeat (av/tensor {:dtype :long :shape []}))))
         equation
         (list '= 'gather '[gathered]
               (list 'map {:index 'i :extent extent
                           :attributes {:stable-array-captures '[data]}}
                     '[indices] '[data]
                     (soac/lambda-form
                      '[index data-array] []
                      '[(clojure.core/aget data-array index)])))]
     (soac/make
      (soac/default-program-facts
       {:values values
        :inputs (vec (distinct (concat '[indices data]
                                       (when (symbol? extent) [extent]))))
        :equations {'gather (soac/default-equation-facts)}
        :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
      [equation] '[gathered]))))

(defn- indexed-owner
  ([] (indexed-owner :long 3 2))
  ([index-dtype extent data-extent]
   (canonical/adapt-indexed-map
    (normalized/linearize-indexed-map-program
     (indexed-source index-dtype extent data-extent)
     (the-ns 'raster.ad.normalized-device-indexed-test)))))

(defn- reordered-indexed-owner []
  (let [source (indexed-source :long)
        reordered (soac/make (assoc (soac/facts source) :inputs '[data indices])
                             (soac/equations source) (soac/outputs source))]
    (canonical/adapt-indexed-map
     (normalized/linearize-indexed-map-program
      reordered (the-ns 'raster.ad.normalized-device-indexed-test)))))

(defn- arguments [owner indices data]
  (mapv (fn [{:keys [id]}]
          (cond (= id 'indices) indices
                (= id 'data) data
                :else (throw (ex-info "unexpected indexed argument" {:id id}))))
        (get-in owner [:interface :inputs])))

(defn- nested-records [value]
  (lazy-seq
   (cons value
         (cond
           (map? value) (mapcat nested-records (vals value))
           (coll? value) (mapcat nested-records value)
           :else []))))

(defn- synthetic-descriptor [target] {:device-id target})

(deftest indexed-owner-assembles-one-complete-ordered-transpose-entry
  (let [owner (indexed-owner)
        captured (atom nil)
        calls (atom [])
        runtime (reify java.io.Closeable (close [_]))]
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate!
                  (fn [module _] (reset! captured module) runtime)
                  link-module/run-and-lease!
                  (fn [_ entry request]
                    (swap! calls conj [entry request])
                    (reify java.io.Closeable (close [_])))]
      (let [source-indices (long-array [0 0 0])
            prepared (device/prepare-indexed-owner
                      owner (arguments owner source-indices (float-array [7.0 11.0]))
                      {:target :ocl:indexed-structure :schedule {:precision :f32-scalar}})
            ports (:ports prepared)
            spine (:spine @captured)
            entries (:entries @captured)
            primal ((:primal! prepared)
                    {'indices source-indices 'data (float-array [7.0 11.0])})
            checked-indices (get-in @calls [0 1 :inputs (:index-input ports)])
            transpose ((:transpose! prepared)
                       (float-array [1.0 2.0 3.0]) (:dependencies primal))
            effect-instance (last (:instances spine))
            zero-instance (second (:instances spine))
            effect-program (get-in effect-instance [:call :program])
            records (nested-records effect-program)
            loops (distinct (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                                    records))]
        (is (= [1 2] (mapv (comp count :instances) entries)))
        (is (= 3 (count (:instances spine))))
        (is (empty? (get-in zero-instance [:call :program :inputs])))
        (is (= :borrowed
               (get-in owner [:interface :cotangent-input :value :ownership])))
        (is (= :owned
               (get-in spine [:values (:seed-input ports) :abstract :ownership])))
        (is (= (get-in effect-instance [:call :buffers (:zero-destination ports)])
               (get-in effect-instance [:call :buffers (:logical-gradient ports)])
               (:gradient-output ports)))
        (is (= (set (:residuals ports))
               (set (map :value (:frontiers @captured)))))
        (is (= 1 (count loops)))
        (is (= :ordered (get-in (first loops) [:attributes :association])))
        (is (not-any? #(instance? raster.compiler.ir.kernel_body.AtomicRMW %) records))
        (is (not (identical? source-indices checked-indices)))
        (is (= (vec source-indices) (vec checked-indices)))
        (is (= :primal (ffirst @calls)))
        (is (= :transpose (first (second @calls))))
        (.close ^java.io.Closeable (:lease transpose))
        (.close ^java.io.Closeable (:lease primal))
        ((:close! prepared))))))

(deftest indexed-input-validation-precedes-every-runtime-call
  (let [owner (indexed-owner)
        calls (atom 0)
        runtime (reify java.io.Closeable (close [_]))]
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [_ _] runtime)
                  link-module/run-and-lease!
                  (fn [& _]
                    (swap! calls inc)
                    (reify java.io.Closeable (close [_])))]
      (let [prepared (device/prepare-indexed-owner
                      owner (arguments owner (long-array [0 0 0])
                                       (float-array [7.0 11.0]))
                      {:target :ocl:indexed-inputs :schedule {:precision :f32-scalar}})]
        (doseq [inputs [{'indices (int-array [0 0 0]) 'data (float-array [7.0 11.0])}
                        {'indices (long-array [0 0]) 'data (float-array [7.0 11.0])}
                        {'indices (long-array [-1 0 0]) 'data (float-array [7.0 11.0])}
                        {'indices (long-array [0 2 0]) 'data (float-array [7.0 11.0])}
                        {'indices (long-array [0 0 0]) 'data (double-array [7.0 11.0])}]]
          (is (some? (reason #((:primal! prepared) inputs)))))
        (is (zero? @calls))
        (doseq [seed [(double-array [1.0 2.0 3.0]) (float-array [1.0 2.0])]]
          (is (= :normalized-device-array
                 (reason #((:transpose! prepared) seed {})))))
        (is (zero? @calls))
        ((:close! prepared)))))
  (testing "dynamic and empty geometry refuse before runtime instantiation"
    (doseq [[make-owner expected]
            [[#(indexed-owner :long 0 2) :normalized-device-indexed-interface]
             [#(indexed-owner :long 3 0) :normalized-device-indexed-interface]
             [#(indexed-owner :long 3 'm) :normalized-device-indexed-interface]
             [#(indexed-owner :long 3 (inc (bigint Integer/MAX_VALUE)))
              :ad-normalized-indexed-map-boundary]]]
      (let [instantiations (atom 0)]
        (with-redefs [hardware/descriptor-for synthetic-descriptor
                      link-module/instantiate!
                      (fn [& _] (swap! instantiations inc)
                        (throw (AssertionError. "runtime observed")))]
          (is (= expected
                 (reason #(let [owner (make-owner)]
                            (device/prepare-indexed-owner
                             owner [(long-array 0) (float-array 2)]
                             {:target :ocl:indexed-refusal})))))
          (is (zero? @instantiations)))))))

(deftest indexed-public-order-is-canonical-and-buffer-transport-is-id-based
  (let [owner (reordered-indexed-owner)
        calls (atom [])
        captured (atom nil)
        runtime (reify java.io.Closeable (close [_]))]
    (is (= '[data indices] (mapv :id (get-in owner [:interface :inputs]))))
    (with-redefs [hardware/descriptor-for synthetic-descriptor
                  link-module/instantiate! (fn [module _]
                                             (reset! captured module)
                                             runtime)
                  link-module/run-and-lease!
                  (fn [_ entry request]
                    (swap! calls conj [entry request])
                    (reify java.io.Closeable (close [_])))]
      (let [indices (long-array [0 1 0])
            data (float-array [7.0 11.0])
            prepared (device/prepare-indexed-owner
                      owner [data indices]
                      {:target :ocl:indexed-order :schedule {:precision :f32-scalar}})
            ports (:ports prepared)
            primal ((:primal! prepared) {'indices indices 'data data})
            input-ports (mapv :id (:inputs ports))
            runtime-inputs (get-in @calls [0 1 :inputs])]
        (is (= '[data indices] input-ports))
        (is (= '[indices data]
               (get-in @captured [:spine :instances 0 :call :program :inputs])))
        (is (= (set (map :value (:inputs ports))) (set (keys runtime-inputs))))
        (is (identical? data (get runtime-inputs
                                  (:value (first (:inputs ports))))))
        (is (not (identical? indices
                             (get runtime-inputs (:value (second (:inputs ports)))))))
        (.close ^java.io.Closeable (:lease primal))
        ((:close! prepared))))))

(defn- bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- download [runtime lease value-id n]
  (let [payload ((ns-resolve 'raster.gpu.link-module 'opaque-payload) runtime)
        resident (get-in @lease [:outputs value-id])
        output (float-array n)]
    (gpu/download-range! (:session payload) resident output {:elements n})
    output))

(defn- index-array [index-dtype values]
  (case index-dtype :int (int-array values) :long (long-array values)))

(defn- run-indexed-device-case! [target index-dtype]
  (let [owner (indexed-owner index-dtype 3 2)
        indices (index-array index-dtype [0 0 0])
        data (float-array [7.0 11.0])
        captured (atom nil)
        instantiate link-module/instantiate!]
    (with-redefs [link-module/instantiate!
                  (fn [module options]
                    (let [runtime (instantiate module options)]
                      (reset! captured runtime)
                      runtime))]
      (let [prepared (device/prepare-indexed-owner
                      owner (arguments owner indices data)
                      {:target target :schedule {:precision :f32-scalar}})]
        (try
          (is (= :normalized-device-index-bounds
                 (reason #((:primal! prepared)
                           {'indices (index-array index-dtype [0 2 0])
                            'data (float-array [7.0 11.0])}))))
          (let [primal ((:primal! prepared) {'indices indices 'data data})]
            (try
              (is (= (bits [7.0 7.0 7.0])
                     (bits (download @captured (:lease primal) (:output-id primal) 3))))
              (is (= [0 0 0] (vec indices)))
              (is (= (bits [7.0 11.0]) (bits data)))
              (dotimes [position 3]
                (case index-dtype
                  :int (aset-int ^ints indices position 1)
                  :long (aset-long ^longs indices position 1)))
              (aset-float data 0 (float 99.0))
              (let [seed (float-array [(float 16777216.0) (float 1.0)
                                       (float -16777216.0)])
                    seed-before (bits seed)
                    ordered ((:transpose! prepared) seed
                             (:dependencies primal))]
                (try
                  (is (= (bits [0.0 0.0])
                         (bits (download @captured (:lease ordered)
                                         (:output-id ordered) 2))))
                  (is (= seed-before (bits seed)))
                  (finally (.close ^java.io.Closeable (:lease ordered)))))
              (let [changed ((:transpose! prepared)
                             (float-array [1.0 2.0 3.0]) (:dependencies primal))]
                (try
                  (is (= (bits [6.0 0.0])
                         (bits (download @captured (:lease changed)
                                         (:output-id changed) 2))))
                  (finally (.close ^java.io.Closeable (:lease changed)))))
              (is (= :link-module-runtime-lease-active
                     (reason #((:primal! prepared)
                               {'indices (index-array index-dtype [1 1 1])
                                'data (float-array [7.0 11.0])}))))
              (finally (.close ^java.io.Closeable (:lease primal))))
            (is (= :link-module-runtime-lease-released
                   (reason #((:transpose! prepared)
                             (float-array [1.0 2.0 3.0]) (:dependencies primal))))))
          (let [next ((:primal! prepared)
                      {'indices (index-array index-dtype [1 1 1])
                       'data (float-array [7.0 11.0])})]
            (try
              (let [gradient ((:transpose! prepared)
                              (float-array [1.0 2.0 3.0]) (:dependencies next))]
                (try
                  (is (= (bits [0.0 6.0])
                         (bits (download @captured (:lease gradient)
                                         (:output-id gradient) 2))))
                  (finally (.close ^java.io.Closeable (:lease gradient)))))
              (finally (.close ^java.io.Closeable (:lease next)))))
          (finally ((:close! prepared))))))))

(deftest normalized-indexed-device-owner-runs-on-opencl
  (if @ocl-probe/opencl-available?
    (doseq [index-dtype [:int :long]]
      (run-indexed-device-case! :ocl:0 index-dtype))
    (ocl-probe/opencl-skip! "normalized indexed device owner acceptance")))

(deftest normalized-indexed-device-owner-runs-on-level-zero
  (if @ze-probe/gpu-available?
    (doseq [index-dtype [:int :long]]
      (run-indexed-device-case! :ze:0 index-dtype))
    (ze-probe/gpu-skip! "normalized indexed device owner acceptance")))
