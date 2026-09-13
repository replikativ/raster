(ns raster.compiler.distributed-numerical-test
  "Numerical interpretation of planned halo rectangles. The device check emulates two workers
   on one GPU; it proves compiled local updates and resident halo copies, not a multi-host runtime."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.ode.pde :as pde]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.numerical-state :as state]
            [raster.runtime.numerical-content :as content]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Files OpenOption StandardOpenOption]
           [java.nio.file.attribute FileAttribute]
           [java.security MessageDigest]
           [java.util HexFormat]))

(deftm heat-step!
  [out :- (Array double), u :- (Array double), rhs :- (Array double),
   nx :- Long, ny :- Long, dt :- Double]
  (pde/heat-rhs-2d! rhs u nx ny 0.25 4.0 9.0)
  (numeric/axpy! out u dt rhs))

(def ^:private width 7)
(def ^:private dt 0.01)
(def ^:private iterations 4)

(def ^:private compiled-step
  (delay (equation/compile #'heat-step! {:target :ze:0 :dtype :double})))

(defn- problem []
  (let [shards [(distributed/shard {:id :left :value :u :device :worker-0
                                   :offsets [0 0] :shape [2 width]})
                (distributed/shard {:id :right :value :u :device :worker-1
                                   :offsets [2 0] :shape [4 width]})]
        abstract (av/tensor {:dtype :double :shape [6 width]
                             :sharding {:kind :partitioned :axis 0
                                        :devices [:worker-0 :worker-1]}})
        halo (distributed/schedule-halo
              (distributed/halo-exchange {:id :exchange :value :u :axis 0 :width 1
                                          :boundary :nonperiodic})
              abstract shards {[:worker-0 :worker-1] [:forward]
                               [:worker-1 :worker-0] [:backward]} [])
        initial (double-array
                 (for [i (range 8) j (range width)]
                   (if (or (zero? i) (= i 7) (zero? j) (= j (dec width)))
                     0.0 (+ (* 0.3 i) (Math/sin (+ (* 0.7 i) j))))))]
    {:shards shards :halo halo :initial initial}))

(defn- local-inputs [{:keys [shards initial]}]
  (into {}
        (map (fn [{:keys [id offsets shape]}]
               (let [buffer (double-array (* (+ 2 (first shape)) width))]
                 ;; Owned rows have one ghost row on either side. Physical domain boundaries
                 ;; remain zero; exchanged ghosts are filled solely from the planned routes.
                 (System/arraycopy initial (* (inc (first offsets)) width)
                                   buffer width (* (first shape) width))
                 [id buffer]))) shards))

(defn- halo-copies [{:keys [shards halo]}]
  (let [by-id (into {} (map (juxt :id identity)) shards)]
    (mapv (fn [{:keys [attributes bytes]}]
            (let [{:keys [source-shard target-shard source-region destination-region]} attributes
                  source-row (- (first (:offsets source-region))
                                (first (:offsets (by-id source-shard))))
                  target-row (first (:offsets destination-region))]
              {:source source-shard :target target-shard
               :src-element (* (inc source-row) width)
               :dst-element (* (inc target-row) width)
               :elements (quot bytes Double/BYTES)})) (:steps halo))))

(defn- reference
  ([initial] (reference initial iterations))
  ([initial nsteps]
   (loop [u (aclone ^doubles initial) step 0]
     (if (= step nsteps) u
         (recur (heat-step! (double-array (alength u)) u (double-array (alength u))
                           8 width dt)
                (inc step))))))

(defn- max-error [expected actual]
  (when-not (= (count expected) (count actual))
    (throw (ex-info "numerical output extent differs from the reference"
                    {:expected (count expected) :actual (count actual)})))
  (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) expected actual)))

(defn- payload-address [^ByteBuffer bytes]
  (let [digest (MessageDigest/getInstance "SHA-256")]
    (.update digest (.duplicate bytes))
    (state/content-address :sha-256 (.formatHex (HexFormat/of) (.digest digest)))))

(defn- checkpoint-manifest [address byte-count]
  (state/certify
   (state/manifest
    {:id :heat/step-2 :logical-coordinate {:step 2 :time-seconds (* 2 dt)}
     :fields [(state/field
               {:id :u :value (av/tensor {:dtype :double :shape [8 width]})
                :chunk-shape [8 width]
                :chunks [(state/chunk
                           {:id :field :offsets [0 0] :shape [8 width]
                            :logical-byte-length byte-count :stored-byte-length byte-count
                            :content address
                            :storage {:format :raw-f64 :byte-order :little-endian}})]})]
     :numerical-contract {:mode :ieee :determinism :reproducible-order
                          :compatibility-id "heat-euler:f64:dirichlet:dt=0.01"}
     :provenance {:program-fingerprint "raster-numerical-acceptance/heat-step-v1"}})))

(defn- open-checkpoint-lease [path certified]
  ;; A real local-file realization fixture, not a production store/provider implementation.
  ;; Verify actual bytes against the certified identity before numerical code consumes them.
  (let [certified (state/verify! certified)
        chunk (get-in certified [:manifest :fields 0 :chunks 0])
        address (:content chunk)
        arena (Arena/ofConfined)]
    (try
      (with-open [channel (FileChannel/open path (into-array OpenOption [StandardOpenOption/READ]))]
        (when-not (= (:stored-byte-length chunk) (.size channel))
          (throw (ex-info "checkpoint payload extent differs" {:reason :checkpoint-size})))
        (let [segment (.map channel FileChannel$MapMode/READ_ONLY 0 (.size channel) arena)]
          (when-not (= address (payload-address (.asByteBuffer segment)))
            (throw (ex-info "checkpoint payload does not match its content address"
                            {:reason :checkpoint-digest})))
          (content/local-content-lease
           {:content address
            :placement (content/content-placement {:provider-id :local-test :tier-id :file
                                                    :content address})
            :segment segment :byte-length (.byteSize segment) :release-fn #(.close arena)})))
      (catch Throwable error
        (.close arena)
        (throw error)))))

(deftest real-mapped-checkpoint-resumes-the-same-numerical-evolution
  (let [initial (:initial (problem))
        checkpoint (reference initial 2)
        byte-count (* Double/BYTES (alength ^doubles checkpoint))
        bytes (doto (ByteBuffer/allocate byte-count) (.order ByteOrder/LITTLE_ENDIAN))
        path (Files/createTempFile "raster-heat-checkpoint-" ".bin" (make-array FileAttribute 0))]
    (try
      (doseq [v checkpoint] (.putDouble bytes v))
      (.flip bytes)
      (let [certified (checkpoint-manifest (payload-address bytes) byte-count)]
        (with-open [channel (FileChannel/open path (into-array OpenOption [StandardOpenOption/WRITE]))]
          (while (.hasRemaining bytes) (.write channel bytes))
          (.force channel true))
        ;; Mapping is reopened after the writer closes. The scoped segment, not the original
        ;; checkpoint array, supplies the restored numerical state.
        (let [lease (open-checkpoint-lease path certified)
              segment (content/lease-segment lease)
              restored (try
                         (is (.isReadOnly ^MemorySegment segment))
                         (.toArray ^MemorySegment segment
                                   (.withOrder ValueLayout/JAVA_DOUBLE_UNALIGNED ByteOrder/LITTLE_ENDIAN))
                         (finally (.close lease)))]
          (is (content/lease-closed? lease))
          (is (false? (.isAlive (.scope ^MemorySegment segment))))
          (is (= (vec checkpoint) (vec restored)))
          (is (< (max-error (reference initial) (reference restored 2)) 1.0e-12)))
        (testing "structural certification cannot disguise corrupt payload bytes"
          (with-open [channel (FileChannel/open path (into-array OpenOption [StandardOpenOption/WRITE]))]
            (.write channel (ByteBuffer/wrap (byte-array [(byte 127)])) 0))
          (is (= :checkpoint-digest
                 (:reason (ex-data (try (open-checkpoint-lease path certified)
                                       (catch clojure.lang.ExceptionInfo error error))))))))
      (finally (Files/deleteIfExists path)))))

(defn- whole-field-executable [initial]
  (let [n (alength ^doubles initial)
        plan (equation/lower @compiled-step
                             [(double-array n) initial (double-array n) 8 width dt])]
    (link/instantiate! plan)))

(defn- source-node [executable source]
  (or (first (keep (fn [[id node]] (when (identical? source (:source node)) id))
                   (get-in executable [:plan :nodes])))
      (throw (ex-info "source boundary missing from lowered plan" {}))))

(defn- advance-device! [executable input steps]
  (let [output (first (get-in executable [:plan :outputs]))]
    (dotimes [_ steps]
      (link/run! executable)
      (gpu/copy-range! (:session executable) (link/node-view executable output)
                       (link/node-view executable input) {:elements (* 8 width)}))
    output))

(deftest mapped-checkpoint-restores-into-a-new-compiled-device-session
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "mapped-checkpoint-device-continuation")
    (let [initial (:initial (problem))
          n (* 8 width) byte-count (* n Double/BYTES)
          path (Files/createTempFile "raster-device-checkpoint-" ".bin" (make-array FileAttribute 0))]
      (try
        ;; The raw payload has a declared endian contract. This fixture performs no endian
        ;; conversion while transferring native f64 device buffers.
        (when-not (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder))
          (throw (ex-info "raw f64 checkpoint transfer needs explicit endian conversion" {})))
        (let [certified
              (with-open [executable (whole-field-executable initial)]
                (let [output (advance-device! executable (source-node executable initial) 2)]
                  (with-open [arena (Arena/ofConfined)
                              channel (FileChannel/open path
                                                        (into-array OpenOption
                                                                    [StandardOpenOption/READ
                                                                     StandardOpenOption/WRITE]))]
                    (.position channel (dec byte-count))
                    (.write channel (ByteBuffer/wrap (byte-array [0])))
                    (let [segment (.map channel FileChannel$MapMode/READ_WRITE 0 byte-count arena)]
                      (gpu/download-range! (:session executable) (link/node-view executable output)
                                           segment {:elements n})
                      (.force segment)
                      (checkpoint-manifest (payload-address (.asByteBuffer segment)) byte-count)))))
              empty-state (double-array n)]
          ;; The old executable/session and mapped writer are closed before a fresh execution
          ;; is initialized directly from the read-only mapping. No intermediate restore array.
          (with-open [executable (whole-field-executable empty-state)
                      lease (open-checkpoint-lease path certified)]
            (let [input (source-node executable empty-state)]
              (gpu/upload-range! (:session executable) (link/node-view executable input)
                                 (content/lease-segment lease) {:elements n})
              ;; Upload is synchronous: closing here is safe and proves replay does not depend
              ;; on a still-live mapped arena. Async ownership is covered by separate contracts.
              (.close lease)
              (is (content/lease-closed? lease))
              (let [output (advance-device! executable input 2)
                    actual (link/download executable output)]
                (is (< (max-error (reference initial) actual) 1.0e-10))))))
        (finally (Files/deleteIfExists path))))))

(defn- owned-values [shards arrays]
  (vec (mapcat (fn [{:keys [id shape]}]
                 (take (* (first shape) width) (drop width (seq (get arrays id))))) shards)))

(defn- host-run [problem exchange?]
  (loop [inputs (local-inputs problem) step 0]
    (if (= step iterations) (owned-values (:shards problem) inputs)
        (do
          (when exchange?
            (doseq [{:keys [source target src-element dst-element elements]} (halo-copies problem)]
              (System/arraycopy (inputs source) src-element (inputs target) dst-element elements)))
          (recur (into {}
                       (map (fn [{:keys [id shape]}]
                              (let [u (inputs id) n (alength ^doubles u)]
                                [id (heat-step! (double-array n) u (double-array n)
                                                (+ 2 (first shape)) width dt)])))
                       (:shards problem))
                 (inc step))))))

(deftest planned-halos-preserve-the-unpartitioned-numerical-evolution
  (let [p (problem)
        expected (subvec (vec (reference (:initial p))) width (* 7 width))]
    (is (= [{:source :left :target :right :src-element 14 :dst-element 0 :elements 7}
            {:source :right :target :left :src-element 7 :dst-element 21 :elements 7}]
           (halo-copies p)))
    (is (< (max-error expected (host-run p true)) 1.0e-12))
    (testing "the oracle detects omitted exchange, not just local stencil arithmetic"
      (is (> (max-error expected (host-run p false)) 1.0e-3)))))

(deftest composed-stencil-and-update-compile-without-driver-allocation
  (let [compilation @compiled-step]
    (is (= :none (get-in compilation [:stats :fallback])))
    (doseq [rows [4 6]]
      (let [n (* rows width)
            plan (equation/lower compilation [(double-array n) (double-array n)
                                              (double-array n) rows width dt])]
        (is (= 0 (get-in plan [:attributes :driver-allocations])))))))

(deftest compiled-two-worker-halos-use-resident-copies
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "compiled-two-worker-resident-halos")
    (let [p (problem)
          inputs (local-inputs p)
          compilation @compiled-step
          session (gpu/make-session :ze:0)
          opened (atom [])]
      (try
        (is (= :none (get-in compilation [:stats :fallback])))
        (let [workers
              (into {}
                    (map (fn [{:keys [id shape]}]
                           (let [u (inputs id) n (alength ^doubles u)
                                 plan (equation/lower compilation
                                                      [(double-array n) u (double-array n)
                                                       (+ 2 (first shape)) width dt])
                                 input-id (first (keep (fn [[id node]]
                                                       (when (identical? u (:source node)) id))
                                                     (:nodes plan)))
                                 _ (when-not input-id
                                     (throw (ex-info "lowering lost the source input boundary" {:shard id})))
                                 executable (link/instantiate! plan {:session session})]
                             (swap! opened conj executable)
                             (is (= 0 (get-in plan [:attributes :driver-allocations])))
                             [id {:executable executable :input input-id
                                  :output (first (:outputs plan)) :elements n}])))
                    (:shards p))]
          (dotimes [_ iterations]
            (doseq [{:keys [source target] :as copy} (halo-copies p)]
              (let [src (workers source) dst (workers target)]
                (gpu/copy-range! session
                                 (link/node-view (:executable src) (:input src))
                                 (link/node-view (:executable dst) (:input dst))
                                 (select-keys copy [:src-element :dst-element :elements]))))
            (doseq [[_ {:keys [executable input output elements]}] workers]
              (link/run! executable)
              (gpu/copy-range! session (link/node-view executable output)
                               (link/node-view executable input) {:elements elements})))
          (let [actual (owned-values (:shards p)
                                     (update-vals workers
                                                  (fn [{:keys [executable output]}]
                                                    (link/download executable output))))
                expected (subvec (vec (reference (:initial p))) width (* 7 width))]
            (is (< (max-error expected actual) 1.0e-10))))
        (finally
          (doseq [executable (reverse @opened)] (link/close! executable))
          (gpu/close-session! session))))))
