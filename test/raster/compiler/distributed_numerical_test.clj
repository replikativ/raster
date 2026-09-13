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
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]))

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

(defn- reference [initial]
  (loop [u (aclone ^doubles initial) step 0]
    (if (= step iterations) u
        (recur (heat-step! (double-array (alength u)) u (double-array (alength u))
                          8 width dt)
               (inc step)))))

(defn- owned-values [shards arrays]
  (vec (mapcat (fn [{:keys [id shape]}]
                 (take (* (first shape) width) (drop width (seq (get arrays id))))) shards)))

(defn- max-error [expected actual]
  (when-not (= (count expected) (count actual))
    (throw (ex-info "numerical output extent differs from the reference"
                    {:expected (count expected) :actual (count actual)})))
  (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) expected actual)))

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
