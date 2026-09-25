(ns raster.compiler.passes.parallel.city-workload-device-test
  "Source-to-device/JVM parity for city-rstr's agent-day kernels. Hardware is optional on CI."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.city-workload-fixture :as city]
            [raster.dl.gpu-grad-parity :as probe]
            [raster.gpu.core :as gpu]))

(defn- inputs []
  (let [rng (java.util.Random. 5)
        n 1024 nc 16 ncell 80
        cell-xy (double-array (mapcat (fn [c] [(+ 9.1 (* 0.002 (mod c 20)))
                                              (+ 48.7 (* 0.0015 (quot c 20)))]) (range ncell)))
        cand-xy (double-array (mapcat (fn [_] [(+ 9.1 (* 0.04 (.nextDouble rng)))
                                               (+ 48.7 (* 0.03 (.nextDouble rng)))]) (range nc)))
        att3 (double-array (for [s (range 3) q (range nc)]
                             (if (zero? (mod (+ q s) 7)) 0.0
                                 (* 50.0 (Math/exp (* 3.0 (.nextDouble rng)))))))
        params3 (double-array [0.8 2.2 400.0 1.1 1.4 900.0 0.0 2.0 600.0])
        totals3 (double-array (* 3 ncell))
        cl (double-array (map #(aget ^doubles cell-xy (* 2 %)) (range ncell)))
        ca (double-array (map #(aget ^doubles cell-xy (inc (* 2 %))) (range ncell)))
        ql (double-array (map #(aget ^doubles cand-xy (* 2 %)) (range nc)))
        qa (double-array (map #(aget ^doubles cand-xy (inc (* 2 %))) (range nc)))]
    (dotimes [s 3]
      (let [att (double-array nc) params (double-array 3) totals (double-array ncell)]
        (System/arraycopy att3 (* s nc) att 0 nc)
        (System/arraycopy params3 (* s 3) params 0 3)
        (city/cell-totals! cl ca ql qa att params totals ncell nc)
        (System/arraycopy totals 0 totals3 (* s ncell) ncell)))
    {:n n :nc nc :ncell ncell :cl cl :ca ca :ql ql :qa qa
     :cell-xy cell-xy :cand-xy cand-xy :att3 att3 :params3 params3 :totals3 totals3
     :ptype (int-array (map #(mod % 2) (range n)))
     :home-cell (int-array (map #(if (zero? (mod % 9)) -1 (mod (* % 13) ncell)) (range n)))
     :work (int-array (map #(if (or (odd? %) (zero? (mod % 9))) (mod % nc) -1) (range n)))
     :work-cell (int-array (map #(mod (* % 31) ncell) (range nc)))
     :type-offsets (int-array [0 1 3]) :type-cdf (double-array [1.0 0.4 1.0])
     :diary-offsets (int-array [0 4 7 9])
     :ep-loc (int-array [0 2 2 0 0 1 2 0 2])
     :ep-minute (int-array [420 600 660 720 420 540 720 420 1080])
     :spend3 (int-array (for [s (range 3) i (range n)] (+ 500 (* 100 s) (mod i 11))))
     :shares (double-array [0.65 0.85 1.0 0.1 0.0 0.3])}))

(defn- typed-scalars [abi values]
  (mapv (fn [{:keys [name kernel-dtype] :as slot}]
          (let [v (get values (str name) ::missing)]
            (when (= ::missing v) (throw (ex-info "missing city test scalar" {:slot slot})))
            {:type kernel-dtype
             :value (case kernel-dtype
                      :int (int v) :long (long v) :float (float v) :double (double v))}))
        (filter #(and (= :scalar (:kind %)) (not= :bound (:role %))) abi)))

(defn- run-device [kernel-key kernel buffers values n outputs]
  (let [session (gpu/make-session :ze:0)]
    (try
      (let [ki (first (gpu/compile! session kernel-key kernel {:dtype :double}))]
        (gpu/alloc! session buffers)
        (gpu/invoke! session kernel-key {} (typed-scalars (:abi ki) values) n)
        (assoc (into {} (map (fn [key] [key (vec (gpu/download session key))]) outputs))
               :required-scalars
               (mapv :name (filter #(and (= :scalar (:kind %))
                                          (not= :bound (:role %))) (:abi ki)))
               :stable-inputs
               (set (map :name (filter #(= :no-write-alias (:aliasing %)) (:abi ki))))))
      (finally (gpu/close-session! session)))))

(defn- common-buffers [{:keys [n nc ncell] :as input}]
  (into {}
        (for [[key dtype size] [[:ptype :int n] [:home-cell :int n] [:work :int n]
                                [:work-cell :int nc] [:type-offsets :int 3]
                                [:type-cdf :double 3] [:diary-offsets :int 4]
                                [:ep-loc :int 9] [:ep-minute :int 9]]]
          [key [dtype size (get input key)]])))

(deftest explicit-floating-to-integer-coordinate-matches-jvm
  (if-not @probe/gpu-available?
    (probe/gpu-skip! "explicit floating-to-integer coordinate")
    (let [values (double-array [0.0 0.124 0.125 0.249 0.5 0.624 0.875 0.999])
          expected (int-array 8)
          _ (city/cast-coordinate-histogram! values expected (alength values))
          device (run-device :cast-coordinate-histogram
                             #'city/cast-coordinate-histogram!
                             {:values [:double (alength values) values]
                              :counts [:int 8 (int-array 8)]}
                             {"n" (alength values)} (alength values) [:counts])]
      (is (= (vec expected) (:counts device)))
      (is (= [2 2 0 0 2 0 0 2] (:counts device))))))

(deftest singleton-do-before-episode-loop-preserves-effects
  (if-not @probe/gpu-available?
    (probe/gpu-skip! "singleton do before episode loop")
    (let [locations (int-array [2 0 2 2 0 2 2 2])
          values (double-array [0.1 0.9 0.6 0.2 0.3 0.8 0.4 0.7])
          expected (int-array 3)
          _ (city/episode-class-histogram! locations values expected 4)
          device (run-device :episode-class-histogram
                             #'city/episode-class-histogram!
                             {:locations [:int 8 locations]
                              :values [:double 8 values]
                              :counts [:int 3 (int-array 3)]}
                             {"n" 4} 4 [:counts])]
      (is (= [4 3 3] (vec expected)))
      (is (= (vec expected) (:counts device))))))

(deftest city-day-kernels-match-their-jvm-source
  (if-not @probe/gpu-available?
    (probe/gpu-skip! "city source-to-device parity")
    (let [{:keys [n nc ncell] :as input} (inputs)
          retail-visits (int-array (* nc 24)) retail-counts (int-array 2)
          spend-visits (int-array (* 3 nc 24)) revenue (int-array (* 3 nc 24)) spend-counts (int-array 8)
          params (double-array (take 3 (:params3 input)))
          r-dims (long-array [n nc 11]) s-dims (long-array [n nc 11 ncell])]
      (city/retail-visits! (:ptype input) (:home-cell input) (:work input) (:work-cell input)
                           (:type-offsets input) (:type-cdf input) (:diary-offsets input)
                           (:ep-loc input) (:ep-minute input) (:cl input) (:ca input)
                           (:ql input) (:qa input) (double-array (take nc (:att3 input)))
                           (double-array (take ncell (:totals3 input))) params
                           retail-visits retail-counts r-dims)
      (city/spend-day! (:ptype input) (:home-cell input) (:work input) (:work-cell input)
                       (:type-offsets input) (:type-cdf input) (:diary-offsets input)
                       (:ep-loc input) (:ep-minute input) (:cell-xy input) (:cand-xy input)
                       (:att3 input) (:totals3 input) (:params3 input) (:shares input)
                       (:spend3 input) revenue spend-visits spend-counts s-dims)
      (let [retail-buffers (merge (common-buffers input)
                                  {:cell-lon [:double ncell (:cl input)] :cell-lat [:double ncell (:ca input)]
                                   :cand-lon [:double nc (:ql input)] :cand-lat [:double nc (:qa input)]
                                   :cand-att [:double nc (double-array (take nc (:att3 input)))]
                                   :totals [:double ncell (double-array (take ncell (:totals3 input)))]
                                   :params [:double 3 params] :dims [:long 3 r-dims]
                                   :visits [:int (* nc 24) (int-array (* nc 24))]
                                   :counts [:int 2 (int-array 2)]})
            retail (run-device :retail #'city/retail-visits! retail-buffers
                               {"n" n "nc" nc "seed" 11 "alpha" (aget params 0)
                                "beta" (aget params 1) "d0" (aget params 2)} n [:visits :counts])
            spend-buffers (merge (common-buffers input)
                                 {:cell-xy [:double (* 2 ncell) (:cell-xy input)]
                                  :cand-xy [:double (* 2 nc) (:cand-xy input)]
                                  :att3 [:double (* 3 nc) (:att3 input)]
                                  :totals3 [:double (* 3 ncell) (:totals3 input)]
                                  :params3 [:double 9 (:params3 input)]
                                  :shares [:double 6 (:shares input)]
                                  :spend3 [:int (* 3 n) (:spend3 input)]
                                  :dims [:long 4 s-dims]
                                  :revenue3 [:int (* 3 nc 24) (int-array (* 3 nc 24))]
                                  :visits3 [:int (* 3 nc 24) (int-array (* 3 nc 24))]
                                  :counts [:int 8 (int-array 8)]})
            spend (run-device :spend #'city/spend-day! spend-buffers
                              {"n" n "nc" nc "seed" 11 "ncell" ncell} n
                              [:revenue3 :visits3 :counts])]
        (is (= (vec retail-visits) (:visits retail)))
        (is (= (vec retail-counts) (:counts retail)))
        (is (= [] (:required-scalars retail))
            "the resident params/dims loads are not extra public launch scalars")
        (is (every? (:stable-inputs retail) '[params dims]))
        (is (= (vec revenue) (:revenue3 spend)))
        (is (= (vec spend-visits) (:visits3 spend)))
        (is (= (vec spend-counts) (:counts spend)))
        (is (= '[n] (:required-scalars spend)))
        (is (every? (:stable-inputs spend) '[params3 dims]))))))
