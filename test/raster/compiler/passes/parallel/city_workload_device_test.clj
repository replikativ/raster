(ns raster.compiler.passes.parallel.city-workload-device-test
  "Source-to-device/JVM parity for city-rstr's agent-day kernels. Hardware is optional on CI."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.city-workload-fixture :as city]
            [raster.dl.gpu-grad-parity :as probe]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
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

(defn- run-device [kernel-key kernel buffers values n outputs & [device]]
  (let [session (gpu/make-session (or device :ze:0))]
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

(deftest branched-effect-steps-match-jvm-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 probe/gpu-available? probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "branch-local effect recurrences on " device))
      (doseq [nc [0 1 2 7]]
        (let [expected (int-array 3)
              expected-state (int-array 3)
              prepared (compiled/lower #'city/branch-local-store-steps! [(int-array 3) (int-array 3) 3 nc]
                                       {:compiler :equation-first :target device :dtype :double
                                        :outputs '[state out]})
              live (compiled/instantiate! prepared)]
          (try
            (dotimes [_ 2]
              (city/branch-local-store-steps! expected-state expected 3 nc)
              (let [result (live {})]
                (is (= (vec expected) (vec (value/->host (:out result))))
                    (str device " trips=" nc))
                (is (= (vec expected-state) (vec (value/->host (:state result)))))))
            (finally (compiled/close! live))))))))

(deftest carried-branches-match-jvm-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 probe/gpu-available? probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-fp64-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "carried effect branches on " device))
      (doseq [kernel [#'city/branch-local-carried-steps! #'city/carried-steps-with-empty-arm!]
              nc [0 1 2 7]]
        (let [expected-state (double-array [-1 1 -1])
              expected (double-array [-77 -77 -77])
              prepared (compiled/lower kernel [(aclone expected-state) (aclone expected) 3 nc]
                                       {:compiler :equation-first :target device :dtype :double
                                        :outputs '[state out]})
              live (compiled/instantiate! prepared)]
          (try
            (dotimes [_ 2]
              (kernel expected-state expected 3 nc)
              (let [result (live {})]
                (is (= (vec expected) (vec (value/->host (:out result))))
                    (str kernel " " device " trips=" nc))
                (is (= (vec expected-state) (vec (value/->host (:state result)))))))
            (finally (compiled/close! live))))))))

(deftest two-exit-search-matches-jvm-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 probe/gpu-available? probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-fp64-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "two-exit search on " device))
      (doseq [nc [1 5]]
        (let [weights (double-array [1 2 3 4])
              targets (double-array [-1 0.5 1 2.5 6 100])
              expected (int-array 6)
              _ (city/two-exit-walk! weights targets expected 6 nc)
              actual (run-device :two-exit-search #'city/two-exit-walk!
                                 {:weights [:double 4 weights] :targets [:double 6 targets]
                                  :out [:int 6 (int-array (repeat 6 -77))]}
                                 {"n" 6 "nc" nc} 6 [:out] device)]
          (is (= (vec expected) (:out actual)) (str device " width=" nc)))))))

(deftest three-carry-effects-match-jvm-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 probe/gpu-available? probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "three carried values on " device))
      (doseq [nc [0 1 2 7]]
        (let [expected (int-array 9)
              initial (int-array 9)
              prepared (compiled/lower #'city/three-carry-effects! [initial 3 nc]
                                       {:compiler :equation-first :target device :dtype :double
                                        :outputs '[out]})
              prior (volatile! nil)
              live (compiled/instantiate! prepared)]
          (try
            (is (= [:fixed]
                   (mapv #(get-in % [:executable :selection]) (compiled/execution-info live)))
                "the public report describes the bound graph, not an unperformed dispatch search")
            (dotimes [_ 2]
              (city/three-carry-effects! expected 3 nc)
              (let [old @prior
                    result (:out (live {}))]
                (when old
                  (is (not (value/live? old)) "replay invalidates the previous state view"))
                (vreset! prior result)
                (is (= (vec expected) (vec (value/->host result)))
                    (str device " trips=" nc))))
            (is (= (vec (int-array 9)) (vec initial)) "resident atomics do not mutate host inputs")
            (finally (compiled/close! live)))
          (is (not (value/live? @prior)) "close invalidates the final explicit output"))))))

(deftest terminal-moments-match-jvm-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 probe/gpu-available? probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "terminal moments on " device))
      (doseq [nc [0 1 7]]
        (let [n 3 weights (double-array (range (max 1 (* n nc))))
              sums (double-array n) squares (double-array n)
              _ (city/terminal-moments! weights sums squares n nc)
              actual (run-device :terminal-moments #'city/terminal-moments!
                                 {:weights [:double (alength weights) weights]
                                  :sums [:double n (double-array (repeat n -77))]
                                  :squares [:double n (double-array (repeat n -77))]}
                                 {"n" n "nc" nc} n [:sums :squares] device)]
          (is (= (vec sums) (:sums actual)) (str device " width=" nc))
          (is (= (vec squares) (:squares actual)) (str device " width=" nc)))))))

(deftest terminal-branch-decision-matches-jvm-on-local-backends
  (doseq [[device available? skip!] [[:ze:0 probe/gpu-available? probe/gpu-skip!]
                                    [:ocl:0 opencl/opencl-available? opencl/opencl-skip!]]]
    (if-not @available?
      (skip! (str "terminal branch snapshot on " device))
      (doseq [nc [0 1 2]]
        (let [left (double-array [1 -1]) right (double-array 2)
              _ (city/terminal-branch-snapshot! left right 2 nc)
              actual (run-device :terminal-branch #'city/terminal-branch-snapshot!
                                 {:left [:double 2 (double-array [1 -1])]
                                  :right [:double 2 (double-array 2)]}
                                 {"n" 2 "nc" nc} 2 [:left :right] device)]
          (is (= (vec left) (:left actual)) (str device " width=" nc))
          (is (= (vec right) (:right actual)) (str device " width=" nc)))))))

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

(deftest plain-diary-store-before-choice-loop-matches-jvm
  (if-not @probe/gpu-available?
    (probe/gpu-skip! "plain store before branch-local effect loop")
    (let [diary (int-array 3)
          choice (int-array 1)
          _ (city/branch-diary-before-choice-loop! diary choice 3)
          device (run-device :branch-diary-before-choice-loop
                             #'city/branch-diary-before-choice-loop!
                             {:diary [:int 3 (int-array 3)]
                              :choice [:int 1 (int-array 1)]}
                             {"n" 3} 3 [:diary :choice])]
      (is (= [1 1 1] (vec diary)))
      (is (= [6] (vec choice)))
      (is (= (vec diary) (:diary device)))
      (is (= (vec choice) (:choice device))))))

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
