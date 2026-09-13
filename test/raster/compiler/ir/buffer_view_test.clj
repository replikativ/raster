(ns raster.compiler.ir.buffer-view-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as view]))

(def allocation
  (view/allocation {:id :kv-cache :byte-size 4096 :memory-space :device
                    :device :ze:0 :alignment 64 :coherence :host-coherent
                    :ownership :owned}))

(deftest shaped-views-have-stable-byte-ranges
  (let [whole (view/view allocation {:id :whole :dtype :float :shape [16 64]})
        prefix (view/subview whole {:id :prefix :dtype :float :shape [4 64]})
        tail (view/subview whole {:id :tail :byte-offset (* 4 4 64)
                                  :shape [12 64]})]
    (is (= [64 1] (:strides whole)))
    (is (= 4096 (:byte-length whole)))
    (is (view/overlaps? whole prefix))
    (is (view/disjoint? prefix tail))
    (is (view/contiguous? prefix))
    (is (view/same-range? whole
                          (view/view allocation {:id :other-name :dtype :float
                                                 :shape [1024]})))))

(deftest physical-range-identity-is-device-scoped-and-empty-views-do-not-overlap
  (let [base (view/view allocation {:dtype :float :shape [8]})
        remote (assoc-in base [:allocation :device] :cuda:0)
        empty (view/view allocation {:dtype :float :byte-offset 8 :shape [0]})]
    (is (not (view/overlaps? base remote)))
    (is (not (view/same-range? base remote)))
    (is (not (view/covered-contiguous? base [remote])))
    (is (not (view/overlaps? base empty)))
    (is (not (view/overlaps? empty base)))
    (is (view/disjoint? base empty))))

(deftest regional-writes-preserve-untouched-initialization
  (let [base (view/view allocation {:dtype :double :shape [8]})
        cut (view/view allocation {:dtype :double :byte-offset 16 :shape [2]})
        remaining (view/subtract-contiguous base cut)]
    (is (= [[0 16] [32 32]] (mapv (juxt :byte-offset :byte-length) remaining)))
    (is (every? view/buffer-view? remaining))
    (is (every? #(= allocation (:allocation %)) remaining))
    (is (not (view/covered-contiguous? base remaining)))
    (is (view/covered-contiguous? base (conj remaining cut)))
    (is (view/covered-contiguous? base (reverse (conj remaining cut))))
    (is (= [] (view/subtract-contiguous base base)))
    (is (= [base] (view/subtract-contiguous base (assoc-in cut [:allocation :device] :cuda:0))))
    (is (= remaining
           (view/subtract-contiguous base (view/view allocation {:dtype :byte :byte-offset 16 :shape [16]}))))
    (is (not (view/covered-contiguous? base [(view/view allocation {:dtype :byte :shape [64]})])))
    (is (view/covered-contiguous? (view/view allocation {:dtype :double :shape [0]}) []))
    (is (= :buffer-view-region-alignment
           (:reason (ex-data (try (view/subtract-contiguous base
                                                           (view/view allocation {:dtype :byte :byte-offset 1 :shape [1]}))
                                 (catch clojure.lang.ExceptionInfo e e))))))))

(deftest region-coverage-does-not-hide-layout-or-allocation-conflicts
  (let [base (view/view allocation {:dtype :float :shape [8]})
        strided (view/view allocation {:dtype :float :shape [4] :strides [2]})
        changed (assoc-in base [:allocation :ownership] :borrowed)
        reason (fn [f] (:reason (ex-data (try (f) (catch clojure.lang.ExceptionInfo e e)))))]
    (is (= :buffer-view-region-layout (reason #(view/subtract-contiguous base strided))))
    (is (= :buffer-view-region-layout (reason #(view/covered-contiguous? base [base strided]))))
    (is (view/covered-contiguous? base [base (assoc strided :byte-offset 128)])
        "a disjoint strided view supplies no evidence and does not prevent valid coverage")
    (doseq [covers [[base changed] [changed base]]]
      (is (= :buffer-view-region-allocation (reason #(view/covered-contiguous? base covers)))))
    (is (= :buffer-view-region-allocation (reason #(view/subtract-contiguous base changed))))))

(deftest contiguous-subtraction-agrees-with-an-element-set-model
  (let [base (view/view allocation {:dtype :float :byte-offset 8 :shape [8]})
        cells (fn [v] (set (range (quot (:byte-offset v) 4)
                                 (quot (+ (:byte-offset v) (:byte-length v)) 4))))]
    (doseq [start (range 13) end (range start 13)]
      (let [cut (view/view allocation {:dtype :float :byte-offset (* 4 start) :shape [(- end start)]})
            remaining (view/subtract-contiguous base cut)
            expected (into #{} (remove (cells cut)) (cells base))]
        (is (= expected (into #{} (mapcat cells) remaining)))
        (is (view/covered-contiguous? base (conj remaining cut)))))))

(deftest prefix-views-require-the-same-allocation-and-typed-range
  (let [base (view/view allocation {:dtype :float :shape [16]})
        prefix (view/subview base {:shape [8]})
        shifted (view/subview base {:shape [8] :byte-offset 4})]
    (is (view/prefix-view? base prefix))
    (is (not (view/prefix-view? prefix base)))
    (is (view/contains-contiguous-view? base shifted))
    (is (not (view/prefix-view? base shifted)))
    (is (not (view/prefix-view? base (assoc-in prefix [:allocation :id] :unrelated))))
    (is (not (view/prefix-view? base (assoc-in prefix [:allocation :ownership] :borrowed))))
    (is (not (view/prefix-view? base (view/view allocation {:dtype :int :shape [8]}))))
    (is (not (view/prefix-view? base (view/view allocation {:dtype :float :shape [8] :strides [2]}))))))

(deftest strided-views-state-their-physical-span
  (let [columns (view/view allocation {:id :columns :dtype :float
                                       :shape [4 8] :strides [64 2]})]
    (is (= (* 4 (inc (+ (* 3 64) (* 7 2)))) (:byte-length columns)))
    (is (not (view/contiguous? columns)))))

(deftest views-fail-loud-on-invalid-physical-claims
  (testing "shape is never inferred implicitly in the physical IR"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicit realized shape"
                          (view/view allocation {:dtype :float}))))
  (testing "shape cannot exceed allocation"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds its allocation"
                          (view/view allocation {:dtype :float :shape [1025]}))))
  (testing "byte length cannot contradict shape and strides"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"differs"
                          (view/view allocation {:dtype :float :shape [4]
                                                 :byte-length 12}))))
  (testing "subviews are contained by their base, not merely by the allocation"
    (let [prefix (view/view allocation {:id :prefix :dtype :float :shape [16]})]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exceeds its base"
                            (view/subview prefix {:byte-offset 32 :dtype :float
                                                  :shape [16]}))))))

(deftest rectangular-regions-preserve-coordinates-and-strides
  (let [padded (view/view allocation {:id :padded :dtype :float :shape [6 7]})
        owned (view/rectangular-subview padded {:offsets [1 0] :shape [4 7]})
        lower-ghost (view/rectangular-subview padded {:offsets [0 0] :shape [1 7]})
        upper-ghost (view/rectangular-subview padded {:offsets [5 0] :shape [1 7]})
        column (view/rectangular-subview owned {:offsets [0 2] :shape [4 1]})]
    (is (= [28 112 [7 1]] ((juxt :byte-offset :byte-length :strides) owned)))
    (is (= (:allocation padded) (:allocation owned)))
    (is (view/contiguous? owned))
    (is (view/disjoint? owned lower-ghost))
    (is (view/disjoint? owned upper-ghost))
    (is (= [36 88 [7 1]] ((juxt :byte-offset :byte-length :strides) column)))
    (is (not (view/contiguous? column)))
    (let [empty (view/rectangular-subview owned {:offsets [4 7] :shape [0 0]})]
      (is (= 0 (:byte-length empty)))
      (is (= (view/byte-end owned) (:byte-offset empty))))))

(deftest rectangles-are-contained-by-axes-not-just-allocation-bytes
  (let [base (view/view allocation {:dtype :float :shape [4 7]})]
    (doseq [region [{:offsets [0 6] :shape [1 2]}
                    {:offsets [-1 0] :shape [1 7]}
                    {:offsets [0] :shape [4]}
                    {:offsets [0 0] :shape [4 -1]}
                    {:offsets [Long/MAX_VALUE 0] :shape [Long/MAX_VALUE 7]}]]
      (is (= :buffer-view-region
             (:reason (ex-data (try (view/rectangular-subview base region)
                                   (catch clojure.lang.ExceptionInfo error error)))))))))
