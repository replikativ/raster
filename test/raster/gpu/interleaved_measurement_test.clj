(ns raster.gpu.interleaved-measurement-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.measurement :as measurement]))

(deftest samples-rotate-and-preserve-chronology
  (let [calls (atom [])
        candidates (mapv (fn [id] {:id id :sample-fn #(do (swap! calls conj id) (count @calls))})
                         [:a :b :c])
        result (measurement/measure-interleaved! candidates :rounds 3 :warmup-rounds 1
                                                  :timing-source :synthetic)]
    (is (= [:a :b :c :a :b :c :b :c :a :c :a :b] @calls))
    (is (= [4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0] (mapv :ns (:samples result))))
    (is (= [0 0 0 1 1 1 2 2 2] (mapv :round (:samples result))))
    (doseq [id [:a :b :c]]
      (is (= #{0 1 2} (set (map :position (filter #(= id (:candidate %)) (:samples result)))))))
    (is (= [4.0 9.0 11.0] (get-in result [:measurements :a :samples-ns])))
    (is (= :synthetic (get-in result [:measurements :a :timing-source])))
    (is (= 1 (get-in result [:measurements :a :warmup-iterations])))))

(deftest rejects-invalid-options-before-device-work
  (let [calls (atom 0)
        candidate {:id :a :sample-fn #(swap! calls inc)}]
    (doseq [candidates [[] [candidate candidate] [{:id :a :sample-fn nil}]]]
      (is (thrown? clojure.lang.ExceptionInfo (measurement/measure-interleaved! candidates))))
    (doseq [opts [[:rounds 0] [:rounds 1.5] [:warmup-rounds -1]
                  [:rounds (inc (bigint Integer/MAX_VALUE))] [:timing-source "host"]
                  [:cv-threshold -1] [:cv-threshold Double/NaN]
                  [:cv-threshold Double/POSITIVE_INFINITY]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (apply measurement/measure-interleaved! [candidate] opts))))
    (is (zero? @calls))))

(deftest invalid-samples-fail-in-warmup-and-measurement
  (doseq [warmup [0 1] sample [nil -1 Double/NaN Double/POSITIVE_INFINITY]]
    (let [calls (atom 0)
          failure (try (measurement/measure-interleaved!
                        [{:id :bad :sample-fn #(do (swap! calls inc) sample)}]
                        :warmup-rounds warmup :rounds 2)
                       nil
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= :bad (:candidate failure)))
      (is (= (if (zero? warmup) :measurement :warmup) (:phase failure)))
      (is (= 1 @calls)))))
